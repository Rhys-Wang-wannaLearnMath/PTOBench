# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP decode front fused kernel (scope1 + scope2 + scope3 + scope4).

Fused pipeline:
- Scope1: RMSNorm + Q/KV projection + RoPE + KV/PE cache update
- Scope2: indexer projection + indexer RoPE + weight reduction + k_cache_idx update
- Scope3: q_idx x k_cache_idx scoring + topk
- Scope4: sparse attention with topk indices, online softmax, latent-to-V projection, dispatch buffer write

Decode contract:
- `seq_lens[b]` must be in `[1, MAX_SEQ]` for active rows.
- Zero-length rows are invalid for this fused decode kernel because scope1/scope2
    write the current token at `pos = seq_len - 1` before scope4 runs.
"""

import pypto.language as pl


from config import (BATCH, MAX_SEQ, HIDDEN, NUM_HEADS, Q_LORA_RANK,
                    KV_LORA_RANK, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM,
                    QK_HEAD_DIM, V_HEAD_DIM, KV_A_OUT, CACHE_ROWS,
                    INDEX_HEADS, INDEX_HEAD_DIM, INDEX_Q_OUT, INDEX_TOPK,
                    EP_NODES, ATTN_OUT, EPS, HIDDEN_INV, INDEX_HEAD_DIM_INV,
                    WEIGHT_SCALE, Q_LORA_INV)

# Scope1 tiles
RMSNORM_K = 512
PROJ_K = 512
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
LORA_CHUNK = 64

# Scope2 tiles
K_CHUNK = 128
IDX_OUT_CHUNK = 128
KIDX_OUT_CHUNK = 64
QREDUCE_OUT_CHUNK = 64
QREDUCE_BATCH_TILE = 16
WEIGHTS_OUT_CHUNK = 16

# Scope3 tiles
SEQ_TILE = 64
MAX_SEQ_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
Q_VALID = 1
Q_PAD = 16
SORT_LEN = 8192
FP32_NEG_INF = -3.4028234663852886e38

# Scope4 tiles
ATTN_SCALE = 1.0 / (QK_HEAD_DIM**0.5)
Q_LATENT_CHUNK = 128
V_OUT_CHUNK = 32
MATMUL_ROW_PAD = 16


def build_deepseek_v3_2_decode_front_scope1234_program():
    @pl.program
    class DeepSeekV32DecodeFrontScope1234:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope1234(
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            q_norm_weight: pl.Tensor[[1, Q_LORA_RANK], pl.FP32],
            kv_norm_weight: pl.Tensor[[1, KV_LORA_RANK], pl.FP32],
            wq_a: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
            wq_b: pl.Tensor[[Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM], pl.BF16],
            wkv_a: pl.Tensor[[HIDDEN, KV_A_OUT], pl.BF16],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            wq_b_idx: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
            wk_idx: pl.Tensor[[HIDDEN, INDEX_HEAD_DIM], pl.BF16],
            weights_proj: pl.Tensor[[HIDDEN, INDEX_HEADS], pl.FP32],
            k_norm_weight: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            k_norm_bias: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            hadamard_q: pl.Tensor[[INDEX_HEAD_DIM, INDEX_HEAD_DIM], pl.BF16],
            hadamard_k: pl.Tensor[[INDEX_HEAD_DIM, INDEX_HEAD_DIM], pl.BF16],
            layer_id_t: pl.Tensor[[1], pl.INT32],
            w_q_nope_to_latent: pl.Tensor[[NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK], pl.BF16],
            w_latent_to_v: pl.Tensor[[NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM], pl.BF16],
            kv_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_LORA_RANK], pl.BF16]],
            pe_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, QK_ROPE_HEAD_DIM], pl.BF16]],
            k_cache_idx: pl.InOut[pl.Tensor[[CACHE_ROWS, INDEX_HEAD_DIM], pl.BF16]],
            dispatch_buf: pl.InOut[pl.Tensor[[EP_NODES, BATCH, ATTN_OUT], pl.BF16]],
        ) -> pl.Tensor[[EP_NODES, BATCH, ATTN_OUT], pl.BF16]:
            # ===== scope1: MLA front path (RMSNorm + Q/KV projection + RoPE + cache writeback) =====
            # Outputs:
            # - kv_cache: normalized KV latent cache row for the current decode token
            # - pe_cache: rotated rope cache row for the current decode token
            # Internal bridge:
            # - q_proj: rotated decode query heads reused by scope4

            # Stage 1.1: RMSNorm on hidden_states.
            normed_states = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="input_rmsnorm"):
                partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for k0 in pl.range(0, HIDDEN, RMSNORM_K):
                    x_chunk = pl.cast(hidden_states[:, k0 : k0 + RMSNORM_K], target_type=pl.FP32)
                    partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]))
                variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
                inv_rms = pl.recip(pl.sqrt(variance))
                for k0 in pl.range(0, HIDDEN, PROJ_K):
                    x_chunk_bf16 = hidden_states[:, k0 : k0 + PROJ_K]
                    x_tile = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                    gamma = input_rms_weight[:, k0 : k0 + PROJ_K]
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_tile, inv_rms), gamma)
                    normed_states = pl.assemble(normed_states, pl.cast(normed, target_type=pl.BF16), [0, k0])

            # Stage 1.2: Project qr = normed @ wq_a.
            qr_fp32 = pl.create_tensor([BATCH, Q_LORA_RANK], dtype=pl.FP32)
            for q0 in pl.parallel(0, Q_LORA_RANK, LORA_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_lora_proj"):
                    q_tile_a = normed_states[:, 0 : PROJ_K]
                    q_tile_b = wq_a[0 : PROJ_K, q0 : q0 + LORA_CHUNK]
                    q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                    for k0 in pl.range(PROJ_K, HIDDEN, PROJ_K):
                        q_tile_a_i = normed_states[:, k0 : k0 + PROJ_K]
                        q_tile_b_i = wq_a[k0 : k0 + PROJ_K, q0 : q0 + LORA_CHUNK]
                        q_acc = pl.matmul_acc(q_acc, q_tile_a_i, q_tile_b_i)
                    qr_fp32 = pl.assemble(qr_fp32, q_acc, [0, q0])

            # Stage 1.3: Apply q_norm on qr.
            qr_out = pl.create_tensor([BATCH, Q_LORA_RANK], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_lora_rmsnorm"):
                q_partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for k0 in pl.range(0, Q_LORA_RANK, LORA_CHUNK):
                    qr_chunk_fp32 = qr_fp32[:, k0 : k0 + LORA_CHUNK]
                    q_partial = pl.reshape(pl.row_sum(pl.mul(qr_chunk_fp32, qr_chunk_fp32)), [1, BATCH])
                    q_partial_sq = pl.add(q_partial_sq, q_partial)
                q_variance = pl.reshape(pl.add(pl.mul(q_partial_sq, Q_LORA_INV), EPS), [BATCH, 1])
                q_inv_rms = pl.recip(pl.sqrt(q_variance))
                for k0 in pl.range(0, Q_LORA_RANK, LORA_CHUNK):
                    qr_chunk_bf16 = pl.cast(qr_fp32[:, k0 : k0 + LORA_CHUNK], target_type=pl.BF16)
                    qr_chunk_fp32 = pl.cast(qr_chunk_bf16, target_type=pl.FP32)
                    q_gamma = q_norm_weight[:, k0 : k0 + LORA_CHUNK]
                    q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk_fp32, q_inv_rms), q_gamma)
                    qr_out = pl.assemble(qr_out, pl.cast(q_normed, target_type=pl.BF16), [0, k0])

            # Stage 1.4: Project q_proj = qr @ wq_b.
            q_proj = pl.create_tensor([BATCH, NUM_HEADS * QK_HEAD_DIM], dtype=pl.BF16)
            for q0 in pl.parallel(0, NUM_HEADS * QK_HEAD_DIM, Q_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_head_proj"):
                    q_chunk_init = qr_out[:, 0 : LORA_CHUNK]
                    wq_b_init = wq_b[0 : LORA_CHUNK, q0 : q0 + Q_OUT_CHUNK]
                    q_out_acc = pl.matmul(q_chunk_init, wq_b_init, out_dtype=pl.FP32)
                    for k0 in pl.range(LORA_CHUNK, Q_LORA_RANK, LORA_CHUNK):
                        q_chunk = qr_out[:, k0 : k0 + LORA_CHUNK]
                        wq_b_chunk = wq_b[k0 : k0 + LORA_CHUNK, q0 : q0 + Q_OUT_CHUNK]
                        q_out_acc = pl.matmul_acc(q_out_acc, q_chunk, wq_b_chunk)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_head_proj_write"):
                    q_proj = pl.assemble(q_proj, pl.cast(q_out_acc, target_type=pl.BF16), [0, q0])

            # Stage 1.5: Project kv_a = normed @ wkv_a.
            kv_a_out = pl.create_tensor([BATCH, KV_A_OUT], dtype=pl.BF16)
            for kv0 in pl.parallel(0, KV_A_OUT, KV_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_a_proj"):
                    kv_tile_a = normed_states[:, 0 : PROJ_K]
                    kv_tile_b = wkv_a[0 : PROJ_K, kv0 : kv0 + KV_OUT_CHUNK]
                    kv_acc = pl.matmul(kv_tile_a, kv_tile_b, out_dtype=pl.FP32)
                    for k0 in pl.range(PROJ_K, HIDDEN, PROJ_K):
                        kv_tile_a_i = normed_states[:, k0 : k0 + PROJ_K]
                        kv_tile_b_i = wkv_a[k0 : k0 + PROJ_K, kv0 : kv0 + KV_OUT_CHUNK]
                        kv_acc = pl.matmul_acc(kv_acc, kv_tile_a_i, kv_tile_b_i)

                # Stage 1.6: Final KV output cast.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_a_write"):
                    kv_a_out = pl.assemble(kv_a_out, pl.cast(kv_acc, target_type=pl.BF16), [0, kv0])

            # Stage 1.7: Apply RoPE on q_proj in-place.
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_rope"):
                    ctx_len = pl.read(seq_lens, [b])
                    pos = ctx_len - 1
                    cos_lo = rope_cos[pos : pos + 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    cos_hi = rope_cos[pos : pos + 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    sin_lo = rope_sin[pos : pos + 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    sin_hi = rope_sin[pos : pos + 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    for q_col in pl.range(QK_NOPE_HEAD_DIM, NUM_HEADS * QK_HEAD_DIM, QK_HEAD_DIM):
                        q_lo = pl.cast(q_proj[b : b + 1, q_col : q_col + QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                        q_hi = pl.cast(q_proj[b : b + 1, q_col + QK_ROPE_HEAD_DIM // 2 : q_col + QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                        q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                        q_proj = pl.assemble(q_proj, pl.cast(q_rot_lo, target_type=pl.BF16), [b, q_col])
                        q_proj = pl.assemble(q_proj, pl.cast(q_rot_hi, target_type=pl.BF16), [b, q_col + QK_ROPE_HEAD_DIM // 2])

            # Stage 1.8: Apply kv_norm on KV latent.
            kv_normed_out = pl.create_tensor([BATCH, KV_LORA_RANK], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rmsnorm"):
                kv_rows = pl.cast(kv_a_out[:, 0 : KV_LORA_RANK], target_type=pl.FP32)
                kv_partial_sq = pl.reshape(pl.row_sum(pl.mul(kv_rows, kv_rows)), [1, BATCH])
                kv_variance = pl.reshape(pl.add(pl.mul(kv_partial_sq, 1.0 / KV_LORA_RANK), EPS), [BATCH, 1])
                kv_inv_rms = pl.recip(pl.sqrt(kv_variance))
                kv_gamma = kv_norm_weight[:, 0 : KV_LORA_RANK]
                kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rows, kv_inv_rms), kv_gamma)
                kv_normed_out = pl.assemble(kv_normed_out, pl.cast(kv_normed, target_type=pl.BF16), [0, 0])

            # Stage 1.9: Write decode caches.
            # - kv_cache: normalized KV latent
            # - pe_cache: rope component from kv_a_out after RoPE
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_cache_write"):
                    ctx_len = pl.read(seq_lens, [b])
                    pos = ctx_len - 1
                    cache_row = b * MAX_SEQ + pos
                    cos_lo = rope_cos[pos : pos + 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    cos_hi = rope_cos[pos : pos + 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    sin_lo = rope_sin[pos : pos + 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    sin_hi = rope_sin[pos : pos + 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    kv_normed_row = kv_normed_out[b : b + 1, 0 : KV_LORA_RANK]
                    pe_lo = pl.cast(kv_a_out[b : b + 1, KV_LORA_RANK : KV_LORA_RANK + QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                    pe_hi = pl.cast(kv_a_out[b : b + 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM // 2 : KV_LORA_RANK + QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                    pe_rot_lo = pl.sub(pl.col_expand_mul(pe_lo, cos_lo), pl.col_expand_mul(pe_hi, sin_lo))
                    pe_rot_hi = pl.add(pl.col_expand_mul(pe_hi, cos_hi), pl.col_expand_mul(pe_lo, sin_hi))
                    kv_cache = pl.assemble(kv_cache, kv_normed_row, [cache_row, 0])
                    pe_cache = pl.assemble(pe_cache, pl.cast(pe_rot_lo, target_type=pl.BF16), [cache_row, 0])
                    pe_cache = pl.assemble(pe_cache, pl.cast(pe_rot_hi, target_type=pl.BF16), [cache_row, QK_ROPE_HEAD_DIM // 2])

            # ===== scope2: indexer path (prepare q_idx/k_cache_idx) =====
            # Outputs:
            # - q_idx_out: BF16 aggregated index query vector for stage3
            # - k_cache_idx: BF16 index key cache row for the current decode token
            #
            # Scope123 switches to an INT8 index path after the Hadamard stage.
            # Scope1234 keeps the BF16 index contract because this standalone file
            # still validates and exports `k_cache_idx` in BF16, and scope3 keeps the
            # matching exact BF16 score path. The staging and comments below stay as
            # close to scope123 as the scope1234 data contract safely allows.

            # Stage 2.1: q_idx_full = wq_b_idx(qr_out).
            q_idx_full = pl.create_tensor([BATCH, INDEX_Q_OUT], dtype=pl.BF16)
            for q0 in pl.parallel(0, INDEX_Q_OUT, IDX_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_q_idx_proj"):
                    s2_q_chunk_init = qr_out[:, 0 : LORA_CHUNK]
                    s2_wq_chunk_init = wq_b_idx[0 : LORA_CHUNK, q0 : q0 + IDX_OUT_CHUNK]
                    s2_q_acc = pl.matmul(s2_q_chunk_init, s2_wq_chunk_init, out_dtype=pl.FP32)
                    for k0 in pl.range(LORA_CHUNK, Q_LORA_RANK, LORA_CHUNK):
                        qr_chunk = qr_out[:, k0 : k0 + LORA_CHUNK]
                        wq_chunk = wq_b_idx[k0 : k0 + LORA_CHUNK, q0 : q0 + IDX_OUT_CHUNK]
                        s2_q_acc = pl.matmul_acc(s2_q_acc, qr_chunk, wq_chunk)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_q_idx_proj_write"):
                    q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_acc, target_type=pl.BF16), [0, q0])

            # Stage 2.2: k_idx = wk_idx(hidden_states).
            k_idx = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)
            for k1 in pl.parallel(0, INDEX_HEAD_DIM, KIDX_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_proj"):
                    s2_x_init = hidden_states[:, 0 : K_CHUNK]
                    wk_init = wk_idx[0 : K_CHUNK, k1 : k1 + KIDX_OUT_CHUNK]
                    s2_k_acc = pl.matmul(s2_x_init, wk_init, out_dtype=pl.FP32)
                    for k0 in pl.range(K_CHUNK, HIDDEN, K_CHUNK):
                        s2_x_chunk = hidden_states[:, k0 : k0 + K_CHUNK]
                        wk_chunk = wk_idx[k0 : k0 + K_CHUNK, k1 : k1 + KIDX_OUT_CHUNK]
                        s2_k_acc = pl.matmul_acc(s2_k_acc, s2_x_chunk, wk_chunk)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_proj_write"):
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_acc, target_type=pl.BF16), [0, k1])

            # Stage 2.3: Apply LayerNorm on k_idx (gamma/beta from k_norm_affine).
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_layernorm"):
                s2_k_tile = pl.cast(k_idx, target_type=pl.FP32)
                s2_mean = pl.row_sum(pl.mul(s2_k_tile, INDEX_HEAD_DIM_INV))
                s2_centered = pl.row_expand_sub(s2_k_tile, s2_mean)
                s2_var = pl.row_sum(pl.mul(pl.mul(s2_centered, s2_centered), INDEX_HEAD_DIM_INV))
                s2_var_eps = pl.add(s2_var, EPS)
                s2_std = pl.reshape(pl.sqrt(pl.reshape(s2_var_eps, [1, BATCH])), [BATCH, 1])
                s2_inv_std = pl.recip(s2_std)
                s2_normed = pl.row_expand_mul(s2_centered, s2_inv_std)
                s2_scaled = pl.col_expand_mul(s2_normed, k_norm_weight)
                s2_ones = pl.add(pl.sub(s2_k_tile, s2_k_tile), 1.0)
                s2_k_normed = pl.add(s2_scaled, pl.col_expand_mul(s2_ones, k_norm_bias))
                k_idx = pl.cast(s2_k_normed, target_type=pl.BF16)

            # Stage 2.4: Apply RoPE on rope dimensions of q_idx_full and k_idx.
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_idx_rope"):
                    pos = pl.read(seq_lens, [b]) - 1
                    cos_lo = rope_cos[pos : pos + 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    cos_hi = rope_cos[pos : pos + 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    sin_lo = rope_sin[pos : pos + 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    sin_hi = rope_sin[pos : pos + 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    for q_col in pl.range(0, INDEX_Q_OUT, INDEX_HEAD_DIM):
                        s2_q_lo = pl.cast(q_idx_full[b : b + 1, q_col : q_col + QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                        s2_q_hi = pl.cast(q_idx_full[b : b + 1, q_col + QK_ROPE_HEAD_DIM // 2 : q_col + QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                        s2_q_rot_lo = pl.sub(pl.col_expand_mul(s2_q_lo, cos_lo), pl.col_expand_mul(s2_q_hi, sin_lo))
                        s2_q_rot_hi = pl.add(pl.col_expand_mul(s2_q_hi, cos_hi), pl.col_expand_mul(s2_q_lo, sin_hi))
                        q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_rot_lo, target_type=pl.BF16), [b, q_col])
                        q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_rot_hi, target_type=pl.BF16), [b, q_col + QK_ROPE_HEAD_DIM // 2])
                    s2_k_lo = pl.cast(k_idx[b : b + 1, 0 : QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                    s2_k_hi = pl.cast(k_idx[b : b + 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                    s2_k_rot_lo = pl.sub(pl.col_expand_mul(s2_k_lo, cos_lo), pl.col_expand_mul(s2_k_hi, sin_lo))
                    s2_k_rot_hi = pl.add(pl.col_expand_mul(s2_k_hi, cos_hi), pl.col_expand_mul(s2_k_lo, sin_hi))
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_rot_lo, target_type=pl.BF16), [b, 0])
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_rot_hi, target_type=pl.BF16), [b, QK_ROPE_HEAD_DIM // 2])

            # Stage 2.5: Apply Hadamard transform (full matrix multiplication).
            # Keep matmul and vector writeback in separate scopes on a2a3 to avoid
            # generating mixed AIC/AIV kernels with zero-work vector lanes.
            for h in pl.parallel(INDEX_HEADS):
                h_offset = h * INDEX_HEAD_DIM
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_q_hadamard_matmul"):
                    q_tile = q_idx_full[:, h_offset : h_offset + INDEX_HEAD_DIM]
                    q_h_tile = pl.matmul(q_tile, hadamard_q, out_dtype=pl.FP32)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_q_hadamard_write"):
                    q_idx_full = pl.assemble(q_idx_full, pl.cast(q_h_tile, target_type=pl.BF16), [0, h_offset])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_hadamard_matmul"):
                k_h_tile = pl.matmul(k_idx, hadamard_k, out_dtype=pl.FP32)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_hadamard_write"):
                k_idx = pl.assemble(k_idx, pl.cast(k_h_tile, target_type=pl.BF16), [0, 0])

            # Stage 2.6: weights = weights_proj(hidden_states) * n_heads^-0.5 * head_dim^-0.5.
            weights_proj_bf16 = pl.create_tensor([HIDDEN, INDEX_HEADS], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_proj_bf16"):
                for k0 in pl.range(0, HIDDEN, K_CHUNK):
                    wp_tile = pl.cast(weights_proj[k0 : k0 + K_CHUNK, :], target_type=pl.BF16)
                    weights_proj_bf16 = pl.assemble(weights_proj_bf16, wp_tile, [k0, 0])

            weights = pl.create_tensor([BATCH, INDEX_HEADS], dtype=pl.FP32)
            for w0 in pl.parallel(0, INDEX_HEADS, WEIGHTS_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_matmul"):
                    s2_x_init = hidden_states[:, 0 : K_CHUNK]
                    wp_init = weights_proj_bf16[0 : K_CHUNK, w0 : w0 + WEIGHTS_OUT_CHUNK]
                    s2_w_acc = pl.matmul(s2_x_init, wp_init, out_dtype=pl.FP32)
                    for k0 in pl.range(K_CHUNK, HIDDEN, K_CHUNK):
                        s2_x_chunk = hidden_states[:, k0 : k0 + K_CHUNK]
                        wp_chunk = weights_proj_bf16[k0 : k0 + K_CHUNK, w0 : w0 + WEIGHTS_OUT_CHUNK]
                        s2_w_acc = pl.matmul_acc(s2_w_acc, s2_x_chunk, wp_chunk)
                    weights = pl.assemble(weights, s2_w_acc, [0, w0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_scale"):
                weights = pl.mul(weights, WEIGHT_SCALE)

            # Stage 2.7: Write current-token k_idx into BF16 cache form.
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_cache_write"):
                    pos = pl.read(seq_lens, [b]) - 1
                    cache_row = b * MAX_SEQ + pos
                    k_cache_idx = pl.assemble(k_cache_idx, k_idx[b : b + 1, :], [cache_row, 0])

            # Stage 2.8: Reduce q_idx_full across heads with weights to get q_idx_out.
            q_idx_out = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)
            weights_flat = pl.reshape(weights, [BATCH * INDEX_HEADS])
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_q_reduce"):
                for d0 in pl.parallel(0, INDEX_HEAD_DIM, QREDUCE_OUT_CHUNK):
                    for b in pl.range(BATCH):
                        s2_acc_b = pl.full([1, QREDUCE_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for h in pl.range(INDEX_HEADS):
                            s2_q_h_b = pl.cast(
                                pl.slice(q_idx_full, [1, QREDUCE_OUT_CHUNK], [b, h * INDEX_HEAD_DIM + d0]),
                                target_type=pl.FP32,
                            )
                            s2_w_h_b = pl.read(weights_flat, [b * INDEX_HEADS + h])
                            s2_acc_b = pl.add(s2_acc_b, pl.mul(s2_q_h_b, s2_w_h_b))
                        q_idx_out = pl.assemble(q_idx_out, pl.cast(s2_acc_b, target_type=pl.BF16), [b, d0])

            # ===== scope3: index score + topk =====
            # Inputs: q_idx_out, k_cache_idx
            # Internal bridge:
            # - topk_idx: top-k positions per batch for scope4
            #
            # Scope123's INT8 path computes quantized per-head scores and topk values.
            # Scope1234 keeps the exact BF16 `q_idx_out x k_cache_idx` score path so the
            # topk indices stay aligned with the exported BF16 cache contract.

            topk_idx = pl.create_tensor([BATCH, INDEX_TOPK], dtype=pl.INT32)
            scores = pl.create_tensor([BATCH, SORT_LEN], dtype=pl.FP32)
            s3_q_padded = pl.create_tensor([BATCH * Q_PAD, INDEX_HEAD_DIM], dtype=pl.BF16)

            for b in pl.parallel(BATCH):
                s3_score_tiles = pl.create_tensor([MAX_SEQ_BLOCKS * Q_PAD, SEQ_TILE], dtype=pl.FP32)

                # Stage 3.0: Pad q_idx_out and pre-fill scores.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_init"):
                    s3_q_row = q_idx_out[b : b + 1, 0 : INDEX_HEAD_DIM]
                    s3_q_padded = pl.assemble(s3_q_padded, s3_q_row, [b * Q_PAD, 0])
                    s3_q_zero_pad = pl.cast(pl.full([Q_PAD - Q_VALID, INDEX_HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
                    s3_q_padded = pl.assemble(s3_q_padded, s3_q_zero_pad, [b * Q_PAD + Q_VALID, 0])
                    s3_neg_inf_row = pl.full([1, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
                    scores = pl.assemble(scores, s3_neg_inf_row, [b, 0])

                s3_ctx_len = pl.read(seq_lens, [b])
                s3_ctx_blocks = (s3_ctx_len + SEQ_TILE - 1) // SEQ_TILE

                # Stage 3.1: Compute tiled BF16 qk logits.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_score_tile"):
                    for sb in pl.range(s3_ctx_blocks):
                        s0 = sb * SEQ_TILE
                        cache_row0 = b * MAX_SEQ + s0
                        s3_q_tile = s3_q_padded[b * Q_PAD : b * Q_PAD + Q_PAD, 0 : INDEX_HEAD_DIM]
                        s3_k_tile = k_cache_idx[cache_row0 : cache_row0 + SEQ_TILE, 0 : INDEX_HEAD_DIM]
                        s3_score_tile = pl.matmul(s3_q_tile, s3_k_tile, b_trans=True, out_dtype=pl.FP32)
                        s3_score_tiles = pl.assemble(s3_score_tiles, s3_score_tile, [sb * Q_PAD, 0])

                # Stage 3.2: Write valid score tiles into the global score row.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_score_write"):
                    for sb in pl.range(s3_ctx_blocks):
                        s0 = sb * SEQ_TILE
                        s3_valid_len = pl.min(SEQ_TILE, s3_ctx_len - s0)
                        score_row0 = sb * Q_PAD
                        s3_score_valid = pl.slice(s3_score_tiles, [1, SEQ_TILE], [score_row0, 0], valid_shape=[1, s3_valid_len])
                        scores = pl.assemble(scores, s3_score_valid, [b, s0])

                s3_sorted_gm = pl.create_tensor([1, 2 * SORT_LEN], dtype=pl.FP32)

                # Stage 3.3: Run sort32 + mrgsort.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_sort"):
                    s3_score_row = scores[b : b + 1, :]
                    idx_init = pl.tensor.arange(0, [1, SORT_LEN], dtype=pl.UINT32)
                    s3_sorted_t = pl.tensor.sort32(s3_score_row, idx_init)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=64)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=256)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=1024)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=4096)
                    s3_sorted_gm = pl.assemble(s3_sorted_gm, s3_sorted_t, [0, 0])

                # Stage 3.4: Split top-k values and raw indices, then pad invalid tail slots.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_topk_extract"):
                    s3_topk_pairs = s3_sorted_gm[:, 0 : 2 * INDEX_TOPK]
                    s3_topk_i_raw = pl.tensor.gather(s3_topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    s3_valid_topk = pl.min(INDEX_TOPK, s3_ctx_len)
                    s3_idx_valid = pl.slice(s3_topk_i_raw, [1, INDEX_TOPK], [0, 0], valid_shape=[1, s3_valid_topk])
                    s3_idx_padded = pl.fillpad(s3_idx_valid, pad_value=pl.PadValue.min)
                    topk_idx = pl.assemble(topk_idx, s3_idx_padded, [b, 0])

            # ===== scope4: sparse attention with online softmax + latent-to-V projection =====
            # Inputs: q_proj, kv_cache, pe_cache, topk_idx
            # Output: dispatch_buf (cross-node dispatch buffer)
            attn_front = pl.create_tensor([BATCH, ATTN_OUT], dtype=pl.BF16)
            topk_idx_flat = pl.reshape(topk_idx, [BATCH * INDEX_TOPK])
            w_q_nope_to_latent_2d = pl.reshape(w_q_nope_to_latent, [NUM_HEADS * QK_NOPE_HEAD_DIM, KV_LORA_RANK])
            w_latent_to_v_2d = pl.reshape(w_latent_to_v, [NUM_HEADS * KV_LORA_RANK, V_HEAD_DIM])

            for b in pl.parallel(BATCH):
                attn_row = pl.create_tensor([1, ATTN_OUT], dtype=pl.FP32)
                sparse_k = pl.min(INDEX_TOPK, pl.read(seq_lens, [b]))
                topk_base = b * INDEX_TOPK

                for h in pl.parallel(NUM_HEADS):
                    q_col = h * QK_HEAD_DIM
                    v_col = h * V_HEAD_DIM

                    # Stage 4.1: Load q_pe and project q_nope into latent space.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_q_pe_load"):
                        q_pe = pl.cast(q_proj[b : b + 1, q_col + QK_NOPE_HEAD_DIM : q_col + QK_HEAD_DIM], target_type=pl.FP32)
                        q_pe_batch = pl.col_expand(
                            pl.full([MATMUL_ROW_PAD, QK_ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
                            q_pe,
                        )
                        q_nope_padded = pl.cast(
                            pl.full([MATMUL_ROW_PAD, QK_NOPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
                            target_type=pl.BF16,
                        )
                        q_nope_padded = pl.col_expand(q_nope_padded, q_proj[b : b + 1, q_col : q_col + QK_NOPE_HEAD_DIM])
                        q_nope_latent_batch = pl.full([MATMUL_ROW_PAD, KV_LORA_RANK], dtype=pl.FP32, value=0.0)

                    # Stage 4.2: Project q_nope to latent space chunk-by-chunk.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_q_nope_latent_proj"):
                        q_w_row = h * QK_NOPE_HEAD_DIM
                        for q0 in pl.range(0, KV_LORA_RANK, Q_LATENT_CHUNK):
                            w_qn_h_chunk = w_q_nope_to_latent_2d[q_w_row : q_w_row + QK_NOPE_HEAD_DIM, q0 : q0 + Q_LATENT_CHUNK]
                            q_nope_latent_part = pl.matmul(q_nope_padded, w_qn_h_chunk, out_dtype=pl.FP32)
                            q_nope_latent_batch = pl.assemble(q_nope_latent_batch, q_nope_latent_part, [0, q0])

                    # Stage 4.3-4.5: Online softmax over the sparse top-k set.
                    #
                    # Keep the dynamic kk loop inside one device task. Emitting
                    # one runtime task per top-k element creates hundreds of
                    # thousands of tiny dependent tasks for this standalone
                    # profile and can leave the a2a3 scheduler stuck waiting.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_softmax"):
                        topk_pos0 = pl.read(topk_idx_flat, [topk_base])
                        cache_s0 = b * MAX_SEQ + topk_pos0
                        kv_s0 = pl.cast(kv_cache[cache_s0 : cache_s0 + 1, 0 : KV_LORA_RANK], target_type=pl.FP32)
                        pe_s0 = pl.cast(pe_cache[cache_s0 : cache_s0 + 1, 0 : QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                        oi = pl.col_expand(pl.full([MATMUL_ROW_PAD, KV_LORA_RANK], dtype=pl.FP32, value=0.0), kv_s0)
                        pe_batch0 = pl.col_expand(pl.full([MATMUL_ROW_PAD, QK_ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0), pe_s0)
                        score_nope0 = pl.row_sum(pl.mul(q_nope_latent_batch, oi))
                        score_pe0 = pl.row_sum(pl.mul(q_pe_batch, pe_batch0))
                        mi = pl.mul(pl.add(score_nope0, score_pe0), ATTN_SCALE)
                        li = pl.exp(pl.sub(mi, mi))

                        for kk in pl.range(1, sparse_k):
                            topk_pos = pl.read(topk_idx_flat, [topk_base + kk])
                            cache_s = b * MAX_SEQ + topk_pos
                            kv_s = pl.cast(kv_cache[cache_s : cache_s + 1, 0 : KV_LORA_RANK], target_type=pl.FP32)
                            pe_s = pl.cast(pe_cache[cache_s : cache_s + 1, 0 : QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                            kv_batch = pl.col_expand(pl.full([MATMUL_ROW_PAD, KV_LORA_RANK], dtype=pl.FP32, value=0.0), kv_s)
                            pe_batch = pl.col_expand(pl.full([MATMUL_ROW_PAD, QK_ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0), pe_s)
                            score_nope = pl.row_sum(pl.mul(q_nope_latent_batch, kv_batch))
                            score_pe = pl.row_sum(pl.mul(q_pe_batch, pe_batch))
                            cur_mi = pl.mul(pl.add(score_nope, score_pe), ATTN_SCALE)
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), beta)
                            oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(kv_batch, beta))
                            mi = mi_new

                        ctx_latent_batch = pl.row_expand_div(oi, li)

                    # Stage 4.6: Project latent context back to V chunks.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_latent_cast"):
                        ctx_latent_bf16_batch = pl.cast(ctx_latent_batch, target_type=pl.BF16)

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_v_proj"):
                        ctx_v_batch = pl.full([MATMUL_ROW_PAD, V_HEAD_DIM], dtype=pl.FP32, value=0.0)
                        v_w_row = h * KV_LORA_RANK
                        for v0 in pl.range(0, V_HEAD_DIM, V_OUT_CHUNK):
                            wv_tile_chunk = w_latent_to_v_2d[v_w_row : v_w_row + KV_LORA_RANK, v0 : v0 + V_OUT_CHUNK]
                            v_part_batch = pl.matmul(ctx_latent_bf16_batch, wv_tile_chunk, out_dtype=pl.FP32)
                            ctx_v_batch = pl.assemble(ctx_v_batch, v_part_batch, [0, v0])

                    # Stage 4.7: Extract the leading V row and assemble it into attn_row.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_v_assemble"):
                        ctx_v = ctx_v_batch[0 : 1, 0 : V_HEAD_DIM]
                        attn_row = pl.assemble(attn_row, ctx_v, [0, v_col])

                # Stage 4.8: Cast and stash the finished attention row for this batch.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_attn_row_write"):
                    attn_front = pl.assemble(attn_front, pl.cast(attn_row, target_type=pl.BF16), [b, 0])

            # Stage 4.9: Route finished attention rows into the cross-node dispatch buffer.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_dispatch"):
                layer_id = pl.read(layer_id_t, [0])
                for b in pl.parallel(BATCH):
                    target_node = (b + layer_id) % EP_NODES
                    token_row = attn_front[b : b + 1, 0 : ATTN_OUT]
                    dispatch_buf = pl.assemble(dispatch_buf, token_row, [target_node, b, 0])

            return dispatch_buf

    return DeepSeekV32DecodeFrontScope1234


