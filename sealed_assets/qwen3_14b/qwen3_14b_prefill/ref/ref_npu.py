# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B prefill Scope 1+2+3.

Fuses the full single-layer prefill path: input RMSNorm, Q/K/V projection,
RoPE, KV cache update, causal attention, output projection, post-attention
RMSNorm, SwiGLU MLP, and the final residual path.

Dynamic batch design
--------------------
Every batch-dependent kernel signature dim is a `pl.dynamic(...)` variable
(`USER_BATCH_DYN` / `KV_CACHE_ROWS_DYN` / `BLOCK_TABLE_FLAT_DYN` /
`SLOT_MAPPING_DYN`), so a single compiled program serves any
`user_batch <= host KV-cache capacity`. Host allocates every
batch-dependent tensor at the user-visible batch (no host pad / no host
trim).

Unlike the decode path, prefill iterates the batch dim with step 1 (one
batch element per outer iteration) and every matmul tile's M dim is
governed by `TOK_TILE` (independent of batch). Therefore prefill does
NOT need decode's `batch_padded` round-up + `valid_shape` zero-pad +
vec-to-vec textract trim machinery on the batch axis. The per-token
`valid_tok = pl.min(TOK_TILE, seq_len_b - p0)` + `valid_shape` pattern
already handles intra-batch sequence-length variation.
"""
import pypto.language as pl

from config import (
    ATTN_SCALE,
    BATCH,
    BLOCK_SIZE,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    INTERMEDIATE,
    KV_HIDDEN,
    MAX_SEQ,
    NUM_HEADS,
    NUM_KV_HEADS,
)


# Dynamic dims for arbitrary user_batch support. Host allocates every
# batch-dependent tensor at the user-visible batch; the prefill kernel
# iterates the batch dim with step 1 so no batch-axis tile-alignment /
# valid_shape pad / textract trim is needed (contrast with decode).
USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
KV_CACHE_ROWS_DYN = pl.dynamic("KV_CACHE_ROWS_DYN")
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")
SLOT_MAPPING_DYN = pl.dynamic("SLOT_MAPPING_DYN")

# Tiling constants shared across the three scopes.
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
TOK_TILE = 64
Q_HEAD_BATCH = 5        # Q heads per attention group
Q_HEAD_BATCH_PAD = 8    # padded to 32-byte alignment (8 * 4 = 32)
Q_HEAD_PAD = 16         # padded Q rows for cube alignment
SEQ_TILE = 256          # sequence tile for attention
SB_BATCH = 64
MLP_OUT_CHUNK = 128


def build_qwen3_14b_prefill_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
):
    # The `batch` parameter documents the host-buffer capacity; it is no
    # longer baked into the program. Every
    # batch-dependent kernel signature dim is a pl.dynamic() variable
    # (USER_BATCH_DYN / KV_CACHE_ROWS_DYN / BLOCK_TABLE_FLAT_DYN /
    # SLOT_MAPPING_DYN), so a single compiled program serves any
    # user_batch <= host capacity.
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    q_per_kv = num_heads // num_kv_heads
    max_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    half_dim = head_dim // 2
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK
    mlp_out_blocks = intermediate_size // MLP_OUT_CHUNK
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)
    max_ctx_blocks = (max_seq + SEQ_TILE - 1) // SEQ_TILE
    hidden_inv = 1.0 / hidden
    head_dim_inv = 1.0 / head_dim

    @pl.program
    class Qwen314BPrefillProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_14b_prefill(
            self,
            hidden_states: pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            q_norm_weight: pl.Tensor[[1, head_dim], pl.FP32],
            k_norm_weight: pl.Tensor[[1, head_dim], pl.FP32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[SLOT_MAPPING_DYN], pl.INT32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, head_dim], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, head_dim], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            w_gate: pl.Tensor[[hidden, intermediate_size], pl.BF16],
            w_up: pl.Tensor[[hidden, intermediate_size], pl.BF16],
            w_down: pl.Tensor[[intermediate_size, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16]],
        ) -> pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16]:
            # Runtime user_batch (host-visible batch). Outer batch loop
            # iterates with step 1 so every matmul tile's M dim is fully
            # determined by TOK_TILE (no batch-axis pad / trim needed).
            user_batch = pl.tensor.dim(hidden_states, 0)
            for b in pl.parallel(0, user_batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    # ── Scope 1: input RMSNorm + Q/K/V projection ──
                    normed_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)

                    # Stage 1.1: RMSNorm (vector ops).
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                        partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                             valid_shape=[1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            partial_sq = pl.add(
                                partial_sq,
                                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, TOK_TILE]),
                            )
                        variance = pl.reshape(
                            pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                            [TOK_TILE, 1],
                        )
                        inv_rms = pl.recip(pl.sqrt(variance))

                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                             valid_shape=[1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                            normed_tile = pl.assemble(normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    # Stage 1.2: Q projection (matmul + matmul_acc, FP32 output).
                    q_proj_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    for ob_chunk in pl.parallel(0, q_out_blocks, 4):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                            for ob in pl.range(ob_chunk, ob_chunk + 4):
                                q0 = ob * Q_OUT_CHUNK
                                tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                tile_w = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [0, q0])
                                q_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    tile_w_i = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                                    q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_w_i)
                                q_proj_tile = pl.assemble(q_proj_tile, q_acc, [0, q0])

                    # Stage 1.3: K/V projection (matmul + matmul_acc in single incore).
                    k_proj_tile = pl.create_tensor([TOK_TILE, kv_hidden], dtype=pl.FP32)
                    v_proj_tile = pl.create_tensor([TOK_TILE, kv_hidden], dtype=pl.FP32)
                    for ob_chunk in pl.parallel(0, kv_out_blocks, 4):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj"):
                            for ob in pl.range(ob_chunk, ob_chunk + 4):
                                kv0 = ob * KV_OUT_CHUNK

                                tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                tile_wk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                                k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    tile_wk_i = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                    k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                                k_proj_tile = pl.assemble(k_proj_tile, k_acc, [0, kv0])

                                tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                tile_wv = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                                v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    tile_wv_i = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                    v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                                v_proj_tile = pl.assemble(v_proj_tile, v_acc, [0, kv0])

                    # Stage 1.4: Q/K per-head RMSNorm (FP32 in-place on proj tiles).
                    for qh_chunk in pl.parallel(0, num_heads, num_heads):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_norm"):
                            for qh in pl.range(qh_chunk, qh_chunk + num_heads):
                                q_col = qh * head_dim
                                q_head = pl.slice(q_proj_tile, [TOK_TILE, head_dim], [0, q_col])
                                q_sq = pl.reshape(
                                    pl.row_sum(pl.mul(q_head, q_head)),
                                    [TOK_TILE, 1],
                                )
                                q_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, head_dim_inv), EPS)))
                                q_normed = pl.col_expand_mul(
                                    pl.row_expand_mul(q_head, q_inv_rms),
                                    q_norm_weight,
                                )
                                q_proj_tile = pl.assemble(q_proj_tile, q_normed, [0, q_col])
                    for kh_chunk in pl.parallel(0, num_kv_heads, num_kv_heads):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_norm"):
                            for kh in pl.range(kh_chunk, kh_chunk + num_kv_heads):
                                k_col = kh * head_dim
                                k_head = pl.slice(k_proj_tile, [TOK_TILE, head_dim], [0, k_col])
                                k_sq = pl.reshape(
                                    pl.row_sum(pl.mul(k_head, k_head)),
                                    [TOK_TILE, 1],
                                )
                                k_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(k_sq, head_dim_inv), EPS)))
                                k_normed = pl.col_expand_mul(
                                    pl.row_expand_mul(k_head, k_inv_rms),
                                    k_norm_weight,
                                )
                                k_proj_tile = pl.assemble(k_proj_tile, k_normed, [0, k_col])

                    # ── Scope 2: RoPE + KV cache update + causal attention ──
                    attn_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)
                    for ti in pl.range(valid_tok):
                        pos = p0 + ti
                        ctx_len = pos + 1
                        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                        cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                        sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                        cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                        cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                        sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                        sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                        # Stage 2.1: K RoPE + cache update + V cache + Q RoPE + pad.
                        all_q_padded = pl.create_tensor([total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)
                        for gi_chunk in pl.parallel(0, total_q_groups, total_q_groups):
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_pad"):
                                for gi in pl.range(gi_chunk, gi_chunk + total_q_groups):
                                    all_q_padded = pl.assemble(
                                        all_q_padded,
                                        pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                                        [gi * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                                    )
                        cache_slot = pl.cast(pl.tensor.read(slot_mapping, [b * max_seq + pos]), pl.INDEX)
                        cache_slot_block = cache_slot // BLOCK_SIZE
                        cache_slot_offset = cache_slot - cache_slot_block * BLOCK_SIZE
                        for ki_chunk in pl.parallel(0, num_kv_heads, 8):
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
                                for ki in pl.range(ki_chunk, ki_chunk + 8):
                                    # K RoPE + cache update.
                                    kv_col = ki * head_dim
                                    k_lo = pl.reshape(pl.slice(k_proj_tile, [1, half_dim], [ti, kv_col]), [1, half_dim])
                                    k_hi = pl.reshape(pl.slice(k_proj_tile, [1, half_dim], [ti, kv_col + half_dim]), [1, half_dim])
                                    rot_lo = pl.sub(
                                        pl.col_expand_mul(k_lo, cos_lo),
                                        pl.col_expand_mul(k_hi, sin_lo),
                                    )
                                    rot_hi = pl.add(
                                        pl.col_expand_mul(k_hi, cos_hi),
                                        pl.col_expand_mul(k_lo, sin_hi),
                                    )
                                    cache_row = (cache_slot_block * num_kv_heads + ki) * BLOCK_SIZE + cache_slot_offset
                                    k_cache = pl.assemble(
                                        k_cache,
                                        pl.cast(rot_lo, target_type=pl.BF16),
                                        [cache_row, 0],
                                    )
                                    k_cache = pl.assemble(
                                        k_cache,
                                        pl.cast(rot_hi, target_type=pl.BF16),
                                        [cache_row, half_dim],
                                    )
                                    # V cache update.
                                    v_cache = pl.assemble(
                                        v_cache,
                                        pl.cast(
                                            pl.reshape(pl.slice(v_proj_tile, [1, head_dim], [ti, ki * head_dim]), [1, head_dim]),
                                            target_type=pl.BF16,
                                        ),
                                        [cache_row, 0],
                                    )
                                    # Q RoPE + pad.
                                    q_base = ki * q_per_kv
                                    for qi in pl.range(Q_HEAD_BATCH):
                                        q_col = (q_base + qi) * head_dim
                                        q_lo = pl.reshape(pl.slice(q_proj_tile, [1, half_dim], [ti, q_col]), [1, half_dim])
                                        q_hi = pl.reshape(pl.slice(q_proj_tile, [1, half_dim], [ti, q_col + half_dim]), [1, half_dim])
                                        rot_lo_bf16 = pl.cast(
                                            pl.sub(
                                                pl.col_expand_mul(q_lo, cos_lo),
                                                pl.col_expand_mul(q_hi, sin_lo),
                                            ),
                                            target_type=pl.BF16,
                                        )
                                        rot_hi_bf16 = pl.cast(
                                            pl.add(
                                                pl.col_expand_mul(q_hi, cos_hi),
                                                pl.col_expand_mul(q_lo, sin_hi),
                                            ),
                                            target_type=pl.BF16,
                                        )
                                        all_q_padded = pl.assemble(all_q_padded, rot_lo_bf16, [ki * Q_HEAD_PAD + qi, 0])
                                        all_q_padded = pl.assemble(all_q_padded, rot_hi_bf16, [ki * Q_HEAD_PAD + qi, half_dim])

                        attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)

                        for gi in pl.range(total_q_groups):
                            kvh = gi // q_groups
                            qg = gi - kvh * q_groups
                            q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH

                            q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, head_dim], [gi * Q_HEAD_PAD, 0])

                            all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                            all_exp_padded = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                            all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                            all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH_PAD, 1], dtype=pl.FP32)
                            all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH_PAD, 1], dtype=pl.FP32)

                            # Stage 2.2: QK matmul for all active sb blocks.
                            for sb_chunk in pl.parallel(0, ctx_blocks, SB_BATCH):
                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                                    for si in pl.range(SB_BATCH):
                                        sb = sb_chunk + si
                                        if sb < ctx_blocks:
                                            block_table_idx = b * max_blocks_per_seq + sb
                                            pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                                            cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                                            k_tile = pl.slice(k_cache, [SEQ_TILE, head_dim], [cache_row0, 0])
                                            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                                            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                            # Stage 2.3: softmax for all active sb blocks.
                            for sb_chunk in pl.parallel(0, ctx_blocks, SB_BATCH):
                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                                    for si in pl.range(SB_BATCH):
                                        sb = sb_chunk + si
                                        if sb < ctx_blocks:
                                            s0 = sb * SEQ_TILE
                                            valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                            scores_valid = pl.slice(
                                                all_raw_scores, [Q_HEAD_BATCH_PAD, SEQ_TILE],
                                                [sb * Q_HEAD_PAD, 0],
                                                valid_shape=[Q_HEAD_BATCH, valid_len],
                                            )
                                            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                                            scores = pl.mul(scores_padded, attn_scale)
                                            cur_mi = pl.row_max(scores)
                                            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                            cur_li = pl.row_sum(pl.cast(exp_scores_bf16, target_type=pl.FP32))
                                            all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                                            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_BATCH_PAD, 0])
                                            all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_BATCH_PAD, 0])

                            # Stage 2.4: SV matmul for all active sb blocks.
                            for sb_chunk in pl.parallel(0, ctx_blocks, SB_BATCH):
                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                                    for si in pl.range(SB_BATCH):
                                        sb = sb_chunk + si
                                        if sb < ctx_blocks:
                                            block_table_idx = b * max_blocks_per_seq + sb
                                            pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                                            cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                                            exp_tile = pl.slice(all_exp_padded, [Q_HEAD_PAD, SEQ_TILE], [sb * Q_HEAD_PAD, 0])
                                            v_tile = pl.slice(v_cache, [SEQ_TILE, head_dim], [cache_row0, 0])
                                            oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                                            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                            # Stage 2.5: online softmax accumulation.
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax_init"):
                                oi = pl.full([Q_HEAD_BATCH_PAD, head_dim], dtype=pl.FP32, value=0.0)
                                li_flat = pl.full([1, Q_HEAD_BATCH_PAD], dtype=pl.FP32, value=0.0)
                                li = pl.reshape(li_flat, [Q_HEAD_BATCH_PAD, 1])
                                mi_flat = pl.full([1, Q_HEAD_BATCH_PAD], dtype=pl.FP32, value=0.0)
                                mi = pl.reshape(mi_flat, [Q_HEAD_BATCH_PAD, 1])

                            for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                                    for si in pl.range(SB_BATCH):
                                        sb = sb0 + si
                                        if sb < ctx_blocks:
                                            oi_sb = pl.slice(all_oi_tmp, [Q_HEAD_BATCH_PAD, head_dim], [sb * Q_HEAD_PAD, 0])
                                            mi_sb = pl.slice(all_cur_mi, [Q_HEAD_BATCH_PAD, 1], [sb * Q_HEAD_BATCH_PAD, 0])
                                            li_sb = pl.slice(all_cur_li, [Q_HEAD_BATCH_PAD, 1], [sb * Q_HEAD_BATCH_PAD, 0])
                                            if sb == 0:
                                                oi = oi_sb
                                                li = li_sb
                                                mi = mi_sb
                                            else:
                                                mi_new = pl.maximum(mi, mi_sb)
                                                alpha = pl.exp(pl.sub(mi, mi_new))
                                                beta = pl.exp(pl.sub(mi_sb, mi_new))
                                                li = pl.add(pl.mul(alpha, li), pl.mul(beta, li_sb))
                                                oi = pl.add(pl.row_expand_mul(oi, alpha),
                                                            pl.row_expand_mul(oi_sb, beta))
                                                mi = mi_new

                            # Finalize ctx = oi / li and write back row by row.
                            ctx_tmp = pl.create_tensor([Q_HEAD_BATCH_PAD, head_dim], dtype=pl.FP32)
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="attention_context"):
                                ctx = pl.row_expand_div(oi, li)
                                ctx_tmp = pl.assemble(ctx_tmp, ctx, [0, 0])
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="attention_writeback"):
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * head_dim
                                    row = pl.slice(ctx_tmp, [1, head_dim], [qi, 0])
                                    row_bf16 = pl.cast(row, target_type=pl.BF16)
                                    attn_row = pl.assemble(attn_row, row_bf16, [0, q_col])

                        # Keep the attention row in the local tile.
                        attn_tile = pl.assemble(attn_tile, attn_row, [ti, 0])
                    # ── Scope 3: output projection + residual + post RMSNorm + MLP ──
                    # Stage 3.1: Output projection + first residual.
                    resid1_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    for ob in pl.range(q_out_blocks):
                        o0 = ob * Q_OUT_CHUNK
                        # Cube matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj"):
                            tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_w = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                            o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_w_i = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                                o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)
                            resid1_tile = pl.assemble(resid1_tile, o_acc, [0, o0])

                        # Add the residual path.
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_residual"):
                            resid_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, Q_OUT_CHUNK], [b, p0, o0],
                                             valid_shape=[1, valid_tok, Q_OUT_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_CHUNK],
                            )
                            mm_out = pl.slice(resid1_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0])
                            resid1_tile = pl.assemble(resid1_tile, pl.add(mm_out, resid_chunk), [0, o0])

                    # Stage 3.2: Post-attention RMSNorm.
                    post_norm_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                        sq_sum = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            post_x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            sq_sum = pl.add(
                                sq_sum,
                                pl.reshape(pl.row_sum(pl.mul(post_x_chunk, post_x_chunk)), [1, TOK_TILE]),
                            )
                        post_inv_rms = pl.recip(pl.sqrt(pl.reshape(
                            pl.add(pl.mul(sq_sum, hidden_inv), EPS),
                            [TOK_TILE, 1],
                        )))

                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            post_x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(post_x_chunk, post_inv_rms),
                                gamma,
                            )
                            post_norm_tile = pl.assemble(post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    # Stage 3.3a: MLP gate/up + SiLU.
                    mlp_silu_tile = pl.create_tensor([TOK_TILE, intermediate_size], dtype=pl.BF16)
                    for ob in pl.range(mlp_out_blocks):
                        o0 = ob * MLP_OUT_CHUNK

                        # Gate matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wgi = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                        # Up matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wui = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                up_acc = pl.matmul_acc(up_acc, pci, wui)

                        # SiLU activation: silu(gate) * up -> BF16.
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                            mlp_silu_tile = pl.assemble(mlp_silu_tile, mlp_chunk_bf16, [0, o0])

                    # Stage 3.3b: Down projection (matmul_acc chain) + final residual -> output.
                    for dob in pl.range(hidden_blocks):
                        d0 = dob * K_CHUNK
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj"):
                            mlp_chunk_0 = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, 0])
                            w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                            for ob in pl.range(1, mlp_out_blocks):
                                o0 = ob * MLP_OUT_CHUNK
                                mlp_chunk_i = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, o0])
                                w_down_chunk_i = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                                down_acc = pl.matmul_acc(down_acc, mlp_chunk_i, w_down_chunk_i)

                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj_residual"):
                            out_chunk = pl.add(
                                down_acc,
                                pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, d0]),
                            )
                            out_chunk_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
                            out = pl.assemble(out, out_chunk_bf16, [b, p0, d0])

            return out

    return Qwen314BPrefillProgram
