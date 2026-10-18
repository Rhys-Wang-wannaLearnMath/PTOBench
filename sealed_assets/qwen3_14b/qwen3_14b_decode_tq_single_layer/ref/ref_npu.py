# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3-14B single-layer TQ decode implementation closure."""

import pypto.language as pl
from turboquant_kv import turboquant_kv_quantize, turboquant_kv_dequant_chunk
from rms_lm_head import rms_lm_head
from config import (
    ATTN_SCALE,
    BATCH,
    BLOCK_SIZE,
    EPS,
    HALF_DIM,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    INTERMEDIATE,
    KV_HIDDEN,
    MAX_BLOCKS_PER_SEQ,
    NUM_HEADS,
    NUM_KV_HEADS,
    Q_GROUPS,
    Q_HEAD_BATCH,
    Q_HEAD_PAD,
    Q_PER_KV,
    TOTAL_Q_GROUPS,
    VOCAB,
    N_LEVELS,
)

BATCH_TILE = 16
K_CHUNK = 256
Q_OUT_CHUNK = 256

USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")
ROPE_SEQ_DYN = pl.dynamic("ROPE_SEQ_DYN")
LAYER_DYN = pl.dynamic("LAYER_DYN")
LAYER_HIDDEN_ROWS_DYN = pl.dynamic("LAYER_HIDDEN_ROWS_DYN")
LAYER_INTER_ROWS_DYN = pl.dynamic("LAYER_INTER_ROWS_DYN")
QUANT_CACHE_ROWS_DYN = pl.dynamic("QUANT_CACHE_ROWS_DYN")

SCOPE1_K_CHUNK = 512
KV_OUT_CHUNK = 64
CMP_CHUNK = 32
MAX_CTX_BLOCKS = MAX_BLOCKS_PER_SEQ
SCOPE1_HIDDEN_BLOCKS = HIDDEN // SCOPE1_K_CHUNK
HIDDEN_BLOCKS = HIDDEN // K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK
MLP_OUT_BLOCKS = INTERMEDIATE // 256

@pl.jit.inline
def decode_layer_tq(
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    quant_k_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
    quant_v_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
    quant_k_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
    quant_v_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
    rot_matrices: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HEAD_DIM], pl.BF16],
    tq_codebook: pl.Tensor[[CMP_CHUNK, N_LEVELS], pl.FP32],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    # NOTE: Q_OUT_BLOCKS, HIDDEN_BLOCKS, MLP_OUT_BLOCKS are used directly
    # in pl.spmd(...) — pl.spmd outlines its body to a top-level function
    # and SSA-verifies the block count outside the JIT-inlined scope, so a
    # local alias defined here would trip "used outside its defining scope".
    head_dim_inv = HEAD_DIM_INV
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cmp_cache_rows = pl.tensor.dim(quant_k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    batch_padded = BATCH
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cmp_base = layer_idx * decode_layer_cmp_cache_rows
    rot_base = layer_idx * HEAD_DIM
    rot_slice = pl.slice(rot_matrices, [HEAD_DIM, HEAD_DIM], [rot_base, 0])

    # Intermediate FP32 tensors between scope 1 and scope 2.
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    # ── Scope 1: input RMSNorm + Q/K/V projection ──
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        normed_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
            partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                sq_k0 = kb * SCOPE1_K_CHUNK
                sq_chunk = pl.cast(
                    pl.slice(current_hidden, [BATCH_TILE, SCOPE1_K_CHUNK], [b0, sq_k0]),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(sq_chunk, sq_chunk)), [1, BATCH_TILE]),
                )
            variance = pl.reshape(
                pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                [BATCH_TILE, 1],
            )
            inv_rms = pl.recip(pl.sqrt(variance))

            for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                norm_k0 = kb * SCOPE1_K_CHUNK
                norm_chunk = pl.cast(
                    pl.slice(current_hidden, [BATCH_TILE, SCOPE1_K_CHUNK], [b0, norm_k0]),
                    target_type=pl.FP32,
                )
                gamma = pl.slice(input_rms_weight, [1, SCOPE1_K_CHUNK], [layer_idx, norm_k0])
                normed = pl.col_expand_mul(pl.row_expand_mul(norm_chunk, inv_rms), gamma)
                normed_tile = pl.assemble(
                    normed_tile, pl.cast(normed, target_type=pl.BF16), [0, norm_k0],
                )

        # Q projection.
        for q0 in pl.parallel(0, HIDDEN, Q_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                    q_k0 = kb * SCOPE1_K_CHUNK
                    q_tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, q_k0])
                    q_tile_b = pl.slice(wq, [SCOPE1_K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + q_k0, q0])
                    if q_k0 == 0:
                        q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                    else:
                        q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
                q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

        # K/V projection.
        for kv0 in pl.parallel(0, KV_HIDDEN, KV_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj"):
                k_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                    k_k0 = kb * SCOPE1_K_CHUNK
                    k_tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k_k0])
                    k_tile_b = pl.slice(wk, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k_k0, kv0])
                    if k_k0 == 0:
                        k_acc = pl.matmul(k_tile_a, k_tile_b, out_dtype=pl.FP32)
                    else:
                        k_acc = pl.matmul_acc(k_acc, k_tile_a, k_tile_b)
                k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj"):
                v_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                    v_k0 = kb * SCOPE1_K_CHUNK
                    v_tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, v_k0])
                    v_tile_b = pl.slice(wv, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + v_k0, kv0])
                    if v_k0 == 0:
                        v_acc = pl.matmul(v_tile_a, v_tile_b, out_dtype=pl.FP32)
                    else:
                        v_acc = pl.matmul_acc(v_acc, v_tile_a, v_tile_b)
                v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

    # Q/K per-head norm (grouped by KV head).
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm"):
            for h in pl.range(NUM_KV_HEADS):
                q0 = h * Q_PER_KV * HEAD_DIM
                q_chunk = pl.reshape(
                    pl.slice(q_proj, [BATCH_TILE, Q_HEAD_BATCH * HEAD_DIM], [b0, q0]),
                    [BATCH_TILE * Q_HEAD_BATCH, HEAD_DIM],
                )
                q_sq_sum = pl.row_sum(pl.mul(q_chunk, q_chunk))
                q_inv_rms = pl.rsqrt(pl.add(pl.mul(q_sq_sum, head_dim_inv), EPS))
                q_chunk_norm = pl.col_expand_mul(
                    pl.row_expand_mul(q_chunk, q_inv_rms),
                    pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                )
                q_chunk_norm_flat = pl.reshape(q_chunk_norm, [BATCH_TILE, Q_HEAD_BATCH * HEAD_DIM])
                q_proj_norm = pl.assemble(q_proj_norm, q_chunk_norm_flat, [b0, q0])

                k0 = h * HEAD_DIM
                k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [b0, k0])
                k_sq_sum = pl.row_sum(pl.mul(k_chunk, k_chunk))
                k_inv_rms = pl.rsqrt(pl.add(pl.mul(k_sq_sum, head_dim_inv), EPS))
                k_chunk_norm = pl.col_expand_mul(
                    pl.row_expand_mul(k_chunk, k_inv_rms),
                    pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                )
                k_proj_norm = pl.assemble(k_proj_norm, k_chunk_norm, [b0, k0])

    # ── Scope 2: RoPE + TQ inline KV quantization + fused dequant attention ──
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    all_q_padded = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16,
    )

    # Per-batch: TQ inline KV quantization (via turboquant_kv_quantize) + Q RoPE.
    for b in pl.parallel(user_batch):
        ctx_len = pl.tensor.read(seq_lens, [b])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE
        cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, HALF_DIM], [0, 0])
        cos_hi = pl.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
        sin_lo = pl.slice(sin_row, [1, HALF_DIM], [0, 0])
        sin_hi = pl.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])

        # Allocate GM buffers for turboquant_kv_quantize.
        # Rows 1-15 are uninitialized; turboquant_kv_quantize processes each row
        # independently and we only consume row 0 of the output.
        k_padded = pl.create_tensor([16, KV_HIDDEN], dtype=pl.FP32)
        v_padded = pl.create_tensor([16, KV_HIDDEN], dtype=pl.FP32)
        cos_lo_all = pl.create_tensor([16, HALF_DIM], dtype=pl.FP32)
        cos_hi_all = pl.create_tensor([16, HALF_DIM], dtype=pl.FP32)
        sin_lo_all = pl.create_tensor([16, HALF_DIM], dtype=pl.FP32)
        sin_hi_all = pl.create_tensor([16, HALF_DIM], dtype=pl.FP32)
        quant_k_temp = pl.create_tensor([16, KV_HIDDEN // 2], dtype=pl.UINT8)
        quant_v_temp = pl.create_tensor([16, KV_HIDDEN // 2], dtype=pl.UINT8)
        k_scales_buf = pl.create_tensor([16, NUM_KV_HEADS], dtype=pl.FP32)
        v_scales_buf = pl.create_tensor([16, NUM_KV_HEADS], dtype=pl.FP32)

        # Pad K/V [1, KV_HIDDEN] -> [16, KV_HIDDEN] and expand cos/sin.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_pad"):
            k_all = pl.slice(k_proj_norm, [1, KV_HIDDEN], [b, 0])
            k_padded = pl.assemble(k_padded, k_all, [0, 0])

            v_all = pl.slice(v_proj, [1, KV_HIDDEN], [b, 0])
            v_padded = pl.assemble(v_padded, v_all, [0, 0])

            _ones = pl.full([16, HALF_DIM], dtype=pl.FP32, value=1.0)
            cos_lo_all = pl.assemble(
                cos_lo_all, pl.col_expand_mul(_ones, cos_lo), [0, 0],
            )
            cos_hi_all = pl.assemble(
                cos_hi_all, pl.col_expand_mul(_ones, cos_hi), [0, 0],
            )
            sin_lo_all = pl.assemble(
                sin_lo_all, pl.col_expand_mul(_ones, sin_lo), [0, 0],
            )
            sin_hi_all = pl.assemble(
                sin_hi_all, pl.col_expand_mul(_ones, sin_hi), [0, 0],
            )

        # Reuse prefill's TQ KV quantization: K RoPE + L2 norm + normalize +
        # rotate + Lloyd-Max quantize for both K and V.
        turboquant_kv_quantize(
            k_padded, v_padded, rot_slice,
            cos_lo_all, cos_hi_all, sin_lo_all, sin_hi_all,
            quant_k_temp, quant_v_temp, k_scales_buf, v_scales_buf,
        )

        # Scatter row 0 results to per-head cache positions.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
            for ki in pl.range(NUM_KV_HEADS):
                kv_col = ki * HALF_DIM
                quant_cache_row = layer_cmp_base + (slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + slot_offset

                # K: extract row 0, scatter to quantized cache + scale.
                # Index temps are nibble-packed, so each head is HALF_DIM wide.
                k_idx_u8 = pl.slice(quant_k_temp, [1, HALF_DIM], [0, kv_col])
                quant_k_cache = pl.assemble(quant_k_cache, k_idx_u8, [quant_cache_row, 0])
                k_scale = pl.read(k_scales_buf, [0, ki])
                pl.write(quant_k_scales, [quant_cache_row, 0], k_scale)

                # V: extract row 0, scatter to quantized cache + scale.
                v_idx_u8 = pl.slice(quant_v_temp, [1, HALF_DIM], [0, kv_col])
                quant_v_cache = pl.assemble(quant_v_cache, v_idx_u8, [quant_cache_row, 0])
                v_scale = pl.read(v_scales_buf, [0, ki])
                pl.write(quant_v_scales, [quant_cache_row, 0], v_scale)

        # Q RoPE + pad into all_q_padded — one head at a time (avoids pl.slice
        # batch bug on [Q_HEAD_BATCH, HEAD_DIM] tiles).
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_q_rope"):
            zero_row = pl.cast(pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
            for ki in pl.range(NUM_KV_HEADS):
                q_base = ki * Q_PER_KV
                row_base = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD
                for qi in pl.range(Q_HEAD_BATCH):
                    q_head = pl.reshape(
                        pl.slice(q_proj_norm, [1, HEAD_DIM], [b, (q_base + qi) * HEAD_DIM]),
                        [1, HEAD_DIM],
                    )
                    q_lo = pl.slice(q_head, [1, HALF_DIM], [0, 0])
                    q_hi = pl.slice(q_head, [1, HALF_DIM], [0, HALF_DIM])
                    rot_lo = pl.cast(
                        pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)),
                        target_type=pl.BF16,
                    )
                    rot_hi = pl.cast(
                        pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)),
                        target_type=pl.BF16,
                    )
                    all_q_padded = pl.assemble(all_q_padded, rot_lo, [row_base + qi, 0])
                    all_q_padded = pl.assemble(all_q_padded, rot_hi, [row_base + qi, HALF_DIM])
                for pi in pl.range(Q_HEAD_PAD - Q_HEAD_BATCH):
                    all_q_padded = pl.assemble(all_q_padded, zero_row,
                        [row_base + Q_HEAD_BATCH + pi, 0])
        # ── K/V scales + norms: TODO — requires scalar write or layout change ──
        # Constraint: Ascend requires FP32 tiles with ≥8 rows (32-byte column align).
        # Cannot create [1, 1] FP32 tiles, so individual scale writes are blocked.
        # Possible fixes:
        #   1. Change quant_k_scales from [ROWS, 1] to [ROWS, 8] (wider columns)
        #   2. Use pl.tensor.write scalar API (if available)
        #   3. Compute scales inline during dequant instead of storing them

        # ── Attention: per Q-group, matching prefill pattern ──
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        for gi in pl.range(TOTAL_Q_GROUPS):
            kvh = gi // Q_GROUPS
            qg = gi - kvh * Q_GROUPS
            q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH

            q_padded = pl.slice(
                all_q_padded, [Q_HEAD_PAD, HEAD_DIM],
                [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0],
            )
            # Per-group GM buffers (matching prefill layout).
            all_raw_scores = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32,
            )
            all_exp_padded = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.BF16,
            )
            all_oi_tmp = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32,
            )
            all_cur_mi = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, 1], dtype=pl.FP32,
            )
            all_cur_li = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, 1], dtype=pl.FP32,
            )

            max_blocks = pl.tensor.dim(block_table, 0) // user_batch

            # Stage 2.2: QK inline dequant + matmul.
            # BF16 buffer filled directly by dequant function.
            k_bf16_buf = pl.create_tensor([BLOCK_SIZE, HEAD_DIM], dtype=pl.BF16)

            for sb in pl.range(ctx_blocks):
                bt_idx = b * max_blocks + sb
                pbid = pl.cast(pl.tensor.read(block_table, [bt_idx]), pl.INDEX)
                row_base = layer_cmp_base + (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE

                # 2.2a: Dequant K: gather → renorm → scale → unrotate → BF16 (scopes inside function).
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_dequant"):
                    k_scales_full = pl.slice(quant_k_scales, [BLOCK_SIZE, 1], [row_base, 0])
                    for k_sub in pl.range(BLOCK_SIZE // CMP_CHUNK):
                        sub_row = k_sub * CMP_CHUNK
                        k_indices = pl.slice(
                            quant_k_cache, [CMP_CHUNK, HALF_DIM], [row_base + sub_row, 0],
                            valid_shape=[CMP_CHUNK, HALF_DIM],
                        )
                        k_scales_sub = pl.slice(k_scales_full, [CMP_CHUNK, 1], [sub_row, 0])
                        chunk_out = turboquant_kv_dequant_chunk(
                            k_indices, k_scales_sub, tq_codebook, rot_slice,
                        )
                        k_bf16_buf = pl.assemble(k_bf16_buf, chunk_out, [sub_row, 0])

                # 2.2e: QK matmul — both inputs as GM-slice BF16.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                    k_tile = pl.slice(k_bf16_buf, [BLOCK_SIZE, HEAD_DIM], [0, 0])
                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                    all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

            # Stage 2.3: softmax.
            for sb in pl.range(ctx_blocks):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                    valid_len = pl.min(BLOCK_SIZE, ctx_len - sb * BLOCK_SIZE)
                    raw = pl.slice(
                        all_raw_scores, [Q_HEAD_PAD, BLOCK_SIZE],
                        [sb * Q_HEAD_PAD, 0],
                    )
                    scores_scaled = pl.mul(raw, decode_attn_scale)
                    scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_PAD // 2, valid_len)
                    scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                    cur_mi = pl.row_max(scores)
                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                    exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                    cur_li = pl.row_sum(exp_scores_fp32)
                    all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                    all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_PAD, 0])
                    all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_PAD, 0])


            # Stage 2.4: SV inline dequant + matmul.
            for sb in pl.range(ctx_blocks):
                bt_idx = b * max_blocks + sb
                pbid = pl.cast(pl.tensor.read(block_table, [bt_idx]), pl.INDEX)
                row_base = layer_cmp_base + (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                v_tile_full = pl.create_tensor([BLOCK_SIZE, HEAD_DIM], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_dequant"):
                    for v_sub in pl.range(BLOCK_SIZE // CMP_CHUNK):
                        v_sub_row = row_base + v_sub * CMP_CHUNK
                        v_indices = pl.slice(
                            quant_v_cache, [CMP_CHUNK, HALF_DIM], [v_sub_row, 0],
                            valid_shape=[CMP_CHUNK, HALF_DIM],
                        )
                        v_scales = pl.slice(
                            quant_v_scales, [CMP_CHUNK, 1], [v_sub_row, 0],
                            valid_shape=[CMP_CHUNK, 1],
                        )
                        chunk_out = turboquant_kv_dequant_chunk(
                            v_indices, v_scales, tq_codebook, rot_slice,
                        )
                        v_tile_full = pl.assemble(v_tile_full, chunk_out, [v_sub * CMP_CHUNK, 0])
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                    exp_tile = pl.slice(all_exp_padded, [Q_HEAD_PAD, BLOCK_SIZE], [sb * Q_HEAD_PAD, 0])
                    oi_tmp = pl.matmul(exp_tile, v_tile_full, out_dtype=pl.FP32)
                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

            # Stage 2.5: online softmax accumulation.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax_init"):
                oi = pl.slice(all_oi_tmp, [Q_HEAD_PAD, HEAD_DIM], [0, 0])
                mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [0, 0])
                li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [0, 0])
            for sb in pl.range(1, ctx_blocks):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                    oi_sb = pl.slice(all_oi_tmp, [Q_HEAD_PAD, HEAD_DIM], [sb * Q_HEAD_PAD, 0])
                    mi_sb = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                    li_sb = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                    mi_new = pl.maximum(mi, mi_sb)
                    alpha = pl.exp(pl.sub(mi, mi_new))
                    beta = pl.exp(pl.sub(mi_sb, mi_new))
                    li = pl.add(pl.mul(alpha, li), pl.mul(beta, li_sb))
                    oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_sb, beta))
                    mi = mi_new

            # Finalize: ctx = oi / li, write to attn_out.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="attention_writeback"):
                ctx = pl.row_expand_div(oi, li)
                ctx_flat_bf16 = pl.cast(
                    pl.reshape(ctx, [1, Q_HEAD_PAD * HEAD_DIM]),
                    target_type=pl.BF16,
                )
                attn_out = pl.assemble(attn_out, ctx_flat_bf16, [b, q_base * HEAD_DIM])

    # ── Scope 3: output projection + residual + MLP (SPMD + UP_DOWN) ──
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.FP32)

        # out_proj: fused cube matmul + vec residual.
        for ob in pl.spmd(Q_OUT_BLOCKS, name_hint="out_proj",
                          optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            o0 = ob * Q_OUT_CHUNK
            a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
            w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, o0])
            o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + k0, o0])
                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)
            resid = pl.cast(pl.slice(current_hidden, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]), target_type=pl.FP32)
            resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

        # Post-attention RMSNorm.
        post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN_BLOCKS):
                post_sq_k0 = kb * K_CHUNK
                post_sq_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_sq_k0])
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(post_sq_chunk, post_sq_chunk)), [1, BATCH_TILE]))
            inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
            for kb in pl.range(HIDDEN_BLOCKS):
                post_norm_k0 = kb * K_CHUNK
                post_norm_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_norm_k0])
                post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, post_norm_k0])
                post_normed = pl.col_expand_mul(
                    pl.row_expand_mul(post_norm_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1])), post_gamma,
                )
                post_norm_tile = pl.assemble(post_norm_tile, pl.cast(post_normed, target_type=pl.BF16), [0, post_norm_k0])

        # gate_up_silu: fused gate + up + SiLU.
        mlp_tile = pl.create_tensor([BATCH_TILE, INTERMEDIATE], dtype=pl.BF16)
        for ob in pl.spmd(MLP_OUT_BLOCKS, name_hint="gate_up_silu",
                          optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            mlp_o0 = ob * 256  # MLP_OUT_CHUNK = 256
            post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
            wg_0 = pl.slice(w_gate, [K_CHUNK, 256], [layer_hidden_base, mlp_o0])
            wu_0 = pl.slice(w_up, [K_CHUNK, 256], [layer_hidden_base, mlp_o0])
            gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
            up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                wg = pl.slice(w_gate, [K_CHUNK, 256], [layer_hidden_base + k0, mlp_o0])
                wu = pl.slice(w_up, [K_CHUNK, 256], [layer_hidden_base + k0, mlp_o0])
                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
            mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, mlp_o0])

        # down_proj: fused cube matmul + vec residual.
        for dob in pl.spmd(HIDDEN_BLOCKS, name_hint="down_proj",
                          optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            d0 = dob * K_CHUNK
            mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, 256], [0, 0])
            w_down_chunk_0 = pl.slice(w_down, [256, K_CHUNK], [layer_inter_base, d0])
            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
            for ob in pl.range(1, MLP_OUT_BLOCKS):
                down_o0 = ob * 256
                down_mlp = pl.slice(mlp_tile, [BATCH_TILE, 256], [0, down_o0])
                w_down_chunk = pl.slice(w_down, [256, K_CHUNK], [layer_inter_base + down_o0, d0])
                down_acc = pl.matmul_acc(down_acc, down_mlp, w_down_chunk)
            resid_chunk_fp32 = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])
            out_chunk = pl.add(down_acc, resid_chunk_fp32)
            next_hidden = pl.assemble(next_hidden, pl.cast(out_chunk, target_type=pl.BF16), [b0, d0])

    return next_hidden

@pl.jit
def decode_fwd_tq(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    quant_k_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
    quant_v_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
    quant_k_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
    quant_v_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
    rot_matrices: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HEAD_DIM], pl.BF16],
    tq_codebook: pl.Tensor[[1, N_LEVELS], pl.FP32],
    # QJL (Algorithm 2): projection matrices + K sign/norm caches.
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    hidden_states.bind_dynamic(0, USER_BATCH_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    slot_mapping.bind_dynamic(0, USER_BATCH_DYN)
    out.bind_dynamic(0, USER_BATCH_DYN)
    block_table.bind_dynamic(0, BLOCK_TABLE_FLAT_DYN)
    rope_cos.bind_dynamic(0, ROPE_SEQ_DYN)
    rope_sin.bind_dynamic(0, ROPE_SEQ_DYN)
    quant_k_cache.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_v_cache.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_k_scales.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_v_scales.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)

    user_batch = pl.tensor.dim(hidden_states, 0)
    batch_padded = BATCH
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    hidden_blocks = HIDDEN // K_CHUNK

    # Expand codebook [1, N_LEVELS] -> [CMP_CHUNK, N_LEVELS] once for all layers.
    tq_codebook_expanded = pl.create_tensor([CMP_CHUNK, N_LEVELS], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="codebook_expand"):
        tq_codebook_expanded = pl.assemble(
            tq_codebook_expanded,
            pl.col_expand_mul(
                pl.full([CMP_CHUNK, N_LEVELS], dtype=pl.FP32, value=1.0),
                tq_codebook,
            ),
            [0, 0],
        )

    # Copy hidden states into padded [batch, hidden] tensor.
    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for kb in pl.range(hidden_blocks):
                copy_k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, copy_k0])

    # Multi-layer loop calling decode_layer_tq().
    for layer_idx in pl.range(num_layers_actual):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        current_hidden = decode_layer_tq(
            current_hidden,
            input_rms_weight,
            wq,
            wk,
            wv,
            q_norm_weight,
            k_norm_weight,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos,
            rope_sin,
            quant_k_cache,
            quant_v_cache,
            quant_k_scales,
            quant_v_scales,
            rot_matrices,
            tq_codebook_expanded,
            wo,
            post_rms_weight,
            w_gate,
            w_up,
            w_down,
            next_hidden,
            layer_idx,
        )

    # Final RMSNorm + LM head projection (same as decode_fwd).
    out = rms_lm_head(current_hidden, final_norm_weight, lm_head_weight, seq_lens, out)

    return out
