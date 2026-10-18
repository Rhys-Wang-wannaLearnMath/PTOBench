# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer (decode). Mirrors model.py Indexer (line 380-433);
golden is a port of forward's decode branch (prefill `start_pos == 0` path is omitted).
The inner Compressor is invoked via golden_compressor (placeholder)."""


import pypto.language as pl

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS
from config import COMPRESS_RATIO, INNER_STATE_MAX_BLOCKS, IDX_CACHE_MAX_BLOCKS
from decode_indexer_compressor import indexer_compressor

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
OFFSET = M.sliding_window

# kernel-local
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = B * INNER_STATE_MAX_BLOCKS
INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_BLOCK_NUM = B * IDX_CACHE_MAX_BLOCKS
SCORE_LEN = IDX_KV_LEN

# tiling
CACHE_TILE = 32
assert BLOCK_SIZE % CACHE_TILE == 0, "CACHE_TILE must not cross a paged idx_kv_cache block"
Q_TILE = 128
Q_OUT_TILE = 256
QR_PROJ_ROW_TILE = 16
HEAD_DIM_TILE = 32
QH_HEAD_DIM_TILE = 128
QH_ROW_TILE = 128
D_TILE = 32
WEIGHTS_ROW_TILE = 32
MM_ROW_TILE = 16          # cube matmul rows must be 16-aligned on current ptoas
T_PAD = ((T + MM_ROW_TILE - 1) // MM_ROW_TILE) * MM_ROW_TILE
B_TILE = 4
TOPK_TILE = 8             # decode T=8; avoid zero-core T//16 topk dispatch
SCORE_INIT_TILE = 4096
SORT_TILE = 2048
assert SCORE_LEN % SCORE_INIT_TILE == 0 and SCORE_LEN % SORT_TILE == 0
assert IDX_TOPK <= SORT_TILE

@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32],
    topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32], 
):
    qr_proj_pad = pl.create_tensor([T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="qr_proj", optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        o0 = idx * Q_OUT_TILE
        qr_acc = pl.create_tensor([MM_ROW_TILE, Q_OUT_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
            q0 = kb * Q_TILE
            qr_tile = pl.slice(qr, [MM_ROW_TILE, Q_TILE], [0, q0], valid_shape=[T, Q_TILE])
            wq_tile = wq_b[q0 : q0 + Q_TILE, o0 : o0 + Q_OUT_TILE]
            if q0 == 0:
                qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
            else:
                qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
        wq_scale = pl.reshape(wq_b_scale[o0 : o0 + Q_OUT_TILE], [1, Q_OUT_TILE])
        qr_scale_pad = pl.slice(qr_scale, [MM_ROW_TILE, 1], [0, 0], valid_shape=[T, 1])
        acc_fp32 = pl.cast(qr_acc, target_type=pl.FP32, mode="none")
        qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale_pad), wq_scale)
        qr_proj_pad[0 : MM_ROW_TILE, o0 : o0 + Q_OUT_TILE] = qr_dequant

    qr_proj_flat = pl.reshape(qr_proj_pad, [T_PAD * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_rope_out = pl.create_tensor([T * IDX_N_HEADS, ROPE_HEAD_DIM], dtype=pl.BF16)
    for idx in pl.spmd(T * IDX_N_HEADS // 32, name_hint="qr_rope"):
        o0 = idx * 32
        token_idx = o0 // IDX_N_HEADS
        batch_idx = token_idx // S
        cos_b = cos[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM // 2]
        qr_rope_slice = qr_proj_flat[o0 : o0 + 32, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
        even_tile = pl.gather(qr_rope_slice, mask_pattern=pl.tile.MaskPattern.P0101)
        odd_tile = pl.gather(qr_rope_slice, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_even = pl.sub(pl.col_expand_mul(even_tile, cos_b), pl.col_expand_mul(odd_tile, sin_b))
        rope_odd = pl.add(pl.col_expand_mul(even_tile, sin_b), pl.col_expand_mul(odd_tile, cos_b))
        rope_buf = pl.full([32, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_buf = pl.tensor.scatter(rope_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_buf)
        rope_buf = pl.tensor.scatter(rope_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_buf)
        qr_rope_out[o0 : o0 + 32, :] = pl.cast(rope_buf, target_type=pl.BF16, mode="rint")

    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // 512, name_hint="qr_hadamard_quant", optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=1)]):
        o0 = idx * 512
        for ro in pl.range(0, 512, QH_ROW_TILE):
            qh_nope = pl.cast(qr_proj_flat[o0 + ro : o0 + ro + QH_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint")
            qh_rope = qr_rope_out[o0 + ro : o0 + ro + QH_ROW_TILE, :]
            qh_acc = pl.matmul(qh_nope, hadamard[0 : IDX_NOPE_HEAD_DIM, :], out_dtype=pl.FP32)
            qr_hadamard_acc = pl.matmul_acc(qh_acc, qh_rope, hadamard[IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM, :])
            qh_a_abs = pl.maximum(qr_hadamard_acc, pl.neg(qr_hadamard_acc))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_ROW_TILE])
            qh_amax = pl.maximum(qh_a_max, INT8_AMAX_EPS)
            qh_scale_quant_row = pl.div(pl.add(pl.sub(qh_amax, qh_amax), INT8_SCALE_MAX), qh_amax)
            qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_ROW_TILE, 1])
            qr_hadamard_scale_dq[o0 + ro : o0 + ro + QH_ROW_TILE, :] = qh_scale_dq
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_ROW_TILE, 1])
            for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
                qh_q_f32 = qr_hadamard_acc[0 : QH_ROW_TILE, h1 : h1 + QH_HEAD_DIM_TILE]
                qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
                qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
                qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
                qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
                qr_hadamard_i8[o0 + ro : o0 + ro + QH_ROW_TILE, h1 : h1 + QH_HEAD_DIM_TILE] = qh_i8

    x_flat = pl.reshape(x, [T, D])
    weights_pad = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    # Decode has only T=8 rows, but current ptoas rejects cube matmul M tiles
    # smaller than 16. Compute a padded 16-row tile and only consume the real T
    # rows below. This is a compatibility shim; math on valid rows is unchanged.
    for idx in pl.spmd(T_PAD // MM_ROW_TILE, name_hint="weights_proj"):
        wrow0 = idx * MM_ROW_TILE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.pipeline(0, D // D_TILE, stage=2):
            d0 = db * D_TILE
            valid_rows = pl.min(MM_ROW_TILE, T - wrow0)
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [wrow0, d0], valid_shape=[valid_rows, D_TILE])
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            if d0 == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
        weights_pad[wrow0 : wrow0 + MM_ROW_TILE, :] = pl.mul(weights_acc, WEIGHTS_SCALE)

    inner_kv, inner_compress_state, idx_kv_cache = indexer_compressor(
        x,
        inner_kv,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        cos,
        sin,
        hadamard,
        idx_kv_cache,
        idx_block_table,
        start_pos,
    )

    kv_cache_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM])
    idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])
    score_flat = pl.reshape(score, [T, SCORE_LEN])
    # Preserve the complete score output while bounding each UB tile.
    for init_block in pl.spmd(T * (SCORE_LEN // SCORE_INIT_TILE), name_hint="score_init"):
        init_row = init_block // (SCORE_LEN // SCORE_INIT_TILE)
        init_col = (init_block % (SCORE_LEN // SCORE_INIT_TILE)) * SCORE_INIT_TILE
        score_flat[init_row : init_row + 1, init_col : init_col + SCORE_INIT_TILE] = pl.full([1, SCORE_INIT_TILE], dtype=pl.FP32, value=FP32_NEG_INF)

    qr_hadamard_scale_flat = pl.reshape(qr_hadamard_scale_dq, [1, T * IDX_N_HEADS])
    for bg in pl.spmd(B // B_TILE, name_hint="score", optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=1)]):
        for bi in pl.range(B_TILE):
            b = bg * B_TILE + bi
            sp_b = pl.read(start_pos, [b])
            clen_b = (sp_b + S) // COMPRESS_RATIO
            cblk_b = (clen_b + CACHE_TILE - 1) // CACHE_TILE
            tb = b * S
            qb = b * S * IDX_N_HEADS
            for cb in pl.range(cblk_b):
                cache0 = cb * CACHE_TILE
                valid_len = pl.min(CACHE_TILE, clen_b - cache0)
                idx_blk_off = cache0 // BLOCK_SIZE
                idx_intra = cache0 % BLOCK_SIZE
                idx_blk_id = pl.cast(
                    pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + idx_blk_off]),
                    pl.INDEX,
                )
                kv0 = idx_blk_id * BLOCK_SIZE + idx_intra
                kv_q_full_f32 = pl.cast(kv_cache_flat[kv0 : kv0 + CACHE_TILE, 0 : IDX_HEAD_DIM], target_type=pl.FP32)
                kv_a_abs = pl.maximum(kv_q_full_f32, pl.neg(kv_q_full_f32))
                kv_a_max = pl.reshape(pl.row_max(kv_a_abs), [1, CACHE_TILE])
                kv_amax = pl.maximum(kv_a_max, INT8_AMAX_EPS)
                kv_scale_quant_row = pl.div(pl.add(pl.sub(kv_amax, kv_amax), INT8_SCALE_MAX), kv_amax)
                kv_cache_scale_dq = pl.reshape(pl.recip(kv_scale_quant_row), [CACHE_TILE, 1])
                kv_scale_quant = pl.reshape(kv_scale_quant_row, [CACHE_TILE, 1])
                kv_q_full_scaled = pl.row_expand_mul(kv_q_full_f32, kv_scale_quant)
                kv_q_full_i32 = pl.cast(kv_q_full_scaled, target_type=pl.INT32, mode="rint")
                kv_q_full_half = pl.cast(kv_q_full_i32, target_type=pl.FP16, mode="round")
                kv_q_i8_full = pl.cast(kv_q_full_half, target_type=pl.INT8, mode="trunc")
                for s in pl.range(S):
                    qr_full = qr_hadamard_i8[qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS, 0 : IDX_HEAD_DIM]
                    score_acc_s = pl.matmul(kv_q_i8_full, qr_full, out_dtype=pl.INT32, b_trans=True)
                    qh_scale_s = qr_hadamard_scale_flat[0 : 1, qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS]
                    score_tile_s = pl.cast(score_acc_s, target_type=pl.FP32, mode="none")
                    score_tile_s = pl.col_expand_mul(pl.row_expand_mul(score_tile_s, kv_cache_scale_dq), qh_scale_s)
                    relu_score_s = pl.maximum(score_tile_s, pl.mul(score_tile_s, 0.0))
                    weights_row_s = pl.reshape(weights_pad[tb + s : tb + s + 1, :], [1, IDX_N_HEADS])
                    weighted_score_s_t = pl.col_expand_mul(relu_score_s, weights_row_s)
                    weighted_score_s = pl.reshape(pl.row_sum(weighted_score_s_t), [1, CACHE_TILE])
                    lane_offset = pl.get_subblock_idx() * (CACHE_TILE // 2)
                    lane_valid = pl.min(CACHE_TILE // 2, pl.max(valid_len, lane_offset) - lane_offset)
                    weighted_score_valid_s = pl.fillpad(pl.set_validshape(weighted_score_s, 1, lane_valid), pad_value=pl.PadValue.min)
                    weighted_score_valid_s = pl.maximum(
                        weighted_score_valid_s,
                        FP32_NEG_INF,
                    )
                    score_flat[tb + s : tb + s + 1, cache0 : cache0 + CACHE_TILE] = weighted_score_valid_s

    topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])
    running_pairs = pl.create_tensor([T, 2 * IDX_TOPK], dtype=pl.FP32)
    # Adapt the earliest official 1M-context selector (1d7c7017, 2026-08-24):
    # sort 2048-value leaves and merge exact top-k prefixes. Keep the old
    # complete score/index ABI and visit every SCORE_LEN column.
    for t in pl.spmd(T, name_hint="topk"):
        for init_col in pl.range(0, SCORE_LEN, SCORE_INIT_TILE):
            topk_idxs_flat[t : t + 1, init_col : init_col + SCORE_INIT_TILE] = pl.full([1, SCORE_INIT_TILE], dtype=pl.INT32, value=-1)
        batch_idx = t // S
        start_pos_b = pl.read(start_pos, [batch_idx])
        cache_len_b = (start_pos_b + S) // COMPRESS_RATIO
        if cache_len_b > 0:
            for leaf_col in pl.range(0, SCORE_LEN, SORT_TILE):
                score_leaf = score_flat[t : t + 1, leaf_col : leaf_col + SORT_TILE]
                idx_init = pl.arange(pl.cast(leaf_col, pl.UINT32), [1, SORT_TILE], dtype=pl.UINT32)
                sorted_leaf = pl.sort32(score_leaf, idx_init)
                sorted_leaf = pl.mrgsort(sorted_leaf, block_len=64)
                sorted_leaf = pl.mrgsort(sorted_leaf, block_len=256)
                sorted_leaf = pl.mrgsort(sorted_leaf, block_len=1024)
                leaf_pairs = sorted_leaf[:, 0 : 2 * IDX_TOPK]
                if leaf_col == 0:
                    running_pairs[t : t + 1, :] = leaf_pairs
                else:
                    prior_pairs = running_pairs[t : t + 1, :]
                    merged_pairs = pl.mrgsort(prior_pairs, leaf_pairs)
                    running_pairs[t : t + 1, :] = merged_pairs[:, 0 : 2 * IDX_TOPK]
            topk_pairs = running_pairs[t : t + 1, :]
            topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid_topk = pl.min(IDX_TOPK, cache_len_b)
            topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
            offset_i32 = pl.cast(offset, target_type=pl.INT32)
            topk_idxs_flat[t : t + 1, 0 : IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)

    return score, idx_kv_cache, topk_idxs


@pl.jit
def indexer_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.INT32]],
    start_pos: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
):
    score, idx_kv_cache, topk_idxs = indexer(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        weights_proj,
        cos,
        sin,
        hadamard,
        inner_kv,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_block_table,
        score,
        topk_idxs,
        start_pos,
        offset,
    )
    return score, idx_kv_cache, topk_idxs
