# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer KV Compressor (decode incremental, ratio=4 overlap)."""


import pypto.language as pl

from config import (
    CASE_CONFIG as M,
    B,
    S,
    D,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    ROPE_HEAD_DIM,
    NOPE_HEAD_DIM,
    MAX_SEQ_LEN,
    COMPRESS_RATIO,
    OVERLAP,
    COFF,
    OUT_DIM,
    STATE_LEN,
    IDX_KV_LEN,
    COMPRESS_STATE_BLOCK_SIZE,
    COMPRESS_STATE_MAX_BLOCKS,
    COMPRESS_STATE_BLOCK_NUM,
    COMPRESS_STATE_DIM,
    BLOCK_SIZE,
    IDX_CACHE_MAX_BLOCKS,
    IDX_CACHE_BLOCK_NUM,
    START_POS,
    FP32_NEG_INF,
)

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
B_TILE = 8
MM_B_TILE = 16  # cube matmul rows must be a 16-row boxed tile
BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
HEAD_TILE = 64
HEAD_DIM_TILE = 128
RMS_TILE = 8
RMS_M_TILE = 16


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj_scratch = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(compress_state, [COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    # Output tensors are not guaranteed to preserve TensorSpec init values on the
    # current runner.  Recreate the golden initial state for the untouched score
    # half; state_scatter below overwrites active rows.
    with pl.spmd((COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE) * (OUT_DIM // OUT_TILE), name_hint="state_score_init") as init_tid:
        init_idx = pl.tile.get_block_idx()
        init_row = init_idx // (OUT_DIM // OUT_TILE)
        init_col = OUT_DIM + (init_idx - init_row * (OUT_DIM // OUT_TILE)) * OUT_TILE
        compress_state_flat[init_row : init_row + 1, init_col : init_col + OUT_TILE] = \
            pl.full([1, OUT_TILE], dtype=pl.FP32, value=FP32_NEG_INF)

    for idx in pl.spmd(BS_PAD * OUT_DIM // (MM_B_TILE * OUT_TILE), name_hint="kv_score_proj"):
        global_row0 = (idx // (OUT_DIM // OUT_TILE)) * MM_B_TILE
        o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            wkv_tile = wkv[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            wgate_tile = wgate[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

        kv_proj_scratch[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_scratch[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = score_acc

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_paged", deps=[init_tid]):
        for c_idx in pl.range(B):
            start_pos_b = pl.read(start_pos, [c_idx])
            ape_row_b = pl.cast(start_pos_b % COMPRESS_RATIO, target_type=pl.INDEX)

            for s in pl.pipeline(S, stage=2):
                abs_pos_s = start_pos_b + s
                state_blk_off = abs_pos_s // COMPRESS_STATE_BLOCK_SIZE
                state_intra = abs_pos_s % COMPRESS_STATE_BLOCK_SIZE
                state_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, state_blk_off]), pl.INDEX)
                state_row = state_blk_id * COMPRESS_STATE_BLOCK_SIZE + state_intra
                proj_row = c_idx * S + s
                token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                kv_tile = kv_proj_scratch[proj_row : proj_row + 1, 0 : OUT_DIM]
                score_tile = score_proj_scratch[proj_row : proj_row + 1, 0 : OUT_DIM]
                ape_tile = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                score_tile = pl.add(score_tile, ape_tile)
                compress_state_flat[state_row : state_row + 1, 0 : OUT_DIM] = kv_tile
                compress_state_flat[state_row : state_row + 1, OUT_DIM : 2 * OUT_DIM] = score_tile

    pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    for c_idx in pl.spmd(B, name_hint="softmax_pool"):
        start_pos_b = pl.read(start_pos, [c_idx])
        pos_b = start_pos_b % COMPRESS_RATIO
        pre_tokens_b = COMPRESS_RATIO - pos_b
        boundary_end_b = start_pos_b + pre_tokens_b - 1
        cur_window_start_b = boundary_end_b - COMPRESS_RATIO + 1
        prev_window_start_b = cur_window_start_b - COMPRESS_RATIO

        if pos_b + S >= COMPRESS_RATIO:
            for hb in pl.range(HEAD_DIM // HEAD_TILE):
                h0 = hb * HEAD_TILE
                last_abs = cur_window_start_b + COMPRESS_RATIO - 1
                last_blk_off = last_abs // COMPRESS_STATE_BLOCK_SIZE
                last_intra = last_abs % COMPRESS_STATE_BLOCK_SIZE
                last_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, last_blk_off]), pl.INDEX)
                last_row = last_blk_id * COMPRESS_STATE_BLOCK_SIZE + last_intra
                last_col0 = OUT_DIM + HEAD_DIM + h0
                mi = compress_state_flat[last_row : last_row + 1, last_col0 : last_col0 + HEAD_TILE]
                li = pl.exp(pl.sub(mi, mi))
                oi = compress_state_flat[last_row : last_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]

                for s in pl.range(0, COMPRESS_RATIO):
                    prev_abs = prev_window_start_b + s
                    if prev_abs >= 0:
                        prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                        prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                        prev_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, prev_blk_off]), pl.INDEX)
                        prev_row = prev_blk_id * COMPRESS_STATE_BLOCK_SIZE + prev_intra
                        front_score = compress_state_flat[prev_row : prev_row + 1, OUT_DIM + h0 : OUT_DIM + h0 + HEAD_TILE]
                        front_kv = compress_state_flat[prev_row : prev_row + 1, h0 : h0 + HEAD_TILE]
                        mi_next_front = pl.maximum(mi, front_score)
                        alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                        beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                        li = pl.add(pl.mul(alpha_front, li), beta_front)
                        oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                        mi = mi_next_front

                for s in pl.range(0, COMPRESS_RATIO - 1):
                    cur_abs = cur_window_start_b + s
                    cur_blk_off = cur_abs // COMPRESS_STATE_BLOCK_SIZE
                    cur_intra = cur_abs % COMPRESS_STATE_BLOCK_SIZE
                    cur_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, cur_blk_off]), pl.INDEX)
                    cur_row = cur_blk_id * COMPRESS_STATE_BLOCK_SIZE + cur_intra
                    back_col0 = OUT_DIM + HEAD_DIM + h0
                    back_score = compress_state_flat[cur_row : cur_row + 1, back_col0 : back_col0 + HEAD_TILE]
                    back_kv = compress_state_flat[cur_row : cur_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_kv[c_idx : c_idx + 1, h0 : h0 + HEAD_TILE] = pooled_chunk

    normed_kv = pl.create_tensor([RMS_M_TILE, HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="rmsnorm_rope"):
        batch_base = batch_base_idx * RMS_TILE
        cos_b = cos[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        partial_sq = pl.full([1, RMS_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE]
            gamma = norm_w_2d[:, k0 : k0 + HEAD_TILE]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE] = pl.cast(normed_chunk, target_type=pl.BF16, mode="rint")

        kv_rope_norm = pooled_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        even_tile = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P0101)
        odd_tile = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_even = pl.sub(pl.mul(even_tile, cos_b), pl.mul(odd_tile, sin_b))
        rope_odd = pl.add(pl.mul(even_tile, sin_b), pl.mul(odd_tile, cos_b))
        rope_buf = pl.full([RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_buf = pl.tensor.scatter(rope_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_buf)
        rope_buf = pl.tensor.scatter(rope_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_buf)
        normed_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM] = pl.cast(rope_buf, target_type=pl.BF16, mode="rint")

    kv_final = pl.create_tensor([RMS_M_TILE, HEAD_DIM], dtype=pl.FP32)
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="kv_hadamard"):
        batch_base = batch_base_idx * RMS_TILE
        kv_proj_tile = pl.slice(normed_kv, [RMS_M_TILE, HEAD_DIM], [batch_base, 0], valid_shape=[RMS_TILE, HEAD_DIM])
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0 : HEAD_DIM, o0 : o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[batch_base : batch_base + RMS_M_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc

    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="kv_and_cache_write"):
        batch_base = batch_base_idx * RMS_TILE
        for inner in pl.range(RMS_TILE):
            c_idx = batch_base + inner
            start_pos_b = pl.read(start_pos, [c_idx])
            pos_b = start_pos_b % COMPRESS_RATIO
            cache_col = start_pos_b // COMPRESS_RATIO
            if pos_b + S >= COMPRESS_RATIO:
                kv_row_fp32 = kv_final[c_idx : c_idx + 1, 0 : HEAD_DIM]
                kv_flat[c_idx * S : c_idx * S + 1, :] = kv_row_fp32
                idx_blk_off = cache_col // BLOCK_SIZE
                idx_intra = cache_col % BLOCK_SIZE
                idx_blk_id = pl.cast(pl.read(idx_block_table, [c_idx, idx_blk_off]), pl.INDEX)
                cache_row = idx_blk_id * BLOCK_SIZE + idx_intra
                idx_kv_cache_flat[cache_row : cache_row + 1, :] = pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint")

    compress_state = pl.reshape(compress_state_flat, [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    idx_kv_cache = pl.reshape(idx_kv_cache_flat, [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv, compress_state, idx_kv_cache


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    kv, compress_state, idx_kv_cache = indexer_compressor(
        x,
        kv,
        compress_state,
        compress_state_block_table,
        wkv,
        wgate,
        ape,
        norm_w,
        cos,
        sin,
        hadamard,
        idx_kv_cache,
        idx_block_table,
        start_pos,
    )
    return kv, compress_state, idx_kv_cache
