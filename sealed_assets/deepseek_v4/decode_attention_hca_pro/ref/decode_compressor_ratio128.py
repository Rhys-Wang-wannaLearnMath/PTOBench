# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=128 non-overlap).

Uses non-overlapping state layout with 128 slots.
Softmax+pool over all slots. No state shift needed."""

import pypto.language as pl

from config import CASE_CONFIG as M, BLOCK_SIZE, C128_COMPRESSOR_BLOCK_SIZE, DECODE_BATCH, DECODE_SEQ

# model config
B = DECODE_BATCH
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-128 non-overlap compressor)
COMPRESS_RATIO = 128
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
COFF = 1
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
# Paged contract:
# - compress_state_block_table is indexed by absolute token position blocks,
#   matching vLLM's DeepSeek V4 compressor state cache contract:
#     state_logical_block = absolute_position // COMPRESS_STATE_BLOCK_SIZE
#     state_intra        = absolute_position % COMPRESS_STATE_BLOCK_SIZE
# - APE remains ratio-local:
#     ape_row = absolute_position % COMPRESS_RATIO
# - COMPRESS_STATE_MAX_BLOCKS=64 is only enough for the current tests
#   (64 * 8 positions per request), not full MAX_SEQ_LEN coverage.
COMPRESS_STATE_MAX_BLOCKS = 64
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_MAX_BLOCKS = 64
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS
START_POS = COMPRESS_RATIO - 1
if IDX_KV_LEN > CMP_MAX_BLOCKS * BLOCK_SIZE:
    raise ValueError("ratio128 compressed KV cache capacity is smaller than max compressed sequence length")

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
HEAD_TILE = 64
B_TILE = 16
BS_PAD = ((B * S + B_TILE - 1) // B_TILE) * B_TILE
RMS_TILE = 8


@pl.jit.inline
def compressor(
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
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    kv_proj_scratch = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    x_flat = pl.reshape(x, [B * S, D])
    for idx in pl.spmd(BS_PAD * OUT_DIM // (B_TILE * OUT_TILE), name_hint="kv_score_proj"):
        global_row0 = (idx // (OUT_DIM // OUT_TILE)) * B_TILE
        o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            wkv_tile = wkv[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            wgate_tile = wgate[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

        kv_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = score_acc

    kv_proj_by_batch = kv_proj_scratch
    score_proj_by_batch = score_proj_scratch
    compress_state_flat = pl.reshape(compress_state, [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE * COMPRESS_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_pre"):
        for global_c_idx in pl.range(B):
            start_pos_b = pl.read(start_pos, [global_c_idx])
            pos_b = start_pos_b % COMPRESS_RATIO
            ape_row_b = pl.cast(pos_b, target_type=pl.INDEX)
            pre_tokens_b = COMPRESS_RATIO - pos_b
            if pos_b + S > COMPRESS_RATIO:
                scatter_n = pre_tokens_b
            else:
                scatter_n = S
            for s in pl.range(S):
                if s < scatter_n:
                    proj_col0 = s * OUT_DIM
                    token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                    state_pos = start_pos_b + s
                    state_logical_blk = state_pos // COMPRESS_STATE_BLOCK_SIZE
                    state_intra = state_pos % COMPRESS_STATE_BLOCK_SIZE
                    state_blk_id = pl.cast(pl.read(compress_state_block_table, [global_c_idx, state_logical_blk]), target_type=pl.INDEX)
                    slot_col0_s = state_intra * COMPRESS_STATE_DIM
                    ape_row = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                    kv_row = kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                    score_row = score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                    score_row = pl.add(score_row, ape_row)
                    compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s : slot_col0_s + OUT_DIM] = kv_row
                    compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s + OUT_DIM : slot_col0_s + 2 * OUT_DIM] = score_row

    pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(B * HEAD_DIM // HEAD_TILE, name_hint="softmax_pool"):
        global_c_idx = idx // (HEAD_DIM // HEAD_TILE)
        h0 = (idx % (HEAD_DIM // HEAD_TILE)) * HEAD_TILE
        start_pos_gate = pl.read(start_pos, [global_c_idx])
        pos_gate = start_pos_gate % COMPRESS_RATIO
        if pos_gate + S >= COMPRESS_RATIO:
            softmax_score_state = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
            softmax_kv_state = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
            for s in pl.pipeline(STATE_LEN, stage=2):
                start_pos_b = pl.read(start_pos, [global_c_idx])
                pos_b = start_pos_b % COMPRESS_RATIO
                compress_pos = start_pos_b + (COMPRESS_RATIO - 1 - pos_b)
                state_pos = compress_pos - (COMPRESS_RATIO - 1) + s
                state_logical_blk = state_pos // COMPRESS_STATE_BLOCK_SIZE
                state_intra = state_pos % COMPRESS_STATE_BLOCK_SIZE
                state_blk_id = pl.cast(pl.read(compress_state_block_table, [global_c_idx, state_logical_blk]), target_type=pl.INDEX)
                kv_col0 = state_intra * COMPRESS_STATE_DIM + h0
                score_col0 = state_intra * COMPRESS_STATE_DIM + OUT_DIM + h0
                slot_score = compress_state_flat[state_blk_id : state_blk_id + 1, score_col0 : score_col0 + HEAD_TILE]
                slot_kv = compress_state_flat[state_blk_id : state_blk_id + 1, kv_col0 : kv_col0 + HEAD_TILE]
                softmax_score_state[s : s + 1, :] = slot_score
                softmax_kv_state[s : s + 1, :] = slot_kv

            softmax_score_state_t = pl.transpose(softmax_score_state, axis1=0, axis2=1)
            softmax_kv_state_t = pl.transpose(softmax_kv_state, axis1=0, axis2=1)
            score_max = pl.row_max(softmax_score_state_t)
            score_exp = pl.exp(pl.row_expand_sub(softmax_score_state_t, score_max))
            score_sum = pl.row_sum(score_exp)
            score_prob = pl.row_expand_div(score_exp, score_sum)
            pooled_chunk_t = pl.row_sum(pl.mul(softmax_kv_state_t, score_prob))
            pooled_chunk = pl.reshape(pooled_chunk_t, [1, HEAD_TILE])
            pooled_kv[global_c_idx : global_c_idx + 1, h0 : h0 + HEAD_TILE] = pooled_chunk

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    normed_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="rmsnorm_rope"):
        batch_base = batch_base_idx * RMS_TILE
        cos_b = cos[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        partial_parts = pl.create_tensor([HEAD_DIM // HEAD_TILE, RMS_TILE], dtype=pl.FP32)
        for rms_kb in pl.range(HEAD_DIM // HEAD_TILE):
            kv_rms_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_TILE])
            partial_parts[rms_kb : rms_kb + 1, :] = kv_rms_rowsum
        partial_sq = pl.col_sum(partial_parts)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        gamma_nope = norm_w_2d[:, 0 : NOPE_HEAD_DIM]
        kv_norm_nope = pooled_kv[batch_base : batch_base + RMS_TILE, 0 : NOPE_HEAD_DIM]
        normed_nope = pl.col_expand_mul(pl.row_expand_mul(kv_norm_nope, inv_rms), gamma_nope)
        normed_kv[batch_base : batch_base + RMS_TILE, 0 : NOPE_HEAD_DIM] = normed_nope

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
        normed_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM] = rope_buf

    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="kv_finalize"):
        batch_base = batch_base_idx * RMS_TILE
        for inner in pl.range(RMS_TILE):
            global_c_idx = batch_base + inner
            start_pos_b = pl.read(start_pos, [global_c_idx])
            pos_b = start_pos_b % COMPRESS_RATIO
            ape_row_b = pl.cast(pos_b, target_type=pl.INDEX)
            pre_tokens_b = COMPRESS_RATIO - pos_b
            cache_col = start_pos_b // COMPRESS_RATIO
            if pos_b + S >= COMPRESS_RATIO:
                kv_row = normed_kv[global_c_idx : global_c_idx + 1, 0 : HEAD_DIM]
                kv_flat[global_c_idx * S : global_c_idx * S + 1, :] = kv_row
                cmp_logical_blk = cache_col // BLOCK_SIZE
                cmp_intra = cache_col % BLOCK_SIZE
                cmp_blk_id = pl.cast(pl.read(cmp_block_table, [global_c_idx, cmp_logical_blk]), target_type=pl.INDEX)
                phys_cmp_row = cmp_blk_id * BLOCK_SIZE + cmp_intra
                cmp_kv_cache_flat[phys_cmp_row : phys_cmp_row + 1, :] = pl.cast(kv_row, target_type=pl.BF16, mode="rint")

                for s in pl.range(S):
                    if s >= pre_tokens_b:
                        proj_col0 = s * OUT_DIM
                        token_ape_row = (ape_row_b + s) % COMPRESS_RATIO
                        state_pos = start_pos_b + s
                        state_logical_blk = state_pos // COMPRESS_STATE_BLOCK_SIZE
                        state_intra = state_pos % COMPRESS_STATE_BLOCK_SIZE
                        state_blk_id = pl.cast(pl.read(compress_state_block_table, [global_c_idx, state_logical_blk]), target_type=pl.INDEX)
                        slot_col0_s = state_intra * COMPRESS_STATE_DIM
                        ape_row = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                        kv_row = kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                        score_row = score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                        score_row = pl.add(score_row, ape_row)
                        compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s : slot_col0_s + OUT_DIM] = kv_row
                        compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s + OUT_DIM : slot_col0_s + 2 * OUT_DIM] = score_row

    compress_state = pl.reshape(compress_state_flat, [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    cmp_kv_cache = pl.reshape(cmp_kv_cache_flat, [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    return kv, compress_state, cmp_kv_cache


