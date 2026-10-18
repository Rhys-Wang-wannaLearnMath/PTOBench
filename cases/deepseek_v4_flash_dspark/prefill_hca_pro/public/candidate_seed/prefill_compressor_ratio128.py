# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pypto.language as pl
from config import BLOCK_SIZE
from config import COMPRESS_RATIO, COMPRESS_STATE_DIM, D, EPS, HCA_STATE_BLOCK_SIZE, HCA_STATE_MAX_BLOCKS, HEAD_DIM, MAX_CMP_WRITES, MAX_SEQ_LEN, NOPE_HEAD_DIM, OUT_DIM, ROPE_HALF, ROPE_HEAD_DIM, STATE_LEN, T
STATE_BLOCK_NUM_DYN = pl.dynamic('PREFILL_HCA_STATE_BLOCK_NUM_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('PREFILL_CMP_BLOCK_NUM_DYN')
K_TILE = 512
OUT_TILE = 32
HEAD_TILE = 64
HCA_C128_RMS_TILE = 8
HCA_C128_RMS_PAD_ROWS = HCA_C128_RMS_TILE

@pl.jit.inline
def prefill_compressor_ratio128(x: pl.Tensor[[T, D], pl.BF16], compress_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32], compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32], wkv: pl.Tensor[[OUT_DIM, D], pl.BF16], wgate: pl.Tensor[[OUT_DIM, D], pl.BF16], ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32], norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], cmp_kv: pl.Out[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], position_ids: pl.Tensor[[T], pl.INT32], num_tokens: pl.Scalar[pl.INT32], cmp_slot_mapping: pl.Tensor[[T], pl.INT64], state_slot_mapping: pl.Tensor[[T], pl.INT64]):
    x_flat = x
    state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    kv_proj_scratch = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(compress_state, [state_block_num * HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    pooled_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    normed_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_hca_c128_pooled_seed') as pooled_seed_tid:
        pooled_kv_pad[:, :] = pl.full([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32, value=0.0)
    for proj_idx in pl.spmd(OUT_DIM // OUT_TILE, name_hint='prefill_hca_c128_kv_score_proj'):
        o0 = proj_idx * OUT_TILE
        kv_acc = pl.create_tensor([T, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([T, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_tile = x_flat[0:T, k0:k0 + K_TILE]
            wkv_tile = wkv[o0:o0 + OUT_TILE, k0:k0 + K_TILE]
            wgate_tile = wgate[o0:o0 + OUT_TILE, k0:k0 + K_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
        kv_proj_scratch[0:T, o0:o0 + OUT_TILE] = kv_acc
        score_proj_scratch[0:T, o0:o0 + OUT_TILE] = score_acc
    write_pos_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_hca_c128_write_map'):
        write_pos_map[0:1, 0:HCA_C128_RMS_TILE] = pl.full([1, HCA_C128_RMS_TILE], dtype=pl.INT32, value=0)
        write_dst_map[0:1, 0:HCA_C128_RMS_TILE] = pl.full([1, HCA_C128_RMS_TILE], dtype=pl.INT32, value=-1)
        map_seen = pl.cast(0, pl.INDEX)
        for map_w in pl.range(T):
            if map_w < num_tokens:
                map_slot_raw = pl.read(cmp_slot_mapping, [map_w])
                if map_slot_raw >= 0:
                    pl.write(write_pos_map, [0, map_seen], pl.read(position_ids, [map_w]))
                    pl.write(write_dst_map, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                    map_seen = map_seen + 1
    for scatter_t in pl.spmd(T, name_hint='prefill_hca_c128_state_scatter_pre'):
        if scatter_t < num_tokens:
            scatter_row_raw = pl.read(state_slot_mapping, [scatter_t])
            if scatter_row_raw >= 0:
                scatter_row = pl.cast(scatter_row_raw, pl.INDEX)
                scatter_pos = pl.read(position_ids, [scatter_t])
                scatter_ape_slot = pl.cast(scatter_pos % COMPRESS_RATIO, pl.INDEX)
                compress_state_flat[scatter_row:scatter_row + 1, 0:OUT_DIM] = kv_proj_scratch[scatter_t:scatter_t + 1, 0:OUT_DIM]
                compress_state_flat[scatter_row:scatter_row + 1, OUT_DIM:COMPRESS_STATE_DIM] = pl.add(score_proj_scratch[scatter_t:scatter_t + 1, 0:OUT_DIM], ape[scatter_ape_slot:scatter_ape_slot + 1, 0:OUT_DIM])
    for pool_idx in pl.spmd(MAX_CMP_WRITES * (HEAD_DIM // HEAD_TILE), name_hint='prefill_hca_c128_softmax_pool', deps=[pooled_seed_tid]):
        write_i = pool_idx // (HEAD_DIM // HEAD_TILE)
        hb = pool_idx - write_i * (HEAD_DIM // HEAD_TILE)
        h0 = hb * HEAD_TILE
        pool_kv_tile = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
        pool_score_tile = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
        write_slot_raw = pl.read(write_dst_map, [0, write_i])
        if write_slot_raw >= 0:
            write_pos = pl.read(write_pos_map, [0, write_i])
            for pool_state_i in pl.range(STATE_LEN):
                pool_kv_tile[pool_state_i:pool_state_i + 1, 0:HEAD_TILE] = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                pool_score_tile[pool_state_i:pool_state_i + 1, 0:HEAD_TILE] = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                pool_abs = write_pos + 1 - COMPRESS_RATIO + pool_state_i
                pool_state_block = pl.cast(pool_abs // HCA_STATE_BLOCK_SIZE, pl.INDEX)
                pool_state_intra = pl.cast(pool_abs - pool_state_block * HCA_STATE_BLOCK_SIZE, pl.INDEX)
                pool_phys_block_raw = pl.read(compress_state_block_table, [pool_state_block])
                if pool_phys_block_raw >= 0:
                    pool_phys_block = pl.cast(pool_phys_block_raw, pl.INDEX)
                    pool_state_row = pool_phys_block * HCA_STATE_BLOCK_SIZE + pool_state_intra
                    pool_kv_tile[pool_state_i:pool_state_i + 1, 0:HEAD_TILE] = compress_state_flat[pool_state_row:pool_state_row + 1, h0:h0 + HEAD_TILE]
                    pool_score_tile[pool_state_i:pool_state_i + 1, 0:HEAD_TILE] = compress_state_flat[pool_state_row:pool_state_row + 1, OUT_DIM + h0:OUT_DIM + h0 + HEAD_TILE]
            pool_score_t = pl.transpose(pool_score_tile, axis1=0, axis2=1)
            pool_kv_t = pl.transpose(pool_kv_tile, axis1=0, axis2=1)
            score_max = pl.row_max(pool_score_t)
            score_exp = pl.exp(pl.row_expand_sub(pool_score_t, score_max))
            score_sum = pl.row_sum(score_exp)
            score_prob = pl.row_expand_div(score_exp, score_sum)
            pooled_chunk_t = pl.row_sum(pl.mul(pool_kv_t, score_prob))
            pooled_kv_pad[write_i:write_i + 1, h0:h0 + HEAD_TILE] = pl.reshape(pooled_chunk_t, [1, HEAD_TILE])
        else:
            pooled_kv_pad[write_i:write_i + 1, h0:h0 + HEAD_TILE] = pooled_kv_pad[write_i:write_i + 1, h0:h0 + HEAD_TILE]
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_hca_c128_rmsnorm_rope', deps=[pooled_seed_tid]):
        cos_b = pl.full([HCA_C128_RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([HCA_C128_RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        for norm_i in pl.range(HCA_C128_RMS_TILE):
            norm_slot_raw = pl.read(write_dst_map, [0, norm_i])
            if norm_slot_raw >= 0:
                norm_cmp_pos = pl.cast(pl.read(write_pos_map, [0, norm_i]) + 1 - COMPRESS_RATIO, pl.INDEX)
                cos_row = pl.cast(freqs_cos[norm_cmp_pos:norm_cmp_pos + 1, 0:ROPE_HALF], target_type=pl.FP32)
                sin_row = pl.cast(freqs_sin[norm_cmp_pos:norm_cmp_pos + 1, 0:ROPE_HALF], target_type=pl.FP32)
                cos_b[norm_i:norm_i + 1, 0:ROPE_HALF] = cos_row
                sin_b[norm_i:norm_i + 1, 0:ROPE_HALF] = sin_row
        partial_sq = pl.full([1, HCA_C128_RMS_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=2):
            rms_h0 = rms_kb * HEAD_TILE
            kv_rms_chunk = pooled_kv_pad[0:HCA_C128_RMS_TILE, rms_h0:rms_h0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, HCA_C128_RMS_TILE]))
        variance = pl.reshape(pl.add(pl.mul(partial_sq, 1.0 / HEAD_DIM), EPS), [HCA_C128_RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for norm_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_TILE, stage=2):
            norm_h0 = norm_kb * HEAD_TILE
            kv_norm_chunk = pooled_kv_pad[0:HCA_C128_RMS_TILE, norm_h0:norm_h0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, norm_h0:norm_h0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv_pad[0:HCA_C128_RMS_TILE, norm_h0:norm_h0 + HEAD_TILE] = normed_chunk
        kv_rope = pooled_kv_pad[0:HCA_C128_RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope, inv_rms), gamma_rope)
        rope_ones = pl.full([HCA_C128_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_kv_pad[0:HCA_C128_RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_rot
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_hca_c128_kv_finalize'):
        for final_i in pl.range(MAX_CMP_WRITES):
            final_cmp_row_raw = pl.read(write_dst_map, [0, final_i])
            if final_cmp_row_raw >= 0:
                final_cmp_row = pl.cast(final_cmp_row_raw, pl.INDEX)
                for final_hb in pl.range(HEAD_DIM // HEAD_TILE):
                    final_h0 = final_hb * HEAD_TILE
                    final_chunk = normed_kv_pad[final_i:final_i + 1, final_h0:final_h0 + HEAD_TILE]
                    cmp_kv_flat[final_cmp_row:final_cmp_row + 1, final_h0:final_h0 + HEAD_TILE] = pl.cast(final_chunk, target_type=pl.BF16, mode='rint')
    return (cmp_kv, compress_state)
