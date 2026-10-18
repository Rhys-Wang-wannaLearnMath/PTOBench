# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill indexer compressor for ratio-4 overlapping KV cache."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    COMPRESS_RATIO,
    D,
    EPS,
    FP32_NEG_INF,
    HEAD_DIM,
    HEAD_DIM_INV,
    IDX_CACHE_MAX_BLOCKS,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    MAX_CMP_WRITES,
    MAX_SEQ_LEN,
    NOPE_HEAD_DIM,
    OUT_DIM,
    PREFILL_IDX_BLOCK_NUM,
    ROPE_HEAD_DIM,
    STATE_LEN,
    T,
)

# Editable implementation tiling retained from the selected source.
HEAD_CHUNK = 32
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK
K_TILE = 512
OUT_TILE = 64
HEAD_TILE = 64
PACKED_PROJ_BLOCKS = OUT_DIM // OUT_TILE
PACKED_POOL_BLOCKS = MAX_CMP_WRITES * HEAD_BLOCKS
PACKED_STATE_UPDATE_TILE = 16
PACKED_RMS_TILE = 16


@pl.jit.inline
def prefill_indexer_compressor(
    x: pl.Tensor[[T, D], pl.BF16],
    kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    # C8 indexer cache: INT8 KV (quant-on-write) + per-position FP32 dequant scale; no bf16 cache.
    idx_kv_cache: pl.Out[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.Out[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    kv_proj_scratch = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    kv_state_flat = pl.reshape(kv_state, [INNER_STATE_BLOCK_NUM * INNER_STATE_BLOCK_SIZE, OUT_DIM])
    score_state_flat = pl.reshape(score_state, [INNER_STATE_BLOCK_NUM * INNER_STATE_BLOCK_SIZE, OUT_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, 1])
    pooled_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.BF16)
    final_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)

    for proj_idx in pl.spmd(PACKED_PROJ_BLOCKS, name_hint="prefill_idx_c4_kv_score_proj"):
        o0 = proj_idx * OUT_TILE
        kv_acc = pl.create_tensor([T, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([T, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_tile = x[0:T, k0 : k0 + K_TILE]
            # Weights stored transposed [OUT_DIM, D] + b_trans=True -> DN2ZN load
            # (K-contiguous long bursts) instead of ND2NZ strided; mirrors the main
            # compressor (prefill_compressor_ratio4) and the decode indexer compressor.
            wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
        kv_proj_scratch[0:T, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_scratch[0:T, o0 : o0 + OUT_TILE] = score_acc

    # Precompute write_i -> (position, dst cache row) once. Input-only deps, so it overlaps the
    # projection matmul, replacing the O(T) write-discovery scan repeated in pool / rmsnorm_rope /
    # cache_write.
    write_pos_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_c4_write_map"):
        write_pos_map[0:1, 0:MAX_CMP_WRITES] = pl.full([1, MAX_CMP_WRITES], dtype=pl.INT32, value=0)
        write_dst_map[0:1, 0:MAX_CMP_WRITES] = pl.full([1, MAX_CMP_WRITES], dtype=pl.INT32, value=-1)
        map_seen = pl.cast(0, pl.INDEX)
        for map_w in pl.range(T):
            if map_w < num_tokens:
                map_slot_raw = pl.read(idx_slot_mapping, [map_w])
                if map_slot_raw >= 0:
                    pl.write(write_pos_map, [0, map_seen], pl.read(position_ids, [map_w]))
                    pl.write(write_dst_map, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                    map_seen = map_seen + 1

    for pool_idx in pl.spmd(PACKED_POOL_BLOCKS, name_hint="prefill_idx_c4_softmax_pool"):
        write_i = pool_idx // HEAD_BLOCKS
        hb = pool_idx - write_i * HEAD_BLOCKS
        h0 = hb * HEAD_CHUNK
        pool_kv_tile = pl.create_tensor([STATE_LEN, HEAD_CHUNK], dtype=pl.FP32)
        pool_score_tile = pl.create_tensor([STATE_LEN, HEAD_CHUNK], dtype=pl.FP32)
        write_slot_raw = pl.read(write_dst_map, [0, write_i])
        if write_slot_raw >= 0:
            write_pos = pl.read(write_pos_map, [0, write_i])
            cur_start = write_pos + 1 - COMPRESS_RATIO
            prev_start = cur_start - COMPRESS_RATIO
            for pool_s in pl.range(COMPRESS_RATIO):
                prev_abs = prev_start + pool_s
                front_slot = pool_s
                pool_kv_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_score_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
                if write_pos >= 2 * COMPRESS_RATIO - 1:
                    prev_state_block = pl.cast(prev_abs // INNER_STATE_BLOCK_SIZE, pl.INDEX)
                    prev_state_intra = pl.cast(prev_abs - prev_state_block * INNER_STATE_BLOCK_SIZE, pl.INDEX)
                    prev_phys_block_raw = pl.read(inner_compress_state_block_table, [prev_state_block])
                    if prev_phys_block_raw >= 0:
                        prev_phys_block = pl.cast(prev_phys_block_raw, pl.INDEX)
                        prev_state_row = prev_phys_block * INNER_STATE_BLOCK_SIZE + prev_state_intra
                        pool_kv_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = kv_state_flat[
                            prev_state_row : prev_state_row + 1,
                            h0 : h0 + HEAD_CHUNK,
                        ]
                        pool_score_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = score_state_flat[
                            prev_state_row : prev_state_row + 1,
                            h0 : h0 + HEAD_CHUNK,
                        ]

                cur_abs = cur_start + pool_s
                back_slot = COMPRESS_RATIO + pool_s
                pool_kv_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_score_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
                cur_state_block = pl.cast(cur_abs // INNER_STATE_BLOCK_SIZE, pl.INDEX)
                cur_state_intra = pl.cast(cur_abs - cur_state_block * INNER_STATE_BLOCK_SIZE, pl.INDEX)
                cur_phys_block_raw = pl.read(inner_compress_state_block_table, [cur_state_block])
                if cur_phys_block_raw >= 0:
                    cur_phys_block = pl.cast(cur_phys_block_raw, pl.INDEX)
                    cur_state_row = cur_phys_block * INNER_STATE_BLOCK_SIZE + cur_state_intra
                    pool_kv_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = kv_state_flat[
                        cur_state_row : cur_state_row + 1,
                        HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                    ]
                    pool_score_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = score_state_flat[
                        cur_state_row : cur_state_row + 1,
                        HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                    ]

            for pool_t in pl.range(T):
                if pool_t < num_tokens:
                    pool_pos = pl.read(position_ids, [pool_t])
                    if pool_pos <= write_pos:
                        if pool_pos >= prev_start:
                            pool_ape_slot = pl.cast(pool_pos % COMPRESS_RATIO, pl.INDEX)
                            if pool_pos < cur_start:
                                pool_slot = pl.cast(pool_pos - prev_start, pl.INDEX)
                                pool_ape = ape[pool_ape_slot : pool_ape_slot + 1, h0 : h0 + HEAD_CHUNK]
                                pool_score = pl.add(
                                    score_proj_scratch[pool_t : pool_t + 1, h0 : h0 + HEAD_CHUNK],
                                    pool_ape,
                                )
                                pool_kv_tile[pool_slot : pool_slot + 1, 0:HEAD_CHUNK] = kv_proj_scratch[
                                    pool_t : pool_t + 1,
                                    h0 : h0 + HEAD_CHUNK,
                                ]
                                pool_score_tile[pool_slot : pool_slot + 1, 0:HEAD_CHUNK] = pool_score
                            else:
                                pool_slot = pl.cast(COMPRESS_RATIO + pool_pos - cur_start, pl.INDEX)
                                pool_ape = ape[
                                    pool_ape_slot : pool_ape_slot + 1,
                                    HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                                ]
                                pool_score = pl.add(
                                    score_proj_scratch[
                                        pool_t : pool_t + 1,
                                        HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                                    ],
                                    pool_ape,
                                )
                                pool_kv_tile[pool_slot : pool_slot + 1, 0:HEAD_CHUNK] = kv_proj_scratch[
                                    pool_t : pool_t + 1,
                                    HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                                ]
                                pool_score_tile[pool_slot : pool_slot + 1, 0:HEAD_CHUNK] = pool_score

            init_slot = STATE_LEN - 1
            mi_buf = pl.create_tensor([1, HEAD_CHUNK], dtype=pl.FP32)
            li_buf = pl.create_tensor([1, HEAD_CHUNK], dtype=pl.FP32)
            oi_buf = pl.create_tensor([1, HEAD_CHUNK], dtype=pl.FP32)
            mi_buf[0:1, 0:HEAD_CHUNK] = pool_score_tile[init_slot : init_slot + 1, 0:HEAD_CHUNK]
            li_buf[0:1, 0:HEAD_CHUNK] = pl.exp(pl.sub(mi_buf[0:1, 0:HEAD_CHUNK], mi_buf[0:1, 0:HEAD_CHUNK]))
            oi_buf[0:1, 0:HEAD_CHUNK] = pool_kv_tile[init_slot : init_slot + 1, 0:HEAD_CHUNK]
            for pool_slot_i in pl.range(STATE_LEN - 1):
                if pool_slot_i >= COMPRESS_RATIO or write_pos >= 2 * COMPRESS_RATIO - 1:
                    mi = mi_buf[0:1, 0:HEAD_CHUNK]
                    li = li_buf[0:1, 0:HEAD_CHUNK]
                    oi = oi_buf[0:1, 0:HEAD_CHUNK]
                    slot_score = pool_score_tile[pool_slot_i : pool_slot_i + 1, 0:HEAD_CHUNK]
                    slot_kv = pool_kv_tile[pool_slot_i : pool_slot_i + 1, 0:HEAD_CHUNK]
                    mi_next = pl.maximum(mi, slot_score)
                    alpha = pl.exp(pl.sub(mi, mi_next))
                    beta = pl.exp(pl.sub(slot_score, mi_next))
                    li_next = pl.add(pl.mul(alpha, li), beta)
                    oi_next = pl.add(pl.mul(oi, alpha), pl.mul(slot_kv, beta))
                    mi_buf[0:1, 0:HEAD_CHUNK] = mi_next
                    li_buf[0:1, 0:HEAD_CHUNK] = li_next
                    oi_buf[0:1, 0:HEAD_CHUNK] = oi_next
            pooled_kv[write_i : write_i + 1, h0 : h0 + HEAD_CHUNK] = pl.div(
                oi_buf[0:1, 0:HEAD_CHUNK],
                li_buf[0:1, 0:HEAD_CHUNK],
            )
        else:
            pooled_kv[write_i : write_i + 1, h0 : h0 + HEAD_CHUNK] = pl.full([1, HEAD_CHUNK], dtype=pl.FP32, value=0.0)

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    for final_block in pl.spmd(MAX_CMP_WRITES // PACKED_RMS_TILE, name_hint="prefill_idx_c4_rmsnorm_rope"):
        final_base = final_block * PACKED_RMS_TILE
        cos_b = pl.full([PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        for final_dt in pl.range(PACKED_RMS_TILE):
            final_i = final_base + final_dt
            write_slot_raw = pl.read(write_dst_map, [0, final_i])
            if write_slot_raw >= 0:
                write_pos = pl.read(write_pos_map, [0, final_i])
                cmp_pos = pl.cast(write_pos + 1 - COMPRESS_RATIO, pl.INDEX)
                cos_b[final_dt : final_dt + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(
                    freqs_cos[cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2],
                    target_type=pl.FP32,
                )
                sin_b[final_dt : final_dt + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(
                    freqs_sin[cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2],
                    target_type=pl.FP32,
                )
        partial_sq = pl.full([1, PACKED_RMS_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[final_base : final_base + PACKED_RMS_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, PACKED_RMS_TILE]))
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [PACKED_RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[final_base : final_base + PACKED_RMS_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[final_base : final_base + PACKED_RMS_TILE, k0 : k0 + HEAD_TILE] = pl.cast(
                normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )
        kv_rope_norm = pooled_kv[final_base : final_base + PACKED_RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        # A3 interleaved swap-gather (matches decode): single data gather + sign trick instead of
        # the P0101/P1010 de-interleave gather + rotate + re-interleave scatter.
        # out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]; idx built in-kernel from pl.arange.
        rope_ones = pl.full([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)                                       # j>>1
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))                                          # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)                                                # [-1,+1,...]
        cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_kv[final_base : final_base + PACKED_RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM] = pl.cast(
            rope_rot,
            target_type=pl.BF16,
            mode="rint",
        )

    for final_block in pl.spmd(MAX_CMP_WRITES // PACKED_RMS_TILE, name_hint="prefill_idx_c4_kv_hadamard"):
        final_base = final_block * PACKED_RMS_TILE
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            final_acc = pl.matmul(
                normed_kv[final_base : final_base + PACKED_RMS_TILE, 0:HEAD_DIM],
                hadamard[0:HEAD_DIM, o0 : o0 + OUT_TILE],
                out_dtype=pl.FP32,
            )
            final_kv[final_base : final_base + PACKED_RMS_TILE, o0 : o0 + OUT_TILE] = final_acc

    for final_block in pl.spmd(MAX_CMP_WRITES // PACKED_RMS_TILE, name_hint="prefill_idx_c4_cache_write"):
        final_base = final_block * PACKED_RMS_TILE
        # C8 quant-on-write: per-row INT8 quant of the bf16-rounded block + per-position dequant scale
        kv_blk_f32 = pl.cast(
            pl.cast(final_kv[final_base : final_base + PACKED_RMS_TILE, 0:HEAD_DIM], target_type=pl.BF16, mode="rint"),
            target_type=pl.FP32)
        # amax = max(|x|); abs-based (max(row_max, -row_min) is wrong on signed KV)
        kv_amax = pl.reshape(pl.row_max(pl.abs(kv_blk_f32)), [1, PACKED_RMS_TILE])
        kv_amax = pl.maximum(kv_amax, pl.full([1, PACKED_RMS_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS))
        kv_scale_q_row = pl.div(pl.full([1, PACKED_RMS_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax)
        kv_scale_dq_col = pl.reshape(pl.recip(kv_scale_q_row), [PACKED_RMS_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [PACKED_RMS_TILE, 1])
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode="rint")
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode="round")
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode="trunc")
        for final_dt in pl.range(PACKED_RMS_TILE):
            final_i = final_base + final_dt
            dst_row_raw = pl.read(write_dst_map, [0, final_i])
            if dst_row_raw >= 0:
                dst_row = pl.cast(dst_row_raw, pl.INDEX)
                idx_kv_cache_flat[dst_row : dst_row + 1, 0:HEAD_DIM] = kv_i8_blk[final_dt : final_dt + 1, :]
                # scale is one value per position; a [1,1] tile store is sub-32B, so scalar-write it
                pl.write(idx_kv_scale_flat, [dst_row, 0], pl.read(kv_scale_dq_col, [final_dt, 0]))
            else:
                keepalive_row = PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE - MAX_CMP_WRITES + final_i
                idx_kv_cache_flat[keepalive_row : keepalive_row + 1, 0:HEAD_DIM] = idx_kv_cache_flat[
                    keepalive_row : keepalive_row + 1,
                    0:HEAD_DIM,
                ]
                pl.write(idx_kv_scale_flat, [keepalive_row, 0], pl.read(idx_kv_scale_flat, [keepalive_row, 0]))

    for update_idx in pl.spmd(T * PACKED_PROJ_BLOCKS, name_hint="prefill_idx_c4_state_update"):
        update_ob = update_idx % PACKED_PROJ_BLOCKS
        update_t = update_idx // PACKED_PROJ_BLOCKS
        update_o0 = update_ob * OUT_TILE
        if update_t < num_tokens:
            state_row_raw = pl.read(inner_state_slot_mapping, [update_t])
            if state_row_raw >= 0:
                state_row = pl.cast(state_row_raw, pl.INDEX)
                update_pos = pl.read(position_ids, [update_t])
                ape_slot = pl.cast(update_pos % COMPRESS_RATIO, pl.INDEX)
                ape_row = ape[ape_slot : ape_slot + 1, update_o0 : update_o0 + OUT_TILE]
                pool_dep = pl.mul(pooled_kv[0:1, 0:OUT_TILE], 0.0)
                kv_state_flat[state_row : state_row + 1, update_o0 : update_o0 + OUT_TILE] = pl.add(
                    kv_proj_scratch[
                        update_t : update_t + 1,
                        update_o0 : update_o0 + OUT_TILE,
                    ],
                    pool_dep,
                )
                score_state_flat[state_row : state_row + 1, update_o0 : update_o0 + OUT_TILE] = pl.add(
                    pl.add(
                        score_proj_scratch[update_t : update_t + 1, update_o0 : update_o0 + OUT_TILE],
                        ape_row,
                    ),
                    pool_dep,
                )

    idx_kv_cache = pl.reshape(idx_kv_cache_flat, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    idx_kv_scale = pl.reshape(idx_kv_scale_flat, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1])
    kv_state = pl.reshape(kv_state_flat, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, OUT_DIM])
    return idx_kv_cache, idx_kv_scale, kv_state, score_state


@pl.jit
def prefill_indexer_compressor_test(
    x: pl.Tensor[[T, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[MAX_CMP_WRITES, HEAD_DIM], pl.INT8]],
    kv_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32]],
    score_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32]],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    prefill_indexer_compressor(
        x, kv_state, score_state, inner_compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        hadamard, idx_kv_cache, idx_kv_scale, idx_block_table, position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping,
    )
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for kv_block in pl.spmd(MAX_CMP_WRITES // PACKED_RMS_TILE, name_hint="prefill_idx_c4_kv_test_extract"):
        kv_base = kv_block * PACKED_RMS_TILE
        for kv_dt in pl.range(PACKED_RMS_TILE):
            kv_i = kv_base + kv_dt
            src_row_raw = pl.cast(-1, pl.INT64)
            write_seen = pl.cast(0, pl.INDEX)
            for scan_w in pl.range(T):
                if scan_w < num_tokens:
                    scan_slot_raw = pl.read(idx_slot_mapping, [scan_w])
                    if scan_slot_raw >= 0:
                        if write_seen == kv_i:
                            src_row_raw = scan_slot_raw
                        write_seen = write_seen + 1
            if src_row_raw >= 0:
                src_row = pl.cast(src_row_raw, pl.INDEX)
                # C8 readback: raw INT8 cache rows in compressed order (dequant scale checked separately
                # via idx_kv_scale). A vector INT8->float widen mis-lanes, so expose the int8 as-is.
                kv[kv_i : kv_i + 1, 0:HEAD_DIM] = idx_kv_cache_flat[src_row : src_row + 1, 0:HEAD_DIM]
            else:
                # INT8 zero via the fp16->int8 cast (a direct pl.full INT8 hits an i8 texpands wall)
                kv[kv_i : kv_i + 1, 0:HEAD_DIM] = pl.cast(
                    pl.full([1, HEAD_DIM], dtype=pl.FP16, value=0.0), target_type=pl.INT8, mode="trunc")
    return kv, kv_state, score_state, idx_kv_cache, idx_kv_scale


