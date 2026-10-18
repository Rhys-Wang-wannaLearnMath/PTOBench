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

from config import (
    CASE_CONFIG as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_IDX_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_indexer_compressor import indexer_compressor
from rope_interleave import rope_interleave

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
COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_PHYSICAL_BLOCKS = 65
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = INNER_STATE_PHYSICAL_BLOCKS
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")
INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
SCORE_LEN = IDX_KV_LEN

# tiling
CACHE_TILE = 64
assert BLOCK_SIZE % CACHE_TILE == 0, "CACHE_TILE must not cross a paged idx_kv_cache block"
# REDUCE_TILE tiles the paged C8 cache one 128-row page per fused matmul+reduce step.
REDUCE_TILE = 128
# the fused score scope fans the cache-page loop across REDUCE_NSPLIT extra lanes: T * NSPLIT.
# T*NSPLIT=16 mixed blocks map to 16 AIC + 32 AIV (1:2), one clean wave on the 24+48 chip.
REDUCE_NSPLIT = 2
Q_TILE = 256
# Q_OUT_TILE is the per-task N granularity (sets idx_qr_proj task count); MM_N_TILE
# is the Mat-safe cube N-tile. Q_OUT_TILE fans Q_OUT_TILE // MM_N_TILE cube ops per
# task so task count halves without growing the [Q_TILE, MM_N_TILE] L1 wq load.
Q_OUT_TILE = 1024
T_PAD = ((T + 16 - 1) // 16) * 16  # T padded up to the 16-row cube M floor
MM_ROW_TILE = T_PAD  # one boxed matmul per task spans every token row
# INT32 Acc is MM_ROW_TILE * MM_N_TILE * 4B and must stay under the 128KiB L0C wall.
MM_N_TILE = min(512, (128 * 1024) // (MM_ROW_TILE * 4))
assert Q_OUT_TILE % MM_N_TILE == 0
# Dequant token tile: a whole-T [T, Q_OUT_TILE] FP32 tile does not fit UB.
DEQUANT_T_TILE = min(T, 8)
assert T % DEQUANT_T_TILE == 0
HEAD_DIM_TILE = 32
D_TILE = 512
# weights_proj splits K, not N: a [D_TILE, IDX_N_HEADS] row block reads contiguous GM,
# while an N slice would take 32B out of every 128B row. Each task writes its own
# partial row block, summed by a separate reduce scope -- a zero-seed + atomic-add
# assemble races here, since T_PAD == MM_ROW_TILE makes the seed a full-extent write.
# WEIGHTS_K_SLICE // D_TILE == 2, so the inner loop is a pl.range: a degenerate
# 2-iteration pl.pipeline(stage=2) miscompiles over matmul.
WEIGHTS_OK = 4
WEIGHTS_K_SLICE = D // WEIGHTS_OK
assert WEIGHTS_K_SLICE % D_TILE == 0
QH_QUANT_TILE = 64
# cube tile for q @ hadamard; L0C caps it at QH_MM_TILE * IDX_HEAD_DIM * 4B <= 64KiB.
QH_MM_TILE = 64
QH_HEAD_DIM_TILE = 64
ROPE_ROW_BLOCK = S * IDX_N_HEADS
# qr_rope SPMD tile == row block: one ROPE_ROW_TILE-row block per SPMD tile.
ROPE_ROW_TILE = 32
TOPK_HALF_LEN = SCORE_LEN // 2
TOPK_HALF_PAIR_OFFSET = 2 * TOPK_HALF_LEN
TOPK_PAIR_WIDTH = 2 * IDX_TOPK
assert SCORE_LEN == 2 * TOPK_HALF_LEN, "decode indexer topk expects an even score length"
assert TOPK_HALF_LEN == 2048, "decode indexer 4096-value topk uses two 2048-value halves"
assert IDX_TOPK <= TOPK_HALF_LEN, "per-half candidate list must cover the final topk width"


@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # C8 indexer cache: INT8 KV (quant-on-write) + per-position FP32 dequant scale; no bf16 cache.
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32],
    topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    qr_acc_pad = pl.create_tensor([T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
    for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="idx_qr_proj_matmul", allow_early_resolve=True):
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(qr, [T_PAD, Q_TILE], [0, q0], valid_shape=[T, Q_TILE])
                wq_tile = wq_b[q0 : q0 + Q_TILE, o_base + ns : o_base + ns + MM_N_TILE]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            qr_acc_pad[0:T_PAD, o_base + ns : o_base + ns + MM_N_TILE] = qr_acc
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="idx_qr_proj_dequant", allow_early_resolve=True):
        o_base = ot * Q_OUT_TILE
        wq_scale = pl.reshape(wq_b_scale[o_base : o_base + Q_OUT_TILE], [1, Q_OUT_TILE])
        for dq_t0 in pl.range(0, T, DEQUANT_T_TILE):
            acc_fp32 = pl.cast(
                qr_acc_pad[dq_t0 : dq_t0 + DEQUANT_T_TILE, o_base : o_base + Q_OUT_TILE],
                target_type=pl.FP32, mode="none")
            qr_scale_tile = qr_scale[dq_t0 : dq_t0 + DEQUANT_T_TILE, :]
            qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
            qr_proj[dq_t0 : dq_t0 + DEQUANT_T_TILE, o_base : o_base + Q_OUT_TILE] = qr_dequant

    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    # BF16 q for the Hadamard matmul: nope half rounded from the FP32 dequant, rope
    # half rotated then rounded.
    qr_bf16 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    # spmd over ROPE_ROW_TILE-row blocks; batch_idx = block base // ROPE_ROW_BLOCK
    # picks the per-batch cos/sin row. cos/sin arrive already interleave-duplicated and
    # sign-folded (built once by the caller), and col_expand_mul folds the [1, ROPE_HEAD_DIM]
    # row broadcast into the rotation multiply -- so no cos_il/sin_il tile is materialized
    # and no per-block dup-gather runs.
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sin_il_signed[j]
    #
    # The j^1 lane-swap index permutes data, so no host table can hold it -- but it is
    # block-invariant, and rebuilding it inside the spmd cost the same arange/trunc-cast/
    # lane/arithmetic chain on all 16 blocks. Built once here instead (same form as the
    # rope_swap scope in decode_sparse_attn_hca) and loaded per block. No pypto bitwise
    # op is reachable at the tensor level, so the fp32 arithmetic chain is the only form.
    rope_swap_idx_t = pl.create_tensor([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rope_swap_idx", allow_early_resolve=True):
        sw_col = pl.col_expand_mul(
            pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))                                                # j%2
        rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)                   # j^1

    for idx in pl.spmd(T * IDX_N_HEADS // ROPE_ROW_TILE, name_hint="qr_rope", allow_early_resolve=True):
        o0 = idx * ROPE_ROW_TILE
        batch_idx = o0 // ROPE_ROW_BLOCK
        rope_swap_idx = rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM]
        cos_row = cos[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM]
        sin_row = sin[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM]
        qr_nope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM]
        qr_rope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(
            pl.col_expand_mul(qr_rope_slice, cos_row), pl.col_expand_mul(qr_swapped, sin_row))
        qr_vec = pl.concat(pl.cast(qr_nope_slice, target_type=pl.BF16, mode="rint"), pl.cast(rope_rot, target_type=pl.BF16, mode="rint"))
        qr_bf16[o0 : o0 + ROPE_ROW_TILE, :] = qr_vec

    # cube-only scope: q @ hadamard lands in GM, keeping the vector amax/quant below
    # in its own scope so the two run as separate cube and vector tasks.
    qh_acc_gm = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_MM_TILE, name_hint="qr_hadamard_matmul", allow_early_resolve=True):
        o0 = idx * QH_MM_TILE
        qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32)
        qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc

    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_TILE, name_hint="qr_hadamard_quant", allow_early_resolve=True):
        o0 = idx * QH_QUANT_TILE
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h0 : h0 + QH_HEAD_DIM_TILE]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE]
            qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE] = qh_i8

    x_flat = pl.reshape(x, [T, D])
    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    weights_partial = pl.create_tensor([WEIGHTS_OK * MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(WEIGHTS_OK, name_hint="weights_proj", deps=[late_dep]) as _weights_tid:
        kb = pl.tile.get_block_idx()
        k_base = kb * WEIGHTS_K_SLICE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [0, d0], valid_shape=[pl.min(MM_ROW_TILE, T), D_TILE])
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
        weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :] = weights_acc

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="weights_proj_reduce"):
        w_sum = weights_partial[0:MM_ROW_TILE, :]
        for kb in pl.unroll(1, WEIGHTS_OK):
            w_sum = pl.add(w_sum, weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :])
        weights[0:MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)

    indexer_compressor(
        x, inner_kv,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        cos, sin, hadamard, idx_kv_cache, idx_kv_scale,
        position_ids, idx_slot_mapping, inner_state_slot_mapping,
        late_dep,
    )

    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_block_num * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * BLOCK_SIZE, 1])
    idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])
    score_flat = pl.reshape(score, [T, SCORE_LEN])

    # No score_init: the loop writes the valid region; the tail is never read (topk re-masks).
    # Fused mixed cube+vector: the per-page C8 matmul feeds the relu/weight/head-sum reduce
    # on chip, so cube page i+1 pipelines against vector page i (no score_acc_gm handoff).
    # slot_num=2: the cube->vector pipe defaults to an 8-slot ring (8 * 32KB = 256KB > UB);
    # a depth-2 ring (64KB) fits alongside the FP32 epilogue. Deeper rings (>=3) would cut the
    # handoff stalls but overflow Vec.
    for unit in pl.spmd(T * REDUCE_NSPLIT, name_hint="score", allow_early_resolve=True,
                        optimizations=[pl.split(pl.SplitMode.NONE, slot_num=2)]):
        tg = unit // REDUCE_NSPLIT
        split = unit - tg * REDUCE_NSPLIT
        b = tg // S
        s = tg - b * S
        clen_b = pl.read(kv_seq_lens, [b]) // COMPRESS_RATIO
        pos_t = pl.read(position_ids, [b, s])
        visible_len_t = pl.min(pl.min(clen_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN)
        cblk_t = (visible_len_t + REDUCE_TILE - 1) // REDUCE_TILE
        tb = b * S
        qb = b * S * IDX_N_HEADS
        qr_full = qr_hadamard_i8[qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS, 0 : IDX_HEAD_DIM]
        qh_scale_s = pl.reshape(qr_hadamard_scale_dq[qb + s * IDX_N_HEADS : qb + (s + 1) * IDX_N_HEADS, :], [1, IDX_N_HEADS])
        weights_row_s = pl.reshape(weights[tb + s : tb + s + 1, :], [1, IDX_N_HEADS])
        lane_iters = (cblk_t - split + REDUCE_NSPLIT - 1) // REDUCE_NSPLIT
        for cb_local in pl.pipeline(0, lane_iters, stage=2):
            cb = split + cb_local * REDUCE_NSPLIT
            cache0 = cb * REDUCE_TILE
            valid_len = pl.min(REDUCE_TILE, visible_len_t - cache0)
            idx_blk_id = pl.cast(
                pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + cb]),
                pl.INDEX,
            )
            kv0 = idx_blk_id * BLOCK_SIZE
            kv_i8_mat = kv_cache_i8_flat[kv0 : kv0 + BLOCK_SIZE, :]
            score_acc_red = pl.matmul(kv_i8_mat, qr_full, out_dtype=pl.INT32, b_trans=True)
            kv_dq_red = kv_scale_flat[kv0 : kv0 + REDUCE_TILE, :]  # paged per-position dequant scale
            score_tile_red = pl.cast(score_acc_red, target_type=pl.FP32, mode="none")
            score_tile_red = pl.col_expand_mul(score_tile_red, qh_scale_s)
            relu_score_red = pl.maximum(score_tile_red, 0.0)
            weighted_score_red = pl.col_expand_mul(relu_score_red, weights_row_s)
            # per-position dequant kv_dq_red applied after the head-sum
            weighted_score_row = pl.mul(pl.row_sum(weighted_score_red), kv_dq_red)
            weighted_score_s = pl.reshape(weighted_score_row, [1, REDUCE_TILE])
            weighted_score_valid_s = pl.fillpad(pl.set_validshape(weighted_score_s, 1, valid_len), pad_value=pl.PadValue.min)
            weighted_score_valid_s = pl.maximum(weighted_score_valid_s, FP32_NEG_INF)
            score_flat[tb + s : tb + s + 1, cache0 : cache0 + REDUCE_TILE] = weighted_score_valid_s

    topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])
    for t in pl.spmd(T, name_hint="topk", allow_early_resolve=True):
        invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
        topk_idxs_flat[t : t + 1, :] = invalid_idxs
        batch_idx = t // S
        token_s = t - batch_idx * S
        cache_len_b = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
        pos_t = pl.read(position_ids, [batch_idx, token_s])
        visible_len_t = pl.min(pl.min(cache_len_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN)
        if visible_len_t > 0:
            offset_i32 = pl.cast(offset, target_type=pl.INT32)
            score_full_raw = score_flat[t : t + 1, 0:SCORE_LEN]
            score_full = pl.fillpad(pl.set_validshape(score_full_raw, 1, visible_len_t), pad_value=pl.PadValue.min)
            score_full = pl.maximum(score_full, pl.full([1, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF))
            idx_init = pl.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
            sorted_full = pl.sort32(score_full, idx_init)
            sorted_full = pl.mrgsort(sorted_full, block_len=64)
            sorted_full = pl.mrgsort(sorted_full, block_len=256)
            sorted_full = pl.mrgsort(sorted_full, block_len=1024)

            # After the 1024 merge, the 4096-score row is two sorted 2048-score
            # runs. sort32/mrgsort keeps score/index pairs interleaved, so the
            # second 2048-score run starts at pair-lane offset 2 * 2048.
            half0_candidates = sorted_full[:, 0:TOPK_PAIR_WIDTH]
            half1_candidates = sorted_full[:, TOPK_HALF_PAIR_OFFSET : TOPK_HALF_PAIR_OFFSET + TOPK_PAIR_WIDTH]
            merged_candidates = pl.mrgsort(half0_candidates, half1_candidates)
            topk_pairs = merged_candidates[:, 0:TOPK_PAIR_WIDTH]
            topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid_topk = pl.min(IDX_TOPK, visible_len_t)
            topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
            topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)

    return score, topk_idxs

