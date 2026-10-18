# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Evaluated implementation closure; fixed operands come from public config."""

from config import CASE_CONFIG as M

import os

import pypto.language as pl

T_DYN = pl.dynamic("T_DYN")  # T = B * S

HC_PRE_IMPL = os.environ.get("DSV4_HC_PRE_IMPL", "separate").lower()

D = M.hidden_size

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim

HC_DIM_INV = 1.0 / HC_DIM

HC_SINKHORN_ITER = M.hc_sinkhorn_iters

HC_EPS = M.hc_eps

NORM_EPS = M.rms_norm_eps

MIX_PAD = 32  # mix_hc (24) padded to a 32-wide cube N / vector row

HC_PAD = 8  # hc (4) padded for 32B-aligned vector ops

NUM_CORES = 24

T_TILE = 8  # vector row-tile (RMS / cast / split / mix_x)

LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile

COMB_T_TILE = 8  # sinkhorn row-tile

RMS_K_CHUNK = 512  # cast / rms K-fragment

LINEAR_K_CHUNK = 256  # cube K-fragment per matmul_acc (32x256x4 FP32 weight fits L0B)

D_CHUNK = 256  # mix_x inner D-fragment (BF16 load = 1KB, 512B-aligned)

D_SPMD = 1024  # mix_x D per spmd block: decode fans 4096 reduce over D/D_SPMD cores

LINEAR_OK = 4

LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK

LINEAR_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // LINEAR_K_CHUNK

RMS_OK = 8

RMS_K_PER_SPLIT = HC_DIM // RMS_OK

RMS_CHUNKS_PER_SPLIT = RMS_K_PER_SPLIT // RMS_K_CHUNK

MIXX_DS = D // D_SPMD  # mix_x tasks per token-tile (4)

assert HC_MULT == 4, (
    f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}; "
    "regenerate the pre0..pre3 / row0..row3 unrolling before using it."
)

assert HC_DIM % LINEAR_K_CHUNK == 0

assert HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0

assert D % D_SPMD == 0 and D_SPMD % D_CHUNK == 0

@pl.jit.inline
def _hc_pre_syncall(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_reshaped = pl.reshape(hc_base, [1, MIX_HC])

    # Cross-barrier intermediates: allocated ONCE in GM/HBM, live across the phases.
    # x arrives as FP32 (hc residual stream is FP32 end-to-end), so there is no x_fp32
    # staging buffer: linear / rms read x_flat directly.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    linear_partial_rows = LINEAR_OK * t_linear
    mixes_partials = pl.create_tensor([linear_partial_rows, MIX_PAD], dtype=pl.FP32)
    # post_pad_store is SAME-CORE scratch: write_post writes its token-tile's post gate then
    # reads it back on the SAME core (no barrier). It is kept (not deleted) because its
    # downstream pl.load->pl.store needs an HC_PAD-wide (32B-aligned) tile -- a narrow
    # [T_TILE, HC_MULT] FP32 tile is 16B rows and pto.alloc_tile rejects it. pre (mix_x) and
    # comb (loaded straight from mixes_raw) are consumed in-tile, so they need no scratch.
    post_pad_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)  # HC_PAD-wide same-core scratch; write_post narrows to post
    # Only the final partial token tile uses these fixed-size staging buffers.
    x_mixed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    comb_tail_store = pl.create_tensor([COMB_T_TILE, HC_PAD * HC_MULT], dtype=pl.FP32)
    # inv_rms same-core scratch for the comb gate: the per-token inv is a [T_TILE,1] COLUMN,
    # which the comb groups (Tiles) need as a Tile too. pto rejects a transposed [T_TILE,1]
    # tile (row-major, 4B row < 32B) and forbids Tile x Tensor ops, so the tensor-world inv
    # is spilled to this [t_linear,1] buffer and loaded back as a [T_TILE,1] Tile (8 floats,
    # vs the 128-float [.,16] comb_logits round-trip this replaces).
    inv_gm = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    sq_sum_acc = pl.create_tensor([1, t_linear], dtype=pl.FP32)  # RMS split-K sum-of-squares (row layout: [1,T_TILE] tiles stay 32B-aligned)
    sq_partials = pl.create_tensor([RMS_OK, t_linear], dtype=pl.FP32)

    # Per-phase grid-stride bounds (dynamic in t_dim; grid-stride round-robins any T over cores).
    tt_n = token_tiles                 # token-tiles (pre / post / comb / rsqrt / sinkhorn / write_post base)
    lin_n = (t_linear // LINEAR_T_TILE) * LINEAR_OK  # linear fans over row-block x OK K-slice
    rms_n = tt_n * RMS_OK             # rms fans over token-tile x K-slice (split-K sum-of-squares)
    linear_reduce_n = t_linear // LINEAR_T_TILE  # linear partials reduced per row-block
    reduce_n = linear_reduce_n + tt_n  # flattened reduce pool: linear row-blocks | rms token-tiles
    mixx_n = tt_n * MIXX_DS           # mix_x fans over token-tile x D-slice
    pool_d = 2 * tt_n + mixx_n        # phase-D flattened pool: sinkhorn(tt_n)|mix_x(mixx_n)|write_post(tt_n)

    with pl.spmd(NUM_CORES, name_hint="hc_pre_fused", sync_start=True, allow_early_resolve=True) as _hc_tid:  # inline form requires the TaskId capture
        core = pl.tile.get_block_idx()  # 0 .. NUM_CORES-1

        # Cube linear partials and vector RMS partials run concurrently.
        for task in pl.range(core, lin_n, NUM_CORES):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)  # last row-block spills past t_dim; valid_shape zero-fills the tail
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
                k0 = k_base + kb * LINEAR_K_CHUNK
                x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, k0], valid_shape=[t_rows, LINEAR_K_CHUNK])
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[MIX_HC, LINEAR_K_CHUNK])
                if kb == 0:
                    acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
            mixes_partials = pl.assemble(
                mixes_partials,
                acc,
                [linear_split * t_linear + t0, 0],
            )
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for task in pl.range(lane, rms_n, NUM_CORES * 2):
                t0 = (task // RMS_OK) * T_TILE
                rms_split = task % RMS_OK
                k_base = rms_split * RMS_K_PER_SPLIT
                valid_rows = pl.min(T_TILE, t_dim - t0)
                sq_part = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(RMS_CHUNKS_PER_SPLIT, stage=4):
                    k0 = k_base + kb * RMS_K_CHUNK
                    x_chunk = pl.slice(
                        x_flat,
                        [T_TILE, RMS_K_CHUNK],
                        [t0, k0],
                        valid_shape=[valid_rows, RMS_K_CHUNK],
                    )
                    sq_part = pl.add(sq_part, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]))
                sq_partials = pl.assemble(sq_partials, sq_part, [rms_split, t0])
        pl.system.syncall(core_type="mix")

        # Split-K partials are reduced in ascending K order.
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for reduce_task in pl.range(lane, reduce_n, NUM_CORES * 2):
                if reduce_task < linear_reduce_n:
                    linear_t0 = reduce_task * LINEAR_T_TILE
                    mixes_total = mixes_partials[
                        linear_t0 : linear_t0 + LINEAR_T_TILE,
                        0:MIX_PAD,
                    ]
                    for linear_split in pl.range(1, LINEAR_OK):
                        partial_t0 = linear_split * t_linear + linear_t0
                        mixes_total = pl.add(
                            mixes_total,
                            mixes_partials[
                                partial_t0 : partial_t0 + LINEAR_T_TILE,
                                0:MIX_PAD,
                            ],
                        )
                    mixes_raw = pl.assemble(mixes_raw, mixes_total, [linear_t0, 0])
                else:
                    rms_t0 = (reduce_task - linear_reduce_n) * T_TILE
                    sq_total = sq_partials[0:1, rms_t0 : rms_t0 + T_TILE]
                    for rms_split in pl.range(1, RMS_OK):
                        sq_total = pl.add(
                            sq_total,
                            sq_partials[rms_split : rms_split + 1, rms_t0 : rms_t0 + T_TILE],
                        )
                    sq_sum_acc = pl.assemble(sq_sum_acc, sq_total, [0, rms_t0])
        pl.system.syncall(core_type="mix")

        # ===================== PHASE C+D FUSED: gate + sinkhorn/mix_x/write_post (AIV) =====
        # Phase C (rsqrt + scale + sigmoid gates) is FOLDED into each Phase-D task: every
        # task recomputes the gate it needs from the published mixes_raw + sq_sum_acc
        # on its OWN core, just before using it. Recompute is nearly free here (latency-bound
        # kernel), and it deletes an entire cross-core barrier plus the pre/post/comb gate
        # round-trips -- the C->D handoff was the reason the old Barrier 3 existed. comb_logits
        # stays only as SAME-CORE scratch (assemble then load back, no barrier).
        #
        # The three kinds share ONE flattened work pool so they run on DISJOINT cores in
        # parallel -- otherwise per-type loops all start at the same lane base, stacking
        # sinkhorn (a 20-iter serial-recurrence latency floor) + that lane's mix_x/write_post
        # onto lane 0 while the other lanes idle. sinkhorn is placed first in the pool so its
        # long chain starts immediately alongside the mix_x lanes. No trailing barrier
        # (kernel end publishes the outputs).
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            # ONE flattened pool [sinkhorn(tt_n) | mix_x(mixx_n) | write_post(tt_n)] so grid-
            # stride lands the three kinds on DISJOINT lanes -- sinkhorn, all mix_x D-slices,
            # and write_post run on different cores concurrently (not serialized per lane).
            for gw in pl.range(lane, pool_d, NUM_CORES * 2):
                if gw < tt_n:
                    t0 = gw * COMB_T_TILE
                    valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
                    # Fold Phase C's comb gate here AND skip the comb_logits GM round-trip:
                    # load the 4 comb groups DIRECTLY from mixes_raw (pad-capable loads at cols
                    # 8/12/16/20 within the 32-wide MIX_PAD row), then apply inv_rms * scale2 +
                    # group bias PER GROUP in-tile. The previous version assembled a [.,16]
                    # comb_logits tile to GM and loaded the 4 misaligned 4-wide groups back
                    # (pto cannot slice a 16B sub-tile in-register); loading straight from the
                    # matmul output drops that store+load, matching the prefill path.
                    # inv_rms for the comb gate must be a TILE (the comb groups below are
                    # Tiles). Compute it in tensor-world, spill to inv_gm, and load it back as a
                    # [T_TILE,1] Tile (a transposed [T_TILE,1] tile is rejected -- 4B row).
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]  # [1,T_TILE] tensor
                    inv_col_tensor = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [COMB_T_TILE, 1])
                    inv_gm = pl.assemble(inv_gm, inv_col_tensor, [t0, 0])
                    inv_col_t = pl.load(
                        inv_gm,
                        [t0, 0],
                        [COMB_T_TILE, 1],
                        valid_shape=[valid_rows, 1],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    comb_off = HC_MULT * 2
                    mix_g0 = pl.load(mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g1 = pl.load(mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g2 = pl.load(mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g3 = pl.load(mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    cb0 = pl.load(hc_reshaped, [0, comb_off + 0 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    cb1 = pl.load(hc_reshaped, [0, comb_off + 1 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    cb2 = pl.load(hc_reshaped, [0, comb_off + 2 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    cb3 = pl.load(hc_reshaped, [0, comb_off + 3 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
                    row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
                    row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
                    row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
                    row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
                    row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
                    row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
                    row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

                    row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    row0_max = pl.row_max(row0_p, row_max_tmp)
                    row1_max = pl.row_max(row1_p, row_max_tmp)
                    row2_max = pl.row_max(row2_p, row_max_tmp)
                    row3_max = pl.row_max(row3_p, row_max_tmp)
                    row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
                    row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
                    row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
                    row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
                    row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
                    row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
                    row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
                    row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
                    row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
                    row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
                    row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
                    row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)

                    row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
                    row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
                    row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
                    row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
                    row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
                    row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
                    row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
                    row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

                    row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
                    col_sum = pl.add(col_sum, HC_EPS)
                    row0_cur = pl.div(row0_eff, col_sum)
                    row1_cur = pl.div(row1_eff, col_sum)
                    row2_cur = pl.div(row2_eff, col_sum)
                    row3_cur = pl.div(row3_eff, col_sum)

                    for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
                        row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
                        row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
                        row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
                        row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
                        row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
                        row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
                        row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
                        row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
                        col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
                        col_sum = pl.add(col_sum, HC_EPS)
                        row0_cur = pl.div(row0_norm, col_sum)
                        row1_cur = pl.div(row1_norm, col_sum)
                        row2_cur = pl.div(row2_norm, col_sum)
                        row3_cur = pl.div(row3_norm, col_sum)

                    if valid_rows == COMB_T_TILE:
                        row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
                        row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
                        row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
                        row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
                        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
                        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
                        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
                        pl.store(row3_out, [t0, 3 * HC_MULT], comb)
                    else:
                        pl.store(row0_cur, [0, 0 * HC_PAD], comb_tail_store)
                        pl.store(row1_cur, [0, 1 * HC_PAD], comb_tail_store)
                        pl.store(row2_cur, [0, 2 * HC_PAD], comb_tail_store)
                        pl.store(row3_cur, [0, 3 * HC_PAD], comb_tail_store)
                        row0_tail = pl.load(
                            comb_tail_store,
                            [0, 0 * HC_PAD],
                            [COMB_T_TILE, HC_PAD],
                            valid_shape=[valid_rows, HC_MULT],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        row1_tail = pl.load(
                            comb_tail_store,
                            [0, 1 * HC_PAD],
                            [COMB_T_TILE, HC_PAD],
                            valid_shape=[valid_rows, HC_MULT],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        row2_tail = pl.load(
                            comb_tail_store,
                            [0, 2 * HC_PAD],
                            [COMB_T_TILE, HC_PAD],
                            valid_shape=[valid_rows, HC_MULT],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        row3_tail = pl.load(
                            comb_tail_store,
                            [0, 3 * HC_PAD],
                            [COMB_T_TILE, HC_PAD],
                            valid_shape=[valid_rows, HC_MULT],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
                        pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
                        pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
                        pl.store(row3_tail, [t0, 3 * HC_MULT], comb)

                elif gw < tt_n + mixx_n:
                    blk = gw - tt_n
                    t0 = (blk // MIXX_DS) * T_TILE
                    d_base = (blk % MIXX_DS) * D_SPMD
                    valid_rows = pl.min(T_TILE, t_dim - t0)
                    # Fold Phase C's pre gate: pre = sigmoid(mix[:, :hc] * inv * scale0 + base) + eps.
                    # Recomputed here from mixes_raw (every D-slice of a token-tile redoes this
                    # tiny [T_TILE, HC_PAD] sigmoid -- free, and it drops the pre_val HBM buffer).
                    ssq_row = sq_sum_acc[0:1, t0:t0 + T_TILE]  # [1,T_TILE] row keeps the rsqrt tile 32B-aligned
                    inv_col = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [T_TILE, 1])
                    pre_base = hc_reshaped[0:1, 0:HC_PAD]
                    pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
                    pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
                    pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
                    # pin the valid shape: the inline gate carries valid=?x? from the dynamic-
                    # offset mixes_raw slice, but the transpose below wants a static 8x8 tile.
                    pre_val = pl.set_validshape(pre_val, T_TILE, HC_PAD)
                    pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
                    pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
                    pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
                    pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
                    pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
                    for db in pl.pipeline(D_SPMD // D_CHUNK, stage=2):
                        d0 = d_base + db * D_CHUNK
                        x0 = pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 0 * D + d0], valid_shape=[valid_rows, D_CHUNK])
                        x1 = pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 1 * D + d0], valid_shape=[valid_rows, D_CHUNK])
                        x2 = pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 2 * D + d0], valid_shape=[valid_rows, D_CHUNK])
                        x3 = pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 3 * D + d0], valid_shape=[valid_rows, D_CHUNK])
                        y0 = pl.row_expand_mul(x0, pre0)
                        y1 = pl.row_expand_mul(x1, pre1)
                        y2 = pl.row_expand_mul(x2, pre2)
                        y3 = pl.row_expand_mul(x3, pre3)
                        y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
                        y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                        if valid_rows == T_TILE:
                            x_mixed = pl.assemble(x_mixed, y_bf16, [t0, d0])
                        else:
                            x_mixed_tail_store = pl.assemble(x_mixed_tail_store, y_bf16, [0, d0])
                            y_out = pl.load(
                                x_mixed_tail_store,
                                [0, d0],
                                [T_TILE, D_CHUNK],
                                valid_shape=[valid_rows, D_CHUNK],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            pl.store(y_out, [t0, d0], x_mixed)

                else:
                    ob = gw - tt_n - mixx_n
                    t0 = ob * COMB_T_TILE
                    valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
                    # Fold Phase C's post gate: post = 2 * sigmoid(mix[:, hc:2hc] * inv * scale1 + base).
                    # The gate is tensor-world (mixes_raw slice -> sigmoid), and pl.store needs a Vec
                    # TILE -- so post_pad is stashed HC_PAD-wide in same-core scratch and loaded back as
                    # an aligned [COMB_T_TILE, HC_PAD] tile (a direct [.,HC_MULT] store would need a
                    # 16B-row FP32 tile, which pto.alloc_tile rejects). pl.store then narrows valid
                    # [COMB_T_TILE, HC_MULT] into the 4-wide post output.
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]  # [1,COMB_T_TILE] row keeps the rsqrt tile 32B-aligned
                    inv_col = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [COMB_T_TILE, 1])
                    post_base = hc_reshaped[0:1, HC_MULT:HC_MULT + HC_PAD]
                    post_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + COMB_T_TILE, HC_MULT:HC_MULT + HC_PAD], inv_col), scale1)
                    post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
                    post_pad = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
                    post_pad_store = pl.assemble(post_pad_store, post_pad, [t0, 0])
                    if valid_rows == COMB_T_TILE:
                        post_full = pl.load(post_pad_store, [t0, 0], [COMB_T_TILE, HC_PAD],
                                            valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                        pl.store(post_full, [t0, 0], post)
                    else:
                        post_tail = pl.load(post_pad_store, [t0, 0], [COMB_T_TILE, HC_PAD],
                                            valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
                        pl.store(post_tail, [t0, 0], post)
    return x_mixed

@pl.jit.inline
def _hc_pre_separate(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """Multi-scope (separate-task) hc_pre -- the pre-#684 structure, applied to ALL T.

    Identical math to _hc_pre_syncall, but each work-type is its OWN pl.spmd task instead
    of one full-occupancy pl.spmd + hard pl.system.syncall barriers. The runtime task
    graph orders the scopes by their GM read/write dependencies (seed -> linear atomic-add
    -> split_pre_post -> write_post / comb_sinkhorn / mix_x), so this path needs dep_gen ON
    (the __main__ harness sets enable_dep_gen accordingly). Tile sizes are aligned to the
    tuned syncall version (D_CHUNK / D_SPMD / LINEAR_* / t_linear round-up); cross-barrier
    buffers are sized to the dynamic t_linear (the 8->16 padded row count), not a static
    T_MAX. Kept as a switchable alternative to the fused kernel (perf is not the goal here).
    """
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # for per-group comb base loads in comb_sinkhorn

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    # x arrives as FP32 from the prior hc_post (the hc residual stream is FP32 end-to-end),
    # so the old BF16->FP32 cast scope + x_fp32 staging buffer are gone: linear / rms read
    # x_flat directly. The 8->16 pad rows of the cube tile are never materialized -- the
    # linear matmul masks them with valid_shape (zero-fill past t_dim), and rms only reads
    # the t_dim real rows.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    linear_partial_rows = LINEAR_OK * t_linear
    mixes_partials = pl.create_tensor([linear_partial_rows, MIX_PAD], dtype=pl.FP32)

    # rms: full-K sum-of-squares per token-tile -> inv_rms (one scope, no split-K).
    for t in pl.spmd(token_tiles, name_hint="hc_pre_rms", allow_early_resolve=True):
        t0 = t * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_CHUNK, stage=4):
            k0 = kb * RMS_K_CHUNK
            if valid_rows == T_TILE:
                x_chunk_full = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk_full, x_chunk_full)), [1, T_TILE]),
                )
            else:
                x_chunk_tail = pl.slice(
                    x_flat,
                    [T_TILE, RMS_K_CHUNK],
                    [t0, k0],
                    valid_shape=[valid_rows, RMS_K_CHUNK],
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk_tail, x_chunk_tail)), [1, T_TILE]),
                )
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS), high_precision=True), [T_TILE, 1])
        inv_rms = pl.assemble(inv_rms, inv, [t0, 0])

    # Linear split-K partials are reduced in ascending K order.
    for task in pl.spmd((t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="hc_pre_linear", allow_early_resolve=True):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        linear_split = task % LINEAR_OK
        k_base = linear_split * LINEAR_K_PER_SPLIT
        t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)  # last row-block spills past t_dim; valid_shape zero-fills the tail
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
            k0 = k_base + kb * LINEAR_K_CHUNK
            x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, k0], valid_shape=[t_rows, LINEAR_K_CHUNK])
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[MIX_HC, LINEAR_K_CHUNK])
            if kb == 0:
                acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
        mixes_partials = pl.assemble(mixes_partials, acc, [linear_split * t_linear + t0, 0])

    for linear_block in pl.spmd(
        t_linear // LINEAR_T_TILE,
        name_hint="hc_pre_linear_reduce",
        allow_early_resolve=True,
    ):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            mixes_total = pl.add(
                mixes_total,
                mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD],
            )
        mixes_raw = pl.assemble(mixes_raw, mixes_total, [linear_t0, 0])

    # split_pre_post: inv_rms-scaled pre gate -> pre_val_store (for mix_x), post gate -> post.
    # Both compute at HC_PAD width; post narrows to HC_MULT via a valid-shape slice (an 8-wide
    # 32B tile, 4 cols valid -- a bare 4-wide slice allocs a 16B tile ptoas rejects). comb gate
    # lives in comb_sinkhorn.
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    # Only the final partial token tile uses these fixed-size staging buffers.
    post_tail_store = pl.create_tensor([T_TILE, HC_PAD], dtype=pl.FP32)
    x_mixed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    comb_tail_store = pl.create_tensor([COMB_T_TILE, HC_PAD * HC_MULT], dtype=pl.FP32)
    for ob in pl.spmd(token_tiles, name_hint="split_pre_post", allow_early_resolve=True):
        t0 = ob * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        inv_col = inv_rms[t0:t0 + T_TILE, 0:1]

        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_val_store = pl.assemble(pre_val_store, pre_val, [t0, 0])

        post_base = pl.reshape(hc_base[HC_MULT:HC_MULT + HC_PAD], [1, HC_PAD])
        post_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, HC_MULT:HC_MULT + HC_PAD], inv_col), scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_pad = pl.mul(post_sig, 2.0)
        if valid_rows == T_TILE:
            post[t0:t0 + T_TILE, 0:HC_MULT] = pl.slice(
                post_pad,
                [T_TILE, HC_PAD],
                [0, 0],
                valid_shape=[T_TILE, HC_MULT],
            )
        else:
            post_tail_store = pl.assemble(post_tail_store, post_pad, [0, 0])
            post_tile = pl.load(
                post_tail_store,
                [0, 0],
                [T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(post_tile, [t0, 0], post)

    # comb_sinkhorn: comb gate (direct from mixes_raw, no comb_logits round-trip) + softmax +
    # 20-iter Sinkhorn (column-first) -> comb. inv_rms is already a [t_linear,1] column buffer,
    # so the [T_TILE,1] inv tile loads directly (no transpose / no spill); the 4 comb groups
    # load pad-capable from mixes_raw at cols 8/12/16/20, then inv_rms * scale2 + group bias.
    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.load(
            inv_rms,
            [t0, 0],
            [COMB_T_TILE, 1],
            valid_shape=[valid_rows, 1],
            target_memory=pl.MemorySpace.Vec,
        )
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g1 = pl.load(mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g2 = pl.load(mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g3 = pl.load(mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[valid_rows, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb0 = pl.load(hc_base_2d, [0, comb_off + 0 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb1 = pl.load(hc_base_2d, [0, comb_off + 1 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb2 = pl.load(hc_base_2d, [0, comb_off + 2 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb3 = pl.load(hc_base_2d, [0, comb_off + 3 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
        row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
        row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
        row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row0_max = pl.row_max(row0_p, row_max_tmp)
        row1_max = pl.row_max(row1_p, row_max_tmp)
        row2_max = pl.row_max(row2_p, row_max_tmp)
        row3_max = pl.row_max(row3_p, row_max_tmp)
        row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
        row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
        row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
        row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
        row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
        row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
        row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
        row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
        row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
        row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
        row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
        row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)

        row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

        row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
        col_sum = pl.add(col_sum, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum)
        row1_cur = pl.div(row1_eff, col_sum)
        row2_cur = pl.div(row2_eff, col_sum)
        row3_cur = pl.div(row3_eff, col_sum)

        for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
            row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
            row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
            row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
            row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
            col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_norm, col_sum)
            row1_cur = pl.div(row1_norm, col_sum)
            row2_cur = pl.div(row2_norm, col_sum)
            row3_cur = pl.div(row3_norm, col_sum)

        if valid_rows == COMB_T_TILE:
            row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
            row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
            row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
            row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
            pl.store(row0_out, [t0, 0 * HC_MULT], comb)
            pl.store(row1_out, [t0, 1 * HC_MULT], comb)
            pl.store(row2_out, [t0, 2 * HC_MULT], comb)
            pl.store(row3_out, [t0, 3 * HC_MULT], comb)
        else:
            pl.store(row0_cur, [0, 0 * HC_PAD], comb_tail_store)
            pl.store(row1_cur, [0, 1 * HC_PAD], comb_tail_store)
            pl.store(row2_cur, [0, 2 * HC_PAD], comb_tail_store)
            pl.store(row3_cur, [0, 3 * HC_PAD], comb_tail_store)
            row0_tail = pl.load(
                comb_tail_store,
                [0, 0 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row1_tail = pl.load(
                comb_tail_store,
                [0, 1 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row2_tail = pl.load(
                comb_tail_store,
                [0, 2 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row3_tail = pl.load(
                comb_tail_store,
                [0, 3 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
            pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
            pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
            pl.store(row3_tail, [t0, 3 * HC_MULT], comb)

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D (D/D_SPMD tasks per tile).
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0:t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_CHUNK, stage=2):
            d0 = d_base + db * D_CHUNK
            x0 = pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 0 * D + d0], valid_shape=[valid_rows, D_CHUNK])
            x1 = pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 1 * D + d0], valid_shape=[valid_rows, D_CHUNK])
            x2 = pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 2 * D + d0], valid_shape=[valid_rows, D_CHUNK])
            x3 = pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 3 * D + d0], valid_shape=[valid_rows, D_CHUNK])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            if valid_rows == T_TILE:
                x_mixed = pl.assemble(x_mixed, y_bf16, [t0, d0])
            else:
                x_mixed_tail_store = pl.assemble(x_mixed_tail_store, y_bf16, [0, d0])
                y_out = pl.load(
                    x_mixed_tail_store,
                    [0, d0],
                    [T_TILE, D_CHUNK],
                    valid_shape=[valid_rows, D_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(y_out, [t0, d0], x_mixed)
    return x_mixed

def _bind_hc_pre():
    """Define and return the public `hc_pre` inline kernel for the selected HC_PRE_IMPL.

    pypto constrains how the choice can be expressed:
      * a kernel-body branch on the module-global string (``if HC_PRE_IMPL == ...``) fails
        -- pypto's frontend treats it as device control flow and cannot resolve the name;
      * an alias (``hc_pre = _hc_pre_syncall``) fails at the call site -- pypto matches the
        call-site NAME against the callee's registered ``__name__`` ("hc_pre" != the impl).
    So `hc_pre` must BE a decorated inline named ``hc_pre`` that calls the chosen impl by
    its literal name; the selection is a plain-Python ``if`` at import (never seen by a
    kernel). Re-invoke to rebind after changing HC_PRE_IMPL (the __main__ --impl path).
    """
    if HC_PRE_IMPL == "separate":
        @pl.jit.inline
        def hc_pre(
            x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
            hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
            hc_scale: pl.Tensor[[3], pl.FP32],
            hc_base: pl.Tensor[[MIX_HC], pl.FP32],
            x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
            post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
            comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
        ):
            _hc_pre_separate(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
            return x_mixed
    else:
        @pl.jit.inline
        def hc_pre(
            x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
            hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
            hc_scale: pl.Tensor[[3], pl.FP32],
            hc_base: pl.Tensor[[MIX_HC], pl.FP32],
            x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
            post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
            comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
        ):
            _hc_pre_syncall(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
            return x_mixed
    return hc_pre

hc_pre = _bind_hc_pre()
