# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime-only HCA implementation closure."""

import os
import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ
T_DYN = pl.dynamic('T_DYN')
HC_PRE_IMPL = os.environ.get('DSV4_HC_PRE_IMPL', 'separate').lower()
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps
MIX_PAD = 32
HC_PAD = 8
NUM_CORES = 24
T_TILE = 8
LINEAR_T_TILE = 16
COMB_T_TILE = 8
RMS_K_CHUNK = 512
LINEAR_K_CHUNK = 256
D_CHUNK = 256
D_SPMD = 1024
CAST_K_SPMD = 2048
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // LINEAR_K_CHUNK
RMS_OK = 8
RMS_K_PER_SPLIT = HC_DIM // RMS_OK
RMS_CHUNKS_PER_SPLIT = RMS_K_PER_SPLIT // RMS_K_CHUNK
CAST_KS = HC_DIM // CAST_K_SPMD
MIXX_DS = D // D_SPMD
assert HC_MULT == 4, f'hc_pre is specialized to HC_MULT == 4, got {HC_MULT}; regenerate the pre0..pre3 / row0..row3 unrolling before using it.'
assert DECODE_BATCH * DECODE_SEQ % T_TILE == 0
assert DECODE_BATCH * DECODE_SEQ % LINEAR_T_TILE == 0 or DECODE_BATCH * DECODE_SEQ <= LINEAR_T_TILE
assert HC_DIM % LINEAR_K_CHUNK == 0
assert HC_DIM % CAST_K_SPMD == 0 and CAST_K_SPMD % RMS_K_CHUNK == 0
assert HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0
assert D % D_SPMD == 0 and D_SPMD % D_CHUNK == 0

@pl.jit.inline
def _hc_pre_syncall(x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_scale: pl.Tensor[[3], pl.FP32], hc_base: pl.Tensor[[MIX_HC], pl.FP32], x_mixed: pl.Tensor[[T_DYN, D], pl.BF16], post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32], comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]):
    t_dim = pl.tensor.dim(x, 0)
    t_linear = (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_reshaped = pl.reshape(hc_base, [1, MIX_HC])
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    post_pad_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    inv_gm = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    sq_sum_acc = pl.create_tensor([1, t_linear], dtype=pl.FP32)
    tt_n = t_dim // T_TILE
    cast_n = tt_n * CAST_KS
    seed_n = t_linear // T_TILE
    lin_n = t_linear // LINEAR_T_TILE * LINEAR_OK
    rms_n = tt_n * RMS_OK
    mixx_n = tt_n * MIXX_DS
    pool_d = 2 * tt_n + mixx_n
    with pl.spmd(NUM_CORES, name_hint='hc_pre_fused', sync_start=True, allow_early_resolve=True) as _hc_tid:
        core = pl.tile.get_block_idx()
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for tc in pl.range(lane, seed_n, NUM_CORES * 2):
                ts0 = tc * T_TILE
                mixes_raw = pl.assemble(mixes_raw, pl.full([T_TILE, MIX_PAD], dtype=pl.FP32, value=0.0), [ts0, 0])
                sq_sum_acc = pl.assemble(sq_sum_acc, pl.full([1, T_TILE], dtype=pl.FP32, value=0.0), [0, ts0])
        pl.system.syncall(core_type='mix')
        for task in pl.range(core, lin_n, NUM_CORES):
            t0 = task // LINEAR_OK * LINEAR_T_TILE
            k_base = task % LINEAR_OK * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
                k0 = k_base + kb * LINEAR_K_CHUNK
                x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, k0], valid_shape=[t_rows, LINEAR_K_CHUNK])
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[MIX_HC, LINEAR_K_CHUNK])
                if kb == 0:
                    acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
            mixes_raw = pl.assemble(mixes_raw, acc, [t0, 0], atomic=pl.AtomicType.Add)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for task in pl.range(lane, rms_n, NUM_CORES * 2):
                t0 = task // RMS_OK * T_TILE
                k_base = task % RMS_OK * RMS_K_PER_SPLIT
                sq_part = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(RMS_CHUNKS_PER_SPLIT, stage=4):
                    k0 = k_base + kb * RMS_K_CHUNK
                    x_chunk = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]
                    sq_part = pl.add(sq_part, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]))
                sq_sum_acc = pl.assemble(sq_sum_acc, sq_part, [0, t0], atomic=pl.AtomicType.Add)
        pl.system.syncall(core_type='mix')
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for gw in pl.range(lane, pool_d, NUM_CORES * 2):
                if gw < tt_n:
                    t0 = gw * COMB_T_TILE
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]
                    inv_col_tensor = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [COMB_T_TILE, 1])
                    inv_gm = pl.assemble(inv_gm, inv_col_tensor, [t0, 0])
                    inv_col_t = pl.load(inv_gm, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
                    comb_off = HC_MULT * 2
                    mix_g0 = pl.load(mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g1 = pl.load(mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g2 = pl.load(mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g3 = pl.load(mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
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
                    row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
                    row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
                    row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
                    row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
                    pl.store(row0_out, [t0, 0 * HC_MULT], comb)
                    pl.store(row1_out, [t0, 1 * HC_MULT], comb)
                    pl.store(row2_out, [t0, 2 * HC_MULT], comb)
                    pl.store(row3_out, [t0, 3 * HC_MULT], comb)
                elif gw < tt_n + mixx_n:
                    blk = gw - tt_n
                    t0 = blk // MIXX_DS * T_TILE
                    d_base = blk % MIXX_DS * D_SPMD
                    ssq_row = sq_sum_acc[0:1, t0:t0 + T_TILE]
                    inv_col = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [T_TILE, 1])
                    pre_base = hc_reshaped[0:1, 0:HC_PAD]
                    pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
                    pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
                    pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
                    pre_val = pl.set_validshape(pre_val, T_TILE, HC_PAD)
                    pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
                    pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
                    pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
                    pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
                    pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
                    for db in pl.pipeline(D_SPMD // D_CHUNK, stage=2):
                        d0 = d_base + db * D_CHUNK
                        x0 = x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_CHUNK]
                        x1 = x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_CHUNK]
                        x2 = x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_CHUNK]
                        x3 = x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_CHUNK]
                        y0 = pl.row_expand_mul(x0, pre0)
                        y1 = pl.row_expand_mul(x1, pre1)
                        y2 = pl.row_expand_mul(x2, pre2)
                        y3 = pl.row_expand_mul(x3, pre3)
                        y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
                        x_mixed = pl.assemble(x_mixed, pl.cast(y_tile, target_type=pl.BF16, mode='rint'), [t0, d0])
                else:
                    ob = gw - tt_n - mixx_n
                    t0 = ob * COMB_T_TILE
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]
                    inv_col = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [COMB_T_TILE, 1])
                    post_base = hc_reshaped[0:1, HC_MULT:HC_MULT + HC_PAD]
                    post_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + COMB_T_TILE, HC_MULT:HC_MULT + HC_PAD], inv_col), scale1)
                    post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
                    post_pad = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
                    post_pad_store = pl.assemble(post_pad_store, post_pad, [t0, 0])
                    post_tile = pl.load(post_pad_store, [t0, 0], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    pl.store(post_tile, [t0, 0], post)
    return x_mixed

@pl.jit.inline
def _hc_pre_separate(x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_scale: pl.Tensor[[3], pl.FP32], hc_base: pl.Tensor[[MIX_HC], pl.FP32], x_mixed: pl.Tensor[[T_DYN, D], pl.BF16], post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32], comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]):
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
    t_linear = (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for t in pl.spmd(t_dim // T_TILE, name_hint='hc_pre_rms', allow_early_resolve=True):
        t0 = t * T_TILE
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_CHUNK, stage=4):
            k0 = kb * RMS_K_CHUNK
            x_chunk = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]))
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS), high_precision=True), [T_TILE, 1])
        inv_rms = pl.assemble(inv_rms, inv, [t0, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='hc_pre_seed', allow_early_resolve=True):
        for ts0 in pl.range(0, t_linear, T_TILE):
            mixes_raw[ts0:ts0 + T_TILE, 0:MIX_PAD] = pl.full([T_TILE, MIX_PAD], dtype=pl.FP32, value=0.0)
    for task in pl.spmd(t_linear // LINEAR_T_TILE * LINEAR_OK, name_hint='hc_pre_linear', allow_early_resolve=True):
        t0 = task // LINEAR_OK * LINEAR_T_TILE
        k_base = task % LINEAR_OK * LINEAR_K_PER_SPLIT
        t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
            k0 = k_base + kb * LINEAR_K_CHUNK
            x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, k0], valid_shape=[t_rows, LINEAR_K_CHUNK])
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[MIX_HC, LINEAR_K_CHUNK])
            if kb == 0:
                acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
        mixes_raw = pl.assemble(mixes_raw, acc, [t0, 0], atomic=pl.AtomicType.Add)
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    for ob in pl.spmd(t_dim // T_TILE, name_hint='split_pre_post', allow_early_resolve=True):
        t0 = ob * T_TILE
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
        post[t0:t0 + T_TILE, 0:HC_MULT] = pl.slice(post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT])
    for ob in pl.spmd(t_dim // COMB_T_TILE, name_hint='comb_sinkhorn', allow_early_resolve=True):
        t0 = ob * COMB_T_TILE
        inv_col_t = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g1 = pl.load(mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g2 = pl.load(mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g3 = pl.load(mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
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
        row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
        pl.store(row3_out, [t0, 3 * HC_MULT], comb)
    for blk in pl.spmd(t_dim // T_TILE * (D // D_SPMD), name_hint='mix_x', allow_early_resolve=True):
        t0 = blk // (D // D_SPMD) * T_TILE
        d_base = blk % (D // D_SPMD) * D_SPMD
        pre_tile_t = pl.transpose(pre_val_store[t0:t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_CHUNK, stage=2):
            d0 = d_base + db * D_CHUNK
            x0 = x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_CHUNK]
            x1 = x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_CHUNK]
            x2 = x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_CHUNK]
            x3 = x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_CHUNK]
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            x_mixed[t0:t0 + T_TILE, d0:d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode='rint')
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
    if HC_PRE_IMPL == 'separate':

        @pl.jit.inline
        def hc_pre(x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_scale: pl.Tensor[[3], pl.FP32], hc_base: pl.Tensor[[MIX_HC], pl.FP32], x_mixed: pl.Tensor[[T_DYN, D], pl.BF16], post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32], comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]):
            _hc_pre_separate(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
            return x_mixed
    else:

        @pl.jit.inline
        def hc_pre(x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_scale: pl.Tensor[[3], pl.FP32], hc_base: pl.Tensor[[MIX_HC], pl.FP32], x_mixed: pl.Tensor[[T_DYN, D], pl.BF16], post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32], comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]):
            _hc_pre_syncall(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
            return x_mixed
    return hc_pre
hc_pre = _bind_hc_pre()
