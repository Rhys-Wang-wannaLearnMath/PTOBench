# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 MoE single-layer (decode), FLASH preset. --ep picks the EP world
size: 2/4/8 run N-rank distributed; each rank keeps 32 experts."""


import pypto.language as pl
import pypto.language.distributed as pld

from config import (CASE_CONFIG as M, EP_WORLD_SIZE, MOE_TOKENS, RECV_MAX,
                    N_EXPERTS_GLOBAL as CONFIG_N_EXPERTS_GLOBAL, N_LOCAL as CONFIG_N_LOCAL)
from hc_pre_v2 import hc_pre
from hc_post_v2 import hc_post
from gate import gate
from expert_shared import expert_shared
from expert_routed import expert_routed


T = MOE_TOKENS
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size

HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MOE_INTER = M.moe_intermediate_size

N_RANKS = EP_WORLD_SIZE
N_EXPERTS_GLOBAL = CONFIG_N_EXPERTS_GLOBAL
N_LOCAL = CONFIG_N_LOCAL
N_ROUTES = T * TOPK

# recv_x/recv_aux laid out [expert, source, slot], flattened to
# [N_LOCAL * RECV_MAX, D]. Lane (e, src, slot) flat row = e * RECV_MAX +
# src * MAX_PER_SRC + slot. One source sends <= T rows to a local expert.
MAX_PER_SRC = T
AUX_PAD = 8  # FP32 pack tile width (32 B min tile); cols: 0=scale 1=weight
AUX_SCALE = 0
AUX_W = 1
IDX_PAD = 8  # INT32 route tile width; route rides a separate window from scale/w
             # (an FP32 tile can't hold it: INDEX->FP32 casts are unsupported).

assert N_RANKS in (2, 4, 8), f"--ep must be one of (2, 4, 8) (got {N_RANKS})"
assert N_EXPERTS_GLOBAL == N_RANKS * N_LOCAL
assert RECV_MAX == N_RANKS * MAX_PER_SRC


# === Dispatch ================================================================
# Lane push, count publish, arrival wait, and cumsum gather run in one
# pl.at(CORE_GROUP) so program order stays push -> notify -> wait -> gather.
@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    # compact per-expert outputs consumed by expert_routed / combine
    recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8],
    recv_scale_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_w_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    # windows
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id; `arrived` is monotonic so waits use `>= moe_epoch`.
    moe_epoch: pl.Scalar[pl.INT32],
):
    # Flat 2-D view kept outside the scope so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])

    # Upstream consumption fence, adapted to one scatter credit per rank.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait") as reuse_tid:
        indices_anchor = pl.read(indices, [0, 0])
        if moe_epoch > 1:
            pld.system.wait(signal=combine_arrived, offsets=[my_rank, 0], expected=pl.cast(moe_epoch - 1, pl.INT32), cmp=pld.WaitCmp.Ge)
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(signal=combine_arrived, offsets=[src, 0], expected=pl.cast(2 * (moe_epoch - 1), pl.INT32), cmp=pld.WaitCmp.Ge)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch", deps=[reuse_tid]):
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        # Push each route to lane (loc_e, my_rank, cursor) on peer=dst.
        cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for d in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                cursor[d * N_LOCAL + e] = 0

        # Pad tiles zeroed once; used cols overwritten per push, then remote_store.
        aux_tile = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
        route_tile = pl.tile.full([1, IDX_PAD], dtype=pl.INT32, value=0)
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                bucket = dst * N_LOCAL + loc_e
                slot = cursor[bucket]
                cursor[bucket] = slot + 1
                # lane (loc_e, my_rank, slot) on peer=dst
                row = loc_e * RECV_MAX + my_rank * MAX_PER_SRC + slot
                pld.tensor.put(
                    dst=recv_x,
                    peer=dst,
                    src=x_norm_i8,
                    dst_offsets=[row, 0],
                    src_offsets=[t, 0],
                    shape=[1, D],
                )
                pl.tile.write(aux_tile, [0, AUX_SCALE], pl.read(x_norm_scale, [t, 0]))
                pl.tile.write(aux_tile, [0, AUX_W], pl.read(weights, [t, k]))
                pld.tile.remote_store(aux_tile, target=recv_aux, peer=dst, offsets=[row, 0])
                pl.tile.write(route_tile, [0, 0], pl.cast(t * TOPK + k, pl.INT32))
                pld.tile.remote_store(route_tile, target=recv_route, peer=dst, offsets=[row, 0])

        # One meta row per dst (all N_LOCAL counts, zeros included), then bump the
        # per-source arrival counter. AtomicAdd(1) is order-independent across the
        # reused window, so a late notify from an earlier epoch cannot clobber it.
        meta_tile = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            pld.tile.remote_store(meta_tile, target=recv_meta, peer=dst, offsets=[my_rank, 0])
            if dst != my_rank:
                pld.system.notify(
                    target=arrived,
                    peer=dst,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        # Wait for every source's flag.
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=arrived,
                    offsets=[src, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        # Cumsum recv_meta over sources -> base row of each (e, src) lane block.
        base = pl.array.create(N_LOCAL * N_RANKS, pl.INT32)
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                base[e * N_RANKS + src] = acc
                acc = acc + pl.read(recv_meta, [src, e])
            pl.write(recv_count_out, [e, 0], acc)

        # Gather lanes into the compact per-expert buffers.
        for e in pl.range(N_LOCAL):
            e_base_row = e * RECV_MAX
            for src in pl.range(N_RANKS):
                n = pl.cast(pl.read(recv_meta, [src, e]), pl.INDEX)
                b = pl.cast(base[e * N_RANKS + src], pl.INDEX)
                for slot in pl.range(n):
                    in_row = e_base_row + src * MAX_PER_SRC + slot
                    out_col = b + slot
                    out_row = e_base_row + out_col
                    recv_x_out_flat[out_row:out_row + 1, :] = recv_x[in_row:in_row + 1, :]
                    pl.write(recv_scale_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_SCALE]))
                    pl.write(recv_w_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_W]))
                    pl.write(recv_r_route_out, [e, out_col], pl.read(recv_route, [in_row, 0]))


# === Combine =================================================================
# Push recv_y rows back to their origin rank keyed by r_route, barrier, then a
# dense reduce ffn_out[t] = sh[t] + Sigma_k routed_y_buf[t*TOPK+k].
@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine"):
        # Rebuild per-(e, src) base rows from recv_meta so the compact slots are
        # walked src-major and the origin rank is the loop index src (static peer,
        # no per-slot origin tensor).
        cmb_base = pl.array.create(N_LOCAL * N_RANKS, pl.INT32)
        for e in pl.range(N_LOCAL):
            base_acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                cmb_base[e * N_RANKS + src] = base_acc
                base_acc = base_acc + pl.read(recv_meta, [src, e])

        # Put each compact slot back to its origin rank (= src) at its route offset.
        for src in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                n = pl.cast(pl.read(recv_meta, [src, e]), pl.INDEX)
                b = pl.cast(cmb_base[e * N_RANKS + src], pl.INDEX)
                for slot in pl.range(n):
                    out_col = b + slot
                    r_route = pl.cast(pl.read(recv_r_route_out, [e, out_col]), pl.INDEX)
                    pld.tensor.put(
                        dst=routed_y_buf,
                        peer=src,
                        src=recv_y_flat,
                        dst_offsets=[r_route, 0],
                        src_offsets=[e * RECV_MAX + out_col, 0],
                        shape=[1, D],
                    )

        # Signal, then wait for every source (AtomicAdd counter, epoch-safe).
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_arrived,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=combine_arrived,
                    offsets=[src, 0],
                    expected=pl.cast(2 * moe_epoch - 1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # ffn_out[t] = sh[t] + Sigma_k routed_y_buf[t*TOPK+k].
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    with pl.spmd(T, name_hint="shared_routed") as reduce_tid:
        t = pl.tile.get_block_idx()
        if t < active_tokens:
            acc = pl.cast(sh[t:t + 1, :], target_type=pl.FP32)
            for k in pl.range(TOPK):
                r = t * TOPK + k
                acc = pl.add(acc, pl.cast(routed_y_buf[r:r + 1, :], target_type=pl.FP32))
            ffn_out[t:t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        else:
            ffn_out[t:t + 1, :] = sh[t:t + 1, :]


    # Whole-grid join ensures every routed result is consumed before reuse.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_consumed", deps=[reduce_tid]):
        for peer in pl.range(N_RANKS):
            pld.system.notify(target=combine_arrived, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)


@pl.jit.inline(auto_scope=False)
def moe(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # final output
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    # windows
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id for the shared flag windows (distinct from layer_id).
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    # Non-output intermediates allocate locally, in their producer's scope.
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn, comb_ffn,
    )

    x_norm = pl.create_tensor([T, D], dtype=pl.BF16)
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, num_tokens, tid2eid, input_ids,
        x_norm, x_norm_i8, x_norm_scale, indices, weights,
    )

    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    dispatch(
        indices, x_norm_i8, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out,
        recv_meta, recv_x, recv_aux, recv_route, arrived, combine_arrived,
        num_tokens, my_rank, moe_epoch,
    )

    with pl.scope():
        recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
        expert_routed(
            recv_x_out, recv_scale_out, recv_w_out, recv_count_out,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            recv_y,
        )

        ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
        combine(
            recv_y, recv_r_route_out, sh,
            ffn_out, recv_meta,
            routed_y_buf, combine_arrived,
            num_tokens, my_rank, moe_epoch,
        )

        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next


