# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from config import (
    AUX_PAD,
    AUX_SCALE,
    AUX_W,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MAX_PER_SRC,
    MIX_HC,
    MOE_INTER,
    MOE_TOKENS,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    TOPK,
    VOCAB,
)

import pypto.language as pl

import pypto.language.distributed as pld

from hc_pre import hc_pre

from hc_post import hc_post

from gate import gate

from expert_shared import expert_shared

from expert_routed import expert_routed

T = MOE_TOKENS

@pl.jit.inline
def clear_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's MoE signal windows after its final MoE completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear"):
        # The final MoE output depends on this rank observing every peer's final
        # meta, payload, and combine notify. No peer can issue another MoE notify
        # to this rank in the current forward after this dependency is satisfied.
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(arrived, [src, 0], zero)
            pl.write(data_arrived, [src, 0], zero)
            pl.write(combine_arrived, [src, 0], zero)

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
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    # windows
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id; `arrived`/`data_arrived` are monotonic so waits use `>= moe_epoch`.
    moe_epoch: pl.Scalar[pl.INT32],
):
    # Flat 2-D view kept outside the scope so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])

    # Meta and payload arrivals ride two independent windows (`arrived` /
    # `data_arrived`), so the two phases barrier separately and overlap freely.

    # Count routes, publish counts, barrier on meta, cumsum -> recv_count_out.
    # Needs every source's counts but none of the bulk payload.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta",
        allow_early_resolve=True,
    ) as _meta_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        # Count how many routes land in each (dst, loc_e) lane (no payload move).
        cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for d in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                cursor[d * N_LOCAL + e] = 0
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        # One meta row per dst (all N_LOCAL counts, zeros included), then bump the
        # per-source arrival counter. AtomicAdd on a monotonic window is
        # order-independent, so a late notify from an earlier epoch cannot clobber it.
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

        # Wait for every source's meta flag.
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=arrived,
                    offsets=[src, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        # Cumsum recv_meta over sources -> per-expert receive count. The host reads
        # recv_count_out to size the routed-expert tile loop, so producing it here
        # lets routed matmuls submit while the payload is still moving.
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Move the bulk payload (x / aux / route) to each destination lane.
    # Split over LOCAL EXPERT INDEX (N_LOCAL blocks): block loc_e handles expert
    # loc_e on EVERY destination rank, so the blocking cross-rank puts fan out
    # across N_LOCAL cores. One slot counter per destination rank; token-major
    # order matches the meta pass's per-(dst, loc_e) cumulative count.
    with pl.spmd(N_LOCAL, name_hint="dispatch_push", allow_early_resolve=True) as _push_tid:
        loc_e = pl.tile.get_block_idx()
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        slot_ctr = pl.array.create(N_RANKS, pl.INT32)
        for d in pl.range(N_RANKS):
            slot_ctr[d] = 0
        e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC

        # Pad tiles zeroed once; used cols overwritten per push, then remote_store.
        aux_tile = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
        route_tile = pl.tile.full([1, IDX_PAD], dtype=pl.INT32, value=0)
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL
                le = eid - dst * N_LOCAL
                if le == loc_e:
                    slot = slot_ctr[dst]
                    slot_ctr[dst] = slot + 1
                    # lane (loc_e, my_rank, slot) on peer=dst
                    row = e_lane_base + slot
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

        # Payload-arrival notify folded into the push: each block signals every peer
        # after its own puts, so a peer sees N_LOCAL notifies per source per epoch
        # and the wait below expects N_LOCAL * moe_epoch. Saves the launch of a
        # separate post-push notify task. Each block bumps the count only after its
        # own puts issue in program order, which is what gates the gather -- recv_aux
        # / recv_route ride a non-draining remote_store and a PIPE_ALL barrier is not
        # a cross-rank DDR fence (PTOAS#872).
        for dst in pl.range(N_RANKS):
            if dst != my_rank:
                pld.system.notify(
                    target=data_arrived,
                    peer=dst,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # Wait only (the notify rides inside dispatch_push). The indices read is an
    # anchor, not data: it deps this task on gate's route tasks, so the wait starts
    # with dispatch_push instead of trailing dispatch_meta's spin. Anchor it to
    # something -- an unanchored wait is dispatched immediately and spins holding a
    # core group, so pipelined layers stack up spinners.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_wait",
        deps=[_push_tid],
        allow_early_resolve=True,
    ) as _wait_tid:
        _idx_anchor = pl.read(indices, [0, 0])
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=data_arrived,
                    offsets=[src, 0],
                    expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Gather lanes into the compact per-expert buffers: one SPMD block per local
    # expert. deps on _wait_tid for the incoming payload; this rank's own
    # dst == my_rank puts are already ordered by the local RAW edges on
    # recv_x / recv_aux / recv_route. deps on _meta_tid for recv_meta_local, which is
    # manual_dep and so has no auto edge from the cumsum.
    with pl.spmd(
        N_LOCAL,
        name_hint="dispatch_gather",
        deps=[_wait_tid, _meta_tid],
        # Keep the routed expert tasks off the cores until this gather retires.
        allow_early_resolve=False,
    ) as _gather_tid:
        e = pl.tile.get_block_idx()
        e_base_row = e * RECV_MAX
        b = pl.cast(0, pl.INDEX)
        for src in pl.range(N_RANKS):
            n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
            src_base_row = e_base_row + src * MAX_PER_SRC
            for slot in pl.range(n):
                in_row = src_base_row + slot
                out_col = b + slot
                out_row = e_base_row + out_col
                recv_x_out_flat[out_row : out_row + 1, :] = recv_x[in_row : in_row + 1, :]
                pl.write(recv_scale_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_SCALE]))
                pl.write(recv_w_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_W]))
                pl.write(recv_r_route_out, [e, out_col], pl.read(recv_route, [in_row, 0]))
            b = b + n

    return _push_tid

@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    dispatch_push_tid: pl.Scalar[pl.TASK_ID],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL * RECV_MAX, D])
    # One SPMD block per LOCAL EXPERT: block e pushes expert e's compact rows back to
    # their origin rank (= the source lane src they arrived on) at their route offset.
    # Rows are src-major, so the per-(e, src) base is a loop-carried prefix sum over
    # src inside the block (same shape as dispatch_gather). Each route maps to a
    # unique (dst, loc_e) and r_route, so the blocks' puts are write-disjoint.
    # Keep the scatter TaskId so the arrival handshake cannot occupy AIV cores
    # until every routed_y_buf put from this rank has completed.
    with pl.spmd(N_LOCAL, name_hint="combine") as _combine_tid:
        e = pl.tile.get_block_idx()
        e_base_row = e * RECV_MAX
        b = pl.cast(0, pl.INDEX)
        for src in pl.range(N_RANKS):
            n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
            for slot in pl.range(n):
                out_col = b + slot
                r_route = pl.cast(pl.read(recv_r_route_out, [e, out_col]), pl.INDEX)
                pld.tensor.put(
                    dst=routed_y_buf,
                    peer=src,
                    src=recv_y_flat,
                    dst_offsets=[r_route, 0],
                    src_offsets=[e_base_row + out_col, 0],
                    shape=[1, D],
                )
            b = b + n

    # Publish only after the complete scatter grid. Keeping this cross-rank wait
    # out of the speculative early-dispatch chain prevents it and shared_routed
    # from reserving the AIV cores needed by the scatter itself.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="combine_wait",
        deps=[_combine_tid, dispatch_push_tid],
    ) as _cwait_tid:
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
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

    # ffn_out[t] = sh[t] + Sigma_k routed_y_buf[t*TOPK+k]. deps on combine_wait for the
    # peers' writes; this rank's own puts ride the local RAW edge on routed_y_buf,
    # which is the only thing ordering them now that the wait is off the scatter.
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    with pl.spmd(
        T,
        name_hint="shared_routed",
        deps=[_cwait_tid],
    ) as _reduce_tid:
        t = pl.tile.get_block_idx()
        if t < active_tokens:
            acc = pl.cast(sh[t:t + 1, :], target_type=pl.FP32)
            for k in pl.range(TOPK):
                r = t * TOPK + k
                acc = pl.add(acc, pl.cast(routed_y_buf[r:r + 1, :], target_type=pl.FP32))
            ffn_out[t:t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        else:
            ffn_out[t:t + 1, :] = sh[t:t + 1, :]

@pl.jit.inline(auto_scope=False)
def moe(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    # windows
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id for the shared flag windows (distinct from layer_id).
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    # Non-output intermediates allocate locally, in their producer's scope.
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32, manual_dep=True)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn, comb_ffn,
    )

    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, num_tokens, tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32, manual_dep=True)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_meta_local = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
    dispatch_push_tid = dispatch(
        indices, x_norm_i8, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out, recv_meta_local,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
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
            ffn_out, recv_meta_local,
            routed_y_buf, combine_arrived,
            num_tokens, my_rank, moe_epoch, dispatch_push_tid,
        )

        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next

