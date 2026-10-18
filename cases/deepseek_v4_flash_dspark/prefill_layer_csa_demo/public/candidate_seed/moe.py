# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num

"""REF closure for deepseek_v4_flash_dspark.prefill_layer_csa_demo; fixed bindings from public config."""
from config import D
from config import HC_DIM
from config import HC_MULT
from config import MAX_PER_SRC
from config import MIX_HC
from config import MOE_INTER
from config import N_EXPERTS_GLOBAL
from config import N_LOCAL
from config import N_RANKS
from config import N_ROUTES
from config import RECV_MAX
from config import T
from config import TOPK
from config import TP_SIZE
from config import VOCAB
from expert_routed import expert_routed
from expert_shared import expert_shared
from gate import gate
from hc_post import hc_post
from hc_pre import hc_pre
from prefill_cp_token_allgather import CP_KV_T_DYN as PREFILL_GROUP_T_DYN
from prefill_cp_token_allgather import CP_Q_T_DYN as PREFILL_LOCAL_T_DYN
from prefill_cp_token_allgather import PREFILL_GROUP_CAP
from prefill_cp_token_allgather import prefill_cp_token_allgather_step
import pypto.language as pl
import pypto.language.distributed as pld

META_N_LOCAL = ((N_LOCAL + 7) // 8) * 8
AUX_PAD = 8
AUX_SCALE = 0
AUX_W = 1
IDX_PAD = 8
PREFILL_INPUT_ID_TILE = 4

@pl.jit.inline
def clear_prefill_moe_signals(
    stage_token: pl.Tensor[[1], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear retained prefill-MoE epochs after the final wave completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_signal_clear") as clear_tid:
        completed_epoch = pl.read(stage_token, [0])
        if completed_epoch >= 0:
            zero = pl.cast(0, pl.INT32)
            for src in pl.range(N_RANKS):
                pl.write(arrived, [src, 0], zero)
                pl.write(data_arrived, [src, 0], zero)
                pl.write(combine_arrived, [src, 0], zero)
                pl.write(stage_done, [src, 0], zero)
    return clear_tid


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
    recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32],
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
        meta_tile = pl.tile.full([1, META_N_LOCAL], dtype=pl.INT32, value=0)
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
    with pl.spmd(N_LOCAL, name_hint="dispatch_push", allow_early_resolve=True):
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

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait") as _wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
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
) -> pl.Scalar[pl.TASK_ID]:
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL * RECV_MAX, D])
    # One SPMD block per LOCAL EXPERT: block e pushes expert e's compact rows back to
    # their origin rank (= the source lane src they arrived on) at their route offset.
    # Rows are src-major, so the per-(e, src) base is a loop-carried prefix sum over
    # src inside the block (same shape as dispatch_gather). Each route maps to a
    # unique (dst, loc_e) and r_route, so the blocks' puts are write-disjoint.
    with pl.spmd(N_LOCAL, name_hint="combine"):
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

        # Each local-expert scatter block publishes one completion per peer.
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_arrived,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_wait") as _cwait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=combine_arrived,
                    offsets=[src, 0],
                    expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
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
    return _reduce_tid


@pl.jit.inline(auto_scope=False)
def _moe_tile(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
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
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one fixed-capacity gate, expert dispatch, and combine tile."""
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, num_tokens, tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    shared_out = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        shared_out,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32, manual_dep=True)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_meta_local = pl.create_tensor([N_RANKS, META_N_LOCAL], dtype=pl.INT32, manual_dep=True)
    dispatch(
        indices, x_norm_i8, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out, recv_meta_local,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        num_tokens, my_rank, moe_epoch,
    )

    recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
    expert_routed(
        recv_x_out, recv_scale_out, recv_w_out, recv_count_out,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )

    completion_tid = combine(
        recv_y, recv_r_route_out, shared_out,
        ffn_out, recv_meta_local,
        routed_y_buf, combine_arrived,
        num_tokens, my_rank, moe_epoch,
    )
    return completion_tid


@pl.jit.inline
def _complete_prefill_moe_wave(
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_token: pl.Tensor[[1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    completion_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Publish one globally complete prefill-MoE wave before window reuse."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_wave_notify", deps=[completion_tid]) as notify_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=stage_done, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_wave_wait") as wait_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.defer_wait(
                    signal=stage_done, offsets=[peer, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_moe_wave_publish",
        deps=[completion_tid, notify_tid, wait_tid],
    ) as barrier_tid:
        pl.write(stage_token, [0], moe_epoch)
    return barrier_tid


@pl.jit.inline(auto_scope=False)
def _prefill_moe_wave(
    x_mixed_full: pl.Tensor[[PREFILL_GROUP_T_DYN, D], pl.BF16],
    input_ids: pl.Tensor[[PREFILL_LOCAL_T_DYN], pl.INT64],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
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
    ffn_out: pl.Tensor[[PREFILL_LOCAL_T_DYN, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_token: pl.Tensor[[1], pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    wave_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Stage and execute one fixed-capacity prefill-MoE wave."""
    local_rows = pl.tensor.dim(ffn_out, 0)
    local_wave_base = wave_id * T
    full_wave_base = tp_rank * local_rows + local_wave_base

    x_mixed_wave = pl.create_tensor([T, D], dtype=pl.BF16)
    for token in pl.spmd(T, name_hint="prefill_moe_hidden_stage"):
        completed_epoch = pl.read(stage_token, [0])
        if completed_epoch >= 0:
            full_token = full_wave_base + token
            x_mixed_wave[token : token + 1, :] = x_mixed_full[full_token : full_token + 1, :]

    input_id_count = pl.tensor.dim(input_ids, 0)
    input_ids_rows = pl.reshape(input_ids, [input_id_count, 1])
    input_ids_wave_rows = pl.create_tensor([T, 1], dtype=pl.INT64)
    for token_block in pl.spmd(T // PREFILL_INPUT_ID_TILE, name_hint="prefill_moe_ids_stage"):
        token0 = token_block * PREFILL_INPUT_ID_TILE
        local_token = local_wave_base + token0
        input_ids_wave_rows[token0 : token0 + PREFILL_INPUT_ID_TILE, 0:1] = input_ids_rows[local_token : local_token + PREFILL_INPUT_ID_TILE, 0:1]
    input_ids_wave = pl.reshape(input_ids_wave_rows, [T])

    ffn_wave = pl.create_tensor([T, D], dtype=pl.BF16)
    completion_tid = _moe_tile(
        x_mixed_wave,
        norm_w, gate_w, gate_bias, tid2eid, input_ids_wave,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        ffn_wave,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        layer_id, pl.const(T, pl.INT32), my_rank, moe_epoch,
    )
    barrier_tid = _complete_prefill_moe_wave(stage_done, stage_token, my_rank, moe_epoch, completion_tid)
    ffn_out[local_wave_base : local_wave_base + T, :] = ffn_wave
    return barrier_tid


@pl.jit.inline(auto_scope=False)
def prefill_moe(
    attn_out: pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[PREFILL_LOCAL_T_DYN], pl.INT64],
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
    x_hc: pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT, D], pl.FP32],
    x_mixed: pl.Tensor[[PREFILL_GROUP_T_DYN, D], pl.BF16],
    post_ffn: pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT], pl.FP32],
    comb_ffn: pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_out: pl.Tensor[[PREFILL_LOCAL_T_DYN, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_token: pl.Tensor[[1], pl.INT32],
    layer_completion: pl.Tensor[[1], pl.INT32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT, D], pl.FP32]:
    """Run one CP-aware prefill MoE layer over a rank-local token shard."""
    with pl.scope():
        hc_pre(attn_out, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed, post_ffn, comb_ffn)

    num_waves = pl.tensor.dim(ffn_out, 0) // T
    num_waves_i32 = pl.cast(num_waves, pl.INT32)
    for wave in pl.range(num_waves):
        wave_i32 = pl.cast(wave, pl.INT32)
        moe_epoch = layer_id * num_waves_i32 + wave_i32 + pl.const(1, pl.INT32)
        with pl.scope():
            _prefill_moe_wave(
                x_mixed, input_ids,
                norm_w, gate_w, gate_bias, tid2eid,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                ffn_out,
                recv_meta, recv_x, recv_aux, recv_route,
                arrived, data_arrived, routed_y_buf, combine_arrived,
                stage_done, stage_token,
                tp_rank, layer_id, wave_i32, my_rank, moe_epoch,
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_layer_complete"):
        completed_epoch = pl.read(stage_token, [0])
        if completed_epoch >= 0:
            pl.write(layer_completion, [0], layer_id + pl.const(1, pl.INT32))

    with pl.scope():
        group_rows = pl.tensor.dim(attn_out, 0)
        ffn_out_full = pl.create_tensor([group_rows, D], dtype=pl.BF16)
        ffn_out_full, _gather_signal = prefill_cp_token_allgather_step(
            ffn_out, ffn_out_full,
            gather_window, gather_signal,
            group_base, tp_rank,
        )
        hc_post(ffn_out_full, attn_out, post_ffn, comb_ffn, x_hc)
    return x_hc


