from config import D, HC_DIM, HC_MULT, MIX_HC, MOE_INTER, N_EXPERTS_GLOBAL, N_LOCAL, N_RANKS, N_ROUTES, TOPK, VOCAB

import pypto.language as pl

import pypto.language.distributed as pld

from config import MOE_TOKENS, RECV_MAX

from hc_pre_v2 import hc_pre

from hc_post_v2 import hc_post

from gate import gate

from expert_shared import expert_shared

from expert_routed import expert_routed

T = MOE_TOKENS

MAX_PER_SRC = T

AUX_PAD = 8  # FP32 pack tile width (32 B min tile); cols: 0=scale 1=weight

AUX_SCALE = 0

AUX_W = 1

IDX_PAD = 8  # INT32 route tile width; route rides a separate window from scale/w
META_PAD = ((N_LOCAL + 7) // 8) * 8  # INT32 metadata tile width (32-byte aligned)

@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8],
    recv_scale_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_w_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, META_PAD], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])

    # Publish route counts first.  The bulk payload uses a separate readiness
    # counter because remote_store is asynchronous with respect to notify.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta") as meta_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for d in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                cursor[d * N_LOCAL + e] = 0

        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                bucket = dst * N_LOCAL + loc_e
                cursor[bucket] = cursor[bucket] + 1

        meta_tile = pl.tile.full([1, META_PAD], dtype=pl.INT32, value=0)
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

        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=arrived,
                    offsets=[src, 0],
                    expected=pl.cast(moe_epoch * 2 - 1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                acc = acc + pl.read(recv_meta, [src, e])
            pl.write(recv_count_out, [e, 0], acc)

    # One producer per destination/local-expert lane.  It fills a complete
    # auxiliary tile before the remote store and notifies only afterwards.
    with pl.spmd(N_RANKS * N_LOCAL, name_hint="dispatch_push", deps=[meta_tid]) as push_tid:
        block = pl.tile.get_block_idx()
        dst = block // N_LOCAL
        loc_e = block - dst * N_LOCAL
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC
        aux_lane = pl.tile.full([MAX_PER_SRC, AUX_PAD], dtype=pl.FP32, value=0.0)
        route_lane = pl.tile.full([MAX_PER_SRC, IDX_PAD], dtype=pl.INT32, value=0)
        slot_ctr = pl.array.create(1, pl.INT32)
        slot_ctr[0] = 0
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                d = eid // N_LOCAL
                le = eid - d * N_LOCAL
                if d == dst and le == loc_e:
                    slot = slot_ctr[0]
                    slot_ctr[0] = slot + 1
                    row = lane_base + slot
                    pld.tensor.put(dst=recv_x, peer=dst, src=x_norm_i8,
                                   dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, D])
                    pl.tile.write(aux_lane, [slot, AUX_SCALE], pl.read(x_norm_scale, [t, 0]))
                    pl.tile.write(aux_lane, [slot, AUX_W], pl.read(weights, [t, k]))
                    pl.tile.write(route_lane, [slot, 0], pl.cast(t * TOPK + k, pl.INT32))
        pld.tile.remote_store(aux_lane, target=recv_aux, peer=dst, offsets=[lane_base, 0])
        pld.tile.remote_store(route_lane, target=recv_route, peer=dst, offsets=[lane_base, 0])
        if dst != my_rank:
            pld.system.notify(target=data_arrived, peer=dst, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_payload_wait", deps=[push_tid]) as payload_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(signal=data_arrived, offsets=[src, 0],
                                expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
                                cmp=pld.WaitCmp.Ge)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_gather", deps=[payload_tid, meta_tid]) as gather_tid:
        for e in pl.range(N_LOCAL):
            e_base_row = e * RECV_MAX
            for src in pl.range(N_RANKS):
                n = pl.cast(pl.read(recv_meta, [src, e]), pl.INDEX)
                b = pl.const(0, pl.INDEX)
                for prior_src in pl.range(src):
                    b = b + pl.cast(pl.read(recv_meta, [prior_src, e]), pl.INDEX)
                for slot in pl.range(n):
                    in_row = e_base_row + src * MAX_PER_SRC + slot
                    out_col = b + slot
                    out_row = e_base_row + out_col
                    recv_x_out_flat[out_row:out_row + 1, :] = recv_x[in_row:in_row + 1, :]
                    pl.write(recv_scale_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_SCALE]))
                    pl.write(recv_w_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_W]))
                    pl.write(recv_r_route_out, [e, out_col], pl.read(recv_route, [in_row, 0]))
    return gather_tid

@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, META_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    expert_completion_tid: pl.Scalar[pl.TASK_ID],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL * RECV_MAX, D])
    with pl.spmd(N_LOCAL, name_hint="combine_scatter", deps=[expert_completion_tid]) as scatter_tid:
        e = pl.tile.get_block_idx()
        for src in pl.range(N_RANKS):
            n = pl.cast(pl.read(recv_meta, [src, e]), pl.INDEX)
            b = pl.const(0, pl.INDEX)
            for prior_src in pl.range(src):
                b = b + pl.cast(pl.read(recv_meta, [prior_src, e]), pl.INDEX)
            for slot in pl.range(n):
                out_col = b + slot
                r_route = pl.cast(pl.read(recv_r_route_out, [e, out_col]), pl.INDEX)
                pld.tensor.put(
                    dst=routed_y_buf, peer=src, src=recv_y_flat,
                    dst_offsets=[r_route, 0],
                    src_offsets=[e * RECV_MAX + out_col, 0],
                    shape=[1, D],
                )
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_arrived, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_wait", deps=[scatter_tid]) as wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=combine_arrived, offsets=[src, 0],
                    expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    for t in pl.spmd(T, name_hint="shared_routed", deps=[wait_tid]):
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
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[N_RANKS, META_PAD], pl.INT32],
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
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
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
    dispatch_completion_tid = dispatch(
        indices, x_norm_i8, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        num_tokens, my_rank, moe_epoch,
    )

    with pl.scope():
        recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
        expert_completion_tid = expert_routed(
            recv_x_out, recv_scale_out, recv_w_out, recv_count_out,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            recv_y, dispatch_completion_tid,
        )

        ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
        combine(
            recv_y, recv_r_route_out, sh,
            ffn_out, recv_meta,
            routed_y_buf, combine_arrived,
            num_tokens, my_rank, moe_epoch, expert_completion_tid,
        )

        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next
