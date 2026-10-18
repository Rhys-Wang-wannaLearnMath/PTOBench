import pypto.language as pl

import pypto.language.distributed as pld

from config import CASE_CONFIG as M, EP_WORLD_SIZE, MOE_TOKENS, RECV_MAX

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

N_EXPERTS_GLOBAL = M.n_routed_experts

N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS

N_ROUTES = T * TOPK

META_PAD = (N_LOCAL + 7) // 8 * 8

MAX_PER_SRC = T

AUX_PAD = 8

AUX_SCALE = 0

AUX_W = 1

IDX_PAD = 8

@pl.jit.inline
def dispatch(indices: pl.Tensor[[T, TOPK], pl.INT32], x_norm_i8: pl.Tensor[[T, D], pl.INT8], x_norm_scale: pl.Tensor[[T, 1], pl.FP32], weights: pl.Tensor[[T, TOPK], pl.FP32], recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8], recv_scale_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32], recv_w_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32], recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32], recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32], recv_meta: pld.DistributedTensor[[N_RANKS, META_PAD], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32], recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], num_tokens: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], moe_epoch: pl.Scalar[pl.INT32]):
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='dispatch'):
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for d in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                cursor[d * N_LOCAL + e] = 0
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
                row = loc_e * RECV_MAX + my_rank * MAX_PER_SRC + slot
                pld.tensor.put(dst=recv_x, peer=dst, src=x_norm_i8, dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, D])
                pl.tile.write(aux_tile, [0, AUX_SCALE], pl.read(x_norm_scale, [t, 0]))
                pl.tile.write(aux_tile, [0, AUX_W], pl.read(weights, [t, k]))
                pld.tile.remote_store(aux_tile, target=recv_aux, peer=dst, offsets=[row, 0])
                pl.tile.write(route_tile, [0, 0], pl.cast(t * TOPK + k, pl.INT32))
                pld.tile.remote_store(route_tile, target=recv_route, peer=dst, offsets=[row, 0])
        meta_tile = pl.tile.full([1, META_PAD], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            pld.tile.remote_store(meta_tile, target=recv_meta, peer=dst, offsets=[my_rank, 0])
            if dst != my_rank:
                pld.system.notify(target=arrived, peer=dst, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(signal=arrived, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)
        base = pl.array.create(N_LOCAL * N_RANKS, pl.INT32)
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                base[e * N_RANKS + src] = acc
                acc = acc + pl.read(recv_meta, [src, e])
            pl.write(recv_count_out, [e, 0], acc)
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

@pl.jit.inline
def combine(recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16], recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32], sh: pl.Tensor[[T, D], pl.BF16], ffn_out: pl.Tensor[[T, D], pl.BF16], recv_meta: pld.DistributedTensor[[N_RANKS, META_PAD], pl.INT32], routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16], combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], num_tokens: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], moe_epoch: pl.Scalar[pl.INT32]):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='combine'):
        cmb_base = pl.array.create(N_LOCAL * N_RANKS, pl.INT32)
        for e in pl.range(N_LOCAL):
            base_acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                cmb_base[e * N_RANKS + src] = base_acc
                base_acc = base_acc + pl.read(recv_meta, [src, e])
        for src in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                n = pl.cast(pl.read(recv_meta, [src, e]), pl.INDEX)
                b = pl.cast(cmb_base[e * N_RANKS + src], pl.INDEX)
                for slot in pl.range(n):
                    out_col = b + slot
                    r_route = pl.cast(pl.read(recv_r_route_out, [e, out_col]), pl.INDEX)
                    pld.tensor.put(dst=routed_y_buf, peer=src, src=recv_y_flat, dst_offsets=[r_route, 0], src_offsets=[e * RECV_MAX + out_col, 0], shape=[1, D])
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(target=combine_arrived, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(signal=combine_arrived, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    for t in pl.spmd(T, name_hint='shared_routed'):
        if t < active_tokens:
            acc = pl.cast(sh[t:t + 1, :], target_type=pl.FP32)
            for k in pl.range(TOPK):
                r = t * TOPK + k
                acc = pl.add(acc, pl.cast(routed_y_buf[r:r + 1, :], target_type=pl.FP32))
            ffn_out[t:t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode='rint')
        else:
            ffn_out[t:t + 1, :] = sh[t:t + 1, :]

@pl.jit.inline(auto_scope=False)
def moe(x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16], hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[3], pl.FP32], hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[T], pl.INT64], routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]], recv_meta: pld.DistributedTensor[[N_RANKS, META_PAD], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32], recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16], combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], layer_id: pl.Scalar[pl.INT32], num_tokens: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], moe_epoch: pl.Scalar[pl.INT32]) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed, post_ffn, comb_ffn)
    x_norm = pl.create_tensor([T, D], dtype=pl.BF16)
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(x_mixed, norm_w, gate_w, gate_bias, layer_id, num_tokens, tid2eid, input_ids, x_norm, x_norm_i8, x_norm_scale, indices, weights)
    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(x_norm_i8, x_norm_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, sh)
    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    dispatch(indices, x_norm_i8, x_norm_scale, weights, recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out, recv_meta, recv_x, recv_aux, recv_route, arrived, num_tokens, my_rank, moe_epoch)
    with pl.scope():
        recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
        expert_routed(recv_x_out, recv_scale_out, recv_w_out, recv_count_out, routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale, recv_y)
        ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
        combine(recv_y, recv_r_route_out, sh, ffn_out, recv_meta, routed_y_buf, combine_arrived, num_tokens, my_rank, moe_epoch)
        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next

@pl.jit
def moe_test(x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16], hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[3], pl.FP32], hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[T], pl.INT64], routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]], recv_meta: pld.DistributedTensor[[N_RANKS, META_PAD], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32], recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16], combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], layer_id: pl.Scalar[pl.INT32], num_tokens: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], moe_epoch: pl.Scalar[pl.INT32]) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    moe(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias, tid2eid, input_ids, routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, x_next, recv_meta, recv_x, recv_aux, recv_route, arrived, routed_y_buf, combine_arrived, layer_id, num_tokens, my_rank, moe_epoch)
    return x_next

@pl.jit.host
def l3_moe(x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16], hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32], hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32], norm_w: pl.Tensor[[N_RANKS, D], pl.BF16], gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[N_RANKS, T], pl.INT64], routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32], x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16]], layer_id: pl.Scalar[pl.INT32], num_tokens: pl.Scalar[pl.INT32]):
    recv_meta_buf = pld.alloc_window_buffer(N_RANKS * META_PAD * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_aux_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * AUX_PAD * 4)
    recv_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)
    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, META_PAD], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        moe_test(x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r], norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r], routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r], routed_w2[r], routed_w2_scale[r], shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r], shared_w2[r], shared_w2_scale[r], x_next[r], recv_meta, recv_x, recv_aux, recv_route, arrived, routed_y_buf, combine_arrived, layer_id, num_tokens, r, pl.const(1, pl.INT32), device=r)
