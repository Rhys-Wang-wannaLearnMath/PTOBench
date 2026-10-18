import pypto.language as pl

import pypto.language.distributed as pld

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX

from hc_pre import hc_pre

from hc_post import hc_post

from gate import gate

from expert_shared import expert_shared

from expert_routed import expert_routed

from dispatch import dispatch_ep

from combine import combine_ep

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

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

W_PAD = 8

IDX_PAD = 8

@pl.jit.inline
def moe_ep(x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16], hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[3], pl.FP32], hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[T], pl.INT64], routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]], pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32], count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32], recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32], recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16], combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], layer_id: pl.Scalar[pl.INT32], num_tokens: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32]) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
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
    dispatch_ep(indices, x_norm_i8, x_norm_scale, weights, recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out, pub_counts, count_done, data_done, recv_x, recv_scale, recv_w, recv_r_route, num_tokens, my_rank)
    recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
    expert_routed(recv_x_out, recv_scale_out, recv_w_out, recv_count_out, routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale, recv_y)
    ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    combine_ep(recv_y, recv_r_route_out, sh, ffn_out, pub_counts, routed_y_buf, combine_done, num_tokens, my_rank)
    x_next = hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    # Retire this protocol's own count matrix and one remote completion credit.
    # Each rank owns its replicated matrix; combine_done has acknowledged all
    # current producers before the complete x_next dependency reaches this task.
    retire_output_view = pl.slice(pl.reshape(x_next, [T, HC_MULT * D]), [1, 32], [0, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='moe_ep_signal_retire'):
        _x_next_anchor = pl.read(retire_output_view, [0, 0])
        for signal_peer in pl.range(N_RANKS):
            if signal_peer != my_rank:
                pld.system.notify(target=count_done, peer=my_rank, offsets=[signal_peer, 0], value=pl.cast(-1, pl.INT32), op=pld.NotifyOp.AtomicAdd)
                pld.system.notify(target=data_done, peer=my_rank, offsets=[signal_peer, 0], value=pl.cast(-1, pl.INT32), op=pld.NotifyOp.AtomicAdd)
                pld.system.notify(target=combine_done, peer=my_rank, offsets=[signal_peer, 0], value=pl.cast(-1, pl.INT32), op=pld.NotifyOp.AtomicAdd)
        for count_row in pl.range(N_RANKS * N_RANKS):
            for count_expert in pl.range(N_LOCAL):
                credit = pl.read(pub_counts, [count_row, count_expert])
                pld.system.notify(target=pub_counts, peer=my_rank, offsets=[count_row, count_expert], value=pl.cast(0 - credit, pl.INT32), op=pld.NotifyOp.AtomicAdd)
    return x_next

@pl.jit
def moe_ep_test(x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16], hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[3], pl.FP32], hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[T], pl.INT64], routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]], pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32], count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32], recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32], recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16], combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], layer_id: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32]) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    x_next = moe_ep(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias, tid2eid, input_ids, routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, x_next, pub_counts, count_done, data_done, recv_x, recv_scale, recv_w, recv_r_route, routed_y_buf, combine_done, layer_id, pl.const(T, pl.INT32), my_rank)
    return x_next

@pl.jit.host
def l3_moe_ep(x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16], hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32], hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32], norm_w: pl.Tensor[[N_RANKS, D], pl.BF16], gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[N_RANKS, T], pl.INT64], routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32], x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16]], layer_id: pl.Scalar[pl.INT32]):
    pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * N_LOCAL * 4)
    count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    data_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_scale_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_w_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_r_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    for r in pl.range(pld.world_size()):
        pub_counts = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
        count_done = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_done = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_w = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_r_route = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_done = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        moe_ep_test(x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r], norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r], routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r], routed_w2[r], routed_w2_scale[r], shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r], shared_w2[r], shared_w2_scale[r], x_next[r], pub_counts, count_done, data_done, recv_x, recv_scale, recv_w, recv_r_route, routed_y_buf, combine_done, layer_id, r, device=r)
