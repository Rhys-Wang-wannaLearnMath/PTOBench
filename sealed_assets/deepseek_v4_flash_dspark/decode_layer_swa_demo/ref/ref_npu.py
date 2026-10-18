"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_swa_demo."""

from config import N_LOCAL
from config import TP_SIZE
from config import N_EXPERTS_GLOBAL
from config import N_RANKS
import decode_swa as swa
import moe as moe_module
import pypto.language as pl
import pypto.language.distributed as pld
from decode_swa import decode_swa
from moe import clear_moe_signals
from moe import moe

MOE_TOKENS = moe_module.T
T_DYN = swa.T_DYN
ORI_BLOCK_NUM_DYN = swa.ORI_BLOCK_NUM_DYN
D = swa.D
H = swa.H
HEAD_DIM = swa.HEAD_DIM
ROPE_HEAD_DIM = swa.ROPE_HEAD_DIM
Q_LORA = swa.Q_LORA
WIN = swa.WIN
HC_MULT = swa.HC_MULT
MIX_HC = swa.MIX_HC
HC_DIM = swa.HC_DIM
O_LORA = swa.O_LORA
O_GROUP_IN = swa.O_GROUP_IN
LOCAL_O_GROUPS = swa.LOCAL_O_GROUPS
LOCAL_O_WIDTH = swa.LOCAL_O_WIDTH
BLOCK_SIZE = swa.BLOCK_SIZE
ATTENTION_WINDOW_ROWS = swa.ATTENTION_WINDOW_ROWS
O_WINDOW_ROWS = swa.O_WINDOW_ROWS
META_N_LOCAL = moe_module.META_N_LOCAL
MOE_INTER = moe_module.MOE_INTER
VOCAB = moe_module.VOCAB
TOPK = moe_module.TOPK
RECV_MAX = moe_module.RECV_MAX
AUX_PAD = moe_module.AUX_PAD
IDX_PAD = moe_module.IDX_PAD
N_ROUTES = moe_module.N_ROUTES

@pl.jit.inline(auto_scope=False)
def decode_layer_swa(x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], swa_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], swa_lens: pl.Tensor[[T_DYN], pl.INT32], position_ids: pl.Tensor[[T_DYN], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[3], pl.FP32], hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64], routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]], x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]], x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]], attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16], attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32], o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32], recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16], combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], layer_id: pl.Scalar[pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], moe_epoch: pl.Scalar[pl.INT32]):
    """Run one SWA attention and one MoE without clearing shared signals."""
    active_t = pl.tensor.dim(x_hc, 0)
    with pl.scope():
        decode_swa(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids, attn_sink, wo_a, wo_b, wo_b_scale, x_attn_active, attention_window, attention_signal, o_window, o_signal, group_base, tp_rank, local_t)
    with pl.scope():
        x_attn_moe = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        for token in pl.spmd(MOE_TOKENS, name_hint='decode_layer_attn_pack'):
            if token < active_t:
                x_attn_moe[token:token + 1, 0:HC_MULT, 0:D] = x_attn_active[token:token + 1, 0:HC_MULT, 0:D]
            else:
                x_attn_moe[token:token + 1, 0:HC_MULT, 0:D] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
        moe(x_attn_moe, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias, tid2eid, input_ids, routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, x_moe_next, recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived, layer_id, local_t, my_rank, moe_epoch)
        for token in pl.spmd(MOE_TOKENS, name_hint='decode_layer_active_trim'):
            if token < active_t:
                x_next[token:token + 1, 0:HC_MULT, 0:D] = x_moe_next[token:token + 1, 0:HC_MULT, 0:D]
    return x_next

@pl.jit(auto_scope=False)
def decode_layer_swa_test(x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], swa_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], swa_lens: pl.Tensor[[T_DYN], pl.INT32], position_ids: pl.Tensor[[T_DYN], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[3], pl.FP32], hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64], routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]], x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]], x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]], attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16], attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32], o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32], recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16], combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], layer_id: pl.Scalar[pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], moe_epoch: pl.Scalar[pl.INT32]):
    """Standalone child: run one layer and clear its fresh MoE signals."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(0, T_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    x_next.bind_dynamic(0, T_DYN)
    decode_layer_swa(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids, attn_sink, wo_a, wo_b, wo_b_scale, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias, tid2eid, input_ids, routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, x_attn_active, x_moe_next, x_next, attention_window, attention_signal, o_window, o_signal, recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived, layer_id, group_base, tp_rank, local_t, my_rank, moe_epoch)
    clear_moe_signals(x_moe_next, arrived, data_arrived, combine_arrived)
    return x_next

@pl.jit.host
def l3_decode_layer_swa(x_hc: pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32], hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16], wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], swa_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64], swa_indices: pl.Tensor[[N_RANKS, T_DYN, WIN], pl.INT32], swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32], position_ids: pl.Tensor[[N_RANKS, T_DYN], pl.INT32], attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32], wo_a: pl.Tensor[[N_RANKS, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[N_RANKS, D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32], hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32], hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32], norm_w: pl.Tensor[[N_RANKS, D], pl.BF16], gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[N_RANKS, MOE_TOKENS], pl.INT64], routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32], x_attn_active: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]], x_moe_next: pl.Out[pl.Tensor[[N_RANKS, MOE_TOKENS, HC_MULT, D], pl.FP32]], x_next: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]], layer_id: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32]):
    """Launch one standalone SWA+MoE layer child on every EP rank."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(1, T_DYN)
    swa_indices.bind_dynamic(1, T_DYN)
    swa_lens.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    x_attn_active.bind_dynamic(1, T_DYN)
    x_next.bind_dynamic(1, T_DYN)
    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, META_N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, META_N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        decode_layer_swa_test(x_hc[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank], kv_cache[rank], swa_slot_mapping[rank], swa_indices[rank], swa_lens[rank], position_ids[rank], attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank], input_ids[rank], routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank], routed_w2[rank], routed_w2_scale[rank], shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank], shared_w2[rank], shared_w2_scale[rank], x_attn_active[rank], x_moe_next[rank], x_next[rank], attention_window, attention_signal, o_window, o_signal, recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived, layer_id, group_base, tp_rank, local_t, rank, pl.const(1, pl.INT32), device=rank)
    return x_next
