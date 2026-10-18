"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_csa_demo."""

from config import N_RANKS
from config import N_LOCAL
from config import TP_SIZE
from config import N_EXPERTS_GLOBAL
from decode_csa import O_LORA as O_LORA
from decode_csa import LOCAL_O_GROUPS as LOCAL_O_GROUPS
from decode_csa import HEAD_DIM as HEAD_DIM
from decode_csa import HC_MULT as HC_MULT
from decode_csa import ORI_BLOCK_NUM_DYN as ORI_BLOCK_NUM_DYN
from decode_csa import ROPE_HEAD_DIM as ROPE_HEAD_DIM
from decode_csa import D as D
from decode_csa import HC_DIM as HC_DIM
from decode_csa import WIN as WIN
from decode_csa import MIX_HC as MIX_HC
from decode_csa import H as H
from decode_csa import T_DYN as T_DYN
from decode_csa import ATTENTION_WINDOW_ROWS as ATTENTION_WINDOW_ROWS
from decode_csa import O_WINDOW_ROWS as O_WINDOW_ROWS
from decode_csa import LOCAL_O_WIDTH as LOCAL_O_WIDTH
from decode_csa import Q_LORA as Q_LORA
from decode_csa import BLOCK_SIZE as BLOCK_SIZE
from decode_csa import O_GROUP_IN as O_GROUP_IN
import decode_csa as csa
import moe as moe_module
import pypto.language as pl
import pypto.language.distributed as pld
from decode_csa import decode_csa
from moe import clear_moe_signals
from moe import moe

MOE_TOKENS = moe_module.T
CSA_B_DYN = csa.B_DYN
CSA_CMP_BLOCK_NUM_DYN = csa.CMP_BLOCK_NUM_DYN
CSA_IDX_CACHE_BLOCK_NUM_DYN = csa.IDX_CACHE_BLOCK_NUM_DYN
CSA_MAIN_STATE_BLOCK_NUM_DYN = csa.MAIN_STATE_BLOCK_NUM_DYN
CSA_INNER_STATE_BLOCK_NUM_DYN = csa.INNER_STATE_BLOCK_NUM_DYN
CSA_IDX_N_HEADS = csa.IDX_N_HEADS
CSA_IDX_HEAD_DIM = csa.IDX_HEAD_DIM
CSA_COMPRESS_RATIO = csa.COMPRESS_RATIO
CSA_MAIN_OUT_DIM = csa.MAIN_OUT_DIM
CSA_MAIN_STATE_BLOCK_SIZE = csa.MAIN_STATE_BLOCK_SIZE
CSA_MAIN_STATE_MAX_BLOCKS = csa.MAIN_STATE_MAX_BLOCKS
CSA_MAIN_STATE_DIM = csa.MAIN_STATE_DIM
CSA_INNER_OUT_DIM = csa.INNER_OUT_DIM
CSA_INNER_STATE_BLOCK_SIZE = csa.INNER_STATE_BLOCK_SIZE
CSA_INNER_STATE_MAX_BLOCKS = csa.INNER_STATE_MAX_BLOCKS
CSA_INNER_STATE_DIM = csa.INNER_STATE_DIM
CSA_CMP_MAX_BLOCKS = csa.CMP_MAX_BLOCKS
CSA_IDX_MAX_BLOCKS = csa.IDX_MAX_BLOCKS
META_N_LOCAL = moe_module.META_N_LOCAL
MOE_INTER = moe_module.MOE_INTER
VOCAB = moe_module.VOCAB
TOPK = moe_module.TOPK
RECV_MAX = moe_module.RECV_MAX
AUX_PAD = moe_module.AUX_PAD
IDX_PAD = moe_module.IDX_PAD
N_ROUTES = moe_module.N_ROUTES

@pl.jit.inline(auto_scope=False)
def decode_layer_csa(x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.InOut[pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[CSA_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32], idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16], hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16], inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor[[CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[CSA_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32], ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], window_swa_lens: pl.Tensor[[T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[CSA_B_DYN], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[3], pl.FP32], hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64], routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]], x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]], x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]], attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16], attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32], o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32], recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16], combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], layer_id: pl.Scalar[pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], moe_epoch: pl.Scalar[pl.INT32]):
    """Run one CSA attention and one MoE without clearing shared signals."""
    active_t = pl.tensor.dim(x_hc, 0)
    with pl.scope():
        decode_csa(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx, inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_compress_state, inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, ori_slot_mapping, window_swa_indices, window_swa_lens, cmp_slot_mapping, idx_slot_mapping, state_slot_mapping, inner_state_slot_mapping, position_ids, kv_seq_lens, attn_sink, wo_a, wo_b, wo_b_scale, x_attn_active, attention_window, attention_signal, o_window, o_signal, group_base, tp_rank, local_t)
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
def decode_layer_csa_test(x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.InOut[pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[CSA_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32], idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16], hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16], inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor[[CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[CSA_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32], ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], window_swa_lens: pl.Tensor[[T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[CSA_B_DYN], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[3], pl.FP32], hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32], norm_w: pl.Tensor[[D], pl.BF16], gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64], routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]], x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]], x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]], attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16], attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32], o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8], recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32], recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32], arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16], combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], layer_id: pl.Scalar[pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], moe_epoch: pl.Scalar[pl.INT32]):
    """Run one standalone CSA+MoE layer and clear its MoE signals."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, CSA_MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, CSA_B_DYN)
    inner_compress_state.bind_dynamic(0, CSA_INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(0, CSA_B_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CSA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, CSA_B_DYN)
    idx_kv_cache.bind_dynamic(0, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(0, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(0, CSA_B_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    idx_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, CSA_B_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    x_next.bind_dynamic(0, T_DYN)
    decode_layer_csa(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx, inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_compress_state, inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, ori_slot_mapping, window_swa_indices, window_swa_lens, cmp_slot_mapping, idx_slot_mapping, state_slot_mapping, inner_state_slot_mapping, position_ids, kv_seq_lens, attn_sink, wo_a, wo_b, wo_b_scale, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias, tid2eid, input_ids, routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, x_attn_active, x_moe_next, x_next, attention_window, attention_signal, o_window, o_signal, recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived, layer_id, group_base, tp_rank, local_t, my_rank, moe_epoch)
    clear_moe_signals(x_moe_next, arrived, data_arrived, combine_arrived)
    return x_next

@pl.jit.host
def l3_decode_layer_csa(x_hc: pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32], hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16], wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_cos: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_sin: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16], compress_state: pl.InOut[pl.Tensor[[N_RANKS, CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32], idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[N_RANKS, D, CSA_IDX_N_HEADS], pl.BF16], hadamard_idx: pl.Tensor[[N_RANKS, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16], inner_wkv: pl.Tensor[[N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[N_RANKS, CSA_IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32], ori_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64], window_swa_indices: pl.Tensor[[N_RANKS, T_DYN, WIN], pl.INT32], window_swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64], idx_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64], position_ids: pl.Tensor[[N_RANKS, T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[N_RANKS, CSA_B_DYN], pl.INT32], attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32], wo_a: pl.Tensor[[N_RANKS, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[N_RANKS, D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32], hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32], hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32], hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32], norm_w: pl.Tensor[[N_RANKS, D], pl.BF16], gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32], gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32], tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32], input_ids: pl.Tensor[[N_RANKS, MOE_TOKENS], pl.INT64], routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32], shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32], x_attn_active: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]], x_moe_next: pl.Out[pl.Tensor[[N_RANKS, MOE_TOKENS, HC_MULT, D], pl.FP32]], x_next: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]], layer_id: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32]):
    """Launch one complete CSA+MoE child per EP-world rank."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    cmp_freqs_cos.bind_dynamic(1, T_DYN)
    cmp_freqs_sin.bind_dynamic(1, T_DYN)
    compress_state.bind_dynamic(1, CSA_MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(1, CSA_B_DYN)
    inner_compress_state.bind_dynamic(1, CSA_INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(1, CSA_B_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(1, CSA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, CSA_B_DYN)
    idx_kv_cache.bind_dynamic(1, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(1, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(1, CSA_B_DYN)
    ori_slot_mapping.bind_dynamic(1, T_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    window_swa_lens.bind_dynamic(1, T_DYN)
    cmp_slot_mapping.bind_dynamic(1, T_DYN)
    idx_slot_mapping.bind_dynamic(1, T_DYN)
    state_slot_mapping.bind_dynamic(1, T_DYN)
    inner_state_slot_mapping.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    kv_seq_lens.bind_dynamic(1, CSA_B_DYN)
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
        decode_layer_csa_test(x_hc[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank], cmp_freqs_cos[rank], cmp_freqs_sin[rank], cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank], compress_state[rank], compress_state_block_table[rank], idx_wq_b[rank], idx_wq_b_scale[rank], weights_proj[rank], hadamard_idx[rank], inner_wkv[rank], inner_wgate[rank], inner_ape[rank], inner_norm_w[rank], inner_compress_state[rank], inner_compress_state_block_table[rank], kv_cache[rank], cmp_kv[rank], cmp_block_table[rank], idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank], ori_slot_mapping[rank], window_swa_indices[rank], window_swa_lens[rank], cmp_slot_mapping[rank], idx_slot_mapping[rank], state_slot_mapping[rank], inner_state_slot_mapping[rank], position_ids[rank], kv_seq_lens[rank], attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank], input_ids[rank], routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank], routed_w2[rank], routed_w2_scale[rank], shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank], shared_w2[rank], shared_w2_scale[rank], x_attn_active[rank], x_moe_next[rank], x_next[rank], attention_window, attention_signal, o_window, o_signal, recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived, layer_id, group_base, tp_rank, local_t, rank, pl.const(1, pl.INT32), device=rank)
    return x_next
