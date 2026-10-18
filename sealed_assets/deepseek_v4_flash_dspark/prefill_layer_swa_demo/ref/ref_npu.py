# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: EP2/TP2 representative single-layer golden
# ci: no-sim    # CI marker: distributed communication oracle requires real devices

"""REF closure for deepseek_v4_flash_dspark.prefill_layer_swa_demo; fixed bindings from public config."""
from config import BLOCK_SIZE
from config import CSA_CMP_BLOCK_NUM
from config import CSA_ORI_BLOCK_NUM
from config import CSA_STATE_BLOCK_NUM
from config import CSA_STATE_BLOCK_SIZE
from config import CSA_STATE_MAX_BLOCKS
from config import D
from config import H
from config import HCA_CMP_BLOCK_NUM
from config import HCA_STATE_BLOCK_NUM
from config import HCA_STATE_BLOCK_SIZE
from config import HCA_STATE_MAX_BLOCKS
from config import HC_DIM
from config import HC_MULT
from config import HEAD_DIM
from config import IDX_HEAD_DIM
from config import IDX_N_HEADS
from config import INNER_COMPRESS_STATE_DIM
from config import INNER_OUT_DIM
from config import INNER_STATE_BLOCK_NUM
from config import INNER_STATE_BLOCK_SIZE
from config import INNER_STATE_MAX_BLOCKS
from config import MAX_SEQ_LEN
from config import MIX_HC
from config import MOE_INTER
from config import N_EXPERTS_GLOBAL
from config import N_LOCAL
from config import N_RANKS
from config import N_ROUTES
from config import O_GROUP_IN
from config import O_LORA
from config import O_PROJ_LOCAL_COLS
from config import O_PROJ_LOCAL_GROUPS
from config import Q_LORA
from config import ROPE_HEAD_DIM
from config import SPARSE_CMP_MAX_BLOCKS
from config import SPARSE_ORI_MAX_BLOCKS
from config import TOPK
from config import TP_SIZE
from config import VOCAB
from moe import AUX_PAD
from moe import IDX_PAD
from moe import META_N_LOCAL
from moe import RECV_MAX
from moe import T
from moe import clear_prefill_moe_signals
from moe import prefill_moe
from prefill_cp_token_allgather import PREFILL_GROUP_CAP
from prefill_csa import IDX_CACHE_BLOCK_NUM
from prefill_csa import IDX_CACHE_MAX_BLOCKS
from prefill_csa import MAIN_COMPRESS_STATE_DIM as CSA_COMPRESS_STATE_DIM
from prefill_csa import MAIN_OUT_DIM as CSA_MAIN_OUT_DIM
from prefill_csa import prefill_attention_csa_cp
from prefill_fwd import FWD_GROUP_TOKENS_DYN
from prefill_fwd import FWD_TOKENS_DYN
from prefill_hca import MAIN_COMPRESS_STATE_DIM as HCA_COMPRESS_STATE_DIM
from prefill_hca import MAIN_OUT_DIM as HCA_MAIN_OUT_DIM
from prefill_hca import prefill_attention_hca_cp
from prefill_o_proj import O_PROJ_SCRATCH_COLS
from prefill_o_proj import O_PROJ_SCRATCH_D
from prefill_o_proj import O_PROJ_SCRATCH_GROUPS
from prefill_o_proj import O_PROJ_SCRATCH_INPUT
from prefill_o_proj import O_PROJ_SCRATCH_RANK
from prefill_o_proj import O_PROJ_WO_A_WINDOW_COLS
from prefill_o_proj import O_PROJ_WO_A_WINDOW_ROWS
from prefill_o_proj import O_PROJ_WO_B_WINDOW_COLS
from prefill_o_proj import O_PROJ_WO_B_WINDOW_ROWS
from prefill_swa import prefill_attention_swa_cp
import pypto.language as pl
import pypto.language.distributed as pld


@pl.jit(auto_scope=False)
def prefill_layer_attention(
    x_hc: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[128, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[4, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[4, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32]],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_cmp_kv: pl.InOut[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_stage: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
):
    """Run one selected DSA-CP attention kind."""
    x_hc.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    ori_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    position_ids_local.bind_dynamic(0, FWD_TOKENS_DYN)
    position_ids_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    attn_stage.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    swa_profile = freqs_cos[0:1, 0:MAX_SEQ_LEN, 0:ROPE_HEAD_DIM]
    swa_cos = pl.reshape(swa_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    swa_profile = freqs_sin[0:1, 0:MAX_SEQ_LEN, 0:ROPE_HEAD_DIM]
    swa_sin = pl.reshape(swa_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    compressed_profile = freqs_cos[1:2, 0:MAX_SEQ_LEN, 0:ROPE_HEAD_DIM]
    compressed_cos = pl.reshape(compressed_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    compressed_profile = freqs_sin[1:2, 0:MAX_SEQ_LEN, 0:ROPE_HEAD_DIM]
    compressed_sin = pl.reshape(compressed_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    wo_a_full = pl.create_tensor([O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], dtype=pl.BF16)
    wo_b_full = pl.create_tensor([O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], dtype=pl.INT8)
    o_proj_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_layer_o_proj_order_init"):
        pl.write(o_proj_order_fence, [0], pl.cast(0, pl.INT32))

    with pl.scope():
        if layer_id == 0:
            kv_cache, attn_stage, gather_signal = prefill_attention_swa_cp(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w,
                wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                swa_cos, swa_sin,
                kv_cache, ori_block_table, ori_slot_mapping_full,
                position_ids_local, position_ids_full,
                attn_sink,
                wo_a, wo_b, wo_b_scale,
                wo_a_full, wo_b_full,
                attn_stage,
                gather_window, gather_signal,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                o_proj_order_fence,
                group_base, tp_rank, pl.const(1, pl.INT32),
            )
        elif layer_id == 2:
            attn_stage, gather_signal = prefill_attention_csa_cp(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w,
                wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                compressed_cos, compressed_sin,
                csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape,
                csa_cmp_norm_w,
                csa_compress_state, csa_compress_state_block_table,
                csa_hadamard_idx, csa_idx_wq_b, csa_idx_wq_b_scale,
                csa_weights_proj,
                csa_inner_wkv, csa_inner_wgate, csa_inner_ape,
                csa_inner_norm_w,
                csa_inner_compress_state, csa_inner_compress_state_block_table,
                kv_cache, ori_block_table, ori_slot_mapping_full,
                csa_cmp_kv, csa_cmp_block_table,
                idx_kv_cache, idx_kv_scale, idx_block_table,
                position_ids_local, position_ids_full,
                csa_cmp_slot_mapping_full, csa_idx_slot_mapping_full, csa_state_slot_mapping_full,
                csa_inner_state_slot_mapping_full,
                attn_sink,
                wo_a, wo_b, wo_b_scale,
                wo_a_full, wo_b_full,
                attn_stage,
                gather_window, gather_signal,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                o_proj_order_fence,
                group_base, tp_rank, pl.const(1, pl.INT32),
            )
        else:
            attn_stage, gather_signal = prefill_attention_hca_cp(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w,
                wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                compressed_cos, compressed_sin,
                hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape,
                hca_cmp_norm_w,
                hca_compress_state, hca_compress_state_block_table,
                kv_cache, ori_slot_mapping_full, ori_block_table,
                hca_cmp_kv, hca_cmp_block_table,
                position_ids_local, position_ids_full,
                hca_cmp_slot_mapping_full, hca_state_slot_mapping_full,
                attn_sink,
                wo_a, wo_b, wo_b_scale,
                wo_a_full, wo_b_full,
                attn_stage,
                gather_window, gather_signal,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                o_proj_order_fence,
                group_base, tp_rank, pl.const(1, pl.INT32),
            )

    return attn_stage


@pl.jit(auto_scope=False)
def prefill_layer_moe(
    attn_stage: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
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
    x_mixed: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16]],
    post_ffn: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32]],
    comb_ffn: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32]],
    ffn_out: pl.Out[pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, META_N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    epoch_init_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the production prefill MoE path for one oracle layer."""
    attn_stage.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    x_mixed.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    post_ffn.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    comb_ffn.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    input_ids.bind_dynamic(0, FWD_TOKENS_DYN)
    ffn_out.bind_dynamic(0, FWD_TOKENS_DYN)
    x_next.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)

    stage_token = pl.create_tensor([1], dtype=pl.INT32)
    layer_completion = pl.create_tensor([1], dtype=pl.INT32)
    layer_i32 = pl.cast(layer_id, pl.INT32)
    num_waves = pl.tensor.dim(ffn_out, 0) // T
    epoch_base = layer_i32 * pl.cast(num_waves, pl.INT32)
    expert_epoch_base = epoch_base * pl.const(N_LOCAL, pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_layer_moe_epoch_seed") as seed_tid:
        for src in pl.range(N_RANKS):
            pl.write(arrived, [src, 0], epoch_base)
            pl.write(data_arrived, [src, 0], expert_epoch_base)
            pl.write(combine_arrived, [src, 0], expert_epoch_base)
            pl.write(stage_done, [src, 0], epoch_base)
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=epoch_init_done, peer=peer,
                    offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_layer_moe_epoch_wait") as wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=epoch_init_done, offsets=[src, 0],
                    expected=pl.cast(1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_layer_moe_order_init", deps=[seed_tid, wait_tid]):
        pl.write(stage_token, [0], epoch_base)
        pl.write(layer_completion, [0], layer_i32)

    prefill_moe(
        attn_stage,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next, x_mixed, post_ffn, comb_ffn, ffn_out,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        stage_done, stage_token, layer_completion,
        gather_window, gather_signal,
        group_base, tp_rank, layer_i32, my_rank,
    )
    clear_prefill_moe_signals(stage_token, arrived, data_arrived, combine_arrived, stage_done)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_layer_epoch_signal_clear"):
        # Clear the two fixed-threshold epoch signals the MoE clear does not own.
        # Reading the final MoE output orders this after every peer's last notify,
        # which the wave-barrier epoch alone does not establish.
        _completion_anchor = pl.read(x_next, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(epoch_init_done, [src, 0], zero)
        for source_tp in pl.range(TP_SIZE):
            pl.write(gather_signal, [source_tp, 0], zero)
    return x_next


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, 128, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[N_RANKS, HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, 4, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[N_RANKS, CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, 4, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32]],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[N_RANKS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    attn_stage: pl.InOut[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, D], pl.BF16]],
    post_ffn: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32]],
    comb_ffn: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32]],
    ffn_out: pl.Out[pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
):
    """Run one DSA-CP layer across all ranks."""
    x_hc.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, FWD_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    input_ids.bind_dynamic(1, FWD_TOKENS_DYN)
    attn_stage.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    x_mixed.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    post_ffn.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    comb_ffn.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    ffn_out.bind_dynamic(1, FWD_TOKENS_DYN)
    x_next.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, META_N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    stage_done_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    epoch_init_done_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_wo_a_window_buf = pld.alloc_window_buffer([O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16)
    o_proj_wo_b_window_buf = pld.alloc_window_buffer([O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8)
    o_proj_weight_ready_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_weight_consumed_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_wo_a_window = pld.window(
            o_proj_wo_a_window_buf, [O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16,
        )
        o_proj_wo_b_window = pld.window(
            o_proj_wo_b_window_buf, [O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8,
        )
        o_proj_weight_ready = pld.window(o_proj_weight_ready_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_weight_consumed = pld.window(o_proj_weight_consumed_buf, [TP_SIZE, 1], dtype=pl.INT32)
        group_base = rank // TP_SIZE * TP_SIZE
        tp_rank = rank % TP_SIZE
        prefill_layer_attention(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank],
            hca_cmp_norm_w[rank],
            hca_compress_state[rank], hca_compress_state_block_table[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank],
            csa_cmp_norm_w[rank],
            csa_compress_state[rank], csa_compress_state_block_table[rank],
            csa_hadamard_idx[rank], csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank],
            csa_weights_proj[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank],
            csa_inner_norm_w[rank],
            csa_inner_compress_state[rank], csa_inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping_full[rank],
            hca_cmp_kv[rank], csa_cmp_kv[rank],
            hca_cmp_block_table[rank], csa_cmp_block_table[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            position_ids_local[rank], position_ids_full[rank],
            hca_cmp_slot_mapping_full[rank], hca_state_slot_mapping_full[rank],
            csa_cmp_slot_mapping_full[rank], csa_idx_slot_mapping_full[rank], csa_state_slot_mapping_full[rank],
            csa_inner_state_slot_mapping_full[rank],
            attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            attn_stage[rank],
            gather_window, gather_signal,
            o_proj_wo_a_window, o_proj_wo_b_window,
            o_proj_weight_ready, o_proj_weight_consumed,
            group_base, tp_rank,
            layer_id,
            device=rank,
        )

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, META_N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        stage_done = pld.window(stage_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        epoch_init_done = pld.window(epoch_init_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        group_base = rank // TP_SIZE * TP_SIZE
        tp_rank = rank % TP_SIZE
        prefill_layer_moe(
            attn_stage[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank],
            gate_w[rank], gate_bias[rank],
            tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_mixed[rank], post_ffn[rank], comb_ffn[rank], ffn_out[rank], x_next[rank],
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            stage_done, epoch_init_done,
            gather_window, gather_signal,
            group_base, tp_rank,
            layer_id, rank,
            device=rank,
        )


