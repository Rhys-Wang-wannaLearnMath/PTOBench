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
from config import BLOCK_SIZE
from config import D
from config import H
from config import HC_DIM
from config import HC_MULT
from config import HEAD_DIM
from config import IDX_TOPK
from config import KV_CMP_BLOCK_NUM as CMP_BLOCK_NUM
from config import KV_ORI_BLOCK_NUM as BLOCK_NUM
from config import MAX_SEQ_LEN
from config import MIX_HC
from config import O_GROUPS
from config import O_GROUP_IN
from config import O_LORA
from config import O_PROJ_LOCAL_COLS
from config import O_PROJ_LOCAL_GROUPS
from config import Q_LORA
from config import ROPE_HEAD_DIM
from config import SPARSE_CMP_MAX_BLOCKS
from config import TP_SIZE
from config import WIN
from hc_post import hc_post
from hc_pre import hc_pre
from prefill_cp_token_allgather import PREFILL_GROUP_CAP
from prefill_cp_token_allgather import prefill_cp_token_allgather_step
from prefill_o_proj import O_PROJ_SCRATCH_COLS
from prefill_o_proj import O_PROJ_SCRATCH_D
from prefill_o_proj import O_PROJ_SCRATCH_GROUPS
from prefill_o_proj import O_PROJ_SCRATCH_INPUT
from prefill_o_proj import O_PROJ_SCRATCH_RANK
from prefill_o_proj import O_PROJ_WO_A_WINDOW_COLS
from prefill_o_proj import O_PROJ_WO_A_WINDOW_ROWS
from prefill_o_proj import O_PROJ_WO_B_WINDOW_COLS
from prefill_o_proj import O_PROJ_WO_B_WINDOW_ROWS
from prefill_o_proj import gather_o_proj_full_weights
from prefill_sparse_attn import PREFILL_ATTN_TILE
from prefill_sparse_attn import SPARSE_BIAS_COLS
from prefill_sparse_attn import VALID_BLOCK_MASK_COLS
from prefill_sparse_attn import sparse_attn_physical
from qkv_proj_rope import kv_proj_rope
from qkv_proj_rope import materialize_rope_rows_dynamic
from qkv_proj_rope import q_proj_rope
from qkv_proj_rope import rope_prepare
from rmsnorm import rms_norm
import pypto.language as pl
import pypto.language.distributed as pld

CP_Q_T_DYN = pl.dynamic("PREFILL_SWA_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("PREFILL_SWA_CP_KV_T_DYN")
BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")

@pl.jit.inline
def prefill_attention_swa_cp_core(
    x_normed_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    x_normed_full: pl.Tensor[[CP_KV_T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[BLOCK_NUM], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
    o_proj_weight_dep: pl.Scalar[pl.TASK_ID],
):
    """SWA attention body: local queries/output projection and replicated full KV."""
    q_dim = pl.tensor.dim(x_normed_local, 0)
    kv_dim = pl.tensor.dim(x_normed_full, 0)

    rope_cos_local = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_local = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows_dynamic(freqs_cos, freqs_sin, position_ids_local, rope_cos_local, rope_sin_local)
    q_cos_il = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    q_sin_signed = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    q_swap_idx = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_local, rope_sin_local, q_cos_il, q_sin_signed, q_swap_idx)

    q = pl.create_tensor([q_dim, H, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([q_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([q_dim, 1], dtype=pl.FP32)
    q_proj_rope(x_normed_local, wq_a, wq_b, wq_b_scale, gamma_cq, q_cos_il, q_sin_signed, q_swap_idx, q, qr, qr_scale)

    rope_cos_full = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_full = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows_dynamic(freqs_cos, freqs_sin, position_ids_full, rope_cos_full, rope_sin_full)
    kv_cos_il = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    kv_sin_signed = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    kv_swap_idx = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_full, rope_sin_full, kv_cos_il, kv_sin_signed, kv_swap_idx)

    kv_full = pl.create_tensor([kv_dim, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_normed_full, wkv, gamma_ckv, kv_cos_il, kv_sin_signed, kv_swap_idx, kv_full, late_dep)

    block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cp_cache_write"):
        for write_t in pl.range(kv_dim):
            write_row_raw = pl.read(ori_slot_mapping_full, [write_t])
            if write_row_raw >= 0:
                write_row = pl.cast(write_row_raw, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, :] = kv_full[write_t : write_t + 1, :]

    swa_indices = pl.create_tensor([q_dim, WIN], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([q_dim, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cp_window_indices"):
        for idx_t in pl.range(q_dim):
            idx_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            mask_row = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
            abs_pos = pl.read(position_ids_local, [idx_t])
            window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for win_col in pl.range(WIN):
                win_col_i32 = pl.cast(win_col, pl.INT32)
                if win_col_i32 < window_valid:
                    key_abs = key_start_abs + win_col_i32
                    blk_slot = key_abs // BLOCK_SIZE
                    blk = pl.read(block_table, [pl.cast(blk_slot, pl.INDEX)])
                    if blk >= 0:
                        block_row = key_abs - blk_slot * BLOCK_SIZE
                        row = pl.cast(blk * BLOCK_SIZE + block_row, pl.INT32)
                        pl.write(idx_row, [0, win_col], row)
                        if win_col < SPARSE_BIAS_COLS:
                            block_col = win_col // PREFILL_ATTN_TILE
                            pl.write(mask_row, [0, block_col], pl.cast(1, pl.INT32))
            swa_indices[idx_t : idx_t + 1, 0:WIN] = idx_row
            valid_block_mask[idx_t : idx_t + 1, 0:VALID_BLOCK_MASK_COLS] = mask_row

    cmp_block_table_dummy = pl.create_tensor([SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    cmp_kv_dummy = pl.create_tensor([CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_indices_dummy = pl.create_tensor([q_dim, IDX_TOPK], dtype=pl.INT32)
    cmp_block_table_dummy_2d = pl.reshape(cmp_block_table_dummy, [1, SPARSE_CMP_MAX_BLOCKS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cp_cmp_dummy_init"):
        cmp_block_table_dummy_2d[:, :] = pl.full([1, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32, value=0)
    for dummy_t in pl.spmd(q_dim, name_hint="prefill_swa_cp_cmp_indices_dummy_init"):
        cmp_indices_dummy[dummy_t : dummy_t + 1, :] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
    attn_out_local = sparse_attn_physical(
        q, kv_cache, swa_indices,
        cmp_kv_dummy, cmp_block_table_dummy, cmp_indices_dummy,
        valid_block_mask, attn_sink,
        rope_cos_local, rope_sin_local,
        wo_a, wo_b, wo_b_scale,
        attn_out_local, o_proj_weight_dep,
    )
    return kv_cache, attn_out_local


@pl.jit.inline
def prefill_attention_swa_cp(
    x_hc_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[BLOCK_NUM], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_local: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    x_out_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    weight_epoch: pl.Scalar[pl.INT32],
):
    """DSA-CP SWA with replicated HC state at both layer boundaries."""
    o_proj_weight_dep = gather_o_proj_full_weights(
        wo_a_local, wo_b_local,
        wo_a_full, wo_b_full,
        o_proj_wo_a_window, o_proj_wo_b_window,
        o_proj_weight_ready, o_proj_weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, weight_epoch,
    )
    q_dim = pl.tensor.dim(position_ids_local, 0)
    kv_dim = pl.tensor.dim(x_hc_full, 0)

    x_mixed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    post_full = pl.create_tensor([kv_dim, HC_MULT], dtype=pl.FP32)
    comb_full = pl.create_tensor([kv_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc_full, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed_full, post_full, comb_full)

    x_normed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed_full, attn_norm_w, x_normed_full)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    local_base = tp_rank * q_dim
    x_normed_local = pl.create_tensor([q_dim, D], dtype=pl.BF16)
    for local_row in pl.spmd(q_dim, name_hint="prefill_swa_cp_query_slice"):
        full_row = local_base + local_row
        query_row = pl.load(x_normed_full, [full_row, 0], [1, D], target_memory=pl.MemorySpace.Vec)
        pl.store(query_row, [local_row, 0], x_normed_local)

    attn_out_local = pl.create_tensor([q_dim, D], dtype=pl.BF16)
    kv_cache, attn_out_local = prefill_attention_swa_cp_core(
        x_normed_local, x_normed_full,
        wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, block_table,
        ori_slot_mapping_full,
        position_ids_local, position_ids_full,
        attn_sink,
        wo_a_full, wo_b_full, wo_b_scale,
        attn_out_local,
        late_dep, o_proj_weight_dep,
    )

    attn_out_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    attn_out_full, gather_signal = prefill_cp_token_allgather_step(
        attn_out_local, attn_out_full,
        gather_window, gather_signal,
        group_base, tp_rank,
    )

    hc_post(attn_out_full, x_hc_full, post_full, comb_full, x_out_full)
    return kv_cache, x_out_full, gather_signal


