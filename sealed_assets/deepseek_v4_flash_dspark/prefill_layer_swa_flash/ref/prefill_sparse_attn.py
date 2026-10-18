# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""REF closure for deepseek_v4_flash_dspark.prefill_layer_swa_flash; fixed bindings from public config."""
from config import BLOCK_SIZE
from config import D
from config import FP32_NEG_INF
from config import H
from config import HEADS_PER_GROUP
from config import HEAD_DIM
from config import IDX_TOPK
from config import INT8_AMAX_EPS
from config import INT8_SCALE_MAX
from config import KV_CMP_MAX_BLOCKS as CMP_MAX_BLOCKS
from config import NOPE_HEAD_DIM as NOPE_DIM
from config import O_GROUPS
from config import O_GROUP_IN
from config import O_LORA
from config import ROPE_HALF
from config import ROPE_HEAD_DIM as ROPE_DIM
from config import SOFTMAX_SCALE
from config import WIN
from prefill_indexer import INDEXER_TOPK_CAP
import pypto.language as pl

PREFILL_DENSE_TILE = 512
PREFILL_QUERY_TILE = 128
T_DYN = pl.dynamic("PREFILL_ATTN_T_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
T_PAD = PREFILL_DENSE_TILE
TOPK = WIN + IDX_TOPK
PREFILL_MAX_COMPRESSED = INDEXER_TOPK_CAP
PREFILL_SPARSE_TOPK = WIN + PREFILL_MAX_COMPRESSED
HEAD_TILE = 16
QK_M_TILE = 32
GATHER_TOKEN_TILE = 2
BIAS_TOKEN_TILE = 16
QUANT_TOKEN_TILE = 8
O_PROJ_PAD_ROWS = 512
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 256
B_K_TILE = 256
QUANT_TILE = 512
PROJ_A_MM_N_TILE = 128
PROJ_B_MM_N_TILE = 256
PROJ_B_D_TILE = 512
PROJ_B_ACT_N_TILE = 512
PROJ_B_ACT_T_TILE = 16
PROJ_B_ACT_TASK_T_TILE = 32
ROPE_CS_T_TILE = 8
PROJ_A_ROW_TILE = min(256, T_PAD)
PROJ_B_ROW_TILE = min(128, T_PAD)
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE
PB_DSLABS = D // PROJ_B_D_TILE
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
PREFILL_QUERY_BLOCKS = T_PAD // PREFILL_QUERY_TILE
PREFILL_QUERY_STATS_ROWS = PREFILL_QUERY_TILE * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
MASK_ALIGN_ELEMS = 32 // 4
VALID_BLOCK_MASK_COLS = ((PREFILL_ATTN_BLOCKS + MASK_ALIGN_ELEMS - 1) // MASK_ALIGN_ELEMS) * MASK_ALIGN_ELEMS
SPARSE_BIAS_COLS = min(TOPK, PREFILL_SPARSE_PAD)
SPARSE_CMP_BIAS_COLS = max(0, SPARSE_BIAS_COLS - WIN)

@pl.jit.inline(auto_scope=False)
def _prepare_sparse_attn_rope(
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[T_PAD, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_PAD, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[HEAD_TILE, ROPE_DIM], pl.INT32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.TASK_ID]:
    """Build one dense tile of inverse-RoPE tables."""
    rope_cs_blocks = (tile_rows + ROPE_CS_T_TILE - 1) // ROPE_CS_T_TILE
    with pl.spmd(ROPE_HALF // ROPE_TILE, name_hint="rope_cs") as rope_cs_tid:
        cp = pl.tile.get_block_idx()
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0

        swap_ones = pl.full([HEAD_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        swap_ramp = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_ramp)
        swap_dup_i32 = pl.cast(pl.mul(swap_col, 0.5), target_type=pl.INT32, mode="trunc")
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))  # j%2
        swap_pair_f = pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0))  # j^1
        swap_idx = pl.cast(swap_pair_f, target_type=pl.INT32)
        rope_swap_idx[:, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = swap_idx[
            :, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE
        ]

        cs_ones = pl.full([ROPE_CS_T_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
        cs_ramp_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
        cs_ramp = pl.cast(cs_ramp_i32, target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_ramp)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)  # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))  # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))  # [+1,-1,...]
        for cs_rb in pl.range(rope_cs_blocks):
            cs_local_t0 = cs_rb * ROPE_CS_T_TILE
            cs_t0 = tile_base + cs_local_t0
            cs_rows = pl.min(ROPE_CS_T_TILE, tile_rows - cs_local_t0)
            cs_cos_rows = pl.slice(
                freqs_cos,
                [ROPE_CS_T_TILE, ROPE_TILE],
                [cs_t0, cp_r0],
                valid_shape=[cs_rows, ROPE_TILE],
            )
            cs_sin_rows = pl.slice(
                freqs_sin,
                [ROPE_CS_T_TILE, ROPE_TILE],
                [cs_t0, cp_r0],
                valid_shape=[cs_rows, ROPE_TILE],
            )
            cs_cos = pl.cast(cs_cos_rows, target_type=pl.FP32)
            cs_sin = pl.cast(cs_sin_rows, target_type=pl.FP32)
            cs_cos_il = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
            cs_sin_il = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
            cs_sin_signed = pl.mul(cs_sin_il, cs_sign)
            rope_cos_il[
                cs_local_t0 : cs_local_t0 + ROPE_CS_T_TILE,
                cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE,
            ] = cs_cos_il
            rope_sin_signed[
                cs_local_t0 : cs_local_t0 + ROPE_CS_T_TILE,
                cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE,
            ] = cs_sin_signed
    return rope_cs_tid


@pl.jit.inline(auto_scope=False)
def _sparse_attn_wave(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[T_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16],
    sparse_kv: pl.Tensor[[PREFILL_QUERY_TILE * PREFILL_SPARSE_PAD, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[PREFILL_QUERY_TILE, PREFILL_SPARSE_PAD], pl.FP32],
    sparse_blk_mi: pl.Tensor[[PREFILL_QUERY_STATS_ROWS, 1], pl.FP32],
    sparse_blk_li: pl.Tensor[[PREFILL_QUERY_STATS_ROWS, 1], pl.FP32],
    sparse_blk_oi: pl.Tensor[[PREFILL_QUERY_STATS_ROWS, HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_PAD, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_PAD, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[HEAD_TILE, ROPE_DIM], pl.INT32],
    rope_cs_tid: pl.Scalar[pl.TASK_ID],
    prior_merge_tid: pl.Scalar[pl.TASK_ID],
    query_base: pl.Scalar[pl.INDEX],
    dense_query_base: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one 128-row gather/QK-PV/merge wave over reusable scratch."""
    t_dim = pl.tensor.dim(q, 0)
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])
    q_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    gather_blocks = PREFILL_QUERY_TILE // GATHER_TOKEN_TILE
    gather_cmp_blocks = gather_blocks * (PREFILL_ATTN_BLOCKS - 1)

    # Stage original-window rows in reverse token-tile order.
    with pl.spmd(
        gather_blocks,
        name_hint="gather_ori_kv",
        deps=[prior_merge_tid],
    ) as gather_ori_tid:
        gather_schedule_block = pl.tile.get_block_idx()
        gather_token_block = gather_blocks - 1 - gather_schedule_block
        gather_local_t0 = gather_token_block * GATHER_TOKEN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_local_t = gather_local_t0 + gather_dt
            gather_t = query_base + gather_local_t
            if gather_t < t_dim:
                if gather_t < active_rows:
                    block_base = gather_local_t * PREFILL_SPARSE_PAD
                    stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    for gather_ki in pl.range(PREFILL_ATTN_TILE):
                        gather_raw = pl.read(swa_indices, [gather_t, gather_ki])
                        if gather_raw >= 0:
                            src = pl.cast(gather_raw, pl.INDEX)
                            stage[gather_ki : gather_ki + 1, :] = ori_kv_flat[src : src + 1, :]
                    sparse_kv[block_base : block_base + PREFILL_ATTN_TILE, :] = stage

    with pl.spmd(
        gather_cmp_blocks,
        name_hint="gather_cmp_kv",
        deps=[prior_merge_tid],
    ) as gather_cmp_tid:
        gather_block = pl.tile.get_block_idx()
        gather_schedule_block = gather_block // (PREFILL_ATTN_BLOCKS - 1)
        gather_token_block = gather_blocks - 1 - gather_schedule_block
        gather_sb = gather_block - gather_schedule_block * (PREFILL_ATTN_BLOCKS - 1) + 1
        gather_local_t0 = gather_token_block * GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_local_t = gather_local_t0 + gather_dt
            gather_t = query_base + gather_local_t
            if gather_t < t_dim:
                if gather_t < active_rows:
                    gather_block_valid = pl.read(valid_block_mask, [gather_t, gather_sb])
                    if gather_block_valid > 0:
                        block_base = gather_local_t * PREFILL_SPARSE_PAD + gather_k0
                        stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        for gather_ki in pl.range(PREFILL_ATTN_TILE):
                            gather_cmp_k = gather_k0 + gather_ki - WIN
                            if gather_cmp_k < IDX_TOPK:
                                gather_raw = pl.read(cmp_indices, [gather_t, gather_cmp_k])
                                if gather_raw >= 0:
                                    cmp_slot = gather_raw
                                    blk_slot = cmp_slot // BLOCK_SIZE
                                    blk = pl.cast(pl.read(cmp_block_table, [blk_slot]), pl.INDEX)
                                    src = blk * BLOCK_SIZE + (cmp_slot - blk_slot * BLOCK_SIZE)
                                    stage[gather_ki : gather_ki + 1, :] = cmp_kv_flat[src : src + 1, :]
                        sparse_kv[block_base : block_base + PREFILL_ATTN_TILE, :] = stage

    # Keep the existing 16-row vectorized bias path inside each query wave.
    with pl.spmd(
        PREFILL_QUERY_TILE // BIAS_TOKEN_TILE,
        name_hint="build_bias",
        deps=[prior_merge_tid],
    ) as bias_tid:
        bias_blk = pl.tile.get_block_idx()
        bias_local_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_t0 = query_base + bias_local_t0
        if bias_t0 < active_rows:
            bias_rows = pl.min(BIAS_TOKEN_TILE, active_rows - bias_t0)
            bias_win_rows = pl.slice(
                swa_indices,
                [BIAS_TOKEN_TILE, WIN],
                [bias_t0, 0],
                valid_shape=[bias_rows, WIN],
            )
            bias_win_idx = pl.cast(bias_win_rows, target_type=pl.FP32)
            bias_win_raw_flag = pl.minimum(pl.maximum(pl.add(bias_win_idx, 1.0), 0.0), 1.0)
            bias_win = pl.mul(pl.sub(bias_win_raw_flag, 1.0), -FP32_NEG_INF)
            sparse_bias[bias_local_t0 : bias_local_t0 + BIAS_TOKEN_TILE, 0:WIN] = bias_win
            if SPARSE_CMP_BIAS_COLS > 0:
                bias_cmp_rows = pl.slice(
                    cmp_indices,
                    [BIAS_TOKEN_TILE, SPARSE_CMP_BIAS_COLS],
                    [bias_t0, 0],
                    valid_shape=[bias_rows, SPARSE_CMP_BIAS_COLS],
                )
                bias_cmp_idx = pl.cast(bias_cmp_rows, target_type=pl.FP32)
                bias_cmp_raw_flag = pl.minimum(pl.maximum(pl.add(bias_cmp_idx, 1.0), 0.0), 1.0)
                bias_cmp = pl.mul(pl.sub(bias_cmp_raw_flag, 1.0), -FP32_NEG_INF)
                sparse_bias[
                    bias_local_t0 : bias_local_t0 + BIAS_TOKEN_TILE,
                    WIN:SPARSE_BIAS_COLS,
                ] = bias_cmp
            if PREFILL_SPARSE_PAD > SPARSE_BIAS_COLS:
                bias_pad_cols = PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS
                bias_pad = pl.full(
                    [BIAS_TOKEN_TILE, bias_pad_cols],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
                sparse_bias[
                    bias_local_t0 : bias_local_t0 + BIAS_TOKEN_TILE,
                    SPARSE_BIAS_COLS:PREFILL_SPARSE_PAD,
                ] = bias_pad

    # Consume the staged wave for QK/PV. Statistics are indexed by the
    # local row so the same 128-row scratch can be reused by every wave.
    with pl.spmd(
        PREFILL_QUERY_TILE,
        name_hint="qk_pv",
        deps=[gather_ori_tid, gather_cmp_tid, bias_tid],
    ) as qk_tid:
        qk_local_t = pl.tile.get_block_idx()
        qk_t = query_base + qk_local_t
        if qk_t < t_dim:
            if qk_t < active_rows:
                qk_kv_base = qk_local_t * PREFILL_SPARSE_PAD
                qk_token_base = qk_local_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
                for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                    qk_s0 = qk_kv_base + qk_sb * PREFILL_ATTN_TILE
                    qk_b0 = qk_sb * PREFILL_ATTN_TILE
                    qk_bias_row = sparse_bias[
                        qk_local_t : qk_local_t + 1,
                        qk_b0 : qk_b0 + PREFILL_ATTN_TILE,
                    ]
                    qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])
                    if qk_sb == 0:
                        qk_block_valid = pl.cast(1, pl.INT32)
                    if qk_block_valid > 0:
                        # Separate QK and PV cache views over the same rows.
                        qk_kv_k = sparse_kv[qk_s0 : qk_s0 + PREFILL_ATTN_TILE, :]
                        qk_kv_v = sparse_kv[qk_s0 : qk_s0 + PREFILL_ATTN_TILE, :]
                        for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                            qk_head_row = qk_t * H + qk_hb * QK_M_TILE
                            qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, :]
                            qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                            qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                            qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                            qk_mi = pl.row_max(qk_scores)
                            qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                            qk_li = pl.row_sum(qk_exp)
                            qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                            qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                            for qk_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                                qk_h_idx = qk_hb * (QK_M_TILE // HEAD_TILE) + qk_sub
                                qk_r0 = qk_sub * HEAD_TILE
                                qk_blk_base = qk_token_base + qk_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
                                qk_row = qk_blk_base + qk_sb * HEAD_TILE
                                sparse_blk_mi[
                                    qk_row : qk_row + HEAD_TILE,
                                    :,
                                ] = qk_mi[qk_r0 : qk_r0 + HEAD_TILE, :]
                                sparse_blk_li[
                                    qk_row : qk_row + HEAD_TILE,
                                    :,
                                ] = qk_li[qk_r0 : qk_r0 + HEAD_TILE, :]
                                sparse_blk_oi[
                                    qk_row : qk_row + HEAD_TILE,
                                    :,
                                ] = qk_oi[qk_r0 : qk_r0 + HEAD_TILE, :]

    # Merge the local statistics and write the global group-major rows.
    with pl.spmd(
        PREFILL_QUERY_TILE,
        name_hint="merge_rope_pack",
        deps=[qk_tid, rope_cs_tid],
    ) as merge_tid:
        m_local_t = pl.tile.get_block_idx()
        m_t = query_base + m_local_t
        if m_t < active_rows:
            m_dense_t = dense_query_base + m_local_t
            m_token_base = m_local_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
            # Load the aligned mask row; only set blocks have statistics.
            m_mask_row = valid_block_mask[m_t : m_t + 1, :]
            m_swap_idx = rope_swap_idx[:, :]
            m_cos_il = rope_cos_il[m_dense_t : m_dense_t + 1, :]
            m_sin_signed = rope_sin_signed[m_dense_t : m_dense_t + 1, :]
            for m_h_idx in pl.range(H // HEAD_TILE):
                m_h0 = m_h_idx * HEAD_TILE
                if m_t < active_rows:
                    m_blk_base = m_token_base + m_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
                    m_mi = sparse_blk_mi[m_blk_base : m_blk_base + HEAD_TILE, :]
                    m_li = sparse_blk_li[m_blk_base : m_blk_base + HEAD_TILE, :]
                    m_oi = sparse_blk_oi[m_blk_base : m_blk_base + HEAD_TILE, :]
                    for m_sb in pl.unroll(1, PREFILL_ATTN_BLOCKS):
                        m_block_valid = pl.read(m_mask_row, [0, m_sb])
                        if m_block_valid > 0:
                            m_row = m_blk_base + m_sb * HEAD_TILE
                            cur_mi = sparse_blk_mi[m_row : m_row + HEAD_TILE, :]
                            cur_li = sparse_blk_li[m_row : m_row + HEAD_TILE, :]
                            cur_oi = sparse_blk_oi[m_row : m_row + HEAD_TILE, :]
                            mi_new = pl.maximum(m_mi, cur_mi)
                            alpha = pl.exp(pl.sub(m_mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            m_li = pl.add(pl.mul(alpha, m_li), pl.mul(beta, cur_li))
                            m_oi_alpha = pl.row_expand_mul(m_oi, alpha)
                            cur_oi_beta = pl.row_expand_mul(cur_oi, beta)
                            m_oi = pl.add(m_oi_alpha, cur_oi_beta)
                            m_mi = mi_new
                    sink_bias = attn_sink_col[m_h0 : m_h0 + HEAD_TILE, :]
                    sink_tile = pl.add(pl.sub(m_mi, m_mi), sink_bias)
                    denom = pl.add(m_li, pl.exp(pl.sub(sink_tile, m_mi)))
                    n_full = pl.row_expand_div(m_oi, denom)[0:HEAD_TILE, :]
                    n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
                else:
                    n_full = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                    n_bf16 = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)

                m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
                m_swapped = pl.gather(m_rope, dim=-1, index=m_swap_idx)
                m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
                m_swapped_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
                m_rot = pl.add(m_rope_cos, m_swapped_sin)
                n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

                if HEAD_TILE % HEADS_PER_GROUP == 0:
                    m_g0 = m_h0 // HEADS_PER_GROUP
                    for m_sg in pl.unroll(HEAD_TILE // HEADS_PER_GROUP):
                        m_src_h0 = m_sg * HEADS_PER_GROUP
                        m_pack_row = (m_g0 + m_sg) * T_PAD + m_dense_t
                        m_dst_head = m_pack_row * HEADS_PER_GROUP
                        # Imperative stores keep this persistent buffer at one GM identity
                        # across the 16 statically unrolled waves.
                        pl.assemble(
                            o_packed_heads,
                            n_bf16[
                                m_src_h0 : m_src_h0 + HEADS_PER_GROUP,
                                0:NOPE_DIM,
                            ],
                            [m_dst_head, 0],
                        )
                        pl.assemble(
                            o_packed_heads,
                            n_rope_bf16[
                                m_src_h0 : m_src_h0 + HEADS_PER_GROUP,
                                0:ROPE_DIM,
                            ],
                            [m_dst_head, NOPE_DIM],
                        )
                else:
                    # Store groups crossing a head tile one head at a time.
                    for m_hi in pl.range(HEAD_TILE):
                        m_gh = m_h0 + m_hi
                        m_g = m_gh // HEADS_PER_GROUP
                        m_pack_row = m_g * T_PAD + m_dense_t
                        m_col = (m_gh - m_g * HEADS_PER_GROUP) * HEAD_DIM
                        o_packed[
                            m_pack_row : m_pack_row + 1,
                            m_col : m_col + NOPE_DIM,
                        ] = n_bf16[m_hi : m_hi + 1, 0:NOPE_DIM]
                        m_rope_row = n_rope_bf16[m_hi : m_hi + 1, 0:ROPE_DIM]
                        o_packed[
                            m_pack_row : m_pack_row + 1,
                            m_col + NOPE_DIM : m_col + HEAD_DIM,
                        ] = m_rope_row

    return merge_tid


@pl.jit.inline(auto_scope=False)
def _sparse_attn_heads(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[T_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16],
    packed_init_tid: pl.Scalar[pl.TASK_ID],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write one bounded dense tile through static 128-row waves."""
    merge_tids = pl.array.create(1, pl.TASK_ID)
    # Keep the manual-dependency zero-fill in the same chain as every head writer and the output projection.
    merge_tids[0] = packed_init_tid
    with pl.scope():
        sparse_kv = pl.create_tensor(
            [PREFILL_QUERY_TILE * PREFILL_SPARSE_PAD, HEAD_DIM],
            dtype=pl.BF16,
        )
        sparse_bias = pl.create_tensor(
            [PREFILL_QUERY_TILE, PREFILL_SPARSE_PAD],
            dtype=pl.FP32,
        )
        sparse_blk_mi = pl.create_tensor([PREFILL_QUERY_STATS_ROWS, 1], dtype=pl.FP32)
        sparse_blk_li = pl.create_tensor([PREFILL_QUERY_STATS_ROWS, 1], dtype=pl.FP32)
        sparse_blk_oi = pl.create_tensor([PREFILL_QUERY_STATS_ROWS, HEAD_DIM], dtype=pl.FP32)
        rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
        rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
        rope_swap_idx = pl.create_tensor([HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
        rope_cs_tid = _prepare_sparse_attn_rope(
            freqs_cos,
            freqs_sin,
            rope_cos_il,
            rope_sin_signed,
            rope_swap_idx,
            tile_base,
            tile_rows,
        )
        for query_block in pl.unroll(PREFILL_QUERY_BLOCKS):
            dense_query_base = query_block * PREFILL_QUERY_TILE
            query_base = tile_base + dense_query_base
            prior_merge_tid = merge_tids[0]
            merge_tid = prior_merge_tid
            if dense_query_base < tile_rows:
                merge_tid = _sparse_attn_wave(
                    q,
                    ori_kv,
                    swa_indices,
                    cmp_kv,
                    cmp_block_table,
                    cmp_indices,
                    valid_block_mask,
                    attn_sink,
                    active_rows,
                    o_packed_heads,
                    sparse_kv,
                    sparse_bias,
                    sparse_blk_mi,
                    sparse_blk_li,
                    sparse_blk_oi,
                    rope_cos_il,
                    rope_sin_signed,
                    rope_swap_idx,
                    rope_cs_tid,
                    prior_merge_tid,
                    query_base,
                    dense_query_base,
                )
            merge_tids[0] = merge_tid

    return o_packed_heads, merge_tids[0]


@pl.jit.inline(auto_scope=False)
def _sparse_attn_o_proj(
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    weight_dep: pl.Scalar[pl.TASK_ID],
):
    """Project one local-token dense tile into BF16 hidden rows."""
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])

    # Chain merge, quantization, and projection through explicit task dependencies.
    o_r = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T_PAD], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    quant_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    proj_b_tids = pl.array.create(PB_DSLABS * O_GROUPS, pl.TASK_ID)
    # Keep arbitrary physical tails out of the Cube/quantization chain.  The packed-head
    # buffer is zero-initialized, so rounding the internal O extent to a complete 512-row
    # quant tile is mathematically neutral; only the final store is cropped to tile_rows.
    o_compute_rows = ((tile_rows + O_PROJ_PAD_ROWS - 1) // O_PROJ_PAD_ROWS) * O_PROJ_PAD_ROWS
    proj_a_rows = o_compute_rows // PROJ_A_ROW_TILE
    quant_rows = o_compute_rows // QUANT_TOKEN_TILE
    proj_b_rows = o_compute_rows // PROJ_B_ROW_TILE

    with pl.manual_scope():
        # proj_a[g, nf]: BF16 grouped GEMM -> o_r[:, group g], peel-first-iter form.
        # Row-blocked over PROJ_A_ROW_TILE with a padded load, so the T_PAD-strided
        # o_packed slab never feeds uninitialized rows into the matmul.
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T_PAD
            out_col_g = g * O_LORA
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.spmd(proj_a_rows, name_hint="proj_a_mm", deps=[heads_dep, weight_dep]) as pa_tid:
                    pa_rb = pl.tile.get_block_idx()
                    pa_r0 = pa_rb * PROJ_A_ROW_TILE
                    pa_src0 = row_base_o + pa_r0
                    xa0_chunk = o_packed[
                        pa_src0 : pa_src0 + PROJ_A_ROW_TILE,
                        0:A_K_TILE,
                    ]
                    wa0_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = o_packed[
                            pa_src0 : pa_src0 + PROJ_A_ROW_TILE,
                            k0 : k0 + A_K_TILE,
                        ]
                        wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
                    o_r = pl.assemble(o_r, acc_a, [pa_r0, out_col_g + n0])
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid

        # quant[g]: per-group amax and symmetric INT8 quant over physical rows.
        for g in pl.parallel(O_GROUPS):
            col_g = g * O_LORA
            with pl.spmd(
                quant_rows,
                name_hint="quant",
                deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)],
            ) as q_tid:
                qt = pl.tile.get_block_idx() * QUANT_TOKEN_TILE
                g_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for k1 in pl.range(0, O_LORA, QUANT_TILE):
                    oc = o_r[
                        qt : qt + QUANT_TOKEN_TILE,
                        col_g + k1 : col_g + k1 + QUANT_TILE,
                    ]
                    oc_abs = pl.maximum(oc, pl.neg(oc))
                    oc_amax = pl.reshape(pl.row_max(oc_abs), [1, QUANT_TOKEN_TILE])
                    g_amax = pl.maximum(g_amax, oc_amax)
                g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                g_sq_row = pl.div(g_scale_num, g_amax)
                act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = pl.recip(g_sq_row)
                g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                for k1 in pl.range(0, O_LORA, QUANT_TILE):
                    oc = o_r[
                        qt : qt + QUANT_TOKEN_TILE,
                        col_g + k1 : col_g + k1 + QUANT_TILE,
                    ]
                    oq_scaled = pl.row_expand_mul(oc, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8[
                        qt : qt + QUANT_TOKEN_TILE,
                        col_g + k1 : col_g + k1 + QUANT_TILE,
                    ] = oq_i8
            quant_tids[g] = q_tid

        # proj_b_mm[dc, g]: INT8 GEMM of group g's contribution to a PROJ_B_D_TILE-wide slab
        # of D, written as INT32 partials[:, g*D+n]. Peel the first matmul: matmul_acc from a
        # zero carry trips TLOAD DN->NZ (pypto#1540).
        for dc in pl.parallel(PB_DSLABS):
            d0 = dc * PROJ_B_D_TILE
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mm", deps=[quant_tids[g]]) as pb_tid:
                    for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        for pb_rb in pl.range(proj_b_rows):
                            pb_r0 = pb_rb * PROJ_B_ROW_TILE
                            b_act0 = o_r_i8[
                                pb_r0 : pb_r0 + PROJ_B_ROW_TILE,
                                col_g : col_g + B_K_TILE,
                            ]
                            b_weight0 = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, col_g : col_g + B_K_TILE]
                            acc_b = pl.matmul(b_act0, b_weight0, b_trans=True, out_dtype=pl.INT32)
                            for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                                k0 = col_g + kb * B_K_TILE
                                b_act = o_r_i8[
                                    pb_r0 : pb_r0 + PROJ_B_ROW_TILE,
                                    k0 : k0 + B_K_TILE,
                                ]
                                b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE]
                                acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True)
                            partials[
                                pb_r0 : pb_r0 + PROJ_B_ROW_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE
                            ] = acc_b
                proj_b_tids[dc * O_GROUPS + g] = pb_tid

    # Dequantize and sum per-group INT32 partials into the BF16 output.
    act_t_blks = (tile_rows + PROJ_B_ACT_TASK_T_TILE - 1) // PROJ_B_ACT_TASK_T_TILE
    act_blocks = (D // PROJ_B_ACT_N_TILE) * act_t_blks
    with pl.spmd(
        act_blocks, name_hint="proj_b_act",
        deps=[proj_b_tids[i] for i in range(PB_DSLABS * O_GROUPS)],
    ) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // act_t_blks
        tblk = act_idx - nreg * act_t_blks
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
            if b_tb < tile_rows:
                out_rows = pl.min(PROJ_B_ACT_T_TILE, tile_rows - b_tb)
                acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
                for g in pl.range(O_GROUPS):
                    p_g = partials[
                        b_tb : b_tb + PROJ_B_ACT_T_TILE,
                        g * D + ob_n0 : g * D + ob_n0 + PROJ_B_ACT_N_TILE,
                    ]
                    g_scale_row = act_scale_dq[g : g + 1, b_tb : b_tb + PROJ_B_ACT_T_TILE]
                    g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                    p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
                    p_g_scaled = pl.row_expand_mul(p_g_f32, g_scale)
                    acc = pl.add(acc, p_g_scaled)
                out_t = pl.col_expand_mul(acc, wb_scale_chunk)
                out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
                out_valid = pl.set_validshape(out_bf16, out_rows, PROJ_B_ACT_N_TILE)
                out_t0 = tile_base + b_tb
                pl.assemble(attn_out, out_valid, [out_t0, ob_n0])

    return attn_out


@pl.jit.inline(auto_scope=False)
def sparse_attn_compute(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[T_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    o_proj_weight_dep: pl.Scalar[pl.TASK_ID],
):
    """Run sparse heads and dense output projection over bounded physical tiles."""
    for tile_base in pl.range(0, active_rows, T_PAD):
        # Keep this explicit because full-forward layer entries deliberately
        # disable compiler-wide scope insertion.  Under an auto-scoped entry it
        # replaces (rather than nests inside) the compiler's loop-body scope.
        with pl.scope():
            tile_rows = pl.min(T_PAD, active_rows - tile_base)
            o_packed_heads = pl.create_tensor(
                [O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM],
                dtype=pl.BF16,
                manual_dep=True,
            )
            with pl.spmd(T_PAD * H // PREFILL_QUERY_TILE, name_hint="prefill_sparse_packed_init") as packed_init_tid:
                packed_row = pl.tile.get_block_idx() * PREFILL_QUERY_TILE
                packed_zero = pl.full([PREFILL_QUERY_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                o_packed_heads[packed_row : packed_row + PREFILL_QUERY_TILE, 0:HEAD_DIM] = packed_zero
            o_packed_heads, heads_dep = _sparse_attn_heads(
                q,
                ori_kv,
                swa_indices,
                cmp_kv,
                cmp_block_table,
                cmp_indices,
                valid_block_mask,
                attn_sink,
                active_rows,
                freqs_cos,
                freqs_sin,
                o_packed_heads,
                packed_init_tid,
                tile_base,
                tile_rows,
            )
            attn_out = _sparse_attn_o_proj(
                o_packed_heads,
                wo_a,
                wo_b,
                wo_b_scale,
                attn_out,
                tile_base,
                tile_rows,
                heads_dep,
                o_proj_weight_dep,
            )
    return attn_out


@pl.jit.inline
def sparse_attn_physical(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[T_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    o_proj_weight_dep: pl.Scalar[pl.TASK_ID],
):
    """Run sparse attention over the full physical token dimension."""
    active_rows = pl.tensor.dim(q, 0)
    return sparse_attn_compute(
        q,
        ori_kv,
        swa_indices,
        cmp_kv,
        cmp_block_table,
        cmp_indices,
        valid_block_mask,
        attn_sink,
        active_rows,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        o_proj_weight_dep,
    )


