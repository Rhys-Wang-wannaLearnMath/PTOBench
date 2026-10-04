# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped output projection (decode).

Corresponds to model.py Attention.forward decode branch lines 533-542:
    o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, softmax_scale)
    apply_rotary_emb(o[..., -rope_dim:], freqs_cis, inverse=True)
    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    x = self.wo_b(o.flatten(2))

Inputs:
- q                 : query tensor from MLA prolog (RoPE already applied)
- ori_kv / cmp_kv   : paged sliding-window and paged compressed KV pools
- cmp_sparse_indices: per-token absolute indices (window + compressed concat)
                      computed by orchestrator (window topk + indexer/HCA topk)
- attn_sink         : per-head sink term added inside softmax
- freqs_cos/sin     : split-half inverse-RoPE tables for the sparse-attn output
- wo_a              : grouped first-stage output-projection weights from model.py:537-541
- wo_b / wo_b_scale : grouped second-stage output-projection W8 per-channel weights

Standalone contract:
- `cmp_sparse_indices[t, :]` may contain `-1` pads.
- entries in `[0, WIN)` address the logical sliding-window ring slots.
- entries in `[WIN, WIN + cmp_valid)` address compressed cache slots, where
    `cmp_valid = max(seqused_kv[b] - min(WIN, seqused_kv[b]), 0)`.
- the grouped projection layout matches:
    `o.view(bsz, seqlen, self.n_local_groups, -1)` with
    `self.n_local_groups == O_GROUPS` and `-1 == O_GROUP_IN`.

"""


import pypto.language as pl

from config import (
    B,
    T,
    H,
    HEAD_DIM,
    ROPE_HEAD_DIM,
    NOPE_HEAD_DIM,
    WIN,
    SPARSE_IDX_TOPK,
    SPARSE_TOPK,
    BLOCK_SIZE,
    ORI_MAX_BLOCKS,
    BLOCK_NUM,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_CMP_BLOCK_NUM,
    SOFTMAX_SCALE,
    D,
    O_LORA,
    O_GROUPS,
    O_GROUP_IN,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)

ROPE_DIM = ROPE_HEAD_DIM
NOPE_DIM = NOPE_HEAD_DIM
IDX_TOPK = SPARSE_IDX_TOPK
TOPK = SPARSE_TOPK
ORI_BLOCK_NUM = BLOCK_NUM
CMP_MAX_BLOCKS = SPARSE_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = SPARSE_CMP_BLOCK_NUM
HALF_ROPE = ROPE_DIM // 2
MATMUL_ROW_PAD = 16

HEADS_PER_GROUP = H // O_GROUPS

A_K_CHUNK = 128
A_N_CHUNK = 128
B_K_CHUNK = 128
B_N_CHUNK = 256

QUANT_CHUNK = 256


@pl.jit.inline
def deepseek_v4_decode_sparse_attn_with_proj(
    q:                  pl.Tensor[[T, H, HEAD_DIM],                               pl.BF16],
    ori_kv:             pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    ori_block_table:    pl.Tensor[[B, ORI_MAX_BLOCKS],                            pl.INT32],
    cmp_kv:             pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    cmp_block_table:    pl.Tensor[[B, CMP_MAX_BLOCKS],                            pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK],                                      pl.INT32],
    attn_sink:          pl.Tensor[[H],                                            pl.FP32],
    seqused_kv:         pl.Tensor[[B],                                            pl.INT32],
    freqs_cos:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    freqs_sin:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    wo_a:               pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],                 pl.BF16],
    wo_b:               pl.Tensor[[D, O_GROUPS * O_LORA],                         pl.INT8],
    wo_b_scale:         pl.Tensor[[D],                                            pl.FP32],
    attn_out:           pl.Tensor[[T, D],                                         pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    A_K_BLOCKS = O_GROUP_IN // A_K_CHUNK
    A_N_BLOCKS = O_LORA // A_N_CHUNK
    B_K_BLOCKS = (O_GROUPS * O_LORA) // B_K_CHUNK
    B_N_BLOCKS = D // B_N_CHUNK

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [B * CMP_MAX_BLOCKS])
    cmp_sparse_indices_flat = pl.reshape(cmp_sparse_indices, [T * TOPK])
    attn_stage = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.BF16)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)

    for b in pl.range(B):
        seq_used = pl.read(seqused_kv, [b])
        window_valid = pl.min(WIN, seq_used)
        cmp_valid = seq_used - window_valid
        cmp_topk_valid = pl.min(IDX_TOPK, cmp_valid)
        sparse_k = window_valid + cmp_topk_valid
        ori_block_base = b * ORI_MAX_BLOCKS
        cmp_block_base = b * CMP_MAX_BLOCKS
        kv_topk_batch = pl.create_tensor([TOPK, HEAD_DIM], dtype=pl.BF16)
        raw_idx_pos = 0

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_gather_kv_topk"):
            raw_idx = pl.read(cmp_sparse_indices_flat, [raw_idx_pos])
            for kk in pl.range(window_valid):
                raw_idx_pos = b * TOPK + kk
                raw_idx = pl.read(cmp_sparse_indices_flat, [raw_idx_pos])
                ori_slot = raw_idx // BLOCK_SIZE
                ori_block_pos = ori_block_base + ori_slot
                ori_blk = pl.cast(pl.read(ori_block_table_flat, [ori_block_pos]), pl.INDEX)
                ori_intra = raw_idx % BLOCK_SIZE
                ori_row = ori_blk * BLOCK_SIZE + ori_intra
                kv_topk_batch = pl.assemble(
                    kv_topk_batch,
                    ori_kv_flat[ori_row : ori_row + 1, 0 : HEAD_DIM],
                    [kk, 0],
                )

            for kk in pl.range(cmp_topk_valid):
                raw_idx_pos = b * TOPK + window_valid + kk
                raw_idx = pl.read(cmp_sparse_indices_flat, [raw_idx_pos])
                cmp_slot = raw_idx - WIN
                cmp_block_slot = cmp_slot // BLOCK_SIZE
                cmp_block_pos = cmp_block_base + cmp_block_slot
                cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [cmp_block_pos]), pl.INDEX)
                cmp_intra = cmp_slot % BLOCK_SIZE
                cmp_row = cmp_blk * BLOCK_SIZE + cmp_intra
                kv_topk_batch = pl.assemble(
                    kv_topk_batch,
                    cmp_kv_flat[cmp_row : cmp_row + 1, 0 : HEAD_DIM],
                    [window_valid + kk, 0],
                )

        for h in pl.parallel(0, H, 1):
            head_row = b * H + h
            group_idx = h // HEADS_PER_GROUP
            head_in_group = h % HEADS_PER_GROUP
            packed_row = group_idx * T + b
            packed_col = head_in_group * HEAD_DIM
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_init"):
                q_batch = pl.col_expand(
                    pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                    pl.cast(q_flat[head_row : head_row + 1, 0 : HEAD_DIM], target_type=pl.FP32),
                )

                kv_batch = pl.col_expand(
                    pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                    pl.cast(kv_topk_batch[0 : 1, 0 : HEAD_DIM], target_type=pl.FP32),
                )
                oi = kv_batch
                score0 = pl.row_sum(pl.mul(q_batch, kv_batch))
                mi = pl.mul(score0, SOFTMAX_SCALE)
                li = pl.exp(pl.sub(mi, mi))

            for kk in pl.range(1, sparse_k):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_accum"):
                    kv_batch = pl.col_expand(
                        pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                        pl.cast(kv_topk_batch[kk : kk + 1, 0 : HEAD_DIM], target_type=pl.FP32),
                    )
                    cur_score = pl.row_sum(pl.mul(q_batch, kv_batch))
                    cur_mi = pl.mul(cur_score, SOFTMAX_SCALE)
                    mi_new = pl.maximum(mi, cur_mi)
                    alpha = pl.exp(pl.sub(mi, mi_new))
                    beta = pl.exp(pl.sub(cur_mi, mi_new))
                    li = pl.add(pl.mul(alpha, li), beta)
                    oi = pl.add(
                        pl.row_expand_mul(oi, alpha),
                        pl.row_expand_mul(kv_batch, beta),
                    )
                    mi = mi_new

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_norm"):
                sink_tile = pl.add(pl.sub(mi, mi), pl.read(attn_sink, [h]))
                denom = pl.add(li, pl.exp(pl.sub(sink_tile, mi)))
                oi_out = pl.row_expand_div(oi, denom)
                attn_stage_row = pl.cast(
                    oi_out[0 : 1, 0 : HEAD_DIM],
                    target_type=pl.BF16,
                )
                attn_stage = pl.assemble(
                    attn_stage,
                    attn_stage_row,
                    [head_row, 0],
                )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_inverse_rope"):
                o_nope = attn_stage[head_row : head_row + 1, 0 : NOPE_DIM]
                cos_lo = pl.cast(freqs_cos[b : b + 1, 0 : HALF_ROPE], target_type=pl.FP32)
                cos_hi = pl.cast(freqs_cos[b : b + 1, HALF_ROPE : ROPE_DIM], target_type=pl.FP32)
                sin_lo = pl.cast(freqs_sin[b : b + 1, 0 : HALF_ROPE], target_type=pl.FP32)
                sin_hi = pl.cast(freqs_sin[b : b + 1, HALF_ROPE : ROPE_DIM], target_type=pl.FP32)
                x_lo = pl.cast(attn_stage[head_row : head_row + 1, NOPE_DIM : NOPE_DIM + HALF_ROPE], target_type=pl.FP32)
                x_hi = pl.cast(attn_stage[head_row : head_row + 1, NOPE_DIM + HALF_ROPE : HEAD_DIM], target_type=pl.FP32)
                y_lo = pl.add(
                    pl.col_expand_mul(x_lo, cos_lo),
                    pl.col_expand_mul(x_hi, sin_lo),
                )
                y_hi = pl.sub(
                    pl.col_expand_mul(x_hi, cos_hi),
                    pl.col_expand_mul(x_lo, sin_hi),
                )
                o_packed = pl.assemble(o_packed, o_nope, [packed_row, packed_col])
                o_packed = pl.assemble(
                    o_packed,
                    pl.cast(y_lo, target_type=pl.BF16),
                    [packed_row, packed_col + NOPE_DIM],
                )
                o_packed = pl.assemble(
                    o_packed,
                    pl.cast(y_hi, target_type=pl.BF16),
                    [packed_row, packed_col + NOPE_DIM + HALF_ROPE],
                )

    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)

    for g in pl.parallel(0, O_GROUPS, 1):
        row_base_o = g * T
        out_col_g = g * O_LORA

        for nb in pl.parallel(0, A_N_BLOCKS, 1):
            n0 = nb * A_N_CHUNK

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_a_accum"):
                xa0_chunk = o_packed[row_base_o:row_base_o + T, 0:A_K_CHUNK]
                wa0_chunk = wo_a[g:g + 1, n0:n0 + A_N_CHUNK, 0:A_K_CHUNK]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, A_K_BLOCKS):
                    k0 = kb * A_K_CHUNK
                    xa_k_chunk = o_packed[row_base_o:row_base_o + T, k0:k0 + A_K_CHUNK]
                    wa_k_chunk = wo_a[g:g + 1, n0:n0 + A_N_CHUNK, k0:k0 + A_K_CHUNK]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_a_store"):
                acc_a_2d = pl.reshape(acc_a, [T, A_N_CHUNK])
                o_r[:, out_col_g + n0:out_col_g + n0 + A_N_CHUNK] = pl.cast(
                    acc_a_2d,
                    target_type=pl.BF16,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_quant"):
        or_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
            or_a_f32 = pl.cast(o_r[:, k0:k0 + QUANT_CHUNK], target_type=pl.FP32)
            or_a_abs = pl.maximum(or_a_f32, pl.neg(or_a_f32))
            or_a_max = pl.reshape(pl.row_max(or_a_abs), [1, T])
            or_amax = pl.maximum(or_amax, or_a_max)
        or_sq_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
        o_r_scale_dq = pl.reshape(pl.recip(or_sq_row), [T, 1])
        or_sq_col = pl.reshape(or_sq_row, [T, 1])
        for k1 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
            or_q_f32 = pl.cast(o_r[:, k1:k1 + QUANT_CHUNK], target_type=pl.FP32)
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="round")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[:, k1:k1 + QUANT_CHUNK] = pl.cast(or_q_half, target_type=pl.INT8, mode="trunc")

    for nb in pl.parallel(0, B_N_BLOCKS, 1):
        n0 = nb * B_N_CHUNK

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_accum"):
            xb0_chunk = o_r_i8[:, 0:B_K_CHUNK]
            wb0_chunk = wo_b[n0:n0 + B_N_CHUNK, 0:B_K_CHUNK]
            acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
            for kb in pl.range(1, B_K_BLOCKS):
                k0 = kb * B_K_CHUNK
                xb_k_chunk = o_r_i8[:, k0:k0 + B_K_CHUNK]
                wb_k_chunk = wo_b[n0:n0 + B_N_CHUNK, k0:k0 + B_K_CHUNK]
                acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_store"):
            wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + B_N_CHUNK], [1, B_N_CHUNK])
            attn_chunk = pl.cast(acc_b, target_type=pl.FP32, mode="none")
            attn_chunk = pl.col_expand_mul(pl.row_expand_mul(attn_chunk, o_r_scale_dq), wb_scale_chunk)
            attn_out[:, n0:n0 + B_N_CHUNK] = pl.cast(attn_chunk, target_type=pl.BF16)

    return attn_out
