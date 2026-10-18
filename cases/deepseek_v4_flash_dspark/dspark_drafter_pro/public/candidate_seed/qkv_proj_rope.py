# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from config import (
    D,
    EPS,
    H,
    HEAD_DIM,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    MAX_SEQ_LEN,
    NOPE_DIM,
    Q_LORA,
    ROPE_DIM,
)

import pypto.language as pl

T_DYN = pl.dynamic("QKV_Q_T_DYN")  # T = B * S

KV_T_DYN = pl.dynamic("QKV_KV_T_DYN")

ROPE_T_DYN = pl.dynamic("QKV_ROPE_T_DYN")

Q_PROJ_TILE = 128       # qproj K-tile (Q_LORA reduction)

QPROJ_MM_N_TILE = 512   # qproj output-column tile

Q_LORA_TILE = 256       # qr rms-norm / quant N granularity

KV_TILE = 64            # kv rms-norm / rope / NOPE N granularity

QUANT_TILE = 256

T_TILE = 8

MATMUL_T_TILE = 16

QR_M_TILE = MATMUL_T_TILE       # qr_proj token (M) tile; cube rows must be a 16-row boxed tile

QR_N_TILE = 128                 # qr_proj Q_LORA (N) per matmul

QR_K_TILE = 256                 # qr_proj D (K) reduction tile   | divides QR_SPLIT_K_TILE

QR_OK = 2                       # qr_proj split-K factor         | D//QR_OK cores share each N-group

QR_SPLIT_K_TILE = D // QR_OK    # qr_proj K per split (=2048)

KV_M_TILE = MATMUL_T_TILE       # kv_proj token (M) tile; decode pads from 8 real rows to 16

KV_N_TILE = 128                 # kv_proj HEAD_DIM (N) per matmul

KV_K_TILE = 256                 # kv_proj D (K) reduction tile   | divides KV_SPLIT_K_TILE

KV_OK = 4                       # kv_proj split-K factor         | D//KV_OK cores share each N-group

KV_SPLIT_K_TILE = D // KV_OK    # kv_proj K per split (=1024)

QPROJ_M_TILE = MATMUL_T_TILE    # qproj token (M) tile; decode pads from 8 real rows to 16

KV_RMS_T_TILE = 8               # kv rms-norm + rope fused token (T) tile

Q_ROPE_T_TILE = 8

Q_ROPE_H_TILE = 4               # heads per fused qproj dequant/rms/rope task

@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(position_ids, 0)
    for rope_t0 in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="qkv_rope_rows"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < num_tokens:
                rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]

@pl.jit.inline
def rope_prepare(
    rope_cos: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.INT32],
):
    """Build the head-invariant interleaved cos / sign-folded sin / swap-index rope rows."""
    t_dim = pl.tensor.dim(rope_cos, 0)
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])

    for qrp_idx in pl.spmd(t_dim // Q_ROPE_T_TILE, name_hint="q_rope_prepare", allow_early_resolve=True):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        qrp_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_dup_idx = pl.cast(qrp_dup_f, target_type=pl.INT32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        qrp_cos_rows = rope_cos_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
        qrp_sin_rows = rope_sin_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
        qrp_cos = pl.cast(qrp_cos_rows, target_type=pl.FP32)
        qrp_sin = pl.cast(qrp_sin_rows, target_type=pl.FP32)
        qrp_cos_il = pl.gather(qrp_cos, dim=-1, index=qrp_dup_idx)
        qrp_sin_il = pl.gather(qrp_sin, dim=-1, index=qrp_dup_idx)
        qrp_sin_signed = pl.mul(qrp_sin_il, qrp_sign)
        rope_cos_il[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_cos_il
        rope_sin_signed[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_sin_signed
        rope_swap_idx[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_swap_idx

@pl.jit.inline
def q_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
):
    """Q LoRA (wq_a -> rms/quant -> wq_b) + per-head RMSNorm + interleaved RoPE."""
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    t_matmul = ((t_dim + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE  # ceil to whole 16-row cube tiles

    # Split-K qr_proj (M=t_dim, K=D=4096, N=Q_LORA=1024): QR_N_TILE N-groups expanded
    # QR_OK-fold into cube blocks that atomic-add their K partials into a zero-seeded
    # output. Seeded on-core, not through create_tensor init_value=0.
    qr_fp32 = pl.create_tensor([t_matmul, Q_LORA], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
        for ts0 in pl.range(0, t_matmul, QR_M_TILE):
            for nseed0 in pl.range(0, Q_LORA, QR_N_TILE):
                qr_seed = pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0)
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = qr_seed

    for qbg_idx in pl.spmd((Q_LORA // QR_N_TILE) * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):
        q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_SPLIT_K_TILE
        for t0 in pl.range(0, t_matmul, QR_M_TILE):
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, t_dim - t0)
                q_x_chunk_bf16 = pl.slice(x_view, [QR_M_TILE, QR_K_TILE], [t0, qr_d0], valid_shape=[qr_rows, QR_K_TILE])
                w_chunk = wq_a[qr_d0 : qr_d0 + QR_K_TILE, q_a_col0 : q_a_col0 + QR_N_TILE]
                if db == 0:
                    q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=pl.AtomicType.Add)

    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    qr_i8_matmul = pl.create_tensor([t_matmul, Q_LORA], dtype=pl.INT8)

    # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
        tg = tg_idx * T_TILE
        qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        qr_amax_g = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for qr_rms_col0 in pl.pipeline(0, Q_LORA, Q_LORA_TILE, stage=2):
            qr_rms_chunk = qr_fp32[tg : tg + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
            qr_rms_sq = pl.mul(qr_rms_chunk, qr_rms_chunk)
            qr_rms_row_sum = pl.reshape(pl.row_sum(qr_rms_sq), [1, T_TILE])
            qr_sq_sum = pl.add(qr_sq_sum, qr_rms_row_sum)
            gamma_rms_cast = pl.cast(gamma_cq[qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE], target_type=pl.FP32)
            gamma_rms_chunk = pl.reshape(gamma_rms_cast, [1, Q_LORA_TILE])
            qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
            qr_g_abs = pl.abs(qr_g)
            qr_g_row_max = pl.reshape(pl.row_max(qr_g_abs), [1, T_TILE])
            qr_amax_g = pl.maximum(qr_amax_g, qr_g_row_max)
        qr_inv_rms = pl.rsqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS), high_precision=True)
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
        qr_amax_floor = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qr_amax_normed = pl.mul(qr_inv_rms, qr_amax_g)
        qr_tile_amax = pl.maximum(qr_amax_floor, qr_amax_normed)

        qr_scale_quant_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
        qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
        qr_scale_view[tg : tg + T_TILE, :] = qr_tile_scale_dq

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = qr_fp32[tg : tg + T_TILE, qa : qa + QUANT_TILE]
            gamma_q_cast = pl.cast(gamma_cq[qa : qa + QUANT_TILE], target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
            qr_view[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8
            qr_i8_matmul[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8

    # Pure-matmul qproj scope (cube, INT32 -> GM), unmixed with downstream vector work.
    q_proj_i32 = pl.create_tensor([t_matmul, H * HEAD_DIM], dtype=pl.INT32)
    # One output-column fragment per task.
    for qproj_n_idx in pl.spmd((H * HEAD_DIM) // QPROJ_MM_N_TILE, name_hint="qproj_matmul"):
        w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
        for t0 in pl.range(0, t_matmul, QPROJ_M_TILE):
            col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
            for qr_proj_col0 in pl.pipeline(0, Q_LORA, Q_PROJ_TILE, stage=2):
                qr_i8_chunk = qr_i8_matmul[t0 : t0 + QPROJ_M_TILE, qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE]
                wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                if qr_proj_col0 == 0:
                    col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                else:
                    col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
            q_proj_i32[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc

    # Fused qproj dequant, per-head RMSNorm, NOPE writeback, and interleaved RoPE.
    # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for hg_idx in pl.spmd(H // Q_ROPE_H_TILE, name_hint="qproj_dequant_rms_nope_rope", allow_early_resolve=True):
        hg = hg_idx * Q_ROPE_H_TILE
        for tg in pl.range(0, t_dim, Q_ROPE_T_TILE):
            qr_scale_dq_t = qr_scale_view[tg : tg + Q_ROPE_T_TILE, :]
            q_cos_il = rope_cos_il[tg : tg + Q_ROPE_T_TILE, :]
            q_sin_signed = rope_sin_signed[tg : tg + Q_ROPE_T_TILE, :]
            q_swap_idx = rope_swap_idx[tg : tg + Q_ROPE_T_TILE, :]
            for h_inner in pl.pipeline(Q_ROPE_H_TILE, stage=2):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_head_acc = q_proj_i32[tg : tg + Q_ROPE_T_TILE, h0 : h0 + HEAD_DIM]
                q_head_scale = pl.reshape(wq_b_scale[h0 : h0 + HEAD_DIM], [1, HEAD_DIM])
                q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                q_head_sq = pl.mul(q_head_dq, q_head_dq)
                q_head_sq_row = pl.row_sum(q_head_sq)
                q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                q_head_var = pl.add(q_head_sq_mean, EPS)
                q_head_inv_rms = pl.rsqrt(q_head_var, high_precision=True)
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])

                q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                q_flat[tg : tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_bf16

                # RoPE writeback on columns [h0+NOPE_DIM:h0+HEAD_DIM). inv_rms is folded in
                # BEFORE the rotation (it is a per-row scalar and commutes); rotating the raw
                # dequantized values first corrupts the query RoPE region on A5.
                q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                q_rope_swapped = pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)
                q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                q_flat[tg : tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + NOPE_DIM + ROPE_DIM] = q_rope_bf16

@pl.jit.inline
def kv_proj_rope(
    x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.INT32],
    kv: pl.Tensor[[KV_T_DYN, HEAD_DIM], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """KV LoRA (wkv) + fused KV RMSNorm and interleaved RoPE."""
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    t_matmul = ((t_dim + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE  # ceil to whole 16-row cube tiles

    # Split-K kv_proj: KV_N_TILE N-groups expanded KV_OK-fold into cube blocks that
    # atomic-add their K partials into a zero-seeded output.
    kv_fp32 = pl.create_tensor([t_matmul, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
        for kts0 in pl.range(0, t_matmul, KV_M_TILE):
            for kvseed0 in pl.range(0, HEAD_DIM, KV_N_TILE):
                kv_seed = pl.full([KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0)
                kv_fp32[kts0 : kts0 + KV_M_TILE, kvseed0 : kvseed0 + KV_N_TILE] = kv_seed

    # `late_dep` fences kv_proj one hop behind rms_norm so qr_proj_matmul takes the cores first.
    with pl.spmd((HEAD_DIM // KV_N_TILE) * KV_OK, name_hint="kv_proj_matmul", deps=[late_dep]) as _kv_tid:
        kbg = pl.tile.get_block_idx()
        kv_col0 = (kbg // KV_OK) * KV_N_TILE
        kv_k_base = (kbg % KV_OK) * KV_SPLIT_K_TILE
        for t0 in pl.range(0, t_matmul, KV_M_TILE):
            kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(KV_SPLIT_K_TILE // KV_K_TILE, stage=2):
                d0 = kv_k_base + db * KV_K_TILE
                kv_rows = pl.min(KV_M_TILE, t_dim - t0)
                kv_x_chunk_bf16 = pl.slice(x_view, [KV_M_TILE, KV_K_TILE], [t0, d0], valid_shape=[kv_rows, KV_K_TILE])
                wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                if db == 0:
                    kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [t0, kv_col0], atomic=pl.AtomicType.Add)

    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])

    # Fused KV RMSNorm + interleaved (CANN A3) RoPE, one spmd task per
    # [KV_RMS_T_TILE, HEAD_DIM] row block. NOPE columns [0:NOPE_DIM) and rope columns
    # [NOPE_DIM:HEAD_DIM) are disjoint, so each task writes a conflict-free row block.
    for tg_idx in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="kv_rms_norm_rope"):
        tg = tg_idx * KV_RMS_T_TILE
        # Pass 1: per-row sum of squares over the full HEAD_DIM -> inv_rms.
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
            kv_sq = pl.mul(kv_chunk, kv_chunk)
            kv_row_sum = pl.reshape(pl.row_sum(kv_sq), [1, KV_RMS_T_TILE])
            kv_sq_sum = pl.add(kv_sq_sum, kv_row_sum)
        kv_inv_rms = pl.rsqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS), high_precision=True)
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

        # NOPE writeback: rms-normalize columns [0:NOPE_DIM) with per-column gamma.
        for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
            gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
            kv_view[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16

        # RoPE writeback on columns [NOPE_DIM:HEAD_DIM), interleaved (CANN A3) swap-gather:
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        # inv_rms and gamma are folded into kv_rope_norm_chunk BEFORE the swap so the
        # swapped lane n[j^1] carries gamma[j^1] — gamma does NOT commute with the rotation.
        gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
        kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
        kv_cos_il = rope_cos_il[tg : tg + KV_RMS_T_TILE, :]
        kv_sin_signed = rope_sin_signed[tg : tg + KV_RMS_T_TILE, :]
        kv_swapped = pl.gather(kv_rope_norm_chunk, dim=-1, index=rope_swap_idx[tg : tg + KV_RMS_T_TILE, :])
        kv_rope_rot = pl.add(pl.mul(kv_rope_norm_chunk, kv_cos_il), pl.mul(kv_swapped, kv_sin_signed))
        kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode="rint")
        kv_view[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM] = kv_rope_i16

