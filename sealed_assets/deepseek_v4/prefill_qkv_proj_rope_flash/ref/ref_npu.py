# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill attn_norm + Q/KV projection + partial RoPE.

The kernel shape is [PREFILL_BATCH, PREFILL_SEQ] from config. The
implementation still splits B * S into smaller internal token chunks.
"""

import pypto.language as pl

from config import CASE_CONFIG as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings

# Prefill QKV tiling. These constants intentionally live in this file instead
# of being imported from decode qkv, because some values depend on this
# kernel's own B/S/T shape.
HEAD_CHUNK = 64
HEAD_GROUP = 8
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_CHUNK = 512
Q_HEAD_RMS_GROUP = 2
Q_PROJ_DEQUANT_GROUP = 16
QR_PROJ_GROUP = 2
KV_PROJ_SPMD_GROUP = 2
Q_LORA_TILE = 32
Q_LORA_CHUNK = Q_LORA_TILE
D_CHUNK = 128 if T >= 128 else (256 if T >= 64 else 512)
KV_CHUNK = 32
PREFILL_ATTN_NORM_T_TILE = 8
QPROJ_T_TILE = 16
KV_RMS_T_TILE = 16
Q_ROPE_T_TILE = 32
KV_ROPE_T_TILE = 32
QUANT_CHUNK = 32 if T >= 128 else (128 if T >= 64 else 256)
assert (H * HEAD_DIM) % (HEAD_CHUNK * HEAD_GROUP) == 0, \
    "HEAD_BLOCKS must be divisible by HEAD_GROUP"
assert H % Q_HEAD_RMS_GROUP == 0, \
    "H must be divisible by Q_HEAD_RMS_GROUP"
assert ((H * HEAD_DIM) // Q_PROJ_OUT_CHUNK) % Q_PROJ_DEQUANT_GROUP == 0, \
    "Q_PROJ_HEAD_BLOCKS must be divisible by Q_PROJ_DEQUANT_GROUP"
assert (Q_LORA // Q_LORA_TILE) % QR_PROJ_GROUP == 0, \
    "Q_BLOCKS must be divisible by QR_PROJ_GROUP"
assert (HEAD_DIM // KV_CHUNK) % KV_PROJ_SPMD_GROUP == 0, \
    "KV_BLOCKS must be divisible by KV_PROJ_SPMD_GROUP"
Q_BLOCKS = Q_LORA // Q_LORA_TILE
Q_PROJ_BLOCKS = Q_LORA // Q_PROJ_CHUNK
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
D_BLOCKS = D // D_CHUNK
KV_BLOCKS = HEAD_DIM // KV_CHUNK

PREFILL_START_POS = 0
PREFILL_ROPE_BATCH_TILE = min(B, max(1, 256 // S))
assert B % PREFILL_ROPE_BATCH_TILE == 0, "B must be divisible by PREFILL_ROPE_BATCH_TILE"


@pl.jit.inline
def prefill_attn_norm(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    norm_w:    pl.Tensor[[D],                    pl.FP32],
    x_normed:  pl.Tensor[[B, S, D],              pl.BF16],
):
    x_flat = pl.reshape(x, [T, D])
    x_normed_flat = pl.reshape(x_normed, [T, D])

    for norm_tg_idx in pl.spmd(T // PREFILL_ATTN_NORM_T_TILE, name_hint="prefill_attn_norm"):
        norm_tg = norm_tg_idx * PREFILL_ATTN_NORM_T_TILE
        x_sq_sum = pl.full([1, PREFILL_ATTN_NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for rms_db in pl.pipeline(D_BLOCKS, stage=2):
            rms_d0 = rms_db * D_CHUNK
            rms_x_chunk = pl.cast(x_flat[norm_tg : norm_tg + PREFILL_ATTN_NORM_T_TILE, rms_d0 : rms_d0 + D_CHUNK], target_type=pl.FP32)
            x_sq_sum = pl.add(
                x_sq_sum,
                pl.reshape(pl.row_sum(pl.mul(rms_x_chunk, rms_x_chunk)), [1, PREFILL_ATTN_NORM_T_TILE]),
            )
        x_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS)))
        x_inv_rms_t = pl.reshape(x_inv_rms, [PREFILL_ATTN_NORM_T_TILE, 1])
        for apply_db in pl.pipeline(D_BLOCKS, stage=2):
            apply_d0 = apply_db * D_CHUNK
            apply_x_chunk = pl.cast(x_flat[norm_tg : norm_tg + PREFILL_ATTN_NORM_T_TILE, apply_d0 : apply_d0 + D_CHUNK], target_type=pl.FP32)
            norm_w_chunk = pl.reshape(norm_w[apply_d0 : apply_d0 + D_CHUNK], [1, D_CHUNK])
            x_normed_chunk = pl.col_expand_mul(pl.row_expand_mul(apply_x_chunk, x_inv_rms_t), norm_w_chunk)
            x_normed_flat[norm_tg : norm_tg + PREFILL_ATTN_NORM_T_TILE, apply_d0 : apply_d0 + D_CHUNK] = pl.cast(
                        x_normed_chunk, target_type=pl.BF16, mode="rint"
            )

    x_normed = pl.reshape(x_normed_flat, [B, S, D])
    return x_normed


@pl.jit.inline
def prefill_qkv_proj_rope_core(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    kv:        pl.Tensor[[T, HEAD_DIM],    pl.BF16],
    qr:        pl.Tensor[[T, Q_LORA],      pl.INT8],
    qr_scale:  pl.Tensor[[T, 1],           pl.FP32],
    start_pos: pl.Scalar[pl.INT32],
):
    rope_cos_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)

    # Stage -1: materialize the absolute-position RoPE rows for this prefill
    # tile. The flattened token order is [batch, seq] -> [B*S], so every batch
    # row in this tile reuses the same contiguous [start_pos, start_pos + S)
    # frequency slice.
    for b0 in pl.range(0, B, PREFILL_ROPE_BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_batch_tile"):
            pos = pl.cast(start_pos, pl.INDEX)
            cos_rows = pl.slice(freqs_cos, [S, ROPE_DIM], [pos, 0])
            sin_rows = pl.slice(freqs_sin, [S, ROPE_DIM], [pos, 0])
            cos_tile = pl.full([PREFILL_ROPE_BATCH_TILE * S, ROPE_DIM], dtype=pl.BF16, value=0.0)
            sin_tile = pl.full([PREFILL_ROPE_BATCH_TILE * S, ROPE_DIM], dtype=pl.BF16, value=0.0)
            for bi in pl.range(PREFILL_ROPE_BATCH_TILE):
                tile_row = bi * S
                cos_tile = pl.assemble(cos_tile, cos_rows, [tile_row, 0])
                sin_tile = pl.assemble(sin_tile, sin_rows, [tile_row, 0])
            rope_offset = b0 * S
            rope_cos_t = pl.assemble(rope_cos_t, cos_tile, [rope_offset, 0])
            rope_sin_t = pl.assemble(rope_sin_t, sin_tile, [rope_offset, 0])

    x_flat = pl.reshape(x, [T, D])
    token_x_bf16 = x_flat

    qr_fp32 = pl.create_tensor([T, Q_LORA], dtype=pl.FP32)
    for qbg_idx in pl.spmd(Q_BLOCKS // QR_PROJ_GROUP, name_hint="prefill_qr_proj_matmul"):
        qbg = qbg_idx * QR_PROJ_GROUP
        for q_inner in pl.pipeline(QR_PROJ_GROUP, stage=2):
            q_a_col0 = (qbg + q_inner) * Q_LORA_CHUNK
            q_acc = pl.create_tensor([T, Q_LORA_CHUNK], dtype=pl.FP32)
            for db in pl.pipeline(0, D_BLOCKS, stage=2):
                qr_d0 = db * D_CHUNK
                q_x_chunk_bf16 = token_x_bf16[:, qr_d0 : qr_d0 + D_CHUNK]
                w_chunk = wq_a[qr_d0 : qr_d0 + D_CHUNK, q_a_col0 : q_a_col0 + Q_LORA_CHUNK]
                if qr_d0 == 0:
                    q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32[:, q_a_col0 : q_a_col0 + Q_LORA_CHUNK] = q_acc

    for qr_tg_idx in pl.spmd(T // PREFILL_ATTN_NORM_T_TILE, name_hint="prefill_qr_rms_norm_quant"):
        qr_tg = qr_tg_idx * PREFILL_ATTN_NORM_T_TILE
        qr_sq_sum = pl.full([1, PREFILL_ATTN_NORM_T_TILE], dtype=pl.FP32, value=0.0)
        qr_tile_amax = pl.full([1, PREFILL_ATTN_NORM_T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for qr_rms_qb in pl.pipeline(Q_BLOCKS, stage=2):
            qr_rms_col0 = qr_rms_qb * Q_LORA_CHUNK
            qr_rms_chunk = qr_fp32[qr_tg : qr_tg + PREFILL_ATTN_NORM_T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_CHUNK]
            qr_sq_sum = pl.add(
                qr_sq_sum,
                pl.reshape(pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk)), [1, PREFILL_ATTN_NORM_T_TILE]),
            )
        qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [PREFILL_ATTN_NORM_T_TILE, 1])

        for qb in pl.pipeline(Q_BLOCKS, stage=2):
            qr_norm_col0 = qb * Q_LORA_CHUNK
            qr_norm_chunk = qr_fp32[qr_tg : qr_tg + PREFILL_ATTN_NORM_T_TILE, qr_norm_col0 : qr_norm_col0 + Q_LORA_CHUNK]
            gamma_chunk = pl.reshape(
                pl.cast(gamma_cq[qr_norm_col0 : qr_norm_col0 + Q_LORA_CHUNK], target_type=pl.FP32),
                [1, Q_LORA_CHUNK],
            )
            qr_normed = pl.col_expand_mul(pl.row_expand_mul(qr_norm_chunk, qr_inv_rms_t), gamma_chunk)
            qr_normed_bf16 = pl.cast(qr_normed, target_type=pl.BF16, mode="rint")
            qr_norm_amax_f32 = pl.cast(qr_normed_bf16, target_type=pl.FP32)
            qr_norm_amax_abs = pl.maximum(qr_norm_amax_f32, pl.neg(qr_norm_amax_f32))
            qr_tile_amax = pl.maximum(
                qr_tile_amax,
                pl.reshape(pl.row_max(qr_norm_amax_abs), [1, PREFILL_ATTN_NORM_T_TILE]),
            )

        qr_scale_quant_row = pl.div(pl.full([1, PREFILL_ATTN_NORM_T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [PREFILL_ATTN_NORM_T_TILE, 1])
        qr_scale[qr_tg : qr_tg + PREFILL_ATTN_NORM_T_TILE, :] = pl.reshape(
            pl.recip(qr_scale_quant_row),
            [PREFILL_ATTN_NORM_T_TILE, 1],
        )

        for qa in pl.pipeline(0, Q_LORA, QUANT_CHUNK, stage=2):
            qr_chunk = qr_fp32[qr_tg : qr_tg + PREFILL_ATTN_NORM_T_TILE, qa : qa + QUANT_CHUNK]
            gamma_q_chunk = pl.reshape(
                pl.cast(gamma_cq[qa : qa + QUANT_CHUNK], target_type=pl.FP32),
                [1, QUANT_CHUNK],
            )
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
            qr_q_normed_bf16 = pl.cast(qr_q_normed, target_type=pl.BF16, mode="rint")
            qr_q_f32 = pl.cast(qr_q_normed_bf16, target_type=pl.FP32)
            qr_q_scaled = pl.row_expand_mul(qr_q_f32, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr[qr_tg : qr_tg + PREFILL_ATTN_NORM_T_TILE, qa : qa + QUANT_CHUNK] = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")

    q_proj_fp32 = pl.create_tensor([T, H * HEAD_DIM], dtype=pl.FP32)
    for hg_idx in pl.spmd(Q_PROJ_HEAD_BLOCKS // Q_PROJ_DEQUANT_GROUP, name_hint="prefill_qproj"):
        hg = hg_idx * Q_PROJ_DEQUANT_GROUP
        col_acc = pl.create_tensor([T, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)
        for h_inner in pl.pipeline(Q_PROJ_DEQUANT_GROUP, stage=2):
            for qb in pl.pipeline(0, Q_PROJ_BLOCKS, stage=2):
                qr_proj_col0 = qb * Q_PROJ_CHUNK
                qr_i8_chunk = qr[:, qr_proj_col0 : qr_proj_col0 + Q_PROJ_CHUNK]
                wq_chunk = wq_b[
                    qr_proj_col0 : qr_proj_col0 + Q_PROJ_CHUNK,
                    (hg + h_inner) * Q_PROJ_OUT_CHUNK : (hg + h_inner) * Q_PROJ_OUT_CHUNK + Q_PROJ_OUT_CHUNK,
                ]
                if qr_proj_col0 == 0:
                    col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                else:
                    col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
            w_col0 = (hg + h_inner) * Q_PROJ_OUT_CHUNK
            w_scale = pl.reshape(wq_b_scale[w_col0 : w_col0 + Q_PROJ_OUT_CHUNK], [1, Q_PROJ_OUT_CHUNK])
            for tc in pl.pipeline(0, T, QPROJ_T_TILE, stage=2):
                col_acc_t = col_acc[tc : tc + QPROJ_T_TILE, :]
                col_fp32 = pl.cast(col_acc_t, target_type=pl.FP32, mode="none")
                qr_scale_dq_t = qr_scale[tc : tc + QPROJ_T_TILE, :]
                col_dequant = pl.col_expand_mul(pl.row_expand_mul(col_fp32, qr_scale_dq_t), w_scale)
                q_proj_fp32[
                    tc : tc + QPROJ_T_TILE,
                    (hg + h_inner) * Q_PROJ_OUT_CHUNK : (hg + h_inner) * Q_PROJ_OUT_CHUNK + Q_PROJ_OUT_CHUNK,
                ] = col_dequant

    q_flat = pl.reshape(q, [T, H * HEAD_DIM])
    q_head_inv_rms_all = pl.create_tensor([H, T], dtype=pl.FP32)
    for hg_idx in pl.spmd(H // Q_HEAD_RMS_GROUP, name_hint="prefill_q_head_rms_nope"):
        hg = hg_idx * Q_HEAD_RMS_GROUP
        for h_inner in pl.range(Q_HEAD_RMS_GROUP):
            h = hg + h_inner
            h0 = h * HEAD_DIM
            q_head_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
            for db in pl.pipeline(HEAD_DIM // HEAD_CHUNK, stage=2):
                d0 = h0 + db * HEAD_CHUNK
                q_head_chunk = q_proj_fp32[:, d0 : d0 + HEAD_CHUNK]
                q_head_sq_sum = pl.add(q_head_sq_sum, pl.reshape(pl.row_sum(pl.mul(q_head_chunk, q_head_chunk)), [1, T]))
            q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
            q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [T, 1])
            q_head_inv_rms_all[h : h + 1, :] = q_head_inv_rms

            for nb in pl.pipeline(NOPE_DIM // HEAD_CHUNK, stage=2):
                n0 = nb * HEAD_CHUNK
                q_nope_chunk = q_proj_fp32[:, h0 + n0 : h0 + n0 + HEAD_CHUNK]
                q_normed = pl.row_expand_mul(q_nope_chunk, q_head_inv_rms_t)
                q_flat[:, h0 + n0 : h0 + n0 + HEAD_CHUNK] = pl.cast(q_normed, target_type=pl.BF16, mode="rint")

    q_rope_stage_fp32 = pl.create_tensor([H * T, ROPE_DIM], dtype=pl.FP32)
    for hg_idx in pl.spmd(H // Q_HEAD_RMS_GROUP, name_hint="prefill_q_head_rope_fused"):
        hg = hg_idx * Q_HEAD_RMS_GROUP
        for h_inner in pl.range(Q_HEAD_RMS_GROUP):
            h = hg + h_inner
            h0 = h * HEAD_DIM
            for q_rope_tg_idx in pl.range(T // Q_ROPE_T_TILE):
                q_rope_tg = q_rope_tg_idx * Q_ROPE_T_TILE
                q_rope_inv_rms_chunk = pl.reshape(
                    q_head_inv_rms_all[h : h + 1, q_rope_tg : q_rope_tg + Q_ROPE_T_TILE],
                    [Q_ROPE_T_TILE, 1],
                )
                q_rope_chunk = q_proj_fp32[q_rope_tg : q_rope_tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + NOPE_DIM + ROPE_DIM]
                q_rope_norm_chunk = pl.row_expand_mul(q_rope_chunk, q_rope_inv_rms_chunk)
                q_rope_even = pl.tensor.gather(q_rope_norm_chunk, mask_pattern=pl.tile.MaskPattern.P0101)
                q_rope_odd = pl.tensor.gather(q_rope_norm_chunk, mask_pattern=pl.tile.MaskPattern.P1010)
                q_rope_cos_chunk = pl.cast(rope_cos_t[q_rope_tg : q_rope_tg + Q_ROPE_T_TILE, :ROPE_HALF], target_type=pl.FP32)
                q_rope_sin_chunk = pl.cast(rope_sin_t[q_rope_tg : q_rope_tg + Q_ROPE_T_TILE, :ROPE_HALF], target_type=pl.FP32)
                q_rot_even = pl.sub(pl.mul(q_rope_even, q_rope_cos_chunk), pl.mul(q_rope_odd, q_rope_sin_chunk))
                q_rot_odd = pl.add(pl.mul(q_rope_even, q_rope_sin_chunk), pl.mul(q_rope_odd, q_rope_cos_chunk))
                q_rope_buf = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                q_rope_buf = pl.tensor.scatter(q_rot_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=q_rope_buf)
                q_rope_buf = pl.tensor.scatter(q_rot_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=q_rope_buf)
                q_rope_stage_fp32[h * T + q_rope_tg : h * T + q_rope_tg + Q_ROPE_T_TILE, :] = q_rope_buf

    for hg_idx in pl.spmd(H // HEAD_GROUP, name_hint="prefill_q_rope_writeback"):
        hg = hg_idx * HEAD_GROUP
        for h_inner in pl.pipeline(HEAD_GROUP, stage=2):
            h = hg + h_inner
            rot_fp32 = q_rope_stage_fp32[h * T : h * T + T, :]
            q_flat[:, h * HEAD_DIM + NOPE_DIM : h * HEAD_DIM + NOPE_DIM + ROPE_DIM] = pl.cast(
                rot_fp32, target_type=pl.BF16, mode="rint"
            )
    q = pl.reshape(q_flat, [T, H, HEAD_DIM])

    kv_fp32 = pl.create_tensor([T, HEAD_DIM], dtype=pl.FP32)
    for kbg_idx in pl.spmd(KV_BLOCKS // KV_PROJ_SPMD_GROUP, name_hint="prefill_kv_proj_matmul"):
        kbg = kbg_idx * KV_PROJ_SPMD_GROUP
        for k_inner in pl.pipeline(KV_PROJ_SPMD_GROUP, stage=2):
            kv_acc = pl.create_tensor([T, KV_CHUNK], dtype=pl.FP32)
            kv_col0 = (kbg + k_inner) * KV_CHUNK
            for db in pl.pipeline(0, D_BLOCKS, stage=2):
                d0 = db * D_CHUNK
                kv_x_chunk_bf16 = token_x_bf16[:, d0 : d0 + D_CHUNK]
                wkv_chunk = wkv[d0 : d0 + D_CHUNK, kv_col0 : kv_col0 + KV_CHUNK]
                if d0 == 0:
                    kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32[:, kv_col0 : kv_col0 + KV_CHUNK] = kv_acc

    kv_inv_rms_tensor = pl.create_tensor([1, T], dtype=pl.FP32)
    for kv_rms_tg_idx in pl.spmd(T // KV_RMS_T_TILE, name_hint="prefill_kv_rms_norm"):
        kv_rms_tg = kv_rms_tg_idx * KV_RMS_T_TILE
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(KV_BLOCKS, stage=2):
            kv_sq_col0 = kb * KV_CHUNK
            kv_chunk = kv_fp32[kv_rms_tg : kv_rms_tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_CHUNK]
            kv_sq_sum = pl.add(kv_sq_sum, pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, KV_RMS_T_TILE]))
        kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])
        kv_inv_rms_tensor[0:1, kv_rms_tg : kv_rms_tg + KV_RMS_T_TILE] = kv_inv_rms

        for nb in pl.pipeline(NOPE_DIM // KV_CHUNK, stage=2):
            n0 = nb * KV_CHUNK
            kv_chunk = kv_fp32[kv_rms_tg : kv_rms_tg + KV_RMS_T_TILE, n0 : n0 + KV_CHUNK]
            gamma_kv_chunk = pl.reshape(
                pl.cast(gamma_ckv[n0 : n0 + KV_CHUNK], target_type=pl.FP32),
                [1, KV_CHUNK],
            )
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv[kv_rms_tg : kv_rms_tg + KV_RMS_T_TILE, n0 : n0 + KV_CHUNK] = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_rope_fused"):
        gamma_rope = pl.reshape(
            pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32),
            [1, ROPE_DIM],
        )
        for kv_rope_tg_idx in pl.range(T // KV_ROPE_T_TILE):
            kv_rope_tg = kv_rope_tg_idx * KV_ROPE_T_TILE
            kv_rope_inv_rms_chunk = pl.reshape(
                kv_inv_rms_tensor[0:1, kv_rope_tg : kv_rope_tg + KV_ROPE_T_TILE],
                [KV_ROPE_T_TILE, 1],
            )
            kv_rope_chunk = kv_fp32[kv_rope_tg : kv_rope_tg + KV_ROPE_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
            kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_rope_inv_rms_chunk), gamma_rope)
            kv_rope_even = pl.tensor.gather(kv_rope_norm_chunk, mask_pattern=pl.tile.MaskPattern.P0101)
            kv_rope_odd = pl.tensor.gather(kv_rope_norm_chunk, mask_pattern=pl.tile.MaskPattern.P1010)
            kv_rope_cos_chunk = pl.cast(rope_cos_t[kv_rope_tg : kv_rope_tg + KV_ROPE_T_TILE, :ROPE_HALF], target_type=pl.FP32)
            kv_rope_sin_chunk = pl.cast(rope_sin_t[kv_rope_tg : kv_rope_tg + KV_ROPE_T_TILE, :ROPE_HALF], target_type=pl.FP32)
            kv_rot_even = pl.sub(pl.mul(kv_rope_even, kv_rope_cos_chunk), pl.mul(kv_rope_odd, kv_rope_sin_chunk))
            kv_rot_odd = pl.add(pl.mul(kv_rope_even, kv_rope_sin_chunk), pl.mul(kv_rope_odd, kv_rope_cos_chunk))
            kv_rope_buf = pl.full([KV_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            kv_rope_buf = pl.tensor.scatter(kv_rot_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=kv_rope_buf)
            kv_rope_buf = pl.tensor.scatter(kv_rot_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=kv_rope_buf)
            kv[kv_rope_tg : kv_rope_tg + KV_ROPE_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM] = pl.cast(
                kv_rope_buf,
                target_type=pl.BF16,
                mode="rint",
            )
    return q, kv, qr, qr_scale


@pl.jit
def prefill_qkv_proj_rope(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    norm_w:    pl.Tensor[[D],                    pl.FP32],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
    kv:        pl.Out[pl.Tensor[[T, HEAD_DIM],    pl.BF16]],
    qr:        pl.Out[pl.Tensor[[T, Q_LORA],      pl.INT8]],
    qr_scale:  pl.Out[pl.Tensor[[T, 1],           pl.FP32]],
    start_pos: pl.Scalar[pl.INT32],
):
    x_normed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    x_normed = prefill_attn_norm(x, norm_w, x_normed)
    q, kv, qr, qr_scale = prefill_qkv_proj_rope_core(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        start_pos,
    )
    return q, kv, qr, qr_scale


