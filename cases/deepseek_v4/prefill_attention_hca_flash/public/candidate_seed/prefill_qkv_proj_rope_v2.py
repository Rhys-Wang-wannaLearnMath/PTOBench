from __future__ import annotations

import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS
T_DYN = pl.dynamic('T_DYN')
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings
Q_PROJ_OUT_TILE = 128
Q_PROJ_TILE = 128
QPROJ_MM_N_TILE = 1024
Q_LORA_TILE = 256
KV_TILE = 64
QUANT_TILE = 256
T_TILE = 8
MATMUL_T_TILE = 16
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)
QR_M_TILE = MATMUL_T_TILE
QR_N_TILE = 128
QR_K_TILE = 256
QR_OK = 2
QR_K_SLICE = D // QR_OK
KV_M_TILE = MATMUL_T_TILE
KV_N_TILE = 128
KV_K_TILE = 256
KV_OK = 4
KV_K_SLICE = D // KV_OK
QPROJ_M_TILE = MATMUL_T_TILE
DEQUANT_T_TILE = 8
KV_RMS_T_TILE = 8
Q_ROPE_T_TILE = 8

@pl.jit.inline
def materialize_rope_rows(freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16], position_ids: pl.Tensor[[T_DYN], pl.INT32], num_tokens: pl.Scalar[pl.INT32], rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16]):
    t_dim = pl.tensor.dim(position_ids, 0)
    for rope_t0 in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint='qkv_rope_rows'):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < num_tokens:
                rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
                rope_cos_t[rope_t:rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos:rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t:rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos:rope_pos + 1, 0:ROPE_DIM]

@pl.jit.inline
def qkv_proj_rope(x: pl.Tensor[[T_DYN, D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16], kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16], qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8], qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32]):
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    t_matmul = pl.max(t_dim, MATMUL_T_TILE)
    qr_fp32 = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    qr_i8_matmul = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.INT8)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='qr_proj_seed'):
        for tc in pl.range(t_matmul // QR_M_TILE):
            ts0 = tc * QR_M_TILE
            for nb in pl.range(Q_LORA // QR_N_TILE):
                nseed0 = nb * QR_N_TILE
                qr_fp32[ts0:ts0 + QR_M_TILE, nseed0:nseed0 + QR_N_TILE] = pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0)
    for qbg_idx in pl.spmd(Q_LORA // QR_N_TILE * QR_OK, name_hint='qr_proj_matmul'):
        q_a_col0 = qbg_idx // QR_OK * QR_N_TILE
        qr_k_base = qbg_idx % QR_OK * QR_K_SLICE
        for tc in pl.range(t_matmul // QR_M_TILE):
            t0 = tc * QR_M_TILE
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_K_SLICE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, t_dim - t0)
                q_x_chunk_bf16 = pl.slice(x_view, [QR_M_TILE, QR_K_TILE], [t0, qr_d0], valid_shape=[qr_rows, QR_K_TILE])
                w_chunk = wq_a[qr_d0:qr_d0 + QR_K_TILE, q_a_col0:q_a_col0 + QR_N_TILE]
                if db == 0:
                    q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=pl.AtomicType.Add)
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint='qr_rms_norm_quant', allow_early_resolve=True):
        tg = tg_idx * T_TILE
        qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        qr_amax_g = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for qr_rms_qb in pl.pipeline(Q_LORA // Q_LORA_TILE, stage=2):
            qr_rms_col0 = qr_rms_qb * Q_LORA_TILE
            qr_rms_chunk = qr_fp32[tg:tg + T_TILE, qr_rms_col0:qr_rms_col0 + Q_LORA_TILE]
            qr_sq_sum = pl.add(qr_sq_sum, pl.reshape(pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk)), [1, T_TILE]))
            gamma_rms_cast = pl.cast(gamma_cq[qr_rms_col0:qr_rms_col0 + Q_LORA_TILE], target_type=pl.FP32)
            gamma_rms_chunk = pl.reshape(gamma_rms_cast, [1, Q_LORA_TILE])
            qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
            qr_g_abs = pl.maximum(qr_g, pl.neg(qr_g))
            qr_amax_g = pl.maximum(qr_amax_g, pl.reshape(pl.row_max(qr_g_abs), [1, T_TILE]))
        qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
        qr_amax_floor = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qr_amax_normed = pl.mul(qr_inv_rms, qr_amax_g)
        qr_tile_amax = pl.maximum(qr_amax_floor, qr_amax_normed)
        qr_scale_quant_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
        qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
        qr_scale_view[tg:tg + T_TILE, :] = qr_tile_scale_dq
        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = qr_fp32[tg:tg + T_TILE, qa:qa + QUANT_TILE]
            gamma_q_cast = pl.cast(gamma_cq[qa:qa + QUANT_TILE], target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode='rint')
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode='round')
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode='trunc')
            qr_view[tg:tg + T_TILE, qa:qa + QUANT_TILE] = qr_q_i8
            qr_i8_matmul[tg:tg + T_TILE, qa:qa + QUANT_TILE] = qr_q_i8
    q_proj_fp32 = pl.create_tensor([T_MAX, H * HEAD_DIM], dtype=pl.FP32)
    q_proj_i32 = pl.create_tensor([T_MAX, H * HEAD_DIM], dtype=pl.INT32)
    for hg_idx in pl.spmd(H * HEAD_DIM // QPROJ_MM_N_TILE // 2, name_hint='qproj_matmul'):
        hg = hg_idx * 2
        for h_inner in pl.range(2):
            w_col0 = (hg + h_inner) * QPROJ_MM_N_TILE
            for tc in pl.range(t_matmul // QPROJ_M_TILE):
                t0 = tc * QPROJ_M_TILE
                col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
                for qb in pl.pipeline(0, Q_LORA // Q_PROJ_TILE, stage=2):
                    qr_proj_col0 = qb * Q_PROJ_TILE
                    qr_i8_chunk = qr_i8_matmul[t0:t0 + QPROJ_M_TILE, qr_proj_col0:qr_proj_col0 + Q_PROJ_TILE]
                    wq_chunk = wq_b[qr_proj_col0:qr_proj_col0 + Q_PROJ_TILE, w_col0:w_col0 + QPROJ_MM_N_TILE]
                    if qr_proj_col0 == 0:
                        col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                    else:
                        col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
                q_proj_i32[t0:t0 + QPROJ_M_TILE, w_col0:w_col0 + QPROJ_MM_N_TILE] = col_acc
    for hg_idx in pl.spmd(H * HEAD_DIM // Q_PROJ_OUT_TILE // 16, name_hint='qproj_dequant'):
        hg = hg_idx * 16
        for h_inner in pl.range(16):
            w_col0 = (hg + h_inner) * Q_PROJ_OUT_TILE
            w_scale = pl.reshape(wq_b_scale[w_col0:w_col0 + Q_PROJ_OUT_TILE], [1, Q_PROJ_OUT_TILE])
            for tc in pl.pipeline(0, t_dim, DEQUANT_T_TILE, stage=2):
                col_acc_t = q_proj_i32[tc:tc + DEQUANT_T_TILE, w_col0:w_col0 + Q_PROJ_OUT_TILE]
                col_fp32 = pl.cast(col_acc_t, target_type=pl.FP32, mode='none')
                qr_scale_dq_t = qr_scale_view[tc:tc + DEQUANT_T_TILE, :]
                col_dequant = pl.col_expand_mul(pl.row_expand_mul(col_fp32, qr_scale_dq_t), w_scale)
                q_proj_fp32[tc:tc + DEQUANT_T_TILE, w_col0:w_col0 + Q_PROJ_OUT_TILE] = col_dequant
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for hg_idx in pl.spmd(H // 2, name_hint='q_head_rms_nope_rope', allow_early_resolve=True):
        hg = hg_idx * 2
        q_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        q_col = pl.col_expand_mul(q_ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        q_dup_f = pl.cast(pl.cast(pl.mul(q_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        q_dup_idx = pl.cast(q_dup_f, target_type=pl.INT32)
        q_lane = pl.sub(q_col, pl.mul(q_dup_f, 2.0))
        q_swap_idx = pl.cast(pl.sub(pl.add(q_col, 1.0), pl.mul(q_lane, 2.0)), target_type=pl.INT32)
        q_sign = pl.sub(pl.mul(q_lane, 2.0), 1.0)
        for tg_idx in pl.range(t_dim // Q_ROPE_T_TILE):
            tg = tg_idx * Q_ROPE_T_TILE
            q_cos_il = pl.gather(pl.cast(rope_cos_view[tg:tg + Q_ROPE_T_TILE, :], target_type=pl.FP32), dim=-1, index=q_dup_idx)
            q_sin_il = pl.gather(pl.cast(rope_sin_view[tg:tg + Q_ROPE_T_TILE, :], target_type=pl.FP32), dim=-1, index=q_dup_idx)
            for h_inner in pl.range(2):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_nope_full = q_proj_fp32[tg:tg + Q_ROPE_T_TILE, h0:h0 + NOPE_DIM]
                q_rope_chunk = q_proj_fp32[tg:tg + Q_ROPE_T_TILE, h0 + NOPE_DIM:h0 + HEAD_DIM]
                q_head_sq_sum = pl.add(pl.reshape(pl.row_sum(pl.mul(q_nope_full, q_nope_full)), [1, Q_ROPE_T_TILE]), pl.reshape(pl.row_sum(pl.mul(q_rope_chunk, q_rope_chunk)), [1, Q_ROPE_T_TILE]))
                q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])
                q_normed = pl.row_expand_mul(q_nope_full, q_head_inv_rms_t)
                q_flat[tg:tg + Q_ROPE_T_TILE, h0:h0 + NOPE_DIM] = pl.cast(q_normed, target_type=pl.BF16, mode='rint')
                q_rope_swapped = pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)
                q_rope_rot = pl.add(pl.mul(q_rope_chunk, q_cos_il), pl.mul(pl.mul(q_rope_swapped, q_sign), q_sin_il))
                q_flat[tg:tg + Q_ROPE_T_TILE, h0 + NOPE_DIM:h0 + NOPE_DIM + ROPE_DIM] = pl.cast(pl.row_expand_mul(q_rope_rot, q_head_inv_rms_t), target_type=pl.BF16, mode='rint')
    kv_fp32 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='kv_proj_seed'):
        for tc in pl.range(t_matmul // KV_M_TILE):
            kts0 = tc * KV_M_TILE
            for nb in pl.range(HEAD_DIM // KV_N_TILE):
                kvseed0 = nb * KV_N_TILE
                kv_fp32[kts0:kts0 + KV_M_TILE, kvseed0:kvseed0 + KV_N_TILE] = pl.full([KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0)
    for kbg in pl.spmd(HEAD_DIM // KV_N_TILE * KV_OK, name_hint='kv_proj_matmul'):
        kv_col0 = kbg // KV_OK * KV_N_TILE
        kv_k_base = kbg % KV_OK * KV_K_SLICE
        for tc in pl.range(t_matmul // KV_M_TILE):
            t0 = tc * KV_M_TILE
            kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(KV_K_SLICE // KV_K_TILE, stage=2):
                d0 = kv_k_base + db * KV_K_TILE
                kv_rows = pl.min(KV_M_TILE, t_dim - t0)
                kv_x_chunk_bf16 = pl.slice(x_view, [KV_M_TILE, KV_K_TILE], [t0, d0], valid_shape=[kv_rows, KV_K_TILE])
                wkv_chunk = wkv[d0:d0 + KV_K_TILE, kv_col0:kv_col0 + KV_N_TILE]
                if db == 0:
                    kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [t0, kv_col0], atomic=pl.AtomicType.Add)
    for tg_idx in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint='kv_rms_norm_rope'):
        tg = tg_idx * KV_RMS_T_TILE
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HEAD_DIM // KV_TILE, stage=2):
            kv_sq_col0 = kb * KV_TILE
            kv_chunk = kv_fp32[tg:tg + KV_RMS_T_TILE, kv_sq_col0:kv_sq_col0 + KV_TILE]
            kv_sq_sum = pl.add(kv_sq_sum, pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, KV_RMS_T_TILE]))
        kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])
        for nb in pl.pipeline(NOPE_DIM // KV_TILE, stage=2):
            n0 = nb * KV_TILE
            kv_chunk = kv_fp32[tg:tg + KV_RMS_T_TILE, n0:n0 + KV_TILE]
            gamma_kv_cast = pl.cast(gamma_ckv[n0:n0 + KV_TILE], target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv_view[tg:tg + KV_RMS_T_TILE, n0:n0 + KV_TILE] = pl.cast(kv_normed, target_type=pl.BF16, mode='rint')
        gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM:NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = kv_fp32[tg:tg + KV_RMS_T_TILE, NOPE_DIM:NOPE_DIM + ROPE_DIM]
        kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
        kv_ones = pl.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        kv_col = pl.col_expand_mul(kv_ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        kv_dup_idx = pl.cast(kv_dup_f, target_type=pl.INT32)
        kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))
        kv_swap_idx = pl.cast(pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0)), target_type=pl.INT32)
        kv_sign = pl.sub(pl.mul(kv_lane, 2.0), 1.0)
        kv_cos_il = pl.gather(pl.cast(rope_cos_view[tg:tg + KV_RMS_T_TILE, :], target_type=pl.FP32), dim=-1, index=kv_dup_idx)
        kv_sin_il = pl.gather(pl.cast(rope_sin_view[tg:tg + KV_RMS_T_TILE, :], target_type=pl.FP32), dim=-1, index=kv_dup_idx)
        kv_swapped = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx)
        kv_rope_rot = pl.add(pl.mul(kv_rope_norm_chunk, kv_cos_il), pl.mul(pl.mul(kv_swapped, kv_sign), kv_sin_il))
        kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode='rint')
        kv_view[tg:tg + KV_RMS_T_TILE, NOPE_DIM:NOPE_DIM + ROPE_DIM] = kv_rope_i16
    return q
