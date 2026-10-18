# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 single-token decode attn_norm fused + Q/KV LoRA + RoPE: produces (q, kv, qr) for the
attention body, with attn_norm fused at the front to save one GM round-trip."""


import pypto.language as pl


from config import (
    B,
    D,
    EPS,
    H,
    HEAD_DIM,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    NOPE_DIM,
    Q_LORA,
    Q_PROJ_HEAD_BLOCKS,
    Q_PROJ_OUT_CHUNK,
    ROPE_DIM,
    ROPE_HALF,
    S,
    T,
)


# Editable implementation decomposition.
ROPE_CHUNK = 32
ROPE_PAIR_CHUNK = ROPE_CHUNK // 2
HEAD_CHUNK = 64
Q_LORA_TILE = 32
Q_PROJ_CHUNK = 128
Q_BLOCKS = Q_LORA // Q_LORA_TILE
Q_PROJ_BLOCKS = Q_LORA // Q_PROJ_CHUNK
HEAD_GROUP = 8
assert (H * HEAD_DIM) % (HEAD_CHUNK * HEAD_GROUP) == 0
HEAD_BLOCKS = (H * HEAD_DIM) // HEAD_CHUNK
HEAD_GROUP_BLOCKS = (H * HEAD_DIM) // (HEAD_CHUNK * HEAD_GROUP)
D_CHUNK = 512
Q_LORA_CHUNK = Q_LORA_TILE
KV_CHUNK = 32
D_BLOCKS = D // D_CHUNK
KV_BLOCKS = HEAD_DIM // KV_CHUNK
QUANT_CHUNK = 256


@pl.jit.inline
def deepseek_v4_decode_qkv_proj_rope(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    norm_w:    pl.Tensor[[D],                    pl.FP32],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    rope_cos:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
    rope_sin:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t:  pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Tensor[[T, H, HEAD_DIM],        pl.BF16],
    kv:        pl.Tensor[[T, HEAD_DIM],           pl.BF16],
    qr:        pl.Tensor[[T, Q_LORA],             pl.INT8],
    qr_scale:  pl.Tensor[[T, 1],                  pl.FP32],
):
    x_flat = pl.reshape(x, [T, D])

    # Stage 0.1: fused attn_norm -> token_x_fp32
    token_x_fp32 = pl.create_tensor([T, D], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        x_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for db in pl.range(D_BLOCKS):
            d0 = db * D_CHUNK
            x_chunk = pl.cast(pl.slice(x_flat, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
            x_sq_sum = pl.add(x_sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T]))
        x_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS)))

    x_inv_rms_t = pl.reshape(x_inv_rms, [T, 1])
    for db in pl.parallel(0, D_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP):
            d0 = db * D_CHUNK
            x_chunk = pl.cast(pl.slice(x_flat, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
            norm_w_chunk = pl.reshape(pl.slice(norm_w, [D_CHUNK], [d0]), [1, D_CHUNK])
            x_normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, x_inv_rms_t), norm_w_chunk)
            token_x_fp32 = pl.assemble(token_x_fp32, x_normed, [0, d0])

    # Stage 0.2: pre-cast token_x for split AIV->AIC flow.
    token_x_bf16 = pl.create_tensor([T, D], dtype=pl.BF16)
    for db in pl.parallel(0, D_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP):
            d0 = db * D_CHUNK
            x_chunk_fp32 = pl.slice(token_x_fp32, [T, D_CHUNK], [0, d0])
            token_x_bf16 = pl.assemble(token_x_bf16, pl.cast(x_chunk_fp32, target_type=pl.BF16), [0, d0])

    # Stage 1/2.1: qr = rms_norm(token_x @ wq_a, gamma_cq)
    qr_fp32 = pl.create_tensor([T, Q_LORA], dtype=pl.FP32)
    for qb in pl.parallel(0, Q_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP):
            q_a_col0 = qb * Q_LORA_CHUNK
            d0_0 = 0
            x_chunk_bf16_0 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0_0])
            w_chunk_0 = pl.slice(wq_a, [D_CHUNK, Q_LORA_CHUNK], [d0_0, q_a_col0])
            q_acc = pl.matmul(x_chunk_bf16_0, w_chunk_0, out_dtype=pl.FP32)
            for db in pl.range(1, D_BLOCKS):
                d0 = db * D_CHUNK
                q_x_chunk_bf16 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0])
                w_chunk = pl.slice(wq_a, [D_CHUNK, Q_LORA_CHUNK], [d0, q_a_col0])
                q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [0, q_a_col0])

    with pl.at(level=pl.Level.CORE_GROUP):
        qr_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for qb in pl.range(Q_BLOCKS):
            qr_sq_col0 = qb * Q_LORA_CHUNK
            qr_chunk = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, qr_sq_col0])
            qr_sq_sum = pl.add(qr_sq_sum, pl.reshape(pl.row_sum(pl.mul(qr_chunk, qr_chunk)), [1, T]))
        qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))

    qr_inv_rms_t = pl.reshape(qr_inv_rms, [T, 1])
    for qb in pl.parallel(0, Q_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP):
            qr_norm_col0 = qb * Q_LORA_CHUNK
            qr_chunk = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, qr_norm_col0])
            gamma_chunk = pl.reshape(
                pl.cast(pl.slice(gamma_cq, [Q_LORA_CHUNK], [qr_norm_col0]), target_type=pl.FP32),
                [1, Q_LORA_CHUNK],
            )
            qr_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_chunk)
            qr_fp32 = pl.assemble(qr_fp32, qr_normed, [0, qr_norm_col0])

    # Stage 2.2: pre-cast normalized qr for the W8A8 dynamic activation path.
    qr_bf16 = pl.create_tensor([T, Q_LORA], dtype=pl.BF16)
    for qb in pl.parallel(0, Q_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP):
            qr_store_col0 = qb * Q_LORA_CHUNK
            qr_chunk_fp32 = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, qr_store_col0])
            qr_bf16 = pl.assemble(qr_bf16, pl.cast(qr_chunk_fp32, target_type=pl.BF16), [0, qr_store_col0])

    # Stage 2.3: W8A8C16 activation path: quantize normalized qr per token.
    qr_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        qr_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for q0 in pl.range(0, Q_LORA, QUANT_CHUNK):
            qr_a_f32 = pl.cast(pl.slice(qr_bf16, [T, QUANT_CHUNK], [0, q0]), target_type=pl.FP32)
            qr_a_abs = pl.maximum(qr_a_f32, pl.neg(qr_a_f32))
            qr_a_max = pl.reshape(pl.row_max(qr_a_abs), [1, T])
            qr_amax = pl.maximum(qr_amax, qr_a_max)
        qr_scale_quant_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_amax)
        qr_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T, 1])
        qr_scale = pl.assemble(qr_scale, qr_scale_dq, [0, 0])
        qr_scale_quant = pl.reshape(qr_scale_quant_row, [T, 1])
        for q1 in pl.range(0, Q_LORA, QUANT_CHUNK):
            qr_q_f32 = pl.cast(pl.slice(qr_bf16, [T, QUANT_CHUNK], [0, q1]), target_type=pl.FP32)
            qr_q_scaled = pl.row_expand_mul(qr_q_f32, qr_scale_quant)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="round")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr = pl.assemble(qr, pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc"), [0, q1])

    # Stage 3: W8A8C16 q_proj = qr_i8 @ wq_b, then dequantize to FP32.
    q_proj_fp32 = pl.create_tensor([T, H * HEAD_DIM], dtype=pl.FP32)
    for hb in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, 1):
        h0 = hb * Q_PROJ_OUT_CHUNK
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qproj_matmul"):
            q0_0 = 0
            qr_i8_0 = pl.slice(qr, [T, Q_PROJ_CHUNK], [0, q0_0])
            wq_0 = pl.slice(wq_b, [Q_PROJ_CHUNK, Q_PROJ_OUT_CHUNK], [q0_0, h0])
            col_acc = pl.matmul(qr_i8_0, wq_0, out_dtype=pl.INT32)
            for qb in pl.range(1, Q_PROJ_BLOCKS):
                qr_proj_col0 = qb * Q_PROJ_CHUNK
                qr_i8_chunk = pl.slice(qr, [T, Q_PROJ_CHUNK], [0, qr_proj_col0])
                wq_chunk = pl.slice(wq_b, [Q_PROJ_CHUNK, Q_PROJ_OUT_CHUNK], [qr_proj_col0, h0])
                col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qproj_dequant"):
            col_fp32 = pl.cast(col_acc, target_type=pl.FP32, mode="none")
            w_scale = pl.slice(wq_b_scale, [1, Q_PROJ_OUT_CHUNK], [hb, 0])
            col_dequant = pl.col_expand_mul(pl.row_expand_mul(col_fp32, qr_scale_dq), w_scale)
            q_proj_fp32 = pl.assemble(q_proj_fp32, col_dequant, [0, h0])

    # Stage 4: per-head RMSNorm + RoPE on q
    q_flat = pl.reshape(q, [T, H * HEAD_DIM])
    for h in pl.parallel(0, H, 1):
        with pl.at(level=pl.Level.CORE_GROUP):
            h0 = h * HEAD_DIM
            q_head_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
            for db in pl.range(HEAD_DIM // HEAD_CHUNK):
                d0 = h0 + db * HEAD_CHUNK
                q_head_chunk = pl.slice(q_proj_fp32, [T, HEAD_CHUNK], [0, d0])
                q_head_sq_sum = pl.add(q_head_sq_sum, pl.reshape(pl.row_sum(pl.mul(q_head_chunk, q_head_chunk)), [1, T]))
            q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
            q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [T, 1])

            for nb in pl.range(NOPE_DIM // HEAD_CHUNK):
                n0 = nb * HEAD_CHUNK
                q_nope_chunk = pl.slice(q_proj_fp32, [T, HEAD_CHUNK], [0, h0 + n0])
                q_normed = pl.row_expand_mul(q_nope_chunk, q_head_inv_rms_t)
                q_flat = pl.assemble(q_flat, pl.cast(q_normed, target_type=pl.BF16), [0, h0 + n0])

            q_rope = pl.slice(q_proj_fp32, [T, ROPE_DIM], [0, h0 + NOPE_DIM])
            q_rope_norm = pl.row_expand_mul(q_rope, q_head_inv_rms_t)
            q_even = pl.tensor.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
            q_odd = pl.tensor.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
            cos = pl.cast(pl.slice(rope_cos, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
            sin = pl.cast(pl.slice(rope_sin, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
            q_rot_even = pl.sub(pl.mul(q_even, cos), pl.mul(q_odd, sin))
            q_rot_odd = pl.add(pl.mul(q_even, sin), pl.mul(q_odd, cos))
            q_rot_even_bf16 = pl.cast(q_rot_even, target_type=pl.BF16)
            q_rot_odd_bf16 = pl.cast(q_rot_odd, target_type=pl.BF16)

        for rope_col in pl.range(0, ROPE_DIM, ROPE_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_rope_reassemble"):
                pair_col = rope_col // 2
                q_rot_even_chunk = pl.slice(q_rot_even_bf16, [T, ROPE_PAIR_CHUNK], [0, pair_col])
                q_rot_odd_chunk = pl.slice(q_rot_odd_bf16, [T, ROPE_PAIR_CHUNK], [0, pair_col])
                q_rot_chunk = pl.matmul(
                    q_rot_even_chunk,
                    pl.slice(even_select_t, [ROPE_PAIR_CHUNK, ROPE_CHUNK], [pair_col, rope_col]),
                    out_dtype=pl.FP32,
                )
                q_rot_chunk = pl.matmul_acc(
                    q_rot_chunk,
                    q_rot_odd_chunk,
                    pl.slice(odd_select_t, [ROPE_PAIR_CHUNK, ROPE_CHUNK], [pair_col, rope_col]),
                )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_rope_write"):
                h0 = h * HEAD_DIM
                q_flat = pl.assemble(q_flat, pl.cast(q_rot_chunk, target_type=pl.BF16), [0, h0 + NOPE_DIM + rope_col])

    q = pl.reshape(q_flat, [T, H, HEAD_DIM])

    # Stage 5/6: kv = rms_norm(token_x @ wkv, gamma_ckv) + RoPE
    kv_fp32 = pl.create_tensor([T, HEAD_DIM], dtype=pl.FP32)
    for kb in pl.parallel(0, KV_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP):
            kv_col0 = kb * KV_CHUNK
            d0_0 = 0
            x_chunk_bf16_0 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0_0])
            wkv_chunk_0 = pl.slice(wkv, [D_CHUNK, KV_CHUNK], [d0_0, kv_col0])
            kv_acc = pl.matmul(x_chunk_bf16_0, wkv_chunk_0, out_dtype=pl.FP32)
            for db in pl.range(1, D_BLOCKS):
                d0 = db * D_CHUNK
                kv_x_chunk_bf16 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0])
                wkv_chunk = pl.slice(wkv, [D_CHUNK, KV_CHUNK], [d0, kv_col0])
                kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [0, kv_col0])

    with pl.at(level=pl.Level.CORE_GROUP):
        kv_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for kb in pl.range(KV_BLOCKS):
            kv_sq_col0 = kb * KV_CHUNK
            kv_chunk = pl.slice(kv_fp32, [T, KV_CHUNK], [0, kv_sq_col0])
            kv_sq_sum = pl.add(kv_sq_sum, pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, T]))
        kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))

    kv_inv_rms_t = pl.reshape(kv_inv_rms, [T, 1])
    for nb in pl.parallel(0, NOPE_DIM // KV_CHUNK, 1):
        with pl.at(level=pl.Level.CORE_GROUP):
            n0 = nb * KV_CHUNK
            kv_chunk = pl.slice(kv_fp32, [T, KV_CHUNK], [0, n0])
            gamma_kv_chunk = pl.reshape(
                pl.cast(pl.slice(gamma_ckv, [KV_CHUNK], [n0]), target_type=pl.FP32),
                [1, KV_CHUNK],
            )
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv = pl.assemble(kv, pl.cast(kv_normed, target_type=pl.BF16), [0, n0])
    kv_rot_even_tmp = pl.create_tensor([T, ROPE_HALF], dtype=pl.BF16)
    kv_rot_odd_tmp = pl.create_tensor([T, ROPE_HALF], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP):
        kv_rope = pl.slice(kv_fp32, [T, ROPE_DIM], [0, NOPE_DIM])
        gamma_rope = pl.reshape(
            pl.cast(pl.slice(gamma_ckv, [ROPE_DIM], [NOPE_DIM]), target_type=pl.FP32),
            [1, ROPE_DIM],
        )
        kv_rope_norm = pl.col_expand_mul(pl.row_expand_mul(kv_rope, kv_inv_rms_t), gamma_rope)
        kv_even = pl.tensor.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
        kv_odd = pl.tensor.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
        cos = pl.cast(pl.slice(rope_cos, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
        sin = pl.cast(pl.slice(rope_sin, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
        kv_rot_even = pl.sub(pl.mul(kv_even, cos), pl.mul(kv_odd, sin))
        kv_rot_odd = pl.add(pl.mul(kv_even, sin), pl.mul(kv_odd, cos))
        kv_rot_even_tmp = pl.assemble(kv_rot_even_tmp, pl.cast(kv_rot_even, target_type=pl.BF16), [0, 0])
        kv_rot_odd_tmp = pl.assemble(kv_rot_odd_tmp, pl.cast(kv_rot_odd, target_type=pl.BF16), [0, 0])

    for rope_col in pl.range(0, ROPE_DIM, ROPE_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_reassemble"):
            pair_col = rope_col // 2
            kv_rot_even_chunk = pl.slice(kv_rot_even_tmp, [T, ROPE_PAIR_CHUNK], [0, pair_col])
            kv_rot_odd_chunk = pl.slice(kv_rot_odd_tmp, [T, ROPE_PAIR_CHUNK], [0, pair_col])
            kv_rot_chunk = pl.matmul(
                kv_rot_even_chunk,
                pl.slice(even_select_t, [ROPE_PAIR_CHUNK, ROPE_CHUNK], [pair_col, rope_col]),
                out_dtype=pl.FP32,
            )
            kv_rot_chunk = pl.matmul_acc(
                kv_rot_chunk,
                kv_rot_odd_chunk,
                pl.slice(odd_select_t, [ROPE_PAIR_CHUNK, ROPE_CHUNK], [pair_col, rope_col]),
            )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_write"):
            kv = pl.assemble(kv, pl.cast(kv_rot_chunk, target_type=pl.BF16), [0, NOPE_DIM + rope_col])

    return q


@pl.jit
def deepseek_v4_decode_qkv_proj_rope_test(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    norm_w:    pl.Tensor[[D],                    pl.FP32],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    rope_cos:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
    rope_sin:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t:  pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
    kv:        pl.Out[pl.Tensor[[T, HEAD_DIM],    pl.BF16]],
    qr:        pl.Out[pl.Tensor[[T, Q_LORA],      pl.INT8]],
    qr_scale:  pl.Out[pl.Tensor[[T, 1],           pl.FP32]],
):
    q = deepseek_v4_decode_qkv_proj_rope(
        x,
        norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos,
        rope_sin,
        even_select_t,
        odd_select_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )
    return q



