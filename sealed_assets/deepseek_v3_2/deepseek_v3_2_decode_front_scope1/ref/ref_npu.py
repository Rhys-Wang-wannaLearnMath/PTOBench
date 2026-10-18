# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer decode front path (batch=16).

Projection stage:
- Compute RMSNorm of hidden_states
- Project to Q latent (qr) via wq_a
- Apply q_norm to Q latent, then project to Q heads via wq_b
- Project to KV latent (kv_a) via wkv_a

Decode cache preparation:
- Split Q heads into q_nope and q_pe outputs
- Apply RoPE to q_pe and k_pe
- Apply kv_norm to KV latent, then update KV/PE cache for the current token

Aligned to official v3.2-exp MLA shapes:
- qk_nope_head_dim = 128
- qk_rope_head_dim = 64
- kv_lora_rank = 512
"""


import pypto.language as pl


from config import (BATCH, MAX_SEQ, HIDDEN, NUM_HEADS, Q_LORA_RANK,
                    KV_LORA_RANK, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM,
                    QK_HEAD_DIM, KV_A_OUT, CACHE_ROWS, EPS, HIDDEN_INV, Q_LORA_INV)

# Tile sizes tuned for the standalone front projection/cache-prep example:
# - PROJ_K = K-dimension chunk for projection matmuls (kept at 512).
# - LORA_CHUNK, KV_OUT_CHUNK = 64 so AIC Right buffer ≤ 65536
#   (512 * 64 * 2 = 65536).
# - Q_OUT_CHUNK = 64 for the same reason on the wq_b side
#   (LORA_CHUNK * Q_OUT_CHUNK * 2 = 64 * 64 * 2 = 8192).
# - LOCAL_PAD_WIDTH removed; the pad tensor was a tuning hint for the
#   combined front pipeline and is not needed here.
RMSNORM_K = 512
PROJ_K = 512
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
LORA_CHUNK = 64

RMSNORM_BLOCKS = (HIDDEN + RMSNORM_K - 1) // RMSNORM_K
PROJ_BLOCKS = (HIDDEN + PROJ_K - 1) // PROJ_K
QR_BLOCKS = (Q_LORA_RANK + LORA_CHUNK - 1) // LORA_CHUNK
Q_OUT_BLOCKS = (NUM_HEADS * QK_HEAD_DIM + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
KV_A_BLOCKS = (KV_A_OUT + KV_OUT_CHUNK - 1) // KV_OUT_CHUNK


def build_deepseek_v3_2_decode_front_scope1_program():
    @pl.program
    class DeepSeekV32DecodeFrontScope1:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope1(
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            wq_a: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
            q_norm_weight: pl.Tensor[[1, Q_LORA_RANK], pl.FP32],
            wq_b: pl.Tensor[[Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM], pl.BF16],
            wkv_a: pl.Tensor[[HIDDEN, KV_A_OUT], pl.BF16],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            kv_norm_weight: pl.Tensor[[1, KV_LORA_RANK], pl.FP32],
            # Output buffers
            qr_out: pl.Out[pl.Tensor[[BATCH, Q_LORA_RANK], pl.BF16]],
            q_nope_out: pl.Out[pl.Tensor[[BATCH, NUM_HEADS * QK_NOPE_HEAD_DIM], pl.BF16]],
            q_pe_out: pl.Out[pl.Tensor[[BATCH, NUM_HEADS * QK_ROPE_HEAD_DIM], pl.BF16]],
            kv_a_out: pl.Out[pl.Tensor[[BATCH, KV_A_OUT], pl.BF16]],
            kv_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, KV_LORA_RANK], pl.BF16]],
            pe_cache: pl.InOut[pl.Tensor[[CACHE_ROWS, QK_ROPE_HEAD_DIM], pl.BF16]],
        ) -> pl.Tensor[[BATCH, NUM_HEADS * QK_ROPE_HEAD_DIM], pl.BF16]:
            q_proj_out = pl.create_tensor([BATCH, NUM_HEADS * QK_HEAD_DIM], dtype=pl.BF16)

            # Front projection: input RMSNorm + Q/KV projections.
            normed_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            qr_fp32_tile = pl.create_tensor([BATCH, Q_LORA_RANK], dtype=pl.FP32)
            kv_a_fp32_tile = pl.create_tensor([BATCH, KV_A_OUT], dtype=pl.FP32)

            # Stage 1: RMSNorm + apply weights, matching the Qwen3
            # batch-tile outer-loop structure.
            with pl.at(level=pl.Level.CORE_GROUP):
                partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb in pl.range(RMSNORM_BLOCKS):
                    k0 = kb * RMSNORM_K
                    x_chunk = pl.cast(
                        pl.slice(hidden_states, [BATCH, RMSNORM_K], [0, k0]),
                        target_type=pl.FP32,
                    )
                    partial_sq = pl.add(
                        partial_sq,
                        pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]),
                    )

                variance = pl.reshape(
                    pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                    [BATCH, 1],
                )
                inv_rms = pl.recip(pl.sqrt(variance))

                for kb in pl.range(PROJ_BLOCKS):
                    k0 = kb * PROJ_K
                    x_chunk_bf16 = pl.slice(hidden_states, [BATCH, PROJ_K], [0, k0])
                    x_tile = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                    gamma = pl.slice(input_rms_weight, [1, PROJ_K], [0, k0])
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_tile, inv_rms), gamma)
                    normed_tile = pl.assemble(
                        normed_tile,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )

            # Stage 2: Q latent projection, accumulated in Cube Acc.
            with pl.at(level=pl.Level.CORE_GROUP):
                for ob in pl.parallel(0, QR_BLOCKS, 1):
                    q0 = ob * LORA_CHUNK
                    q_tile_a = pl.slice(normed_tile, [BATCH, PROJ_K], [0, 0])
                    q_tile_b = pl.slice(wq_a, [PROJ_K, LORA_CHUNK], [0, q0])
                    q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                    for kb in pl.range(1, PROJ_BLOCKS):
                        k0 = kb * PROJ_K
                        q_tile_a_i = pl.slice(normed_tile, [BATCH, PROJ_K], [0, k0])
                        q_tile_b_i = pl.slice(wq_a, [PROJ_K, LORA_CHUNK], [k0, q0])
                        q_acc = pl.matmul_acc(q_acc, q_tile_a_i, q_tile_b_i)
                    qr_fp32_tile = pl.assemble(qr_fp32_tile, q_acc, [0, q0])

            # Stage 3: q_norm(wq_a(normed)), matching ds32.py MLA.forward.
            with pl.at(level=pl.Level.CORE_GROUP):
                q_partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb in pl.range(QR_BLOCKS):
                    k0 = kb * LORA_CHUNK
                    qr_chunk_fp32 = pl.slice(qr_fp32_tile, [BATCH, LORA_CHUNK], [0, k0])
                    q_partial_sq = pl.add(
                        q_partial_sq,
                        pl.reshape(pl.row_sum(pl.mul(qr_chunk_fp32, qr_chunk_fp32)), [1, BATCH]),
                    )

                q_variance = pl.reshape(
                    pl.add(pl.mul(q_partial_sq, Q_LORA_INV), EPS),
                    [BATCH, 1],
                )
                q_inv_rms = pl.recip(pl.sqrt(q_variance))

                for kb in pl.range(QR_BLOCKS):
                    k0 = kb * LORA_CHUNK
                    qr_chunk_bf16 = pl.cast(
                        pl.slice(qr_fp32_tile, [BATCH, LORA_CHUNK], [0, k0]),
                        target_type=pl.BF16,
                    )
                    qr_chunk_fp32 = pl.cast(qr_chunk_bf16, target_type=pl.FP32)
                    q_gamma = pl.slice(q_norm_weight, [1, LORA_CHUNK], [0, k0])
                    q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk_fp32, q_inv_rms), q_gamma)
                    q_normed_bf16 = pl.cast(q_normed, target_type=pl.BF16)
                    qr_out = pl.assemble(qr_out, q_normed_bf16, [0, k0])

            # Stage 4: Q head projection. Keep the original per-K-block
            # accumulation form; materializing the full FP32 q_proj
            # temporary exposes a block-write issue for this 24576-wide output.
            with pl.at(level=pl.Level.CORE_GROUP):
                for ob in pl.parallel(0, Q_OUT_BLOCKS, 1):
                    q0 = ob * Q_OUT_CHUNK
                    q_out_acc = pl.full([BATCH, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(QR_BLOCKS):
                        k0 = kb * LORA_CHUNK
                        q_chunk = pl.slice(qr_out, [BATCH, LORA_CHUNK], [0, k0])
                        wq_b_chunk = pl.slice(wq_b, [LORA_CHUNK, Q_OUT_CHUNK], [k0, q0])
                        q_out_acc = pl.add(
                            q_out_acc,
                            pl.matmul(q_chunk, wq_b_chunk),
                        )
                    q_proj_out = pl.assemble(q_proj_out, pl.cast(q_out_acc, target_type=pl.BF16), [0, q0])

            # Stage 5: KV latent projection, accumulated in Cube Acc.
            with pl.at(level=pl.Level.CORE_GROUP):
                for ob in pl.parallel(0, KV_A_BLOCKS, 1):
                    kv0 = ob * KV_OUT_CHUNK
                    kv_tile_a = pl.slice(normed_tile, [BATCH, PROJ_K], [0, 0])
                    kv_tile_b = pl.slice(wkv_a, [PROJ_K, KV_OUT_CHUNK], [0, kv0])
                    kv_acc = pl.matmul(kv_tile_a, kv_tile_b, out_dtype=pl.FP32)
                    for kb in pl.range(1, PROJ_BLOCKS):
                        k0 = kb * PROJ_K
                        kv_tile_a_i = pl.slice(normed_tile, [BATCH, PROJ_K], [0, k0])
                        kv_tile_b_i = pl.slice(wkv_a, [PROJ_K, KV_OUT_CHUNK], [k0, kv0])
                        kv_acc = pl.matmul_acc(kv_acc, kv_tile_a_i, kv_tile_b_i)
                    kv_a_fp32_tile = pl.assemble(kv_a_fp32_tile, kv_acc, [0, kv0])

            # Stage 6: final KV output cast from FP32 projection temporary.
            with pl.at(level=pl.Level.CORE_GROUP):
                for ob in pl.parallel(0, KV_A_BLOCKS, 1):
                    kv0 = ob * KV_OUT_CHUNK
                    kv_chunk = pl.cast(
                        pl.slice(kv_a_fp32_tile, [BATCH, KV_OUT_CHUNK], [0, kv0]),
                        target_type=pl.BF16,
                    )
                    kv_a_out = pl.assemble(kv_a_out, kv_chunk, [0, kv0])

            kv_normed_out = pl.create_tensor([BATCH, KV_LORA_RANK], dtype=pl.BF16)

            with pl.at(level=pl.Level.CORE_GROUP):
                kv_rows = pl.cast(pl.slice(kv_a_out, [BATCH, KV_LORA_RANK], [0, 0]), target_type=pl.FP32)
                kv_partial_sq = pl.reshape(pl.row_sum(pl.mul(kv_rows, kv_rows)), [1, BATCH])
                kv_variance = pl.reshape(
                    pl.add(pl.mul(kv_partial_sq, 1.0 / KV_LORA_RANK), EPS),
                    [BATCH, 1],
                )
                kv_inv_rms = pl.recip(pl.sqrt(kv_variance))
                kv_gamma = pl.slice(kv_norm_weight, [1, KV_LORA_RANK], [0, 0])
                kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rows, kv_inv_rms), kv_gamma)
                kv_normed_out = pl.assemble(kv_normed_out, pl.cast(kv_normed, target_type=pl.BF16), [0, 0])

            # Q split + q_rope: produce q_nope/q_pe while keeping the internal
            # full Q layout available for the in-place RoPE writes.
            # Cache preparation: write RMS-normalized KV latent and rotated
            # k_pe for the current decode token.
            with pl.at(level=pl.Level.CORE_GROUP):
                for b in pl.parallel(0, BATCH, 1):
                    ctx_len = pl.tensor.read(seq_lens, [b])
                    pos = ctx_len - 1
                    cache_row = b * MAX_SEQ + pos

                    cos_lo = pl.slice(rope_cos, [1, QK_ROPE_HEAD_DIM // 2], [pos, 0])
                    cos_hi = pl.slice(
                        rope_cos, [1, QK_ROPE_HEAD_DIM // 2], [pos, QK_ROPE_HEAD_DIM // 2]
                    )
                    sin_lo = pl.slice(rope_sin, [1, QK_ROPE_HEAD_DIM // 2], [pos, 0])
                    sin_hi = pl.slice(
                        rope_sin, [1, QK_ROPE_HEAD_DIM // 2], [pos, QK_ROPE_HEAD_DIM // 2]
                    )

                    for h in pl.range(NUM_HEADS):
                        q_col = h * QK_HEAD_DIM
                        q_nope_col = h * QK_NOPE_HEAD_DIM
                        q_pe_col = h * QK_ROPE_HEAD_DIM
                        q_nope = pl.slice(q_proj_out, [1, QK_NOPE_HEAD_DIM], [b, q_col])
                        q_nope_out = pl.assemble(q_nope_out, q_nope, [b, q_nope_col])
                        q_lo = pl.cast(
                            pl.slice(
                                q_proj_out,
                                [1, QK_ROPE_HEAD_DIM // 2],
                                [b, q_col + QK_NOPE_HEAD_DIM],
                            ),
                            target_type=pl.FP32,
                        )
                        q_hi = pl.cast(
                            pl.slice(
                                q_proj_out,
                                [1, QK_ROPE_HEAD_DIM // 2],
                                [b, q_col + QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM // 2],
                            ),
                            target_type=pl.FP32,
                        )
                        q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                        q_proj_out = pl.assemble(
                            q_proj_out,
                            pl.cast(q_rot_lo, target_type=pl.BF16),
                            [b, q_col + QK_NOPE_HEAD_DIM],
                        )
                        q_proj_out = pl.assemble(
                            q_proj_out,
                            pl.cast(q_rot_hi, target_type=pl.BF16),
                            [b, q_col + QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM // 2],
                        )
                        q_pe_out = pl.assemble(q_pe_out, pl.cast(q_rot_lo, target_type=pl.BF16), [b, q_pe_col])
                        q_pe_out = pl.assemble(
                            q_pe_out,
                            pl.cast(q_rot_hi, target_type=pl.BF16),
                            [b, q_pe_col + QK_ROPE_HEAD_DIM // 2],
                        )

            with pl.at(level=pl.Level.CORE_GROUP):
                for b in pl.parallel(0, BATCH, 1):
                    ctx_len = pl.tensor.read(seq_lens, [b])
                    pos = ctx_len - 1
                    cache_row = b * MAX_SEQ + pos

                    cos_lo = pl.slice(rope_cos, [1, QK_ROPE_HEAD_DIM // 2], [pos, 0])
                    cos_hi = pl.slice(
                        rope_cos, [1, QK_ROPE_HEAD_DIM // 2], [pos, QK_ROPE_HEAD_DIM // 2]
                    )
                    sin_lo = pl.slice(rope_sin, [1, QK_ROPE_HEAD_DIM // 2], [pos, 0])
                    sin_hi = pl.slice(
                        rope_sin, [1, QK_ROPE_HEAD_DIM // 2], [pos, QK_ROPE_HEAD_DIM // 2]
                    )
                    kv_normed_row = pl.slice(kv_normed_out, [1, KV_LORA_RANK], [b, 0])

                    pe_lo = pl.cast(
                        pl.slice(kv_a_out, [1, QK_ROPE_HEAD_DIM // 2], [b, KV_LORA_RANK]),
                        target_type=pl.FP32,
                    )
                    pe_hi = pl.cast(
                        pl.slice(
                            kv_a_out,
                            [1, QK_ROPE_HEAD_DIM // 2],
                            [b, KV_LORA_RANK + QK_ROPE_HEAD_DIM // 2],
                        ),
                        target_type=pl.FP32,
                    )

                    pe_rot_lo = pl.sub(pl.col_expand_mul(pe_lo, cos_lo), pl.col_expand_mul(pe_hi, sin_lo))
                    pe_rot_hi = pl.add(pl.col_expand_mul(pe_hi, cos_hi), pl.col_expand_mul(pe_lo, sin_hi))

                    kv_cache = pl.assemble(kv_cache, kv_normed_row, [cache_row, 0])
                    pe_cache = pl.assemble(pe_cache, pl.cast(pe_rot_lo, target_type=pl.BF16), [cache_row, 0])
                    pe_cache = pl.assemble(
                        pe_cache,
                        pl.cast(pe_rot_hi, target_type=pl.BF16),
                        [cache_row, QK_ROPE_HEAD_DIM // 2],
                    )

            return q_pe_out

    return DeepSeekV32DecodeFrontScope1

