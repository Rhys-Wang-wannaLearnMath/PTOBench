# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP decode front scope2 (indexer prepare).

This file prepares the indexer outputs consumed by scope3:
- q = wq_b(qr), split as q_pe/q_nope with q_pe first
- k = LayerNorm(wk(hidden_states)), split as k_pe/k_nope with k_pe first
- apply non-interleaved RoPE to q_pe and k_pe
- compute per-index-head weights
- quantize q_idx_full and current-token k_idx to row-wise INT8
- update k_cache_idx_i8 and k_cache_idx_scale at the current decode position
"""

import pypto.language as pl


from config import (BATCH, MAX_SEQ, HIDDEN, Q_LORA_RANK, QK_ROPE_HEAD_DIM,
                    INDEX_HEADS, INDEX_HEAD_DIM, EPS, INT8_GROUP_SIZE,
                    INT8_SCALE_MAX, INT8_AMAX_EPS, INT8_SCALE_PACK, CACHE_ROWS)

K_CHUNK = 128
LORA_CHUNK = 128
IDX_OUT_CHUNK = 128
WEIGHTS_OUT_CHUNK = 64

if INDEX_HEAD_DIM != INT8_GROUP_SIZE:
    raise ValueError(
        f"INT8 quant path expects INDEX_HEAD_DIM == {INT8_GROUP_SIZE}, "
        f"got {INDEX_HEAD_DIM}"
    )

INDEX_Q_OUT = INDEX_HEADS * INDEX_HEAD_DIM
INDEX_Q_ROWS = BATCH * INDEX_HEADS

HIDDEN_BLOCKS = (HIDDEN + K_CHUNK - 1) // K_CHUNK
QR_BLOCKS = (Q_LORA_RANK + LORA_CHUNK - 1) // LORA_CHUNK
IDX_OUT_BLOCKS = (INDEX_Q_OUT + IDX_OUT_CHUNK - 1) // IDX_OUT_CHUNK
WK_OUT_BLOCKS = (INDEX_HEAD_DIM + IDX_OUT_CHUNK - 1) // IDX_OUT_CHUNK
WEIGHTS_BLOCKS = (INDEX_HEADS + WEIGHTS_OUT_CHUNK - 1) // WEIGHTS_OUT_CHUNK
INDEX_HEAD_DIM_INV = 1.0 / INDEX_HEAD_DIM
WEIGHT_SCALE = (INDEX_HEADS ** -0.5) * (INDEX_HEAD_DIM ** -0.5)


def build_deepseek_v3_2_decode_front_indexer_program():
    @pl.program
    class DeepSeekV32DecodeFrontIndexer:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_indexer(
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            qr: pl.Tensor[[BATCH, Q_LORA_RANK], pl.BF16],
            wq_b_idx: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
            wk_idx: pl.Tensor[[HIDDEN, INDEX_HEAD_DIM], pl.BF16],
            k_norm_weight: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            k_norm_bias: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            weights_proj: pl.Tensor[[HIDDEN, INDEX_HEADS], pl.FP32],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            k_cache_idx_i8: pl.InOut[pl.Tensor[[CACHE_ROWS, INDEX_HEAD_DIM], pl.INT8]],
            k_cache_idx_scale: pl.InOut[pl.Tensor[[BATCH, MAX_SEQ], pl.FP32]],
            q_idx_full_i8_out: pl.Out[pl.Tensor[[INDEX_Q_ROWS, INDEX_HEAD_DIM], pl.INT8]],
            q_idx_scale_heads_out: pl.Out[pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32]],
            weights_out: pl.Out[pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[CACHE_ROWS, INDEX_HEAD_DIM], pl.INT8],
            pl.Tensor[[BATCH, MAX_SEQ], pl.FP32],
            pl.Tensor[[INDEX_Q_ROWS, INDEX_HEAD_DIM], pl.INT8],
            pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32],
            pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32],
        ]:
            q_idx_full = pl.create_tensor([BATCH, INDEX_Q_OUT], dtype=pl.BF16)
            k_idx = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)
            q_idx_full_i8 = pl.create_tensor([INDEX_Q_ROWS, INDEX_HEAD_DIM], dtype=pl.INT8)
            k_idx_i8 = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.INT8)
            k_idx_scale = pl.create_tensor([BATCH, INT8_SCALE_PACK], dtype=pl.FP32)
            weights = pl.create_tensor([BATCH, INDEX_HEADS], dtype=pl.FP32)
            q_idx_scale_heads = pl.create_tensor([BATCH, INDEX_HEADS], dtype=pl.FP32)

            # Stage 0: q_idx_full = wq_b_idx(qr), shaped as [B, INDEX_HEADS, INDEX_HEAD_DIM].
            with pl.at(level=pl.Level.CORE_GROUP):
                for ob in pl.parallel(0, IDX_OUT_BLOCKS, 1):
                    q0 = ob * IDX_OUT_CHUNK
                    q_acc = pl.full([BATCH, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(QR_BLOCKS):
                        k0 = kb * LORA_CHUNK
                        qr_chunk = pl.slice(qr, [BATCH, LORA_CHUNK], [0, k0])
                        wq_chunk = pl.slice(wq_b_idx, [LORA_CHUNK, IDX_OUT_CHUNK], [k0, q0])
                        q_acc = pl.add(q_acc, pl.matmul(qr_chunk, wq_chunk, out_dtype=pl.FP32))
                    q_idx_full = pl.assemble(q_idx_full, pl.cast(q_acc, target_type=pl.BF16), [0, q0])

            # Stage 1: k_idx pre-projection = wk_idx(hidden_states).
            with pl.at(level=pl.Level.CORE_GROUP):
                for ob in pl.parallel(0, WK_OUT_BLOCKS, 1):
                    k1 = ob * IDX_OUT_CHUNK
                    k_acc = pl.full([BATCH, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(hidden_states, [BATCH, K_CHUNK], [0, k0])
                        wk_chunk = pl.slice(wk_idx, [K_CHUNK, IDX_OUT_CHUNK], [k0, k1])
                        k_acc = pl.add(k_acc, pl.matmul(x_chunk, wk_chunk, out_dtype=pl.FP32))
                    k_idx = pl.assemble(k_idx, pl.cast(k_acc, target_type=pl.BF16), [0, k1])

            # Stage 2: LayerNorm on k_idx.
            with pl.at(level=pl.Level.CORE_GROUP):
                k_tile = pl.cast(pl.slice(k_idx, [BATCH, INDEX_HEAD_DIM], [0, 0]), target_type=pl.FP32)
                mean = pl.row_sum(pl.mul(k_tile, INDEX_HEAD_DIM_INV))
                centered = pl.row_expand_sub(k_tile, mean)
                var_eps = pl.row_sum(pl.mul(pl.add(pl.mul(centered, centered), EPS), INDEX_HEAD_DIM_INV))
                std = pl.reshape(
                    pl.sqrt(pl.reshape(var_eps, [1, BATCH])),
                    [BATCH, 1],
                )
                inv_std = pl.recip(std)
                normed = pl.row_expand_mul(centered, inv_std)
                gamma = pl.slice(k_norm_weight, [1, INDEX_HEAD_DIM], [0, 0])
                beta = pl.slice(k_norm_bias, [1, INDEX_HEAD_DIM], [0, 0])
                scaled = pl.col_expand_mul(normed, gamma)
                ones = pl.add(pl.sub(k_tile, k_tile), 1.0)
                k_normed = pl.add(scaled, pl.col_expand_mul(ones, beta))
                k_idx = pl.assemble(k_idx, pl.cast(k_normed, target_type=pl.BF16), [0, 0])

            # Stage 3: non-interleaved RoPE on indexer q_pe and k_pe.
            with pl.at(level=pl.Level.CORE_GROUP):
                for b in pl.parallel(0, BATCH, 1):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cos_lo = pl.slice(rope_cos, [1, QK_ROPE_HEAD_DIM // 2], [pos, 0])
                    cos_hi = pl.slice(
                        rope_cos,
                        [1, QK_ROPE_HEAD_DIM // 2],
                        [pos, QK_ROPE_HEAD_DIM // 2],
                    )
                    sin_lo = pl.slice(rope_sin, [1, QK_ROPE_HEAD_DIM // 2], [pos, 0])
                    sin_hi = pl.slice(
                        rope_sin,
                        [1, QK_ROPE_HEAD_DIM // 2],
                        [pos, QK_ROPE_HEAD_DIM // 2],
                    )

                    for h in pl.range(INDEX_HEADS):
                        q_col = h * INDEX_HEAD_DIM
                        q_lo = pl.cast(
                            pl.slice(q_idx_full, [1, QK_ROPE_HEAD_DIM // 2], [b, q_col]),
                            target_type=pl.FP32,
                        )
                        q_hi = pl.cast(
                            pl.slice(
                                q_idx_full,
                                [1, QK_ROPE_HEAD_DIM // 2],
                                [b, q_col + QK_ROPE_HEAD_DIM // 2],
                            ),
                            target_type=pl.FP32,
                        )
                        q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                        q_idx_full = pl.assemble(
                            q_idx_full,
                            pl.cast(q_rot_lo, target_type=pl.BF16),
                            [b, q_col],
                        )
                        q_idx_full = pl.assemble(
                            q_idx_full,
                            pl.cast(q_rot_hi, target_type=pl.BF16),
                            [b, q_col + QK_ROPE_HEAD_DIM // 2],
                        )

                    k_lo = pl.cast(pl.slice(k_idx, [1, QK_ROPE_HEAD_DIM // 2], [b, 0]), target_type=pl.FP32)
                    k_hi = pl.cast(
                        pl.slice(k_idx, [1, QK_ROPE_HEAD_DIM // 2], [b, QK_ROPE_HEAD_DIM // 2]),
                        target_type=pl.FP32,
                    )
                    k_rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                    k_rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                    k_idx = pl.assemble(k_idx, pl.cast(k_rot_lo, target_type=pl.BF16), [b, 0])
                    k_idx = pl.assemble(
                        k_idx,
                        pl.cast(k_rot_hi, target_type=pl.BF16),
                        [b, QK_ROPE_HEAD_DIM // 2],
                    )

            # Stage 4: weights = weights_proj(hidden_states) * n_heads^-0.5 * head_dim^-0.5.
            with pl.at(level=pl.Level.CORE_GROUP):
                for ob in pl.parallel(0, WEIGHTS_BLOCKS, 1):
                    w0 = ob * WEIGHTS_OUT_CHUNK
                    w_acc = pl.full([BATCH, WEIGHTS_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(hidden_states, [BATCH, K_CHUNK], [0, k0])
                        wp_chunk = pl.slice(weights_proj, [K_CHUNK, WEIGHTS_OUT_CHUNK], [k0, w0])
                        w_acc = pl.add(
                            w_acc,
                            pl.matmul(
                                x_chunk,
                                pl.cast(wp_chunk, target_type=pl.BF16),
                                out_dtype=pl.FP32,
                            ),
                        )
                    w_scaled = pl.mul(w_acc, WEIGHT_SCALE)
                    weights = pl.assemble(weights, w_scaled, [0, w0])

            # Stage 5: Quantize indexer q/k tensors and keep INT8 outputs for scope3.
            # Keep q and k in separate Vec scopes: current ptoas accounts all
            # temporaries in one CORE_GROUP, and the combined scope exceeds the
            # A2/A3 Vec buffer limit even though the two paths are independent.
            q_idx_grouped = pl.reshape(q_idx_full, [INDEX_Q_ROWS, INDEX_HEAD_DIM])
            with pl.at(level=pl.Level.CORE_GROUP):
                for b in pl.parallel(0, BATCH, 1):
                    for h0 in pl.range(0, INDEX_HEADS, BATCH):
                        r0 = b * INDEX_HEADS + h0
                        q_block = pl.cast(
                            pl.slice(q_idx_grouped, [BATCH, INDEX_HEAD_DIM], [r0, 0]),
                            target_type=pl.FP32,
                            mode="none",
                        )
                        q_abs = pl.maximum(q_block, pl.neg(q_block))
                        q_amax_row = pl.reshape(pl.row_max(q_abs), [1, BATCH])
                        q_amax_row = pl.maximum(
                            q_amax_row,
                            pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS),
                        )
                        q_scale_quant_row = pl.div(
                            pl.full([1, BATCH], dtype=pl.FP32, value=INT8_SCALE_MAX),
                            q_amax_row,
                        )
                        q_scale_dequant_row = pl.div(
                            pl.full([1, BATCH], dtype=pl.FP32, value=1.0),
                            q_scale_quant_row,
                        )
                        q_scale_quant = pl.reshape(q_scale_quant_row, [BATCH, 1])
                        q_scaled = pl.row_expand_mul(q_block, q_scale_quant)
                        q_i32 = pl.cast(q_scaled, target_type=pl.INT32, mode="round")
                        q_half = pl.cast(q_i32, target_type=pl.FP16, mode="round")
                        q_i8 = pl.cast(q_half, target_type=pl.INT8, mode="trunc")
                        q_idx_full_i8 = pl.assemble(q_idx_full_i8, q_i8, [r0, 0])
                        q_idx_scale_heads = pl.assemble(q_idx_scale_heads, q_scale_dequant_row, [b, h0])

            with pl.at(level=pl.Level.CORE_GROUP):
                for r0 in pl.parallel(0, BATCH, BATCH):
                    k_block = pl.cast(
                        pl.slice(k_idx, [BATCH, INDEX_HEAD_DIM], [r0, 0]),
                        target_type=pl.FP32,
                        mode="none",
                    )
                    k_abs = pl.maximum(k_block, pl.neg(k_block))
                    k_amax_row = pl.reshape(pl.row_max(k_abs), [1, BATCH])
                    k_amax_row = pl.maximum(
                        k_amax_row,
                        pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS),
                    )
                    k_scale_quant_row = pl.div(
                        pl.full([1, BATCH], dtype=pl.FP32, value=INT8_SCALE_MAX),
                        k_amax_row,
                    )
                    k_scale_dequant_row = pl.div(
                        pl.full([1, BATCH], dtype=pl.FP32, value=1.0),
                        k_scale_quant_row,
                    )
                    k_scale_quant = pl.reshape(k_scale_quant_row, [BATCH, 1])
                    k_scale_dequant = pl.reshape(k_scale_dequant_row, [BATCH, 1])
                    k_scale_pack_target = pl.full([BATCH, INT8_SCALE_PACK], dtype=pl.FP32, value=0.0)
                    k_scale_pack = pl.row_expand(k_scale_pack_target, k_scale_dequant)
                    k_scaled = pl.row_expand_mul(k_block, k_scale_quant)
                    k_i32 = pl.cast(k_scaled, target_type=pl.INT32, mode="round")
                    k_half = pl.cast(k_i32, target_type=pl.FP16, mode="round")
                    k_i8 = pl.cast(k_half, target_type=pl.INT8, mode="trunc")
                    k_idx_i8 = pl.assemble(k_idx_i8, k_i8, [r0, 0])
                    k_idx_scale = pl.assemble(k_idx_scale, k_scale_pack, [r0, 0])

            # Stage 6: update k_cache_idx_i8 and k_cache_idx_scale for the current decode token.
            k_idx_scale_flat = pl.reshape(k_idx_scale, [BATCH * INT8_SCALE_PACK])
            k_cache_idx_scale_flat = pl.reshape(k_cache_idx_scale, [BATCH * MAX_SEQ])
            with pl.at(level=pl.Level.CORE_GROUP):
                for b in pl.parallel(0, BATCH, 1):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cache_row = b * MAX_SEQ + pos
                    k_row_i8 = pl.slice(k_idx_i8, [1, INDEX_HEAD_DIM], [b, 0])
                    k_row_scale = pl.tensor.read(k_idx_scale_flat, [b * INT8_SCALE_PACK])
                    k_cache_idx_i8 = pl.assemble(k_cache_idx_i8, k_row_i8, [cache_row, 0])
                    pl.tensor.write(k_cache_idx_scale_flat, [b * MAX_SEQ + pos], k_row_scale)

            # Stage 7: export q scales, q INT8 rows, and weights for scope3.
            with pl.at(level=pl.Level.CORE_GROUP):
                for r0 in pl.parallel(0, INDEX_Q_ROWS, BATCH):
                    q_i8_tile = pl.slice(q_idx_full_i8, [BATCH, INDEX_HEAD_DIM], [r0, 0])
                    q_idx_full_i8_out = pl.assemble(q_idx_full_i8_out, q_i8_tile, [r0, 0])

                for b0 in pl.parallel(0, BATCH, BATCH):
                    q_scale_tile = pl.slice(q_idx_scale_heads, [BATCH, INDEX_HEADS], [b0, 0])
                    weights_tile = pl.slice(weights, [BATCH, INDEX_HEADS], [b0, 0])
                    q_idx_scale_heads_out = pl.assemble(q_idx_scale_heads_out, q_scale_tile, [b0, 0])
                    weights_out = pl.assemble(weights_out, weights_tile, [b0, 0])

            return k_cache_idx_i8, k_cache_idx_scale, q_idx_full_i8_out, q_idx_scale_heads_out, weights_out

    return DeepSeekV32DecodeFrontIndexer


def build_deepseek_v3_2_decode_front_scope2_program():
    return build_deepseek_v3_2_decode_front_indexer_program()

