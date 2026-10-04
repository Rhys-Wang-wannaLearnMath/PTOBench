# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 1 — input RMSNorm + Q/K/V projection.
  1. RMSNorm of input hidden states
  2. Q/K/V projection via matmul

Input hidden states are BF16; weights are BF16; projections output FP32.
"""
import pypto.language as pl
from config import BATCH, EPS, HIDDEN, HIDDEN_INV, KV_HIDDEN

# Vector TILELET budget (2 KB = 2048 B, FP32 = 4 B/elem):
K_CHUNK = 512
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
BATCH_TILE = 16

HIDDEN_BLOCKS = HIDDEN // K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK
Q_SPMD_BLOCKS = Q_OUT_BLOCKS // 4
KV_SPMD_BLOCKS = KV_OUT_BLOCKS // 4


def build_qwen3_scope1_program():
    @pl.program
    class Qwen3Scope1:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_scope1(
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
            wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
            wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
            q_proj: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
            k_proj: pl.Out[pl.Tensor[[BATCH, KV_HIDDEN], pl.FP32]],
            v_proj: pl.Out[pl.Tensor[[BATCH, KV_HIDDEN], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[BATCH, HIDDEN], pl.FP32],
            pl.Tensor[[BATCH, KV_HIDDEN], pl.FP32],
            pl.Tensor[[BATCH, KV_HIDDEN], pl.FP32],
        ]:
            for b0 in pl.parallel(0, BATCH, BATCH_TILE):
                normed_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)

                # Stage 1: RMSNorm + apply weights (vector ops only).
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                    partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.pipeline(HIDDEN_BLOCKS, stage=2):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(
                            partial_sq,
                            pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]),
                        )
                    # Compute variance in [1, BATCH_TILE], then reshape to [BATCH_TILE, 1]
                    # for row_expand_mul broadcasting.
                    variance = pl.reshape(
                        pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                        [BATCH_TILE, 1],
                    )

                    inv_rms = pl.recip(pl.sqrt(variance))

                    for kb in pl.pipeline(HIDDEN_BLOCKS, stage=2):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                        normed_tile = pl.assemble(normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                # Stage 2: Q projection — SPMD over output columns.
                for ob0 in pl.spmd(Q_SPMD_BLOCKS, name_hint="q_proj"):
                    for ob in pl.range(ob0 * 4, (ob0 + 1) * 4):
                        q0 = ob * Q_OUT_CHUNK

                        tile_a = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        tile_b = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [0, q0])
                        q_acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)

                        tile_a_1 = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, K_CHUNK])
                        tile_b_1 = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [K_CHUNK, q0])
                        q_acc = pl.matmul_acc(q_acc, tile_a_1, tile_b_1)

                        for kb in pl.pipeline(2, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            tile_b_i = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)

                        q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

                # Stage 3: K/V projection in a single SPMD block.
                for ob0 in pl.spmd(KV_SPMD_BLOCKS, name_hint="kv_proj"):
                    for ob in pl.range(ob0 * 4, (ob0 + 1) * 4):
                        kv0 = ob * KV_OUT_CHUNK

                        tile_a = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        tile_wk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                        k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)

                        tile_a_k1 = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, K_CHUNK])
                        tile_wk_1 = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [K_CHUNK, kv0])
                        k_acc = pl.matmul_acc(k_acc, tile_a_k1, tile_wk_1)

                        for kb in pl.pipeline(2, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            tile_wk_i = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)

                        k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

                        tile_a = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        tile_wv = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                        v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)

                        tile_a_v1 = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, K_CHUNK])
                        tile_wv_1 = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [K_CHUNK, kv0])
                        v_acc = pl.matmul_acc(v_acc, tile_a_v1, tile_wv_1)

                        for kb in pl.pipeline(2, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            tile_wv_i = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)

                        v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

            return q_proj, k_proj, v_proj

    return Qwen3Scope1
