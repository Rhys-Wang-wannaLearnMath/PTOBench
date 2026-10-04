# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer decode BACK part (batch=16, max_seq=4096).

BACK boundary:
- start from combine buffer read
- run full residual + MLP + output path
"""


import pypto.language as pl


from config import (BATCH, MAX_SEQ, HIDDEN, INTERMEDIATE, NUM_HEADS,
                    V_HEAD_DIM, ATTN_OUT, EP_NODES, EPS, HIDDEN_INV)

# tiling constants.
K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 256
BATCH_TILE = 16


def build_deepseek_v3_2_decode_back_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
):
    BATCH_SIZE = batch
    HIDDEN_SIZE = hidden_size
    INTER_SIZE = intermediate_size
    ATTN_OUT_SIZE = attn_out_size
    EP_NODES_SIZE = ep_nodes

    ATTN_BLOCKS = (ATTN_OUT_SIZE + K_CHUNK - 1) // K_CHUNK
    HIDDEN_BLOCKS = (HIDDEN_SIZE + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_SIZE + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_SIZE + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK

    @pl.program
    class DeepSeekV32DecodeBack:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_back_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16],
            node_id_t: pl.Tensor[[1], pl.INT32],
            # combine buffer from cross-node communication
            combine_buf: pl.Tensor[[EP_NODES_SIZE, BATCH_SIZE, ATTN_OUT_SIZE], pl.BF16],
            wo: pl.Tensor[[ATTN_OUT_SIZE, HIDDEN_SIZE], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_SIZE], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_down: pl.Tensor[[INTER_SIZE, HIDDEN_SIZE], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16]],
        ) -> pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16]:
            # Scope: output projection + residual + post-rms + MLP + residual.
            node_id = pl.cast(pl.tensor.read(node_id_t, [0]), pl.INDEX)
            for b0 in pl.range(0, BATCH_SIZE, BATCH_TILE):
                resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_SIZE], dtype=pl.FP32)
                # Read combine results from this node view.
                combined_3d = pl.slice(
                    combine_buf, [1, BATCH_TILE, ATTN_OUT_SIZE], [node_id, b0, 0]
                )
                combined = pl.reshape(combined_3d, [BATCH_TILE, ATTN_OUT_SIZE])

                # O projection and residual.
                for ob in pl.range(Q_OUT_BLOCKS):
                    o0 = ob * Q_OUT_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP):
                        a_chunk_0 = pl.slice(combined, [BATCH_TILE, K_CHUNK], [0, 0])
                        w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                        o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, ATTN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(combined, [BATCH_TILE, K_CHUNK], [0, k0])
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]), target_type=pl.FP32
                        )
                        resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                # Post RMSNorm.
                post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_SIZE], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]))
                    inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [BATCH_TILE, 1])), gamma
                        )
                        post_norm_tile = pl.assemble(
                            post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                        )

                # MLP.
                mlp_tile = pl.create_tensor([BATCH_TILE, INTER_SIZE], dtype=pl.BF16)
                for ob in pl.range(MLP_OUT_BLOCKS):
                    o0 = ob * MLP_OUT_CHUNK

                    with pl.at(level=pl.Level.CORE_GROUP):
                        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                        mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

                # Down projection + final residual writeback.
                for dob in pl.range(HIDDEN_BLOCKS):
                    d0 = dob * K_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP):
                        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
                        w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                        for ob in pl.range(1, MLP_OUT_BLOCKS):
                            o0 = ob * MLP_OUT_CHUNK
                            down_mlp_chunk_bf16 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, o0])
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        out_chunk = pl.add(
                            down_acc,
                            pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0]),
                        )
                        out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [b0, d0])

            return out

    return DeepSeekV32DecodeBack


