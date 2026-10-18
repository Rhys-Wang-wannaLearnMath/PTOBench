# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer prefill BACK part (batch=16, max_seq=4096).
[NOTE] Current test is reduced to batch=4, max_seq=128 for faster iteration and dodge HBM OOM issue.

BACK boundary:
- read combine tensor
- run complete residual + MLP + output path
"""

import pypto.language as pl

from config import BATCH, MAX_SEQ, HIDDEN, INTERMEDIATE, NUM_HEADS, V_HEAD_DIM, ATTN_OUT, EP_NODES, EPS, HIDDEN_INV


K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 128
TOK_TILE = 64


def build_deepseek_v3_2_prefill_back_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
):
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    INTER_CFG = intermediate_size
    ATTN_OUT_CFG = attn_out_size
    EP_NODES_CFG = ep_nodes

    ATTN_BLOCKS = (ATTN_OUT_CFG + K_CHUNK - 1) // K_CHUNK
    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_CFG + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK

    @pl.program
    class DeepSeekV32PrefillBack:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_prefill_back_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            combine_buf: pl.Tensor[[EP_NODES_CFG, BATCH_CFG, MAX_SEQ_CFG, ATTN_OUT_CFG], pl.BF16],
            wo: pl.Tensor[[ATTN_OUT_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16]],
        ) -> pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16]:
            for b in pl.parallel(0, BATCH_CFG, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    # GM intermediate tensors.
                    resid1_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    attn_tile = pl.create_tensor([TOK_TILE, ATTN_OUT_CFG], dtype=pl.BF16)

                    # Stage 1: Copy combine_buf 4D -> attn_tile 2D.
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for kb in pl.range(ATTN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk_fp32 = pl.reshape(
                                pl.cast(
                                    pl.slice(combine_buf, [1, 1, TOK_TILE, K_CHUNK], [0, b, p0, k0],
                                             valid_shape=[1, 1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            a_chunk_bf16 = pl.cast(a_chunk_fp32, target_type=pl.BF16)
                            attn_tile = pl.assemble(attn_tile, a_chunk_bf16, [0, k0])

                    # Stage 2: Output projection + first residual.
                    for ob in pl.range(Q_OUT_BLOCKS):
                        o0 = ob * Q_OUT_CHUNK

                        # Cube: chained matmul.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_w = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                            o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, ATTN_BLOCKS):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_w_i = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                                o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)

                            resid1_tile = pl.assemble(resid1_tile, o_acc, [0, o0])

                        # Vector: add residual.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            resid_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, Q_OUT_CHUNK], [b, p0, o0],
                                             valid_shape=[1, valid_tok, Q_OUT_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_CHUNK],
                            )
                            mm_out = pl.slice(resid1_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0])
                            resid_sum = pl.add(mm_out, resid_chunk)
                            resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                    # Stage 3: Post-attention RMSNorm.
                    post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.BF16)
                    down_fp32_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.FP32)

                    # 3a: Compute inv_rms (reduction — sequential).
                    with pl.at(level=pl.Level.CORE_GROUP):
                        sq_sum = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            sq_sum = pl.add(
                                sq_sum,
                                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, TOK_TILE]),
                            )
                        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                    # 3b: Normalize + gamma + zero-init down_proj (parallel — independent offsets).
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for kb in pl.parallel(0, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [TOK_TILE, 1])),
                                gamma,
                            )
                            normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                            post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, k0])
                            down_zero_chunk = pl.full([TOK_TILE, K_CHUNK], dtype=pl.FP32, value=0.0)
                            down_fp32_tile = pl.assemble(down_fp32_tile, down_zero_chunk, [0, k0])

                    # Stage 4: MLP gate/up + SiLU + down projection.
                    for ob in pl.range(MLP_OUT_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK

                        # Gate matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                            for kb in pl.range(1, HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wgi = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                        # Up matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                            for kb in pl.range(1, HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wui = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                up_acc = pl.matmul_acc(up_acc, pci, wui)

                        # SiLU activation.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                        # Down projection: cube matmul + vector accumulate.
                        for dob in pl.range(HIDDEN_BLOCKS):
                            d0 = dob * K_CHUNK

                            with pl.at(level=pl.Level.CORE_GROUP):
                                w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                                down_next = pl.matmul(mlp_chunk_bf16, w_down_chunk, out_dtype=pl.FP32)

                            with pl.at(level=pl.Level.CORE_GROUP):
                                down_prev = pl.slice(down_fp32_tile, [TOK_TILE, K_CHUNK], [0, d0])
                                accum = pl.add(down_prev, down_next)
                                down_fp32_tile = pl.assemble(down_fp32_tile, accum, [0, d0])

                    # Stage 5: Final residual add -> BF16 output (parallel — independent offsets).
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for ob in pl.parallel(0, HIDDEN_BLOCKS):
                            o0 = ob * K_CHUNK
                            final_sum = pl.add(
                                pl.slice(down_fp32_tile, [TOK_TILE, K_CHUNK], [0, o0]),
                                pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, o0]),
                            )
                            final_bf16 = pl.cast(final_sum, target_type=pl.BF16)
                            out = pl.assemble(out, final_bf16, [b, p0, o0])

            return out

    return DeepSeekV32PrefillBack


