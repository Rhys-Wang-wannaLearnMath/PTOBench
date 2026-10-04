# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE local expert + shared expert compute (decode, EP single-card)."""


import pypto.language as pl


from config import (
    B,
    S,
    T,
    D,
    MOE_INTER,
    TOPK,
    SWIGLU_LIMIT,
    EP_WORLD_SIZE,
    EP_RANK,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    EXPERTS_START_IDX,
    RECV_TOTAL_MAX,
)

K_CHUNK = 256
INTER_K = 256
INTER_CHUNK = 64
D_OUT_CHUNK = 64
RECV_Y_INIT_CHUNK = 512


def build_deepseek_v4_decode_moe_expert_program():
    @pl.program
    class DeepSeekV4DecodeMoEExpert:
        @pl.function(type=pl.FunctionType.InCore)
        def compute_scale_col(
            self,
            recv_expert_id: pl.Tensor[[RECV_TOTAL_MAX, 1], pl.INT32],
            recv_weights: pl.Tensor[[RECV_TOTAL_MAX, 1], pl.FP32],
            local_i: pl.Scalar[pl.INDEX],
            scale_col: pl.Out[pl.Tensor[[RECV_TOTAL_MAX, 1], pl.FP32]],
        ) -> pl.Tensor[[RECV_TOTAL_MAX, 1], pl.FP32]:
            # Use [1, RECV_TOTAL_MAX] form: tile.cmps mask packs into the trailing
            # dim, and layout resolution rejects column-1 sel inputs.
            local_i_i32: pl.Scalar[pl.INT32] = pl.cast(local_i, pl.INT32)
            expert_id_col: pl.Tile[[RECV_TOTAL_MAX, 1], pl.INT32] = pl.load(recv_expert_id, [0, 0], [RECV_TOTAL_MAX, 1])
            weights_col: pl.Tile[[RECV_TOTAL_MAX, 1], pl.FP32] = pl.load(recv_weights, [0, 0], [RECV_TOTAL_MAX, 1])
            expert_id_row: pl.Tile[[1, RECV_TOTAL_MAX], pl.INT32] = pl.reshape(expert_id_col, [1, RECV_TOTAL_MAX])
            weights_row: pl.Tile[[1, RECV_TOTAL_MAX], pl.FP32] = pl.reshape(weights_col, [1, RECV_TOTAL_MAX])
            one_tile: pl.Tile[[1, RECV_TOTAL_MAX], pl.FP32] = pl.tile.full([1, RECV_TOTAL_MAX], dtype=pl.FP32, value=1.0)
            zero_tile: pl.Tile[[1, RECV_TOTAL_MAX], pl.FP32] = pl.tile.full([1, RECV_TOTAL_MAX], dtype=pl.FP32, value=0.0)
            tmp: pl.Tile[[1, 32], pl.FP32] = pl.tile.create([1, 32], dtype=pl.FP32)
            eq_mask: pl.Tile[[1, 32], pl.UINT8] = pl.tile.cmps(expert_id_row, local_i_i32, cmp_type=0)
            mask_fp32: pl.Tile[[1, RECV_TOTAL_MAX], pl.FP32] = pl.tile.sel(eq_mask, one_tile, zero_tile, tmp)
            scaled_row: pl.Tile[[1, RECV_TOTAL_MAX], pl.FP32] = pl.mul(weights_row, mask_fp32)
            scaled_col: pl.Tile[[RECV_TOTAL_MAX, 1], pl.FP32] = pl.reshape(scaled_row, [RECV_TOTAL_MAX, 1])
            scale_col = pl.store(scaled_col, [0, 0], scale_col)
            return scale_col

        @pl.function(type=pl.FunctionType.Orchestration)
        def deepseek_v4_decode_moe_expert(
            self,
            recv_x: pl.Tensor[[RECV_TOTAL_MAX, D], pl.BF16],
            # Global expert id per row; padding rows must carry an id outside
            # [EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS) (e.g. -1).
            recv_expert_id: pl.Tensor[[RECV_TOTAL_MAX, 1], pl.INT32],
            recv_weights: pl.Tensor[[RECV_TOTAL_MAX, 1], pl.FP32],
            x_local: pl.Tensor[[T, D], pl.BF16],
            expert_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.BF16],
            expert_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.BF16],
            expert_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.BF16],
            shared_w1: pl.Tensor[[MOE_INTER, D], pl.BF16],
            shared_w3: pl.Tensor[[MOE_INTER, D], pl.BF16],
            shared_w2: pl.Tensor[[D, MOE_INTER], pl.BF16],
            recv_y: pl.Out[pl.Tensor[[RECV_TOTAL_MAX, D], pl.BF16]],
            sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
        ):
            # Stage 0: zero-init recv_y so cross-expert accumulation is exact.
            for d0 in pl.parallel(0, D, RECV_Y_INIT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="recv_y_zero"):
                    zero_chunk = pl.full([RECV_TOTAL_MAX, RECV_Y_INIT_CHUNK], dtype=pl.FP32, value=0.0)
                    recv_y[:, d0 : d0 + RECV_Y_INIT_CHUNK] = pl.cast(zero_chunk, target_type=pl.BF16)

            # Stage 1: routed local experts.
            #   - local_i iterates GLOBAL expert ids (matches recv_expert_id values).
            #   - expert_w* are local-only, indexed by local_offset.
            #   - Each row matches at most one local_i, so mask zeros out non-matching
            #     rows and cross-expert `recv_y +=` reduces to scatter-by-mask.
            for local_i in pl.range(EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS):
                local_offset = local_i - EXPERTS_START_IDX
                scale_col = pl.create_tensor([RECV_TOTAL_MAX, 1], dtype=pl.FP32)
                scale_col = self.compute_scale_col(recv_expert_id, recv_weights, local_i, scale_col)

                h_tile = pl.create_tensor([RECV_TOTAL_MAX, MOE_INTER], dtype=pl.BF16)

                for n0 in pl.parallel(0, MOE_INTER, INTER_CHUNK):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_gate_up_matmul"):
                        x_init = recv_x[:, 0 : K_CHUNK]
                        w1_init = expert_w1[local_offset, n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                        w3_init = expert_w3[local_offset, n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                        gate_acc = pl.matmul(x_init, w1_init, b_trans=True, out_dtype=pl.FP32)
                        up_acc = pl.matmul(x_init, w3_init, b_trans=True, out_dtype=pl.FP32)
                        for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                            x_k = recv_x[:, k0 : k0 + K_CHUNK]
                            w1_k = expert_w1[local_offset, n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                            w3_k = expert_w3[local_offset, n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                            gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                            up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)
                        # Scalar-indexed expert_w* keeps a leading size-1 dim; matmul promotes
                        # that to a 3D batched output. Drop it so downstream stays 2D.
                        gate_2d = pl.reshape(gate_acc, [RECV_TOTAL_MAX, INTER_CHUNK])
                        up_2d = pl.reshape(up_acc, [RECV_TOTAL_MAX, INTER_CHUNK])

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_swiglu"):
                        # TODO: re-enable once pl.mins/maxs accept TensorType (currently
                        # tile-only). Demo config has SWIGLU_LIMIT=0.0 so this is a no-op,
                        # but v4-pro config (SWIGLU_LIMIT=10.0) needs the clamp.
                        # if SWIGLU_LIMIT > 0.0:
                        #     gate_2d = pl.mins(gate_2d, SWIGLU_LIMIT)
                        #     up_2d = pl.maxs(pl.mins(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                        silu = pl.mul(gate_2d, sigmoid)
                        gated = pl.mul(silu, up_2d)
                        h_chunk = pl.row_expand_mul(gated, scale_col)
                        h_tile[:, n0 : n0 + INTER_CHUNK] = pl.cast(h_chunk, target_type=pl.BF16)

                for d0 in pl.parallel(0, D, D_OUT_CHUNK):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_w2_matmul"):
                        h_init = h_tile[:, 0 : INTER_K]
                        w2_init = expert_w2[local_offset, d0 : d0 + D_OUT_CHUNK, 0 : INTER_K]
                        y_acc = pl.matmul(h_init, w2_init, b_trans=True, out_dtype=pl.FP32)
                        for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                            h_k = h_tile[:, k0 : k0 + INTER_K]
                            w2_k = expert_w2[local_offset, d0 : d0 + D_OUT_CHUNK, k0 : k0 + INTER_K]
                            y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)
                        y_2d = pl.reshape(y_acc, [RECV_TOTAL_MAX, D_OUT_CHUNK])

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_recv_y_accum"):
                        existing = pl.cast(recv_y[:, d0 : d0 + D_OUT_CHUNK], target_type=pl.FP32)
                        summed = pl.add(existing, y_2d)
                        recv_y[:, d0 : d0 + D_OUT_CHUNK] = pl.cast(summed, target_type=pl.BF16)

            # Stage 2: shared expert
            sh_tile = pl.create_tensor([T, MOE_INTER], dtype=pl.BF16)

            for n0 in pl.parallel(0, MOE_INTER, INTER_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up_matmul"):
                    xs_init = x_local[:, 0 : K_CHUNK]
                    sw1_init = shared_w1[n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                    sw3_init = shared_w3[n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                    sh_gate_acc = pl.matmul(xs_init, sw1_init, b_trans=True, out_dtype=pl.FP32)
                    sh_up_acc = pl.matmul(xs_init, sw3_init, b_trans=True, out_dtype=pl.FP32)
                    for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                        xs_k = x_local[:, k0 : k0 + K_CHUNK]
                        sw1_k = shared_w1[n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                        sw3_k = shared_w3[n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                        sh_gate_acc = pl.matmul_acc(sh_gate_acc, xs_k, sw1_k, b_trans=True)
                        sh_up_acc = pl.matmul_acc(sh_up_acc, xs_k, sw3_k, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_swiglu"):
                    sh_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(sh_gate_acc)), 1.0))
                    sh_silu = pl.mul(sh_gate_acc, sh_sigmoid)
                    sh_gated = pl.mul(sh_silu, sh_up_acc)
                    sh_tile[:, n0 : n0 + INTER_CHUNK] = pl.cast(sh_gated, target_type=pl.BF16)

            for d0 in pl.parallel(0, D, D_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_w2_matmul"):
                    hs_init = sh_tile[:, 0 : INTER_K]
                    sw2_init = shared_w2[d0 : d0 + D_OUT_CHUNK, 0 : INTER_K]
                    sh_y_acc = pl.matmul(hs_init, sw2_init, b_trans=True, out_dtype=pl.FP32)
                    for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                        hs_k = sh_tile[:, k0 : k0 + INTER_K]
                        sw2_k = shared_w2[d0 : d0 + D_OUT_CHUNK, k0 : k0 + INTER_K]
                        sh_y_acc = pl.matmul_acc(sh_y_acc, hs_k, sw2_k, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_write"):
                    sh[:, d0 : d0 + D_OUT_CHUNK] = pl.cast(sh_y_acc, target_type=pl.BF16)

            return recv_y, sh

    return DeepSeekV4DecodeMoEExpert
