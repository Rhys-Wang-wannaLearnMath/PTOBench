# Copyright (c) PyPTO Contributors.
"""NPU Ref for DeepSeek-V4 packed dispatch."""

import pypto.language as pl

from config import (
    CASE_CONFIG as M,
    DECODE_BATCH,
    DECODE_SEQ,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    EP_WORLD_SIZE,
    EP_RANK,
    RECV_MAX,
)

B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS

# Agent-editable implementation tiling.
QUANT_CHUNK = 256


@pl.jit.inline
def moe_dispatch(
    x_norm: pl.Tensor[[T, D], pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
):
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_norm_q"):
        x_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, D, QUANT_CHUNK):
            x_f32 = pl.cast(x_norm[:, k0:k0 + QUANT_CHUNK], target_type=pl.FP32)
            x_abs = pl.maximum(x_f32, pl.neg(x_f32))
            x_max = pl.reshape(pl.row_max(x_abs), [1, T])
            x_amax = pl.maximum(x_amax, x_max)
        scale_q = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), x_amax)
        x_norm_scale_dq[:, 0:1] = pl.reshape(pl.recip(scale_q), [T, 1])
        scale_col = pl.reshape(scale_q, [T, 1])
        for k1 in pl.range(0, D, QUANT_CHUNK):
            x_f32 = pl.cast(x_norm[:, k1:k1 + QUANT_CHUNK], target_type=pl.FP32)
            x_scaled = pl.row_expand_mul(x_f32, scale_col)
            x_i32 = pl.cast(x_scaled, target_type=pl.INT32, mode="rint")
            x_half = pl.cast(x_i32, target_type=pl.FP16, mode="round")
            x_norm_i8[:, k1:k1 + QUANT_CHUNK] = pl.cast(x_half, target_type=pl.INT8, mode="trunc")

    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="packed_dispatch"):
        for e in pl.range(N_LOCAL_EXPERTS):
            for s in pl.range(RECV_MAX):
                pl.write(recv_scale_dq, [e, s], 0.0)
                pl.write(recv_weights, [e, s], 0.0)
                pl.write(recv_token, [e, s], pl.cast(0, pl.INT32))
            pl.write(recv_expert_count, [e, 0], pl.cast(0, pl.INT32))
        for t in pl.range(T):
            for k in pl.unroll(TOPK):
                e_global = pl.read(indices, [t, k])
                e = pl.cast(e_global - EXPERTS_START_IDX, pl.INDEX)
                slot_i32 = pl.read(recv_expert_count, [e, 0])
                slot = pl.cast(slot_i32, pl.INDEX)
                dst = e * RECV_MAX + slot
                recv_x_flat = pl.assemble(recv_x_flat, pl.slice(x_norm_i8, [1, D], [t, 0]), [dst, 0])
                pl.write(recv_scale_dq, [e, slot], pl.read(x_norm_scale_dq, [t, 0]))
                pl.write(recv_weights, [e, slot], pl.read(weights, [t, k]))
                pl.write(recv_token, [e, slot], pl.cast(t, pl.INT32))
                pl.write(recv_expert_count, [e, 0], pl.cast(slot_i32 + 1, pl.INT32))


@pl.jit
def moe_dispatch_test(
    x_norm: pl.Tensor[[T, D], pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8]],
    recv_scale_dq: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32]],
    recv_weights: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32]],
    recv_token: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32]],
    recv_expert_count: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]],
):
    moe_dispatch(x_norm, indices, weights, recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count)
    return recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count
