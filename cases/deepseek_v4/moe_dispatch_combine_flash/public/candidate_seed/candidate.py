"""DeepSeek-V4 fused local MoE dispatch, stand-in expert, and combine."""

import pypto.language as pl

from config import (
    CASE_CONFIG as M,
    DECODE_BATCH,
    DECODE_SEQ,
    EP_WORLD_SIZE,
    EXPERTS_START_IDX,
    RECV_MAX,
)


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

# Agent-editable implementation tiling.
COL_CHUNK = 128


@pl.jit.inline
def moe_dispatch_combine(
    x_norm: pl.Tensor[[T, D], pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_y: pl.Tensor[[T, D], pl.BF16],
):
    recv_x = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.BF16)
    recv_y = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.BF16)
    recv_weights = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX], dtype=pl.FP32)
    recv_token = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX], dtype=pl.INT32)
    routed_y_buf = pl.create_tensor([T * N_LOCAL_EXPERTS, D], dtype=pl.BF16)

    recv_weights_col = pl.reshape(recv_weights, [N_LOCAL_EXPERTS * RECV_MAX, 1])
    count_flat = pl.reshape(recv_expert_count, [N_LOCAL_EXPERTS])
    indices_flat = pl.reshape(indices, [T * TOPK])
    weights_flat = pl.reshape(weights, [T * TOPK])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch"):
        for t in pl.range(T):
            for k in pl.unroll(TOPK):
                p = t * TOPK + k
                e_global = pl.read(indices_flat, [p])
                e = pl.cast(e_global - EXPERTS_START_IDX, pl.INDEX)
                slot_i32 = pl.read(count_flat, [e])
                dst = e * RECV_MAX + pl.cast(slot_i32, pl.INDEX)

                recv_x = pl.assemble(
                    recv_x, pl.slice(x_norm, [1, D], [t, 0]), [dst, 0]
                )
                pl.write(recv_weights, [dst], pl.read(weights_flat, [p]))
                pl.write(recv_token, [dst], pl.cast(t, pl.INT32))
                pl.write(count_flat, [e], pl.cast(slot_i32 + 1, pl.INT32))

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_expert"):
        for r0 in pl.range(0, N_LOCAL_EXPERTS * RECV_MAX, RECV_MAX):
            for d0 in pl.range(0, D, COL_CHUNK):
                x_col = pl.cast(
                    recv_x[r0:r0 + RECV_MAX, d0:d0 + COL_CHUNK],
                    target_type=pl.FP32,
                )
                y_col = pl.row_expand_mul(
                    x_col, recv_weights_col[r0:r0 + RECV_MAX, :]
                )
                recv_y[r0:r0 + RECV_MAX, d0:d0 + COL_CHUNK] = pl.cast(
                    y_col, target_type=pl.BF16, mode="rint"
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="routed_y_buf_init"):
        for r0 in pl.range(0, T * N_LOCAL_EXPERTS, N_LOCAL_EXPERTS):
            for d0 in pl.range(0, D, COL_CHUNK):
                routed_y_buf[
                    r0:r0 + N_LOCAL_EXPERTS, d0:d0 + COL_CHUNK
                ] = pl.full(
                    [N_LOCAL_EXPERTS, COL_CHUNK], dtype=pl.BF16, value=0.0
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine"):
        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(count_flat, [e]), pl.INDEX)
            for s in pl.range(n_rows):
                i = e * RECV_MAX + s
                t = pl.cast(pl.read(recv_token, [i]), pl.INDEX)
                routed_y_buf = pl.assemble(
                    routed_y_buf,
                    pl.slice(recv_y, [1, D], [i, 0]),
                    [t * N_LOCAL_EXPERTS + e, 0],
                )
        for t in pl.range(T):
            base = t * N_LOCAL_EXPERTS
            for d0 in pl.range(0, D, COL_CHUNK):
                acc = pl.cast(
                    routed_y_buf[base:base + 1, d0:d0 + COL_CHUNK],
                    target_type=pl.FP32,
                )
                for e in pl.range(1, N_LOCAL_EXPERTS):
                    row = pl.cast(
                        routed_y_buf[
                            base + e:base + e + 1, d0:d0 + COL_CHUNK
                        ],
                        target_type=pl.FP32,
                    )
                    acc = pl.add(acc, row)
                routed_y[t:t + 1, d0:d0 + COL_CHUNK] = pl.cast(
                    acc, target_type=pl.BF16, mode="rint"
                )


@pl.jit
def moe_dispatch_combine_test(
    x_norm: pl.Tensor[[T, D], pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_expert_count: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]],
    routed_y: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    moe_dispatch_combine(x_norm, indices, weights, recv_expert_count, routed_y)
    return routed_y
