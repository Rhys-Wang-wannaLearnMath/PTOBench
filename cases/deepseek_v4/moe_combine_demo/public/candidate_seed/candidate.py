"""DeepSeek-V4 packed weighted MoE combine NPU implementation."""

import pypto.language as pl

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

# Agent-editable implementation tiling.
COL_CHUNK = 512


@pl.jit.inline
def moe_combine(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[B, S, D], pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    routed_y_buf = pl.create_tensor([T * N_LOCAL_EXPERTS, D], dtype=pl.BF16)
    routed_w_buf = pl.create_tensor([T, N_LOCAL_EXPERTS], dtype=pl.FP32)
    ffn_out_flat = pl.reshape(ffn_out, [T, D])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="packed_combine_init"):
        for r0 in pl.range(0, T * N_LOCAL_EXPERTS, N_LOCAL_EXPERTS):
            for d0 in pl.range(0, D, COL_CHUNK):
                routed_y_buf[r0:r0 + N_LOCAL_EXPERTS, d0:d0 + COL_CHUNK] = pl.full(
                    [N_LOCAL_EXPERTS, COL_CHUNK], dtype=pl.BF16, value=0.0
                )
        for t in pl.range(T):
            for e in pl.range(N_LOCAL_EXPERTS):
                pl.write(routed_w_buf, [t, e], 0.0)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="packed_combine"):
        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(recv_expert_count, [e, 0]), pl.INDEX)
            for s in pl.range(n_rows):
                i = e * RECV_MAX + s
                t = pl.cast(pl.read(recv_token, [e, s]), pl.INDEX)
                dst = t * N_LOCAL_EXPERTS + e
                routed_y_buf = pl.assemble(
                    routed_y_buf,
                    pl.slice(recv_y_flat, [1, D], [i, 0]),
                    [dst, 0],
                )
                pl.write(routed_w_buf, [t, e], pl.read(recv_weights, [e, s]))
        for t in pl.range(T):
            base = t * N_LOCAL_EXPERTS
            for d0 in pl.range(0, D, COL_CHUNK):
                acc = pl.cast(
                    sh[t:t + 1, d0:d0 + COL_CHUNK], target_type=pl.FP32
                )
                for e in pl.range(N_LOCAL_EXPERTS):
                    row = pl.cast(
                        routed_y_buf[base + e:base + e + 1, d0:d0 + COL_CHUNK],
                        target_type=pl.FP32,
                    )
                    w = pl.read(routed_w_buf, [t, e])
                    acc = pl.add(acc, pl.mul(row, w))
                ffn_out_flat[t:t + 1, d0:d0 + COL_CHUNK] = pl.cast(
                    acc, target_type=pl.BF16, mode="rint"
                )


@pl.jit
def moe_combine_test(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
):
    moe_combine(recv_y, recv_token, recv_weights, recv_expert_count, sh, ffn_out)
    return ffn_out
