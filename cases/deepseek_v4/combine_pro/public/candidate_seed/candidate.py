import pypto.language as pl

from config import (
    B, D, S, T, N_LOCAL_EXPERTS, RECV_MAX,
)


@pl.jit.inline
def combine(recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
            recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
            recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
            sh: pl.Tensor[[T, D], pl.BF16],
            ffn_out: pl.Tensor[[B, S, D], pl.BF16]):
    ffn_out_flat = pl.reshape(ffn_out, [T, D])
    routed_y_buf = pl.create_tensor([T, N_LOCAL_EXPERTS, D], dtype=pl.BF16)
    routed_y_buf_flat = pl.reshape(routed_y_buf, [T * N_LOCAL_EXPERTS, D])
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_scatter"):
        for rb in pl.range(T * N_LOCAL_EXPERTS // 16):
            r0 = rb * 16
            for db in pl.range(2):
                d0 = db * (D // 2)
                routed_y_buf_flat[r0:r0 + 16, d0:d0 + D // 2] = pl.full(
                    [16, D // 2], dtype=pl.BF16, value=0.0
                )
        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(recv_expert_count, [e, 0]), pl.INDEX)
            for s in pl.range(n_rows):
                i = e * RECV_MAX + s
                t = pl.cast(pl.read(recv_token, [e, s]), pl.INDEX)
                dst = t * N_LOCAL_EXPERTS + e
                routed_y_buf_flat[dst:dst + 1, :] = recv_y_flat[i:i + 1, :]
    for tb in pl.spmd(T // 4, name_hint="combine_reduce"):
        for tt in pl.range(4):
            t = tb * 4 + tt
            base = t * N_LOCAL_EXPERTS
            acc = pl.cast(sh[t:t + 1, :], target_type=pl.FP32)
            # D=7168 exceeds the A2/A3 vector-buffer limit with two pipeline stages.
            for e in pl.pipeline(N_LOCAL_EXPERTS, stage=1):
                row_fp32 = pl.cast(routed_y_buf_flat[base + e:base + e + 1, :], target_type=pl.FP32)
                acc = pl.add(acc, row_fp32)
            ffn_out_flat[t:t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")


@pl.jit
def combine_test(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
):
    combine(recv_y, recv_token, recv_expert_count, sh, ffn_out)
    return ffn_out
