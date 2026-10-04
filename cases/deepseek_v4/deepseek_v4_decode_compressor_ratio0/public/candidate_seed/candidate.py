"""NPU Ref for the fixed DeepSeek-V4 ratio-0 decode compressor."""

import pypto.language as pl

from config import (
    APE_SHAPE,
    B,
    COFF,
    COMPRESS_RATIO,
    D,
    EPS,
    HADAMARD_SHAPE,
    HEAD_DIM,
    NOPE_HEAD_DIM,
    OUT_DIM,
    ROPE_HEAD_DIM,
    S,
    STATE_LEN,
    STATE_SHAPE,
    W_SHAPE,
    X_SHAPE,
)


# HEAD_CHUNK is an implementation tile and remains editable after publication.
HEAD_CHUNK = 128
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK


@pl.jit.inline
def deepseek_v4_decode_compressor(
    x: pl.Tensor[X_SHAPE, pl.BF16],
    kv_state: pl.Tensor[STATE_SHAPE, pl.FP32],
    score_state: pl.Tensor[STATE_SHAPE, pl.FP32],
    wkv: pl.Tensor[W_SHAPE, pl.BF16],
    wgate: pl.Tensor[W_SHAPE, pl.BF16],
    ape: pl.Tensor[APE_SHAPE, pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[HADAMARD_SHAPE, pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Tensor[[B, OUT_DIM], pl.BF16],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="zeros"):
        for ob in pl.parallel(HEAD_BLOCKS):
            h0 = ob * HEAD_CHUNK
            zero_fp32 = pl.full([B, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
            out = pl.assemble(out, pl.cast(zero_fp32, target_type=pl.BF16), [0, h0])
    return out


@pl.jit
def deepseek_v4_decode_compressor_test(
    x: pl.Tensor[X_SHAPE, pl.BF16],
    kv_state: pl.InOut[pl.Tensor[STATE_SHAPE, pl.FP32]],
    score_state: pl.InOut[pl.Tensor[STATE_SHAPE, pl.FP32]],
    wkv: pl.Tensor[W_SHAPE, pl.BF16],
    wgate: pl.Tensor[W_SHAPE, pl.BF16],
    ape: pl.Tensor[APE_SHAPE, pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[HADAMARD_SHAPE, pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[B, OUT_DIM], pl.BF16]],
):
    return deepseek_v4_decode_compressor(
        x, kv_state, score_state, wkv, wgate, ape, norm_w, cos, sin,
        hadamard, start_pos, out,
    )
