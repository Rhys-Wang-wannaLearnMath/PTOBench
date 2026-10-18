# Copyright (c) PyPTO Contributors.
import pypto.language as pl

from config import B, S, D, HC_MULT, HC_DIM


T = B * S
D_CHUNK = 512
D_BLOCKS = D // D_CHUNK
HC_POST_TOKEN_TILE = 16
HC_POST_TOKEN_BLOCKS = T // HC_POST_TOKEN_TILE
HC_POST_SPMD_BLOCKS = HC_MULT * HC_POST_TOKEN_BLOCKS * D_BLOCKS
assert T % HC_POST_TOKEN_TILE == 0, "T must be divisible by HC_POST_TOKEN_TILE"
assert D % D_CHUNK == 0, "D must be divisible by D_CHUNK"


@pl.jit.inline
def prefill_hc_post(
    x: pl.Tensor[[B, S, D], pl.BF16],
    residual: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[B, S, HC_MULT], pl.FP32],
    comb: pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
    y: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
):
    x_flat = pl.reshape(x, [T, D])
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    post_t = pl.reshape(post, [T, HC_MULT])
    comb_t = pl.reshape(comb, [T, HC_MULT * HC_MULT])
    y_flat = pl.reshape(y, [T, HC_DIM])

    for block in pl.spmd(HC_POST_SPMD_BLOCKS, name_hint="prefill_hc_post"):
        db = block % D_BLOCKS
        token_block_and_h = block // D_BLOCKS
        token_block = token_block_and_h % HC_POST_TOKEN_BLOCKS
        out_h = token_block_and_h // HC_POST_TOKEN_BLOCKS
        t0 = token_block * HC_POST_TOKEN_TILE
        d0 = db * D_CHUNK

        for dt in pl.range(HC_POST_TOKEN_TILE):
            t = t0 + dt
            post_w = pl.read(post_t, [t, out_h])
            comb0_w = pl.read(comb_t, [t, 0 * HC_MULT + out_h])
            comb1_w = pl.read(comb_t, [t, 1 * HC_MULT + out_h])
            comb2_w = pl.read(comb_t, [t, 2 * HC_MULT + out_h])
            comb3_w = pl.read(comb_t, [t, 3 * HC_MULT + out_h])
            x_row = pl.cast(
                pl.slice(x_flat, [1, D_CHUNK], [t, d0]),
                target_type=pl.FP32,
            )
            y_row = pl.mul(x_row, post_w)
            residual0 = pl.cast(
                pl.slice(residual_flat, [1, D_CHUNK], [t, 0 * D + d0]),
                target_type=pl.FP32,
            )
            residual1 = pl.cast(
                pl.slice(residual_flat, [1, D_CHUNK], [t, 1 * D + d0]),
                target_type=pl.FP32,
            )
            residual2 = pl.cast(
                pl.slice(residual_flat, [1, D_CHUNK], [t, 2 * D + d0]),
                target_type=pl.FP32,
            )
            residual3 = pl.cast(
                pl.slice(residual_flat, [1, D_CHUNK], [t, 3 * D + d0]),
                target_type=pl.FP32,
            )
            y_row = pl.add(y_row, pl.mul(residual0, comb0_w))
            y_row = pl.add(y_row, pl.mul(residual1, comb1_w))
            y_row = pl.add(y_row, pl.mul(residual2, comb2_w))
            y_row = pl.add(y_row, pl.mul(residual3, comb3_w))
            y_flat[
                t:t + 1,
                out_h * D + d0:out_h * D + d0 + D_CHUNK,
            ] = pl.cast(y_row, target_type=pl.BF16, mode="rint")
    return pl.reshape(y_flat, [B, S, HC_MULT, D])


@pl.jit
def prefill_hc_post_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    residual: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[B, S, HC_MULT], pl.FP32],
    comb: pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
):
    return prefill_hc_post(x, residual, post, comb, y)

