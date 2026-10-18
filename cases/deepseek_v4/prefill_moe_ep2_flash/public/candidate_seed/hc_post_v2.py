import pypto.language as pl
from config import D, HC_MULT, HC_DIM, PREFILL_BATCH, PREFILL_SEQ
T_DYN = pl.dynamic("T_DYN")


T_TILE = 4

assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0

@pl.jit.inline
def hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    t_dim = pl.tensor.dim(x, 0)

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    for block in pl.spmd((t_dim // T_TILE) * HC_MULT, name_hint="hc_post"):
        token_block = block // HC_MULT
        out_h = block % HC_MULT
        t0 = token_block * T_TILE
        for t in pl.pipeline(t0, t0 + T_TILE, stage=2):
            post_w = pl.read(post, [t, out_h])
            x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
            y_row = pl.mul(x_row, post_w)
            for in_h in pl.pipeline(HC_MULT, stage=4):
                comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                res_d = in_h * D
                res_row = pl.cast(residual_flat[t : t + 1, res_d : res_d + D], target_type=pl.FP32)
                weighted = pl.mul(res_row, comb_w)
                y_row = pl.add(y_row, weighted)
            y_out = pl.cast(y_row, target_type=pl.BF16, mode="rint")
            y_flat[t : t + 1, out_h * D : out_h * D + D] = y_out
    return y
