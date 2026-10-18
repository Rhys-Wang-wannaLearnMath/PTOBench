import pypto.language as pl

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

D = M.hidden_size

HC_MULT = M.hc_mult

HC_DIM = M.hc_dim

D_TILE = 512

@pl.jit.inline
def hc_post(x: pl.Tensor[[B, S, D], pl.BF16], residual: pl.Tensor[[B, S, HC_MULT, D], pl.BF16], post: pl.Tensor[[B, S, HC_MULT], pl.FP32], comb: pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32], y: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]]):
    x_flat = pl.reshape(x, [T, D])
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    post_t = pl.reshape(post, [T, HC_MULT])
    comb_t = pl.reshape(comb, [T, HC_MULT * HC_MULT])
    y_flat = pl.reshape(y, [T, HC_DIM])
    for tb in pl.spmd(T // 2, name_hint='hc_post'):
        for tt in pl.range(2):
            t = tb * 2 + tt
            for out_h in pl.range(HC_MULT):
                post_w = pl.read(post_t, [t, out_h])
                y_d = out_h * D
                for d0 in pl.range(0, D, D_TILE):
                    x_chunk = pl.cast(x_flat[t:t + 1, d0:d0 + D_TILE], target_type=pl.FP32)
                    y_chunk = pl.mul(x_chunk, post_w)
                    for in_h in pl.range(HC_MULT):
                        comb_w = pl.read(comb_t, [t, in_h * HC_MULT + out_h])
                        res_d = in_h * D + d0
                        residual_chunk = pl.cast(residual_flat[t:t + 1, res_d:res_d + D_TILE], target_type=pl.FP32)
                        y_chunk = pl.add(y_chunk, pl.mul(residual_chunk, comb_w))
                    y_flat[t:t + 1, y_d + d0:y_d + d0 + D_TILE] = pl.cast(y_chunk, target_type=pl.BF16, mode='rint')
    y = pl.reshape(y_flat, [B, S, HC_MULT, D])
    return y
