# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pypto.language as pl
from config import B, S, D, HC_MULT, T, D_CHUNK, D_BLOCKS, HC_DIM


@pl.jit.inline
def deepseek_v4_decode_hc_post(
    x:        pl.Tensor[[B, S, D],                    pl.BF16],
    residual: pl.Tensor[[B, S, HC_MULT, D],           pl.BF16],
    post:     pl.Tensor[[B, S, HC_MULT],              pl.FP32],
    comb:     pl.Tensor[[B, S, HC_MULT, HC_MULT],     pl.FP32],
    y:        pl.Out[pl.Tensor[[B, S, HC_MULT, D],    pl.BF16]],
):
    x_flat = pl.reshape(x, [T, D])
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    post_flat = pl.reshape(post, [T * HC_MULT])
    comb_flat = pl.reshape(comb, [T * HC_MULT * HC_MULT])
    y_flat = pl.reshape(y, [T, HC_DIM])

    for out_h in pl.parallel(HC_MULT):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_post"):
            for t in pl.parallel(0, T, 1):
                post_w = pl.read(post_flat, [t * HC_MULT + out_h])
                for db in pl.range(D_BLOCKS):
                    d0 = db * D_CHUNK
                    x_row = pl.cast(
                        pl.slice(x_flat, [1, D_CHUNK], [t, d0]),
                        target_type=pl.FP32,
                    )
                    y_row = pl.mul(x_row, post_w)
                    for in_h in pl.range(HC_MULT):
                        comb_w = pl.read(
                            comb_flat,
                            [t * HC_MULT * HC_MULT + in_h * HC_MULT + out_h],
                        )
                        residual_row = pl.cast(
                            pl.slice(residual_flat, [1, D_CHUNK], [t, in_h * D + d0]),
                            target_type=pl.FP32,
                        )
                        y_row = pl.add(y_row, pl.mul(residual_row, comb_w))
                    y_flat = pl.assemble(
                        y_flat,
                        pl.cast(y_row, target_type=pl.BF16, mode="rint"),
                        [t, out_h * D + d0],
                    )
    y = pl.reshape(y_flat, [B, S, HC_MULT, D])
    return y


@pl.jit
def deepseek_v4_decode_hc_post_test(
    x:        pl.Tensor[[B, S, D],                    pl.BF16],
    residual: pl.Tensor[[B, S, HC_MULT, D],           pl.BF16],
    post:     pl.Tensor[[B, S, HC_MULT],              pl.FP32],
    comb:     pl.Tensor[[B, S, HC_MULT, HC_MULT],     pl.FP32],
    y:        pl.Out[pl.Tensor[[B, S, HC_MULT, D],    pl.BF16]],
):
    y = deepseek_v4_decode_hc_post(x, residual, post, comb, y)
    return y
