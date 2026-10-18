# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""REF closure for deepseek_v4_flash_dspark.prefill_layer_swa_demo; fixed bindings from public config."""
from config import D
from config import HC_DIM
from config import HC_MULT
import pypto.language as pl

T_DYN = pl.dynamic("T_DYN")
T_TILE = 4

@pl.jit.inline
def hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    t_dim = pl.tensor.dim(x, 0)

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for token_block in pl.spmd(token_tiles, name_hint="hc_post"):
        t0 = token_block * T_TILE
        for t in pl.pipeline(t0, t0 + T_TILE, stage=2):
            if t < t_dim:
                # One cast per token: all HC_MULT outputs share the x row.
                x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
                for out_h in pl.unroll(HC_MULT):
                    post_w = pl.read(post, [t, out_h])
                    y_row = pl.mul(x_row, post_w)
                    for in_h in pl.pipeline(HC_MULT, stage=4):
                        comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                        res_d = in_h * D
                        # residual is already FP32 (hc stream is FP32 end-to-end): no cast, read straight.
                        res_row = residual_flat[t : t + 1, res_d : res_d + D]
                        weighted = pl.mul(res_row, comb_w)
                        y_row = pl.add(y_row, weighted)
                    # y is FP32 (feeds the next layer's hc_pre directly): no BF16 output cast.
                    y_flat[t : t + 1, out_h * D : out_h * D + D] = y_row
    return y


