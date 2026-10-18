# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from config import (
    D,
    MAIN_HIDDEN_DIM,
)

import pypto.language as pl

from rmsnorm import rms_norm

T_DYN = pl.dynamic("DSPARK_PROJ_T_DYN")

T_TILE = 16                              # cube M-tile; matmul rows must be a multiple of 16

N_TILE = 128                             # 32 output blocks over D

K_TILE = 512                             # 24 reduction steps over MAIN_HIDDEN_DIM

@pl.jit.inline
def dspark_proj(
    main_hidden: pl.Tensor[[T_DYN, MAIN_HIDDEN_DIM], pl.BF16],
    main_proj_w: pl.Tensor[[D, MAIN_HIDDEN_DIM], pl.BF16],
    main_norm_w: pl.Tensor[[D], pl.BF16],
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(main_hidden, 0)
    t_matmul = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE

    projected = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    for proj_idx in pl.spmd(
        (t_matmul // T_TILE) * (D // N_TILE), name_hint="dspark_main_proj", allow_early_resolve=True
    ):
        t0 = (proj_idx // (D // N_TILE)) * T_TILE
        n0 = (proj_idx % (D // N_TILE)) * N_TILE
        proj_rows = pl.min(T_TILE, t_dim - t0)
        hidden_k0 = pl.slice(main_hidden, [T_TILE, K_TILE], [t0, 0], valid_shape=[proj_rows, K_TILE])
        weight_k0 = main_proj_w[n0 : n0 + N_TILE, 0:K_TILE]
        proj_acc = pl.matmul(hidden_k0, weight_k0, b_trans=True, out_dtype=pl.FP32)
        for k0 in pl.pipeline(K_TILE, MAIN_HIDDEN_DIM, K_TILE, stage=2):
            hidden_k = pl.slice(main_hidden, [T_TILE, K_TILE], [t0, k0], valid_shape=[proj_rows, K_TILE])
            weight_k = main_proj_w[n0 : n0 + N_TILE, k0 : k0 + K_TILE]
            proj_acc = pl.matmul_acc(proj_acc, hidden_k, weight_k, b_trans=True)
        proj_bf16 = pl.cast(proj_acc, target_type=pl.BF16, mode="rint")
        proj_valid = pl.set_validshape(proj_bf16, proj_rows, N_TILE)
        projected[t0 : t0 + T_TILE, n0 : n0 + N_TILE] = proj_valid

    rms_norm(projected, main_norm_w, main_x)
    return main_x

