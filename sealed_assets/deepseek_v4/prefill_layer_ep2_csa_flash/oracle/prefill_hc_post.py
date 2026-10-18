# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections post-mix for prefill attention."""


from config import CASE_CONFIG as M, PREFILL_BATCH, PREFILL_SEQ


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim

# tiling
D_CHUNK = 512
D_BLOCKS = D // D_CHUNK
HC_POST_TOKEN_TILE = 16
HC_POST_TOKEN_BLOCKS = T // HC_POST_TOKEN_TILE
HC_POST_SPMD_BLOCKS = HC_MULT * HC_POST_TOKEN_BLOCKS * D_BLOCKS
assert T % HC_POST_TOKEN_TILE == 0, "T must be divisible by HC_POST_TOKEN_TILE"






def golden_prefill_hc_post(tensors):
    import torch

    x = tensors["x"].float()
    residual = tensors["residual"].float()
    post = tensors["post"].float()
    comb = tensors["comb"].float()

    y_fp32 = torch.zeros(B, S, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, :, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, :, in_h, :] * comb[:, :, in_h, out_h:out_h + 1]
        y_fp32[:, :, out_h, :] = y_row
    tensors["y"][:] = y_fp32.to(torch.bfloat16)




