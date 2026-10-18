# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections post-mix (dynamic shape): combines a sublayer
output with its hc-residual into the next hc-stack via post and comb weights.
Supports both decode and prefill batch/sequence sizes via dynamic-shape tensors."""

# ``hc_post`` processes a dynamic tensor whose rows are all valid (decode/MoE).
# ``hc_post_prefill`` processes only the active prefix of a static prefill tensor
# and deterministically zero-fills its inactive tail.


from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ

# Dynamic shape variables.

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim

# tiling
T_TILE = 4
INACTIVE_FILL_T_TILE = 16
INACTIVE_FILL_D_TILE = 256
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0



def golden_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    import torch

    x        = tensors["x"].float()
    residual = tensors["residual"].float()
    post     = tensors["post"].float()
    comb     = tensors["comb"].float().reshape(-1, HC_MULT, HC_MULT)  # [B, S, HC, HC]

    T = x.shape[0]
    y_fp32 = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, in_h, :] * comb[:, in_h, out_h:out_h + 1]
        y_fp32[:, out_h, :] = y_row
    y = y_fp32  # hc residual stream is FP32 end-to-end: no BF16 rounding

    tensors["y"][:] = y


def golden_hc_post_prefill(tensors):
    """Prefill reference: compute active rows and zero the static tail."""
    golden_hc_post(tensors)
    num_tokens = int(tensors["num_tokens"])
    tensors["y"][num_tokens:] = 0


