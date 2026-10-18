# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections pre-mix for prefill attention."""


from config import CASE_CONFIG as M, PREFILL_BATCH, PREFILL_SEQ


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps

# kernel-local
MIX_PAD = 32
HC_PAD = 8

# tiling
T_TILE = 16
RMS_T_TILE = 16
LINEAR_T_TILE = 16
COMB_T_TILE = 16
RMS_K_CHUNK = 128
LINEAR_K_CHUNK = 512
D_CHUNK = 512
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK
RMS_PIPE_STAGE = 1 if T >= 64 else 4






def golden_prefill_hc_pre(tensors):
    import torch

    x = tensors["x"].float()
    hc_fn = tensors["hc_fn"].float()
    hc_scale = tensors["hc_scale"].float()
    hc_base = tensors["hc_base"].float()

    x_flat = x.flatten(2)
    x_flat_2d = x_flat.reshape(T, HC_DIM)
    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)

    mix_cols = []
    for m in range(MIX_HC):
        mix_col = torch.zeros(T, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_CHUNK):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_CHUNK]
            w_chunk = hc_fn[m:m + 1, k0:k0 + LINEAR_K_CHUNK]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes_2d = torch.cat(mix_cols, dim=1)
    mixes = mixes_2d.reshape(B, S, MIX_HC)

    pre_logits = mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]
    pre = torch.sigmoid(pre_logits) + HC_EPS

    post_logits = (
        mixes_2d[:, HC_MULT:HC_MULT + HC_PAD] * hc_scale[1]
        + hc_base[HC_MULT:HC_MULT + HC_PAD]
    )
    post = (2 * torch.sigmoid(post_logits[:, :HC_MULT])).reshape(B, S, HC_MULT)

    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]).view(
        B, S, HC_MULT, HC_MULT
    )
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y01 = x[:, :, 0, :] * pre[:, :, 0:1] + x[:, :, 1, :] * pre[:, :, 1:2]
    y23 = x[:, :, 2, :] * pre[:, :, 2:3] + x[:, :, 3, :] * pre[:, :, 3:4]
    y = y01 + y23

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["x_mixed"][:] = _to_device_bf16(y)
    tensors["post"][:] = post
    tensors["comb"][:] = comb_t




