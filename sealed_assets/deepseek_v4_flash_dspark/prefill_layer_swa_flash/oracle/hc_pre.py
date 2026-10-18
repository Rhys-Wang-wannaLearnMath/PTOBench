# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ORACLE closure for deepseek_v4_flash_dspark.prefill_layer_swa_flash; fixed bindings from public config."""
from config import D
from config import HC_DIM
from config import HC_DIM_INV
from config import HC_EPS
from config import HC_MULT
from config import HC_SINKHORN_ITER
from config import MIX_HC
from config import NORM_EPS

RMS_K_TILE = 512
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK

def _golden_a2a3_cube_linear(x_flat_2d, hc_fn):
    """Emulate the A2/A3 f32 Cube MAD reduction used by the HC projection."""
    import torch

    cube_acc_k = 4
    group_block = 64
    assert LINEAR_K_PER_SPLIT % cube_acc_k == 0
    groups_per_split = LINEAR_K_PER_SPLIT // cube_acc_k

    t_dim = x_flat_2d.shape[0]
    x_groups = x_flat_2d.reshape(t_dim, 1, LINEAR_OK, groups_per_split, cube_acc_k)
    w_groups = hc_fn.reshape(1, MIX_HC, LINEAR_OK, groups_per_split, cube_acc_k)
    split_mixes = torch.zeros(t_dim, MIX_HC, LINEAR_OK, dtype=torch.float32)
    for group0 in range(0, groups_per_split, group_block):
        x_block = x_groups[..., group0:group0 + group_block, :].double()
        w_block = w_groups[..., group0:group0 + group_block, :].double()
        group_dots = (x_block * w_block).sum(dim=-1)
        for group in range(group_dots.shape[-1]):
            split_mixes = (split_mixes.double() + group_dots[..., group]).float()

    # Match the kernel's ascending split order.
    mixes = torch.zeros(t_dim, MIX_HC, dtype=torch.float32)
    for split in range(LINEAR_OK):
        mixes += split_mixes[..., split]
    return mixes


def golden_hc_pre(tensors):
    """Torch reference matching the target HC-pre reduction and nonlinearities."""
    import torch

    x = tensors["x"].float()  # [T, hc, D]
    hc_fn = tensors["hc_fn"].float()  # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()  # [3]
    hc_base = tensors["hc_base"].float()  # [mix_hc]

    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)

    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)

    # A2/A3 f32 Cube MAD accumulates four consecutive products before rounding
    # the updated accumulator to FP32.  Reproduce that target-defined reduction
    # instead of using Torch's different reduction tree.
    mixes = _golden_a2a3_cube_linear(x_flat_2d, hc_fn)
    mixes *= rsqrt

    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(t_dim, HC_MULT, HC_MULT)

    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = (y0 + y1) + (y2 + y3)

    # Match the kernel's mode="rint" cast (round to nearest, ties to even).
    tensors["x_mixed"][:] = y.to(torch.bfloat16).reshape(t_dim, D)
    tensors["post"][:] = post_t.reshape(t_dim, HC_MULT)
    tensors["comb"][:] = comb_t.reshape(t_dim, HC_MULT * HC_MULT)


