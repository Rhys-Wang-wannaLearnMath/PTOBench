"""Independent CPU Oracle for DeepSeek V4 HC-pre-v2 prefill."""

import torch

from config import (
    D,
    HC_DIM,
    HC_DIM_INV,
    HC_EPS,
    HC_MULT,
    HC_SINKHORN_ITER,
    MIX_HC,
    NORM_EPS,
    PREFILL_TOKENS,
)


RMS_CHUNK = 512
LINEAR_CHUNK = 256


def _device_bf16(value):
    rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
    return rounded.view(torch.float32).to(torch.bfloat16)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"hc_pre has no scalars: {sorted(scalars)}")

    source = inputs["x"]
    x = source.reshape(PREFILL_TOKENS, HC_DIM).to(torch.float32)
    weights = inputs["hc_fn"].to(torch.float32)

    square_sum = torch.zeros((PREFILL_TOKENS, 1), dtype=torch.float32)
    for start in range(0, HC_DIM, RMS_CHUNK):
        part = x[:, start : start + RMS_CHUNK]
        square_sum += (part * part).sum(dim=1, keepdim=True)
    inverse_rms = torch.rsqrt(square_sum * HC_DIM_INV + NORM_EPS)

    columns = []
    for row in range(MIX_HC):
        dot = torch.zeros((PREFILL_TOKENS, 1), dtype=torch.float32)
        for start in range(0, HC_DIM, LINEAR_CHUNK):
            dot += (
                x[:, start : start + LINEAR_CHUNK]
                * weights[row : row + 1, start : start + LINEAR_CHUNK]
            ).sum(dim=1, keepdim=True)
        columns.append(dot * inverse_rms)
    mixes = torch.cat(columns, dim=1)

    scale = inputs["hc_scale"].to(torch.float32)
    base = inputs["hc_base"].to(torch.float32)
    pre = (
        torch.sigmoid(mixes[:, :HC_MULT] * scale[0] + base[:HC_MULT])
        + HC_EPS
    )
    post = 2 * torch.sigmoid(
        mixes[:, HC_MULT : 2 * HC_MULT] * scale[1]
        + base[HC_MULT : 2 * HC_MULT]
    )
    comb = (
        mixes[:, 2 * HC_MULT :] * scale[2] + base[2 * HC_MULT :]
    ).reshape(PREFILL_TOKENS, HC_MULT, HC_MULT)

    comb = torch.softmax(comb, dim=-1) + HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)

    mixed = torch.zeros((PREFILL_TOKENS, D), dtype=torch.float32)
    for lane in range(HC_MULT):
        mixed += source[:, lane].to(torch.float32) * pre[:, lane : lane + 1]

    return {
        "x_mixed": _device_bf16(mixed),
        "post": post,
        "comb": comb.reshape(PREFILL_TOKENS, HC_MULT * HC_MULT),
    }

