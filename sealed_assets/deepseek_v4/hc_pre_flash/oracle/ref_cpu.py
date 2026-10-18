"""Independent CPU Oracle for DeepSeek-V4 HC pre-mix."""

import torch

from config import (
    B,
    D,
    HC_DIM,
    HC_DIM_INV,
    HC_EPS,
    HC_MULT,
    HC_SINKHORN_ITER,
    MIX_HC,
    NORM_EPS,
    S,
    T,
)


CHUNK = 128


def _device_bf16(value):
    rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
    return rounded.view(torch.float32).to(torch.bfloat16)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"hc_pre has no scalars: {sorted(scalars)}")

    x_source = inputs["x"]
    x = x_source.flatten(2).reshape(T, HC_DIM).to(torch.float32)
    weights = inputs["hc_fn"].to(torch.float32)

    square_sum = torch.zeros((T, 1), dtype=torch.float32)
    for start in range(0, HC_DIM, CHUNK):
        part = x[:, start : start + CHUNK]
        square_sum += (part * part).sum(dim=1, keepdim=True)
    inverse_rms = torch.rsqrt(square_sum * HC_DIM_INV + NORM_EPS)

    columns = []
    for row in range(MIX_HC):
        dot = torch.zeros((T, 1), dtype=torch.float32)
        for start in range(0, HC_DIM, CHUNK):
            dot += (
                x[:, start : start + CHUNK]
                * weights[row : row + 1, start : start + CHUNK]
            ).sum(dim=1, keepdim=True)
        columns.append(dot * inverse_rms)
    mixes = torch.cat(columns, dim=1).reshape(B, S, MIX_HC)

    scale = inputs["hc_scale"].to(torch.float32)
    base = inputs["hc_base"].to(torch.float32)
    pre = (
        torch.sigmoid(mixes[..., :HC_MULT] * scale[0] + base[:HC_MULT])
        + HC_EPS
    )
    post = 2 * torch.sigmoid(
        mixes[..., HC_MULT : 2 * HC_MULT] * scale[1]
        + base[HC_MULT : 2 * HC_MULT]
    )
    comb = (
        mixes[..., 2 * HC_MULT :] * scale[2] + base[2 * HC_MULT :]
    ).reshape(B, S, HC_MULT, HC_MULT)

    comb = torch.softmax(comb, dim=-1) + HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)

    lanes = x_source.to(torch.float32)
    mixed = torch.zeros((B, S, D), dtype=torch.float32)
    for lane in range(HC_MULT):
        mixed += lanes[:, :, lane] * pre[:, :, lane : lane + 1]

    return {"x_mixed": _device_bf16(mixed), "post": post, "comb": comb}
