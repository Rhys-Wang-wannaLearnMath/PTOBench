"""Independent CPU Oracle for DeepSeek-V4 HC head."""

import torch

from config import B, D, EPS, HC_DIM, HC_EPS, HC_MULT, S, T


CHUNK = 512


def _device_bf16(value):
    rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
    return rounded.view(torch.float32).to(torch.bfloat16)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"hc_head has no scalars: {sorted(scalars)}")
    x_source = inputs["x_hc"]
    x = x_source.reshape(T, HC_DIM).to(torch.float32)
    weights = inputs["hc_head_fn"].to(torch.float32)

    square_sum = torch.zeros((T, 1), dtype=torch.float32)
    for start in range(0, HC_DIM, CHUNK):
        part = x[:, start : start + CHUNK]
        square_sum += (part * part).sum(dim=1, keepdim=True)
    inverse_rms = torch.rsqrt(square_sum * (1.0 / HC_DIM) + EPS)

    columns = []
    for lane in range(HC_MULT):
        dot = torch.zeros((T, 1), dtype=torch.float32)
        for start in range(0, HC_DIM, CHUNK):
            dot += (
                x[:, start : start + CHUNK]
                * weights[lane : lane + 1, start : start + CHUNK]
            ).sum(dim=1, keepdim=True)
        columns.append(dot * inverse_rms)
    mixes = torch.cat(columns, dim=1).reshape(B, S, HC_MULT)
    gates = (
        torch.sigmoid(
            mixes * inputs["hc_head_scale"].float()
            + inputs["hc_head_base"].float()
        )
        + HC_EPS
    )
    lanes = x_source.float().reshape(B, S, HC_MULT, D)
    output = (
        lanes[:, :, 0] * gates[:, :, 0:1]
        + lanes[:, :, 1] * gates[:, :, 1:2]
    ) + (
        lanes[:, :, 2] * gates[:, :, 2:3]
        + lanes[:, :, 3] * gates[:, :, 3:4]
    )
    return {"y": _device_bf16(output)}
