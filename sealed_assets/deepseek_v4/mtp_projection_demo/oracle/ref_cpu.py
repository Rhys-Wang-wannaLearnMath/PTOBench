"""Candidate-external CPU Oracle for quantized MTP projection."""

import torch
import config

T, D = config.DECODE_TOKENS, config.D
CHUNK = config.D_CHUNK
EPS = config.EPS
AMAX_EPS = config.INT8_AMAX_EPS
SCALE_MAX = config.INT8_SCALE_MAX


def _norm(x, weight):
    flat = x.reshape(T, D).to(torch.float32)
    square_sum = torch.zeros((T, 1), dtype=torch.float32)
    for start in range(0, D, CHUNK):
        part = flat[:, start:start + CHUNK]
        square_sum += (part * part).sum(dim=1, keepdim=True)
    inverse = torch.rsqrt(square_sum * (1.0 / D) + EPS)
    return (flat * inverse * weight.to(torch.float32).reshape(1, D)).reshape_as(x)


def _quantize_rows(value):
    amax = value.abs().amax(dim=-1, keepdim=True).clamp_min(AMAX_EPS)
    quant_scale = SCALE_MAX / amax
    integer = torch.round(value * quant_scale).to(torch.int32)
    return integer.to(torch.float16).to(torch.int8), 1.0 / quant_scale


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"mtp_projection has no scalars: {sorted(scalars)}")
    e_norm = (_norm(inputs["hidden_states"], inputs["enorm_w"]) * inputs["e_proj_smooth"].float()).to(torch.bfloat16)
    h_norm = (_norm(inputs["prev_hidden_states"], inputs["hnorm_w"]) * inputs["h_proj_smooth"].float()).to(torch.bfloat16)
    e_i8, e_row_scale = _quantize_rows(e_norm.float())
    h_i8, h_row_scale = _quantize_rows(h_norm.float())
    e_acc = e_i8.to(torch.int32).matmul(inputs["e_proj_w"].to(torch.int32).transpose(0, 1)).float()
    h_acc = h_i8.to(torch.int32).matmul(inputs["h_proj_w"].to(torch.int32).transpose(0, 1)).float()
    e_value = e_acc * e_row_scale * inputs["e_proj_w_scale"].float().reshape(1, 1, D)
    h_value = h_acc * h_row_scale * inputs["h_proj_w_scale"].float().reshape(1, 1, D)
    return {"hidden_states_out": (e_value + h_value).to(torch.bfloat16)}
