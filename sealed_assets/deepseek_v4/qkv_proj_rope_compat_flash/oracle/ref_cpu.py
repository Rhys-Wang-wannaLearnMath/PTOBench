"""Independent CPU oracle for one invocation of the compatibility workload."""

import torch

from config import (
    D, EPS, H, HEAD_DIM, INT8_AMAX_EPS, INT8_SCALE_MAX, NOPE_DIM,
    Q_LORA, ROPE_HALF, T,
)


def _rms(value, gamma):
    inverse = torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)
    return value * inverse * gamma


def _bf16_matmul(left, right):
    return torch.matmul(left.to(torch.bfloat16).float(), right.to(torch.bfloat16).float()).float()


def _quantize_rows(value):
    rows = value.reshape(-1, value.shape[-1]).float()
    amax = rows.abs().amax(-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    quant_scale = INT8_SCALE_MAX / amax
    integer = torch.round(rows * quant_scale).to(torch.int32)
    quantized = integer.to(torch.float16).to(torch.int8).reshape_as(value)
    return quantized, (1.0 / quant_scale).reshape(*value.shape[:-1], 1)


def _rope(value, cos, sin):
    pairs = value.unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    cos_value, sin_value = cos[..., :ROPE_HALF], sin[..., :ROPE_HALF]
    while cos_value.ndim < even.ndim:
        cos_value, sin_value = cos_value.unsqueeze(-2), sin_value.unsqueeze(-2)
    rotated_even = (even * cos_value - odd * sin_value).to(torch.bfloat16)
    rotated_odd = (even * sin_value + odd * cos_value).to(torch.bfloat16)
    return torch.stack((rotated_even, rotated_odd), -1).flatten(-2)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"qkv_proj_rope compatibility Case has no scalars: {sorted(scalars)}")
    # even_select_t and odd_select_t are retained in the fixed ABI but the
    # selected gather/scatter RoPE implementation does not consume them.
    token_x = _rms(inputs["x"].float().reshape(T, D), inputs["norm_w"].float())
    qr_float = _rms(_bf16_matmul(token_x, inputs["wq_a"].float()), inputs["gamma_cq"].float())
    qr, qr_scale = _quantize_rows(qr_float.to(torch.bfloat16).float())
    q_integer = torch.matmul(qr.to(torch.int32), inputs["wq_b"].to(torch.int32))
    q_full = (
        q_integer.float() * qr_scale * inputs["wq_b_scale"].float().reshape(1, -1)
    ).reshape(T, H, HEAD_DIM)
    q_full = q_full * torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q = torch.cat((q_full[..., :NOPE_DIM], _rope(
        q_full[..., NOPE_DIM:], inputs["rope_cos"].float(), inputs["rope_sin"].float(),
    )), -1)
    kv_full = _rms(_bf16_matmul(token_x, inputs["wkv"].float()), inputs["gamma_ckv"].float())
    kv_rope = _rope(
        kv_full[..., NOPE_DIM:].unsqueeze(1), inputs["rope_cos"].float(), inputs["rope_sin"].float(),
    ).squeeze(1)
    kv = torch.cat((kv_full[..., :NOPE_DIM], kv_rope), -1)
    return {"q": q.to(torch.bfloat16), "kv": kv.to(torch.bfloat16), "qr": qr, "qr_scale": qr_scale}
