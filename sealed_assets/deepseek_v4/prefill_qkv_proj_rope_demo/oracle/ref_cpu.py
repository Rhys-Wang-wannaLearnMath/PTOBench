"""Independent CPU Oracle for fixed-shape prefill QKV projection and RoPE."""

import torch

from config import (
    B,
    D,
    EPS,
    H,
    HEAD_DIM,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    MAX_SEQ_LEN,
    NOPE_DIM,
    Q_LORA,
    ROPE_DIM,
    ROPE_HALF,
    S,
    T,
)


def _rms(value, gamma):
    return (
        value
        * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + EPS)
        * gamma
    )


def _matmul_bf16(left, right):
    return left.to(torch.bfloat16).float().matmul(
        right.to(torch.bfloat16).float()
    )


def _quantize_rows(value):
    rows = value.reshape(-1, value.shape[-1]).float()
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    multiplier = INT8_SCALE_MAX / amax
    quantized = (
        torch.round(rows * multiplier)
        .to(torch.int32)
        .to(torch.float16)
        .to(torch.int8)
        .reshape_as(value)
    )
    scale = (1.0 / multiplier).reshape(*value.shape[:-1], 1)
    return quantized, scale


def _rope(value, cosine, sine):
    pairs = value.unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    cosine = cosine[..., :ROPE_HALF]
    sine = sine[..., :ROPE_HALF]
    while cosine.ndim < even.ndim:
        cosine = cosine.unsqueeze(-2)
        sine = sine.unsqueeze(-2)
    return torch.stack(
        (
            (even * cosine - odd * sine).to(torch.bfloat16),
            (even * sine + odd * cosine).to(torch.bfloat16),
        ),
        dim=-1,
    ).flatten(-2)


def compute(inputs, scalars):
    if set(scalars) != {"start_pos"}:
        raise ValueError(
            f"expected only start_pos scalar, got {sorted(scalars)}"
        )
    start_pos = int(scalars["start_pos"])
    if not 0 <= start_pos <= MAX_SEQ_LEN - S:
        raise ValueError(
            f"start_pos {start_pos} outside [0, {MAX_SEQ_LEN - S}]"
        )

    positions = torch.arange(start_pos, start_pos + S)
    cosine = (
        inputs["freqs_cos"]
        .index_select(0, positions)
        .reshape(B, S, ROPE_DIM)
        .reshape(T, ROPE_DIM)
        .float()
    )
    sine = (
        inputs["freqs_sin"]
        .index_select(0, positions)
        .reshape(B, S, ROPE_DIM)
        .reshape(T, ROPE_DIM)
        .float()
    )

    normalized_x = _rms(
        inputs["x"].float(), inputs["norm_w"].float()
    ).to(torch.bfloat16).float().reshape(T, D)
    lora = _rms(
        _matmul_bf16(normalized_x, inputs["wq_a"]),
        inputs["gamma_cq"].float(),
    )
    qr, qr_scale = _quantize_rows(lora.to(torch.bfloat16).float())

    q_accumulator = qr.to(torch.int32).matmul(
        inputs["wq_b"].to(torch.int32)
    )
    q_full = (
        q_accumulator.float()
        * qr_scale
        * inputs["wq_b_scale"].float().reshape(1, H * HEAD_DIM)
    ).reshape(T, H, HEAD_DIM)
    q_full = q_full * torch.rsqrt(
        q_full.square().mean(dim=-1, keepdim=True) + EPS
    )
    q = torch.cat(
        (
            q_full[..., :NOPE_DIM],
            _rope(q_full[..., NOPE_DIM:], cosine, sine),
        ),
        dim=-1,
    )

    kv_full = _rms(
        _matmul_bf16(normalized_x, inputs["wkv"]),
        inputs["gamma_ckv"].float(),
    )
    kv = torch.cat(
        (
            kv_full[..., :NOPE_DIM],
            _rope(
                kv_full[..., NOPE_DIM:].unsqueeze(1), cosine, sine
            ).squeeze(1),
        ),
        dim=-1,
    )
    return {
        "q": q.to(torch.bfloat16),
        "kv": kv.to(torch.bfloat16),
        "qr": qr.reshape(T, Q_LORA),
        "qr_scale": qr_scale,
    }

