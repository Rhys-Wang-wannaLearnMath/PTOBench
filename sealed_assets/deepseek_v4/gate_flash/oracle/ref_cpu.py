"""Independent CPU Oracle for the selected DeepSeek-V4 gate preset."""

import torch

from config import (
    D,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    N_EXPERTS,
    N_HASH_LAYERS,
    NORM_EPS,
    ROUTE_SCALE,
    T,
    TOPK,
)


def _quantize_tokens(x_bf16):
    x = x_bf16.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    quantized = (
        torch.round(x * scale_quant)
        .to(torch.int32)
        .to(torch.float16)
        .to(torch.int8)
    )
    return quantized, (1.0 / scale_quant).reshape(-1)


def compute(inputs, scalars):
    if set(scalars) != {"layer_id", "num_tokens"}:
        raise ValueError(f"unexpected gate scalars: {sorted(scalars)}")
    layer_id = int(scalars["layer_id"])
    num_tokens = max(0, min(T, int(scalars["num_tokens"])))
    x = inputs["x_mixed"].float().view(T, D)
    norm_w = inputs["norm_w"].float()
    square_sum = (x * x).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(square_sum * (1.0 / D) + NORM_EPS)
    x_norm = ((x * inv_rms) * norm_w.view(1, D)).to(torch.bfloat16)
    x_norm_i8, x_norm_scale = _quantize_tokens(x_norm)
    logits = x_norm.float() @ inputs["gate_w"].float().T
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + inputs["gate_bias"].float().view(1, N_EXPERTS)
    if layer_id < N_HASH_LAYERS:
        indices = inputs["tid2eid"][inputs["input_ids"].flatten().long()]
    else:
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]
    topk_scores = torch.gather(scores, dim=-1, index=indices.long())
    weights = (
        topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    ) * ROUTE_SCALE
    if num_tokens < T:
        x_norm_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0
    return {
        "x_norm": x_norm,
        "x_norm_i8": x_norm_i8,
        "x_norm_scale": x_norm_scale.reshape(T, 1),
        "indices": indices.to(torch.int32),
        "weights": weights.to(torch.float32),
    }
