"""External CPU Oracle for the direct fused decode QKV/RoPE entry."""

import torch
from config import CASE_CONFIG, DECODE_BATCH, DECODE_SEQ, INT8_AMAX_EPS

TOKENS = DECODE_BATCH * DECODE_SEQ
HIDDEN = CASE_CONFIG.hidden_size
HEADS = CASE_CONFIG.num_attention_heads
HEAD = CASE_CONFIG.head_dim
NOPE = CASE_CONFIG.nope_head_dim
HALF = CASE_CONFIG.qk_rope_head_dim // 2
LORA = CASE_CONFIG.q_lora_rank
EPSILON, QUANT_FLOOR, QUANT_MAX = CASE_CONFIG.rms_norm_eps, INT8_AMAX_EPS, 127.0


def _normalized(value, gamma):
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + EPSILON) * gamma


def _projection(left, right):
    # Both operands enter cube matmul as BF16 even though the CPU accumulator is FP32.
    return left.to(torch.bfloat16).float().matmul(right.to(torch.bfloat16).float())


def _row_quant(value):
    flat = value.reshape(-1, value.shape[-1]).float()
    maximum = flat.abs().amax(dim=-1, keepdim=True).clamp_min(QUANT_FLOOR)
    multiplier = QUANT_MAX / maximum
    rounded = torch.round(flat * multiplier).to(torch.int32)
    q = rounded.to(torch.float16).to(torch.int8).reshape_as(value)
    return q, (1.0 / multiplier).reshape(*value.shape[:-1], 1)


def _rotate(rotary, cosine, sine):
    paired = rotary.reshape(*rotary.shape[:-1], HALF, 2)
    even = paired[..., 0]
    odd = paired[..., 1]
    c = cosine[:, :HALF]
    s = sine[:, :HALF]
    while c.ndim < even.ndim:
        c = c.unsqueeze(-2); s = s.unsqueeze(-2)
    out_even = (even * c - odd * s).to(torch.bfloat16)
    out_odd = (even * s + odd * c).to(torch.bfloat16)
    return torch.stack((out_even, out_odd), dim=-1).reshape_as(rotary)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"direct decode QKV has no scalars: {sorted(scalars)}")
    normalized_x = _normalized(inputs["x"].float(), inputs["norm_w"].float()).to(torch.bfloat16).float()
    token_x = normalized_x.reshape(TOKENS, HIDDEN)

    lora = _normalized(_projection(token_x, inputs["wq_a"]), inputs["gamma_cq"].float())
    qr, qr_scale = _row_quant(lora.to(torch.bfloat16).float())
    q_acc = qr.to(torch.int32).matmul(inputs["wq_b"].to(torch.int32))
    q_full = (q_acc.float() * qr_scale * inputs["wq_b_scale"].float().reshape(1, -1)).reshape(TOKENS, HEADS, HEAD)
    q_full = q_full * torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPSILON)
    q = torch.cat((_normalized_slice(q_full, NOPE), _rotate(q_full[..., NOPE:], inputs["rope_cos"].float(), inputs["rope_sin"].float())), dim=-1)

    kv_full = _normalized(_projection(token_x, inputs["wkv"]), inputs["gamma_ckv"].float())
    kv_rotary = _rotate(kv_full[..., NOPE:].unsqueeze(1), inputs["rope_cos"].float(), inputs["rope_sin"].float()).squeeze(1)
    kv = torch.cat((kv_full[..., :NOPE], kv_rotary), dim=-1)
    return {"q": q.to(torch.bfloat16), "kv": kv.to(torch.bfloat16), "qr": qr, "qr_scale": qr_scale}


def _normalized_slice(q_full, end):
    # q_full has already received the per-head no-gamma RMS factor.
    return q_full[..., :end]
