"""Independent CPU reference for the single shared-expert FFN."""

import torch
import torch.nn.functional as F
from config import CASE_CONFIG, MOE_TOKENS, INT8_SCALE_MAX, INT8_AMAX_EPS


T = MOE_TOKENS
D = CASE_CONFIG.hidden_size
SWIGLU_LIMIT = CASE_CONFIG.swiglu_limit


def _dequant_weight(weight, scale):
    return weight.to(torch.float32) * scale.unsqueeze(-1)


def _quantize_rows(x):
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    quantized = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    return quantized.reshape_as(x), (1.0 / scale_quant).reshape(*x.shape[:-1], 1)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"expert_shared has no scalars: {sorted(scalars)}")
    x = inputs["x_local_i8"].float() * inputs["x_local_scale_dq"].float()
    w1 = _dequant_weight(inputs["shared_w1"], inputs["shared_w1_scale"].float())
    w3 = _dequant_weight(inputs["shared_w3"], inputs["shared_w3_scale"].float())
    w2 = _dequant_weight(inputs["shared_w2"], inputs["shared_w2_scale"].float())
    gate = x @ w1.T
    up = x @ w3.T
    if SWIGLU_LIMIT > 0:
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    hidden = F.silu(gate) * up
    hidden_i8, hidden_scale = _quantize_rows(hidden)
    output = (hidden_i8.float() * hidden_scale) @ w2.T
    return {"sh": output.to(torch.bfloat16)}
