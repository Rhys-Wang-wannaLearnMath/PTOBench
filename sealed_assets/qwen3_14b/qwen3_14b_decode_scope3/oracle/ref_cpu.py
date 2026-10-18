"""Independent CPU Oracle for the Qwen3 Scope 3 contract."""
from __future__ import annotations

import torch

from config import EPS


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    attn_out = tensors["attn_out"]
    hidden_states = tensors["hidden_states"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()
    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + EPS)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())
    out = (down + resid1).bfloat16()
    return {"out": out.detach().cpu().contiguous()}
