"""Candidate-external FP32 RMSNorm Oracle for the prefill workload."""

import torch

from config import EPS


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"rmsnorm_prefill_pro has no scalars: {sorted(scalars)}")
    x = inputs["x"].to(torch.float32)
    norm_w = inputs["norm_w"].to(torch.float32)
    inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)
    return {"x_normed": (x * inv_rms * norm_w).to(torch.bfloat16)}
