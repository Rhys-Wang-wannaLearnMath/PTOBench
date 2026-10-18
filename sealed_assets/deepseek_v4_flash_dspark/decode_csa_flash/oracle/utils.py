from __future__ import annotations
import torch

def int8_quant_per_row(x: torch.Tensor):
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = 127.0 / amax
    q = torch.round(x.float() * scale).clamp(-127, 127).to(torch.int8)
    return q, (1.0 / scale).float()
