"""Independent CPU golden for packed SWA prefill."""

from config import CASE_CONFIG as M

EPS = M.rms_norm_eps

def golden_rms_norm(x, norm_w):
    import torch

    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)
