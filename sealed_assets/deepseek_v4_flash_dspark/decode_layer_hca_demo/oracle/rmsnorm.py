"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_hca_demo."""

from config import CASE_CONFIG as M
import torch

EPS = M.rms_norm_eps

def golden_rms_norm(x, norm_w):
    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)
