import torch

from config import CASE_CONFIG


def compute(values):
    hidden = values["main_hidden"].to(torch.bfloat16).float()
    weight = values["main_proj_w"].to(torch.bfloat16).float()
    projected = torch.matmul(hidden, weight.t()).to(torch.bfloat16)
    norm_weight = values["main_norm_w"].to(torch.bfloat16).float()
    projected_float = projected.float()
    inverse_rms = torch.rsqrt(projected_float.square().mean(-1, keepdim=True) + CASE_CONFIG.rms_norm_eps)
    return {"main_x": (projected_float * inverse_rms * norm_weight).to(torch.bfloat16)}
