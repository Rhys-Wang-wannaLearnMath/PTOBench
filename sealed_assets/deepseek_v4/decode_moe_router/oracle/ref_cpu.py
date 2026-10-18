"""Independent CPU reference for decode HC-pre, RMSNorm, and learned top-k routing."""

import torch
import torch.nn.functional as F

B, S, H, D, MIX = 16, 1, 4, 4096, 24
TOPK = 2
EPS = 1e-6


def compute(i, scalars):
    if scalars:
        raise ValueError("decode_moe_router has no scalars")
    x = i["x_hc"].float()
    flat = x.flatten(2)
    mixed = (flat @ i["hc_ffn_fn"].float().T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + EPS)
    scale, base = i["hc_ffn_scale"].float(), i["hc_ffn_base"].float()
    pre = torch.sigmoid(mixed[..., :4] * scale[0] + base[:4]) + EPS
    post = 2 * torch.sigmoid(mixed[..., 4:8] * scale[1] + base[4:8])
    comb = (mixed[..., 8:] * scale[2] + base[8:]).view(B, S, H, H)
    comb = torch.softmax(comb, -1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    x_mixed = (pre.unsqueeze(-1) * x).sum(2).to(torch.bfloat16)
    xf = x_mixed.float()
    x_norm = (i["norm_w"].float() * xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)).to(torch.bfloat16).view(B * S, D)
    scores = F.softplus(x_norm.float() @ i["gate_w"].float().T).sqrt()
    indices = (scores + i["gate_bias"].float()).topk(TOPK, dim=-1).indices
    weights = scores.gather(1, indices) 
    weights = weights / weights.sum(-1, keepdim=True)
    return {"x_norm": x_norm, "indices": indices.to(torch.int32), "weights": weights.float(), "post_ffn": post, "comb_ffn": comb}
