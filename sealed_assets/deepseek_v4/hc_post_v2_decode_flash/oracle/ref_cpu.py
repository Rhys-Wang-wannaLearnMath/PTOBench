"""Case-local CPU Oracle for DeepSeek-V4 HC post mix."""

import torch
from config import D, HC_MULT


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"hc_post has no scalars: {sorted(scalars)}")
    x = inputs["x"].float()
    residual = inputs["residual"].float()
    post = inputs["post"].float()
    comb = inputs["comb"].float().reshape(-1, HC_MULT, HC_MULT)
    output = torch.zeros((x.shape[0], HC_MULT, D), dtype=torch.float32)
    for out_lane in range(HC_MULT):
        row = x * post[:, out_lane:out_lane + 1]
        for in_lane in range(HC_MULT):
            row = row + residual[:, in_lane] * comb[:, in_lane, out_lane:out_lane + 1]
        output[:, out_lane] = row
    return {"y": output.to(torch.bfloat16)}
