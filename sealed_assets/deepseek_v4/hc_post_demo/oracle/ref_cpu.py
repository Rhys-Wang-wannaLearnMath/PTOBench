"""Independent CPU Oracle for the flattened HC post contract."""

import torch
from config import T, D, HC_MULT


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"hc_post has no scalars: {sorted(scalars)}")
    x = inputs["x"].float()
    residual = inputs["residual"].float()
    post = inputs["post"].float()
    comb = inputs["comb"].float().reshape(T, HC_MULT, HC_MULT)
    output = torch.zeros((T, HC_MULT, D), dtype=torch.float32)
    for out_lane in range(HC_MULT):
        row = x * post[:, out_lane:out_lane + 1]
        for in_lane in range(HC_MULT):
            row = row + residual[:, in_lane, :] * comb[:, in_lane, out_lane:out_lane + 1]
        output[:, out_lane, :] = row
    return {"y": output.to(torch.bfloat16)}
