"""Independent CPU oracle for prefill HC-post."""
import torch

from config import B, S, D, HC_MULT


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"prefill_hc_post has no scalars: {sorted(scalars)}")
    x = inputs["x"].float()
    residual = inputs["residual"].float()
    post = inputs["post"].float()
    comb = inputs["comb"].float()
    output = torch.zeros((B, S, HC_MULT, D), dtype=torch.float32)
    for out_h in range(HC_MULT):
        value = x * post[:, :, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            value = value + residual[:, :, in_h, :] * comb[:, :, in_h, out_h:out_h + 1]
        output[:, :, out_h, :] = value
    return {"y": output.to(torch.bfloat16)}
