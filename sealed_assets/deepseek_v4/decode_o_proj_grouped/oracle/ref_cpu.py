"""External CPU Oracle for the grouped two-stage decode output projection."""

import torch

T, GROUPS, GROUP_INPUT, RANK, MODEL = 16, 8, 4096, 1024, 4096


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"decode_o_proj has no scalars: {sorted(scalars)}")
    packed = inputs["o"].float().reshape(T, GROUPS, GROUP_INPUT)
    stage_a_parts = []
    for group in range(GROUPS):
        stage_a_parts.append(packed[:, group].matmul(inputs["wo_a"][group].float().transpose(0, 1)))
    stage_a = torch.stack(stage_a_parts, dim=1).to(torch.bfloat16).float().reshape(T, GROUPS * RANK)
    output = stage_a.matmul(inputs["wo_b"].float().transpose(0, 1))
    return {"attn_out": output.to(torch.bfloat16)}
