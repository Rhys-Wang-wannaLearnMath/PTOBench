"""Independent CPU reference for the decode routed-local and shared experts."""

import torch
import torch.nn.functional as F


from config import (
    D,
    MOE_INTER,
    N_LOCAL_EXPERTS,
    RECV_TOTAL_MAX,
    SWIGLU_LIMIT,
)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"decode_moe_expert has no scalars: {sorted(scalars)}")
    recv_x = inputs["recv_x"].float()
    expert_ids = inputs["recv_expert_id"][:, 0]
    weights = inputs["recv_weights"][:, 0].float()
    w1 = inputs["expert_w1"].float()
    w3 = inputs["expert_w3"].float()
    w2 = inputs["expert_w2"].float()
    recv_y = torch.zeros(RECV_TOTAL_MAX, D)
    for expert in range(N_LOCAL_EXPERTS):
        mask = expert_ids == expert
        if not bool(mask.any()):
            continue
        gate = recv_x[mask] @ w1[expert].T
        up = recv_x[mask] @ w3[expert].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        hidden = F.silu(gate) * up
        hidden = (hidden * weights[mask].unsqueeze(-1)).to(torch.bfloat16).float()
        recv_y[mask] = hidden @ w2[expert].T

    x_local = inputs["x_local"].float()
    shared_gate = x_local @ inputs["shared_w1"].float().T
    shared_up = x_local @ inputs["shared_w3"].float().T
    shared_hidden = (F.silu(shared_gate) * shared_up).to(torch.bfloat16).float()
    sh = shared_hidden @ inputs["shared_w2"].float().T
    return {"recv_y": recv_y.to(torch.bfloat16), "sh": sh.to(torch.bfloat16)}
