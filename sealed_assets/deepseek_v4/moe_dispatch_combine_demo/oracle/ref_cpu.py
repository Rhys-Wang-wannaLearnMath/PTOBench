"""Independent CPU Oracle for fused dispatch, scale, and combine."""

import torch

from config import D, N_LOCAL_EXPERTS, RECV_MAX, T, TOPK


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"moe_dispatch_combine has no scalars: {sorted(scalars)}")

    recv_x = torch.zeros(
        N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16
    )
    recv_weights = torch.zeros(
        N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32
    )
    recv_token = torch.zeros(
        N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32
    )
    cursor = [0] * N_LOCAL_EXPERTS

    for token in range(T):
        seen = set()
        for k in range(TOPK):
            expert = int(inputs["indices"][token, k].item())
            if not 0 <= expert < N_LOCAL_EXPERTS:
                raise ValueError(
                    f"token {token} expert {expert} outside [0,{N_LOCAL_EXPERTS})"
                )
            if expert in seen:
                raise ValueError(f"token {token} repeats expert {expert}")
            seen.add(expert)
            slot = cursor[expert]
            if slot >= RECV_MAX:
                raise ValueError(f"expert {expert} exceeds capacity {RECV_MAX}")
            recv_x[expert, slot] = inputs["x_norm"][token]
            recv_weights[expert, slot] = float(
                inputs["weights"][token, k].item()
            )
            recv_token[expert, slot] = token
            cursor[expert] = slot + 1

    recv_y = (recv_x.float() * recv_weights.unsqueeze(-1)).to(torch.bfloat16)
    routed = torch.zeros(
        T, N_LOCAL_EXPERTS, D, dtype=torch.bfloat16
    )
    for expert in range(N_LOCAL_EXPERTS):
        for slot in range(cursor[expert]):
            token = int(recv_token[expert, slot].item())
            routed[token, expert] = recv_y[expert, slot]

    accumulator = routed[:, 0].float()
    for expert in range(1, N_LOCAL_EXPERTS):
        accumulator = accumulator + routed[:, expert].float()
    counts = torch.tensor(cursor, dtype=torch.int32).reshape(
        N_LOCAL_EXPERTS, 1
    )
    return {
        "recv_expert_count": counts,
        "routed_y": accumulator.to(torch.bfloat16),
    }
