"""Independent CPU Oracle for packed weighted MoE combine."""

import torch

from config import B, D, N_LOCAL_EXPERTS, RECV_MAX, S, T


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"moe_combine has no scalars: {sorted(scalars)}")
    routed = torch.zeros(T, N_LOCAL_EXPERTS, D, dtype=torch.bfloat16)
    weights = torch.zeros(T, N_LOCAL_EXPERTS, dtype=torch.float32)
    for expert in range(N_LOCAL_EXPERTS):
        count = int(inputs["recv_expert_count"][expert, 0].item())
        if not 0 <= count <= RECV_MAX:
            raise ValueError(f"expert {expert} count {count} exceeds capacity")
        seen = set()
        for slot in range(count):
            token = int(inputs["recv_token"][expert, slot].item())
            if not 0 <= token < T:
                raise ValueError(
                    f"expert {expert} slot {slot} token {token} is outside [0,{T})"
                )
            if token in seen:
                raise ValueError(
                    f"expert {expert} repeats token {token} in its active prefix"
                )
            seen.add(token)
            routed[token, expert] = inputs["recv_y"][expert, slot]
            weights[token, expert] = float(inputs["recv_weights"][expert, slot].item())
    output = inputs["sh"].float()
    for expert in range(N_LOCAL_EXPERTS):
        output = output + routed[:, expert].float() * weights[:, expert:expert + 1]
    return {"ffn_out": output.to(torch.bfloat16).reshape(B, S, D)}
