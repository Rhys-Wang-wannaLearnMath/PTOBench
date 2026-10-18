import torch
from config import D, T, N_LOCAL_EXPERTS


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"combine has no scalars: {sorted(scalars)}")
    recv_y, recv_token = inputs["recv_y"], inputs["recv_token"]
    counts = inputs["recv_expert_count"]
    acc = inputs["sh"].to(torch.float32).clone()
    for expert in range(N_LOCAL_EXPERTS):
        count = int(counts[expert, 0].item())
        if not 0 <= count <= T:
            raise ValueError(f"expert {expert} count outside domain")
        seen = set()
        for slot in range(count):
            token = int(recv_token[expert, slot].item())
            if not 0 <= token < T or token in seen:
                raise ValueError(f"invalid token at expert {expert}, slot {slot}")
            seen.add(token)
            acc[token] += recv_y[expert, slot].to(torch.float32)
    return {"ffn_out": acc.to(torch.bfloat16).reshape(T, 1, D)}
