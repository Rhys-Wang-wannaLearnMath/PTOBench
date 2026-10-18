"""Independent CPU Oracle for DeepSeek-V4 packed dispatch."""

import torch

from config import (
    CASE_CONFIG,
    DECODE_TOKENS,
    EP_WORLD_SIZE,
    EP_RANK,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    RECV_MAX,
)

T = DECODE_TOKENS
D = CASE_CONFIG.hidden_size
TOPK = CASE_CONFIG.num_experts_per_tok
N_LOCAL_EXPERTS = CASE_CONFIG.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS


def _quantize(x_bf16):
    x = x_bf16.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = INT8_SCALE_MAX / amax
    x_i8 = torch.round(x * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
    return x_i8, (1.0 / scale_q).reshape(T)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"moe_dispatch has no scalars: {sorted(scalars)}")
    x_i8, scale_dq = _quantize(inputs["x_norm"])
    indices = inputs["indices"]
    weights = inputs["weights"]
    recv_x = torch.zeros((N_LOCAL_EXPERTS, RECV_MAX, D), dtype=torch.int8)
    recv_scale = torch.zeros((N_LOCAL_EXPERTS, RECV_MAX), dtype=torch.float32)
    recv_weights = torch.zeros((N_LOCAL_EXPERTS, RECV_MAX), dtype=torch.float32)
    recv_token = torch.zeros((N_LOCAL_EXPERTS, RECV_MAX), dtype=torch.int32)
    cursor = torch.zeros((N_LOCAL_EXPERTS,), dtype=torch.int32)
    for token in range(T):
        row_experts = [int(value) for value in indices[token].tolist()]
        if len(set(row_experts)) != TOPK:
            raise ValueError(f"token {token} does not route to {TOPK} distinct experts")
        for route, global_expert in enumerate(row_experts):
            expert = global_expert - EXPERTS_START_IDX
            if not 0 <= expert < N_LOCAL_EXPERTS:
                raise ValueError(f"token {token} route {route} expert {expert} outside local shard")
            slot = int(cursor[expert].item())
            if slot >= RECV_MAX:
                raise ValueError(f"expert {expert} exceeds RECV_MAX")
            recv_x[expert, slot] = x_i8[token]
            recv_scale[expert, slot] = scale_dq[token]
            recv_weights[expert, slot] = weights[token, route]
            recv_token[expert, slot] = token
            cursor[expert] += 1
    return {
        "recv_x": recv_x,
        "recv_scale_dq": recv_scale,
        "recv_weights": recv_weights,
        "recv_token": recv_token,
        "recv_expert_count": cursor.reshape(N_LOCAL_EXPERTS, 1),
    }
