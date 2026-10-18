"""Candidate-external CPU Oracle for local packed MoE dispatch."""

import torch
from config import (CASE_CONFIG, DECODE_TOKENS, EP_WORLD_SIZE, EP_RANK,
                    RECV_MAX)

T = DECODE_TOKENS
TOPK = CASE_CONFIG.num_experts_per_tok
D = CASE_CONFIG.hidden_size
EXPERTS = CASE_CONFIG.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * EXPERTS


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"dispatch has no scalars: {sorted(scalars)}")
    x = inputs["x_norm_i8"]
    scale = inputs["x_norm_scale"]
    indices = inputs["indices"]
    weights = inputs["weights"]
    recv_x = torch.zeros((EXPERTS, RECV_MAX, D), dtype=torch.int8)
    recv_scale = torch.zeros((EXPERTS, RECV_MAX), dtype=torch.float32)
    recv_weights = torch.zeros((EXPERTS, RECV_MAX), dtype=torch.float32)
    recv_token = torch.zeros((EXPERTS, RECV_MAX), dtype=torch.int32)
    cursor = torch.zeros((EXPERTS,), dtype=torch.int32)
    for token in range(T):
        row_experts = [int(value) for value in indices[token].tolist()]
        if len(set(row_experts)) != TOPK:
            raise ValueError(f"token {token} does not route to {TOPK} distinct experts")
        for route in range(TOPK):
            expert = row_experts[route] - EXPERTS_START_IDX
            if not 0 <= expert < EXPERTS:
                raise ValueError(f"token {token} route {route} expert {expert} outside local shard")
            slot = int(cursor[expert].item())
            if slot >= RECV_MAX:
                raise ValueError(f"expert {expert} exceeds RECV_MAX")
            recv_x[expert, slot] = x[token]
            recv_scale[expert, slot] = scale[token, 0]
            recv_weights[expert, slot] = weights[token, route]
            recv_token[expert, slot] = token
            cursor[expert] += 1
    return {
        "recv_x": recv_x,
        "recv_scale_dq": recv_scale,
        "recv_weights": recv_weights,
        "recv_token": recv_token,
        "recv_expert_count": cursor.reshape(EXPERTS, 1),
    }
