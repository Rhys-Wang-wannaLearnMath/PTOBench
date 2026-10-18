"""Independent CPU reference for combined routed-local and shared experts."""

import torch
import torch.nn.functional as functional

from config import (
    T, D, MOE_INTER, N_LOCAL_EXPERTS, RECV_MAX, SWIGLU_LIMIT,
    INT8_SCALE_MAX, INT8_AMAX_EPS,
)


def _dequant(weight, scale):
    return weight.to(torch.float32) * scale.unsqueeze(-1)


def _quantize_rows(values):
    rows = values.float().reshape(-1, values.shape[-1])
    amax = rows.abs().amax(-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    quantized = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    return quantized.reshape_as(values), (1.0 / scale_quant).reshape(*values.shape[:-1], 1)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"moe_expert has no scalars: {sorted(scalars)}")
    weight1 = _dequant(inputs["expert_w1"], inputs["expert_w1_scale"].float())
    weight3 = _dequant(inputs["expert_w3"], inputs["expert_w3_scale"].float())
    weight2 = _dequant(inputs["expert_w2"], inputs["expert_w2_scale"].float())
    shared1 = _dequant(inputs["shared_w1"], inputs["shared_w1_scale"].float())
    shared3 = _dequant(inputs["shared_w3"], inputs["shared_w3_scale"].float())
    shared2 = _dequant(inputs["shared_w2"], inputs["shared_w2_scale"].float())
    local_int8, local_scale = _quantize_rows(inputs["x_local"].float())
    local = local_int8.float() * local_scale
    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for expert in range(N_LOCAL_EXPERTS):
        rows = int(inputs["recv_expert_count"][expert, 0].item())
        if not 0 <= rows <= RECV_MAX:
            raise ValueError(f"expert {expert} count {rows} exceeds capacity")
        if rows == 0:
            continue
        received = (
            inputs["recv_x"][expert, :rows].float()
            * inputs["recv_scale_dq"][expert, :rows].float().reshape(-1, 1)
        )
        gate = received @ weight1[expert].T
        up = received @ weight3[expert].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        hidden = functional.silu(gate) * up
        hidden_int8, hidden_scale = _quantize_rows(hidden)
        recv_y[expert, :rows] = (hidden_int8.float() * hidden_scale) @ weight2[expert].T
    shared_gate = local @ shared1.T
    shared_up = local @ shared3.T
    shared_hidden = functional.silu(shared_gate) * shared_up
    shared_int8, shared_scale = _quantize_rows(shared_hidden)
    sh = (shared_int8.float() * shared_scale) @ shared2.T
    return {"recv_y": recv_y.to(torch.bfloat16), "sh": sh.to(torch.bfloat16)}
