"""Independent CPU Oracle for the routed expert workload."""
import torch
import torch.nn.functional as F
from config import CASE_CONFIG, EP_WORLD_SIZE, RECV_MAX, INT8_SCALE_MAX, INT8_AMAX_EPS

N_LOCAL_EXPERTS = CASE_CONFIG.n_routed_experts // EP_WORLD_SIZE
D = CASE_CONFIG.hidden_size
MOE_INTER = CASE_CONFIG.moe_intermediate_size
SWIGLU_LIMIT = CASE_CONFIG.swiglu_limit

def _dequant_weight(weight, scale):
    return weight.to(torch.float32) * scale.unsqueeze(-1)

def _quantize_rows(value):
    rows = value.float().reshape(-1, value.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    quantized = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    return quantized.reshape_as(value), (1.0 / scale_quant).reshape(*value.shape[:-1], 1)

def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"expert_routed has no scalars: {sorted(scalars)}")
    counts = inputs["recv_expert_count"]
    if tuple(counts.shape) != (N_LOCAL_EXPERTS, 1):
        raise ValueError(f"unexpected expert-count shape: {tuple(counts.shape)}")
    w1 = _dequant_weight(inputs["routed_w1"], inputs["routed_w1_scale"].float())
    w3 = _dequant_weight(inputs["routed_w3"], inputs["routed_w3_scale"].float())
    w2 = _dequant_weight(inputs["routed_w2"], inputs["routed_w2_scale"].float())
    output = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.float32)
    for expert in range(N_LOCAL_EXPERTS):
        rows = int(counts[expert, 0].item())
        if not 0 <= rows <= RECV_MAX:
            raise ValueError(f"expert {expert} count {rows} exceeds packed capacity")
        if rows == 0:
            continue
        x = inputs["recv_x"][expert, :rows].float() * inputs["recv_scale_dq"][expert, :rows].float().reshape(-1, 1)
        gate = x @ w1[expert].T
        up = x @ w3[expert].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        hidden = F.silu(gate) * up
        hidden_i8, hidden_scale = _quantize_rows(hidden)
        hidden = hidden_i8.float() * (hidden_scale * inputs["recv_weights"][expert, :rows].float().reshape(-1, 1))
        output[expert, :rows] = hidden @ w2[expert].T
    return {"recv_y": output.to(torch.bfloat16)}
