from __future__ import annotations
from config import CASE_CONFIG as M, INT8_SCALE_MAX, INT8_AMAX_EPS, RECV_MAX
D = M.hidden_size
SWIGLU_LIMIT = M.swiglu_limit
from config import N_LOCAL as N_LOCAL_EXPERTS

def _int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return (out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1))

def golden_expert_routed(tensors):
    """Torch reference for the routed expert. recv_y is the per-row routing-
    weight-scaled SwiGLU output, ready for combine reduce to simply sum.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)
    recv_x_i8 = tensors['recv_x']
    recv_scale_dq = tensors['recv_scale_dq'].float()
    recv_weights = tensors['recv_weights'].float()
    recv_expert_count = tensors['recv_expert_count']
    w1 = dequant_w(tensors['routed_w1'], tensors['routed_w1_scale'].float())
    w3 = dequant_w(tensors['routed_w3'], tensors['routed_w3_scale'].float())
    w2 = dequant_w(tensors['routed_w2'], tensors['routed_w2_scale'].float())
    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub_i8 = recv_x_i8[e, :n_rows, :]
        x_sub_sd = recv_scale_dq[e, :n_rows].reshape(-1, 1)
        x_sub_q = x_sub_i8.float() * x_sub_sd
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)
        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        (h_i8, h_sd) = _int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_per_row)
        recv_y[e, :n_rows, :] = h @ w2[e].T
    tensors['recv_y'][:] = recv_y.to(torch.bfloat16)
