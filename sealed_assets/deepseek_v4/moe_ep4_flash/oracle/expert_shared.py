from __future__ import annotations
from config import CASE_CONFIG as M, INT8_SCALE_MAX, INT8_AMAX_EPS
SWIGLU_LIMIT = M.swiglu_limit

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

def golden_expert_shared(tensors):
    """Torch reference for the shared expert.

    Input is the per-token INT8 quant produced by gate (shared with
    dispatch / routed expert); we dequant inside to match the kernel's
    dequant-then-matmul pattern."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)
    x_local_i8 = tensors['x_local_i8']
    x_local_scale_dq = tensors['x_local_scale_dq'].float()
    x_local = x_local_i8.float() * x_local_scale_dq
    sw1 = dequant_w(tensors['shared_w1'], tensors['shared_w1_scale'].float())
    sw3 = dequant_w(tensors['shared_w3'], tensors['shared_w3_scale'].float())
    sw2 = dequant_w(tensors['shared_w2'], tensors['shared_w2_scale'].float())
    sh_gate = x_local @ sw1.T
    sh_up = x_local @ sw3.T
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up
    (sh_h_i8, sh_h_sd) = _int8_quant_per_row(sh_h)
    sh_h = sh_h_i8.float() * sh_h_sd
    sh = sh_h @ sw2.T
    tensors['sh'][:] = sh.to(torch.bfloat16)
