"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_swa_flash."""

from config import CASE_CONFIG as M
from config import N_EXPERTS_GLOBAL
from config import EP
from config import RECV_MAX
from utils import int8_quant_per_row
import torch
import torch.nn.functional as F

D = M.hidden_size
SWIGLU_LIMIT = M.swiglu_limit
N_LOCAL_EXPERTS = N_EXPERTS_GLOBAL // EP

def golden_expert_routed(tensors):
    """Torch reference for the routed expert. recv_y is the per-row routing-
    weight-scaled SwiGLU output, ready for combine reduce to simply sum.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero."""

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
        (h_i8, h_sd) = int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_per_row)
        recv_y[e, :n_rows, :] = h @ w2[e].T
    tensors['recv_y'][:] = recv_y.to(torch.bfloat16)
