"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_swa_flash."""

from config import CASE_CONFIG as M
from utils import int8_quant_per_row
import torch
import torch.nn.functional as F

SWIGLU_LIMIT = M.swiglu_limit

def golden_expert_shared(tensors):
    """Torch reference for the shared expert.

    Input is the per-token INT8 quant produced by gate (shared with
    dispatch / routed expert); we dequant inside to match the kernel's
    dequant-then-matmul pattern."""

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
    (sh_h_i8, sh_h_sd) = int8_quant_per_row(sh_h)
    sh_h = sh_h_i8.float() * sh_h_sd
    sh = sh_h @ sw2.T
    tensors['sh'][:] = sh.to(torch.bfloat16)
