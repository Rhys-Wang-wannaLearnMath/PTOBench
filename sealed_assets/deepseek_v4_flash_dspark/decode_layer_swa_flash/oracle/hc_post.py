"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_swa_flash."""

from config import CASE_CONFIG as M
import torch

D = M.hidden_size
HC_MULT = M.hc_mult

def golden_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    x = tensors['x'].float()
    residual = tensors['residual'].float()
    post = tensors['post'].float()
    comb = tensors['comb'].float().reshape(-1, HC_MULT, HC_MULT)
    T = x.shape[0]
    y_fp32 = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, in_h, :] * comb[:, in_h, out_h:out_h + 1]
        y_fp32[:, out_h, :] = y_row
    y = y_fp32
    tensors['y'][:] = y
