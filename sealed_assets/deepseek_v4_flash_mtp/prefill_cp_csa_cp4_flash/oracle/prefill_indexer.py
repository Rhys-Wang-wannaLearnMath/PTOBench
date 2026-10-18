from config import CASE_CONFIG as M, INT8_SCALE_MAX, INT8_AMAX_EPS

IDX_N_HEADS = M.index_n_heads

IDX_HEAD_DIM = M.index_head_dim

IDX_TOPK = M.index_topk

def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path.

    Mirrors decode_indexer._int8_quant_per_row: round-to-int8 via fp16 to match the device
    rounding, return the dequant scale (1/scale_quant) per row.
    """
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    out_i8 = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return (out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1))
