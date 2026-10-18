'Independent CPU golden for packed SWA prefill.'

import torch

import torch

from config import BLOCK_SIZE, INT8_AMAX_EPS, INT8_SCALE_MAX

def cache_row_from_table(
    table: torch.Tensor,
    slot: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Map one logical slot through a single-request 1D block table.

    Scalar counterpart of :func:`paged_slot_mapping`, which takes a ``[B, blocks]``
    table. Returns ``-1`` for an unmapped page.
    """
    block = slot // block_size
    intra = slot % block_size
    phys_block = int(table[block].item())
    if phys_block < 0:
        return -1
    return phys_block * block_size + intra

def int8_quant_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path.

    Rounds to int8 through fp16 to match the device rounding; returns the
    per-row dequant scale (``1 / scale_quant``).
    """
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)
