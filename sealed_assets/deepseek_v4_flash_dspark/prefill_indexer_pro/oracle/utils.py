"""Reachable source CPU row-quantization helper."""
import torch
import importlib.util
from pathlib import Path

_config_spec = importlib.util.spec_from_file_location(
    "oracle_config_" + Path(__file__).parent.parent.name,
    Path(__file__).with_name("config.py"),
)
_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config)
INT8_AMAX_EPS = _config.INT8_AMAX_EPS
INT8_SCALE_MAX = _config.INT8_SCALE_MAX
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

