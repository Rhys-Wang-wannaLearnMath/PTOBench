"""Independent CPU Oracle for the ratio-0 compressor."""

from __future__ import annotations

import torch

from config import B, HEAD_DIM, OUT_DTYPE


def compute(inputs, scalars):
    del scalars
    outputs = {name: value.clone() for name, value in inputs.items()}
    outputs["out"] = torch.zeros((B, HEAD_DIM), dtype=torch.bfloat16)
    return {
        "out": outputs["out"].detach().cpu().contiguous(),
        "kv_state": outputs["kv_state"].detach().cpu().contiguous(),
        "score_state": outputs["score_state"].detach().cpu().contiguous(),
    }
