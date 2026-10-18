"""Independent CPU Oracle for Qwen3-14B sampled-token embedding lookup."""

from __future__ import annotations

import torch

from config import BATCH, HIDDEN


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors["next_hidden"] = torch.zeros((BATCH, HIDDEN), dtype=torch.bfloat16)
    sampled_ids = tensors["sampled_ids"][:, :1].long().view(-1)
    tensors["next_hidden"][:] = tensors["embed_weight"].index_select(
        0, sampled_ids
    ).to(tensors["next_hidden"].dtype)
    return {"next_hidden": tensors["next_hidden"].detach().cpu().contiguous()}
