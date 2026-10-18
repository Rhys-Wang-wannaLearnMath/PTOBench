"""Independent CPU Oracle for Qwen3-14B greedy sampling."""
from __future__ import annotations

import torch

from config import BATCH, REAL_VOCAB, SAMPLED_IDS_PAD


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    logits = inputs["logits"].float()
    token_ids = torch.argmax(logits[:, :REAL_VOCAB], dim=-1).to(torch.int32)
    output = torch.zeros((BATCH, SAMPLED_IDS_PAD), dtype=torch.int32)
    output[:, 0] = token_ids
    return {"sampled_ids": output.contiguous()}
