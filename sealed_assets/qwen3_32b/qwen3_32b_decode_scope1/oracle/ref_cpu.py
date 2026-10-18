"""Independent CPU oracle for Qwen3-32B decode Scope 1."""
from __future__ import annotations

import torch

from config import BATCH, EPS, HIDDEN, HIDDEN_INV, KV_HIDDEN

K_CHUNK = 512


def _golden(tensors):
    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]

    outputs = {
        "q_proj": torch.zeros((BATCH, HIDDEN), dtype=torch.float32),
        "k_proj": torch.zeros((BATCH, KV_HIDDEN), dtype=torch.float32),
        "v_proj": torch.zeros((BATCH, KV_HIDDEN), dtype=torch.float32),
    }
    for b0 in range(0, BATCH, BATCH):
        x_tile = hidden_states[b0:b0 + BATCH, :].float()
        sq_sum = torch.zeros((x_tile.shape[0], 1), dtype=torch.float32)
        for k0 in range(0, HIDDEN, K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + K_CHUNK]
            sq_sum = sq_sum + (x_chunk * x_chunk).sum(dim=-1, keepdim=True)
        variance = sq_sum * HIDDEN_INV + EPS
        normed = (x_tile / torch.sqrt(variance) * input_rms_weight.float()).bfloat16()
        outputs["q_proj"][b0:b0 + BATCH] = (normed.float() @ wq.float()).float()
        outputs["k_proj"][b0:b0 + BATCH] = (normed.float() @ wk.float()).float()
        outputs["v_proj"][b0:b0 + BATCH] = (normed.float() @ wv.float()).float()
    return outputs


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.detach().clone() for name, value in inputs.items()}
    outputs = _golden(tensors)
    return {name: value.cpu().contiguous() for name, value in outputs.items()}
