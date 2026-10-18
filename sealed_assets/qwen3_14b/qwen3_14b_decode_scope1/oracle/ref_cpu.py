"""Independent CPU Oracle for Qwen3-14B decode Scope 1."""
from __future__ import annotations

import torch

from config import EPS

K_CHUNK = 512
BATCH_TILE = 16


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]
    q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()
        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + K_CHUNK]
            sq_sum = sq_sum + (x_chunk * x_chunk).sum(dim=-1, keepdim=True)
        rms = torch.sqrt(sq_sum / hidden_size + EPS)
        normed = (x_tile / rms * input_rms_weight.float()).bfloat16()
        q_proj[b0:b_end] = (normed.float() @ wq.float()).float()
        k_proj[b0:b_end] = (normed.float() @ wk.float()).float()
        v_proj[b0:b_end] = (normed.float() @ wv.float()).float()
    return {
        "q_proj": q_proj.detach().cpu().contiguous(),
        "k_proj": k_proj.detach().cpu().contiguous(),
        "v_proj": v_proj.detach().cpu().contiguous(),
    }
