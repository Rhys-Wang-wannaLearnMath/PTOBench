"""Independent CPU oracle for one-step decode CSA."""
from __future__ import annotations
import torch
from original import golden_attention_csa

def compute(values):
    tensors = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in values.items()}
    golden_attention_csa(tensors)
    return {"x_out": tensors["x_out"], "kv_cache": tensors["kv_cache"]}
