"""Independent CPU Oracle for one HCA invocation."""
import torch
from original import golden_attention_hca

def compute(values):
    tensors = {name: value.clone() if isinstance(value, torch.Tensor) else value
               for name, value in values.items()}
    golden_attention_hca(tensors)
    return {"x_out": tensors["x_out"], "kv_cache": tensors["kv_cache"]}
