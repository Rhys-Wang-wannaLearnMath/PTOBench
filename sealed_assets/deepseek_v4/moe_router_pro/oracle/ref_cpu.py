"""DeepSeek-V4 MoE FFN router (decode): RMSNorm + gate + topk + normalize."""
from config import T, D, NORM_EPS, TOPK, ROUTE_SCALE, N_HASH_LAYERS, LAYER_ID

def golden_moe_router_core(tensors):
    import torch
    import torch.nn.functional as F
    norm_w = tensors['norm_w'].float()
    x_f = tensors['x_mixed'].float()
    var = x_f.square().mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + NORM_EPS)
    x_normalized = (norm_w * x_n).to(torch.bfloat16)
    x_flat = x_normalized.view(T, D)
    gate_w = tensors['gate_w'].float()
    gate_bias = tensors['gate_bias'].float()
    scores = F.softplus(x_flat.float() @ gate_w.T).sqrt()
    original_scores = scores
    layer_id = int(tensors['layer_id'])
    if layer_id < N_HASH_LAYERS:
        tid2eid = tensors['tid2eid']
        input_ids = tensors['input_ids']
        indices = tid2eid[input_ids.flatten().long()]
    else:
        biased = scores + gate_bias
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]
    weights = original_scores.gather(1, indices.long())
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE
    tensors['x_norm'][:] = x_flat
    tensors['indices'][:] = indices.to(torch.int32)
    tensors['weights'][:] = weights.to(torch.float32)

def compute(inputs, scalars):
    import torch
    values = dict(inputs)
    values.update(scalars)
    if 'layer_id' not in values:
        raise ValueError('missing fixed layer_id')
    if int(values['layer_id']) != LAYER_ID:
        raise ValueError('only the original fixed layer_id=0 is supported')
    values['x_norm'] = torch.empty((T, D), dtype=torch.bfloat16)
    values['indices'] = torch.empty((T, TOPK), dtype=torch.int32)
    values['weights'] = torch.empty((T, TOPK), dtype=torch.float32)
    golden_moe_router_core(values)
    return {name: values[name] for name in ('x_norm', 'indices', 'weights')}
