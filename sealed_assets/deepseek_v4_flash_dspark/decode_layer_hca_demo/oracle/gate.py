"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_hca_demo."""

from config import CASE_CONFIG as M
from config import INT8_SCALE_MAX
from config import INT8_AMAX_EPS
from config import MOE_TOKENS
import torch

T = MOE_TOKENS
D = M.hidden_size
NORM_EPS = M.rms_norm_eps
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor
N_HASH_LAYERS = M.num_hash_layers

def _per_token_int8_quant(x_bf16):
    x_f32 = x_bf16.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = INT8_SCALE_MAX / amax
    scaled = x_f32 * scale_q
    x_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = (1.0 / scale_q).reshape(-1)
    return (x_i8, scale_dq)

def golden_gate_core(tensors):
    num_tokens = max(0, min(T, int(tensors.get('num_tokens', T))))
    x_f = tensors['x_mixed'].float().view(T, D)
    norm_w = tensors['norm_w'].float()
    sq_sum = (x_f * x_f).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)
    xg = x_f * norm_w.view(1, D)
    (x_norm_i8, scale_dq_g) = _per_token_int8_quant(xg)
    x_norm_scale = scale_dq_g.reshape(T, 1) * inv_rms
    gate_w = tensors['gate_w'].float()
    gate_bias = tensors['gate_bias'].float()
    logits = inv_rms * (xg.float() @ gate_w.T)
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.view(1, -1)
    layer_id = int(tensors['layer_id'])
    if layer_id < N_HASH_LAYERS:
        tid2eid = tensors['tid2eid']
        input_ids = tensors['input_ids']
        indices = tid2eid[input_ids.flatten().long()]
    else:
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]
    topk_vals = torch.gather(scores, dim=-1, index=indices.long())
    denom = topk_vals.sum(dim=-1, keepdim=True)
    weights = topk_vals / denom * ROUTE_SCALE
    if num_tokens < T:
        x_norm_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0
    tensors['x_norm_i8'][:] = x_norm_i8
    tensors['x_norm_scale'][:] = x_norm_scale.reshape(T, 1)
    tensors['indices'][:] = indices.to(torch.int32)
    tensors['weights'][:] = weights.to(torch.float32)
