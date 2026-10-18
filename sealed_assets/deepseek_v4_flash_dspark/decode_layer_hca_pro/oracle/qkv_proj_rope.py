"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_hca_pro."""

from config import CASE_CONFIG as M
import torch
from utils import int8_quant_per_row

D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
EPS = M.rms_norm_eps

def golden_qkv_proj_rope(tensors):
    """Torch reference: Q/KV LoRA + RoPE for an already attention-normalized input."""
    x = tensors['x'].float()
    wq_a = tensors['wq_a'].float()
    wq_b = tensors['wq_b']
    wq_b_scale = tensors['wq_b_scale'].float().view(-1)
    wkv = tensors['wkv'].float()
    rope_cos = tensors['rope_cos'].float()
    rope_sin = tensors['rope_sin'].float()
    gamma_cq = tensors['gamma_cq'].float()
    gamma_ckv = tensors['gamma_ckv'].float()

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

    def matmul_bf16_input_fp32(a, b):
        a_fp32 = a.to(torch.bfloat16).float()
        b_fp32 = b.to(torch.bfloat16).float()
        return torch.matmul(a_fp32, b_fp32).float()

    def apply_rope(x_rope, cos, sin):
        x_pair = x_rope.unflatten(-1, (-1, 2))
        (x_even, x_odd) = (x_pair[..., 0], x_pair[..., 1])
        cos_v = cos[..., :ROPE_HALF]
        sin_v = sin[..., :ROPE_HALF]
        while cos_v.ndim < x_even.ndim:
            cos_v = cos_v.unsqueeze(-2)
            sin_v = sin_v.unsqueeze(-2)
        y_even = (x_even * cos_v - x_odd * sin_v).to(torch.bfloat16)
        y_odd = (x_even * sin_v + x_odd * cos_v).to(torch.bfloat16)
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    t_dim = x.shape[0]
    token_x = x.view(t_dim, D)
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)
    (qr_i8, qr_scale) = int8_quant_per_row(qr_out.float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(t_dim, H, HEAD_DIM)
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)
    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)
    tensors['q'][:] = q_out.to(torch.bfloat16)
    tensors['kv'][:] = kv_out.to(torch.bfloat16)
    tensors['qr'][:] = qr_i8
    tensors['qr_scale'][:] = qr_scale
