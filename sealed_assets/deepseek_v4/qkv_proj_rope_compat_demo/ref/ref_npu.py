"""NPU entry for the fixed DeepSeek-V4 DEMO QKV/RoPE compatibility ABI."""

import pypto.language as pl

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ
from decode_qkv_proj_rope import attn_norm, qkv_proj_rope

B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
Q_LORA = M.q_lora_rank


@pl.jit.inline
def qkv_proj_rope_compat(
    x: pl.Tensor[[B, S, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
):
    # The selector inputs remain in the historical ABI; this frozen
    # gather/scatter implementation does not consume them.
    x_normed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    x_normed = attn_norm(x, norm_w, x_normed)
    return qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv, rope_cos, rope_sin,
        gamma_cq, gamma_ckv, q, kv, qr, qr_scale,
    )


@pl.jit
def qkv_proj_rope_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
):
    return qkv_proj_rope_compat(
        x, norm_w, wq_a, wq_b, wq_b_scale, wkv, rope_cos, rope_sin,
        even_select_t, odd_select_t, gamma_cq, gamma_ckv, q, kv, qr, qr_scale,
    )
