"""Case-local CPU Oracle for Qwen3 32B qwen3_32b_decode_scope2.py."""
from __future__ import annotations
import torch
from config import BATCH, HEAD_DIM, MAX_SEQ, NUM_HEADS, NUM_KV_HEADS
Q_HEAD_BATCH = 8        # Q heads batched per attention group
Q_HEAD_PAD = 16         # padded Q rows for cube fractal alignment
SEQ_TILE = 64           # sequence tile for attention loop
SB_BATCH = 64
def golden_qwen3_scope2(tensors):
    """PyTorch reference matching kernel BF16 precision path.

    Simulates the kernel's tiled online-softmax with BF16 matmuls:
      - Q cast to BF16 after RoPE (matching kernel QK matmul input).
      - QK/SV matmuls use BF16 inputs with FP32 accumulation.
      - BF16 round-trip on exp_scores before row_sum (matching kernel).
      - Full SEQ_TILE K/V loads with fillpad masking on scores.
    """
    import math

    import torch

    q_proj = tensors["q_proj"]              # already FP32
    k_proj = tensors["k_proj"]              # already FP32
    v_proj = tensors["v_proj"]              # FP32, cast to BF16 for cache writes
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()  # BF16, cloned to avoid side effects
    v_cache = tensors["v_cache"].clone()

    batch = q_proj.shape[0]
    hidden = q_proj.shape[1]
    kv_hidden = k_proj.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    attn_out = torch.zeros(batch, hidden, dtype=torch.float32)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        # K RoPE: all KV heads together.
        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_lo, k_hi = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [
                k_lo * cos_lo - k_hi * sin_lo,
                k_hi * cos_hi + k_lo * sin_hi,
            ],
            dim=-1,
        )

        # Update caches.
        for ki in range(num_kv_heads):
            cr = b * num_kv_heads * max_seq + ki * max_seq + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        # Q RoPE: all Q heads together.
        q_heads = q_proj[b].view(num_heads, head_dim)
        q_lo, q_hi = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [
                q_lo * cos_lo - q_hi * sin_lo,
                q_hi * cos_hi + q_lo * sin_hi,
            ],
            dim=-1,
        )

        # Grouped-query attention (tiled online softmax, matching kernel BF16 path).
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp = q_rot[q_base : q_base + Q_HEAD_BATCH, :]
                # Match kernel: Q cast to BF16 for QK matmul.
                q_grp_bf16 = q_grp.to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cb = b * num_kv_heads * max_seq + kvh * max_seq + s0

                    # Load full SEQ_TILE K/V tiles as BF16 (matching kernel).
                    k_tile = k_cache[cb : cb + SEQ_TILE, :]
                    v_tile = v_cache[cb : cb + SEQ_TILE, :]

                    # QK matmul: BF16 * BF16 → FP32.
                    raw_scores = q_grp_bf16.float() @ k_tile.float().T

                    # Fillpad invalid positions before scale.
                    if valid_len < SEQ_TILE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    # Online softmax: row_max → exp → BF16 round-trip → row_sum.
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                    # SV matmul: BF16 * BF16 → FP32.
                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                for qi in range(Q_HEAD_BATCH):
                    qh = q_base + qi
                    attn_out[b, qh * head_dim : (qh + 1) * head_dim] = ctx[qi]

    tensors["attn_out"][:] = attn_out.to(torch.bfloat16)
def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors["attn_out"] = torch.zeros(inputs["q_proj"].shape[0], inputs["q_proj"].shape[1], dtype=torch.bfloat16)
    golden_qwen3_scope2(tensors)
    return {"attn_out": tensors["attn_out"].detach().cpu().contiguous()}
