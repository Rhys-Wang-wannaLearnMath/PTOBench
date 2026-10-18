"""Independent CPU Oracle for the selected single-layer Qwen3 decode Case."""
from __future__ import annotations

import torch

from config import (
    ATTN_SCALE, BATCH, BLOCK_SIZE, EPS, HEAD_DIM, HALF_DIM, HIDDEN, NUM_HEADS,
    NUM_KV_HEADS, Q_PER_KV, MAX_BLOCKS_PER_SEQ,
)

_REF_ATTN_SCALE = ATTN_SCALE

def _bf16(t: torch.Tensor) -> torch.Tensor:
    """Round through bf16 then back to fp32 — emulates a kernel ``pl.cast(BF16)``."""
    return t.to(torch.bfloat16).to(torch.float32)

def _rmsnorm_inv(x: torch.Tensor) -> torch.Tensor:
    """1/sqrt(mean(x^2, last) + eps), keepdim — the kernel's deferred denominator."""
    return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)

def _rope_half(vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Kernel RoPE (NeoX half-split): rot_lo = lo*cos_lo - hi*sin_lo ; rot_hi = hi*cos_hi + lo*sin_hi."""
    lo, hi = vec[..., :HALF_DIM], vec[..., HALF_DIM:]
    cos_lo, cos_hi = cos[..., :HALF_DIM], cos[..., HALF_DIM:]
    sin_lo, sin_hi = sin[..., :HALF_DIM], sin[..., HALF_DIM:]
    return torch.cat([lo * cos_lo - hi * sin_lo, hi * cos_hi + lo * sin_hi], dim=-1)

def golden_decode_layer(values: dict) -> None:
    """Torch reference for ONE Qwen3 decode layer; fills ``values['out']`` in place.

    Mirrors decode_fwd_layers (N==1) / _decode_layer at layer_idx 0: RMSNorm ->
    Q/K/V proj -> per-head QK-norm -> RoPE -> KV-cache write at pos=seq_len-1 -> GQA
    flash attention over [0, seq_len) -> out_proj -> residual -> post-RMSNorm ->
    SwiGLU MLP -> residual. The deferred input-RMSNorm inv_rms and the QK-norm
    control scale cancel to the standard math (QK-norm is scale-invariant).

    The inter-layer hidden is carried in FP32 now: the residual stream (h1, out)
    stays FP32 with NO intermediate bf16 rounding; only the chunk boundary casts
    (BF16 embed-in at copy_hidden, FP32->BF16 hidden-out at copy_out) round. So the
    only bf16 cast points are normed, the QKV inputs, q/k/v cache, attn_out, the
    gate/up input, the SwiGLU output, and the final copy_out — NOT the residual add.
    """
    x = values["hidden_states"].float()                          # [B,H], residual source
    gamma_in = values["input_rms_weight"].float()[0]             # [H]
    inv_rms = _rmsnorm_inv(x)                                    # [B,1] (deferred)

    normed = _bf16(x * gamma_in)                                 # bf16 normed (no inv_rms yet)
    q_proj = normed @ values["wq"].float()                      # [B,H]
    k_proj = normed @ values["wk"].float()                      # [B,KVH]
    v_proj = normed @ values["wv"].float()                      # [B,KVH]

    qn = values["q_norm_weight"].float()[0]
    kn = values["k_norm_weight"].float()[0]
    qh = (q_proj * inv_rms).reshape(BATCH, NUM_HEADS, HEAD_DIM)
    qh = qh * _rmsnorm_inv(qh) * qn                              # per-head QK-norm
    kh = (k_proj * inv_rms).reshape(BATCH, NUM_KV_HEADS, HEAD_DIM)
    kh = kh * _rmsnorm_inv(kh) * kn
    v_heads = (v_proj * inv_rms).reshape(BATCH, NUM_KV_HEADS, HEAD_DIM)

    seq_lens = values["seq_lens"]
    block_table = values["block_table"]
    rope_cos = values["rope_cos"].float()
    rope_sin = values["rope_sin"].float()
    k_cache = values["k_cache"].float()
    v_cache = values["v_cache"].float()

    attn_out = torch.zeros(BATCH, HIDDEN)
    for b in range(BATCH):
        slen = int(seq_lens[b].item())
        p = slen - 1
        cos_p, sin_p = rope_cos[p], rope_sin[p]
        q_b = _bf16(_rope_half(qh[b], cos_p, sin_p))             # [40,128] current Q (bf16)
        k_cur = _bf16(_rope_half(kh[b], cos_p, sin_p))           # [8,128] current K (bf16)
        v_cur = _bf16(v_heads[b])                                # [8,128]
        n_blocks = (slen + BLOCK_SIZE - 1) // BLOCK_SIZE
        for kvh in range(NUM_KV_HEADS):
            # Gather this (b, kvh) lane's past KV from the PAGED pool exactly as the
            # kernel does: logical block sb -> physical page block_table[b*MBPS + sb],
            # whose (page, kvh) tile starts at (pbid*NUM_KV_HEADS + kvh)*BLOCK_SIZE.
            k_lane = torch.empty(slen, HEAD_DIM)
            v_lane = torch.empty(slen, HEAD_DIM)
            for sb in range(n_blocks):
                pbid = int(block_table[b * MAX_BLOCKS_PER_SEQ + sb].item())
                row = (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                lo = sb * BLOCK_SIZE
                blk = min(BLOCK_SIZE, slen - lo)
                k_lane[lo : lo + blk] = k_cache[row : row + blk]
                v_lane[lo : lo + blk] = v_cache[row : row + blk]
            k_lane[p] = k_cur[kvh]                               # current token (kernel writes it first)
            v_lane[p] = v_cur[kvh]
            for j in range(Q_PER_KV):
                hq = kvh * Q_PER_KV + j
                scores = (q_b[hq].unsqueeze(0) * k_lane).sum(-1) * _REF_ATTN_SCALE
                w = torch.softmax(scores, dim=-1)
                attn_out[b, hq * HEAD_DIM : (hq + 1) * HEAD_DIM] = (w.unsqueeze(-1) * v_lane).sum(0)
    attn_out = _bf16(attn_out)

    attn_proj = attn_out @ values["wo"].float()                  # out_proj (FP32)
    h1 = x + attn_proj                                           # raw residual (FP32, NOT bf16-rounded)
    post_gamma = values["post_rms_weight"].float()[0]
    post_inv = _rmsnorm_inv(h1)                                  # deferred into silu (FP32 h1)
    mlp_in = _bf16(h1 * post_gamma)                              # gamma before inv_rms
    gate = mlp_in @ values["w_gate"].float()
    up = mlp_in @ values["w_up"].float()
    sg = gate * post_inv
    su = up * post_inv
    mlp = _bf16(sg * torch.sigmoid(sg) * su)                     # SwiGLU
    down = mlp @ values["w_down"].float()
    # FP32-carry residual: out = down + h1 in FP32 (no per-layer bf16(h1) rounding);
    # the single FP32->BF16 round is decode_fwd_layers' copy_out at the chunk tail.
    values["out"] = (down + h1).to(torch.bfloat16)

def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors["out"] = torch.zeros(
        (inputs["hidden_states"].shape[0], inputs["hidden_states"].shape[1]),
        dtype=torch.bfloat16,
    )
    golden_decode_layer(tensors)
    return {"out": tensors["out"].detach().cpu().contiguous()}
