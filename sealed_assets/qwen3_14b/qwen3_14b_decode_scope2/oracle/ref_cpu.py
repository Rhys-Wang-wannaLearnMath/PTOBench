"""Independent CPU Oracle for the Qwen3-14B decode Scope 2 Case."""

from __future__ import annotations

import math

import torch

from config import BLOCK_SIZE, EPS, Q_HEAD_BATCH, Q_HEAD_PAD


def golden_qwen3_scope2(tensors):
    q_proj = tensors["q_proj"]
    k_proj = tensors["k_proj"]
    v_proj = tensors["v_proj"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()

    batch = q_proj.shape[0]
    hidden = q_proj.shape[1]
    kv_hidden = k_proj.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    max_ctx_blocks = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    q_heads_all = q_proj.view(batch, num_heads, head_dim)
    k_heads_all = k_proj.view(batch, num_kv_heads, head_dim)
    q_var = q_heads_all.pow(2).mean(dim=-1, keepdim=True)
    k_var = k_heads_all.pow(2).mean(dim=-1, keepdim=True)
    q_heads_all = q_heads_all * torch.rsqrt(q_var + EPS) * q_norm_weight.float()
    k_heads_all = k_heads_all * torch.rsqrt(k_var + EPS) * k_norm_weight.float()

    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_out = torch.zeros(batch, hidden, dtype=torch.bfloat16)
    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_heads_all[b]
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [
                k_lo_h * cos_lo - k_hi_h * sin_lo,
                k_hi_h * cos_hi + k_lo_h * sin_hi,
            ],
            dim=-1,
        )
        slot = int(slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE
        for ki in range(num_kv_heads):
            cache_row = (
                slot_block * num_kv_heads + ki
            ) * BLOCK_SIZE + slot_offset
            k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_row, :] = v_proj[
                b, ki * head_dim : (ki + 1) * head_dim
            ].to(torch.bfloat16)

        q_heads = q_heads_all[b]
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [
                q_lo_h * cos_lo - q_hi_h * sin_lo,
                q_hi_h * cos_hi + q_lo_h * sin_hi,
            ],
            dim=-1,
        )

        attn_row_padded = torch.zeros(
            1,
            total_q_groups * Q_HEAD_PAD * head_dim,
            dtype=torch.bfloat16,
        )
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                gi = kvh * q_groups + qg
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[
                    q_base : q_base + Q_HEAD_BATCH, :
                ].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = min(BLOCK_SIZE, ctx_len - s0)
                    pbid = int(
                        block_table[b * max_ctx_blocks + sb].item()
                    )
                    cache_row0 = (
                        pbid * num_kv_heads + kvh
                    ) * BLOCK_SIZE
                    k_tile = k_cache[
                        cache_row0 : cache_row0 + BLOCK_SIZE, :
                    ]
                    v_tile = v_cache[
                        cache_row0 : cache_row0 + BLOCK_SIZE, :
                    ]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < BLOCK_SIZE:
                        raw_scores[:, valid_len:] = torch.finfo(
                            torch.float32
                        ).min
                    scores = raw_scores * scale
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(
                        dim=-1, keepdim=True
                    )
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
                ctx_flat_padded_bf16 = torch.zeros(
                    1, Q_HEAD_PAD * head_dim, dtype=torch.bfloat16
                )
                ctx_flat_padded_bf16[
                    :, : Q_HEAD_BATCH * head_dim
                ] = ctx.reshape(1, -1).to(torch.bfloat16)
                attn_row_padded[
                    :,
                    gi
                    * Q_HEAD_PAD
                    * head_dim : (gi + 1)
                    * Q_HEAD_PAD
                    * head_dim,
                ] = ctx_flat_padded_bf16

        attn_row = torch.zeros(1, hidden, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                gi = kvh * q_groups + qg
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                ctx_flat_bf16 = attn_row_padded[
                    :,
                    gi
                    * Q_HEAD_PAD
                    * head_dim : gi
                    * Q_HEAD_PAD
                    * head_dim
                    + Q_HEAD_BATCH
                    * head_dim,
                ]
                attn_row[
                    :,
                    q_base
                    * head_dim : (q_base + Q_HEAD_BATCH)
                    * head_dim,
                ] = ctx_flat_bf16

        attn_out[b : b + 1, :] = attn_row

    tensors["attn_out"][:] = attn_out


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors["attn_out"] = torch.zeros(
        inputs["q_proj"].shape[0],
        inputs["q_proj"].shape[1],
        dtype=torch.bfloat16,
    )
    golden_qwen3_scope2(tensors)
    return {
        "attn_out": tensors["attn_out"].detach().cpu().contiguous()
    }
