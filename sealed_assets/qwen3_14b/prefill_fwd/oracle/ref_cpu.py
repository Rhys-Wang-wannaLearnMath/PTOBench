"""Case-local CPU Oracle for Qwen3 14B prefill_fwd.py."""

from __future__ import annotations

import torch

Q_HEAD_BATCH = 5
EPS = 1e-6
SEQ_TILE = 128
BLOCK_SIZE = 128
LM_HEAD_K_CHUNK = 128
VOCAB_CHUNK = 64

def golden_qwen3_14b_prefill(tensors):
    """Reference implementation for full-layer prefill plus final logits.

    Mirrors the kernel precision path through RMSNorm, Q/K/V projection, RoPE,
    KV cache update, online-softmax attention, output projection, post RMSNorm,
    SwiGLU MLP, final RMSNorm, and the LM head projection.
    """
    import math

    import torch

    hidden_states = tensors["hidden_states"]
    seq_lens = tensors["seq_lens"]
    chunk_lens = tensors["chunk_lens"]
    chunk_offsets = tensors["chunk_offsets"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    final_norm_weight = tensors["final_norm_weight"]
    lm_head_weight = tensors["lm_head_weight"]

    batch = seq_lens.shape[0]
    total_tokens = hidden_states.shape[0]
    max_seq = rope_cos.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = EPS
    max_blocks_per_seq = block_table.shape[0] // batch
    num_layers = input_rms_weight.shape[0]
    intermediate_size = w_gate.shape[1]
    layer_cache_rows = batch * max_blocks_per_seq * num_kv_heads * BLOCK_SIZE

    def tiled_lm_head(lhs, rhs_t, k_chunk, vocab_chunk):
        out = torch.zeros(lhs.shape[0], rhs_t.shape[0], dtype=torch.float32)
        for k0 in range(0, lhs.shape[1], k_chunk):
            lhs_chunk = lhs[:, k0:k0 + k_chunk].float()
            out += lhs_chunk @ rhs_t[:, k0:k0 + k_chunk].float().T
        return out

    hidden = hidden_states.clone()
    for layer_idx in range(num_layers):
        layer_hidden_base = layer_idx * hidden_size
        layer_inter_base = layer_idx * intermediate_size
        layer_cache_base = layer_idx * layer_cache_rows
        input_rms_weight_f = input_rms_weight[layer_idx:layer_idx + 1, :].float()
        wq_f = wq[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wk_f = wk[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wv_f = wv[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wo_f = wo[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        post_rms_f = post_rms_weight[layer_idx:layer_idx + 1, :].float()
        w_gate_f = w_gate[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        w_up_f = w_up[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        w_down_f = w_down[layer_inter_base:layer_inter_base + intermediate_size, :].float()

        out_t = torch.zeros(total_tokens, hidden_size, dtype=torch.float32)
        for b in range(batch):
            seq_len_b = int(seq_lens[b].item())
            chunk_len_b = int(chunk_lens[b].item())
            token_base = int(chunk_offsets[b].item())
            if chunk_len_b <= 0:
                continue

            S = chunk_len_b
            chunk_start_b = seq_len_b - chunk_len_b

            # ── Scope 1: RMSNorm + Q/K/V projection ──
            x = hidden[token_base:token_base + S, :].float()
            variance = x.square().mean(dim=-1, keepdim=True) + eps
            inv_rms = 1.0 / torch.sqrt(variance)
            normed_bf16 = (x * inv_rms * input_rms_weight_f).to(torch.bfloat16)

            normed_f32 = normed_bf16.float()
            q_proj_f = normed_f32 @ wq_f
            k_proj_f = normed_f32 @ wk_f
            v_proj_f = normed_f32 @ wv_f

            # Per-head RMSNorm on Q and K (FP32).
            q_norm_f = q_norm_weight[layer_idx:layer_idx + 1, :].float().view(1, 1, head_dim)
            k_norm_f = k_norm_weight[layer_idx:layer_idx + 1, :].float().view(1, 1, head_dim)
            q_proj_view = q_proj_f.view(S, num_heads, head_dim)
            q_var = q_proj_view.pow(2).mean(dim=-1, keepdim=True)
            q_proj_view = q_proj_view * torch.rsqrt(q_var + eps) * q_norm_f
            q_proj_f = q_proj_view.reshape(S, hidden_size)
            k_proj_view = k_proj_f.view(S, num_kv_heads, head_dim)
            k_var = k_proj_view.pow(2).mean(dim=-1, keepdim=True)
            k_proj_view = k_proj_view * torch.rsqrt(k_var + eps) * k_norm_f
            k_proj_f = k_proj_view.reshape(S, kv_hidden)

            # ── Scope 2: RoPE + KV cache + causal attention ──
            cos_row = rope_cos[chunk_start_b:seq_len_b, :]
            sin_row = rope_sin[chunk_start_b:seq_len_b, :]
            cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
            sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

            # K RoPE + cache write.
            k_row = k_proj_f.view(S, num_kv_heads, head_dim)
            k_lo, k_hi = k_row[:, :, :half], k_row[:, :, half:]
            k_rot = torch.cat([
                k_lo * cos_lo.unsqueeze(1) - k_hi * sin_lo.unsqueeze(1),
                k_hi * cos_hi.unsqueeze(1) + k_lo * sin_hi.unsqueeze(1),
            ], dim=-1).to(torch.bfloat16)
            for pos in range(S):
                slot = int(slot_mapping[token_base + pos].item())
                slot_block = slot // BLOCK_SIZE
                slot_offset = slot % BLOCK_SIZE
                for ki in range(num_kv_heads):
                    cache_row = layer_cache_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                    k_cache[cache_row, :] = k_rot[pos, ki, :]

            # V cache write.
            v_row_bf16 = v_proj_f.view(S, num_kv_heads, head_dim).to(torch.bfloat16)
            for pos in range(S):
                slot = int(slot_mapping[token_base + pos].item())
                slot_block = slot // BLOCK_SIZE
                slot_offset = slot % BLOCK_SIZE
                for ki in range(num_kv_heads):
                    cache_row = layer_cache_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                    v_cache[cache_row, :] = v_row_bf16[pos, ki, :]

            # Q RoPE -> BF16.
            q_row = q_proj_f.view(S, num_heads, head_dim)
            q_lo, q_hi = q_row[:, :, :half], q_row[:, :, half:]
            q_rot_bf16 = torch.cat([
                q_lo * cos_lo.unsqueeze(1) - q_hi * sin_lo.unsqueeze(1),
                q_hi * cos_hi.unsqueeze(1) + q_lo * sin_hi.unsqueeze(1),
            ], dim=-1).to(torch.bfloat16)

            # Causal attention with tiled online softmax.
            max_blocks = (seq_len_b + SEQ_TILE - 1) // SEQ_TILE
            padded_len = max_blocks * SEQ_TILE
            ctx_lens = torch.arange(chunk_start_b + 1, seq_len_b + 1)
            col_idx = torch.arange(SEQ_TILE)
            attn_result = torch.zeros(S, hidden_size, dtype=torch.float32)

            for kvh in range(num_kv_heads):
                for qg in range(q_groups):
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_grp = q_rot_bf16[:S, q_base:q_base + Q_HEAD_BATCH, :]

                    cache_base = b * max_blocks_per_seq
                    k_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                    v_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                    for sb in range(max_blocks):
                        pbid = int(block_table[cache_base + sb].item())
                        cache_row0 = layer_cache_base + (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                        s0 = sb * SEQ_TILE
                        k_padded[s0:s0 + BLOCK_SIZE] = k_cache[cache_row0:cache_row0 + BLOCK_SIZE, :]
                        v_padded[s0:s0 + BLOCK_SIZE] = v_cache[cache_row0:cache_row0 + BLOCK_SIZE, :]

                    oi = li = mi = None

                    for sb in range(max_blocks):
                        s0 = sb * SEQ_TILE
                        k_tile = k_padded[s0:s0 + SEQ_TILE]
                        v_tile = v_padded[s0:s0 + SEQ_TILE]

                        raw_scores = (q_grp @ k_tile.T).float()

                        valid_lens = torch.clamp(ctx_lens - s0, min=0, max=SEQ_TILE)
                        mask = col_idx.unsqueeze(0) < valid_lens.unsqueeze(1)
                        raw_scores[~mask.unsqueeze(1).expand_as(raw_scores)] = torch.finfo(torch.float32).min
                        scores = raw_scores * scale

                        cur_mi = scores.max(dim=-1, keepdim=True).values
                        exp_scores = torch.exp(scores - cur_mi)
                        exp_bf16 = exp_scores.to(torch.bfloat16)
                        cur_li = exp_bf16.float().sum(dim=-1, keepdim=True)

                        oi_tmp = (exp_bf16 @ v_tile).float()

                        if sb == 0:
                            oi, li, mi = oi_tmp, cur_li, cur_mi
                        else:
                            active = valid_lens > 0
                            if active.any():
                                a = active
                                mi_new = torch.maximum(mi[a], cur_mi[a])
                                alpha = torch.exp(mi[a] - mi_new)
                                beta = torch.exp(cur_mi[a] - mi_new)
                                oi[a] = oi[a] * alpha + oi_tmp[a] * beta
                                li[a] = alpha * li[a] + beta * cur_li[a]
                                mi[a] = mi_new

                    ctx = oi / li
                    attn_result[:, q_base * head_dim:(q_base + Q_HEAD_BATCH) * head_dim] = \
                        ctx.reshape(S, Q_HEAD_BATCH * head_dim)

            attn_bf16 = attn_result.to(torch.bfloat16)

            # ── Scope 3: output projection + residual + post RMSNorm + MLP ──
            attn_f = attn_bf16.float()
            hs = hidden[token_base:token_base + S, :].float()

            # Output projection + first residual.
            resid1 = torch.matmul(attn_f, wo_f) + hs

            # Post-attention RMSNorm.
            variance = resid1.pow(2).mean(dim=-1, keepdim=True)
            post_inv_rms = torch.rsqrt(variance + eps)
            normed_bf16 = (resid1 * post_inv_rms * post_rms_f).bfloat16()

            # SwiGLU MLP.
            normed_post_f = normed_bf16.float()
            gate = torch.matmul(normed_post_f, w_gate_f)
            up = torch.matmul(normed_post_f, w_up_f)
            mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
            down = torch.matmul(mlp_bf16.float(), w_down_f)

            # Final residual -> BF16.
            out_t[token_base:token_base + S, :] = (down + resid1).bfloat16().float()

        hidden = out_t.to(torch.bfloat16)

    final_hidden = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    for b in range(batch):
        chunk_len_b = int(chunk_lens[b].item())
        token_base = int(chunk_offsets[b].item())
        if chunk_len_b > 0:
            final_hidden[b, :] = hidden[token_base + chunk_len_b - 1, :]

    variance = final_hidden.float().pow(2).mean(dim=-1, keepdim=True)
    final_normed = (
        final_hidden.float()
        * torch.rsqrt(variance + eps)
        * final_norm_weight.float()
    ).bfloat16()
    tensors["out"][:] = tiled_lm_head(final_normed, lm_head_weight, LM_HEAD_K_CHUNK, VOCAB_CHUNK)

def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalar inputs: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors["out"] = torch.zeros(
        (inputs["seq_lens"].shape[0], inputs["lm_head_weight"].shape[0]),
        dtype=torch.float32,
    )
    golden_qwen3_14b_prefill(tensors)
    return {"out": tensors["out"].detach().cpu().contiguous()}
