from __future__ import annotations

import torch

from config import (
    BATCH, DOWN_K_BLOCKS, DOWN_K_CHUNK, DOWN_N_BLOCKS, DOWN_N_CHUNK, EPS,
    HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HIDDEN, MLP_OUT_BLOCKS,
    MLP_OUT_CHUNK, NUM_HEADS, NUM_KV_HEADS, OUT_PROJ_K_BLOCKS,
    OUT_PROJ_K_CHUNK, OUT_PROJ_N_BLOCKS, Q_GROUPS, Q_HEAD_BATCH,
    Q_OUT_CHUNK, Q_PER_KV, SEQ_TILE,
)


def golden_qwen3_decode(tensors):
    """PyTorch reference: scope1 (RMSNorm + projection), scope2 (attention), scope3 (output + MLP)."""
    import math

    hidden_states_chunked = tensors["hidden_states"]
    input_rms_weight_chunked = tensors["input_rms_weight"]
    wq_chunked = tensors["wq"]
    wk_chunked = tensors["wk"]
    wv_chunked = tensors["wv"]
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo_chunked = tensors["wo"]
    post_rms_weight_chunked = tensors["post_rms_weight"]
    w_gate_chunked = tensors["w_gate"]
    w_up_chunked = tensors["w_up"]
    w_down_chunked = tensors["w_down"]

    hidden_states = hidden_states_chunked[:, 0, :, :].permute(1, 0, 2).reshape(BATCH, HIDDEN)
    input_rms_weight = input_rms_weight_chunked[:, 0, :, :].permute(1, 0, 2).reshape(1, HIDDEN)
    post_rms_weight = post_rms_weight_chunked[:, 0, :, :].permute(1, 0, 2).reshape(1, HIDDEN)

    half = HEAD_DIM // 2
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # ── Scope 1 golden: RMSNorm + Q/K/V projection ──
    x_tile = hidden_states.float()
    sq_sum = (x_tile ** 2).sum(dim=-1, keepdim=True)
    variance = sq_sum / HIDDEN + EPS
    rms = torch.sqrt(variance)
    normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

    wq = wq_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, HIDDEN)
    wk = wk_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, KV_HIDDEN)
    wv = wv_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, KV_HIDDEN)
    q_proj = (normed.float() @ wq.float()).float()
    k_proj = (normed.float() @ wk.float()).float()
    v_proj = (normed.float() @ wv.float()).float()

    # ── Scope 2 golden: RoPE + cache update + attention ──
    attn_proj_tile = torch.zeros(OUT_PROJ_K_BLOCKS, BATCH, OUT_PROJ_K_CHUNK, dtype=torch.bfloat16)

    for b in range(BATCH):
        ctx_len = seq_lens[b, 0, 0, 0].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos, 0, :, :]
        sin_row = rope_sin[pos, 0, :, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(NUM_KV_HEADS, HEAD_DIM)
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat([k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi], dim=-1)

        for ki in range(NUM_KV_HEADS):
            cache_idx = b * NUM_KV_HEADS + ki
            pos_block = pos // SEQ_TILE
            pos_offset = pos - pos_block * SEQ_TILE
            k_cache[cache_idx, pos_block, pos_offset, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_idx, pos_block, pos_offset, :] = v_proj[b, ki * HEAD_DIM : (ki + 1) * HEAD_DIM].to(torch.bfloat16)

        q_heads = q_proj[b].view(NUM_HEADS, HEAD_DIM)
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat([q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi], dim=-1)

        for kvh in range(NUM_KV_HEADS):
            for qg in range(Q_GROUPS):
                q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, HEAD_DIM, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cache_idx = b * NUM_KV_HEADS + kvh
                    k_tile = k_cache[cache_idx, sb, :, :]
                    v_tile = v_cache[cache_idx, sb, :, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < SEQ_TILE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

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
                    attn_proj_tile[qh, b, :] = ctx[qi].to(torch.bfloat16)

    # ── Scope 3 golden: output projection + residual + post RMSNorm + MLP + residual ──
    out_proj_chunks = []
    for oi in range(OUT_PROJ_N_BLOCKS):
        o_acc_lo = torch.matmul(attn_proj_tile[0].float(), wo_chunked[0, oi, :, :Q_OUT_CHUNK].float())
        o_acc_hi = torch.matmul(attn_proj_tile[0].float(), wo_chunked[0, oi, :, Q_OUT_CHUNK:].float())
        for kb in range(1, OUT_PROJ_K_BLOCKS):
            o_acc_lo = o_acc_lo + torch.matmul(attn_proj_tile[kb].float(), wo_chunked[kb, oi, :, :Q_OUT_CHUNK].float())
            o_acc_hi = o_acc_hi + torch.matmul(attn_proj_tile[kb].float(), wo_chunked[kb, oi, :, Q_OUT_CHUNK:].float())
        out_proj_chunks.append(torch.cat([o_acc_lo, o_acc_hi], dim=-1))
    o_proj = torch.cat(out_proj_chunks, dim=-1)
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + EPS)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    w_gate = w_gate_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, INTERMEDIATE)
    w_up = w_up_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, INTERMEDIATE)
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    mlp_blocks = mlp_bf16.reshape(BATCH, MLP_OUT_BLOCKS, MLP_OUT_CHUNK)
    down_chunks = []
    for di in range(DOWN_N_BLOCKS):
        down_acc = None
        for ob in range(DOWN_K_BLOCKS):
            mlp_block = ob // (MLP_OUT_CHUNK // DOWN_K_CHUNK)
            mlp_offset = (ob - mlp_block * (MLP_OUT_CHUNK // DOWN_K_CHUNK)) * DOWN_K_CHUNK
            mlp_chunk = mlp_blocks[:, mlp_block, mlp_offset : mlp_offset + DOWN_K_CHUNK].float()
            w_down_chunk = w_down_chunked[ob, di, :, :].float()
            down_part = torch.matmul(mlp_chunk, w_down_chunk)
            down_acc = down_part if down_acc is None else down_acc + down_part
        down_chunks.append(down_acc)
    down = torch.stack(down_chunks, dim=1).reshape(BATCH, HIDDEN)

    out_flat = (down + resid1).bfloat16()
    tensors["out"][:] = out_flat.reshape(BATCH, DOWN_N_BLOCKS, DOWN_N_CHUNK).permute(1, 0, 2).unsqueeze(1)


def compute(tensors, scalars):
    """Compute one independent CPU result for the blocked decode ABI."""
    if scalars:
        raise ValueError("qwen3_32b_decode_4d has no scalar inputs")
    local = {name: value.detach().clone() for name, value in tensors.items()}
    if "out" not in local:
        local["out"] = torch.zeros(
            (DOWN_N_BLOCKS, 1, BATCH, DOWN_N_CHUNK), dtype=torch.bfloat16
        )
    golden_qwen3_decode(local)
    return {"out": local["out"].contiguous()}
