"""Independent CPU Oracle for prefill SWA sparse attention."""

import torch

from config import (
    B, S, T, D, H, HEAD_DIM, ROPE_HEAD_DIM, NOPE_DIM, HALF_ROPE,
    WIN, SOFTMAX_SCALE, O_LORA, O_GROUPS, HEADS_PER_GROUP, O_GROUP_IN,
    BLOCK_SIZE, MAX_BLOCKS, BLOCK_NUM, INT8_AMAX_EPS, INT8_SCALE_MAX,
    CASE_CONFIG,
)

SPARSE_ATTN_TILE = 64

def _int8_quant_per_row(x):
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)

def _golden_prefill_swa_values(tensors):
    start_pos = int(tensors["start_pos"])
    q = tensors["q"].float().view(B, S, H, HEAD_DIM)
    kv = tensors["kv"].float().view(B, S, HEAD_DIM)
    kv_cache = tensors["kv_cache"].float()
    block_table = tensors["block_table"]
    attn_sink = tensors["attn_sink"].float()

    o = torch.zeros(B, S, H, HEAD_DIM, dtype=torch.float32)
    for b in range(B):
        for s in range(S):
            abs_pos = start_pos + s
            first = max(0, abs_pos - WIN + 1)
            kv_rows = []
            for key_abs in range(first, abs_pos + 1):
                if key_abs >= start_pos:
                    kv_rows.append(kv[b, key_abs - start_pos])
                else:
                    ori_slot = key_abs % WIN
                    blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
                    kv_rows.append(kv_cache[blk_id, ori_slot % BLOCK_SIZE, 0])
            kv_b = torch.stack(kv_rows, dim=0)
            q_t = q[b, s]

            block_mi = []
            block_li = []
            block_oi = []
            for tile_start in range(0, kv_b.shape[0], SPARSE_ATTN_TILE):
                kv_tile = kv_b[tile_start:tile_start + SPARSE_ATTN_TILE]
                scores = torch.einsum("hd,kd->hk", q_t, kv_tile) * SOFTMAX_SCALE
                cur_mi = scores.max(dim=-1, keepdim=True).values
                exp_scores = torch.exp(scores - cur_mi).to(torch.bfloat16).float()
                cur_li = exp_scores.sum(dim=-1, keepdim=True)
                cur_oi = exp_scores @ kv_tile.to(torch.bfloat16).float()
                block_mi.append(cur_mi)
                block_li.append(cur_li)
                block_oi.append(cur_oi)

            merge_mi = block_mi[0]
            merge_li = block_li[0]
            merge_oi = block_oi[0]
            for cur_mi, cur_li, cur_oi in zip(block_mi[1:], block_li[1:], block_oi[1:], strict=True):
                mi_new = torch.maximum(merge_mi, cur_mi)
                alpha = torch.exp(merge_mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                merge_li = alpha * merge_li + beta * cur_li
                merge_oi = alpha * merge_oi + beta * cur_oi
                merge_mi = mi_new

            denom = merge_li + torch.exp(attn_sink.unsqueeze(-1) - merge_mi)
            o[b, s] = (merge_oi / denom).to(torch.bfloat16).float()

    return o

def _golden_swa_attention_o_proj(tensors):
    cos = tensors["freqs_cos_t"].float().view(B, S, ROPE_HEAD_DIM)
    sin = tensors["freqs_sin_t"].float().view(B, S, ROPE_HEAD_DIM)
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = _golden_prefill_swa_values(tensors)

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[..., :HALF_ROPE].unsqueeze(2)
    sin_half = sin[..., :HALF_ROPE].unsqueeze(2)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(B, S, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)
    tensors["attn_out"][:] = out.to(torch.bfloat16)

def golden_prefill_swa_sparse_attn(tensors):
    start_pos = int(tensors["start_pos"])

    _golden_swa_attention_o_proj(tensors)

    kv = tensors["kv"].view(B, S, HEAD_DIM)
    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]
    for b in range(B):
        for s in range(S):
            ori_slot = (start_pos + s) % WIN
            blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
            kv_cache[blk_id, ori_slot % BLOCK_SIZE, 0] = kv[b, s]

def compute(inputs, scalars):
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors.update(dict(scalars))
    tensors["attn_out"] = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_swa_sparse_attn(tensors)
    return {"kv_cache": tensors["kv_cache"].detach().cpu().contiguous(), "attn_out": tensors["attn_out"].detach().cpu().contiguous()}
