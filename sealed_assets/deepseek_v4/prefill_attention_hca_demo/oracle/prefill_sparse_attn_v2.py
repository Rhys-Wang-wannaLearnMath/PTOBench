from __future__ import annotations
from config import BLOCK_SIZE, CASE_CONFIG as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_CMP_MAX_BLOCKS, PREFILL_SEQ
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
ROPE_HALF = HALF_ROPE
NOPE_DIM = M.nope_head_dim
IDX_TOPK = M.index_topk
WIN = M.sliding_window
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
PREFILL_SPARSE_TOPK = min(TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE

def golden_prefill_sparse_attn(tensors):
    """Self-contained torch reference for the unified overlay sparse-attn entry."""
    import torch
    num_tokens = int(tensors['num_tokens'])
    q = tensors['q'].float()
    ori_kv = tensors['ori_kv'].float()
    kv_overlay = tensors['kv_overlay'].float()
    cmp_kv = tensors['cmp_kv'].float()
    ori_block_table = tensors['ori_block_table']
    cmp_block_table = tensors['cmp_block_table']
    cmp_sparse_indices = tensors['cmp_sparse_indices']
    cmp_sparse_lens = tensors['cmp_sparse_lens']
    attn_sink = tensors['attn_sink'].float()
    cos = tensors['freqs_cos'].float()
    sin = tensors['freqs_sin'].float()
    wo_a = tensors['wo_a'].float()
    wo_b_i8 = tensors['wo_b']
    wo_b_scale = tensors['wo_b_scale'].float()
    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(num_tokens):
        gathered = []
        sparse_len = max(0, min(int(cmp_sparse_lens[t].item()), PREFILL_SPARSE_PAD, TOPK))
        for raw_i in cmp_sparse_indices[t, :sparse_len].tolist():
            raw = int(raw_i)
            if raw < 0:
                continue
            if raw < WIN:
                block_id = int(ori_block_table[raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[block_id, intra, 0])
            elif raw < WIN + T:
                overlay_t = raw - WIN
                if 0 <= overlay_t < num_tokens:
                    gathered.append(kv_overlay[overlay_t])
            else:
                cmp_slot = raw - (WIN + T)
                if cmp_slot < 0 or cmp_slot >= CMP_MAX_BLOCKS * BLOCK_SIZE:
                    continue
                block_id = int(cmp_block_table[cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[block_id, intra, 0])
        if not gathered:
            continue
        kv_rows = torch.stack(gathered, dim=0)
        mi = None
        li = None
        oi = None
        for tile_start in range(0, kv_rows.shape[0], PREFILL_ATTN_TILE):
            kv_tile = kv_rows[tile_start:tile_start + PREFILL_ATTN_TILE]
            scores = q[t] @ kv_tile.T * SOFTMAX_SCALE
            cur_mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - cur_mi)
            cur_li = exp_scores.sum(dim=-1, keepdim=True)
            exp_scores_bf16 = exp_scores.to(torch.bfloat16)
            cur_oi = exp_scores_bf16.float() @ kv_tile.to(torch.bfloat16).float()
            if mi is None:
                mi = cur_mi
                li = cur_li
                oi = cur_oi
            else:
                mi_new = torch.maximum(mi, cur_mi)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                li = alpha * li + beta * cur_li
                oi = oi * alpha + cur_oi * beta
                mi = mi_new
        if mi is not None:
            denom = li + torch.exp(attn_sink.unsqueeze(-1) - mi)
            o[t] = oi / denom
    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :ROPE_HALF].unsqueeze(1)
    sin_half = sin[:, :ROPE_HALF].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)
    o_model = o.float().view(T, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum('tgd,grd->tgr', o_model, wo_a)
    o_r_g = o_r.reshape(T, O_GROUPS, O_LORA)
    amax_g = o_r_g.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q_g = INT8_SCALE_MAX / amax_g
    o_r_i8_g = torch.round(o_r_g * scale_q_g).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq_g = 1.0 / scale_q_g
    wo_b_g = wo_b_i8.reshape(D, O_GROUPS, O_LORA)
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        p_g = o_r_i8_g[:, g].to(torch.int32) @ wo_b_g[:, g].to(torch.int32).T
        out = out + p_g.float() * scale_dq_g[:, g]
    out = out * wo_b_scale.unsqueeze(0)
    tensors['attn_out'][:] = out.to(torch.bfloat16)
