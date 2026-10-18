"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_csa_flash."""

from config import TP
from config import CASE_CONFIG as M
from config import COMPRESS_RATIO
from config import DECODE_BATCH
from config import DECODE_SEQ
from config import BLOCK_SIZE
import torch

B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
IDX_TOPK = M.index_topk
CMP_TOPK = IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
NEG_INF = -1e+20
ATTN_K_TILE = 128
T_PAD = (T + 16 - 1) // 16 * 16
TOPK = WIN + CMP_TOPK
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

def golden_sparse_attn(tensors):
    """Torch reference for the CSA sparse-attention heads."""
    q = tensors['q'].float()
    tokens = q.shape[0]
    batch = tokens // S
    ori_kv = tensors['ori_kv'].float()
    window_swa_indices = tensors['window_swa_indices']
    cmp_kv = tensors['cmp_kv'].float()
    cmp_block_table = tensors['cmp_block_table']
    raw = tensors['idx_topk'][:, :CMP_TOPK].to(torch.int64)
    bound = ((tensors['position_ids'][:, 0].to(torch.int64) + 1) // COMPRESS_RATIO).unsqueeze(1)
    keep = (raw >= 0) & (raw < bound)
    cmp_sparse_indices = torch.where(keep, raw, torch.full_like(raw, -1)).to(torch.int32)
    attn_sink = tensors['attn_sink'].float()
    cos = tensors['freqs_cos'].float()
    sin = tensors['freqs_sin'].float()
    o = torch.zeros(tokens, H, HEAD_DIM)
    for t in range(tokens):
        b = t // S
        kv_rows = []
        valid = []
        for raw in window_swa_indices[t].tolist():
            slot = int(raw)
            if slot >= 0:
                blk_id = slot // BLOCK_SIZE
                intra = slot % BLOCK_SIZE
                kv_rows.append(ori_kv[blk_id, intra, 0])
                valid.append(True)
            else:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)
        for raw in cmp_sparse_indices[t].tolist():
            if raw < 0:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)
                continue
            cmp_slot = int(raw)
            blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
            intra = cmp_slot % BLOCK_SIZE
            kv_rows.append(cmp_kv[blk_id, intra, 0])
            valid.append(True)
        if not any(valid):
            continue
        pad_k = PADDED_TOPK - TOPK
        if pad_k:
            kv_rows.extend((torch.zeros(HEAD_DIM, dtype=ori_kv.dtype) for _ in range(pad_k)))
            valid.extend((False for _ in range(pad_k)))
        kv_b = torch.stack(kv_rows, dim=0)
        valid_b = torch.tensor(valid, dtype=torch.bool)
        q_t = q[t]
        block_mi = []
        block_li = []
        block_oi = []
        for tile_start in range(0, PADDED_TOPK, ATTN_K_TILE):
            kv_tile = kv_b[tile_start:tile_start + ATTN_K_TILE]
            valid_tile = valid_b[tile_start:tile_start + ATTN_K_TILE]
            scores = q_t @ kv_tile.T * SOFTMAX_SCALE
            scores = scores.masked_fill(~valid_tile.unsqueeze(0), NEG_INF)
            mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - mi).masked_fill(~valid_tile.unsqueeze(0), 0.0)
            li = exp_scores.sum(dim=-1, keepdim=True)
            oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
            block_mi.append(mi)
            block_li.append(li)
            block_oi.append(oi)
        score_max = block_mi[0]
        li = block_li[0]
        oi_num = block_oi[0]
        for (mi_cur, li_cur, oi_cur) in zip(block_mi[1:], block_li[1:], block_oi[1:]):
            score_max_new = torch.maximum(score_max, mi_cur)
            alpha = torch.exp(score_max - score_max_new)
            beta = torch.exp(mi_cur - score_max_new)
            li = alpha * li + beta * li_cur
            oi_num = alpha * oi_num + beta * oi_cur
            score_max = score_max_new
        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[t] = oi_num / denom
    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)
    packed = tensors['o_packed_heads']
    packed[:, :tokens] = o.float().view(tokens, O_GROUPS, O_GROUP_IN).permute(1, 0, 2).to(torch.bfloat16)
