"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_swa_demo."""

from config import TP
from config import CASE_CONFIG as M
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
SOFTMAX_SCALE = M.softmax_scale
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
NEG_INF = -1e+20
ATTN_K_TILE = 128
T_PAD = (T + 16 - 1) // 16 * 16
SPARSE_BLOCKS = 1

def golden_sparse_attn(tensors):
    """Torch reference for the SWA sparse-attention heads."""
    q = tensors['q'].float()
    ori_kv = tensors['ori_kv'].float()
    ori_kv_flat = ori_kv.reshape(ori_kv.shape[0] * BLOCK_SIZE, HEAD_DIM)
    swa_indices = tensors['swa_indices']
    swa_lens = tensors['swa_lens']
    attn_sink = tensors['attn_sink'].float()
    cos = tensors['freqs_cos'].float()
    sin = tensors['freqs_sin'].float()
    tokens = q.shape[0]
    o = torch.zeros(tokens, H, HEAD_DIM)
    for t in range(tokens):
        valid_len = int(swa_lens[t].item())
        valid_slots = [int(v) for v in swa_indices[t, :valid_len].tolist() if int(v) >= 0]
        if not valid_slots:
            continue
        q_t = q[t]
        block_mi = []
        block_li = []
        block_oi = []
        for sb in range(SPARSE_BLOCKS):
            start = sb * ATTN_K_TILE
            end = min(start + ATTN_K_TILE, WIN)
            slots = swa_indices[t, start:end].tolist()
            valid_tile = torch.tensor([start + i < valid_len and int(slot) >= 0 for (i, slot) in enumerate(slots)], dtype=torch.bool)
            if end - start < ATTN_K_TILE:
                valid_tile = torch.cat([valid_tile, torch.zeros(ATTN_K_TILE - (end - start), dtype=torch.bool)])
            valid_tile = valid_tile.to(device=ori_kv.device)
            kv_tile = torch.zeros(ATTN_K_TILE, HEAD_DIM, dtype=ori_kv.dtype, device=ori_kv.device)
            for (r, slot) in enumerate(slots):
                if r >= ATTN_K_TILE:
                    break
                slot_i = int(slot)
                if slot_i >= 0:
                    kv_tile[r] = ori_kv_flat[slot_i]
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
    packed = tensors['o_packed_heads'].view(O_GROUPS, T_PAD, HEADS_PER_GROUP, HEAD_DIM)
    o_grouped = o.view(tokens, O_GROUPS, HEADS_PER_GROUP, HEAD_DIM)
    packed[:, :tokens] = o_grouped.permute(1, 0, 2, 3).to(torch.bfloat16)
