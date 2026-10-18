"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_hca_demo."""

from config import COMPRESS_RATIO
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
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
NEG_INF = -1e+20
HCA_MAX_COMPRESSED_ROWS = 1048576 // COMPRESS_RATIO
ATTN_K_TILE = 128
T_PAD = (T + 16 - 1) // 16 * 16

def golden_sparse_attn(tensors):
    """Torch reference for the HCA sparse-attention heads."""
    q = tensors['q'].float()
    tokens = q.shape[0]
    batch = tokens // S
    ori_kv = tensors['ori_kv'].float()
    window_swa_indices = tensors['window_swa_indices']
    cmp_kv = tensors['cmp_kv'].float()
    cmp_block_table = tensors['cmp_block_table']
    position_ids = tensors['position_ids'].to(torch.int64)
    kv_seq_lens = tensors['kv_seq_lens'].to(torch.int64)
    attn_sink = tensors['attn_sink'].float()
    cos = tensors['freqs_cos'].float()
    sin = tensors['freqs_sin'].float()
    o = torch.zeros(tokens, H, HEAD_DIM)
    for t in range(tokens):
        b = t // S
        item_rows = []
        item_valid = []
        raw_rows = []
        raw_valid = []
        for raw in window_swa_indices[t].tolist():
            slot = int(raw)
            if slot >= 0:
                blk_id = slot // BLOCK_SIZE
                intra = slot % BLOCK_SIZE
                raw_rows.append(ori_kv[blk_id, intra, 0])
                raw_valid.append(True)
            else:
                raw_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                raw_valid.append(False)
        raw_rows = torch.stack(raw_rows, dim=0)
        raw_valid = torch.tensor(raw_valid, dtype=torch.bool)
        for row_begin in range(0, WIN, ATTN_K_TILE):
            item_rows.append(raw_rows[row_begin:row_begin + ATTN_K_TILE])
            item_valid.append(raw_valid[row_begin:row_begin + ATTN_K_TILE])
        compressed_rows = min((int(position_ids[t].item()) + 1) // COMPRESS_RATIO, int(kv_seq_lens[b].item()) // COMPRESS_RATIO, HCA_MAX_COMPRESSED_ROWS)
        for row_begin in range(0, max(compressed_rows, 0), ATTN_K_TILE):
            valid_rows = min(ATTN_K_TILE, compressed_rows - row_begin)
            rows = []
            valid = []
            for lane in range(ATTN_K_TILE):
                logical_row = row_begin + lane
                if lane < valid_rows:
                    logical_page = logical_row // BLOCK_SIZE
                    physical_page = -1
                    if logical_page < cmp_block_table.shape[1]:
                        physical_page = int(cmp_block_table[b, logical_page].item())
                    if 0 <= physical_page < cmp_kv.shape[0]:
                        rows.append(cmp_kv[physical_page, logical_row % BLOCK_SIZE, 0])
                        valid.append(True)
                        continue
                rows.append(torch.zeros(HEAD_DIM, dtype=cmp_kv.dtype))
                valid.append(False)
            item_rows.append(torch.stack(rows, dim=0))
            item_valid.append(torch.tensor(valid, dtype=torch.bool))
        q_t = q[t]
        block_mi = []
        block_li = []
        block_oi = []
        for (kv_tile, valid_tile) in zip(item_rows, item_valid):
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
