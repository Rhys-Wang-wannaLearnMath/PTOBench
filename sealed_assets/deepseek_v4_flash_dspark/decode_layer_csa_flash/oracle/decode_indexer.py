"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_csa_flash."""

from config import CASE_CONFIG as M
from config import COMPRESS_RATIO
from config import MAX_SEQ_LEN
from config import FP32_NEG_INF
from config import DECODE_SEQ
from config import BLOCK_SIZE
import torch
from decode_indexer_compressor import golden_compressor
from utils import int8_quant_per_row

S = DECODE_SEQ
D = M.hidden_size
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
WEIGHTS_SCALE = M.index_weights_scale
IDX_TOPK = M.index_topk
IDX_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
TOPK_CANDIDATES_PER_LEAF = 2048
TOPK_MAX_CANDIDATES = IDX_MAX_ROWS

def golden_indexer(tensors):
    """Torch reference for Indexer.forward decode branch; prefill `start_pos == 0` path is omitted."""
    x = tensors['x'].float()
    qr = tensors['qr']
    qr_scale = tensors['qr_scale'].float()
    wq_b = tensors['wq_b']
    wq_b_scale = tensors['wq_b_scale'].float()
    weights_proj = tensors['weights_proj'].float()
    cos = tensors['cos']
    sin = tensors['sin']
    hadamard = tensors['hadamard'].float()
    kv_seq_lens = tensors['kv_seq_lens'].to(torch.int64)
    tokens = x.shape[0]
    (bsz, seqlen) = (tokens // S, S)
    x = x.view(tokens, D)
    (ratio, rd) = (COMPRESS_RATIO, ROPE_HEAD_DIM)
    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(tokens, IDX_N_HEADS, IDX_HEAD_DIM)
    q_rope = q[..., -rd:]
    q_rope_swapped = q_rope.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    q_rope = q_rope * cos[:, None, :] + q_rope_swapped * sin[:, None, :]
    q = torch.cat([q[..., :-rd], q_rope], dim=-1)
    q = q.to(torch.bfloat16).float() @ hadamard
    inner_tensors = {'x': tensors['x'], 'kv': tensors['inner_kv'], 'wkv': tensors['inner_wkv'], 'wgate': tensors['inner_wgate'], 'ape': tensors['inner_ape'], 'norm_w': tensors['inner_norm_w'], 'cos': tensors['cmp_cos'], 'sin': tensors['cmp_sin'], 'hadamard': tensors['hadamard'], 'compress_state': tensors['inner_compress_state'], 'compress_state_block_table': tensors['inner_compress_state_block_table'], 'idx_kv_cache': tensors['idx_kv_cache'], 'idx_kv_scale': tensors['idx_kv_scale'], 'position_ids': tensors['position_ids'], 'idx_slot_mapping': tensors['idx_slot_mapping'], 'inner_state_slot_mapping': tensors['inner_state_slot_mapping']}
    golden_compressor(inner_tensors)
    weights = x @ weights_proj * WEIGHTS_SCALE
    idx_kv_cache_i8 = tensors['idx_kv_cache']
    idx_kv_scale = tensors['idx_kv_scale'].float()
    idx_block_table = tensors['idx_block_table']
    topk_scores = torch.full((tokens, IDX_TOPK), FP32_NEG_INF, dtype=torch.float32)
    topk_idxs = torch.full((tokens, IDX_TOPK), -1, dtype=torch.int32)
    (q_i8, q_scale) = int8_quant_per_row(q.reshape(tokens * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(tokens, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(tokens, IDX_N_HEADS, 1)
    flat_cache = idx_kv_cache_i8.reshape(-1, IDX_HEAD_DIM)
    flat_scale = idx_kv_scale.reshape(-1, 1)
    for b in range(bsz):
        cache_len = min(int(kv_seq_lens[b].item()) // ratio, TOPK_MAX_CANDIDATES)
        if cache_len <= 0:
            continue
        logical_rows = torch.arange(cache_len, dtype=torch.int64)
        physical_pages = idx_block_table[b, logical_rows // BLOCK_SIZE].to(torch.int64)
        valid_pages = (physical_pages >= 0) & (physical_pages < idx_kv_cache_i8.shape[0])
        physical_rows = physical_pages.clamp(min=0) * BLOCK_SIZE + logical_rows % BLOCK_SIZE
        kv_i8 = flat_cache[physical_rows]
        kv_scale = flat_scale[physical_rows]
        for s in range(seqlen):
            token = b * S + s
            visible_len = min(cache_len, int(tensors['position_ids'][token].item() + 1) // ratio, TOPK_MAX_CANDIDATES)
            if visible_len <= 0:
                continue
            running_scores = torch.empty(0, dtype=torch.float32)
            running_indices = torch.empty(0, dtype=torch.int64)
            for begin in range(0, visible_len, TOPK_CANDIDATES_PER_LEAF):
                end = min(begin + TOPK_CANDIDATES_PER_LEAF, visible_len)
                score_i32 = torch.einsum('hd,td->ht', q_i8[token].to(torch.int32), kv_i8[begin:end].to(torch.int32))
                score = score_i32.float() * q_scale[token]
                score = (torch.relu(score) * weights[token].unsqueeze(-1)).sum(dim=0)
                score = score * kv_scale[begin:end, 0]
                score = torch.where(valid_pages[begin:end], score, torch.full_like(score, FP32_NEG_INF))
                indices = torch.arange(begin, end, dtype=torch.int64)
                merged_scores = torch.cat([running_scores, score])
                merged_indices = torch.cat([running_indices, indices])
                keep = min(IDX_TOPK, merged_scores.numel())
                (running_scores, selected) = torch.topk(merged_scores, keep)
                running_indices = merged_indices[selected]
            topk_scores[token, :running_scores.numel()] = running_scores
            topk_idxs[token, :running_indices.numel()] = running_indices.to(torch.int32)
    tensors['topk_scores'][:] = topk_scores
    tensors['topk_idxs'][:] = topk_idxs
