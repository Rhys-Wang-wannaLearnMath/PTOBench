from config import CASE_CONFIG as M, BLOCK_SIZE, FP32_NEG_INF, PREFILL_IDX_BLOCK_NUM
from prefill_indexer_compressor import golden_prefill_indexer_compressor
import torch
from utils import int8_quant_per_row
from config import ROPE_HEAD_DIM, IDX_N_HEADS, IDX_HEAD_DIM, WEIGHTS_SCALE, COMPRESS_RATIO, IDX_TOPK, B, S, T, INDEXER_SCORE_MAX_BLOCKS, INDEXER_SCORE_CAP, INDEXER_TOPK_CAP, MAX_CMP_WRITES

def topk_prefix_contract_error(topk_indices, position_ids, num_tokens):
    """Return an error string if the top-k prefix contract is broken."""
    if hasattr(num_tokens, 'item'):
        num_tokens = num_tokens.item()
    num_tokens = int(num_tokens)
    for t in range(T):
        row = topk_indices[t]
        if t >= num_tokens:
            non_padding = int((row != -1).count_nonzero().item())
            if non_padding:
                return f'inactive top-k row {t} contains {non_padding} non--1 entries'
            continue
        visible = min(int((int(position_ids[t].item()) + 1) // COMPRESS_RATIO), INDEXER_TOPK_CAP)
        prefix = row[:visible]
        if visible:
            out_of_range = int(((prefix < 0) | (prefix >= visible)).count_nonzero().item())
            if out_of_range:
                return f'top-k row {t} has {out_of_range} entries outside [0, {visible}) in its visible prefix'
            unique_count = int(torch.unique(prefix).numel())
            if unique_count != visible:
                return f'top-k row {t} visible prefix has {unique_count}/{visible} unique entries'
        tail_non_padding = int((row[visible:] != -1).count_nonzero().item())
        if tail_non_padding:
            return f'top-k row {t} tail contains {tail_non_padding} non--1 entries'
    return None

def golden_prefill_indexer_core(tensors):
    compressor_tensors = {'x': tensors['x'], 'kv': torch.zeros(MAX_CMP_WRITES, IDX_HEAD_DIM, dtype=torch.bfloat16), 'compress_state': tensors['inner_compress_state'], 'inner_compress_state_block_table': tensors['inner_compress_state_block_table'], 'wkv': tensors['inner_wkv'], 'wgate': tensors['inner_wgate'], 'ape': tensors['inner_ape'], 'norm_w': tensors['inner_norm_w'], 'freqs_cos': tensors['freqs_cos'], 'freqs_sin': tensors['freqs_sin'], 'hadamard': tensors['hadamard'], 'idx_kv_cache': tensors['idx_kv_cache'], 'idx_kv_scale': tensors['idx_kv_scale'], 'idx_block_table': tensors['idx_block_table'], 'position_ids': tensors['position_ids'], 'num_tokens': tensors['num_tokens'], 'idx_slot_mapping': tensors['idx_slot_mapping'], 'inner_state_slot_mapping': tensors['inner_state_slot_mapping']}
    golden_prefill_indexer_compressor(compressor_tensors)
    tensors['idx_kv_cache'][:] = compressor_tensors['idx_kv_cache']
    tensors['idx_kv_scale'][:] = compressor_tensors['idx_kv_scale']
    num_tokens = int(tensors['num_tokens'])
    position_ids = tensors['position_ids'].long()
    rd = ROPE_HEAD_DIM
    cmp_topk_indices = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
    score_full = torch.full((T, INDEXER_SCORE_CAP), FP32_NEG_INF, dtype=torch.float32)
    visible = ((position_ids + 1) // COMPRESS_RATIO).clamp(max=INDEXER_SCORE_CAP)
    max_visible = int(visible[:num_tokens].max().item()) if num_tokens > 0 else 0
    if max_visible == 0:
        return (cmp_topk_indices, score_full)
    qr = tensors['qr']
    qr_scale = tensors['qr_scale'].float()
    wq_b = tensors['wq_b']
    wq_b_scale = tensors['wq_b_scale'].float()
    hadamard = tensors['hadamard'].float()
    cos = tensors['cos'].float().view(T, 1, -1)
    sin = tensors['sin'].float().view(T, 1, -1)
    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(T, IDX_N_HEADS, IDX_HEAD_DIM)
    q_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    (q0, q1) = (q_pair[..., 0], q_pair[..., 1])
    y0 = (q0 * cos - q1 * sin).to(torch.bfloat16)
    y1 = (q0 * sin + q1 * cos).to(torch.bfloat16)
    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
    q = q.to(torch.bfloat16).float() @ hadamard
    weights = tensors['x'].float() @ tensors['weights_proj'].float() * WEIGHTS_SCALE
    cache_flat_i8 = tensors['idx_kv_cache'].reshape(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
    scale_flat = tensors['idx_kv_scale'].float().reshape(PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE, 1)
    idx_block_table = tensors['idx_block_table']
    rows = [int(idx_block_table[c // BLOCK_SIZE].item()) * BLOCK_SIZE + c % BLOCK_SIZE for c in range(max_visible)]
    kv_i8 = torch.stack([cache_flat_i8[r] for r in rows], dim=0).to(torch.int32)
    kv_sc = torch.stack([scale_flat[r] for r in rows], dim=0).view(1, 1, max_visible)
    (q_i8, q_sc) = int8_quant_per_row(q.reshape(T * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(T, IDX_N_HEADS, IDX_HEAD_DIM).to(torch.int32)
    q_sc = q_sc.view(T, IDX_N_HEADS, 1)
    score_i32 = torch.einsum('thd,cd->thc', q_i8, kv_i8)
    score = score_i32.float() * q_sc * kv_sc
    score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)
    col = torch.arange(max_visible).unsqueeze(0)
    score = score.masked_fill(col >= visible.unsqueeze(1), FP32_NEG_INF)
    score_full[:, :max_visible] = score
    for t in range(num_tokens):
        k = int(min(INDEXER_TOPK_CAP, int(visible[t].item())))
        if k > 0:
            sel = score[t].topk(k, dim=-1)[1]
            cmp_topk_indices[t, :k] = sel.to(torch.int32)
    return (cmp_topk_indices, score_full)
