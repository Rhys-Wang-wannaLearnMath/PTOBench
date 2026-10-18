"""Case-local CPU Oracle for DeepSeek V3.2 decode front scope3."""

import torch

BATCH = 16
MAX_SEQ = 4096
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
SEQ_TILE = 64
SORT_LEN = 8192
FP32_NEG_INF = -3.4028234663852886e38


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"decode_front_scope3 has no scalars: {sorted(scalars)}")
    q_idx_full_i8 = inputs["q_idx_full_i8"]
    q_idx_scale_heads = inputs["q_idx_scale_heads"].float()
    weights = inputs["weights"].float()
    k_cache_idx_i8 = inputs["k_cache_idx_i8"]
    k_cache_idx_scale = inputs["k_cache_idx_scale"].float()
    seq_lens = inputs["seq_lens"]
    topk_vals_out = inputs["topk_vals_out"].clone()
    topk_idx_out = inputs["topk_idx_out"].clone()

    q_idx_i8_view = q_idx_full_i8.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM).to(torch.int32)
    q_s = weights * q_idx_scale_heads
    scores = torch.full((BATCH, SORT_LEN), FP32_NEG_INF, dtype=torch.float32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
        for sb in range(ctx_blocks):
            s0 = sb * SEQ_TILE
            valid_len = min(SEQ_TILE, ctx_len - s0)
            cache_row0 = b * MAX_SEQ + s0
            k_tile = k_cache_idx_i8[cache_row0:cache_row0 + SEQ_TILE].to(torch.int32)
            logits = torch.matmul(q_idx_i8_view[b, :INDEX_HEADS], k_tile.transpose(0, 1)).float()
            score = (torch.relu(logits[:, :valid_len]) * q_s[b, :INDEX_HEADS, None]).sum(dim=0)
            score = score * k_cache_idx_scale[b, s0:s0 + valid_len]
            scores[b, s0:s0 + valid_len] = score

    sorted_vals, sorted_idx = torch.sort(scores, dim=1, descending=True, stable=True)
    topk_vals_out.copy_(sorted_vals[:, :INDEX_TOPK])
    idx = sorted_idx[:, :INDEX_TOPK].to(torch.int32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        valid_topk = min(INDEX_TOPK, ctx_len)
        idx[b, valid_topk:] = torch.iinfo(torch.int32).min
    topk_idx_out.copy_(idx)
    return {
        "topk_vals_out": topk_vals_out.contiguous(),
        "topk_idx_out": topk_idx_out.contiguous(),
    }

