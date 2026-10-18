"""Case-local CPU Oracle for the complete DeepSeek V4 decode indexer."""

from __future__ import annotations

import torch


from config import (
    CASE_CONFIG as M,
    DECODE_BATCH as B,
    DECODE_SEQ as S,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    COMPRESS_RATIO,
)

D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
OUT_DIM = (1 + int(COMPRESS_RATIO == 4)) * IDX_HEAD_DIM
STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
IDX_TOPK = M.index_topk
SCORE_LEN = M.max_position_embeddings // COMPRESS_RATIO
EPS = M.rms_norm_eps


def _int8_quant_per_row(value: torch.Tensor):
    rows = value.float().reshape(-1, value.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    quantized = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return quantized.reshape_as(value), scale_dequant.reshape(*value.shape[:-1], 1)


def _compress(inputs: dict[str, torch.Tensor]) -> None:
    x = inputs["x"].float()
    state = inputs["inner_compress_state"]
    state_table = inputs["inner_compress_state_block_table"]
    kv = x @ inputs["inner_wkv"].float()
    score = x @ inputs["inner_wgate"].float()
    ape = inputs["inner_ape"]
    start_positions = inputs["start_pos"]
    pooled = torch.zeros(B, 1, IDX_HEAD_DIM, dtype=torch.float32)
    should_compress_rows = torch.zeros(B, dtype=torch.bool)

    for batch in range(B):
        start = int(start_positions[batch].item())
        pre_tokens = min(S, COMPRESS_RATIO - (start % COMPRESS_RATIO))
        should_compress = pre_tokens < S or (start + S) % COMPRESS_RATIO == 0
        boundary_end = start + pre_tokens - 1
        current_window = boundary_end - COMPRESS_RATIO + 1
        previous_window = current_window - COMPRESS_RATIO
        ape_row = start % COMPRESS_RATIO
        for token in range(S):
            score[batch, token] += ape[(ape_row + token) % COMPRESS_RATIO]
            absolute = start + token
            block = int(state_table[batch, absolute // STATE_BLOCK_SIZE].item())
            intra = absolute % STATE_BLOCK_SIZE
            state[block, intra, :OUT_DIM] = kv[batch, token]
            state[block, intra, OUT_DIM:] = score[batch, token]
        if not should_compress:
            continue
        should_compress_rows[batch] = True
        kv_rows, score_rows = [], []
        for token in range(COMPRESS_RATIO):
            absolute = previous_window + token
            if absolute < 0:
                kv_rows.append(torch.zeros(IDX_HEAD_DIM))
                score_rows.append(torch.full((IDX_HEAD_DIM,), float("-inf")))
                continue
            block = int(state_table[batch, absolute // STATE_BLOCK_SIZE].item())
            intra = absolute % STATE_BLOCK_SIZE
            kv_rows.append(state[block, intra, :IDX_HEAD_DIM])
            score_rows.append(state[block, intra, OUT_DIM : OUT_DIM + IDX_HEAD_DIM])
        for token in range(COMPRESS_RATIO):
            absolute = current_window + token
            block = int(state_table[batch, absolute // STATE_BLOCK_SIZE].item())
            intra = absolute % STATE_BLOCK_SIZE
            kv_rows.append(state[block, intra, IDX_HEAD_DIM:OUT_DIM])
            score_rows.append(state[block, intra, OUT_DIM + IDX_HEAD_DIM :])
        kv_window = torch.stack(kv_rows).unsqueeze(0)
        score_window = torch.stack(score_rows).unsqueeze(0)
        pooled[batch : batch + 1] = (kv_window * score_window.softmax(dim=1)).sum(dim=1, keepdim=True)

    inputs["inner_compress_state"][:] = state
    if not bool(should_compress_rows.any()):
        return
    idx_cache = inputs["idx_kv_cache"]
    idx_table = inputs["idx_block_table"]
    for batch in range(B):
        if not bool(should_compress_rows[batch]):
            continue
        start = int(start_positions[batch].item())
        value = pooled[batch : batch + 1]
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)
        value = value * inputs["inner_norm_w"]
        pairs = value[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        x0, x1 = pairs[..., 0], pairs[..., 1]
        cos, sin = inputs["cos"][batch].view(-1), inputs["sin"][batch].view(-1)
        rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)
        value = torch.cat((value[..., :-ROPE_HEAD_DIM], rotated), dim=-1)
        value = value.to(torch.bfloat16).float() @ inputs["hadamard"].float()
        cache_column = start // COMPRESS_RATIO
        block = int(idx_table[batch, cache_column // BLOCK_SIZE].item())
        idx_cache[block, cache_column % BLOCK_SIZE, 0] = value[0, 0]
    inputs["idx_kv_cache"][:] = idx_cache


def compute(inputs, scalars):
    if set(scalars) != {"offset"}:
        raise ValueError(f"unexpected scalars: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    x = tensors["x"].float()
    qr = tensors["qr"]
    q_i32 = qr.to(torch.int32) @ tensors["wq_b"].to(torch.int32)
    q = (
        q_i32.float()
        * tensors["qr_scale"].float()
        * tensors["wq_b_scale"].float().view(1, -1)
    ).view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
    pairs = q[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
    x0, x1 = pairs[..., 0], pairs[..., 1]
    cos = tensors["cos"].view(B, 1, 1, -1)
    sin = tensors["sin"].view(B, 1, 1, -1)
    y0 = (x0 * cos - x1 * sin).to(torch.bfloat16)
    y1 = (x0 * sin + x1 * cos).to(torch.bfloat16)
    q = torch.cat((q[..., :-ROPE_HEAD_DIM], torch.stack((y0, y1), dim=-1).flatten(-2)), dim=-1)
    q = q @ tensors["hadamard"].float()

    _compress(tensors)
    weights = (x @ tensors["weights_proj"].float()) * WEIGHTS_SCALE
    idx_cache = tensors["idx_kv_cache"].float()
    idx_table = tensors["idx_block_table"]
    start_positions = tensors["start_pos"]
    score_full = torch.full((B, S, SCORE_LEN), FP32_NEG_INF, dtype=torch.float32)
    topk_indices = torch.full((B, S, SCORE_LEN), -1, dtype=torch.int32)
    q_i8, q_scale = _int8_quant_per_row(q.reshape(B * S * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(B, S, IDX_N_HEADS, 1)

    for batch in range(B):
        start = int(start_positions[batch].item())
        cache_len = (start + S) // COMPRESS_RATIO
        if cache_len <= 0:
            continue
        cache_rows = []
        for slot in range(cache_len):
            block = int(idx_table[batch, slot // BLOCK_SIZE].item())
            cache_rows.append(idx_cache[block, slot % BLOCK_SIZE, 0])
        cache_view = torch.stack(cache_rows).unsqueeze(0)
        kv_i8, kv_scale = _int8_quant_per_row(cache_view.reshape(cache_len, IDX_HEAD_DIM))
        kv_i8 = kv_i8.view(cache_len, IDX_HEAD_DIM)
        kv_scale = kv_scale.view(cache_len, 1)
        score_i32 = torch.einsum("shd,td->sht", q_i8[batch].to(torch.int32), kv_i8.to(torch.int32))
        score = score_i32.float() * q_scale[batch] * kv_scale.view(1, 1, cache_len)
        score = (torch.relu(score) * weights[batch].unsqueeze(-1)).sum(dim=1)
        score_full[batch, :, :cache_len] = score.float()
        k = min(IDX_TOPK, cache_len)
        _, indices = score.topk(k, dim=-1)
        topk_indices[batch, :, :k] = indices.to(torch.int32) + int(scalars["offset"])

    return {
        "idx_kv_cache": tensors["idx_kv_cache"].detach().cpu().contiguous(),
        "score": score_full,
        "topk_idxs": topk_indices,
    }

