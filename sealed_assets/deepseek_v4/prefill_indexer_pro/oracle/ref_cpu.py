"""Case-local CPU Oracle for the complete packed prefill indexer."""

from __future__ import annotations

import torch

from config import (
    BLOCK_SIZE,
    CASE_CONFIG,
    COMPRESS_RATIO as RATIO,
    D,
    FP32_NEG_INF,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_NOPE_HEAD_DIM as NOPE_HEAD_DIM,
    INDEXER_OFFSET as INDEX_OFFSET,
    INDEXER_SCORE_CAP as SCORE_CAP,
    INNER_OUT_DIM as OUT_DIM,
    INNER_STATE_BLOCK_NUM as STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE as STATE_BLOCK_SIZE,
    INNER_STATE_LEN as STATE_LEN,
    INT8_AMAX_EPS as INT8_EPS,
    INT8_SCALE_MAX as INT8_MAX,
    MAX_SEQ_LEN,
    PREFILL_IDX_BLOCK_NUM as CACHE_BLOCK_NUM,
    Q_LORA,
    ROPE_HEAD_DIM,
    T,
    WEIGHTS_SCALE,
)

EPS = CASE_CONFIG.rms_norm_eps


def _quantize_rows(value):
    rows = value.float().reshape(-1, value.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_EPS)
    scale_quant = INT8_MAX / amax
    quantized = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    return quantized.reshape_as(value), (1.0 / scale_quant).reshape(*value.shape[:-1], 1)


def _compress(tensors):
    kv_proj = tensors["x"].float() @ tensors["inner_wkv"].float().t()
    score_proj = tensors["x"].float() @ tensors["inner_wgate"].float().t()
    kv_state_flat = tensors["inner_kv_state"].view(STATE_BLOCK_NUM * STATE_BLOCK_SIZE, OUT_DIM)
    score_state_flat = tensors["inner_score_state"].view(STATE_BLOCK_NUM * STATE_BLOCK_SIZE, OUT_DIM)
    state_table = tensors["inner_compress_state_block_table"]
    cache = tensors["idx_kv_cache"]
    scale = tensors["idx_kv_scale"]
    cache_rows = cache.view(CACHE_BLOCK_NUM * BLOCK_SIZE, 1, IDX_HEAD_DIM)[:, 0]
    scale_rows = scale.view(CACHE_BLOCK_NUM * BLOCK_SIZE, 1, 1)[:, 0, 0]
    positions = tensors["position_ids"]
    num_tokens = int(tensors["num_tokens"])
    ape = tensors["inner_ape"]
    norm_w = tensors["inner_norm_w"]
    hadamard = tensors["hadamard"].float()

    def state_row(absolute):
        if absolute < 0 or absolute >= MAX_SEQ_LEN:
            return -1
        block, intra = absolute // STATE_BLOCK_SIZE, absolute % STATE_BLOCK_SIZE
        physical = int(state_table[block].item())
        return -1 if physical < 0 else physical * STATE_BLOCK_SIZE + intra

    write_i = 0
    for token_id in range(num_tokens):
        destination = int(tensors["idx_slot_mapping"][token_id].item())
        if destination < 0:
            continue
        write_pos = int(positions[token_id].item())
        current_start = write_pos + 1 - RATIO
        previous_start = current_start - RATIO
        pool_kv = torch.zeros(STATE_LEN, IDX_HEAD_DIM, dtype=torch.float32)
        pool_score = torch.full((STATE_LEN, IDX_HEAD_DIM), float("-inf"), dtype=torch.float32)
        for slot in range(RATIO):
            previous = previous_start + slot
            if write_pos >= 2 * RATIO - 1:
                row = state_row(previous)
                if row >= 0:
                    pool_kv[slot] = kv_state_flat[row, :IDX_HEAD_DIM]
                    pool_score[slot] = score_state_flat[row, :IDX_HEAD_DIM]
            current = current_start + slot
            row = state_row(current)
            if row >= 0:
                pool_kv[RATIO + slot] = kv_state_flat[row, IDX_HEAD_DIM:OUT_DIM]
                pool_score[RATIO + slot] = score_state_flat[row, IDX_HEAD_DIM:OUT_DIM]
        for token in range(num_tokens):
            pos = int(positions[token].item())
            if pos < previous_start or pos > write_pos:
                continue
            ape_slot = pos % RATIO
            if pos < current_start:
                pool_slot, col0 = pos - previous_start, 0
            else:
                pool_slot, col0 = RATIO + pos - current_start, IDX_HEAD_DIM
            pool_kv[pool_slot] = kv_proj[token, col0 : col0 + IDX_HEAD_DIM]
            pool_score[pool_slot] = score_proj[token, col0 : col0 + IDX_HEAD_DIM] + ape[ape_slot, col0 : col0 + IDX_HEAD_DIM]

        mi = pool_score[STATE_LEN - 1 : STATE_LEN].clone()
        li = torch.exp(mi - mi)
        oi = pool_kv[STATE_LEN - 1 : STATE_LEN].clone()
        for slot in range(STATE_LEN - 1):
            if slot < RATIO and write_pos < 2 * RATIO - 1:
                continue
            slot_score = pool_score[slot : slot + 1]
            slot_kv = pool_kv[slot : slot + 1]
            mi_next = torch.maximum(mi, slot_score)
            alpha, beta = torch.exp(mi - mi_next), torch.exp(slot_score - mi_next)
            li = alpha * li + beta
            oi = oi * alpha + slot_kv * beta
            mi = mi_next
        pooled = oi / li
        normed_fp32 = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + EPS) * norm_w.float().view(1, IDX_HEAD_DIM)
        normed = normed_fp32.clone()
        normed[:, :NOPE_HEAD_DIM] = normed_fp32[:, :NOPE_HEAD_DIM].to(torch.bfloat16).float()
        pair = normed_fp32[..., NOPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        even, odd = pair[..., 0], pair[..., 1]
        compressed_position = write_pos + 1 - RATIO
        cos = tensors["freqs_cos"][compressed_position : compressed_position + 1, : ROPE_HEAD_DIM // 2].float()
        sin = tensors["freqs_sin"][compressed_position : compressed_position + 1, : ROPE_HEAD_DIM // 2].float()
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        normed[:, NOPE_HEAD_DIM:] = rotated.to(torch.bfloat16).float()
        final = normed.to(torch.bfloat16).float() @ hadamard
        row_bf16 = final.to(torch.bfloat16)[0].float()
        amax = row_bf16.abs().amax().clamp_min(INT8_EPS)
        quant_scale = INT8_MAX / amax
        row_i8 = torch.round(row_bf16 * quant_scale).to(torch.int32).to(torch.float16).to(torch.int8)
        cache_rows[destination] = row_i8
        scale_rows[destination] = 1.0 / quant_scale
        write_i += 1

    for token in range(num_tokens):
        pos = int(positions[token].item())
        row = int(tensors["inner_state_slot_mapping"][token].item())
        if row < 0:
            continue
        kv_state_flat[row] = kv_proj[token]
        score_state_flat[row] = score_proj[token] + ape[pos % RATIO]
    tensors["inner_kv_state"][:] = kv_state_flat.view_as(tensors["inner_kv_state"])
    tensors["inner_score_state"][:] = score_state_flat.view_as(tensors["inner_score_state"])
    tensors["idx_kv_cache"][:] = cache
    tensors["idx_kv_scale"][:] = scale


def compute(inputs, scalars):
    if set(scalars) != {"num_tokens"}:
        raise ValueError(f"unexpected scalars: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors["num_tokens"] = int(scalars["num_tokens"])
    _compress(tensors)

    num_tokens = int(scalars["num_tokens"])
    positions = tensors["position_ids"].long()
    visible = ((positions + 1) // RATIO).clamp(max=SCORE_CAP)
    topk = torch.full((T, SCORE_CAP), -1, dtype=torch.int32)
    score_full = torch.full((T, SCORE_CAP), FP32_NEG_INF, dtype=torch.float32)
    max_visible = int(visible[:num_tokens].max().item()) if num_tokens > 0 else 0
    if max_visible > 0:
        q_i32 = tensors["qr"].to(torch.int32) @ tensors["wq_b"].to(torch.int32)
        q = (
            q_i32.float()
            * tensors["qr_scale"].float()
            * tensors["wq_b_scale"].float().view(1, -1)
        ).view(T, IDX_N_HEADS, IDX_HEAD_DIM)
        pair = q[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        q0, q1 = pair[..., 0], pair[..., 1]
        cos = tensors["cos"].float().view(T, 1, -1)
        sin = tensors["sin"].float().view(T, 1, -1)
        y0 = (q0 * cos - q1 * sin).to(torch.bfloat16)
        y1 = (q0 * sin + q1 * cos).to(torch.bfloat16)
        q = torch.cat((q[..., :-ROPE_HEAD_DIM], torch.stack((y0, y1), dim=-1).flatten(-2)), dim=-1)
        q = q.to(torch.bfloat16).float() @ tensors["hadamard"].float()
        weights = (tensors["x"].float() @ tensors["weights_proj"].float()) * WEIGHTS_SCALE
        cache_flat = tensors["idx_kv_cache"].reshape(CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
        scale_flat = tensors["idx_kv_scale"].float().reshape(CACHE_BLOCK_NUM * BLOCK_SIZE, 1)
        table = tensors["idx_block_table"]
        rows = [int(table[col // BLOCK_SIZE].item()) * BLOCK_SIZE + col % BLOCK_SIZE for col in range(max_visible)]
        kv_i8 = torch.stack([cache_flat[row] for row in rows]).to(torch.int32)
        kv_scale = torch.stack([scale_flat[row] for row in rows]).view(1, 1, max_visible)
        q_i8, q_scale = _quantize_rows(q.reshape(T * IDX_N_HEADS, IDX_HEAD_DIM))
        q_i8 = q_i8.view(T, IDX_N_HEADS, IDX_HEAD_DIM).to(torch.int32)
        q_scale = q_scale.view(T, IDX_N_HEADS, 1)
        score_i32 = torch.einsum("thd,cd->thc", q_i8, kv_i8)
        score = score_i32.float() * q_scale * kv_scale
        score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)
        columns = torch.arange(max_visible).unsqueeze(0)
        score = score.masked_fill(columns >= visible.unsqueeze(1), FP32_NEG_INF)
        score_full[:, :max_visible] = score
        for token in range(num_tokens):
            k = min(SCORE_CAP, int(visible[token].item()))
            if k > 0:
                selected = score[token].topk(k, dim=-1)[1]
                topk[token, :k] = selected.to(torch.int32) + INDEX_OFFSET

    return {
        "idx_kv_cache": tensors["idx_kv_cache"].detach().cpu().contiguous(),
        "idx_kv_scale": tensors["idx_kv_scale"].detach().cpu().contiguous(),
        "score": score_full,
        "topk_idxs": topk,
    }

