"""Case-local CPU Oracle for the standalone packed prefill compressor."""

from __future__ import annotations

import torch


from config import (
    BLOCK_SIZE,
    COMPRESS_RATIO as RATIO,
    D,
    EPS,
    HEAD_DIM,
    INNER_STATE_BLOCK_NUM as STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE as STATE_BLOCK_SIZE,
    INT8_AMAX_EPS as INT8_EPS,
    INT8_SCALE_MAX as INT8_MAX,
    MAX_CMP_WRITES as MAX_WRITES,
    MAX_SEQ_LEN,
    NOPE_HEAD_DIM,
    OUT_DIM,
    PREFILL_IDX_BLOCK_NUM as CACHE_BLOCK_NUM,
    ROPE_HEAD_DIM,
    STATE_LEN,
    T,
)


def compute(inputs, scalars):
    if set(scalars) != {"num_tokens"}:
        raise ValueError(f"unexpected scalars: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    num_tokens = int(scalars["num_tokens"])
    kv_proj = tensors["x"].float() @ tensors["wkv"].float().t()
    score_proj = tensors["x"].float() @ tensors["wgate"].float().t()
    kv_state_flat = tensors["kv_state"].view(STATE_BLOCK_NUM * STATE_BLOCK_SIZE, OUT_DIM)
    score_state_flat = tensors["score_state"].view(STATE_BLOCK_NUM * STATE_BLOCK_SIZE, OUT_DIM)
    state_table = tensors["inner_compress_state_block_table"]
    cache = tensors["idx_kv_cache"]
    scale = tensors["idx_kv_scale"]
    cache_rows = cache.view(CACHE_BLOCK_NUM * BLOCK_SIZE, 1, HEAD_DIM)[:, 0]
    scale_rows = scale.view(CACHE_BLOCK_NUM * BLOCK_SIZE, 1, 1)[:, 0, 0]
    positions = tensors["position_ids"]
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    hadamard = tensors["hadamard"].float()
    kv_output = torch.zeros(MAX_WRITES, HEAD_DIM, dtype=torch.int8)

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
        pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
        pool_score = torch.full((STATE_LEN, HEAD_DIM), float("-inf"), dtype=torch.float32)
        for slot in range(RATIO):
            previous = previous_start + slot
            if write_pos >= 2 * RATIO - 1:
                row = state_row(previous)
                if row >= 0:
                    pool_kv[slot] = kv_state_flat[row, :HEAD_DIM]
                    pool_score[slot] = score_state_flat[row, :HEAD_DIM]
            current = current_start + slot
            row = state_row(current)
            if row >= 0:
                pool_kv[RATIO + slot] = kv_state_flat[row, HEAD_DIM:OUT_DIM]
                pool_score[RATIO + slot] = score_state_flat[row, HEAD_DIM:OUT_DIM]
        for token in range(num_tokens):
            pos = int(positions[token].item())
            if pos < previous_start or pos > write_pos:
                continue
            ape_slot = pos % RATIO
            if pos < current_start:
                pool_slot, col0 = pos - previous_start, 0
            else:
                pool_slot, col0 = RATIO + pos - current_start, HEAD_DIM
            pool_kv[pool_slot] = kv_proj[token, col0 : col0 + HEAD_DIM]
            pool_score[pool_slot] = score_proj[token, col0 : col0 + HEAD_DIM] + ape[ape_slot, col0 : col0 + HEAD_DIM]

        mi = pool_score[STATE_LEN - 1 : STATE_LEN].clone()
        li = torch.exp(mi - mi)
        oi = pool_kv[STATE_LEN - 1 : STATE_LEN].clone()
        for slot in range(STATE_LEN - 1):
            if slot < RATIO and write_pos < 2 * RATIO - 1:
                continue
            slot_score, slot_kv = pool_score[slot : slot + 1], pool_kv[slot : slot + 1]
            mi_next = torch.maximum(mi, slot_score)
            alpha, beta = torch.exp(mi - mi_next), torch.exp(slot_score - mi_next)
            li = alpha * li + beta
            oi = oi * alpha + slot_kv * beta
            mi = mi_next
        pooled = oi / li
        normed_fp32 = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + EPS) * norm_w.float().view(1, HEAD_DIM)
        normed = normed_fp32.clone()
        normed[:, :NOPE_HEAD_DIM] = normed_fp32[:, :NOPE_HEAD_DIM].to(torch.bfloat16).float()
        pair = normed_fp32[..., NOPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        even, odd = pair[..., 0], pair[..., 1]
        compressed_position = write_pos + 1 - RATIO
        cos = tensors["freqs_cos"][compressed_position : compressed_position + 1, : ROPE_HEAD_DIM // 2].float()
        sin = tensors["freqs_sin"][compressed_position : compressed_position + 1, : ROPE_HEAD_DIM // 2].float()
        normed[:, NOPE_HEAD_DIM:] = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2).to(torch.bfloat16).float()
        final = normed.to(torch.bfloat16).float() @ hadamard
        row_bf16 = final.to(torch.bfloat16)[0].float()
        amax = row_bf16.abs().amax().clamp_min(INT8_EPS)
        quant_scale = INT8_MAX / amax
        row_i8 = torch.round(row_bf16 * quant_scale).to(torch.int32).to(torch.float16).to(torch.int8)
        cache_rows[destination] = row_i8
        scale_rows[destination] = 1.0 / quant_scale
        if write_i < MAX_WRITES:
            kv_output[write_i] = row_i8
        write_i += 1

    for token in range(num_tokens):
        pos = int(positions[token].item())
        row = int(tensors["inner_state_slot_mapping"][token].item())
        if row < 0:
            continue
        kv_state_flat[row] = kv_proj[token]
        score_state_flat[row] = score_proj[token] + ape[pos % RATIO]

    return {
        "kv": kv_output,
        "kv_state": kv_state_flat.view_as(tensors["kv_state"]).detach().cpu().contiguous(),
        "score_state": score_state_flat.view_as(tensors["score_state"]).detach().cpu().contiguous(),
        "idx_kv_cache": cache.detach().cpu().contiguous(),
        "idx_kv_scale": scale.detach().cpu().contiguous(),
    }
