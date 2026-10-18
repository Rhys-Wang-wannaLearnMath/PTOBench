"""Case-local CPU Oracle for the standalone decode indexer compressor."""

from __future__ import annotations

import torch


B, S, HEAD_DIM = 8, 1, 128
COMPRESS_RATIO, OUT_DIM = 4, 256
ROPE_HEAD_DIM, STATE_BLOCK_SIZE, BLOCK_SIZE = 64, 4, 128
EPS = 1e-6


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalars: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    x = tensors["x"].float()
    state = tensors["compress_state"]
    state_table = tensors["compress_state_block_table"]
    kv_projection = x @ tensors["wkv"].float()
    score_projection = x @ tensors["wgate"].float()
    ape = tensors["ape"]
    starts = tensors["start_pos"]
    pooled = torch.zeros(B, 1, HEAD_DIM, dtype=torch.float32)
    should_compress_rows = torch.zeros(B, dtype=torch.bool)

    for batch in range(B):
        start = int(starts[batch].item())
        pre_tokens = min(S, COMPRESS_RATIO - (start % COMPRESS_RATIO))
        should_compress = pre_tokens < S or (start + S) % COMPRESS_RATIO == 0
        boundary_end = start + pre_tokens - 1
        current_window = boundary_end - COMPRESS_RATIO + 1
        previous_window = current_window - COMPRESS_RATIO
        ape_row = start % COMPRESS_RATIO
        for token in range(S):
            score_projection[batch, token] += ape[(ape_row + token) % COMPRESS_RATIO]
            absolute = start + token
            block = int(state_table[batch, absolute // STATE_BLOCK_SIZE].item())
            intra = absolute % STATE_BLOCK_SIZE
            state[block, intra, :OUT_DIM] = kv_projection[batch, token]
            state[block, intra, OUT_DIM:] = score_projection[batch, token]
        if not should_compress:
            continue
        should_compress_rows[batch] = True
        kv_rows, score_rows = [], []
        for token in range(COMPRESS_RATIO):
            absolute = previous_window + token
            if absolute < 0:
                kv_rows.append(torch.zeros(HEAD_DIM))
                score_rows.append(torch.full((HEAD_DIM,), float("-inf")))
                continue
            block = int(state_table[batch, absolute // STATE_BLOCK_SIZE].item())
            intra = absolute % STATE_BLOCK_SIZE
            kv_rows.append(state[block, intra, :HEAD_DIM])
            score_rows.append(state[block, intra, OUT_DIM : OUT_DIM + HEAD_DIM])
        for token in range(COMPRESS_RATIO):
            absolute = current_window + token
            block = int(state_table[batch, absolute // STATE_BLOCK_SIZE].item())
            intra = absolute % STATE_BLOCK_SIZE
            kv_rows.append(state[block, intra, HEAD_DIM:OUT_DIM])
            score_rows.append(state[block, intra, OUT_DIM + HEAD_DIM :])
        kv_window = torch.stack(kv_rows).unsqueeze(0)
        score_window = torch.stack(score_rows).unsqueeze(0)
        pooled[batch : batch + 1] = (kv_window * score_window.softmax(dim=1)).sum(dim=1, keepdim=True)

    output = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    cache = tensors["idx_kv_cache"]
    cache_table = tensors["idx_block_table"]
    for batch in range(B):
        if not bool(should_compress_rows[batch]):
            continue
        value = pooled[batch : batch + 1]
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)
        value = value * tensors["norm_w"]
        pair = value[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        x0, x1 = pair[..., 0], pair[..., 1]
        cos, sin = tensors["cos"][batch].view(-1), tensors["sin"][batch].view(-1)
        rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)
        value = torch.cat((value[..., :-ROPE_HEAD_DIM], rotated), dim=-1)
        value = value.to(torch.bfloat16).float() @ tensors["hadamard"].float()
        output[batch : batch + 1, 0:1] = value
        cache_column = int(starts[batch].item()) // COMPRESS_RATIO
        block = int(cache_table[batch, cache_column // BLOCK_SIZE].item())
        cache[block, cache_column % BLOCK_SIZE, 0] = value[0, 0]

    return {
        "kv": output,
        "compress_state": state.detach().cpu().contiguous(),
        "idx_kv_cache": cache.detach().cpu().contiguous(),
    }
