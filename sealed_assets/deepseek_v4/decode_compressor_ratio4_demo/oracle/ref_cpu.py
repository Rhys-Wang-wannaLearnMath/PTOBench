"""Case-local External CPU Oracle for deepseek_v4.decode_compressor_ratio4.

This file copies the reviewed Torch reference logic from the frozen official
Golden source and returns every output/inout tensor required by the frozen
TensorSpec contract. It does not import candidate code or read NPU outputs.
"""

from __future__ import annotations

import torch

BLOCK_SIZE = 128
COMPRESS_RATIO = 4
COMPRESS_STATE_BLOCK_SIZE = 4
EPS = 1e-06
HEAD_DIM = 512
OUT_DIM = 1024
ROPE_HEAD_DIM = 64
S = 1


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

    x = tensors["x"].float()
    compress_state = tensors["compress_state"]
    compress_state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    cmp_kv_cache = tensors["cmp_kv_cache"]
    cmp_block_table = tensors["cmp_block_table"]
    start_pos_t = tensors["start_pos"]
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]

    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        start_pos = int(start_pos_t[b].item())
        pre_tokens = min(S, ratio - (start_pos % ratio))
        should_compress = pre_tokens < S or (start_pos + S) % ratio == 0
        boundary_end = start_pos + pre_tokens - 1
        cur_window_start = boundary_end - ratio + 1
        prev_window_start = cur_window_start - ratio

        # Per-token ape add + state scatter through the compressor-state block table.
        ape_row_g = start_pos % ratio
        for s in range(S):
            token_ape_row = (ape_row_g + s) % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            abs_pos = start_pos + s
            blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
            intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
            compress_state[blk_id, intra, :OUT_DIM] = kv[b, s, :]
            compress_state[blk_id, intra, OUT_DIM:] = score[b, s, :]

        if should_compress:
            should_compress_rows[b] = True
            kv_rows = []
            score_rows = []
            for s in range(ratio):
                abs_pos = prev_window_start + s
                if abs_pos < 0:
                    kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device))
                    score_rows.append(torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device))
                    continue
                blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
                intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
                kv_rows.append(compress_state[blk_id, intra, :HEAD_DIM])
                score_rows.append(compress_state[blk_id, intra, OUT_DIM:OUT_DIM + HEAD_DIM])
            for s in range(ratio):
                abs_pos = cur_window_start + s
                blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
                intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
                kv_rows.append(compress_state[blk_id, intra, HEAD_DIM:OUT_DIM])
                score_rows.append(compress_state[blk_id, intra, OUT_DIM + HEAD_DIM:])
            kvs = torch.stack(kv_rows, dim=0).unsqueeze(0)
            scs = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)

    tensors["compress_state"][:] = compress_state

    if not bool(should_compress_rows.any()):
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x

    for b in range(bsz):
        if not bool(should_compress_rows[b]):
            continue
        start_pos = int(start_pos_t[b].item())
        kv_b = rmsnorm(pooled[b : b + 1], norm_w)

        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_v, sin_v = cos[b].view(-1), sin[b].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0
        # so the golden matches its [B, S, HEAD_DIM] zero-init.
        tensors["kv"][b : b + 1, 0:1, :] = kv_b

        cmp_slot = start_pos // ratio
        blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
        cmp_kv_cache[blk_id, cmp_slot % BLOCK_SIZE, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


_OUTPUT_SPECS = [{'name': 'kv', 'shape': [8, 1, 512], 'dtype': 'torch.float32', 'role': 'output'}, {'name': 'compress_state', 'shape': [512, 4, 2048], 'dtype': 'torch.float32', 'role': 'inout'}, {'name': 'cmp_kv_cache', 'shape': [512, 128, 1, 512], 'dtype': 'torch.bfloat16', 'role': 'inout'}]


def _dtype(name: str) -> torch.dtype:
    value = name.removeprefix("torch.")
    dtype = getattr(torch, value)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported dtype {name}")
    return dtype


def compute(inputs, scalars):
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors.update(dict(scalars))
    for spec in _OUTPUT_SPECS:
        if spec["name"] not in tensors:
            tensors[spec["name"]] = torch.zeros(tuple(spec["shape"]), dtype=_dtype(spec["dtype"]))
    golden_compressor(tensors)
    return {spec["name"]: tensors[spec["name"]].detach().cpu().contiguous() for spec in _OUTPUT_SPECS}
