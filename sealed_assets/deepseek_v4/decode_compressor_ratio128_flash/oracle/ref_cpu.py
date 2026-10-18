"""Case-local External CPU Oracle for deepseek_v4.decode_compressor_ratio128.

This file copies the reviewed Torch reference logic from the frozen official
Golden source and returns every output/inout tensor required by the frozen
TensorSpec contract. It does not import candidate code or read NPU outputs.
"""

from __future__ import annotations

import torch

BLOCK_SIZE = 128
COMPRESS_RATIO = 128
COMPRESS_STATE_BLOCK_SIZE = 8
EPS = 1e-06
HEAD_DIM = 512
OUT_DIM = 512
ROPE_HEAD_DIM = 64
S = 1


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap).

    Operates on paged caches: compress_state (kv + score channels merged) and cmp_kv_cache,
    each addressed via the corresponding block_table.
    """
    import torch

    x = tensors["x"].float()
    compress_state_block_table = tensors["compress_state_block_table"]
    cmp_block_table = tensors["cmp_block_table"]
    # Read/write state cache through absolute-position block-table addressing,
    # matching vLLM's DeepSeek V4 compressor contract. APE remains modulo ratio.
    compress_state = tensors["compress_state"]

    def read_state_row(b, pos):
        logical_blk = pos // COMPRESS_STATE_BLOCK_SIZE
        intra = pos % COMPRESS_STATE_BLOCK_SIZE
        sblk = int(compress_state_block_table[b, logical_blk].item())
        return (
            compress_state[sblk, intra, :OUT_DIM],
            compress_state[sblk, intra, OUT_DIM:2 * OUT_DIM],
        )

    def write_state_row(b, pos, kv_row, score_row):
        logical_blk = pos // COMPRESS_STATE_BLOCK_SIZE
        intra = pos % COMPRESS_STATE_BLOCK_SIZE
        sblk = int(compress_state_block_table[b, logical_blk].item())
        compress_state[sblk, intra, :OUT_DIM] = kv_row
        compress_state[sblk, intra, OUT_DIM:2 * OUT_DIM] = score_row

    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    cmp_kv_cache = tensors["cmp_kv_cache"]
    start_pos_t = tensors["start_pos"]
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]
    kv_proj = kv
    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        start_pos = int(start_pos_t[b].item())
        pre_tokens = min(S, ratio - (start_pos % ratio))
        should_compress = pre_tokens < S or (start_pos + S) % ratio == 0
        ape_row_g = start_pos % ratio

        for s in range(pre_tokens):
            token_ape_row = (ape_row_g + s) % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            write_state_row(b, start_pos + s, kv[b, s, :], score[b, s, :])

        if should_compress:
            should_compress_rows[b] = True
            compress_pos = start_pos + (ratio - 1 - (start_pos % ratio))
            kv_rows = []
            score_rows = []
            for pos in range(compress_pos - ratio + 1, compress_pos + 1):
                kv_row, score_row = read_state_row(b, pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            kv_state = torch.stack(kv_rows, dim=0).unsqueeze(0)
            score_state = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)

        if pre_tokens < S:
            for s in range(pre_tokens, S):
                token_ape_row = (ape_row_g + s) % ratio
                score[b, s, :] = score[b, s, :] + ape[token_ape_row]
                write_state_row(b, start_pos + s, kv_proj[b, s, :], score[b, s, :])
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

        # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0.
        tensors["kv"][b : b + 1, 0:1, :] = kv_b

        cache_col = start_pos // ratio
        logical_blk = cache_col // BLOCK_SIZE
        intra_offset = cache_col % BLOCK_SIZE
        cblk = int(cmp_block_table[b, logical_blk].item())
        cmp_kv_cache[cblk, intra_offset, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


_OUTPUT_SPECS = [{'name': 'kv', 'shape': [8, 1, 512], 'dtype': 'torch.float32', 'role': 'output'}, {'name': 'compress_state', 'shape': [512, 8, 1024], 'dtype': 'torch.float32', 'role': 'inout'}, {'name': 'cmp_kv_cache', 'shape': [512, 128, 1, 512], 'dtype': 'torch.bfloat16', 'role': 'inout'}]


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
