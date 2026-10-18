"""Case-local External CPU Oracle for deepseek_v4.compressor_ratio4_flash.

This file copies the reviewed Torch reference logic from the frozen official
Golden source and returns every output/inout tensor required by the frozen
TensorSpec contract. It does not import candidate code or read NPU outputs.
"""

from __future__ import annotations

import torch

from config import B, COMPRESS_RATIO, EPS, HEAD_DIM, IDX_KV_LEN, OUT_DIM, S, STATE_LEN


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    x = tensors["x"].float()
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()
    kv_cache = tensors["kv_cache"]
    start_pos = int(tensors["start_pos"])
    rotate = bool(tensors["rotate"])
    bsz, _, _ = x.shape
    ratio, d, rd = COMPRESS_RATIO, hadamard.shape[0], tensors["even_select"].shape[0]

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]
    kv_proj = kv

    pre_tokens = min(S, ratio - (start_pos % ratio))
    should_compress = pre_tokens < S or (start_pos + S) % ratio == 0

    # Per-token ape add + state scatter; slots wrap when the S-token step crosses a window boundary.
    ape_row_g = start_pos % ratio
    for s in range(pre_tokens):
        token_ape_row = (ape_row_g + s) % ratio
        score[:, s, :] = score[:, s, :] + ape[token_ape_row]
        kv_state[:bsz, ratio + token_ape_row] = kv[:, s, :]
        score_state[:bsz, ratio + token_ape_row] = score[:, s, :]

    if should_compress:
        kvs = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)
        scs = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1)
        kv = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)   # [B, 1, HEAD_DIM]
        kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
        score_state[:bsz, :ratio] = score_state[:bsz, ratio:]

    if pre_tokens < S:
        for s in range(pre_tokens, S):
            token_ape_row = (ape_row_g + s) % ratio
            score[:, s, :] = score[:, s, :] + ape[token_ape_row]
            kv_state[:bsz, ratio + token_ape_row] = kv_proj[:, s, :]
            score_state[:bsz, ratio + token_ape_row] = score[:, s, :]

    tensors["kv_state"][:] = kv_state
    tensors["score_state"][:] = score_state

    if not should_compress:
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return (w * x).to(torch.bfloat16)

    kv = rmsnorm(kv.to(torch.bfloat16), norm_w)

    x_pair = kv[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
    y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)

    kv = torch.cat([kv[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1).float()

    if rotate:
        kv = kv @ hadamard
    # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0
    # so the golden matches its [B, S, HEAD_DIM] zero-init.
    tensors["kv"][:, 0:1, :] = kv

    kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)

    tensors["kv_cache"][:] = kv_cache


_OUTPUT_SPECS = [
    {"name": "kv", "shape": [B, S, HEAD_DIM], "dtype": "torch.float32", "role": "output"},
    {"name": "kv_state", "shape": [B, STATE_LEN, OUT_DIM], "dtype": "torch.float32", "role": "inout"},
    {"name": "score_state", "shape": [B, STATE_LEN, OUT_DIM], "dtype": "torch.float32", "role": "inout"},
    {"name": "kv_cache", "shape": [B, IDX_KV_LEN, HEAD_DIM], "dtype": "torch.bfloat16", "role": "inout"},
]


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
