"""Independent CPU Oracle for the legacy ratio-128 decode compressor."""

from __future__ import annotations

import torch

from config import (
    B,
    COMPRESS_RATIO,
    EPS,
    HEAD_DIM,
    OUT_SHAPE,
    OVERLAP,
    ROPE_HEAD_DIM,
    ROTATE,
    START_POS,
)


def golden_deepseek_v4_decode_compressor(tensors):
    x = tensors["x"]
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"].float()
    norm_w = tensors["norm_w"].float()
    cos = tensors["cos"].float()
    sin = tensors["sin"].float()
    hadamard = tensors["hadamard"].float()

    batch_size, _, _ = x.shape
    ratio, overlap, rotate = COMPRESS_RATIO, OVERLAP, ROTATE
    head_dim, rope_head_dim = HEAD_DIM, ROPE_HEAD_DIM
    x = x.float()
    kv = x.view(batch_size, -1) @ wkv.T
    score = x.view(batch_size, -1) @ wgate.T

    should_compress = (START_POS + 1) % ratio == 0
    score = score + ape[START_POS % ratio]
    if overlap:
        raise ValueError("ratio-128 Oracle requires the fixed non-overlap contract")
    kv_state[:batch_size, START_POS % ratio] = kv
    score_state[:batch_size, START_POS % ratio] = score
    if should_compress:
        kv = (kv_state[:batch_size] * score_state[:batch_size].softmax(dim=1)).sum(
            dim=1, keepdim=True,
        )

    if not should_compress:
        tensors["out"][:] = torch.zeros(OUT_SHAPE, dtype=torch.bfloat16)
        return

    kv_compressed = kv.squeeze(1)
    kv_compressed = (
        kv_compressed
        * torch.rsqrt(kv_compressed.square().mean(-1, keepdim=True) + EPS)
        * norm_w
    )

    half_rope_dim = rope_head_dim // 2
    x_lo = kv_compressed[..., -rope_head_dim:-half_rope_dim]
    x_hi = kv_compressed[..., -half_rope_dim:]
    cos_values, sin_values = cos.view(-1), sin.view(-1)
    y_lo = x_lo * cos_values - x_hi * sin_values
    y_hi = x_lo * sin_values + x_hi * cos_values
    kv_compressed = torch.cat(
        [kv_compressed[..., :-rope_head_dim], y_lo, y_hi], dim=-1,
    )

    if rotate:
        kv_compressed = (kv_compressed @ hadamard).to(torch.bfloat16).float()

    tensors["out"][:] = kv_compressed.to(torch.bfloat16)


def compute(inputs, scalars):
    tensors = {name: value.clone() for name, value in inputs.items()}
    tensors.update(dict(scalars))
    tensors["out"] = torch.zeros(OUT_SHAPE, dtype=torch.bfloat16)
    golden_deepseek_v4_decode_compressor(tensors)
    return {
        name: tensors[name].detach().cpu().contiguous()
        for name in ("kv_state", "score_state", "out")
    }
