"""Case-local External CPU Oracle for deepseek_v4.deepseek_v4_decode_compressor_ratio4.

This file copies the reviewed Torch reference logic from the frozen official
Golden source and returns every output/inout tensor required by the frozen
TensorSpec contract. It does not import candidate code or read NPU outputs.
"""

from __future__ import annotations

import torch

from config import B, COMPRESS_RATIO, EPS, HEAD_DIM, OVERLAP, ROPE_HEAD_DIM, ROTATE, START_POS


def golden_deepseek_v4_decode_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

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

    bsz, _, _ = x.shape
    ratio, overlap, rotate, d, rd = COMPRESS_RATIO, OVERLAP, ROTATE, HEAD_DIM, ROPE_HEAD_DIM
    dtype = x.dtype
    x = x.float()
    kv = x.view(bsz, -1) @ wkv.T
    score = x.view(bsz, -1) @ wgate.T

    should_compress = (START_POS + 1) % ratio == 0
    score = score + ape[START_POS % ratio]
    if overlap:
        kv_state[:bsz, ratio + START_POS % ratio] = kv
        score_state[:bsz, ratio + START_POS % ratio] = score
        if should_compress:
            kvs = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)
            scs = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1)
            kv = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)
            kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
            score_state[:bsz, :ratio] = score_state[:bsz, ratio:]

    if not should_compress:
        tensors["out"][:] = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)
        return

    kv_c = kv.squeeze(1)
    kv_c = kv_c * torch.rsqrt(kv_c.square().mean(-1, keepdim=True) + EPS) * norm_w

    half_rd = rd // 2
    x_lo = kv_c[..., -rd:-half_rd]
    x_hi = kv_c[..., -half_rd:]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y_lo = x_lo * cos_v - x_hi * sin_v
    y_hi = x_lo * sin_v + x_hi * cos_v
    kv_c = torch.cat([kv_c[..., :-rd], y_lo, y_hi], dim=-1)

    if rotate:
        kv_c = (kv_c @ hadamard).to(torch.bfloat16).float()
    else:
        pass

    tensors["out"][:] = kv_c.to(torch.bfloat16)


_OUTPUT_SPECS = [{'name': 'kv_state', 'shape': [16, 8, 1024], 'dtype': 'torch.float32', 'role': 'inout'}, {'name': 'score_state', 'shape': [16, 8, 1024], 'dtype': 'torch.float32', 'role': 'inout'}, {'name': 'out', 'shape': [16, 512], 'dtype': 'torch.bfloat16', 'role': 'output'}]


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
    golden_deepseek_v4_decode_compressor(tensors)
    return {spec["name"]: tensors[spec["name"]].detach().cpu().contiguous() for spec in _OUTPUT_SPECS}
