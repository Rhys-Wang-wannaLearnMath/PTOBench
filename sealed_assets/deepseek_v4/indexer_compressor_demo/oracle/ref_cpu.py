"""Case-local CPU Oracle for the standalone overlap indexer compressor."""

from __future__ import annotations

import torch


B, S, HEAD_DIM = 8, 1, 128
RATIO, OUT_DIM, ROPE_HEAD_DIM = 4, 256, 64
EPS = 1e-6


def compute(inputs, scalars):
    if set(scalars) != {"start_pos", "rotate"}:
        raise ValueError(f"unexpected scalars: {sorted(scalars)}")
    tensors = {name: value.clone() for name, value in inputs.items()}
    start_pos = int(scalars["start_pos"])
    rotate = bool(scalars["rotate"])
    x = tensors["x"].float()
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    kv_cache = tensors["kv_cache"]
    kv_projection = x @ tensors["wkv"].float()
    score = x @ tensors["wgate"].float()
    pre_tokens = min(S, RATIO - (start_pos % RATIO))
    should_compress = pre_tokens < S or (start_pos + S) % RATIO == 0
    ape_row = start_pos % RATIO

    for token in range(pre_tokens):
        slot = (ape_row + token) % RATIO
        score[:, token] += tensors["ape"][slot]
        kv_state[:, RATIO + slot] = kv_projection[:, token]
        score_state[:, RATIO + slot] = score[:, token]

    if should_compress:
        kv_window = torch.cat((kv_state[:, :RATIO, :HEAD_DIM], kv_state[:, RATIO:, HEAD_DIM:]), dim=1)
        score_window = torch.cat((score_state[:, :RATIO, :HEAD_DIM], score_state[:, RATIO:, HEAD_DIM:]), dim=1)
        kv = (kv_window * score_window.softmax(dim=1)).sum(dim=1, keepdim=True)
        kv_state[:, :RATIO] = kv_state[:, RATIO:]
        score_state[:, :RATIO] = score_state[:, RATIO:]
    else:
        kv = kv_projection

    for token in range(pre_tokens, S):
        slot = (ape_row + token) % RATIO
        score[:, token] += tensors["ape"][slot]
        kv_state[:, RATIO + slot] = kv_projection[:, token]
        score_state[:, RATIO + slot] = score[:, token]

    output = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    if should_compress:
        kv = kv.to(torch.bfloat16).float()
        kv = kv * torch.rsqrt(kv.square().mean(-1, keepdim=True) + EPS)
        kv = (tensors["norm_w"] * kv).to(torch.bfloat16)
        pair = kv[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        x0, x1 = pair[..., 0], pair[..., 1]
        cos, sin = tensors["cos"].view(-1), tensors["sin"].view(-1)
        y0 = (x0 * cos - x1 * sin).to(torch.bfloat16)
        y1 = (x0 * sin + x1 * cos).to(torch.bfloat16)
        kv = torch.cat((kv[..., :-ROPE_HEAD_DIM], torch.stack((y0, y1), dim=-1).flatten(-2)), dim=-1).float()
        if rotate:
            kv = kv @ tensors["hadamard"].float()
        output[:, 0:1] = kv
        kv_cache[:, start_pos // RATIO] = kv.squeeze(1)

    return {
        "kv": output,
        "kv_state": kv_state.detach().cpu().contiguous(),
        "score_state": score_state.detach().cpu().contiguous(),
        "kv_cache": kv_cache.detach().cpu().contiguous(),
    }
