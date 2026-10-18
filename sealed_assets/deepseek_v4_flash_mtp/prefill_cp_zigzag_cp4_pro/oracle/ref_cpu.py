"""Independent CPU oracle for the CP4 zigzag tail exchange."""

import torch


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalars: {sorted(scalars)}")
    local = inputs["local_kv_tail"]
    reverse_index = inputs["reverse_index"].to(torch.int64)
    cp_size, _, head_dim = local.shape
    tail_rows = local.shape[1] // 2
    segments = reverse_index.numel()
    logical = torch.empty(segments * tail_rows, head_dim, dtype=local.dtype)
    for segment in range(segments):
        rank_major = int(reverse_index[segment].item())
        rank, part = divmod(rank_major, 2)
        start = part * tail_rows
        logical[segment * tail_rows:(segment + 1) * tail_rows] = local[rank, start:start + tail_rows]
    logical = logical.unsqueeze(0).expand(cp_size, -1, -1).contiguous()
    return {
        "logical_tails_out": logical,
        "decode_raw_window_out": logical[:, -tail_rows:, :].contiguous(),
    }
