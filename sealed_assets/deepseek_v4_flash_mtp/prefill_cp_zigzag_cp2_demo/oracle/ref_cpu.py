"""Independent CPU golden for the selected fixed CP2 marker fixture."""
import torch
from config import TAIL_ROWS, HEAD_DIM, CP_SIZE, NUM_SEGMENTS, CP_TAIL_WINDOW_ROWS

def _segment_abs_starts(segment_lens):
    starts = []
    position = 0
    for length in segment_lens:
        starts.append(position)
        position += length
    return starts


def _encode_marker(abs_pos: int):
    import torch

    marker = torch.zeros(HEAD_DIM, dtype=torch.bfloat16)
    marker[1] = float(abs_pos // TAIL_ROWS)
    marker[2] = float(abs_pos % TAIL_ROWS)
    return marker


def golden_prefill_cp_zigzag_kv_tail_exchange(tensors):
    import torch

    segment_lens = [TAIL_ROWS] * NUM_SEGMENTS
    starts = _segment_abs_starts(segment_lens)
    logical = torch.zeros(CP_TAIL_WINDOW_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    for segment in range(NUM_SEGMENTS):
        for row in range(TAIL_ROWS):
            logical[segment * TAIL_ROWS + row] = _encode_marker(starts[segment] + row)
    tensors["logical_tails_out"][:] = logical.unsqueeze(0).expand(CP_SIZE, -1, -1)

    final_start = NUM_SEGMENTS * TAIL_ROWS - TAIL_ROWS
    decode_window = torch.stack([_encode_marker(final_start + row) for row in range(TAIL_ROWS)])
    tensors["decode_raw_window_out"][:] = decode_window.unsqueeze(0).expand(CP_SIZE, -1, -1)


def compute(inputs, scalars):
    if scalars:
        raise ValueError(f"unexpected scalars: {sorted(scalars)}")
    tensors = {
        "logical_tails_out": torch.zeros((CP_SIZE, CP_TAIL_WINDOW_ROWS, HEAD_DIM), dtype=torch.bfloat16),
        "decode_raw_window_out": torch.zeros((CP_SIZE, TAIL_ROWS, HEAD_DIM), dtype=torch.bfloat16),
    }
    golden_prefill_cp_zigzag_kv_tail_exchange(tensors)
    return tensors
