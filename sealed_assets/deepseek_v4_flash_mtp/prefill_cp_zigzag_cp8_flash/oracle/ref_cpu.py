import torch

from config import CP_SIZE, CP_TAIL_WINDOW_ROWS, HEAD_DIM, NUM_SEGMENTS, TAIL_ROWS, TENSOR_DTYPE


def compute(inputs):
    del inputs
    dtype = getattr(torch, TENSOR_DTYPE)
    logical = torch.zeros(CP_TAIL_WINDOW_ROWS, HEAD_DIM, dtype=dtype)
    positions = torch.arange(NUM_SEGMENTS * TAIL_ROWS)
    logical[:, 1] = (positions // TAIL_ROWS).to(dtype)
    logical[:, 2] = (positions % TAIL_ROWS).to(dtype)
    decode_window = logical[-TAIL_ROWS:].clone()
    return {
        "logical_tails_out": logical.unsqueeze(0).expand(CP_SIZE, -1, -1).clone(),
        "decode_raw_window_out": decode_window.unsqueeze(0).expand(CP_SIZE, -1, -1).clone(),
    }
