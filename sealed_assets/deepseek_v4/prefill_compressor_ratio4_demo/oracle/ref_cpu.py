"""Case-local External CPU Oracle for deepseek_v4.prefill_compressor_ratio4.

This file copies the reviewed Torch reference logic from the frozen official
Golden source and returns every output/inout tensor required by the frozen
TensorSpec contract. It does not import candidate code or read NPU outputs.
"""

from __future__ import annotations

import torch

from config import (
    BLOCK_SIZE, COMPRESS_RATIO, CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE,
    D, EPS, HEAD_DIM, MAX_SEQ_LEN, NOPE_HEAD_DIM, OUT_DIM,
    PREFILL_CMP_BLOCK_NUM, ROPE_HEAD_DIM, STATE_LEN, T,
)


def golden_prefill_compressor_ratio4(tensors):
    """Packed token-major torch reference for ratio-4 prefill compressor."""
    import torch

    x = tensors["x"].view(T, D).float()
    kv_state_flat = tensors["kv_state"].view(CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, OUT_DIM)
    score_state_flat = tensors["score_state"].view(CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, OUT_DIM)
    state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cmp_kv = tensors["cmp_kv"]
    cache_rows = cmp_kv.view(cmp_kv.shape[0] * BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    position_ids = tensors["position_ids"]

    kv_proj = x @ wkv.t()    # wkv stored [OUT_DIM, D] for b_trans
    score_proj = x @ wgate.t()

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * CSA_STATE_BLOCK_SIZE + intra

    for token_id in range(int(tensors["num_tokens"])):
        dst_row = int(tensors["cmp_slot_mapping"][token_id].item())
        if dst_row < 0:
            continue
        write_pos = int(position_ids[token_id].item())
        cur_start = write_pos + 1 - COMPRESS_RATIO
        prev_start = cur_start - COMPRESS_RATIO
        pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
        pool_score = torch.full((STATE_LEN, HEAD_DIM), float("-inf"), dtype=torch.float32)

        for s in range(COMPRESS_RATIO):
            prev_abs = prev_start + s
            if write_pos >= 2 * COMPRESS_RATIO - 1:
                prev_row = state_row(prev_abs)
                if prev_row >= 0:
                    pool_kv[s] = kv_state_flat[prev_row, :HEAD_DIM]
                    pool_score[s] = score_state_flat[prev_row, :HEAD_DIM]

            cur_abs = cur_start + s
            cur_row = state_row(cur_abs)
            if cur_row >= 0:
                pool_kv[COMPRESS_RATIO + s] = kv_state_flat[cur_row, HEAD_DIM:OUT_DIM]
                pool_score[COMPRESS_RATIO + s] = score_state_flat[cur_row, HEAD_DIM:OUT_DIM]

        for t in range(int(tensors["num_tokens"])):
            pos = int(position_ids[t].item())
            if pos < prev_start or pos > write_pos:
                continue
            if pos < cur_start:
                pool_slot = pos - prev_start
                col0 = 0
            else:
                pool_slot = COMPRESS_RATIO + pos - cur_start
                col0 = HEAD_DIM
            ape_slot = pos % COMPRESS_RATIO
            pool_kv[pool_slot] = kv_proj[t, col0 : col0 + HEAD_DIM]
            pool_score[pool_slot] = score_proj[t, col0 : col0 + HEAD_DIM] + ape[ape_slot, col0 : col0 + HEAD_DIM]

        init_slot = STATE_LEN - 1
        mi = pool_score[init_slot : init_slot + 1].clone()
        li = torch.exp(mi - mi)
        oi = pool_kv[init_slot : init_slot + 1].clone()
        for slot_i in range(STATE_LEN - 1):
            if slot_i < COMPRESS_RATIO and write_pos < 2 * COMPRESS_RATIO - 1:
                continue
            slot_score = pool_score[slot_i : slot_i + 1]
            slot_kv = pool_kv[slot_i : slot_i + 1]
            mi_next = torch.maximum(mi, slot_score)
            alpha = torch.exp(mi - mi_next)
            beta = torch.exp(slot_score - mi_next)
            li = alpha * li + beta
            oi = oi * alpha + slot_kv * beta
            mi = mi_next
        pooled = oi / li
        inv_rms = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = pooled * inv_rms * norm_w.float().view(1, HEAD_DIM)
        rope_pair = normed[..., NOPE_HEAD_DIM:HEAD_DIM].unflatten(-1, (-1, 2))
        rope_even = rope_pair[..., 0]
        rope_odd = rope_pair[..., 1]
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
        rot_even = rope_even * cos - rope_odd * sin
        rot_odd = rope_even * sin + rope_odd * cos
        normed[:, NOPE_HEAD_DIM:HEAD_DIM] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cache_rows[dst_row] = normed.to(torch.bfloat16)[0]

    for t in range(int(tensors["num_tokens"])):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        ape_slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][ape_slot]
    tensors["cmp_kv"][:] = cmp_kv
    tensors["kv_state"][:] = kv_state_flat.view_as(tensors["kv_state"])
    tensors["score_state"][:] = score_state_flat.view_as(tensors["score_state"])


_OUTPUT_SPECS = [
    {"name": "kv_state", "shape": [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM], "dtype": "torch.float32", "role": "inout"},
    {"name": "score_state", "shape": [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM], "dtype": "torch.float32", "role": "inout"},
    {"name": "cmp_kv", "shape": [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], "dtype": "torch.bfloat16", "role": "inout"},
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
    tensors.setdefault("num_tokens", T)
    for spec in _OUTPUT_SPECS:
        if spec["name"] not in tensors:
            tensors[spec["name"]] = torch.zeros(tuple(spec["shape"]), dtype=_dtype(spec["dtype"]))
    golden_prefill_compressor_ratio4(tensors)
    return {spec["name"]: tensors[spec["name"]].detach().cpu().contiguous() for spec in _OUTPUT_SPECS}
