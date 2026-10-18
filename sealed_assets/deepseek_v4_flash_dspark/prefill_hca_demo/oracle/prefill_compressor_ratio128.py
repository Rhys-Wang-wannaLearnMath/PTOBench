# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from config import BLOCK_SIZE
from config import COMPRESS_RATIO, COMPRESS_STATE_DIM, EPS, HCA_CMP_BLOCK_NUM, HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HEAD_DIM, MAX_SEQ_LEN, NOPE_HEAD_DIM, OUT_DIM, ROPE_HALF, STATE_LEN

def golden_prefill_compressor_ratio128(tensors):
    import torch
    num_tokens = int(tensors['num_tokens'])
    kv_proj = tensors['x'].float() @ tensors['wkv'].float().t()
    score_proj = tensors['x'].float() @ tensors['wgate'].float().t()
    compress_state_flat = tensors['compress_state'].view(HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors['compress_state_block_table']
    cmp_kv_flat = tensors['cmp_kv'].view(HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * HCA_STATE_BLOCK_SIZE + intra
    for token_id in range(num_tokens):
        dst_row = int(tensors['cmp_slot_mapping'][token_id].item())
        if dst_row < 0:
            continue
        write_pos = int(tensors['position_ids'][token_id].item())
        pool_kv_state = torch.zeros(STATE_LEN, OUT_DIM, dtype=torch.float32)
        pool_score_state = torch.zeros(STATE_LEN, OUT_DIM, dtype=torch.float32)
        for slot in range(STATE_LEN):
            row = state_row(write_pos + 1 - COMPRESS_RATIO + slot)
            if row >= 0:
                pool_kv_state[slot] = kv_state_flat[row]
                pool_score_state[slot] = score_state_flat[row]
        for t in range(num_tokens):
            pos = int(tensors['position_ids'][t].item())
            if pos > write_pos:
                continue
            slot = pos % COMPRESS_RATIO
            pool_kv_state[slot] = kv_proj[t]
            pool_score_state[slot] = score_proj[t] + tensors['ape'][slot]
        pooled = (pool_kv_state * pool_score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
        inv = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = pooled * inv * tensors['norm_w'].float().view(1, HEAD_DIM)
        rope_pair = normed[..., NOPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        even = rope_pair[..., 0].float()
        odd = rope_pair[..., 1].float()
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors['freqs_cos'][cmp_pos:cmp_pos + 1, 0:ROPE_HALF].float()
        sin = tensors['freqs_sin'][cmp_pos:cmp_pos + 1, 0:ROPE_HALF].float()
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        normed[:, NOPE_HEAD_DIM:] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cmp_kv_flat[dst_row] = normed[0]
    for t in range(num_tokens):
        pos = int(tensors['position_ids'][t].item())
        dst_row = int(tensors['state_slot_mapping'][t].item())
        if dst_row < 0:
            continue
        slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors['ape'][slot]
