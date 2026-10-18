# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ORACLE closure for deepseek_v4_flash_dspark.prefill_layer_hca_pro; fixed bindings from public config."""
from config import BLOCK_SIZE
from config import CSA_COMPRESS_RATIO as COMPRESS_RATIO
from config import EPS
from config import IDX_HEAD_DIM as HEAD_DIM
from config import IDX_NOPE_HEAD_DIM as NOPE_HEAD_DIM
from config import INNER_COMPRESS_STATE_DIM as COMPRESS_STATE_DIM
from config import INNER_OUT_DIM as OUT_DIM
from config import INNER_STATE_BLOCK_SIZE
from config import INNER_STATE_LEN as STATE_LEN
from config import INT8_AMAX_EPS
from config import INT8_SCALE_MAX
from config import MAX_SEQ_LEN
from config import ROPE_HEAD_DIM

PREFILL_STATE_TILE = 512

def golden_prefill_indexer_compressor(tensors):
    import torch

    token_count = tensors["x"].shape[0]
    kv_proj = tensors["x"].float() @ tensors["wkv"].float().t()  # wkv stored [OUT_DIM, D] for b_trans
    score_proj = tensors["x"].float() @ tensors["wgate"].float().t()
    compress_state_flat = tensors["compress_state"].view(
        tensors["compress_state"].shape[0] * INNER_STATE_BLOCK_SIZE,
        COMPRESS_STATE_DIM,
    )
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors["inner_compress_state_block_table"]
    idx_kv_cache = tensors["idx_kv_cache"]  # C8: INT8 KV
    idx_kv_scale = tensors["idx_kv_scale"]  # C8: per-position FP32 dequant scale
    cache_rows = idx_kv_cache.view(idx_kv_cache.shape[0] * BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    scale_rows = idx_kv_scale.view(idx_kv_scale.shape[0] * BLOCK_SIZE, 1, 1)[:, 0, 0]
    position_ids = tensors["position_ids"]
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    hadamard = tensors["hadamard"].float()

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // INNER_STATE_BLOCK_SIZE
        intra = abs_pos % INNER_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * INNER_STATE_BLOCK_SIZE + intra

    for tile_base in range(0, token_count, PREFILL_STATE_TILE):
        tile_end = min(tile_base + PREFILL_STATE_TILE, token_count)

        # Match the device ordering exactly: scatter the complete physical
        # state tile first, then pool every ratio-4 boundary in that tile.
        for token_id in range(tile_base, tile_end):
            dst_row = int(tensors["inner_state_slot_mapping"][token_id].item())
            if dst_row < 0:
                continue
            pos = int(position_ids[token_id].item())
            kv_state_flat[dst_row] = kv_proj[token_id]
            score_state_flat[dst_row] = score_proj[token_id] + ape[pos % COMPRESS_RATIO]

        for token_id in range(tile_base, tile_end):
            dst_row = int(tensors["idx_slot_mapping"][token_id].item())
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
            normed_fp32 = pooled * inv_rms * norm_w.float().view(1, HEAD_DIM)
            normed = normed_fp32.clone()
            normed[:, 0:NOPE_HEAD_DIM] = normed_fp32[:, 0:NOPE_HEAD_DIM].to(torch.bfloat16).float()
            rope_pair = normed_fp32[..., NOPE_HEAD_DIM:HEAD_DIM].unflatten(-1, (-1, 2))
            rope_even = rope_pair[..., 0]
            rope_odd = rope_pair[..., 1]
            cmp_pos = write_pos + 1 - COMPRESS_RATIO
            cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
            sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
            rot_even = rope_even * cos - rope_odd * sin
            rot_odd = rope_even * sin + rope_odd * cos
            normed[:, NOPE_HEAD_DIM:HEAD_DIM] = (
                torch.stack([rot_even, rot_odd], dim=-1).flatten(-2).to(torch.bfloat16).float()
            )
            final = normed.to(torch.bfloat16).float() @ hadamard
            final_bf16 = final.to(torch.bfloat16)[0]
            # C8 quant-on-write: int8 + per-position dequant scale of the bf16-rounded compressed row.
            row_bf16 = final_bf16.float()
            amax = row_bf16.abs().amax().clamp_min(INT8_AMAX_EPS)
            scale_q = INT8_SCALE_MAX / amax
            row_i8 = torch.round(row_bf16 * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
            cache_rows[dst_row] = row_i8
            scale_rows[dst_row] = 1.0 / scale_q
    tensors["idx_kv_cache"][:] = idx_kv_cache
    tensors["idx_kv_scale"][:] = idx_kv_scale


