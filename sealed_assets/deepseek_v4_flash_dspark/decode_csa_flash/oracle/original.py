# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 CSA (Compressed Sparse Attention) decode orchestration.

This standalone harness targets the ratio-4 compression step used by the current
workflow checkpoint. It composes:

- hc_pre
- qkv_proj_rope
- main compressor (ratio=4, rotate=False)
- inner compressor (ratio=4, rotate=True)
- indexer
- sparse_attn_csa (with fused grouped o_proj)
- hc_post

The helper stack in this repo has already moved to the refreshed v4 contracts:
q_proj runs through the W8A8 path, sparse_attn_csa owns grouped o_proj, and the
indexer consumes a prepared `idx_kv_cache` instead of owning the inner
compressor itself. This file aligns to that stack instead of the older draft
surface.
"""



from config import (
    CASE_CONFIG as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_CMP_BLOCK_NUM,
    DECODE_IDX_BLOCK_NUM,
    DECODE_ORI_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    KV_ORI_TABLE_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_compressor_ratio4 import golden_compressor as _golden_main_compressor
from hc_post import golden_hc_post as _golden_hc_post
from hc_pre import golden_hc_pre as _golden_hc_pre
from decode_indexer import golden_indexer as _golden_indexer
from qkv_proj_rope import golden_qkv_proj_rope as _golden_qkv_proj_rope
from rmsnorm import golden_rms_norm as _golden_rms_norm
from decode_sparse_attn_csa import golden_sparse_attn as _golden_sparse_attn

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
INDEXER_SCORE_LEN = MAX_SEQ_LEN // 4
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
MAIN_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
MAIN_STATE_PHYSICAL_BLOCKS = 65
MAIN_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
MAIN_STATE_BLOCK_NUM = MAIN_STATE_PHYSICAL_BLOCKS
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_PHYSICAL_BLOCKS = 65
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = INNER_STATE_PHYSICAL_BLOCKS
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_TABLE_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM

# tiling
CSA_WB_TOKEN_TILE = 8


def golden_attention_csa(tensors):
    """Torch reference for the ratio-4 compression-step CSA orchestration."""
    import torch

    golden_compressor = _golden_main_compressor
    golden_hc_pre = _golden_hc_pre
    golden_indexer = _golden_indexer
    golden_qkv_proj_rope = _golden_qkv_proj_rope
    golden_rms_norm = _golden_rms_norm
    golden_sparse_attn = _golden_sparse_attn
    golden_hc_post = _golden_hc_post

    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    position_ids = tensors["position_ids"].to(torch.int64)
    position_ids_bsd = position_ids.reshape(B, S).to(torch.int32).contiguous()
    cmp_slot_mapping_bsd = tensors["cmp_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()
    idx_slot_mapping_bsd = tensors["idx_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()
    state_slot_mapping_bsd = tensors["state_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()
    inner_state_slot_mapping_bsd = tensors["inner_state_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_t = freqs_cos[position_ids].contiguous()
    rope_sin_t = freqs_sin[position_ids].contiguous()
    first_pos = position_ids.reshape(B, S)[:, 0]
    step_cos = freqs_cos[first_pos, :HALF_ROPE].float().contiguous()
    step_sin = freqs_sin[first_pos, :HALF_ROPE].float().contiguous()
    cmp_pos = first_pos + (COMPRESS_RATIO - (first_pos % COMPRESS_RATIO)) - COMPRESS_RATIO
    cmp_cos = freqs_cos[cmp_pos, :HALF_ROPE].float().contiguous()
    cmp_sin = freqs_sin[cmp_pos, :HALF_ROPE].float().contiguous()

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr_i8 = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_t,
        "rope_sin": rope_sin_t,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr_i8,
        "qr_scale": qr_scale,
    })

    kv_cache = tensors["kv_cache"]
    window_swa_indices = tensors["window_swa_indices"]
    window_swa_lens = tensors["window_swa_lens"]
    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]

    cmp_out = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    golden_compressor({
        "x": x_normed.reshape(B, S, D),
        "kv": cmp_out,
        "compress_state": tensors["compress_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": cmp_cos,
        "sin": cmp_sin,
        "cmp_kv_cache": cmp_kv,
        "position_ids": position_ids_bsd,
        "cmp_slot_mapping": cmp_slot_mapping_bsd,
        "state_slot_mapping": state_slot_mapping_bsd,
    })

    idx_kv = torch.zeros(B, S, IDX_HEAD_DIM, dtype=torch.float32)
    idx_score = torch.zeros(B, S, INDEXER_SCORE_LEN, dtype=torch.float32)
    idx_topk_full = torch.full((B, S, INDEXER_SCORE_LEN), -1, dtype=torch.int32)
    golden_indexer({
        "x": x_normed.reshape(B, S, D),
        "qr": qr_i8,
        "qr_scale": qr_scale,
        "wq_b": tensors["idx_wq_b"],
        "wq_b_scale": tensors["idx_wq_b_scale"],
        "weights_proj": tensors["weights_proj"],
        "cos": step_cos,
        "sin": step_sin,
        "hadamard": tensors["hadamard_idx"],
        "inner_kv": idx_kv,
        "inner_compress_state": tensors["inner_compress_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "inner_wkv": tensors["inner_wkv"],
        "inner_wgate": tensors["inner_wgate"],
        "inner_ape": tensors["inner_ape"],
        "inner_norm_w": tensors["inner_norm_w"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "idx_block_table": tensors["idx_block_table"],
        "score": idx_score,
        "topk_idxs": idx_topk_full,
        "position_ids": position_ids_bsd,
        "idx_slot_mapping": idx_slot_mapping_bsd,
        "inner_state_slot_mapping": inner_state_slot_mapping_bsd,
        "kv_seq_lens": tensors["kv_seq_lens"],
        "offset": torch.tensor(0, dtype=torch.int32),
    })

    ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    for t in range(T):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            blk_id = write_row // BLOCK_SIZE
            intra = write_row % BLOCK_SIZE
            kv_cache[blk_id, intra, 0] = kv[t]

    idx_topk_flat = idx_topk_full.view(T, INDEXER_SCORE_LEN)

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    # sparse_attn_csa folds the compressed-slot masking in (0 <= raw < floor((pos+1)/
    # COMPRESS_RATIO)); pass raw idx_topk + position so the golden masks the same way.
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "window_swa_indices": window_swa_indices,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "idx_topk": idx_topk_flat,
        "position_ids": position_ids.view(T, 1),
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    y = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })
    tensors["x_out"][:] = y


