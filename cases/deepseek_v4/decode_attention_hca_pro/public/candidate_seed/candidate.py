# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 HCA (Hierarchical Compressed Attention) decode orchestration — `compress_ratio == 128` path.
Active in layers 3/5 of the model (2 of the 8 layers in demo). Has the main compressor (ratio=128,
overlap=False) but NO indexer; the compressed-portion topk for sparse_attn comes from a deterministic
index computation, not from a learned indexer score.
Companion files: attention_swa.py (ratio=0)
                 attention_csa_draft.py (ratio=4)."""


import pypto.language as pl

from config import (
    CASE_CONFIG as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from hc_pre import hc_pre
from hc_post import hc_post
from decode_qkv_proj_rope import attn_norm, qkv_proj_rope
from decode_compressor_ratio128 import compressor
from decode_sparse_attn import sparse_attn


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local (HCA: ratio-128 main compressor, no indexer)
COMPRESS_RATIO = 128  # HCA
OVERLAP = COMPRESS_RATIO == 4   # always False for HCA
COFF = 1 + int(OVERLAP)         # always 1 for HCA
MAIN_OUT_DIM = COFF * HEAD_DIM
ORI_MAX_BLOCKS = 1                  # WIN==BLOCK_SIZE → 1 ori block per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS  # ori KV pool size (matches sparse_attn ORI_BLOCK_NUM)
CMP_MAX_BLOCKS = 64                 # matches sparse_attn CMP_MAX_BLOCKS
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS  # cmp KV pool size (shared with the compressor's cmp_kv_cache)
# Main compressor state pool (kv + score channels merged into one paged FP32 buffer).
COMPRESS_STATE_MAX_BLOCKS = 64
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
SPARSE_IDX_TOPK = M.index_topk             # sparse_attn module's IDX_TOPK (static shape contract)
CMP_TOPK_CAPACITY = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_TOPK = min(CMP_TOPK_CAPACITY, SPARSE_IDX_TOPK)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK        # sparse_attn module's TOPK (= 640 for demo)
# Default fixture is a post-window non-compression step. The --start-pos fixture
# option covers post-window no-compression/aligned/boundary positions.
START_POS = 128
# tiling
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE


@pl.jit.inline
def attention_hca(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # main compressor (head_dim=HEAD_DIM, ratio=128, overlap=False)
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    # KV cache split into ori (sliding window) and cmp (compressed) pools to match sparse_attn's contract.
    # cmp_kv is shared with the compressor: it writes the compressed row directly into this pool.
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    # o_proj (fused into sparse_attn)
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    """HCA decode orchestration for compress_ratio=128."""
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post_t = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_mixed = hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post_t,
        comb_t,
    )

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    cmp_start_pos = pl.create_tensor([B], dtype=pl.INT32)
    cmp_cos = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_rope"):
        for b in pl.range(B):
            start_pos_b = pl.read(start_pos, [b])
            pl.write(cmp_start_pos, [b], pl.cast(start_pos_b, pl.INT32))
            cmp_offset_b = COMPRESS_RATIO - (start_pos_b % COMPRESS_RATIO)
            cmp_pos_b = pl.cast(start_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos_row = freqs_cos[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_sin_row = freqs_sin[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_cos[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_cos_row, target_type=pl.FP32)
            cmp_sin[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_sin_row, target_type=pl.FP32)
            for s in pl.range(S):
                t = b * S + s
                pos_b = pl.cast(start_pos_b + s, pl.INDEX)
                step_cos_row = pl.cast(freqs_cos[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                step_sin_row = pl.cast(freqs_sin[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(step_cos_row, target_type=pl.BF16, mode="rint")
                rope_sin_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(step_sin_row, target_type=pl.BF16, mode="rint")

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)        # unused on HCA path
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    x_normed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    x_normed = attn_norm(x_mixed, attn_norm_w, x_normed)
    q = qkv_proj_rope(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos_t,
        rope_sin_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )

    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_scatter_ori"):
        for b in pl.range(B):
            start_pos_b = pl.read(start_pos, [b])
            for s_idx in pl.range(S):
                ori_slot = (start_pos_b + s_idx) % WIN
                blk_id = pl.cast(pl.read(ori_block_table_flat, [b * ORI_MAX_BLOCKS + ori_slot // BLOCK_SIZE]), pl.INDEX)
                dst_row = blk_id * BLOCK_SIZE + ori_slot % BLOCK_SIZE
                kv_cache_flat[dst_row : dst_row + 1, 0 : HEAD_DIM] = kv[b * S + s_idx : b * S + s_idx + 1, 0 : HEAD_DIM]
    kv_cache = pl.reshape(kv_cache_flat, [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    cmp_kv_proj = pl.create_tensor([B, S, HEAD_DIM], dtype=pl.FP32)
    cmp_kv_proj, compress_state, cmp_kv = compressor(
        x_normed,
        cmp_kv_proj,
        compress_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_cos,
        cmp_sin,
        cmp_kv,
        cmp_block_table,
        cmp_start_pos,
    )

    # topk_idxs: [0..WIN) window plus deterministic compressed slots.
    # sparse_attn's static TOPK contract is SPARSE_TOPK (= WIN+IDX_TOPK = 640 in demo);
    # the actual valid count is bounded by seqused_kv inside sparse_attn. Build one
    # row in Vec, then broadcast-write to all T rows via a sequential range — a full
    # [T, SPARSE_TOPK] INT32 tile (320KB at T=128) busts the Vec budget.
    topk_idxs = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_topk"):
        win_idx = pl.arange(0, [1, WIN], dtype=pl.INT32)
        cmp_idx = pl.add(
            pl.arange(0, [1, CMP_TOPK], dtype=pl.INT32),
            pl.full([1, CMP_TOPK], dtype=pl.INT32, value=WIN),
        )
        if CMP_TOPK < SPARSE_IDX_TOPK:
            pad_idx = pl.full([1, SPARSE_IDX_TOPK - CMP_TOPK], dtype=pl.INT32, value=-1)
            topk_row = pl.concat(pl.concat(win_idx, cmp_idx), pad_idx)
        else:
            topk_row = pl.concat(win_idx, cmp_idx)
        for t in pl.range(T):
            topk_idxs[t : t + 1, 0 : SPARSE_TOPK] = topk_row

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn(
        q,
        kv_cache,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        topk_idxs,
        attn_sink,
        seqused_kv,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )

    attn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out = hc_post(
        attn_out_3d,
        x_hc,
        post_t,
        comb_t,
        x_out,
    )
    return x_out


@pl.jit
def attention_hca_test(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_out = attention_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, ori_block_table, cmp_kv, cmp_block_table,
        attn_sink, seqused_kv,
        wo_a, wo_b, wo_b_scale,
        x_out,
        start_pos,
    )
    return x_out
