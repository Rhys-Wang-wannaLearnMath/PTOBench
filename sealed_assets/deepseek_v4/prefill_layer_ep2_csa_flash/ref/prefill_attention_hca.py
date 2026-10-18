# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill HCA attention bring-up.

Correctness-first standalone for the ratio-128 HCA prefill path. The public
contract is single-request token-major prefill: the layer owns the
per-request loop and feeds this op one contiguous run of <=T tokens. HCA
consumes lowered metadata such as position_ids, dense slot mappings, and sparse
indices.
"""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    CASE_CONFIG as M,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_SEQ,
)
from hc_post_v2 import hc_post
from hc_pre_v2 import hc_pre
from prefill_compressor_ratio128 import (
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    prefill_compressor_ratio128,
)
from prefill_qkv_proj_rope_v2 import materialize_rope_rows, qkv_proj_rope
from rmsnorm import rms_norm
from prefill_sparse_attn_v2 import prefill_sparse_attn


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
ROPE_HALF = ROPE_DIM // 2
NOPE_HEAD_DIM = M.nope_head_dim
NOPE_DIM = NOPE_HEAD_DIM
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
IDX_TOPK = M.index_topk
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# prefill_sparse_attn cache/topk contract (mirrors prefill_sparse_attn).
SPARSE_TOPK = WIN + IDX_TOPK
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_ORI_BLOCK_NUM = B * SPARSE_ORI_MAX_BLOCKS
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
SPARSE_CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM

COMPRESS_RATIO = 128
MAIN_OUT_DIM = HEAD_DIM
MAIN_STATE_LEN = COMPRESS_RATIO
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
START_POS = 0
HCA_ORI_BLOCK_NUM = SPARSE_ORI_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = SPARSE_CMP_BLOCK_NUM
WRITEBACK_GUARD_TILE = 16

assert S == COMPRESS_RATIO, "first prefill HCA bring-up targets one ratio-128 prompt chunk"
assert WIN == BLOCK_SIZE, "prefill HCA currently assumes one window page per batch"
assert SPARSE_ORI_MAX_BLOCKS * BLOCK_SIZE >= S, "prefill HCA ori cache pool is too small"
assert SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE >= PREFILL_MAX_COMPRESSED, "prefill HCA cmp table is too small"
assert SPARSE_CMP_BLOCK_NUM >= SPARSE_CMP_MAX_BLOCKS, "prefill HCA cmp physical pool is too small"
assert PREFILL_COMPRESSED_LEN == 1


@pl.jit.inline
def prefill_attention_hca(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_kv_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_norm(x_mixed, attn_norm_w, x_normed)

    rope_cos_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        position_ids,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
    )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale,
    )

    prefill_compressor_ratio128(
        x_normed, cmp_kv_state, cmp_score_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        freqs_cos, freqs_sin, cmp_kv,
        position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    prefill_sparse_attn(
        q, kv_cache, ori_block_table, kv,
        cmp_kv, cmp_block_table,
        cmp_sparse_indices, cmp_sparse_lens,
        attn_sink, num_tokens,
        rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )
    # Commit new tokens to the cache AFTER sparse_attn reads the pre-update
    # history (the current tokens reach attention via the `kv` overlay).
    kv_cache_flat = pl.reshape(kv_cache, [HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cache_writeback"):
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    # Fold a zeroed attn_out read into the first 32B of the
                    # cache row so writeback is ordered after sparse_attn.
                    write_guard = pl.cast(attn_out[write_t : write_t + 1, 0:WRITEBACK_GUARD_TILE], target_type=pl.FP32)
                    write_zero = pl.mul(write_guard, 0.0)
                    write_kv_head = pl.cast(kv[write_t : write_t + 1, 0:WRITEBACK_GUARD_TILE], target_type=pl.FP32)
                    kv_cache_flat[write_row : write_row + 1, 0:WRITEBACK_GUARD_TILE] = pl.cast(
                        pl.add(write_kv_head, write_zero), target_type=pl.BF16)
                    kv_cache_flat[write_row : write_row + 1, WRITEBACK_GUARD_TILE:HEAD_DIM] = kv[
                        write_t : write_t + 1, WRITEBACK_GUARD_TILE:HEAD_DIM]

    hc_post(attn_out, x_hc, post, comb, x_out)
    return x_out

