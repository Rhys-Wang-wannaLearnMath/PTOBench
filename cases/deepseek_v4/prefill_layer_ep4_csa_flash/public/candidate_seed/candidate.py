# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2 # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 packed (request-aware) chunked prefill single layer with MoE EP4.

This is the Qwen-style packed-prefill variant of ``prefill_layer.py``. The packed
``prefill_layer_core`` expands the batch and sequence dimensions internally: it
loops over requests and over fixed-``T`` token tiles, builds tile-local ``[T, ...]``
inputs from the packed buffers, and directly calls the existing fixed-``T`` child
kernels (``prefill_attention_{swa,hca,csa}`` and ``moe``) per tile, scattering valid
rows back into the packed output.

Coordinate system (shared by JIT and golden, see issue #591):
  * packed token buffers : global packed row (``chunk_offsets[r] + tile_id*T + t``)
  * cache/state/tables   : request-local (each request owns a contiguous slice)
  * sparse-index overlay : tile-local (``WIN + t`` for the current tile's tokens)
  * position_ids         : absolute position (``chunk_start + tile_id*T + t``)

All tile-local metadata (slot mappings, sparse indices, positions) is produced on
the host by reusing the child ``build_*_tensor_specs(start_pos=tile_ctx,
num_tokens=valid)`` builders, which already encode the absolute-position ring /
overlay / compressed / state formulas for a single ``[T]`` tile. The kernel just
gathers the precomputed packed metadata per tile.
"""

import pypto.language as pl
import pypto.language.distributed as pld

# The prefill path routes PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe (which freezes recv shapes and derives RECV_MAX = EP * MOE_TOKENS at import).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS
config.PREFILL_LAYER_STATIC_CHILDREN = True
# Import moe first so dependent modules freeze the EP4 shapes from CASE_CONFIG.
# modules bake config-derived MoE shapes.
from moe import (
    AUX_PAD,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    T,
    TOPK,
    VOCAB,
    moe,
)
from config import CASE_CONFIG as MODEL_CONFIG
from prefill_attention_swa import (
    BLOCK_NUM as SWA_ORI_BLOCK_NUM,
    BLOCK_SIZE as SWA_BLOCK_SIZE,
    prefill_attention_swa,
)
from prefill_attention_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    HCA_CMP_BLOCK_NUM,
    HCA_ORI_BLOCK_NUM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    prefill_attention_hca,
)
from prefill_attention_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    PREFILL_IDX_BLOCK_NUM,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_TOPK,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    prefill_attention_csa,
)
assert SWA_BLOCK_SIZE == BLOCK_SIZE, "SWA/HCA/CSA must share the PyPTO block size"
assert SWA_ORI_BLOCK_NUM == HCA_ORI_BLOCK_NUM == CSA_ORI_BLOCK_NUM

# ``T`` is the fixed child-kernel token-tile capacity (Qwen's ``TOK_TILE``). It is
# NOT the packed token total. The packed prefill only ever feeds the children a
# fixed ``[T, ...]`` tile at a time.
TOK_TILE = T
PREFILL_CHUNK_TOKENS = T
DEFAULT_CHUNK_LENS = (T, T + T // 2)
DEFAULT_USER_BATCH = len(DEFAULT_CHUNK_LENS)

# Per-request contiguous block/state counts (each request owns one such slice; the
# packed buffer dim0 is ``user_batch * <count>``). The cache/state block tables are
# request-local so the children see block ids starting at 0 inside their slice.
ORI_CACHE_BLOCKS = CSA_ORI_BLOCK_NUM
# HCA in sources_target still exposes the compact ratio-128 cache, while the
# newer CSA uses the packed/config-sized ratio-4 cache. The packed layer allocates
# one shared cmp_kv buffer, so use the larger per-request capacity; active child
# specs still drive initialization and golden per layer.
CMP_CACHE_BLOCKS = max(HCA_CMP_BLOCK_NUM, CSA_CMP_BLOCK_NUM)
IDX_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
ORI_TABLE_BLOCKS = SPARSE_ORI_MAX_BLOCKS
CMP_TABLE_BLOCKS = SPARSE_CMP_MAX_BLOCKS
IDX_TABLE_BLOCKS = IDX_CACHE_MAX_BLOCKS

# Dynamic (batch-dependent) kernel-signature dims. Kept local to this file so
# config.py and the fixed-T child kernels stay untouched (issue #591 §1).
USER_BATCH_DYN = pl.dynamic("DEEPSEEK_PREFILL_USER_BATCH_DYN")
PREFILL_TOKENS_DYN = pl.dynamic("DEEPSEEK_PREFILL_TOKENS_DYN")

PREFILL_ORI_CACHE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_ORI_CACHE_BLOCKS_DYN")
PREFILL_CMP_CACHE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_CMP_CACHE_BLOCKS_DYN")
PREFILL_IDX_CACHE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_IDX_CACHE_BLOCKS_DYN")

PREFILL_ORI_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_ORI_BLOCK_TABLE_DYN")
PREFILL_CMP_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_CMP_BLOCK_TABLE_DYN")
PREFILL_IDX_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_IDX_BLOCK_TABLE_DYN")

PREFILL_HCA_STATE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_HCA_STATE_BLOCKS_DYN")
PREFILL_CSA_STATE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_CSA_STATE_BLOCKS_DYN")
PREFILL_INNER_STATE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_INNER_STATE_BLOCKS_DYN")


@pl.jit
def prefill_layer_core(
    x_hc: pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_tile_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
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
    hca_cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.InOut[pl.Tensor[
        [PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_cmp_score_state: pl.InOut[pl.Tensor[
        [PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[[PREFILL_HCA_STATE_BLOCKS_DYN], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.InOut[
        pl.Tensor[[PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_cmp_score_state: pl.InOut[
        pl.Tensor[[PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[PREFILL_CSA_STATE_BLOCKS_DYN], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.InOut[
        pl.Tensor[[PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_score_state: pl.InOut[
        pl.Tensor[[PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[PREFILL_INNER_STATE_BLOCKS_DYN], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[PREFILL_ORI_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[PREFILL_ORI_BLOCK_TABLE_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[PREFILL_CMP_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_BLOCK_TABLE_DYN], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[PREFILL_TOKENS_DYN, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[PREFILL_IDX_BLOCK_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16]:
    x_hc.bind_dynamic(0, PREFILL_TOKENS_DYN)
    ori_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    cmp_sparse_indices.bind_dynamic(0, PREFILL_TOKENS_DYN)
    cmp_sparse_lens.bind_dynamic(0, PREFILL_TOKENS_DYN)
    position_ids.bind_dynamic(0, PREFILL_TOKENS_DYN)
    hca_cmp_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    hca_state_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    csa_cmp_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    csa_idx_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    csa_state_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    csa_inner_state_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    input_ids.bind_dynamic(0, PREFILL_TOKENS_DYN)
    x_next.bind_dynamic(0, PREFILL_TOKENS_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    chunk_lens.bind_dynamic(0, USER_BATCH_DYN)
    chunk_offsets.bind_dynamic(0, USER_BATCH_DYN)
    chunk_tile_offsets.bind_dynamic(0, USER_BATCH_DYN)
    kv_cache.bind_dynamic(0, PREFILL_ORI_CACHE_BLOCKS_DYN)
    ori_block_table.bind_dynamic(0, PREFILL_ORI_BLOCK_TABLE_DYN)
    cmp_kv.bind_dynamic(0, PREFILL_CMP_CACHE_BLOCKS_DYN)
    cmp_block_table.bind_dynamic(0, PREFILL_CMP_BLOCK_TABLE_DYN)
    idx_kv_cache.bind_dynamic(0, PREFILL_IDX_CACHE_BLOCKS_DYN)
    idx_kv_scale.bind_dynamic(0, PREFILL_IDX_CACHE_BLOCKS_DYN)
    idx_block_table.bind_dynamic(0, PREFILL_IDX_BLOCK_TABLE_DYN)
    hca_cmp_kv_state.bind_dynamic(0, PREFILL_HCA_STATE_BLOCKS_DYN)
    hca_cmp_score_state.bind_dynamic(0, PREFILL_HCA_STATE_BLOCKS_DYN)
    hca_compress_state_block_table.bind_dynamic(0, PREFILL_HCA_STATE_BLOCKS_DYN)
    csa_cmp_kv_state.bind_dynamic(0, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_cmp_score_state.bind_dynamic(0, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_compress_state_block_table.bind_dynamic(0, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_inner_kv_state.bind_dynamic(0, PREFILL_INNER_STATE_BLOCKS_DYN)
    csa_inner_score_state.bind_dynamic(0, PREFILL_INNER_STATE_BLOCKS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(0, PREFILL_INNER_STATE_BLOCKS_DYN)
    user_batch = pl.tensor.dim(seq_lens, 0)
    for request_id in pl.range(user_batch):
        chunk_len_b = pl.tensor.read(chunk_lens, [request_id])
        chunk_base = pl.cast(pl.tensor.read(chunk_offsets, [request_id]), pl.INDEX)
        tile_ord_base = pl.cast(pl.tensor.read(chunk_tile_offsets, [request_id]), pl.INDEX)
        tok_blocks = (chunk_len_b + TOK_TILE - 1) // TOK_TILE
        ridx = pl.cast(request_id, pl.INDEX)

        # Request-local cache/state/table views (one slice per request; persist
        # across the request's tiles via in-place mutation by the children).
        kv_cache_req = pl.slice(kv_cache, [ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
                                [ridx * ORI_CACHE_BLOCKS, 0, 0, 0])
        ori_block_table_req = pl.slice(ori_block_table, [ORI_TABLE_BLOCKS], [ridx * ORI_TABLE_BLOCKS])
        cmp_kv_req = pl.slice(cmp_kv, [CMP_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
                              [ridx * CMP_CACHE_BLOCKS, 0, 0, 0])
        cmp_block_table_req = pl.slice(cmp_block_table, [CMP_TABLE_BLOCKS], [ridx * CMP_TABLE_BLOCKS])
        idx_kv_cache_req = pl.slice(idx_kv_cache, [IDX_CACHE_BLOCKS, BLOCK_SIZE, 1, IDX_HEAD_DIM],
                                    [ridx * IDX_CACHE_BLOCKS, 0, 0, 0])
        idx_kv_scale_req = pl.slice(idx_kv_scale, [IDX_CACHE_BLOCKS, BLOCK_SIZE, 1, 1],
                                    [ridx * IDX_CACHE_BLOCKS, 0, 0, 0])
        idx_block_table_req = pl.slice(idx_block_table, [IDX_TABLE_BLOCKS], [ridx * IDX_TABLE_BLOCKS])
        hca_kv_state_req = pl.slice(hca_cmp_kv_state, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
                                    [ridx * HCA_STATE_BLOCK_NUM, 0, 0])
        hca_score_state_req = pl.slice(hca_cmp_score_state, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
                                       [ridx * HCA_STATE_BLOCK_NUM, 0, 0])
        hca_state_table_req = pl.slice(hca_compress_state_block_table, [HCA_STATE_MAX_BLOCKS],
                                       [ridx * HCA_STATE_MAX_BLOCKS])
        csa_kv_state_req = pl.slice(csa_cmp_kv_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM],
                                    [ridx * CSA_STATE_BLOCK_NUM, 0, 0])
        csa_score_state_req = pl.slice(csa_cmp_score_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM],
                                       [ridx * CSA_STATE_BLOCK_NUM, 0, 0])
        csa_state_table_req = pl.slice(csa_compress_state_block_table, [CSA_STATE_MAX_BLOCKS],
                                       [ridx * CSA_STATE_MAX_BLOCKS])
        csa_inner_kv_state_req = pl.slice(csa_inner_kv_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM],
                                          [ridx * INNER_STATE_BLOCK_NUM, 0, 0])
        csa_inner_score_state_req = pl.slice(csa_inner_score_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM],
                                             [ridx * INNER_STATE_BLOCK_NUM, 0, 0])
        csa_inner_state_table_req = pl.slice(csa_inner_compress_state_block_table, [INNER_STATE_MAX_BLOCKS],
                                             [ridx * INNER_STATE_MAX_BLOCKS])

        for tile_id in pl.range(tok_blocks):
            p0 = tile_id * TOK_TILE
            tile_base = chunk_base + p0
            valid_tok = pl.min(TOK_TILE, chunk_len_b - p0)
            valid_n = pl.cast(valid_tok, pl.INT32)  # child num_tokens scalar (INT32)
            # Global execution ordinal of this MoE call (1-based). The done
            # windows are monotonic counters, so each serial MoE call needs a
            # unique, gap-free epoch == its execution order; chunk_tile_offsets
            # is the exclusive prefix sum of tok_blocks over requests.
            moe_epoch = pl.cast(tile_ord_base + tile_id + 1, pl.INT32)

            # Tile-local fixed-[T] inputs gathered from the packed buffers. The
            # children only read the leading ``valid_tok`` rows, so the padded
            # tail (when valid_tok < T) is ignored. No explicit per-tile pl.scope:
            # rely on auto_scope + pl.range's sequential semantics so the
            # request-local cache/state RAW dependency is carried across tiles
            # (tile N's writeback ordered before tile N+1's gather).
            x_hc_tile = pl.slice(x_hc, [TOK_TILE, HC_MULT, D], [tile_base, 0, 0])
            ori_slot_tile = pl.slice(ori_slot_mapping, [TOK_TILE], [tile_base])
            cmp_sparse_indices_tile = pl.slice(cmp_sparse_indices, [TOK_TILE, SPARSE_TOPK], [tile_base, 0])
            cmp_sparse_lens_tile = pl.slice(cmp_sparse_lens, [TOK_TILE], [tile_base])
            position_ids_tile = pl.slice(position_ids, [TOK_TILE], [tile_base])
            hca_cmp_slot_tile = pl.slice(hca_cmp_slot_mapping, [TOK_TILE], [tile_base])
            hca_state_slot_tile = pl.slice(hca_state_slot_mapping, [TOK_TILE], [tile_base])
            csa_cmp_slot_tile = pl.slice(csa_cmp_slot_mapping, [TOK_TILE], [tile_base])
            csa_idx_slot_tile = pl.slice(csa_idx_slot_mapping, [TOK_TILE], [tile_base])
            csa_state_slot_tile = pl.slice(csa_state_slot_mapping, [TOK_TILE], [tile_base])
            csa_inner_state_slot_tile = pl.slice(csa_inner_state_slot_mapping, [TOK_TILE], [tile_base])
            input_ids_tile = pl.slice(input_ids, [TOK_TILE], [tile_base])

            x_attn_tile = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.BF16)
            if layer_id < 2:
                prefill_attention_swa(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    kv_cache_req, ori_block_table_req, ori_slot_tile,
                    cmp_sparse_indices_tile, cmp_sparse_lens_tile, position_ids_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )
            elif layer_id % 2 == 1:
                prefill_attention_hca(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
                    hca_kv_state_req, hca_score_state_req, hca_state_table_req,
                    kv_cache_req, ori_slot_tile, ori_block_table_req,
                    cmp_kv_req, cmp_block_table_req, cmp_sparse_indices_tile, cmp_sparse_lens_tile,
                    position_ids_tile, hca_cmp_slot_tile, hca_state_slot_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )
            else:
                prefill_attention_csa(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
                    csa_kv_state_req, csa_score_state_req, csa_state_table_req,
                    csa_hadamard_idx,
                    csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj,
                    csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
                    csa_inner_kv_state_req, csa_inner_score_state_req, csa_inner_state_table_req,
                    kv_cache_req, ori_block_table_req, ori_slot_tile,
                    cmp_kv_req, cmp_block_table_req, idx_kv_cache_req, idx_kv_scale_req, idx_block_table_req,
                    position_ids_tile, csa_cmp_slot_tile, csa_idx_slot_tile,
                    csa_state_slot_tile, csa_inner_state_slot_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )

            x_next_tile = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.BF16)
            moe(
                x_attn_tile,
                hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                norm_w, gate_w, gate_bias, tid2eid, input_ids_tile,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                x_next_tile,
                recv_meta, recv_x, recv_aux, recv_route, arrived,
                routed_y_buf, combine_arrived,
                layer_id, valid_n, my_rank, moe_epoch,
            )

            # Scatter the tile back into the padded physical output. Each
            # request's physical span is rounded up to whole-T tiles, so a
            # full-T write is safe even for a partial logical tail tile.
            x_next = pl.assemble(x_next, x_next_tile, [tile_base, 0, 0])
    # Adapt official 7a57215c (2026-08-14): persistent windows outlive
    # dispatches, while each dispatch restarts its MoE epochs at one.
    # FLASH dispatch/combine each deposit one credit per remote source
    # per epoch; this implementation has no data_arrived window.
    last_req = user_batch - 1
    last_tiles = (pl.tensor.read(chunk_lens, [last_req]) + TOK_TILE - 1) // TOK_TILE
    total_epochs = pl.cast(pl.tensor.read(chunk_tile_offsets, [last_req]), pl.INDEX) + pl.cast(last_tiles, pl.INDEX)
    neg_epochs = pl.cast(0 - total_epochs, pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_retire"):
        _x_next_anchor = pl.read(x_next, [0, 0, 0])
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.notify(target=arrived, peer=my_rank, offsets=[src, 0], value=neg_epochs, op=pld.NotifyOp.AtomicAdd)
                pld.system.notify(target=combine_arrived, peer=my_rank, offsets=[src, 0], value=neg_epochs, op=pld.NotifyOp.AtomicAdd)
    return x_next


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16],
    seq_lens: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    chunk_tile_offsets: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.InOut[pl.Tensor[
        [N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_cmp_score_state: pl.InOut[pl.Tensor[
        [N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_cmp_score_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_score_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_ORI_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, PREFILL_ORI_BLOCK_TABLE_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_CMP_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, PREFILL_CMP_BLOCK_TABLE_DYN], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[N_RANKS, PREFILL_IDX_BLOCK_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16]],
    layer_id: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer(N_RANKS * N_LOCAL * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_aux_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * AUX_PAD * 4)
    recv_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        prefill_layer_core(
            x_hc[rank],
            seq_lens[rank], chunk_lens[rank], chunk_offsets[rank], chunk_tile_offsets[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank], hca_cmp_norm_w[rank],
            hca_cmp_kv_state[rank], hca_cmp_score_state[rank], hca_compress_state_block_table[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank],
            csa_cmp_kv_state[rank], csa_cmp_score_state[rank], csa_compress_state_block_table[rank],
            csa_hadamard_idx[rank],
            csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank], csa_weights_proj[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_kv_state[rank], csa_inner_score_state[rank],
            csa_inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping[rank],
            cmp_kv[rank], cmp_block_table[rank],
            cmp_sparse_indices[rank], cmp_sparse_lens[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            position_ids[rank],
            hca_cmp_slot_mapping[rank], hca_state_slot_mapping[rank],
            csa_cmp_slot_mapping[rank], csa_idx_slot_mapping[rank],
            csa_state_slot_mapping[rank], csa_inner_state_slot_mapping[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_next[rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived,
            routed_y_buf, combine_arrived,
            layer_id, rank,
            device=rank,
        )
