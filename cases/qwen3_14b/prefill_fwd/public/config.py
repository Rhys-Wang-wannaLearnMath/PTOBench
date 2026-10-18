# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed evaluation contract for qwen3_14b.prefill_fwd.

The selected historical config is flat and has no named model presets.
Model capacity and the two-layer synthetic Fixture bindings remain distinct.
"""

import pypto.language as pl

# Dynamic dimensions used by the selected entry and helper signatures.
USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
KV_CACHE_ROWS_DYN = pl.dynamic("KV_CACHE_ROWS_DYN")
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")
ROPE_SEQ_DYN = pl.dynamic("ROPE_SEQ_DYN")
LAYER_DYN = pl.dynamic("LAYER_DYN")
LAYER_HIDDEN_ROWS_DYN = pl.dynamic("LAYER_HIDDEN_ROWS_DYN")
LAYER_INTER_ROWS_DYN = pl.dynamic("LAYER_INTER_ROWS_DYN")

# Selected historical model shape.
BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
INTERMEDIATE = 17408
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
VOCAB = 152064
NUM_LAYERS = 40

# Fixed numeric semantics.
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)
HALF_DIM = HEAD_DIM // 2

# Query grouping and external paged-cache layout.
Q_HEAD_BATCH = 5
SEQ_TILE = 128
BLOCK_SIZE = SEQ_TILE
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
assert Q_PER_KV == Q_HEAD_BATCH, "The selected workload has one Q group per KV head"
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS

# Fixed original builder workload; model capacity above is unchanged.
CASE_BATCH = BATCH
CASE_MAX_SEQ = 128
CASE_NUM_LAYERS = 2
CASE_USE_MAX_SEQ = False
CASE_CHUNK_START = 0
CASE_CHUNK_SIZE = 0
CASE_MAX_BLOCKS_PER_SEQ = (CASE_MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
CASE_NUM_BLOCKS = CASE_BATCH * CASE_MAX_BLOCKS_PER_SEQ
CASE_LAYER_CACHE_ROWS = CASE_NUM_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE
CASE_CACHE_ROWS = CASE_NUM_LAYERS * CASE_LAYER_CACHE_ROWS

# The official builder's TOK_TILE operand fixes the Fixture distribution.
# Runtime tiling belongs to the editable implementation, independently of this value.
FIXTURE_TOKEN_QUANTUM = 64
FIXTURE_N_BLOCKS = max(
    1, (CASE_MAX_SEQ + FIXTURE_TOKEN_QUANTUM - 1) // FIXTURE_TOKEN_QUANTUM
)
SEQ_LENS = tuple(
    min(((b % FIXTURE_N_BLOCKS) + 1) * FIXTURE_TOKEN_QUANTUM, CASE_MAX_SEQ)
    for b in range(CASE_BATCH)
)
CHUNK_LENS = SEQ_LENS
CHUNK_OFFSETS = tuple(sum(CHUNK_LENS[:b]) for b in range(CASE_BATCH))
TOTAL_TOKENS = sum(CHUNK_LENS)

FIXTURE_BUILDER_ARGS = (
    CASE_BATCH, CASE_MAX_SEQ, HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
    INTERMEDIATE, CASE_NUM_LAYERS, VOCAB, CASE_USE_MAX_SEQ,
    CASE_CHUNK_START, CASE_CHUNK_SIZE,
)
