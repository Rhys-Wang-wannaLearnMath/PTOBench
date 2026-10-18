# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Top-k candidate selection for Qwen3-14B logits."""

from __future__ import annotations

import pypto.language as pl

from config import CASE_CONFIG as M
from config import FP32_NEG_INF, TOPK


BATCH = M.batch
VOCAB = M.vocab
REAL_VOCAB = M.real_vocab
# This is an implementation tiling choice retained from the selected source;
# fixed workload dimensions come from the public Case contract.
VOCAB_CHUNK = 512
NUM_VOCAB_CHUNKS = VOCAB // VOCAB_CHUNK
REAL_NUM_FULL_VOCAB_CHUNKS = REAL_VOCAB // VOCAB_CHUNK
REAL_VOCAB_TAIL = REAL_VOCAB % VOCAB_CHUNK
REAL_NUM_VOCAB_CHUNKS = REAL_NUM_FULL_VOCAB_CHUNKS + (1 if REAL_VOCAB_TAIL != 0 else 0)
TOPK_GROUP_WIDTH = 2048
TOPK_NUM_FULL_GROUPS = REAL_VOCAB // TOPK_GROUP_WIDTH
TOPK_GROUP_TAIL = REAL_VOCAB % TOPK_GROUP_WIDTH
TOPK_NUM_GROUPS = TOPK_NUM_FULL_GROUPS + (1 if TOPK_GROUP_TAIL != 0 else 0)
TOPK_CANDIDATE_PAD = 4096
TOPK_FINAL_HALF = TOPK_CANDIDATE_PAD // 2
GREEDY_CHUNK_PAD = VOCAB_CHUNK
GREEDY_SORT_TOPK = 16

assert VOCAB % VOCAB_CHUNK == 0
assert REAL_VOCAB <= VOCAB
assert TOPK <= VOCAB_CHUNK
assert VOCAB_CHUNK == 512
assert TOPK_GROUP_WIDTH % VOCAB_CHUNK == 0
assert TOPK_GROUP_TAIL == REAL_VOCAB_TAIL
assert TOPK_NUM_GROUPS * TOPK <= TOPK_CANDIDATE_PAD
assert NUM_VOCAB_CHUNKS <= GREEDY_CHUNK_PAD


@pl.jit.inline
def _topk_group_pairs(
    logits: pl.Tensor[[BATCH, VOCAB], pl.FP32],
    batch_idx: pl.Scalar[pl.INDEX],
    group_idx: pl.Scalar[pl.INDEX],
):
    group_start = group_idx * TOPK_GROUP_WIDTH
    scores = logits[batch_idx : batch_idx + 1, group_start : group_start + TOPK_GROUP_WIDTH]
    indices = pl.arange(
        pl.cast(group_start, target_type=pl.UINT32),
        [1, TOPK_GROUP_WIDTH],
        dtype=pl.UINT32,
    )
    pairs = pl.sort32(scores, indices)
    pairs = pl.mrgsort(pairs, block_len=64)
    pairs = pl.mrgsort(pairs, block_len=256)
    pairs = pl.mrgsort(pairs, block_len=1024)
    return pairs[:, 0 : 2 * TOPK]


@pl.jit
def topk_select_fwd(
    logits: pl.Tensor[[BATCH, VOCAB], pl.FP32],
    sampling_control: pl.Tensor[[2], pl.INT32],
    topk_values: pl.Out[pl.Tensor[[BATCH, TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
):
    """Return greedy top-1 or the largest TOPK real-vocab candidates per row."""
    for b in pl.parallel(BATCH):
        if b < pl.read(sampling_control, [0]):
            if pl.read(sampling_control, [1]) == 1:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="greedy_select"):
                    idx_init = pl.arange(0, [1, VOCAB_CHUNK], dtype=pl.UINT32)
                    chunk_vals = pl.create_tensor([1, GREEDY_CHUNK_PAD], dtype=pl.FP32)
                    chunk_vals[:, :] = pl.full([1, GREEDY_CHUNK_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
                    for c in pl.range(REAL_NUM_VOCAB_CHUNKS):
                        c0 = c * VOCAB_CHUNK
                        local_scores = logits[b : b + 1, c0 : c0 + VOCAB_CHUNK]
                        if REAL_VOCAB_TAIL != 0:
                            if c == REAL_NUM_FULL_VOCAB_CHUNKS:
                                local_scores_valid = pl.set_validshape(local_scores, 1, REAL_VOCAB_TAIL)
                                local_scores_padded = pl.fillpad(
                                    local_scores_valid, pad_value=pl.PadValue.min
                                )
                                sorted_pairs = pl.sort32(local_scores_padded, idx_init)
                            else:
                                sorted_pairs = pl.sort32(local_scores, idx_init)
                        else:
                            sorted_pairs = pl.sort32(local_scores, idx_init)
                        sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
                        sorted_pairs = pl.mrgsort(sorted_pairs, block_len=256)
                        top_pairs = sorted_pairs[:, 0 : 2 * GREEDY_SORT_TOPK]
                        top_vals = pl.gather(top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                        pl.write(chunk_vals, [0, c], pl.read(top_vals, [0, 0]))

                    chunk_sorted = pl.sort32(chunk_vals, idx_init)
                    chunk_sorted = pl.mrgsort(chunk_sorted, block_len=64)
                    chunk_sorted = pl.mrgsort(chunk_sorted, block_len=256)
                    chunk_top_pairs = chunk_sorted[:, 0 : 2 * GREEDY_SORT_TOPK]
                    chunk_top_vals = pl.gather(chunk_top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                    best_val = pl.read(chunk_top_vals, [0, 0])
                    chunk_i32 = pl.cast(0, pl.INT32)
                    for c in pl.range(REAL_NUM_VOCAB_CHUNKS):
                        scan_c = (REAL_NUM_VOCAB_CHUNKS - 1) - c
                        if pl.read(chunk_vals, [0, scan_c]) == best_val:
                            chunk_i32 = pl.cast(scan_c, pl.INT32)

                    local_token = pl.cast(0, pl.INT32)
                    chunk_base = chunk_i32 * pl.cast(VOCAB_CHUNK, target_type=pl.INT32)
                    winning_logits = pl.slice(
                        logits,
                        [1, VOCAB_CHUNK],
                        [pl.cast(b, pl.INDEX), pl.cast(chunk_base, target_type=pl.INDEX)],
                    )
                    if REAL_VOCAB_TAIL != 0:
                        if chunk_i32 == pl.cast(REAL_NUM_FULL_VOCAB_CHUNKS, target_type=pl.INT32):
                            winning_logits_valid = pl.set_validshape(winning_logits, 1, REAL_VOCAB_TAIL)
                            winning_logits_padded = pl.fillpad(
                                winning_logits_valid, pad_value=pl.PadValue.min
                            )
                            for t in pl.range(VOCAB_CHUNK):
                                scan_t = (VOCAB_CHUNK - 1) - t
                                if (
                                    pl.read(
                                        winning_logits_padded,
                                        [0, pl.cast(scan_t, pl.INDEX)],
                                    )
                                    == best_val
                                ):
                                    local_token = pl.cast(scan_t, pl.INT32)
                        else:
                            for t in pl.range(VOCAB_CHUNK):
                                scan_t = (VOCAB_CHUNK - 1) - t
                                if pl.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)]) == best_val:
                                    local_token = pl.cast(scan_t, pl.INT32)
                    else:
                        for t in pl.range(VOCAB_CHUNK):
                            scan_t = (VOCAB_CHUNK - 1) - t
                            if pl.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)]) == best_val:
                                local_token = pl.cast(scan_t, pl.INT32)

                    token_id = chunk_base + local_token
                    if token_id >= pl.cast(REAL_VOCAB, target_type=pl.INT32):
                        token_id = pl.cast(0, pl.INT32)
                    # Keep output GM writes tile-only. Mixing a slice store with
                    # scalar GM writes is rejected by the current MTE3 lowering.
                    greedy_values = pl.full([1, TOPK], dtype=pl.FP32, value=FP32_NEG_INF)
                    greedy_indices = pl.full([1, TOPK], dtype=pl.INT32, value=0)
                    pl.write(greedy_values, [0, 0], best_val)
                    pl.write(greedy_indices, [0, 0], token_id)
                    topk_values[b : b + 1, :] = greedy_values
                    topk_indices[b : b + 1, :] = greedy_indices
            else:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="topk_select"):
                    candidate_vals = pl.create_tensor([1, TOPK_CANDIDATE_PAD], dtype=pl.FP32)
                    candidate_vals[:, :] = pl.full(
                        [1, TOPK_CANDIDATE_PAD], dtype=pl.FP32, value=FP32_NEG_INF
                    )
                    candidate_ids = pl.create_tensor([1, TOPK_CANDIDATE_PAD], dtype=pl.INT32)
                    candidate_ids[:, :] = pl.full([1, TOPK_CANDIDATE_PAD], dtype=pl.INT32, value=0)

                    for g in pl.range(TOPK_NUM_FULL_GROUPS):
                        group_pairs = _topk_group_pairs(logits, b, g)
                        group_vals = pl.gather(group_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                        group_ids = pl.gather(
                            group_pairs,
                            mask_pattern=pl.tile.MaskPattern.P1010,
                            output_dtype=pl.INT32,
                        )
                        candidate_offset = g * TOPK
                        for k in pl.range(TOPK):
                            pl.write(
                                candidate_vals,
                                [0, candidate_offset + k],
                                pl.read(group_vals, [0, k]),
                            )
                            pl.write(
                                candidate_ids,
                                [0, candidate_offset + k],
                                pl.read(group_ids, [0, k]),
                            )

                    if TOPK_GROUP_TAIL != 0:
                        tail_start = TOPK_NUM_FULL_GROUPS * TOPK_GROUP_WIDTH
                        tail_scores_raw = logits[
                            b : b + 1, tail_start : tail_start + VOCAB_CHUNK
                        ]
                        tail_scores = pl.fillpad(
                            pl.set_validshape(tail_scores_raw, 1, TOPK_GROUP_TAIL),
                            pad_value=pl.PadValue.min,
                        )
                        tail_indices = pl.arange(
                            pl.cast(tail_start, target_type=pl.UINT32),
                            [1, VOCAB_CHUNK],
                            dtype=pl.UINT32,
                        )
                        tail_pairs = pl.sort32(tail_scores, tail_indices)
                        tail_pairs = pl.mrgsort(tail_pairs, block_len=64)
                        tail_pairs = pl.mrgsort(tail_pairs, block_len=256)
                        tail_pairs = tail_pairs[:, 0 : 2 * TOPK]
                        tail_vals = pl.gather(tail_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                        tail_ids = pl.gather(
                            tail_pairs,
                            mask_pattern=pl.tile.MaskPattern.P1010,
                            output_dtype=pl.INT32,
                        )
                        tail_offset = TOPK_NUM_FULL_GROUPS * TOPK
                        for k in pl.range(TOPK):
                            pl.write(
                                candidate_vals,
                                [0, tail_offset + k],
                                pl.read(tail_vals, [0, k]),
                            )
                            pl.write(
                                candidate_ids,
                                [0, tail_offset + k],
                                pl.read(tail_ids, [0, k]),
                            )

                    candidate_positions = pl.arange(0, [1, TOPK_CANDIDATE_PAD], dtype=pl.UINT32)
                    candidate_sorted = pl.sort32(candidate_vals, candidate_positions)
                    candidate_sorted = pl.mrgsort(candidate_sorted, block_len=64)
                    candidate_sorted = pl.mrgsort(candidate_sorted, block_len=256)
                    candidate_sorted = pl.mrgsort(candidate_sorted, block_len=1024)
                    half0_pairs = candidate_sorted[:, 0 : 2 * TOPK]
                    half1_pairs = candidate_sorted[
                        :, 2 * TOPK_FINAL_HALF : 2 * TOPK_FINAL_HALF + 2 * TOPK
                    ]
                    candidate_pairs = pl.mrgsort(half0_pairs, half1_pairs)[:, 0 : 2 * TOPK]
                    topk_values[b : b + 1, :] = pl.gather(
                        candidate_pairs,
                        mask_pattern=pl.tile.MaskPattern.P0101,
                    )
                    selected_positions = pl.gather(
                        candidate_pairs,
                        mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    for k in pl.range(TOPK):
                        candidate_pos = pl.read(selected_positions, [0, k])
                        token_id = pl.read(
                            candidate_ids,
                            [0, pl.cast(candidate_pos, target_type=pl.INDEX)],
                        )
                        pl.write(topk_indices, [b, k], token_id)
        else:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="topk_select_inactive"):
                topk_values[b : b + 1, :] = pl.full([1, TOPK], dtype=pl.FP32, value=FP32_NEG_INF)
                topk_indices[b : b + 1, :] = pl.full([1, TOPK], dtype=pl.INT32, value=0)

    return topk_values, topk_indices
