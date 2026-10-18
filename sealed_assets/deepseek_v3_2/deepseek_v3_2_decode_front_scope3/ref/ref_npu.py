# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer decode FRONT - Scope 3: INT8 indexer score + topk.

Pipeline: scope1 (qkv proj/RoPE) -> scope2 (indexer INT8 quant + k_cache_idx_i8
update) -> scope3 (this file: INT8 score then topk).

Scoring matches the current fused scope123 path:
    logits_i32[h, s] = q_idx_full_i8[b, h] dot k_cache_idx_i8[b, s]
    score[b, s] = sum_h relu(logits_i32[h, s]) * weights[b, h] * q_scale[b, h] * k_scale[b, s]

Topk is done by sort32 + 4-way mrgsort merge, then gather to split sorted
(val, idx) pairs, then GM reload with valid_shape+fillpad to mark idx slots
past ctx_len. Outputs (topk_vals_out, topk_idx_out) = top-INDEX_TOPK entries
per batch; invalid tail idx = INT32_MIN (< 0), compatible with scope4's
`topk_pos >= 0` filter.
"""
import pypto.language as pl

from config import (
    BATCH,
    CACHE_ROWS_IDX,
    FP32_NEG_INF,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_Q_ROWS,
    INDEX_TOPK,
    MAX_SEQ,
    MAX_SEQ_BLOCKS,
    Q_PAD,
    Q_VALID,
    SEQ_TILE,
    SORT_LEN,
)


def build_deepseek_v3_2_decode_front_scope3_program():
    @pl.program
    class DeepSeekV32DecodeFrontScope3:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope3(
            self,
            q_idx_full_i8: pl.Tensor[[INDEX_Q_ROWS, INDEX_HEAD_DIM], pl.INT8],
            q_idx_scale_heads: pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32],
            weights: pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32],
            k_cache_idx_i8: pl.Tensor[[CACHE_ROWS_IDX, INDEX_HEAD_DIM], pl.INT8],
            k_cache_idx_scale: pl.Tensor[[BATCH, MAX_SEQ], pl.FP32],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            topk_vals_out: pl.Out[pl.Tensor[[BATCH, INDEX_TOPK], pl.FP32]],
            topk_idx_out: pl.Out[pl.Tensor[[BATCH, INDEX_TOPK], pl.INT32]],
        ) -> tuple[
            pl.Tensor[[BATCH, INDEX_TOPK], pl.FP32],
            pl.Tensor[[BATCH, INDEX_TOPK], pl.INT32],
        ]:
            # Pad q_idx_full_i8 to Q_PAD rows per (batch, head) for INT8 qk validation.
            q_i8_padded = pl.create_tensor([BATCH * INDEX_HEADS * Q_PAD, INDEX_HEAD_DIM], dtype=pl.INT8)
            q_s_padded = pl.create_tensor([BATCH * Q_PAD, INDEX_HEADS], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="init_q_pad"):
                for b in pl.range(BATCH):
                    for h in pl.range(INDEX_HEADS):
                        q_row0 = b * INDEX_HEADS + h
                        q_i8_valid = pl.slice(q_idx_full_i8, [1, INDEX_HEAD_DIM], [q_row0, 0])
                        q_i8_padded = pl.assemble(q_i8_padded, q_i8_valid, [q_row0 * Q_PAD, 0])
                        q_i8_zero_pad = pl.cast(
                            pl.full([Q_PAD - Q_VALID, INDEX_HEAD_DIM], dtype=pl.INT16, value=0),
                            target_type=pl.INT8,
                        )
                        q_i8_padded = pl.assemble(q_i8_padded, q_i8_zero_pad, [q_row0 * Q_PAD + Q_VALID, 0])

                for b in pl.range(BATCH):
                    weights_row = pl.slice(weights, [1, INDEX_HEADS], [b, 0])
                    q_scales_row = pl.slice(q_idx_scale_heads, [1, INDEX_HEADS], [b, 0])
                    q_s_row = pl.mul(weights_row, q_scales_row)
                    q_s_padded = pl.assemble(q_s_padded, q_s_row, [b * Q_PAD, 0])
                    q_s_padded = pl.assemble(
                        q_s_padded,
                        pl.full([Q_PAD - Q_VALID, INDEX_HEADS], dtype=pl.FP32, value=0.0),
                        [b * Q_PAD + Q_VALID, 0],
                    )

            # Transient GM buffers (shared across batches).
            all_scores_i8 = pl.create_tensor(
                [BATCH * MAX_SEQ_BLOCKS * INDEX_HEADS * Q_PAD, SEQ_TILE],
                dtype=pl.INT32,
            )
            relu_rows = pl.create_tensor([BATCH * MAX_SEQ_BLOCKS * INDEX_HEADS, SEQ_TILE], dtype=pl.FP32)
            weighted_scores = pl.create_tensor([BATCH * MAX_SEQ_BLOCKS * Q_PAD, SEQ_TILE], dtype=pl.FP32)
            score_tiles = pl.create_tensor([BATCH * MAX_SEQ_BLOCKS, SEQ_TILE], dtype=pl.FP32)

            for b in pl.parallel(0, BATCH, 1):
                ctx_len = pl.read(seq_lens, [b])
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

                # Stage 0: pre-fill scores[0, 0:SORT_LEN] with -inf.
                scores = pl.create_tensor([1, SORT_LEN], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_neg_inf"):
                    neg_inf_row = pl.full([1, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
                    scores = pl.assemble(scores, neg_inf_row, [0, 0])

                # Stage 1: Compute tiled INT8 qk logits between q_idx_full_i8 and k_cache_idx_i8.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="int8_qk_logits"):
                    for sb in pl.range(ctx_blocks):
                        s0 = sb * SEQ_TILE
                        cache_row0 = b * MAX_SEQ + s0
                        k_tile_i8 = pl.slice(k_cache_idx_i8, [SEQ_TILE, INDEX_HEAD_DIM], [cache_row0, 0])
                        for h in pl.range(INDEX_HEADS):
                            q_row0 = (b * INDEX_HEADS + h) * Q_PAD
                            tile_row0 = ((b * MAX_SEQ_BLOCKS + sb) * INDEX_HEADS + h) * Q_PAD
                            q_tile_i8 = pl.slice(q_i8_padded, [Q_PAD, INDEX_HEAD_DIM], [q_row0, 0])
                            logits_i32 = pl.matmul(q_tile_i8, k_tile_i8, b_trans=True, out_dtype=pl.INT32)
                            all_scores_i8 = pl.assemble(all_scores_i8, logits_i32, [tile_row0, 0])

                # Stage 2: Cast staged INT32 logits to FP32, then extract q row and apply ReLU.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="relu_logits"):
                    for sb in pl.range(ctx_blocks):
                        for h in pl.range(INDEX_HEADS):
                            tile_row0 = ((b * MAX_SEQ_BLOCKS + sb) * INDEX_HEADS + h) * Q_PAD
                            logits_row_i32 = pl.slice(all_scores_i8, [1, SEQ_TILE], [tile_row0, 0])
                            logits_row_f32 = pl.cast(logits_row_i32, target_type=pl.FP32, mode="none")
                            relu_logits = pl.maximum(logits_row_f32, pl.mul(logits_row_f32, 0.0))
                            relu_row0 = (b * MAX_SEQ_BLOCKS + sb) * INDEX_HEADS + h
                            relu_rows = pl.assemble(relu_rows, relu_logits, [relu_row0, 0])

                # Stage 3: Reduce per-head ReLU logits with weights * q_scale.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="weighted_reduce"):
                    for sb in pl.range(ctx_blocks):
                        q_s_tile = pl.slice(q_s_padded, [Q_PAD, INDEX_HEADS], [b * Q_PAD, 0])
                        relu_row0 = (b * MAX_SEQ_BLOCKS + sb) * INDEX_HEADS
                        relu_tile = pl.slice(relu_rows, [INDEX_HEADS, SEQ_TILE], [relu_row0, 0])
                        weighted_tile = pl.matmul(q_s_tile, relu_tile, out_dtype=pl.FP32)
                        weighted_row0 = (b * MAX_SEQ_BLOCKS + sb) * Q_PAD
                        weighted_scores = pl.assemble(weighted_scores, weighted_tile, [weighted_row0, 0])

                # Stage 4: Apply k scale and write valid score tiles to scores.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_scale_score"):
                    for sb in pl.range(ctx_blocks):
                        s0 = sb * SEQ_TILE
                        valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                        weighted_row0 = (b * MAX_SEQ_BLOCKS + sb) * Q_PAD
                        k_scale = pl.slice(k_cache_idx_scale, [1, SEQ_TILE], [b, s0])
                        score_acc = pl.slice(weighted_scores, [1, SEQ_TILE], [weighted_row0, 0])
                        score_tile = pl.mul(score_acc, k_scale)
                        score_row0 = b * MAX_SEQ_BLOCKS + sb
                        score_tiles = pl.assemble(score_tiles, score_tile, [score_row0, 0])
                        score_valid = pl.slice(
                            score_tiles,
                            [1, SEQ_TILE],
                            [score_row0, 0],
                            valid_shape=[1, valid_len],
                        )
                        scores = pl.assemble(scores, score_valid, [0, s0])

                # Stage 5: sort32 + 4 mrgsort
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sort_mrgsort"):
                    idx_init = pl.tensor.arange(0, [1, SORT_LEN], dtype=pl.UINT32)
                    sorted_t = pl.tensor.sort32(scores, idx_init)
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=64)
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=256)
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=1024)
                    sorted_gm = pl.tensor.mrgsort(sorted_t, block_len=4096)

                # Stage 6: extract topk vals and idx
                raw_idx_gm = pl.create_tensor([1, INDEX_TOPK], dtype=pl.INT32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_split_idx"):
                    topk_pairs = pl.slice(sorted_gm, [1, 2 * INDEX_TOPK], [0, 0])
                    topk_v = pl.tensor.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                    topk_i_raw = pl.tensor.gather(
                        topk_pairs,
                        mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    topk_vals_out = pl.assemble(topk_vals_out, topk_v, [b, 0])
                    raw_idx_gm = pl.assemble(raw_idx_gm, topk_i_raw, [0, 0])
                    valid_topk = pl.min(INDEX_TOPK, ctx_len)
                    idx_valid = pl.slice(
                        raw_idx_gm,
                        [1, INDEX_TOPK],
                        [0, 0],
                        valid_shape=[1, valid_topk],
                    )
                    idx_padded = pl.fillpad(idx_valid, pad_value=pl.PadValue.min)
                    topk_idx_out = pl.assemble(topk_idx_out, idx_padded, [b, 0])

            return topk_vals_out, topk_idx_out

    return DeepSeekV32DecodeFrontScope3
