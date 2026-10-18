# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Certified NPU implementation for the Qwen3-14B decode Scope 2 Case."""

import pypto.language as pl

from config import (
    BATCH,
    BLOCK_SIZE,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    MAX_SEQ,
    NUM_HEADS,
    NUM_KV_HEADS,
    Q_HEAD_BATCH,
    Q_HEAD_PAD,
)


BATCH_TILE = 16


def build_qwen3_scope2_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    q_per_kv = num_heads // num_kv_heads
    half_dim = head_dim // 2
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)
    max_ctx_blocks = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks_per_seq = max_ctx_blocks
    num_blocks = batch * max_blocks_per_seq

    @pl.program
    class Qwen3Scope2:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_scope2(
            self,
            q_proj: pl.Tensor[[batch, hidden], pl.FP32],
            k_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            v_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            q_norm_weight: pl.Tensor[[1, head_dim], pl.FP32],
            k_norm_weight: pl.Tensor[[1, head_dim], pl.FP32],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            block_table: pl.Tensor[[batch * max_blocks_per_seq], pl.INT32],
            slot_mapping: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[num_blocks * num_kv_heads * BLOCK_SIZE, head_dim], pl.BF16],
            v_cache: pl.Tensor[[num_blocks * num_kv_heads * BLOCK_SIZE, head_dim], pl.BF16],
            attn_out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            q_proj_norm = pl.create_tensor([batch, hidden], dtype=pl.FP32)
            k_proj_norm = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)
            v_proj_bf16 = pl.create_tensor([batch, kv_hidden], dtype=pl.BF16)

            for b0 in pl.range(0, batch, BATCH_TILE):
                with pl.at(level=pl.Level.CORE_GROUP):
                    for h in pl.range(num_heads):
                        q0 = h * head_dim
                        q_chunk = pl.slice(q_proj, [BATCH_TILE, head_dim], [b0, q0])
                        q_sq_sum = pl.row_sum(pl.mul(q_chunk, q_chunk))
                        q_inv_rms = pl.rsqrt(pl.add(pl.mul(q_sq_sum, HEAD_DIM_INV), EPS))
                        q_chunk_norm = pl.col_expand_mul(
                            pl.row_expand_mul(q_chunk, q_inv_rms),
                            q_norm_weight,
                        )
                        q_proj_norm = pl.assemble(q_proj_norm, q_chunk_norm, [b0, q0])

                    for h in pl.range(num_kv_heads):
                        k0 = h * head_dim
                        k_chunk = pl.slice(k_proj, [BATCH_TILE, head_dim], [b0, k0])
                        k_sq_sum = pl.row_sum(pl.mul(k_chunk, k_chunk))
                        k_inv_rms = pl.rsqrt(pl.add(pl.mul(k_sq_sum, HEAD_DIM_INV), EPS))
                        k_chunk_norm = pl.col_expand_mul(
                            pl.row_expand_mul(k_chunk, k_inv_rms),
                            k_norm_weight,
                        )
                        k_proj_norm = pl.assemble(k_proj_norm, k_chunk_norm, [b0, k0])
                        v_proj_bf16 = pl.assemble(
                            v_proj_bf16,
                            pl.cast(
                                pl.slice(v_proj, [BATCH_TILE, head_dim], [b0, k0]),
                                target_type=pl.BF16,
                            ),
                            [b0, k0],
                        )

            all_q_padded = pl.create_tensor([batch * total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP):
                for idx in pl.range(batch * total_q_groups):
                    all_q_padded = pl.assemble(
                        all_q_padded,
                        pl.cast(
                            pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0),
                            target_type=pl.BF16,
                        ),
                        [idx * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                    )

            for b in pl.range(batch):
                ctx_len = pl.tensor.read(seq_lens, [b])
                pos = ctx_len - 1
                block_table_base = b * max_blocks_per_seq
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                with pl.at(level=pl.Level.CORE_GROUP):
                    for ki in pl.range(0, num_kv_heads):
                        kv_col = ki * head_dim
                        q_base = ki * q_per_kv
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * head_dim
                            q_lo = pl.slice(q_proj_norm, [1, half_dim], [b, q_col])
                            q_hi = pl.slice(q_proj_norm, [1, half_dim], [b, q_col + half_dim])
                            rot_lo_bf16 = pl.cast(
                                pl.sub(
                                    pl.col_expand_mul(q_lo, cos_lo),
                                    pl.col_expand_mul(q_hi, sin_lo),
                                ),
                                target_type=pl.BF16,
                            )
                            rot_hi_bf16 = pl.cast(
                                pl.add(
                                    pl.col_expand_mul(q_hi, cos_hi),
                                    pl.col_expand_mul(q_lo, sin_hi),
                                ),
                                target_type=pl.BF16,
                            )
                            all_q_padded = pl.assemble(
                                all_q_padded,
                                rot_lo_bf16,
                                [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + qi, 0],
                            )
                            all_q_padded = pl.assemble(
                                all_q_padded,
                                rot_hi_bf16,
                                [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + qi, half_dim],
                            )

                attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                attn_row_padded = pl.create_tensor(
                    [1, total_q_groups * Q_HEAD_PAD * head_dim],
                    dtype=pl.BF16,
                )
                for gi in pl.range(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_padded = pl.slice(
                        all_q_padded,
                        [Q_HEAD_PAD, head_dim],
                        [b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0],
                    )
                    prefix_len = ctx_len - 1
                    prefix_blocks = (prefix_len + BLOCK_SIZE - 1) // BLOCK_SIZE
                    all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32)
                    all_exp_padded = pl.create_tensor(
                        [max_ctx_blocks * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.BF16
                    )
                    all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                    all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32)
                    all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        for sb in pl.parallel(prefix_blocks):
                            block_table_idx = block_table_base + sb
                            pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                            cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                            k_tile = pl.slice(k_cache, [BLOCK_SIZE, head_dim], [cache_row0, 0])
                            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP):
                        for sb in pl.parallel(prefix_blocks):
                            s0 = sb * BLOCK_SIZE
                            valid_len = pl.min(BLOCK_SIZE, prefix_len - s0)
                            scores_valid = pl.slice(
                                all_raw_scores,
                                [Q_HEAD_PAD, BLOCK_SIZE],
                                [sb * Q_HEAD_PAD, 0],
                                valid_shape=[Q_HEAD_PAD, valid_len],
                            )
                            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                            scores = pl.mul(scores_padded, attn_scale)
                            cur_mi = pl.row_max(scores)
                            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                            exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                            cur_li = pl.row_sum(exp_scores_fp32)
                            all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_PAD, 0])
                            all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP):
                        for sb in pl.parallel(prefix_blocks):
                            block_table_idx = block_table_base + sb
                            pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                            cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                            exp_tile = pl.slice(
                                all_exp_padded,
                                [Q_HEAD_PAD, BLOCK_SIZE],
                                [sb * Q_HEAD_PAD, 0],
                            )
                            v_tile = pl.slice(v_cache, [BLOCK_SIZE, head_dim], [cache_row0, 0])
                            oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP):
                        kv_col = kvh * head_dim
                        k_lo = pl.slice(k_proj_norm, [1, half_dim], [b, kv_col])
                        k_hi = pl.slice(
                            k_proj_norm, [1, half_dim], [b, kv_col + half_dim]
                        )
                        current_k_lo = pl.cast(
                            pl.sub(
                                pl.col_expand_mul(k_lo, cos_lo),
                                pl.col_expand_mul(k_hi, sin_lo),
                            ),
                            target_type=pl.BF16,
                        )
                        current_k_hi = pl.cast(
                            pl.add(
                                pl.col_expand_mul(k_hi, cos_hi),
                                pl.col_expand_mul(k_lo, sin_hi),
                            ),
                            target_type=pl.BF16,
                        )
                        current_k_lo_box = pl.cast(
                            pl.col_expand_add(
                                pl.full(
                                    [Q_HEAD_PAD, half_dim], dtype=pl.FP32, value=0.0
                                ),
                                pl.cast(current_k_lo, target_type=pl.FP32),
                            ),
                            target_type=pl.BF16,
                        )
                        current_k_hi_box = pl.cast(
                            pl.col_expand_add(
                                pl.full(
                                    [Q_HEAD_PAD, half_dim], dtype=pl.FP32, value=0.0
                                ),
                                pl.cast(current_k_hi, target_type=pl.FP32),
                            ),
                            target_type=pl.BF16,
                        )
                        q_score_lo = pl.slice(
                            q_padded, [Q_HEAD_PAD, half_dim], [0, 0]
                        )
                        q_score_hi = pl.slice(
                            q_padded, [Q_HEAD_PAD, half_dim], [0, half_dim]
                        )
                        mi = pl.row_max(
                            pl.mul(
                                pl.add(
                                    pl.matmul(
                                        q_score_lo,
                                        current_k_lo_box,
                                        b_trans=True,
                                        out_dtype=pl.FP32,
                                    ),
                                    pl.matmul(
                                        q_score_hi,
                                        current_k_hi_box,
                                        b_trans=True,
                                        out_dtype=pl.FP32,
                                    ),
                                ),
                                attn_scale,
                            )
                        )
                        li = pl.row_sum(
                            pl.full(
                                [Q_HEAD_PAD, Q_HEAD_PAD],
                                dtype=pl.FP32,
                                value=0.0625,
                            )
                        )
                        current_v = pl.slice(
                            v_proj_bf16, [1, head_dim], [b, kv_col]
                        )
                        oi = pl.col_expand_add(
                            pl.full(
                                [Q_HEAD_PAD, head_dim], dtype=pl.FP32, value=0.0
                            ),
                            pl.cast(current_v, target_type=pl.FP32),
                        )
                        for sb in pl.range(prefix_blocks):
                            oi_tmp_valid = pl.slice(all_oi_tmp, [Q_HEAD_PAD, head_dim], [sb * Q_HEAD_PAD, 0])
                            cur_mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                            cur_li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                            oi = pl.add(
                                pl.row_expand_mul(oi, alpha),
                                pl.row_expand_mul(oi_tmp_valid, beta),
                            )
                            mi = mi_new
                        ctx = pl.row_expand_div(oi, li)
                        ctx_flat_padded = pl.reshape(ctx, [1, Q_HEAD_PAD * head_dim])
                        ctx_flat_padded_bf16 = pl.cast(ctx_flat_padded, target_type=pl.BF16)
                        attn_row_padded = pl.assemble(
                            attn_row_padded,
                            ctx_flat_padded_bf16,
                            [0, gi * Q_HEAD_PAD * head_dim],
                        )

                for gi in pl.range(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    with pl.at(level=pl.Level.CORE_GROUP):
                        ctx_flat_bf16 = pl.slice(
                            attn_row_padded,
                            [1, Q_HEAD_BATCH * head_dim],
                            [0, gi * Q_HEAD_PAD * head_dim],
                        )
                        attn_row = pl.assemble(attn_row, ctx_flat_bf16, [0, q_base * head_dim])

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            return attn_out

    return Qwen3Scope2
