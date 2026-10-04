# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 2 — RoPE + grouped-query attention.
  1. K/V projection and Q RoPE + pad
  2. QK matmul
  3. Softmax
  4. SV matmul
  5. Online-softmax accumulation + final normalisation

Input projections are FP32; cos/sin tables are FP32; KV caches are BF16.
Output attention is BF16.
"""
from __future__ import annotations

import pypto.language as pl
from config import BATCH, HEAD_DIM, MAX_SEQ, NUM_HEADS, NUM_KV_HEADS

# Tiling constants (aligned to qwen3_32b_decode_tilelet).
Q_HEAD_BATCH = 8        # Q heads batched per attention group
Q_HEAD_PAD = 16         # padded Q rows for cube fractal alignment
SEQ_TILE = 64           # sequence tile for attention loop
SB_BATCH = 16

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
    cache_rows = batch * num_kv_heads * max_seq
    half_dim = head_dim // 2
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)
    max_ctx_blocks = (max_seq + SEQ_TILE - 1) // SEQ_TILE

    @pl.program
    class Qwen3Scope2:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_scope2(
            self,
            q_proj: pl.Tensor[[batch, hidden], pl.FP32],
            k_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            v_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            attn_out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            all_q_padded = pl.create_tensor([batch * total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)

            for b in pl.range(batch):
                ctx_len = pl.read(seq_lens, [b])
                pos = ctx_len - 1
                prefix_len = ctx_len - 1
                ctx_blocks = (prefix_len + SEQ_TILE - 1) // SEQ_TILE
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                # Stage 1: Q RoPE + pad. Keep input caches read-only; the current
                # token is merged explicitly after the historical cache tiles.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
                    for ki in pl.range(num_kv_heads):
                        kv_col = ki * head_dim
                        # Q RoPE + pad (ki == kvh since q_groups == 1).
                        q_base = ki * q_per_kv
                        q_block = pl.reshape(
                            pl.slice(q_proj, [1, Q_HEAD_BATCH * head_dim], [b, q_base * head_dim]),
                            [Q_HEAD_BATCH, head_dim],
                        )
                        q_lo = pl.slice(q_block, [Q_HEAD_BATCH, half_dim], [0, 0])
                        q_hi = pl.slice(q_block, [Q_HEAD_BATCH, half_dim], [0, half_dim])
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
                        all_q_padded = pl.assemble(all_q_padded, rot_lo_bf16, [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD, 0])
                        all_q_padded = pl.assemble(all_q_padded, rot_hi_bf16, [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD, half_dim])
                        all_q_padded = pl.assemble(
                            all_q_padded,
                            pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                            [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                        )

                attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                for gi in pl.parallel(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, head_dim], [b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0])

                    # Stages 2-5: compute each sequence tile and merge online.
                    s0 = 0
                    cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq
                    k_tile = pl.slice(k_cache, [SEQ_TILE, head_dim], [cache_row0, 0])
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                        raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                        valid_len = pl.min(SEQ_TILE, prefix_len)
                        scores_valid = pl.slice(raw_scores, [Q_HEAD_BATCH, SEQ_TILE], [0, 0], valid_shape=[Q_HEAD_BATCH, valid_len])
                        scores = pl.mul(pl.fillpad(scores_valid, pad_value=pl.PadValue.min), attn_scale)
                        mi = pl.row_max(scores)
                        exp_scores = pl.cast(pl.exp(pl.row_expand_sub(scores, mi)), target_type=pl.BF16)
                        li = pl.row_sum(pl.cast(exp_scores, target_type=pl.FP32))
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                        oi = pl.matmul(exp_scores, pl.slice(v_cache, [SEQ_TILE, head_dim], [cache_row0, 0]), out_dtype=pl.FP32)
                    for sb in pl.range(1, ctx_blocks):
                        s0 = sb * SEQ_TILE
                        cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                        k_tile = pl.slice(k_cache, [SEQ_TILE, head_dim], [cache_row0, 0])
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                            valid_len = pl.min(SEQ_TILE, prefix_len - s0)
                            scores_valid = pl.slice(raw_scores, [Q_HEAD_BATCH, SEQ_TILE], [0, 0], valid_shape=[Q_HEAD_BATCH, valid_len])
                            scores = pl.mul(pl.fillpad(scores_valid, pad_value=pl.PadValue.min), attn_scale)
                            cur_mi = pl.row_max(scores)
                            exp_scores = pl.cast(pl.exp(pl.row_expand_sub(scores, cur_mi)), target_type=pl.BF16)
                            cur_li = pl.row_sum(pl.cast(exp_scores, target_type=pl.FP32))
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                            oi_tmp = pl.matmul(exp_scores, pl.slice(v_cache, [SEQ_TILE, head_dim], [cache_row0, 0]), out_dtype=pl.FP32)
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                            oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp, beta))
                            mi = mi_new

                    # Merge the current token without a write/read dependency on
                    # the large input cache tensors.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="current_token"):
                        kv_col = kvh * head_dim
                        current_k_lo = pl.cast(
                            pl.sub(
                                pl.col_expand_mul(pl.slice(k_proj, [1, half_dim], [b, kv_col]), cos_lo),
                                pl.col_expand_mul(pl.slice(k_proj, [1, half_dim], [b, kv_col + half_dim]), sin_lo),
                            ),
                            target_type=pl.BF16,
                        )
                        current_k_hi = pl.cast(
                            pl.add(
                                pl.col_expand_mul(pl.slice(k_proj, [1, half_dim], [b, kv_col + half_dim]), cos_hi),
                                pl.col_expand_mul(pl.slice(k_proj, [1, half_dim], [b, kv_col]), sin_hi),
                            ),
                            target_type=pl.BF16,
                        )
                        current_k_lo_box = pl.cast(
                            pl.col_expand_add(
                                pl.full([Q_HEAD_PAD, half_dim], dtype=pl.FP32, value=0.0),
                                pl.cast(current_k_lo, target_type=pl.FP32),
                            ),
                            target_type=pl.BF16,
                        )
                        current_k_hi_box = pl.cast(
                            pl.col_expand_add(
                                pl.full([Q_HEAD_PAD, half_dim], dtype=pl.FP32, value=0.0),
                                pl.cast(current_k_hi, target_type=pl.FP32),
                            ),
                            target_type=pl.BF16,
                        )
                        current_mi_physical = pl.row_max(
                            pl.mul(
                                pl.add(
                                    pl.matmul(
                                        pl.slice(q_padded, [Q_HEAD_PAD, half_dim], [0, 0]),
                                        current_k_lo_box,
                                        b_trans=True,
                                        out_dtype=pl.FP32,
                                    ),
                                    pl.matmul(
                                        pl.slice(q_padded, [Q_HEAD_PAD, half_dim], [0, half_dim]),
                                        current_k_hi_box,
                                        b_trans=True,
                                        out_dtype=pl.FP32,
                                    ),
                                ),
                                attn_scale,
                            )
                        )
                        current_mi = pl.slice(current_mi_physical, [Q_HEAD_BATCH, 1], [0, 0])
                        current_li = pl.row_sum(
                            pl.full([Q_HEAD_BATCH, Q_HEAD_PAD], dtype=pl.FP32, value=0.0625)
                        )
                        current_v = pl.cast(
                            pl.slice(v_proj, [1, head_dim], [b, kv_col]), target_type=pl.BF16
                        )
                        current_oi = pl.col_expand_add(
                            pl.full([Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0),
                            pl.cast(current_v, target_type=pl.FP32),
                        )
                        mi_new = pl.maximum(mi, current_mi)
                        alpha = pl.exp(pl.sub(mi, mi_new))
                        beta = pl.exp(pl.sub(current_mi, mi_new))
                        li = pl.add(pl.mul(alpha, li), pl.mul(beta, current_li))
                        oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(current_oi, beta))

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                        ctx = pl.row_expand_div(oi, li)
                        ctx_flat_bf16 = pl.cast(pl.reshape(ctx, [1, Q_HEAD_BATCH * head_dim]), target_type=pl.BF16)
                        attn_row = pl.assemble(attn_row, ctx_flat_bf16, [0, q_base * head_dim])

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            return attn_out

    return Qwen3Scope2
