# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE FFN router decode orchestration.
Runs FFN half-compress pre-processing, FFN RMSNorm, learned routing scores,
top-k extraction, and route-weight normalization. Outputs `x_norm`, per-token
expert indices/weights, and the post/comb tensors required by FFN hc_post."""


import pypto.language as pl

from config import (
    B,
    D,
    HC_DIM,
    HC_MULT,
    LAYER_ID,
    MIX_HC,
    NORM_EPS,
    N_EXPERTS,
    N_HASH_LAYERS,
    ROUTE_SCALE,
    S,
    TOPK,
    T,
    VOCAB,
)
from deepseek_v4_decode_hc_pre import deepseek_v4_decode_hc_pre

# Per-layer routing mode is fixed at build time: layers with LAYER_ID < N_HASH_LAYERS
# do tid2eid lookup (no scores, no bias, no topk); the rest do learned-score + bias + topk.
# This implementation currently runs the learned-score path. The public entrypoint
# keeps tid2eid/input_ids so hash-routed layers can share the same call contract.
# hc_pre (ffn)
# ffn_norm + gate (decode chunking).
D_CHUNK          = 512
D_BLOCKS         = D // D_CHUNK

# Routing topk via sort32. SCORE_PAD == sort32 row width; PAIR_PAD covers the
# (val, idx) interleaved output of the topk slice; FP32_NEG_INF fills the
# unused tail so padded slots always rank below real expert scores.
SCORE_PAD        = 32
PAIR_PAD         = 32
TOPK_GATHER_PAD  = PAIR_PAD // 2
FP32_NEG_INF     = -1.0e30


@pl.jit.inline
def _deepseek_v4_decode_moe_router_kernel(
    x_mixed:      pl.Tensor[[B, S, D],                    pl.BF16],
    norm_w:       pl.Tensor[[D],                           pl.FP32],
    gate_w:       pl.Tensor[[N_EXPERTS, D],                pl.FP32],
    gate_bias:    pl.Tensor[[N_EXPERTS],                   pl.FP32],
    x_norm:       pl.Tensor[[T, D],                        pl.BF16],
    indices:      pl.Tensor[[T, TOPK],                     pl.INT32],
    weights:      pl.Tensor[[T, TOPK],                     pl.FP32],
):
    # ---- ffn_norm: RMSNorm over x_mixed (mirrors hc_pre rms). ----
    # x_mixed parameter is rank-3 [B, S, D]; flatten to [T, D] for chunked slice.
    x_mixed_flat = pl.reshape(x_mixed, [T, D])
    inv_rms = pl.create_tensor([1, T], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_norm_rms"):
        sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for db in pl.pipeline(D_BLOCKS, stage=4):
            d0 = db * D_CHUNK
            x_chunk = pl.cast(pl.slice(x_mixed_flat, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T]),
            )
        inv_rms_val = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
        inv_rms = pl.assemble(inv_rms, inv_rms_val, [0, 0])

    # Materialize x_norm as a normal intermediate; the gate dot below reads it
    # back. The same chunk also assembles into the entry's `x_norm` output.
    x_norm_bf16 = pl.create_tensor([T, D], dtype=pl.BF16)
    for db in pl.parallel(0, D_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_norm_apply"):
            d0 = db * D_CHUNK
            inv_rms_col = pl.reshape(pl.slice(inv_rms, [1, T], [0, 0]), [T, 1])
            x_chunk = pl.cast(pl.slice(x_mixed_flat, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
            norm_w_chunk = pl.reshape(pl.slice(norm_w, [D_CHUNK], [d0]), [1, D_CHUNK])
            x_normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_col), norm_w_chunk)
            x_normed_bf16 = pl.cast(x_normed, target_type=pl.BF16)
            x_norm_bf16 = pl.assemble(x_norm_bf16, x_normed_bf16, [0, d0])
            x_norm = pl.assemble(x_norm, x_normed_bf16, [0, d0])

    # ---- Gate.forward: dot(x_norm, gate_w) -> sqrt(softplus(.)) -> +bias. ----
    # Pad-tail of `biased_scores` is initialized to -inf so sort32 ranks padded
    # slots after every real expert score (otherwise their pad-zero would beat
    # negative real scores and topk would pick padding columns 8..31).
    biased_scores = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_init_score_table"):
        biased_scores = pl.assemble(
            biased_scores,
            pl.full([T, SCORE_PAD], dtype=pl.FP32, value=FP32_NEG_INF),
            [0, 0],
        )

    # Single fused kernel: for each expert, reduce dot via row_sum, apply
    # sqrt(softplus(.)) + bias, scatter the per-token scalar into the [T,
    # SCORE_PAD] table. Per-expert kernels (one `with pl.at` each) lower to
    # 8 kernels whose hardware-side outputs collapse to identical values
    # for tokens 1..15; the fused form below reduces correctly on hardware.
    biased_flat = pl.reshape(biased_scores, [T * SCORE_PAD])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_dot"):
        for expert_i in pl.range(N_EXPERTS):
            score_acc = pl.full([1, T], dtype=pl.FP32, value=0.0)
            for db in pl.range(D_BLOCKS):
                d0 = db * D_CHUNK
                x_chunk = pl.cast(pl.slice(x_norm_bf16, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
                w_row = pl.slice(gate_w, [1, D_CHUNK], [expert_i, d0])
                prod = pl.col_expand_mul(x_chunk, w_row)
                score_acc = pl.add(
                    score_acc,
                    pl.reshape(pl.row_sum(prod), [1, T]),
                )
            bias = pl.read(gate_bias, [expert_i])
            logits = score_acc
            zero = pl.mul(logits, 0.0)
            relu_logits = pl.maximum(logits, zero)
            abs_logits = pl.maximum(logits, pl.neg(logits))
            softplus = pl.add(relu_logits, pl.log(pl.add(pl.exp(pl.neg(abs_logits)), 1.0)))
            score_tile = pl.sqrt(softplus)
            biased_tile = pl.add(score_tile, bias)
            biased_row_flat = pl.reshape(biased_tile, [T])
            for t in pl.unroll(T):
                pl.write(biased_flat, [t * SCORE_PAD + expert_i], pl.read(biased_row_flat, [t]))
    biased_scores = pl.reshape(biased_flat, [T, SCORE_PAD])

    # ---- topk via sort32: per-token sort with paired INT32 indices. ----
    sorted_rows = pl.create_tensor([T, 2 * SCORE_PAD], dtype=pl.FP32)
    for t in pl.unroll(T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_sort_top2"):
            score_row = pl.slice(biased_scores, [1, SCORE_PAD], [t, 0])
            idx_init = pl.tensor.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
            sorted_t = pl.tensor.sort32(score_row, idx_init)
            sorted_rows = pl.assemble(sorted_rows, sorted_t, [t, 0])

    # Declare topk_vals_pad before topk_idx_pad: orchestration codegen
    # binds GM buffers in declaration order, and the extract block below
    # writes vals before idx. Reordering these two lines silently swaps
    # which buffer downstream reads see.
    topk_vals_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    topk_idx_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.INT32)
    weight_out_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_extract_top2"):
        # Only init vals (FP32). topk_idx_pad is fully covered by the
        # per-token assemble below, so no explicit zero-init is needed.
        topk_vals_pad = pl.assemble(
            topk_vals_pad,
            pl.full([T, SCORE_PAD], dtype=pl.FP32, value=0.0),
            [0, 0],
        )
        for t in pl.unroll(T):
            topk_pairs = pl.slice(sorted_rows, [1, PAIR_PAD], [t, 0])
            topk_vals = pl.tensor.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
            topk_i_raw = pl.tensor.gather(
                topk_pairs,
                mask_pattern=pl.tile.MaskPattern.P1010,
                output_dtype=pl.INT32,
            )
            topk_vals_valid = pl.slice(
                topk_vals,
                [1, TOPK_GATHER_PAD],
                [0, 0],
                valid_shape=[1, TOPK],
            )
            topk_vals_padded = pl.fillpad(topk_vals_valid, pad_value=pl.PadValue.zero)
            # NOTE on current bias handling:
            #   weights are read straight from the sort32 stream, so they include
            #   the route bias. The zero-bias fixture makes those values equal to
            #   the unbiased route scores. Non-zero bias needs indirect gather from
            #   the original score table once the ptoas backend supports
            #   `pl.tensor.gather(input, dim, index)`.
            topk_vals_pad = pl.assemble(topk_vals_pad, topk_vals_padded, [t, 0])
            topk_idx_pad = pl.assemble(topk_idx_pad, topk_i_raw, [t, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_normalize_weights"):
        denom = pl.reshape(pl.row_sum(topk_vals_pad), [T, 1])
        weights_pad = pl.mul(pl.row_expand_div(topk_vals_pad, denom), ROUTE_SCALE)
        weight_out_pad = pl.assemble(weight_out_pad, weights_pad, [0, 0])

    # Scatter the first TOPK columns of the padded tables to the entry's
    # output [T, TOPK] tensors.
    indices_flat = pl.reshape(indices, [T * TOPK])
    weights_flat = pl.reshape(weights, [T * TOPK])
    topk_idx_flat = pl.reshape(topk_idx_pad, [T * SCORE_PAD])
    weight_out_flat = pl.reshape(weight_out_pad, [T * SCORE_PAD])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="write_route_outputs"):
        for t in pl.unroll(T):
            dst_base = t * TOPK
            src_base = t * SCORE_PAD
            for k in pl.unroll(TOPK):
                indices_flat = pl.write(indices_flat, [dst_base + k], pl.read(topk_idx_flat, [src_base + k]))
                weights_flat = pl.write(weights_flat, [dst_base + k], pl.read(weight_out_flat, [src_base + k]))


@pl.jit
def deepseek_v4_decode_moe_router(
    x_hc:         pl.Tensor[[B, S, HC_MULT, D],            pl.BF16],
    hc_ffn_fn:    pl.Tensor[[MIX_HC, HC_DIM],              pl.FP32],
    hc_ffn_scale: pl.Tensor[[3],                           pl.FP32],
    hc_ffn_base:  pl.Tensor[[MIX_HC],                      pl.FP32],
    norm_w:       pl.Tensor[[D],                           pl.FP32],
    gate_w:       pl.Tensor[[N_EXPERTS, D],                pl.FP32],
    gate_bias:    pl.Tensor[[N_EXPERTS],                   pl.FP32],
    tid2eid:      pl.Tensor[[VOCAB, TOPK],                 pl.INT32],
    input_ids:    pl.Tensor[[B, S],                        pl.INT64],
    x_norm:       pl.Out[pl.Tensor[[T, D],                 pl.BF16]],
    indices:      pl.Out[pl.Tensor[[T, TOPK],              pl.INT32]],
    weights:      pl.Out[pl.Tensor[[T, TOPK],              pl.FP32]],
    post_ffn:     pl.Out[pl.Tensor[[B, S, HC_MULT],        pl.FP32]],
    comb_ffn:     pl.Out[pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32]],
):
    # Stage 1: Block.hc_pre (ffn) -- shared @pl.jit.inline kernel.
    # Discard the returned reshape-view; pl.write side-effects already wrote
    # post_ffn / comb_ffn / x_mixed in-place on the underlying buffers.
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    deepseek_v4_decode_hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn, comb_ffn,
    )

    # Stage 2: ffn_norm + Gate.forward + topk + weight normalize + scatter.
    # tid2eid/input_ids are intentionally unused by the learned-score branch but
    # remain on the public signature for hash-routing compatibility.
    _deepseek_v4_decode_moe_router_kernel(
        x_mixed,
        norm_w, gate_w, gate_bias,
        x_norm, indices, weights,
    )
    return x_norm, indices, weights, post_ffn, comb_ffn


