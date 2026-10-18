# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Extracted evaluated implementation; fixed operands come from public config."""

import pypto.language as pl
import pypto.language.distributed as pld
from dspark_attention import dspark_attention
from dspark_context_kv import dspark_context_kv
from dspark_proj import dspark_proj
from hc_head import hc_head
from hc_post import hc_post_prefill
from hc_pre import hc_pre
from lookup_embedding import lookup_embedding
from rmsnorm import rms_norm
from moe import AUX_PAD, IDX_PAD, MOE_INTER, N_EXPERTS_GLOBAL, N_LOCAL, N_RANKS, N_ROUTES, RECV_MAX, TOPK, VOCAB, clear_moe_signals, moe
from config import D, DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_MOE_TOKENS, DSPARK_NOISE_TOKEN_ID, DSPARK_QUERY_PAD, DSPARK_QUERY_TOKENS, DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH, H, HC_DIM, HC_MULT, HEAD_DIM, MAIN_IN, MAX_SEQ_LEN, MIX_HC, ORI_BLOCK_NUM, ORI_MAX_BLOCKS, O_GROUPS, O_GROUP_IN, O_LORA, Q_LORA, ROPE_DIM, T, T_QUERY, WIN, BLOCK_SIZE, KV_ORI_BLOCK_NUM


T_MAIN_DYN = pl.dynamic("DSPARK_BACKBONE_T_MAIN_DYN")


B_DYN = pl.dynamic("DSPARK_BACKBONE_B_DYN")


PREPARE_T_MAIN_DYN = pl.dynamic("DSPARK_PREPARE_T_MAIN_DYN")


PREPARE_B_DYN = pl.dynamic("DSPARK_PREPARE_B_DYN")


METADATA_B_DYN = pl.dynamic("DSPARK_METADATA_B_DYN")


assert DSPARK_QUERY_WIDTH < DSPARK_QUERY_PAD


assert KV_ORI_BLOCK_NUM % DSPARK_MAX_BATCH == 0


PAD_D_TILE = 512


@pl.jit.inline
def prepare_dspark_inputs(
    target_hidden: pl.Tensor[[PREPARE_T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[PREPARE_B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[PREPARE_B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[PREPARE_B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    main_x: pl.Tensor[[PREPARE_T_MAIN_DYN, D], pl.BF16],
    query_token_ids: pl.Tensor[[DSPARK_MOE_TOKENS], pl.INT64],
    query_hc_flat: pl.Tensor[[DSPARK_MOE_TOKENS, HC_MULT * D], pl.FP32],
):
    dspark_proj(target_hidden, main_proj_weight, main_norm_weight, main_x)

    batch = pl.tensor.dim(num_sampled, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_query_ids"):
        for token in pl.range(DSPARK_MOE_TOKENS):
            token_id = pl.cast(0, pl.INT64)
            if token < active_tokens:
                request = token // DSPARK_QUERY_WIDTH
                query_offset = token % DSPARK_QUERY_WIDTH
                token_id = pl.cast(DSPARK_NOISE_TOKEN_ID, pl.INT64)
                if query_offset == 0:
                    token_id = pl.read(next_prefill_tokens, [request])
                    if pl.read(num_sampled, [request]) > 0:
                        token_id = pl.read(last_sampled, [request])
            pl.write(query_token_ids, [token], token_id)

    lookup_hidden = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    query_hc = pl.reshape(query_hc_flat, [DSPARK_MOE_TOKENS, HC_MULT, D])
    lookup_embedding(
        query_token_ids,
        embedding_weight,
        lookup_hidden,
        query_hc,
    )
    return main_x, query_token_ids, query_hc_flat


@pl.jit.inline
def build_dspark_metadata(
    anchor_positions: pl.Tensor[[METADATA_B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, METADATA_B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    query_slot_mapping: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_QUERY_TOKENS], pl.INT64],
    swa_indices: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH], pl.INT32],
    query_positions: pl.Tensor[[DSPARK_QUERY_TOKENS], pl.INT32],
):
    batch = pl.tensor.dim(anchor_positions, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    for metadata_core in pl.spmd(1, name_hint="dspark_query_metadata"):
        for token in pl.range(metadata_core, DSPARK_QUERY_TOKENS):
            pl.write(query_positions, [token], pl.cast(0, pl.INT32))
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(query_slot_mapping, [layer, token], pl.cast(-1, pl.INT64))
            if token < active_tokens:
                request = token // DSPARK_QUERY_WIDTH
                query_offset = token % DSPARK_QUERY_WIDTH
                anchor_position = pl.read(anchor_positions, [request])
                query_position = anchor_position + 1 + query_offset
                pl.write(query_positions, [token], pl.cast(query_position, pl.INT32))
                for layer in pl.range(DSPARK_DRAFT_LAYERS):
                    logical_block = query_position // BLOCK_SIZE
                    block_offset = query_position % BLOCK_SIZE
                    physical_block = pl.read(
                        block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)]
                    )
                    query_slot = physical_block * BLOCK_SIZE + block_offset
                    pl.write(query_slot_mapping, [layer, token], pl.cast(query_slot, pl.INT64))

    for request in pl.spmd(DSPARK_MAX_BATCH, name_hint="dspark_visible_metadata"):
        start_position = pl.cast(0, pl.INT32)
        visible_len = pl.cast(0, pl.INT32)
        if request < batch:
            anchor_position = pl.read(anchor_positions, [request])
            prefix_len = anchor_position + 1
            start_position = pl.cast(pl.max(prefix_len - WIN, 0), pl.INT32)
            visible_len = pl.cast(
                prefix_len + DSPARK_QUERY_WIDTH - start_position,
                pl.INT32,
            )
        for layer in pl.range(DSPARK_DRAFT_LAYERS):
            pl.write(swa_lens, [layer, request], visible_len)
            for visible_offset in pl.range(DSPARK_SWA_INDEX_WIDTH):
                visible_slot = pl.cast(-1, pl.INT32)
                if visible_offset < visible_len:
                    visible_position = start_position + visible_offset
                    logical_block = visible_position // BLOCK_SIZE
                    block_offset = visible_position % BLOCK_SIZE
                    physical_block = pl.read(
                        block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)]
                    )
                    visible_slot = pl.cast(
                        physical_block * BLOCK_SIZE + block_offset,
                        pl.INT32,
                    )
                pl.write(swa_indices, [layer, request, visible_offset], visible_slot)
    return (
        query_slot_mapping,
        swa_indices,
        swa_lens,
        query_positions,
    )


@pl.jit.inline(auto_scope=False)
def draft_layer(
    query_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    draft_layer_index: pl.Scalar[pl.INT32],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    query_positions: pl.Tensor[[T_QUERY], pl.INT32],
    kv_cache: pl.Tensor[[KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_slot_mapping: pl.Tensor[[T_QUERY], pl.INT64],
    swa_indices: pl.Tensor[[DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[DSPARK_MAX_BATCH], pl.INT32],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    query_token_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    output_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    active_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    layer_hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [draft_layer_index * MIX_HC, 0])
    layer_hc_attn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [draft_layer_index * 3])
    layer_hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [draft_layer_index * MIX_HC])
    layer_attn_norm_w: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [draft_layer_index * D])
    layer_wq_a: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [draft_layer_index * D, 0])
    layer_wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [draft_layer_index * Q_LORA, 0])
    layer_wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [draft_layer_index * H * HEAD_DIM])
    layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [draft_layer_index * D, 0])
    layer_gamma_cq: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [draft_layer_index * Q_LORA])
    layer_gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [draft_layer_index * HEAD_DIM])
    layer_attn_sink: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [draft_layer_index * H])
    layer_wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [draft_layer_index * O_GROUPS, 0, 0])
    layer_wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [draft_layer_index * D, 0])
    layer_wo_b_scale: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [draft_layer_index * D])
    layer_hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [draft_layer_index * MIX_HC, 0])
    layer_hc_ffn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [draft_layer_index * 3])
    layer_hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [draft_layer_index * MIX_HC])
    layer_ffn_norm_w: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_w, [D], [draft_layer_index * D])
    layer_gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [draft_layer_index * N_EXPERTS_GLOBAL, 0])
    layer_gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [draft_layer_index * N_EXPERTS_GLOBAL])
    layer_tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [draft_layer_index * VOCAB, 0])
    layer_routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [draft_layer_index * N_LOCAL, 0, 0])
    layer_routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [draft_layer_index * N_LOCAL, 0])
    layer_routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [draft_layer_index * N_LOCAL, 0, 0])
    layer_routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [draft_layer_index * N_LOCAL, 0])
    layer_routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [draft_layer_index * N_LOCAL, 0, 0])
    layer_routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [draft_layer_index * N_LOCAL, 0])
    layer_shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [draft_layer_index * MOE_INTER, 0])
    layer_shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [draft_layer_index * MOE_INTER])
    layer_shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [draft_layer_index * MOE_INTER, 0])
    layer_shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [draft_layer_index * MOE_INTER])
    layer_shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [draft_layer_index * D, 0])
    layer_shared_w2_scale: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [draft_layer_index * D])

    query_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    combine = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        query_hc,
        layer_hc_attn_fn,
        layer_hc_attn_scale,
        layer_hc_attn_base,
        query_mixed,
        post,
        combine,
    )

    query_normed = pl.create_tensor([T_QUERY, D], dtype=pl.BF16)
    query_mixed_active: pl.Tensor[[T_QUERY, D], pl.BF16] = pl.slice(
        query_mixed,
        [T_QUERY, D],
        [0, 0],
    )
    rms_norm(query_mixed_active, layer_attn_norm_w, query_normed)
    attention_output = pl.create_tensor([T_QUERY, D], dtype=pl.BF16)
    dspark_attention(
        query_normed,
        layer_wq_a, layer_wq_b, layer_wq_b_scale, layer_wkv, layer_gamma_cq, layer_gamma_ckv,
        freqs_cos, freqs_sin, query_positions,
        kv_cache, query_slot_mapping, swa_indices, swa_lens,
        layer_attn_sink, layer_wo_a, layer_wo_b, layer_wo_b_scale,
        attention_output,
    )

    padded_attention = pl.create_tensor([T, D], dtype=pl.BF16)
    for pad_idx in pl.spmd(T * (D // PAD_D_TILE), name_hint="dspark_attention_pad"):
        pad_token = pad_idx // (D // PAD_D_TILE)
        pad_col = (pad_idx % (D // PAD_D_TILE)) * PAD_D_TILE
        output_tile = pl.full([1, PAD_D_TILE], dtype=pl.BF16, value=0.0)
        if pad_token < T_QUERY:
            output_tile = attention_output[pad_token : pad_token + 1, pad_col : pad_col + PAD_D_TILE]
        padded_attention[pad_token : pad_token + 1, pad_col : pad_col + PAD_D_TILE] = output_tile

    attention_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    hc_post_prefill(
        padded_attention,
        query_hc,
        post,
        combine,
        attention_hc,
        active_tokens,
    )

    moe(
        attention_hc,
        layer_hc_ffn_fn, layer_hc_ffn_scale, layer_hc_ffn_base,
        layer_ffn_norm_w, layer_gate_w, layer_gate_bias, layer_tid2eid, query_token_ids,
        layer_routed_w1, layer_routed_w1_scale, layer_routed_w3, layer_routed_w3_scale, layer_routed_w2, layer_routed_w2_scale,
        layer_shared_w1, layer_shared_w1_scale, layer_shared_w3, layer_shared_w3_scale, layer_shared_w2, layer_shared_w2_scale,
        output_hc,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        layer_id, active_tokens, my_rank, moe_epoch,
    )
    return output_hc


@pl.jit
def dspark_drafter(
    target_hidden: pl.Tensor[[T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    context_position_ids: pl.Tensor[[T_MAIN_DYN], pl.INT32],
    context_slot_mapping: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_MAIN_DYN], pl.INT64],
    anchor_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS],
        pl.INT32,
    ],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    initial_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    intermediate_hidden: pl.Out[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, T, HC_MULT, D], pl.FP32]
    ],
    head_hidden: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    target_hidden.bind_dynamic(0, T_MAIN_DYN)
    context_position_ids.bind_dynamic(0, T_MAIN_DYN)
    context_slot_mapping.bind_dynamic(1, T_MAIN_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    anchor_positions.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    head_hidden.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(num_sampled, 0)
    target_tokens = pl.tensor.dim(target_hidden, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH

    kv_cache_0 = kv_caches[0]
    kv_cache_1 = kv_caches[1]
    kv_cache_2 = kv_caches[2]
    wkv_0 = pl.slice(wkv, [D, HEAD_DIM], [0, 0])
    wkv_1 = pl.slice(wkv, [D, HEAD_DIM], [D, 0])
    wkv_2 = pl.slice(wkv, [D, HEAD_DIM], [2 * D, 0])
    gamma_ckv_0 = pl.slice(gamma_ckv, [HEAD_DIM], [0])
    gamma_ckv_1 = pl.slice(gamma_ckv, [HEAD_DIM], [HEAD_DIM])
    gamma_ckv_2 = pl.slice(gamma_ckv, [HEAD_DIM], [2 * HEAD_DIM])
    main_x = pl.create_tensor([target_tokens, D], dtype=pl.BF16)
    query_token_ids = pl.create_tensor([T], dtype=pl.INT64)
    hidden_0_flat = pl.reshape(initial_hidden, [T, HC_MULT * D])
    main_x, query_token_ids, hidden_0_flat = prepare_dspark_inputs(
        target_hidden,
        main_proj_weight,
        main_norm_weight,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        embedding_weight,
        main_x,
        query_token_ids,
        hidden_0_flat,
    )
    context_slots_0 = pl.create_tensor([target_tokens], dtype=pl.INT64)
    context_slots_1 = pl.create_tensor([target_tokens], dtype=pl.INT64)
    context_slots_2 = pl.create_tensor([target_tokens], dtype=pl.INT64)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_context_slots"):
        for token in pl.range(target_tokens):
            pl.write(context_slots_0, [token], pl.read(context_slot_mapping, [0, token]))
            pl.write(context_slots_1, [token], pl.read(context_slot_mapping, [1, token]))
            pl.write(context_slots_2, [token], pl.read(context_slot_mapping, [2, token]))

    dspark_context_kv(
        main_x, wkv_0, gamma_ckv_0, freqs_cos, freqs_sin,
        context_position_ids, context_slots_0, kv_cache_0,
    )
    dspark_context_kv(
        main_x, wkv_1, gamma_ckv_1, freqs_cos, freqs_sin,
        context_position_ids, context_slots_1, kv_cache_1,
    )
    dspark_context_kv(
        main_x, wkv_2, gamma_ckv_2, freqs_cos, freqs_sin,
        context_position_ids, context_slots_2, kv_cache_2,
    )

    query_slot_mapping = pl.create_tensor([DSPARK_DRAFT_LAYERS, T_QUERY], dtype=pl.INT64)
    swa_indices = pl.create_tensor(
        [DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH],
        dtype=pl.INT32,
    )
    swa_lens = pl.create_tensor([DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH], dtype=pl.INT32)
    query_positions = pl.create_tensor([T_QUERY], dtype=pl.INT32)
    (
        query_slot_mapping,
        swa_indices,
        swa_lens,
        query_positions,
    ) = build_dspark_metadata(
        anchor_positions,
        block_tables,
        query_slot_mapping,
        swa_indices,
        swa_lens,
        query_positions,
    )
    query_slot_mapping_0 = query_slot_mapping[0]
    query_slot_mapping_1 = query_slot_mapping[1]
    query_slot_mapping_2 = query_slot_mapping[2]
    swa_indices_0 = swa_indices[0]
    swa_indices_1 = swa_indices[1]
    swa_indices_2 = swa_indices[2]
    swa_lens_0 = swa_lens[0]
    swa_lens_1 = swa_lens[1]
    swa_lens_2 = swa_lens[2]
    hidden_1 = intermediate_hidden[0]
    draft_layer(
        initial_hidden, pl.const(0, pl.INT32),
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, query_positions,
        kv_cache_0, query_slot_mapping_0, swa_indices_0, swa_lens_0,
        attn_sink, wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
        gate_w, gate_bias, tid2eid, query_token_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        hidden_1,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(40, pl.INT32), pl.cast(active_tokens, pl.INT32), my_rank, pl.const(1, pl.INT32),
    )

    hidden_2 = intermediate_hidden[1]
    draft_layer(
        hidden_1, pl.const(1, pl.INT32),
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, query_positions,
        kv_cache_1, query_slot_mapping_1, swa_indices_1, swa_lens_1,
        attn_sink, wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
        gate_w, gate_bias, tid2eid, query_token_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        hidden_2,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(41, pl.INT32), pl.cast(active_tokens, pl.INT32), my_rank, pl.const(2, pl.INT32),
    )

    hidden_3 = intermediate_hidden[2]
    draft_layer(
        hidden_2, pl.const(2, pl.INT32),
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, query_positions,
        kv_cache_2, query_slot_mapping_2, swa_indices_2, swa_lens_2,
        attn_sink, wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
        gate_w, gate_bias, tid2eid, query_token_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        hidden_3,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(42, pl.INT32), pl.cast(active_tokens, pl.INT32), my_rank, pl.const(3, pl.INT32),
    )
    clear_moe_signals(hidden_3, arrived, data_arrived, combine_arrived)

    padded_head_hidden = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_head(hidden_3, hc_head_fn, hc_head_scale, hc_head_base, padded_head_hidden)
    head_hidden_flat = pl.reshape(head_hidden, [batch * DSPARK_QUERY_WIDTH, D])
    for token in pl.spmd(T, name_hint="dspark_head_unpad"):
        if token < active_tokens:
            head_hidden_flat[token : token + 1, :] = padded_head_hidden[
                token : token + 1,
                :,
            ]
    return head_hidden


@pl.jit.host
def l3_dspark_drafter(
    target_hidden: pl.Tensor[[N_RANKS, T_MAIN_DYN, MAIN_IN], pl.BF16],
    initial_hidden: pl.Out[
        pl.Tensor[[N_RANKS, DSPARK_MAX_BATCH * DSPARK_QUERY_PAD, HC_MULT, D], pl.FP32]
    ],
    intermediate_hidden: pl.Out[
        pl.Tensor[
            [N_RANKS, DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH * DSPARK_QUERY_PAD, HC_MULT, D],
            pl.FP32,
        ]
    ],
    main_proj_weight: pl.Tensor[[N_RANKS, D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[N_RANKS, D], pl.BF16],
    num_sampled: pl.Tensor[[N_RANKS, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[N_RANKS, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[N_RANKS, B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[N_RANKS, VOCAB, D], pl.BF16],
    context_position_ids: pl.Tensor[[N_RANKS, T_MAIN_DYN], pl.INT32],
    context_slot_mapping: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, T_MAIN_DYN], pl.INT64],
    anchor_positions: pl.Tensor[[N_RANKS, B_DYN], pl.INT32],
    block_tables: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[
        pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    head_hidden: pl.Out[pl.Tensor[[N_RANKS, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
):
    target_hidden.bind_dynamic(1, T_MAIN_DYN)
    context_position_ids.bind_dynamic(1, T_MAIN_DYN)
    context_slot_mapping.bind_dynamic(2, T_MAIN_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    anchor_positions.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, B_DYN)
    head_hidden.bind_dynamic(1, B_DYN)

    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        dspark_drafter(
            target_hidden[rank], main_proj_weight[rank], main_norm_weight[rank],
            num_sampled[rank], last_sampled[rank], next_prefill_tokens[rank],
            embedding_weight[rank],
            context_position_ids[rank], context_slot_mapping[rank], anchor_positions[rank], block_tables[rank],
            freqs_cos[rank], freqs_sin[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_w[rank],
            wq_a[rank], wq_b[rank], wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            kv_caches[rank], attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], ffn_norm_w[rank],
            gate_w[rank], gate_bias[rank], tid2eid[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank], initial_hidden[rank],
            intermediate_hidden[rank], head_hidden[rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
            rank,
            device=rank,
        )
