# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 SWA (Sliding Window Attention) decode orchestration — `compress_ratio == 0` path.
Active in layers 0/1/7 of the model (3 of the 8 layers in demo). No KV compression, so neither
compressor nor indexer is invoked; topk for sparse_attn is window_topk_idxs only and the KV cache
holds only the sliding window (no compressed portion). YaRN frequency scaling is also disabled
in this path (model.py:478-479 selects base rope_theta when compress_ratio==0).
Companion files: deepseek_v4_decode_csa.py (ratio=4)
                 deepseek_v4_decode_hca.py (ratio=128)."""
import pypto.language as pl
import importlib.util as _importlib_util
from pathlib import Path as _Path

from deepseek_v4_decode_hc_pre import deepseek_v4_decode_hc_pre
from deepseek_v4_decode_hc_post import deepseek_v4_decode_hc_post
from deepseek_v4_decode_qkv_proj_rope import deepseek_v4_decode_qkv_proj_rope

_SPARSE_ATTN_WITH_PROJ_PATH = (
    _Path(__file__).resolve().parent / "deepseek_v4_decode_sparse_attn.py"
)
_sparse_attn_spec = _importlib_util.spec_from_file_location(
    "_deepseek_v4_sparse_attn_with_proj_compat",
    _SPARSE_ATTN_WITH_PROJ_PATH,
)
if _sparse_attn_spec is None or _sparse_attn_spec.loader is None:
    raise ImportError(f"cannot load sparse-attn with-proj compat module from {_SPARSE_ATTN_WITH_PROJ_PATH}")
_sparse_attn_mod = _importlib_util.module_from_spec(_sparse_attn_spec)
_sparse_attn_spec.loader.exec_module(_sparse_attn_mod)
deepseek_v4_decode_sparse_attn_with_proj = _sparse_attn_mod.deepseek_v4_decode_sparse_attn_with_proj
from config import (
    B, S, T, EPS, D, H, HEAD_DIM, ROPE_HEAD_DIM, NOPE_HEAD_DIM,
    Q_LORA, Q_PROJ_OUT_CHUNK, Q_PROJ_HEAD_BLOCKS, WIN, SOFTMAX_SCALE,
    INT8_SCALE_MAX, INT8_AMAX_EPS, HC_MULT, MIX_HC, HC_DIM,
    HC_SINKHORN_ITER, HC_EPS, MAX_SEQ_LEN, O_LORA, O_GROUPS, O_GROUP_IN,
    BLOCK_SIZE, ORI_MAX_BLOCKS, MAX_BLOCKS, BLOCK_NUM, TOPK,
    SPARSE_IDX_TOPK, SPARSE_TOPK, SPARSE_CMP_MAX_BLOCKS,
    SPARSE_CMP_BLOCK_NUM, START_POS,
)


@pl.jit
def deepseek_v4_decode_swa(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post_t = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_mixed = deepseek_v4_decode_hc_pre(
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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_step"):
        pos = pl.cast(start_pos, pl.INDEX)
        cos_row = pl.cast(pl.slice(freqs_cos, [1, ROPE_HEAD_DIM], [pos, 0]), target_type=pl.FP32)
        sin_row = pl.cast(pl.slice(freqs_sin, [1, ROPE_HEAD_DIM], [pos, 0]), target_type=pl.FP32)
        rope_cos_fp32 = pl.col_expand(
            pl.full([T, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
            cos_row,
        )
        rope_sin_fp32 = pl.col_expand(
            pl.full([T, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
            sin_row,
        )
        rope_cos_t = pl.cast(rope_cos_fp32, target_type=pl.BF16)
        rope_sin_t = pl.cast(rope_sin_fp32, target_type=pl.BF16)

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q = deepseek_v4_decode_qkv_proj_rope(
        x_mixed,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos_t,
        rope_sin_t,
        even_select_t,
        odd_select_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )

    kv_cache_flat = pl.reshape(kv_cache, [BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    block_table_flat = pl.reshape(block_table, [B * MAX_BLOCKS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_scatter_kv"):
        ori_slot = start_pos % WIN
        for b in pl.parallel(0, B, 1):
            blk_id = pl.cast(pl.read(block_table_flat, [b]), pl.INDEX)
            dst_row = blk_id * BLOCK_SIZE + ori_slot
            kv_cache_flat = pl.assemble(
                kv_cache_flat,
                kv[b:b + 1, 0:HEAD_DIM],
                [dst_row, 0],
            )
    kv_cache = pl.reshape(kv_cache_flat, [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    sparse_topk = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_topk"):
        idx_row = pl.arange(0, [1, WIN], dtype=pl.INT32)
        pad_row = pl.full([1, SPARSE_IDX_TOPK], dtype=pl.INT32, value=-1)
        sparse_topk_row = pl.concat(idx_row, pad_row)
        sparse_topk = pl.col_expand(
            pl.full([T, SPARSE_TOPK], dtype=pl.INT32, value=-1),
            sparse_topk_row,
        )

    cmp_kv_dummy = pl.create_tensor([SPARSE_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_block_table_dummy = pl.create_tensor([B, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_cmp_dummy"):
        cmp_block_table_dummy = pl.full([B, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32, value=-1)

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    deepseek_v4_decode_sparse_attn_with_proj(
        q,
        kv_cache,
        block_table,
        cmp_kv_dummy,
        cmp_block_table_dummy,
        sparse_topk,
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
    x_out = deepseek_v4_decode_hc_post(
        attn_out_3d,
        x_hc,
        post_t,
        comb_t,
        x_out,
    )
    return x_out


