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
from qkv_proj_rope import kv_proj_rope, materialize_rope_rows, rope_prepare
from config import BLOCK_SIZE, CASE_CONFIG as M


T_DYN = pl.dynamic("DSPARK_CONTEXT_KV_T_DYN")


ORI_BLOCK_NUM_DYN = pl.dynamic("DSPARK_CONTEXT_KV_ORI_BLOCK_NUM_DYN")


D = M.hidden_size


HEAD_DIM = M.head_dim


ROPE_DIM = M.qk_rope_head_dim


MAX_SEQ_LEN = M.max_position_embeddings


@pl.jit.inline
def dspark_context_kv(
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(main_x, 0)

    rope_cos_t = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.BF16)
    materialize_rope_rows(freqs_cos, freqs_sin, position_ids, t_dim, rope_cos_t, rope_sin_t)

    rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_t, rope_sin_t, rope_cos_il, rope_sin_signed, rope_swap_idx)

    # This no-work source task seeds the explicit kv_proj dependency chain.
    late_dep = pl.system.task_dummy(deps=[])
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(main_x, wkv, gamma_ckv, rope_cos_il, rope_sin_signed, rope_swap_idx, kv, late_dep)

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_context_kv_scatter"):
        for write_t in pl.range(t_dim):
            write_row_i64 = pl.read(slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

    return kv_cache
