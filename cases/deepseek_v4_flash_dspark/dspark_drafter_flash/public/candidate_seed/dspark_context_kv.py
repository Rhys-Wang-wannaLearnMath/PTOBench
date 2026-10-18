# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from config import (
    BLOCK_SIZE,
    D,
    DSPARK_QUERY_TOKENS,
    HEAD_DIM,
    KV_ORI_BLOCK_NUM,
    MAX_SEQ_LEN,
    ROPE_DIM,
)

import pypto.language as pl

from qkv_proj_rope import kv_proj_rope, materialize_rope_rows, rope_prepare

@pl.jit.inline
def dspark_context_kv_query(
    main_x: pl.Tensor[[DSPARK_QUERY_TOKENS, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[DSPARK_QUERY_TOKENS], pl.INT32],
    slot_mapping: pl.Tensor[[DSPARK_QUERY_TOKENS], pl.INT64],
    kv_cache: pl.Tensor[[KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
):
    """Project a padded DSpark query-width context without dynamic shape variables."""
    rope_cos_t = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        position_ids,
        DSPARK_QUERY_TOKENS,
        rope_cos_t,
        rope_sin_t,
    )

    rope_cos_il = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_t, rope_sin_t, rope_cos_il, rope_sin_signed, rope_swap_idx)

    kv = pl.create_tensor([DSPARK_QUERY_TOKENS, HEAD_DIM], dtype=pl.BF16)
    late_dep = pl.system.task_dummy(deps=[])
    kv_proj_rope(
        main_x,
        wkv,
        gamma_ckv,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        kv,
        late_dep,
    )

    kv_cache_flat = pl.reshape(kv_cache, [KV_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_context_kv_query_scatter"):
        for write_t in pl.range(DSPARK_QUERY_TOKENS):
            write_row_i64 = pl.read(slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

