# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NPU reference for Qwen3-14B sampled-token embedding lookup."""

from __future__ import annotations

import pypto.language as pl

from config import BATCH, HIDDEN, SAMPLED_IDS_PAD, VOCAB


HIDDEN_CHUNK = 256

assert HIDDEN % HIDDEN_CHUNK == 0


@pl.jit
def token_embed_fwd(
    sampled_ids: pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    next_hidden: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
):
    """Gather embedding rows for sampled token ids."""
    for b in pl.parallel(0, BATCH, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_embed"):
            token_id = pl.read(sampled_ids, [b, 0])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            for k0 in pl.range(0, HIDDEN, HIDDEN_CHUNK):
                hidden_chunk = pl.slice(
                    embed_weight, [1, HIDDEN_CHUNK], [token_row, k0]
                )
                next_hidden = pl.assemble(next_hidden, hidden_chunk, [b, k0])

    return next_hidden
