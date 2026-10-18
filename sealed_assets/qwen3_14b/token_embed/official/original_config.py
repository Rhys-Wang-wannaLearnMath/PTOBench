"""Compatibility config aliases for Qwen3-14B leaf kernels.

This adapter keeps the byte-exact structured constants authoritative while
exporting the legacy flat names used by greedy_sample.py and token_embed.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pypto.language as pl

from constants import QWEN3_14B as QWEN3_14B
from constants import QWEN3_14B_TILING as QWEN3_14B_TILING


@dataclass(frozen=True)
class Qwen3DynamicDims:
    """Dynamic dimensions used by the JIT/program signatures."""

    user_batch: Any
    kv_cache_rows: Any
    block_table_flat: Any
    rope_seq: Any
    layer: Any
    layer_hidden_rows: Any
    layer_inter_rows: Any


QWEN3_14B_DIMS = Qwen3DynamicDims(
    user_batch=pl.dynamic("USER_BATCH_DYN"),
    kv_cache_rows=pl.dynamic("KV_CACHE_ROWS_DYN"),
    block_table_flat=pl.dynamic("BLOCK_TABLE_FLAT_DYN"),
    rope_seq=pl.dynamic("ROPE_SEQ_DYN"),
    layer=pl.dynamic("LAYER_DYN"),
    layer_hidden_rows=pl.dynamic("LAYER_HIDDEN_ROWS_DYN"),
    layer_inter_rows=pl.dynamic("LAYER_INTER_ROWS_DYN"),
)

BATCH = QWEN3_14B.batch
HIDDEN = QWEN3_14B.hidden
VOCAB = QWEN3_14B.vocab
REAL_VOCAB = QWEN3_14B.real_vocab
SAMPLED_IDS_PAD = QWEN3_14B.sampled_ids_pad
