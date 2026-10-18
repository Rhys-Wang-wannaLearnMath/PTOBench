"""Compatibility config aliases for the Qwen3-14B TQ decode source.

The read-only corpus source imports legacy flat constants. This adapter keeps
the byte-exact structured constants authoritative and exposes only the alias
surface needed by qwen3_14b_decode_tq.py and its upstream helper dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pypto.language as pl

from constants import QWEN3_14B as _QWEN3_14B
from constants import QWEN3_14B_TILING as QWEN3_14B_TILING


class _Qwen3CompatModel:
    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    @property
    def batch(self) -> int:
        return self._base.batch

    @property
    def batch_pad(self) -> int:
        return self._base.batch


@dataclass(frozen=True)
class Qwen3DynamicDims:
    batch: Any
    user_batch: Any
    kv_cache_rows: Any
    block_table_flat: Any
    rope_seq: Any
    layer: Any
    layer_hidden_rows: Any
    layer_inter_rows: Any


USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
QWEN3_14B_DIMS = Qwen3DynamicDims(
    batch=USER_BATCH_DYN,
    user_batch=USER_BATCH_DYN,
    kv_cache_rows=pl.dynamic("KV_CACHE_ROWS_DYN"),
    block_table_flat=pl.dynamic("BLOCK_TABLE_FLAT_DYN"),
    rope_seq=pl.dynamic("ROPE_SEQ_DYN"),
    layer=pl.dynamic("LAYER_DYN"),
    layer_hidden_rows=pl.dynamic("LAYER_HIDDEN_ROWS_DYN"),
    layer_inter_rows=pl.dynamic("LAYER_INTER_ROWS_DYN"),
)

QWEN3_14B = _Qwen3CompatModel(_QWEN3_14B)

BATCH = QWEN3_14B.batch
MAX_SEQ = QWEN3_14B.max_seq
NUM_HEADS = QWEN3_14B.num_heads
NUM_KV_HEADS = QWEN3_14B.num_kv_heads
HEAD_DIM = QWEN3_14B.head_dim
HIDDEN = QWEN3_14B.hidden
INTERMEDIATE = QWEN3_14B.intermediate
KV_HIDDEN = QWEN3_14B.kv_hidden
VOCAB = QWEN3_14B.vocab
REAL_VOCAB = QWEN3_14B.real_vocab
NUM_LAYERS = QWEN3_14B.num_layers

EPS = QWEN3_14B.eps
HIDDEN_INV = QWEN3_14B.hidden_inv
HEAD_DIM_INV = QWEN3_14B.head_dim_inv
ATTN_SCALE = QWEN3_14B.attn_scale
HALF_DIM = QWEN3_14B.half_dim
Q_PER_KV = QWEN3_14B.q_per_kv
Q_HEAD_BATCH = QWEN3_14B.q_head_batch
Q_HEAD_PAD = QWEN3_14B.q_head_pad
Q_GROUPS = QWEN3_14B.q_groups
TOTAL_Q_GROUPS = QWEN3_14B.total_q_groups

BATCH_TILE = 16
SEQ_TILE = QWEN3_14B_TILING.seq_tile
BLOCK_SIZE = QWEN3_14B_TILING.block_size
MAX_BLOCKS_PER_SEQ = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE

K_CHUNK = 256
Q_OUT_CHUNK = 256

KV_CACHE_ROWS_DYN = QWEN3_14B_DIMS.kv_cache_rows
BLOCK_TABLE_FLAT_DYN = QWEN3_14B_DIMS.block_table_flat
ROPE_SEQ_DYN = QWEN3_14B_DIMS.rope_seq
LAYER_DYN = QWEN3_14B_DIMS.layer
LAYER_HIDDEN_ROWS_DYN = QWEN3_14B_DIMS.layer_hidden_rows
LAYER_INTER_ROWS_DYN = QWEN3_14B_DIMS.layer_inter_rows

FINAL_RMS_K_CHUNK = 128
LM_HEAD_K_CHUNK = 128
VOCAB_CHUNK = 64
