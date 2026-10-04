"""Fixed public contract for qwen3_14b.decode_layer_single_layer_max_seq_127.

The selected QWEN3_14B preset is the complete model configuration needed by
the byte-exact decode source.  Implementation tuning constants remain in the
Ref/Candidate closure.
"""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl


@dataclass(frozen=True)
class Qwen3TilingConfig:
    """Selected Qwen3 page-size ABI values."""

    seq_tile: int
    vocab_chunk: int

    @property
    def block_size(self) -> int:
        return self.seq_tile


@dataclass(frozen=True)
class Qwen3Config:
    """Source-shaped fixed Qwen3-14B model preset surface."""

    name: str
    family: str
    variant: str
    batch: int
    max_seq: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden: int
    intermediate: int
    vocab: int
    real_vocab: int
    num_layers: int
    sampled_ids_pad: int
    eps: float
    q_head_batch: int
    q_head_pad: int

    @property
    def batch_pad(self) -> int:
        return self.batch

    @property
    def kv_hidden(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def hidden_inv(self) -> float:
        return 1.0 / self.hidden

    @property
    def head_dim_inv(self) -> float:
        return 1.0 / self.head_dim

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def half_dim(self) -> int:
        return self.head_dim // 2

    @property
    def q_per_kv(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def q_groups(self) -> int:
        return self.q_per_kv // self.q_head_batch

    @property
    def total_q_groups(self) -> int:
        return self.num_kv_heads * self.q_groups


# The source's PTO2_MANUAL_MAX_SEQ binding is fixed to 127 for this Case.
QWEN3_14B = Qwen3Config(
    name="qwen3-14b",
    family="qwen3",
    variant="14b",
    batch=16,
    max_seq=127,
    num_heads=40,
    num_kv_heads=8,
    head_dim=128,
    hidden=40 * 128,
    intermediate=17408,
    vocab=152064,
    real_vocab=151936,
    num_layers=40,
    sampled_ids_pad=8,
    eps=1e-6,
    q_head_batch=5,
    q_head_pad=16,
)
QWEN3_14B_TILING = Qwen3TilingConfig(seq_tile=128, vocab_chunk=512)


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

# Compatibility aliases required by the preserved official entry annotations.
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
KV_CACHE_ROWS_DYN = QWEN3_14B_DIMS.kv_cache_rows
BLOCK_TABLE_FLAT_DYN = QWEN3_14B_DIMS.block_table_flat
ROPE_SEQ_DYN = QWEN3_14B_DIMS.rope_seq
LAYER_DYN = QWEN3_14B_DIMS.layer
LAYER_HIDDEN_ROWS_DYN = QWEN3_14B_DIMS.layer_hidden_rows
LAYER_INTER_ROWS_DYN = QWEN3_14B_DIMS.layer_inter_rows

CASE_CONFIG = QWEN3_14B

# Fixed operator and paged-cache contract bindings.
BATCH = CASE_CONFIG.batch
MAX_SEQ = CASE_CONFIG.max_seq
NUM_HEADS = CASE_CONFIG.num_heads
NUM_KV_HEADS = CASE_CONFIG.num_kv_heads
HEAD_DIM = CASE_CONFIG.head_dim
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = CASE_CONFIG.intermediate
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
HALF_DIM = HEAD_DIM // 2
EPS = CASE_CONFIG.eps
SEQ_TILE = QWEN3_14B_TILING.seq_tile
BLOCK_SIZE = SEQ_TILE
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
MAX_BLOCKS_PER_SEQ = MAX_CTX_BLOCKS
NUM_PAGES = BATCH * MAX_BLOCKS_PER_SEQ
CACHE_ROWS = NUM_PAGES * NUM_KV_HEADS * BLOCK_SIZE
KV_SPLITS = MAX_CTX_BLOCKS
Q_HEAD_BATCH = Q_PER_KV
Q_HEAD_PAD = ((Q_HEAD_BATCH + 15) // 16) * 16
