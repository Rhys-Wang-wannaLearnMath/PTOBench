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
CASE_CONFIG = QWEN3_14B

# Only the external page size belongs to this decode contract.
QWEN3_14B_TILING = Qwen3TilingConfig(seq_tile=128)


@dataclass(frozen=True)
class Qwen3DynamicDims:
    batch: Any
    user_batch: Any
    kv_cache_rows: Any


USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
QWEN3_14B_DIMS = Qwen3DynamicDims(
    batch=USER_BATCH_DYN,
    user_batch=USER_BATCH_DYN,
    kv_cache_rows=pl.dynamic("KV_CACHE_ROWS_DYN"),
)

# Compatibility aliases required by the preserved official entry annotations.
KV_CACHE_ROWS_DYN = QWEN3_14B_DIMS.kv_cache_rows
VOCAB = CASE_CONFIG.vocab
REAL_VOCAB = CASE_CONFIG.real_vocab
NUM_LAYERS = CASE_CONFIG.num_layers

# Fixed operator and paged-cache contract bindings.
BATCH = CASE_CONFIG.batch
MAX_SEQ = CASE_CONFIG.max_seq
# The selected default runner samples varied sequence lengths.
FULL_SEQ = False
NUM_HEADS = CASE_CONFIG.num_heads
NUM_KV_HEADS = CASE_CONFIG.num_kv_heads
HEAD_DIM = CASE_CONFIG.head_dim
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = CASE_CONFIG.intermediate
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
HALF_DIM = HEAD_DIM // 2
EPS = CASE_CONFIG.eps
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM
SEQ_TILE = QWEN3_14B_TILING.seq_tile
BLOCK_SIZE = SEQ_TILE
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
MAX_BLOCKS_PER_SEQ = MAX_CTX_BLOCKS
NUM_PAGES = BATCH * MAX_BLOCKS_PER_SEQ
CACHE_ROWS = NUM_PAGES * NUM_KV_HEADS * BLOCK_SIZE
Q_HEAD_BATCH = Q_PER_KV
Q_HEAD_PAD = ((Q_HEAD_BATCH + 15) // 16) * 16
Q_GROUPS = CASE_CONFIG.q_groups
TOTAL_Q_GROUPS = CASE_CONFIG.total_q_groups
