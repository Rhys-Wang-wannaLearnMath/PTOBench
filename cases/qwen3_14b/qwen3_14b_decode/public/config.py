"""Fixed evaluation contract for qwen3_14b.qwen3_14b_decode."""

from __future__ import annotations

class Qwen3DecodeConfig:
    """Readable fixed values for the single-layer Qwen3 decode ABI."""

    def __init__(
        self,
        *,
        name,
        family,
        variant,
        batch,
        max_seq,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden,
        intermediate,
        eps,
        q_head_batch,
        q_head_pad,
        block_size,
    ):
        self.name = name
        self.family = family
        self.variant = variant
        self.batch = batch
        self.max_seq = max_seq
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden = hidden
        self.intermediate = intermediate
        self.eps = eps
        self.q_head_batch = q_head_batch
        self.q_head_pad = q_head_pad
        self.block_size = block_size

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

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def max_blocks_per_seq(self) -> int:
        return (self.max_seq + self.block_size - 1) // self.block_size

    @property
    def num_blocks(self) -> int:
        return self.batch * self.max_blocks_per_seq

    @property
    def cache_rows(self) -> int:
        return self.num_blocks * self.num_kv_heads * self.block_size


QWEN3_14B_DECODE = Qwen3DecodeConfig(
    name="qwen3-14b-single-layer-decode",
    family="qwen3",
    variant="14b",
    batch=16,
    max_seq=4096,
    num_heads=40,
    num_kv_heads=8,
    head_dim=128,
    hidden=5120,
    intermediate=17408,
    eps=1e-6,
    q_head_batch=5,
    q_head_pad=16,
    block_size=256,
)

CASE_CONFIG = QWEN3_14B_DECODE


# Fixed operator and paged-cache ABI bindings.
BATCH = CASE_CONFIG.batch
MAX_SEQ = CASE_CONFIG.max_seq
NUM_HEADS = CASE_CONFIG.num_heads
NUM_KV_HEADS = CASE_CONFIG.num_kv_heads
HEAD_DIM = CASE_CONFIG.head_dim
HIDDEN = CASE_CONFIG.hidden
KV_HIDDEN = CASE_CONFIG.kv_hidden
INTERMEDIATE = CASE_CONFIG.intermediate
EPS = CASE_CONFIG.eps
Q_HEAD_BATCH = CASE_CONFIG.q_head_batch
Q_HEAD_PAD = CASE_CONFIG.q_head_pad
SEQ_TILE = CASE_CONFIG.block_size
BLOCK_SIZE = CASE_CONFIG.block_size
HALF_DIM = CASE_CONFIG.half_dim
Q_PER_KV = CASE_CONFIG.q_per_kv
Q_GROUPS = CASE_CONFIG.q_groups
TOTAL_Q_GROUPS = CASE_CONFIG.total_q_groups
HIDDEN_INV = CASE_CONFIG.hidden_inv
HEAD_DIM_INV = CASE_CONFIG.head_dim_inv
ATTN_SCALE = CASE_CONFIG.attn_scale
MAX_BLOCKS_PER_SEQ = CASE_CONFIG.max_blocks_per_seq
NUM_BLOCKS = CASE_CONFIG.num_blocks
CACHE_ROWS = CASE_CONFIG.cache_rows
