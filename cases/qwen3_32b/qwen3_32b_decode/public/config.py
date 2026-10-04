"""Fixed evaluation contract for qwen3_32b.qwen3_32b_decode."""

from __future__ import annotations


class Qwen3DecodeConfig:
    """Readable fixed values for the single-layer Qwen3-32B decode ABI."""

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
        intermediate,
        eps,
        q_head_batch,
        q_head_pad,
        seq_tile,
    ):
        self.name = name
        self.family = family
        self.variant = variant
        self.batch = batch
        self.max_seq = max_seq
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate = intermediate
        self.eps = eps
        self.q_head_batch = q_head_batch
        self.q_head_pad = q_head_pad
        self.seq_tile = seq_tile

    @property
    def hidden(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_hidden(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def hidden_inv(self) -> float:
        return 1.0 / self.hidden

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
        return 1.0 / (self.head_dim ** 0.5)

    @property
    def max_ctx_blocks(self) -> int:
        return (self.max_seq + self.seq_tile - 1) // self.seq_tile

    @property
    def cache_rows(self) -> int:
        return self.batch * self.num_kv_heads * self.max_seq


QWEN3_32B_DECODE = Qwen3DecodeConfig(
    name="qwen3-32b-single-layer-decode",
    family="qwen3",
    variant="32b",
    batch=16,
    max_seq=4096,
    num_heads=64,
    num_kv_heads=8,
    head_dim=128,
    intermediate=25600,
    eps=1e-6,
    q_head_batch=8,
    q_head_pad=16,
    seq_tile=256,
)

CASE_CONFIG = QWEN3_32B_DECODE


# Fixed operator ABI bindings used by Ref, Candidate, Oracle, and evaluator.
BATCH = CASE_CONFIG.batch
MAX_SEQ = CASE_CONFIG.max_seq
NUM_HEADS = CASE_CONFIG.num_heads
NUM_KV_HEADS = CASE_CONFIG.num_kv_heads
HEAD_DIM = CASE_CONFIG.head_dim
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = CASE_CONFIG.kv_hidden
INTERMEDIATE = CASE_CONFIG.intermediate
EPS = CASE_CONFIG.eps
Q_HEAD_BATCH = CASE_CONFIG.q_head_batch
Q_HEAD_PAD = CASE_CONFIG.q_head_pad
SEQ_TILE = CASE_CONFIG.seq_tile
HALF_DIM = CASE_CONFIG.half_dim
Q_PER_KV = CASE_CONFIG.q_per_kv
Q_GROUPS = CASE_CONFIG.q_groups
TOTAL_Q_GROUPS = CASE_CONFIG.total_q_groups
HIDDEN_INV = CASE_CONFIG.hidden_inv
ATTN_SCALE = CASE_CONFIG.attn_scale
MAX_CTX_BLOCKS = CASE_CONFIG.max_ctx_blocks
CACHE_ROWS = CASE_CONFIG.cache_rows
