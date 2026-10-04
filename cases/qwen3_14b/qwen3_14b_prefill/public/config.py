"""Fixed evaluation contract for qwen3_14b.qwen3_14b_prefill."""

from __future__ import annotations


class Qwen3PrefillConfig:
    """Readable fixed values for the Qwen3-14B prefill ABI."""

    def __init__(
        self,
        *,
        name: str,
        batch: int,
        max_seq: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate: int,
        eps: float,
        seq_len_quantum: int,
        block_size: int,
    ) -> None:
        self.name = name
        self.batch = batch
        self.max_seq = max_seq
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate = intermediate
        self.eps = eps
        self.seq_len_quantum = seq_len_quantum
        self.block_size = block_size

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
    def head_dim_inv(self) -> float:
        return 1.0 / self.head_dim

    @property
    def half_dim(self) -> int:
        return self.head_dim // 2

    @property
    def q_per_kv(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def attn_scale(self) -> float:
        return 1.0 / (self.head_dim ** 0.5)

    @property
    def max_blocks_per_seq(self) -> int:
        return (self.max_seq + self.block_size - 1) // self.block_size

    @property
    def num_blocks(self) -> int:
        return self.batch * self.max_blocks_per_seq

    @property
    def cache_rows(self) -> int:
        return self.num_blocks * self.num_kv_heads * self.block_size


QWEN3_14B_PREFILL = Qwen3PrefillConfig(
    name="qwen3-14b-single-layer-prefill",
    batch=16,
    max_seq=128,
    num_heads=40,
    num_kv_heads=8,
    head_dim=128,
    intermediate=17408,
    eps=1e-6,
    seq_len_quantum=64,
    block_size=256,
)

CASE_CONFIG = QWEN3_14B_PREFILL


# Fixed operator, layout, and paged-cache ABI bindings.
BATCH = CASE_CONFIG.batch
MAX_SEQ = CASE_CONFIG.max_seq
NUM_HEADS = CASE_CONFIG.num_heads
NUM_KV_HEADS = CASE_CONFIG.num_kv_heads
HEAD_DIM = CASE_CONFIG.head_dim
HIDDEN = CASE_CONFIG.hidden
KV_HIDDEN = CASE_CONFIG.kv_hidden
INTERMEDIATE = CASE_CONFIG.intermediate
EPS = CASE_CONFIG.eps
HIDDEN_INV = CASE_CONFIG.hidden_inv
HEAD_DIM_INV = CASE_CONFIG.head_dim_inv
HALF_DIM = CASE_CONFIG.half_dim
Q_PER_KV = CASE_CONFIG.q_per_kv
ATTN_SCALE = CASE_CONFIG.attn_scale
SEQ_LEN_QUANTUM = CASE_CONFIG.seq_len_quantum
BLOCK_SIZE = CASE_CONFIG.block_size
MAX_BLOCKS_PER_SEQ = CASE_CONFIG.max_blocks_per_seq
NUM_BLOCKS = CASE_CONFIG.num_blocks
CACHE_ROWS = CASE_CONFIG.cache_rows

# Fixed tensor dtypes and externally visible shapes.
HIDDEN_STATES_DTYPE = "bfloat16"
SEQ_LENS_DTYPE = "int32"
MODEL_WEIGHT_DTYPE = "bfloat16"
NORM_WEIGHT_DTYPE = "float32"
INDEX_DTYPE = "int32"
HIDDEN_STATES_SHAPE = (BATCH, MAX_SEQ, HIDDEN)
SEQ_LENS_SHAPE = (BATCH,)
BLOCK_TABLE_SHAPE = (NUM_BLOCKS,)
SLOT_MAPPING_SHAPE = (BATCH * MAX_SEQ,)
CACHE_SHAPE = (CACHE_ROWS, HEAD_DIM)
OUTPUT_SHAPE = HIDDEN_STATES_SHAPE
