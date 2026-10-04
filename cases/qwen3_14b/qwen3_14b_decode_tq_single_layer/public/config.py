"""Fixed contract for qwen3_14b.qwen3_14b_decode_tq_single_layer."""

from dataclasses import dataclass
from math import log2, sqrt


@dataclass(frozen=True)
class Qwen3Config:
    """Single-source structured constants for the Qwen3-14B kernels."""

    name: str
    family: str
    variant: str

    # Model shape.
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

    # Shared numeric/model invariants.
    eps: float

    # Qwen3-14B attention grouping invariant.
    q_head_batch: int
    q_head_pad: int

    # Quantization constants.
    int8_scale_max: float
    int8_amax_eps: float

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


QWEN3_14B = Qwen3Config(
    name="qwen3-14b",
    family="qwen3",
    variant="14b",
    batch=16,
    max_seq=4096,
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
    int8_scale_max=127.0,
    int8_amax_eps=1e-4,
)

CASE_CONFIG = QWEN3_14B

assert QWEN3_14B.q_per_kv == QWEN3_14B.q_head_batch, (
    f"q_per_kv ({QWEN3_14B.q_per_kv}) must equal q_head_batch ({QWEN3_14B.q_head_batch}) "
    f"(qk_norm / rope_kv_cache assume one Q group per KV head)"
)
assert QWEN3_14B.q_head_pad % 4 == 0 and QWEN3_14B.q_head_pad // 2 >= QWEN3_14B.q_head_batch, (
    f"q_head_pad ({QWEN3_14B.q_head_pad}) must be a multiple of 4 with "
    f"q_head_pad // 2 ({QWEN3_14B.q_head_pad // 2}) >= q_head_batch ({QWEN3_14B.q_head_batch})"
)
assert QWEN3_14B.total_q_groups % 2 == 0, (
    f"total_q_groups ({QWEN3_14B.total_q_groups}) must be even (fa_fused pairs Q groups)"
)

# Fixed operator and deployment bindings; model defaults remain distinct.
BATCH = CASE_CONFIG.batch
MAX_SEQ = CASE_CONFIG.max_seq
NUM_HEADS = CASE_CONFIG.num_heads
NUM_KV_HEADS = CASE_CONFIG.num_kv_heads
HEAD_DIM = CASE_CONFIG.head_dim
HIDDEN = CASE_CONFIG.hidden
INTERMEDIATE = CASE_CONFIG.intermediate
KV_HIDDEN = CASE_CONFIG.kv_hidden
VOCAB = CASE_CONFIG.vocab
REAL_VOCAB = CASE_CONFIG.real_vocab
NUM_LAYERS = CASE_CONFIG.num_layers
EPS = CASE_CONFIG.eps
HIDDEN_INV = CASE_CONFIG.hidden_inv
HEAD_DIM_INV = CASE_CONFIG.head_dim_inv
ATTN_SCALE = CASE_CONFIG.attn_scale
HALF_DIM = CASE_CONFIG.half_dim
Q_PER_KV = CASE_CONFIG.q_per_kv
Q_HEAD_BATCH = CASE_CONFIG.q_head_batch
Q_HEAD_PAD = CASE_CONFIG.q_head_pad
Q_GROUPS = CASE_CONFIG.q_groups
TOTAL_Q_GROUPS = CASE_CONFIG.total_q_groups

EVAL_BATCH = CASE_CONFIG.batch
EVAL_MAX_SEQ = 127
EVAL_NUM_LAYERS = 1
BLOCK_SIZE = 128  # Physical paged-cache layout, not an implementation tile.
N_LEVELS = 16
TQ_BITS = min(int(log2(N_LEVELS)), 4)
# The source Lloyd-Max solver fixes these numerical semantics.
TQ_MAX_ITER = 200
TQ_TOL = 1e-10
TQ_SIGMA = 1.0 / sqrt(HEAD_DIM)
TQ_INIT_LO = -3.5 * TQ_SIGMA
TQ_INIT_HI = 3.5 * TQ_SIGMA
TQ_CDF_GUARD = 1e-15
MAX_BLOCKS_PER_SEQ = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
EVAL_MAX_BLOCKS_PER_SEQ = (EVAL_MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
EVAL_LAYER_CACHE_ROWS = EVAL_BATCH * EVAL_MAX_BLOCKS_PER_SEQ * NUM_KV_HEADS * BLOCK_SIZE
EVAL_CMP_CACHE_ROWS = EVAL_NUM_LAYERS * EVAL_LAYER_CACHE_ROWS
EVAL_LAYER_HIDDEN_ROWS = EVAL_NUM_LAYERS * HIDDEN
EVAL_LAYER_INTER_ROWS = EVAL_NUM_LAYERS * INTERMEDIATE
EVAL_ROT_ROWS = EVAL_NUM_LAYERS * HEAD_DIM

# External tensors are contiguous row-major/ND. Projection weights are
# input-by-output; the LM head is vocabulary-by-hidden. No public scalars.
TENSOR_SHAPES = {
    "hidden_states": (EVAL_BATCH, HIDDEN),
    "input_rms_weight": (EVAL_NUM_LAYERS, HIDDEN),
    "wq": (EVAL_LAYER_HIDDEN_ROWS, HIDDEN),
    "wk": (EVAL_LAYER_HIDDEN_ROWS, KV_HIDDEN),
    "wv": (EVAL_LAYER_HIDDEN_ROWS, KV_HIDDEN),
    "q_norm_weight": (EVAL_NUM_LAYERS, HEAD_DIM),
    "k_norm_weight": (EVAL_NUM_LAYERS, HEAD_DIM),
    "seq_lens": (EVAL_BATCH,),
    "block_table": (EVAL_BATCH * EVAL_MAX_BLOCKS_PER_SEQ,),
    "slot_mapping": (EVAL_BATCH,),
    "rope_cos": (EVAL_MAX_SEQ, HEAD_DIM),
    "rope_sin": (EVAL_MAX_SEQ, HEAD_DIM),
    "quant_k_cache": (EVAL_CMP_CACHE_ROWS, HALF_DIM),
    "quant_v_cache": (EVAL_CMP_CACHE_ROWS, HALF_DIM),
    "quant_k_scales": (EVAL_CMP_CACHE_ROWS, 1),
    "quant_v_scales": (EVAL_CMP_CACHE_ROWS, 1),
    "rot_matrices": (EVAL_ROT_ROWS, HEAD_DIM),
    "tq_codebook": (1, N_LEVELS),
    "wo": (EVAL_LAYER_HIDDEN_ROWS, HIDDEN),
    "post_rms_weight": (EVAL_NUM_LAYERS, HIDDEN),
    "w_gate": (EVAL_LAYER_HIDDEN_ROWS, INTERMEDIATE),
    "w_up": (EVAL_LAYER_HIDDEN_ROWS, INTERMEDIATE),
    "w_down": (EVAL_LAYER_INTER_ROWS, HIDDEN),
    "final_norm_weight": (1, HIDDEN),
    "lm_head_weight": (VOCAB, HIDDEN),
    "out": (EVAL_BATCH, VOCAB),
}
ARGUMENT_ORDER = tuple(TENSOR_SHAPES)
TENSOR_DTYPES = {
    "hidden_states": "bfloat16", "input_rms_weight": "float32",
    "wq": "bfloat16", "wk": "bfloat16", "wv": "bfloat16",
    "q_norm_weight": "float32", "k_norm_weight": "float32",
    "seq_lens": "int32", "block_table": "int32", "slot_mapping": "int32",
    "rope_cos": "float32", "rope_sin": "float32",
    "quant_k_cache": "uint8", "quant_v_cache": "uint8",
    "quant_k_scales": "float32", "quant_v_scales": "float32",
    "rot_matrices": "bfloat16", "tq_codebook": "float32",
    "wo": "bfloat16", "post_rms_weight": "float32",
    "w_gate": "bfloat16", "w_up": "bfloat16", "w_down": "bfloat16",
    "final_norm_weight": "float32", "lm_head_weight": "bfloat16", "out": "float32",
}
