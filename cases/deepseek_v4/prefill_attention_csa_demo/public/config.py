"""Fixed DEMO contract for DeepSeek V4 packed prefill CSA attention."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Follows the source DeepSeek V4 configuration surface."""

    name: str
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]
    routed_scaling_factor: float
    swiglu_limit: float
    num_hidden_layers: int
    num_hash_layers: int
    num_nextn_predict_layers: int
    compress_ratios: Tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    max_position_embeddings: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_max_position_embeddings: int
    dtype: Literal["bf16", "fp8"]
    scale_fmt: Optional[Literal["ue8m0"]]
    expert_dtype: Optional[Literal["fp4"]]
    scale_dtype: Literal["fp32", "fp8"]
    max_batch_size: int

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def index_nope_head_dim(self) -> int:
        return self.index_head_dim - self.qk_rope_head_dim

    @property
    def index_weights_scale(self) -> float:
        return self.index_head_dim ** -0.5 * self.index_n_heads ** -0.5

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


DEMO = DeepSeekV4Config(
    name="demo",
    hidden_size=4096,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1024,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=4096,
    n_routed_experts=16,
    n_shared_experts=1,
    num_experts_per_tok=2,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.0,
    swiglu_limit=0.0,
    num_hidden_layers=8,
    num_hash_layers=0,
    num_nextn_predict_layers=1,
    compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=4096,
    rope_theta=10000.0,
    compress_rope_theta=40000.0,
    rope_factor=40.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=0,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype=None,
    scale_dtype="fp8",
    max_batch_size=4,
)

CASE_CONFIG = DEMO


# Fixed deployment and operator bindings for packed CSA prefill.
DECODE_BATCH = 8
DECODE_SEQ = 1
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
BLOCK_SIZE = 128
C4A_COMPRESSOR_BLOCK_SIZE = 4
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38

KV_CMP_MAX_BLOCKS = 32
IDX_CACHE_MAX_BLOCKS = 64
DECODE_CMP_BLOCK_NUM = DECODE_BATCH * KV_CMP_MAX_BLOCKS
DECODE_IDX_BLOCK_NUM = DECODE_BATCH * IDX_CACHE_MAX_BLOCKS
PREFILL_ORI_MAX_BLOCKS = (PREFILL_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
PREFILL_ORI_BLOCK_NUM = PREFILL_BATCH * PREFILL_ORI_MAX_BLOCKS
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
PREFILL_IDX_BLOCK_NUM = DECODE_IDX_BLOCK_NUM

# Fixed packed-CSA ABI and derived dimensions (mirrors the selected source).
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = CASE_CONFIG.q_lora_rank
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
WIN = CASE_CONFIG.sliding_window
COMPRESS_RATIO = 4
START_POS = 0
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
IDX_HEAD_DIM = CASE_CONFIG.index_head_dim
IDX_N_HEADS = CASE_CONFIG.index_n_heads
IDX_TOPK = CASE_CONFIG.index_topk
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
SPARSE_TOPK = WIN + IDX_TOPK
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
COFF = 2
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_LEN = COFF * COMPRESS_RATIO
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = PREFILL_BATCH * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = CMP_MAX_BLOCKS
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
PREFILL_SPARSE_TOPK = min(SPARSE_TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
SPARSE_PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)
CSA_ORI_BLOCK_NUM = SPARSE_ORI_MAX_BLOCKS
CSA_CMP_BLOCK_NUM = CMP_BLOCK_NUM
