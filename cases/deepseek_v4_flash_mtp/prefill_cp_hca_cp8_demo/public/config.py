"""Fixed contract for deepseek_v4_flash_mtp.prefill_cp_hca_cp8_demo."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Follows the selected source DeepSeek-V4 configuration."""

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
    name="demo", hidden_size=4096, num_attention_heads=64, head_dim=512,
    qk_rope_head_dim=64, q_lora_rank=1024, o_lora_rank=1024, o_groups=8,
    sliding_window=128, rms_norm_eps=1e-6, vocab_size=129280,
    moe_intermediate_size=4096, n_routed_experts=16, n_shared_experts=1,
    num_experts_per_tok=2, scoring_func="sqrtsoftplus", routed_scaling_factor=1.0,
    swiglu_limit=0.0, num_hidden_layers=8, num_hash_layers=0,
    num_nextn_predict_layers=1, compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
    index_n_heads=64, index_head_dim=128, index_topk=512, hc_mult=4,
    hc_sinkhorn_iters=20, hc_eps=1e-6, max_position_embeddings=4096,
    rope_theta=10000.0, compress_rope_theta=40000.0, rope_factor=40.0,
    beta_fast=32, beta_slow=1, original_max_position_embeddings=0,
    dtype="fp8", scale_fmt="ue8m0", expert_dtype=None, scale_dtype="fp8",
    max_batch_size=4,
)

CASE_CONFIG = DEMO
FLASH = CASE_CONFIG

DECODE_BATCH = 4
DECODE_SEQ = 2
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ

BLOCK_SIZE = 128
C4A_COMPRESSOR_BLOCK_SIZE = 4
C128_COMPRESSOR_BLOCK_SIZE = 8
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
KV_ORI_TABLE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
KV_CMP_MAX_BLOCKS = 32
IDX_CACHE_MAX_BLOCKS = 64
ORI_KV_BLOCK_NUM = 128
CMP_KV_BLOCK_NUM = 32
IDX_KV_BLOCK_NUM = 64
HCA_STATE_PHYSICAL_BLOCKS = 64
CSA_STATE_PHYSICAL_BLOCKS = 65
CSA_INNER_STATE_PHYSICAL_BLOCKS = 65
DECODE_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
DECODE_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM
DECODE_IDX_BLOCK_NUM = IDX_KV_BLOCK_NUM
PREFILL_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
PREFILL_ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
PREFILL_IDX_BLOCK_NUM = DECODE_IDX_BLOCK_NUM

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38

D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HC_MULT = CASE_CONFIG.hc_mult
HC_DIM = CASE_CONFIG.hc_dim
MIX_HC = CASE_CONFIG.mix_hc
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_DIM = CASE_CONFIG.qk_rope_head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
Q_LORA = CASE_CONFIG.q_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
O_LORA = CASE_CONFIG.o_lora_rank
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
IDX_TOPK = CASE_CONFIG.index_topk
WIN = CASE_CONFIG.sliding_window
SOFTMAX_SCALE = CASE_CONFIG.softmax_scale
EPS = CASE_CONFIG.rms_norm_eps

TAIL_ROWS = 128
CP_CHOICES = (2, 4, 8)
CP_SIZE = 8
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
EPOCHS = 1
NUM_SEGMENTS = 2 * CP_SIZE
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
CP_TAIL_WINDOW_ROWS = NUM_SEGMENTS * TAIL_ROWS
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_SIZE
OVERLAY_BASE = ORI_CACHE_ROWS
PRED_OVERLAY_ROWS = TAIL_ROWS
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2
MAX_COMPRESSED_ROWS_PER_SEGMENT = MAX_SEGMENT_TILES
MAX_COMPRESS_LEAVES = 1 + MAX_SEGMENT_TILES
COMPRESS_RATIO = 128
OUT_DIM = HEAD_DIM
COMPRESS_STATE_DIM = 2 * OUT_DIM
HCA_STATE_BLOCK_SIZE = 8
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = HCA_STATE_PHYSICAL_BLOCKS
HCA_CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
STATE_ROWS = HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE
LEAF_CMP_BLOCKS = HCA_CMP_BLOCK_NUM
LEAF_CMP_ROWS = LOCAL_PARTS * MAX_COMPRESS_LEAVES * LEAF_CMP_BLOCKS * BLOCK_SIZE
CMP_ROWS_PER_SEGMENT = MAX_COMPRESSED_ROWS_PER_SEGMENT
CMP_ROWS_PER_RANK = LOCAL_PARTS * CMP_ROWS_PER_SEGMENT
CMP_WINDOW_ROWS = CP_SIZE * CMP_ROWS_PER_RANK
CMP_META_DIM = 8
STATE_META_DIM = 8
STATE_WINDOW_ROWS = CP_SIZE * TAIL_ROWS

