"""Fixed contract for deepseek_v4_flash_dspark.prefill_csa_pro."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Follows HuggingFace config.json"""

    name: str

    # ---- attention / hidden ----
    hidden_size: int
    num_attention_heads: int
    head_dim: int                  # MLA value-head dim
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int                  # grouped output projection
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int

    # ---- MoE ----
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]
    routed_scaling_factor: float
    swiglu_limit: float

    # ---- layers ----
    num_hidden_layers: int
    num_hash_layers: int           # first layers use hash routing
    num_nextn_predict_layers: int  # multi-token-prediction layers
    compress_ratios: Tuple[int, ...]   # per-layer KV compression ratio (0 / 4 / 128)

    # ---- lightning indexer ----
    index_n_heads: int
    index_head_dim: int
    index_topk: int                # compressed positions kept by the indexer

    # ---- hyper-connections (HC) ----
    hc_mult: int                   # hc-stack width
    hc_sinkhorn_iters: int
    hc_eps: float

    # ---- context length / RoPE (YaRN; rope_scaling.* flattened) ----
    max_position_embeddings: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float                       # rope_scaling.factor
    beta_fast: int                           # rope_scaling.beta_fast
    beta_slow: int                           # rope_scaling.beta_slow
    original_max_position_embeddings: int    # rope_scaling.original_max_position_embeddings

    # ---- precision / quantization (quantization_config.* flattened; unused by decode kernels) ----
    dtype: Literal["bf16", "fp8"]              # quantization_config.quant_method
    scale_fmt: Optional[Literal["ue8m0"]]     # quantization_config.scale_fmt
    expert_dtype: Optional[Literal["fp4"]]    # MoE-expert weight dtype (None = same as `dtype`)
    scale_dtype: Literal["fp32", "fp8"]       # dequant-scale storage dtype

    # ---- deployment (not consumed by the decode kernels) ----
    max_batch_size: int            # max supported batch size (cache sizing)

    # ---- derived ----
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


PRO = DeepSeekV4Config(
    name="pro",
    hidden_size=7168,
    num_attention_heads=128,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1536,
    o_lora_rank=1024,
    o_groups=16,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=3072,
    n_routed_experts=384,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=2.5,
    swiglu_limit=10.0,
    num_hidden_layers=61,
    num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(
        128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
    ),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=1024,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=1048576,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype=None,
    scale_dtype="fp8",
    max_batch_size=4,
)

CASE_CONFIG = PRO
CASE_ID = "deepseek_v4_flash_dspark.prefill_csa_pro"

# Fixed packed CSA geometry, cache capacities, and numeric semantics.
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH*PREFILL_SEQ
BLOCK_SIZE = 128
C4A_COMPRESSOR_BLOCK_SIZE = 4
KV_ORI_TABLE_MAX_POSITION_EMBEDDINGS = CASE_CONFIG.max_position_embeddings
KV_ORI_TABLE_MAX_BLOCKS = (KV_ORI_TABLE_MAX_POSITION_EMBEDDINGS+BLOCK_SIZE-1)//BLOCK_SIZE
KV_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
KV_CMP_MAX_BLOCKS = 32
IDX_CACHE_MAX_BLOCKS = 64
ORI_KV_BLOCK_NUM = 128
CMP_KV_BLOCK_NUM = 64
IDX_KV_BLOCK_NUM = 64
DECODE_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
DECODE_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM
DECODE_IDX_BLOCK_NUM = IDX_KV_BLOCK_NUM
CSA_STATE_PHYSICAL_BLOCKS = 65
CSA_INNER_STATE_PHYSICAL_BLOCKS = 65
PREFILL_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
PREFILL_ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
PREFILL_IDX_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B*S
NUM_TOKENS = T
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM//2
Q_LORA = CASE_CONFIG.q_lora_rank
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
WIN = CASE_CONFIG.sliding_window
COMPRESS_RATIO = 4
START_POS = 0
IDX_HEAD_DIM = CASE_CONFIG.index_head_dim
IDX_N_HEADS = CASE_CONFIG.index_n_heads
IDX_TOPK = CASE_CONFIG.index_topk
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
O_GROUP_IN = H*HEAD_DIM//O_GROUPS
OVERLAP = COMPRESS_RATIO==4
COFF = 1+int(OVERLAP)
MAIN_OUT_DIM = COFF*HEAD_DIM
MAIN_COMPRESS_STATE_DIM = 2*MAIN_OUT_DIM
MAIN_STATE_LEN = COFF*COMPRESS_RATIO
INNER_OUT_DIM = COFF*IDX_HEAD_DIM
INNER_COMPRESS_STATE_DIM = 2*INNER_OUT_DIM
INNER_STATE_LEN = COFF*COMPRESS_RATIO
MAX_CMP_WRITES = max(1,T//COMPRESS_RATIO)
CSA_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN+CSA_STATE_BLOCK_SIZE-1)//CSA_STATE_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN+INNER_STATE_BLOCK_SIZE-1)//INNER_STATE_BLOCK_SIZE
CSA_STATE_BLOCK_NUM = CSA_STATE_PHYSICAL_BLOCKS
INNER_STATE_BLOCK_NUM = CSA_INNER_STATE_PHYSICAL_BLOCKS
ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = CMP_MAX_BLOCKS
CSA_ORI_BLOCK_NUM = ORI_BLOCK_NUM
CSA_CMP_BLOCK_NUM = CMP_BLOCK_NUM
INDEXER_SCORE_MAX_BLOCKS = 2
INDEXER_SCORE_CAP = INDEXER_SCORE_MAX_BLOCKS*BLOCK_SIZE
INDEXER_TOPK_CAP = min(IDX_TOPK,INDEXER_SCORE_CAP)
TOPK = WIN+IDX_TOPK
PREFILL_MAX_COMPRESSED = max(1,min(IDX_TOPK,WIN+WIN//2))
PREFILL_SPARSE_TOPK = min(TOPK,min(WIN,S)+PREFILL_MAX_COMPRESSED)
EPS = CASE_CONFIG.rms_norm_eps
NORM_EPS = CASE_CONFIG.rms_norm_eps
NOPE_HEAD_DIM = HEAD_DIM-ROPE_HEAD_DIM
IDX_NOPE_HEAD_DIM = IDX_HEAD_DIM-ROPE_HEAD_DIM
HC_EPS = CASE_CONFIG.hc_eps
HC_SINKHORN_ITER = CASE_CONFIG.hc_sinkhorn_iters
SOFTMAX_SCALE = CASE_CONFIG.softmax_scale
WEIGHTS_SCALE = CASE_CONFIG.index_weights_scale
BF16 = "bfloat16"
FP32 = "float32"
INT8 = "int8"
INT32 = "int32"
INT64 = "int64"
