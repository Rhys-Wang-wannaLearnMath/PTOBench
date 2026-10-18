"""Fixed contract for deepseek_v4_flash_dspark.decode_layer_swa_demo; DEMO SWA, TP2/EP2."""

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

# Fixed evaluated operator ABI and official builder bindings.
CASE_ID = 'deepseek_v4_flash_dspark.decode_layer_swa_demo'
ATTENTION_KIND = 'swa'
OFFICIAL_BUILDER = 'build_swa_layer_specs'
LAYER_ID = 0
# Fixed decode deployment. MOE_TOKENS is computed before final TP/EP binding.
DECODE_BATCH = 64
DSPARK_SPEC_TOKENS = 7
DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
DECODE_START_POS = 8192
SOURCE_TP = 4
DP = 4
SOURCE_EP = 16
MOE_TOKENS = DECODE_TOKENS * DP // SOURCE_EP
TP_SIZE = 2
EP_SIZE = 2
TP = TP_SIZE
EP = EP_SIZE
TP_GROUPS = EP_SIZE // TP_SIZE
FIXTURE_START_POS = None
FIXTURE_BATCH = MOE_TOKENS // DECODE_SEQ
FIXTURE_LOCAL_T = FIXTURE_BATCH * DECODE_SEQ
LOCAL_T = FIXTURE_LOCAL_T
B = DECODE_BATCH // TP_SIZE
S = DECODE_SEQ
T = B * S
N_RANKS = EP_SIZE
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts // SOURCE_EP * EP_SIZE
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
MOE_INTER = CASE_CONFIG.moe_intermediate_size
TOPK = CASE_CONFIG.num_experts_per_tok
VOCAB = CASE_CONFIG.vocab_size
RECV_MAX = EP_SIZE * MOE_TOKENS
N_ROUTES = MOE_TOKENS * TOPK
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
NOPE_HEAD_DIM = CASE_CONFIG.nope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = CASE_CONFIG.q_lora_rank
SOFTMAX_SCALE = CASE_CONFIG.softmax_scale
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
HC_SINKHORN_ITER = CASE_CONFIG.hc_sinkhorn_iters
HC_EPS = CASE_CONFIG.hc_eps
EPS = CASE_CONFIG.rms_norm_eps
WIN = CASE_CONFIG.sliding_window
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
LOCAL_O_GROUPS = O_GROUPS // TP_SIZE
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA
BLOCK_SIZE = 32
KV_ORI_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_ORI_BLOCK_NUM = 512
# Attention source explicitly uses a 1M context ceiling for every preset.
MAX_SEQ_LEN = 1_048_576
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
# Tensor/scalar storage contracts; model precision metadata stays on CASE_CONFIG.
HC_DTYPE = "float32"
ACTIVATION_DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
QUANTIZED_WEIGHT_DTYPE = "int8"
SCALE_DTYPE = "float32"
MAPPING_DTYPE = "int64"
INDEX_DTYPE = "int32"
SCALAR_DTYPE = "int32"

# Official default Fixture start positions (the builder receives None).
DEFAULT_START_PATTERN = (DECODE_START_POS, WIN - 1, 31)
FIXTURE_START_POSITIONS = tuple(DEFAULT_START_PATTERN[i % len(DEFAULT_START_PATTERN)]
                                for i in range(FIXTURE_BATCH))
FIXTURE_KV_SEQ_LENS = tuple(start + S for start in FIXTURE_START_POSITIONS)
