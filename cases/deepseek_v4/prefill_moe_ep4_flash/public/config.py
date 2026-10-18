"""Fixed FLASH contract for deepseek_v4.prefill_moe_ep4_flash."""

from dataclasses import dataclass, replace
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


FLASH = DeepSeekV4Config(
    name="flash",
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
    moe_intermediate_size=2048,
    n_routed_experts=256,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hidden_layers=43,
    num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
    ),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=16384,  # 8k prompt + 512 decode steps target; official 1M;
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype="fp4",
    scale_dtype="fp8",
    max_batch_size=4,
)
CASE_CONFIG = replace(FLASH, n_routed_experts=FLASH.n_routed_experts // 8 * 4)

# Fixed EP4 prefill operator ABI; implementation tiling stays in Ref/Candidate.
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
MOE_TOKENS = PREFILL_TOKENS
EP_WORLD_SIZE = 4
EP_RANK = 0
LAYER_ID = 0
NUM_TOKENS = PREFILL_TOKENS
SCALAR_DTYPE = "int32"
B = PREFILL_BATCH
S = PREFILL_SEQ
T = PREFILL_TOKENS
D = CASE_CONFIG.hidden_size
TOPK = CASE_CONFIG.num_experts_per_tok
VOCAB = CASE_CONFIG.vocab_size
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
MOE_INTER = CASE_CONFIG.moe_intermediate_size
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // EP_WORLD_SIZE
N_RANKS = EP_WORLD_SIZE
N_ROUTES = T * TOPK
MAX_PER_SRC = T
RECV_MAX = EP_WORLD_SIZE * MOE_TOKENS
PREFILL_RECV_MAX = EP_WORLD_SIZE * PREFILL_TOKENS
NORM_EPS = CASE_CONFIG.rms_norm_eps
HC_EPS = CASE_CONFIG.hc_eps
HC_SINKHORN_ITER = CASE_CONFIG.hc_sinkhorn_iters
ROUTE_SCALE = CASE_CONFIG.routed_scaling_factor
N_HASH_LAYERS = CASE_CONFIG.num_hash_layers
SWIGLU_LIMIT = CASE_CONFIG.swiglu_limit
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38

X_HC_SHAPE = (EP_WORLD_SIZE, T, HC_MULT, D)
HC_FFN_FN_SHAPE = (EP_WORLD_SIZE, MIX_HC, HC_DIM)
HC_FFN_SCALE_SHAPE = (EP_WORLD_SIZE, 3)
HC_FFN_BASE_SHAPE = (EP_WORLD_SIZE, MIX_HC)
NORM_W_SHAPE = (EP_WORLD_SIZE, D)
GATE_W_SHAPE = (EP_WORLD_SIZE, N_EXPERTS_GLOBAL, D)
GATE_BIAS_SHAPE = (EP_WORLD_SIZE, N_EXPERTS_GLOBAL)
TID2EID_SHAPE = (EP_WORLD_SIZE, VOCAB, TOPK)
INPUT_IDS_SHAPE = (EP_WORLD_SIZE, T)
ROUTED_W1_SHAPE = (EP_WORLD_SIZE, N_LOCAL, MOE_INTER, D)
ROUTED_W1_SCALE_SHAPE = (EP_WORLD_SIZE, N_LOCAL, MOE_INTER)
ROUTED_W3_SHAPE = ROUTED_W1_SHAPE
ROUTED_W3_SCALE_SHAPE = ROUTED_W1_SCALE_SHAPE
ROUTED_W2_SHAPE = (EP_WORLD_SIZE, N_LOCAL, D, MOE_INTER)
ROUTED_W2_SCALE_SHAPE = (EP_WORLD_SIZE, N_LOCAL, D)
SHARED_W1_SHAPE = (EP_WORLD_SIZE, MOE_INTER, D)
SHARED_W1_SCALE_SHAPE = (EP_WORLD_SIZE, MOE_INTER)
SHARED_W3_SHAPE = SHARED_W1_SHAPE
SHARED_W3_SCALE_SHAPE = SHARED_W1_SCALE_SHAPE
SHARED_W2_SHAPE = (EP_WORLD_SIZE, D, MOE_INTER)
SHARED_W2_SCALE_SHAPE = (EP_WORLD_SIZE, D)
X_NEXT_SHAPE = X_HC_SHAPE

X_HC_DTYPE = "bfloat16"
HC_FFN_FN_DTYPE = "float32"
HC_FFN_SCALE_DTYPE = "float32"
HC_FFN_BASE_DTYPE = "float32"
NORM_W_DTYPE = "bfloat16"
GATE_W_DTYPE = "float32"
GATE_BIAS_DTYPE = "float32"
TID2EID_DTYPE = "int32"
INPUT_IDS_DTYPE = "int64"
ROUTED_W_DTYPE = "int8"
ROUTED_W_SCALE_DTYPE = "float32"
SHARED_W_DTYPE = "int8"
SHARED_W_SCALE_DTYPE = "float32"
X_NEXT_DTYPE = "bfloat16"
