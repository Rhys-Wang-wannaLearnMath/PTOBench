"""Fixed deepseek_v4.moe_ep_ep2_pro contract derived from the selected PRO preset."""
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

EP_WORLD_SIZE = 2
CASE_CONFIG = replace(PRO, n_routed_experts=PRO.n_routed_experts // 8 * EP_WORLD_SIZE)

DECODE_BATCH = 8
DECODE_SEQ = 1
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
MOE_TOKENS = DECODE_TOKENS
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EP_RANK = 0
EP_ROUTING_GLOBAL = True
N_RANKS = EP_WORLD_SIZE
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
RECEIVE_CAPACITY_WORLD_SIZE = 8
DECODE_RECV_MAX = RECEIVE_CAPACITY_WORLD_SIZE * DECODE_TOKENS
PREFILL_RECV_MAX = RECEIVE_CAPACITY_WORLD_SIZE * PREFILL_TOKENS
PREFILL_LAYER_RECV_MAX = PREFILL_RECV_MAX
RECV_MAX = max(DECODE_RECV_MAX, PREFILL_LAYER_RECV_MAX)
D = CASE_CONFIG.hidden_size
TOPK = CASE_CONFIG.num_experts_per_tok
VOCAB = CASE_CONFIG.vocab_size
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
MOE_INTER = CASE_CONFIG.moe_intermediate_size
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
N_ROUTES = T * TOPK
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
LAYER_ID = 0

# Fixed external tensor ABI: rank-leading contiguous ND tensors.
X_HC_SHAPE = (N_RANKS, T, HC_MULT, D)
HC_FFN_FN_SHAPE = (N_RANKS, MIX_HC, HC_DIM)
HC_FFN_SCALE_SHAPE = (N_RANKS, 3)
HC_FFN_BASE_SHAPE = (N_RANKS, MIX_HC)
NORM_W_SHAPE = (N_RANKS, D)
GATE_W_SHAPE = (N_RANKS, N_EXPERTS_GLOBAL, D)
GATE_BIAS_SHAPE = (N_RANKS, N_EXPERTS_GLOBAL)
TID2EID_SHAPE = (N_RANKS, VOCAB, TOPK)
INPUT_IDS_SHAPE = (N_RANKS, T)
ROUTED_W1_SHAPE = (N_RANKS, N_LOCAL, MOE_INTER, D)
ROUTED_W1_SCALE_SHAPE = (N_RANKS, N_LOCAL, MOE_INTER)
ROUTED_W3_SHAPE = ROUTED_W1_SHAPE
ROUTED_W3_SCALE_SHAPE = ROUTED_W1_SCALE_SHAPE
ROUTED_W2_SHAPE = (N_RANKS, N_LOCAL, D, MOE_INTER)
ROUTED_W2_SCALE_SHAPE = (N_RANKS, N_LOCAL, D)
SHARED_W1_SHAPE = (N_RANKS, MOE_INTER, D)
SHARED_W1_SCALE_SHAPE = (N_RANKS, MOE_INTER)
SHARED_W3_SHAPE = SHARED_W1_SHAPE
SHARED_W3_SCALE_SHAPE = SHARED_W1_SCALE_SHAPE
SHARED_W2_SHAPE = (N_RANKS, D, MOE_INTER)
SHARED_W2_SCALE_SHAPE = (N_RANKS, D)
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
ROUTED_W1_DTYPE = "int8"
ROUTED_W1_SCALE_DTYPE = "float32"
ROUTED_W3_DTYPE = "int8"
ROUTED_W3_SCALE_DTYPE = "float32"
ROUTED_W2_DTYPE = "int8"
ROUTED_W2_SCALE_DTYPE = "float32"
SHARED_W1_DTYPE = "int8"
SHARED_W1_SCALE_DTYPE = "float32"
SHARED_W3_DTYPE = "int8"
SHARED_W3_SCALE_DTYPE = "float32"
SHARED_W2_DTYPE = "int8"
SHARED_W2_SCALE_DTYPE = "float32"
X_NEXT_DTYPE = "bfloat16"
LAYER_ID_DTYPE = "int32"
