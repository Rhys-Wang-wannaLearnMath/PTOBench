"""Fixed contract: deepseek_v4_flash_dspark.moe_flash.

The FLASH model preset stays unnormalized; the composed EP2 operator has
32 global experts and preserves the source's import-time 32-token capacity.
"""
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
CASE_CONFIG = FLASH
CASE_ID = "deepseek_v4_flash_dspark.moe_flash"

DECODE_BATCH = 64
DECODE_SEQ = 2
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
DP = 4
# Source config computes token capacity under EP16 before moe.py binds EP2.
EP = 16
MOE_TOKENS = DECODE_TOKENS * DP // EP
# The entry changes ranks/receive capacity without recomputing MOE_TOKENS.
EP = 2
RECV_MAX = EP * MOE_TOKENS
BLOCK_SIZE = 128
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
TOPK = CASE_CONFIG.num_experts_per_tok
VOCAB = CASE_CONFIG.vocab_size
MOE_INTER = CASE_CONFIG.moe_intermediate_size
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts // 16 * EP
N_LOCAL = N_EXPERTS_GLOBAL // EP
N_RANKS = EP
T = MOE_TOKENS
D = CASE_CONFIG.hidden_size
N_ROUTES = T * TOPK
LAYER_ID = 0
NUM_TOKENS = T
BALANCED_ROUTING = False
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38

TENSOR_SHAPES = {
    "x_hc": (N_RANKS, T, HC_MULT, D),
    "hc_ffn_fn": (N_RANKS, MIX_HC, HC_DIM),
    "hc_ffn_scale": (N_RANKS, 3),
    "hc_ffn_base": (N_RANKS, MIX_HC),
    "norm_w": (N_RANKS, D),
    "gate_w": (N_RANKS, N_EXPERTS_GLOBAL, D),
    "gate_bias": (N_RANKS, N_EXPERTS_GLOBAL),
    "tid2eid": (N_RANKS, VOCAB, TOPK),
    "input_ids": (N_RANKS, T),
    "routed_w1": (N_RANKS, N_LOCAL, MOE_INTER, D),
    "routed_w1_scale": (N_RANKS, N_LOCAL, MOE_INTER),
    "routed_w3": (N_RANKS, N_LOCAL, MOE_INTER, D),
    "routed_w3_scale": (N_RANKS, N_LOCAL, MOE_INTER),
    "routed_w2": (N_RANKS, N_LOCAL, D, MOE_INTER),
    "routed_w2_scale": (N_RANKS, N_LOCAL, D),
    "shared_w1": (N_RANKS, MOE_INTER, D),
    "shared_w1_scale": (N_RANKS, MOE_INTER),
    "shared_w3": (N_RANKS, MOE_INTER, D),
    "shared_w3_scale": (N_RANKS, MOE_INTER),
    "shared_w2": (N_RANKS, D, MOE_INTER),
    "shared_w2_scale": (N_RANKS, D),
    "x_next": (N_RANKS, T, HC_MULT, D),
}
TENSOR_DTYPES = {
    "x_hc": "float32", "hc_ffn_fn": "float32", "hc_ffn_scale": "float32",
    "hc_ffn_base": "float32", "norm_w": "bfloat16", "gate_w": "float32",
    "gate_bias": "float32", "tid2eid": "int32", "input_ids": "int64",
    "routed_w1": "int8", "routed_w1_scale": "float32",
    "routed_w3": "int8", "routed_w3_scale": "float32",
    "routed_w2": "int8", "routed_w2_scale": "float32",
    "shared_w1": "int8", "shared_w1_scale": "float32",
    "shared_w3": "int8", "shared_w3_scale": "float32",
    "shared_w2": "int8", "shared_w2_scale": "float32", "x_next": "float32",
}
SCALAR_DTYPES = {"layer_id": "int32", "num_tokens": "int32"}
