"""Fixed FLASH preset and ABI for deepseek_v4.prefill_moe_ep2_flash."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekV4Config:
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
    max_position_embeddings=16384,
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

PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
MOE_TOKENS = PREFILL_TOKENS
T = MOE_TOKENS
EP_WORLD_SIZE = 2
N_RANKS = EP_WORLD_SIZE
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts // 8 * EP_WORLD_SIZE
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
RECV_MAX = EP_WORLD_SIZE * MOE_TOKENS
D = CASE_CONFIG.hidden_size
MOE_INTER = CASE_CONFIG.moe_intermediate_size
TOPK = CASE_CONFIG.num_experts_per_tok
VOCAB = CASE_CONFIG.vocab_size
HC_MULT = CASE_CONFIG.hc_mult
HC_DIM = CASE_CONFIG.hc_dim
MIX_HC = CASE_CONFIG.mix_hc
N_ROUTES = T * TOPK
MAX_PER_SRC = T
NORM_EPS = CASE_CONFIG.rms_norm_eps
HC_EPS = CASE_CONFIG.hc_eps
HC_SINKHORN_ITER = CASE_CONFIG.hc_sinkhorn_iters
ROUTE_SCALE = CASE_CONFIG.routed_scaling_factor
N_HASH_LAYERS = CASE_CONFIG.num_hash_layers
SWIGLU_LIMIT = CASE_CONFIG.swiglu_limit
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
LAYER_ID = 0
NUM_TOKENS = T

TENSOR_LAYOUT = "ND"
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
    "x_hc": "bfloat16",
    "hc_ffn_fn": "float32",
    "hc_ffn_scale": "float32",
    "hc_ffn_base": "float32",
    "norm_w": "bfloat16",
    "gate_w": "float32",
    "gate_bias": "float32",
    "tid2eid": "int32",
    "input_ids": "int64",
    "routed_w1": "int8",
    "routed_w1_scale": "float32",
    "routed_w3": "int8",
    "routed_w3_scale": "float32",
    "routed_w2": "int8",
    "routed_w2_scale": "float32",
    "shared_w1": "int8",
    "shared_w1_scale": "float32",
    "shared_w3": "int8",
    "shared_w3_scale": "float32",
    "shared_w2": "int8",
    "shared_w2_scale": "float32",
    "x_next": "bfloat16",
}
RESIDENT_INPUTS = (
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
)
RESIDENT_LAYOUT = "stacked"
SCALAR_DTYPES = {"layer_id": "int32", "num_tokens": "int32"}
