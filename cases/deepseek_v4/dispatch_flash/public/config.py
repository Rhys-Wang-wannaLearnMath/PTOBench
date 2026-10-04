"""Fixed contract for deepseek_v4.dispatch_flash."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Source-shaped DeepSeek-V4 model configuration."""

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
    name="flash", hidden_size=4096, num_attention_heads=64, head_dim=512,
    qk_rope_head_dim=64, q_lora_rank=1024, o_lora_rank=1024, o_groups=8,
    sliding_window=128, rms_norm_eps=1e-6, vocab_size=129280,
    moe_intermediate_size=2048, n_routed_experts=256, n_shared_experts=1,
    num_experts_per_tok=6, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
    swiglu_limit=10.0, num_hidden_layers=43, num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0),
    index_n_heads=64, index_head_dim=128, index_topk=512, hc_mult=4,
    hc_sinkhorn_iters=20, hc_eps=1e-6, max_position_embeddings=16384,
    rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=16.0,
    beta_fast=32, beta_slow=1, original_max_position_embeddings=65536,
    dtype="fp8", scale_fmt="ue8m0", expert_dtype="fp4", scale_dtype="fp8",
    max_batch_size=4,
)

CASE_CONFIG = FLASH

# Fixed decode and single-card EP bindings from the selected source workload.
DECODE_BATCH = 8
DECODE_SEQ = 1
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
EP_WORLD_SIZE = 8
EP_RANK = 0
RECV_MAX = EP_WORLD_SIZE * DECODE_TOKENS
D = CASE_CONFIG.hidden_size
TOPK = CASE_CONFIG.num_experts_per_tok
N_LOCAL_EXPERTS = CASE_CONFIG.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS

X_NORM_I8_SHAPE = (DECODE_TOKENS, D)
X_NORM_SCALE_SHAPE = (DECODE_TOKENS, 1)
INDICES_SHAPE = (DECODE_TOKENS, TOPK)
WEIGHTS_SHAPE = (DECODE_TOKENS, TOPK)
RECV_X_SHAPE = (N_LOCAL_EXPERTS, RECV_MAX, D)
RECV_SCALE_DQ_SHAPE = (N_LOCAL_EXPERTS, RECV_MAX)
RECV_WEIGHTS_SHAPE = (N_LOCAL_EXPERTS, RECV_MAX)
RECV_TOKEN_SHAPE = (N_LOCAL_EXPERTS, RECV_MAX)
RECV_EXPERT_COUNT_SHAPE = (N_LOCAL_EXPERTS, 1)

X_NORM_I8_DTYPE = "int8"
X_NORM_SCALE_DTYPE = "float32"
INDICES_DTYPE = "int32"
WEIGHTS_DTYPE = "float32"
RECV_X_DTYPE = "int8"
RECV_SCALE_DQ_DTYPE = "float32"
RECV_WEIGHTS_DTYPE = "float32"
RECV_TOKEN_DTYPE = "int32"
RECV_EXPERT_COUNT_DTYPE = "int32"
