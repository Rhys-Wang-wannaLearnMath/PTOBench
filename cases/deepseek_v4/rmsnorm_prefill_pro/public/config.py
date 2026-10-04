"""Fixed DeepSeek-V4 PRO RMSNorm prefill contract."""

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


PRO = DeepSeekV4Config(
    name="pro", hidden_size=7168, num_attention_heads=128, head_dim=512,
    qk_rope_head_dim=64, q_lora_rank=1536, o_lora_rank=1024, o_groups=16,
    sliding_window=128, rms_norm_eps=1e-6, vocab_size=129280,
    moe_intermediate_size=3072, n_routed_experts=384, n_shared_experts=1,
    num_experts_per_tok=6, scoring_func="sqrtsoftplus", routed_scaling_factor=2.5,
    swiglu_limit=10.0, num_hidden_layers=61, num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                     4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                     4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                     4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                     4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                     4, 0),
    index_n_heads=64, index_head_dim=128, index_topk=1024, hc_mult=4,
    hc_sinkhorn_iters=20, hc_eps=1e-6, max_position_embeddings=1048576,
    rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=16.0,
    beta_fast=32, beta_slow=1, original_max_position_embeddings=65536,
    dtype="fp8", scale_fmt="ue8m0", expert_dtype=None, scale_dtype="fp8",
    max_batch_size=4,
)

CASE_CONFIG = PRO

# Fixed prefill ABI. Implementation tile and pipeline choices stay in the Ref.
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
T = PREFILL_TOKENS
D = CASE_CONFIG.hidden_size
EPS = CASE_CONFIG.rms_norm_eps
X_SHAPE = (T, D)
NORM_W_SHAPE = (D,)
OUTPUT_SHAPE = (T, D)
X_DTYPE = "bfloat16"
NORM_W_DTYPE = "bfloat16"
OUTPUT_DTYPE = "bfloat16"
