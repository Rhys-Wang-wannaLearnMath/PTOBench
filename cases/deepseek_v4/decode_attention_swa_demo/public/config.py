# Copyright (c) PyPTO Contributors.
"""Fixed DEMO contract for DeepSeek-V4 sliding-window decode."""

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


# Selected complete source preset; sibling presets are separate Cases.
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

DECODE_BATCH = 8
DECODE_SEQ = 1
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
START_POS = 127
BLOCK_SIZE = 128
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4

B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
NOPE_HEAD_DIM = CASE_CONFIG.nope_head_dim
Q_LORA = CASE_CONFIG.q_lora_rank
WIN = CASE_CONFIG.sliding_window
EPS = CASE_CONFIG.rms_norm_eps
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
SPARSE_IDX_TOPK = CASE_CONFIG.index_topk
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK
SPARSE_CMP_MAX_BLOCKS = 64
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
ORI_MAX_BLOCKS = 1

BF16 = "bfloat16"
FP32 = "float32"
INT8 = "int8"
INT32 = "int32"
X_OUT_SHAPE = (B, S, HC_MULT, D)
X_OUT_DTYPE = BF16
