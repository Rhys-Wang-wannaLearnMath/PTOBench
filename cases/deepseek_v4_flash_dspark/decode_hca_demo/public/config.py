# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed DEMO contract for deepseek_v4_flash_dspark.decode_hca_demo."""

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
CASE_ID = "deepseek_v4_flash_dspark.decode_hca_demo"

# Fixed one-step HCA operator contract; implementation tuning stays in runtime code.
DECODE_BATCH = 64
DECODE_SEQ = 2
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
DECODE_START_POS = 8192
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S

D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
HALF_ROPE = ROPE_DIM // 2
NOPE_HEAD_DIM = CASE_CONFIG.nope_head_dim
Q_LORA = CASE_CONFIG.q_lora_rank
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
WIN = CASE_CONFIG.sliding_window
EPS = CASE_CONFIG.rms_norm_eps
SOFTMAX_SCALE = CASE_CONFIG.softmax_scale
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM = CASE_CONFIG.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = CASE_CONFIG.hc_sinkhorn_iters
HC_EPS = CASE_CONFIG.hc_eps

BLOCK_SIZE = 128
C128_COMPRESSOR_BLOCK_SIZE = 8
COMPRESS_RATIO = 128
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
HEAD_DIM_INV = 1.0 / HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO

# Authorized operator-local context; nominal DEMO model context remains 4096.
HCA_MAX_SEQ_LEN = 1_048_576
KV_ORI_TABLE_MAX_BLOCKS = (HCA_MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
# Preserve independent physical ownership of the original raw KV pool.
ORI_KV_BLOCK_NUM = 128
KV_ORI_MAX_BLOCKS = ORI_KV_BLOCK_NUM
KV_CMP_MAX_BLOCKS = (HCA_MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
CMP_KV_BLOCK_NUM = DECODE_BATCH * KV_CMP_MAX_BLOCKS

MAX_SEQ_LEN = HCA_MAX_SEQ_LEN
COMPRESS_TOPK = MAX_SEQ_LEN // COMPRESS_RATIO
HCA_TOPK_LIMIT = COMPRESS_TOPK
HCA_CMP_TOPK = COMPRESS_TOPK
CMP_TOPK = HCA_CMP_TOPK
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_CAPACITY = MAX_SEQ_LEN // COMPRESS_RATIO
HCA_STATE_PHYSICAL_BLOCKS = 64
COMPRESS_STATE_PHYSICAL_BLOCKS = HCA_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
DECODE_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
DECODE_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
SPARSE_NEG_INF = -1.0e20
BF16 = "bfloat16"
FP32 = "float32"
INT8 = "int8"
INT32 = "int32"
INT64 = "int64"
X_OUT_SHAPE = (T, HC_MULT, D)
X_OUT_DTYPE = FP32
KV_CACHE_SHAPE = (DECODE_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
KV_CACHE_DTYPE = BF16
