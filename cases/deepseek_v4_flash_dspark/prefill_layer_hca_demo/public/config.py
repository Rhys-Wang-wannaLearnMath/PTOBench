"""Fixed evaluation contract for deepseek_v4_flash_dspark.prefill_layer_hca_demo."""

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 configuration"""

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

# Fixed single-layer operator and deployment bindings.
DECODE_BATCH = 64
DSPARK_SPEC_TOKENS = 7
DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
DP = 4
SOURCE_EP = 16
MOE_TOKENS = DECODE_TOKENS * DP // SOURCE_EP
EP = 2
TP_SIZE = 2
N_RANKS = EP
T = MOE_TOKENS
GROUP_TOKENS = TP_SIZE * T
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts // SOURCE_EP * EP
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
MAX_PER_SRC = T
RECV_MAX = EP * MOE_TOKENS
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = CASE_CONFIG.nope_head_dim
Q_LORA = CASE_CONFIG.q_lora_rank
MOE_INTER = CASE_CONFIG.moe_intermediate_size
TOPK = CASE_CONFIG.num_experts_per_tok
N_ROUTES = T * TOPK
VOCAB = CASE_CONFIG.vocab_size
HC_MULT = CASE_CONFIG.hc_mult
HC_DIM = CASE_CONFIG.hc_dim
MIX_HC = CASE_CONFIG.mix_hc
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = CASE_CONFIG.hc_sinkhorn_iters
HC_EPS = CASE_CONFIG.hc_eps
NORM_EPS = CASE_CONFIG.rms_norm_eps
SWIGLU_LIMIT = CASE_CONFIG.swiglu_limit
ROUTE_SCALE = CASE_CONFIG.routed_scaling_factor
N_HASH_LAYERS = CASE_CONFIG.num_hash_layers
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
WIN = CASE_CONFIG.sliding_window
SOFTMAX_SCALE = CASE_CONFIG.softmax_scale
IDX_N_HEADS = CASE_CONFIG.index_n_heads
IDX_HEAD_DIM = CASE_CONFIG.index_head_dim
IDX_TOPK = CASE_CONFIG.index_topk
IDX_NOPE_HEAD_DIM = CASE_CONFIG.index_nope_head_dim
WEIGHTS_SCALE = CASE_CONFIG.index_weights_scale
BLOCK_SIZE = 32
C4A_COMPRESSOR_BLOCK_SIZE = 2
C128_COMPRESSOR_BLOCK_SIZE = 8
KV_ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_CMP_MAX_BLOCKS = (MAX_SEQ_LEN // 4 + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_CACHE_MAX_BLOCKS = 2 * KV_CMP_MAX_BLOCKS
KV_ORI_BLOCK_NUM = 512
KV_CMP_BLOCK_NUM = 256
IDX_CACHE_BLOCK_NUM = 256
HCA_STATE_PHYSICAL_BLOCKS = 256
CSA_STATE_PHYSICAL_BLOCKS = 260
CSA_INNER_STATE_PHYSICAL_BLOCKS = 260
HCA_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = HCA_STATE_PHYSICAL_BLOCKS
CSA_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE
CSA_STATE_BLOCK_NUM = CSA_STATE_PHYSICAL_BLOCKS
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = CSA_INNER_STATE_PHYSICAL_BLOCKS
HCA_COMPRESS_RATIO = 128
HCA_MAIN_OUT_DIM = HEAD_DIM
HCA_COMPRESS_STATE_DIM = 2 * HCA_MAIN_OUT_DIM
HCA_STATE_LEN = HCA_COMPRESS_RATIO
CSA_COMPRESS_RATIO = 4
CSA_OVERLAP = CSA_COMPRESS_RATIO == 4
CSA_COFF = 1 + int(CSA_OVERLAP)
CSA_MAIN_OUT_DIM = CSA_COFF * HEAD_DIM
CSA_COMPRESS_STATE_DIM = 2 * CSA_MAIN_OUT_DIM
CSA_STATE_LEN = CSA_COFF * CSA_COMPRESS_RATIO
INNER_OUT_DIM = CSA_COFF * IDX_HEAD_DIM
INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
INNER_STATE_LEN = CSA_COFF * CSA_COMPRESS_RATIO
SPARSE_ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CSA_ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CSA_CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM
HCA_CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
O_PROJ_LOCAL_GROUPS = O_GROUPS // TP_SIZE
O_PROJ_LOCAL_COLS = O_PROJ_LOCAL_GROUPS * O_LORA
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
START_POS = 0
FIXTURE_SEED = 20260825
# Each result appends its single LAYER_ID, ATTENTION_KIND and compression ratio.
CASE_ID = 'deepseek_v4_flash_dspark.prefill_layer_hca_demo'
ATTENTION_KIND = 'hca'
LAYER_ID = 3
COMPRESS_RATIO = CASE_CONFIG.compress_ratios[LAYER_ID]
ORI_BLOCK_TABLE_COLS = KV_ORI_BLOCK_NUM if ATTENTION_KIND == "swa" else KV_ORI_MAX_BLOCKS
MODEL_CONFIG = replace(CASE_CONFIG, n_routed_experts=N_EXPERTS_GLOBAL)
HEAD_DIM_INV = 1.0 / HEAD_DIM
NOPE_DIM = NOPE_HEAD_DIM
ROPE_DIM = ROPE_HEAD_DIM
EPS = NORM_EPS

# Tensor ABI uses these concrete storage types, independent of model precision labels.
FP32_DTYPE = "float32"
BF16_DTYPE = "bfloat16"
INT8_DTYPE = "int8"
INT32_DTYPE = "int32"
INT64_DTYPE = "int64"
TENSOR_LAYOUT = "ND"
