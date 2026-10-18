# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed configuration for deepseek_v4_flash_mtp.prefill_cp_swa_cp8_demo."""

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

# Fixed CP8 SWA prefill contract. DECODE_BATCH/SEQ remain because preserved
# HC, RMS and QKV helpers import them as compatibility operands.
DECODE_BATCH = 4
DECODE_SEQ = 2
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ

M = CASE_CONFIG
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
HC_MULT = M.hc_mult
HC_DIM = HC_MULT * D
MIX_HC = (2 + HC_MULT) * HC_MULT
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = HEAD_DIM - ROPE_DIM
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
EPS = M.rms_norm_eps
HC_EPS = M.hc_eps
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
SOFTMAX_SCALE = HEAD_DIM ** -0.5

# Frozen deployment capacity comes from the original FLASH preset, before the
# operator preset is rebound; DEMO's shorter RoPE table does not resize cache.
SOURCE_RAW_CACHE_MAX_POSITION_EMBEDDINGS = 16384
BLOCK_SIZE = 128
BLOCK_ROWS = BLOCK_SIZE
KV_ORI_TABLE_MAX_BLOCKS = (SOURCE_RAW_CACHE_MAX_POSITION_EMBEDDINGS + BLOCK_SIZE - 1) // BLOCK_SIZE
PREFILL_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_ROWS
ORI_KV_BLOCK_NUM = 128
PREFILL_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
KV_CMP_MAX_BLOCKS = 32
CMP_KV_BLOCK_NUM = 32
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38

CP_SIZE = 8
CP_CHOICES = (2, 4, 8)
CP_DEFAULT = CP_SIZE
NUM_SEGMENTS = 2 * CP_SIZE
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
TAIL_ROWS = WIN
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
CP_TAIL_WINDOW_ROWS = NUM_SEGMENTS * TAIL_ROWS
OVERLAY_BASE = ORI_CACHE_ROWS
PRED_OVERLAY_ROWS = TAIL_ROWS
CUR_OVERLAY_ROWS = TAIL_ROWS
OVERLAY_ROWS = PRED_OVERLAY_ROWS + CUR_OVERLAY_ROWS
OVERLAY_SOURCES = 2
NUM_RING_BLOCKS = ORI_CACHE_ROWS // BLOCK_ROWS
KV_DTYPE = "bfloat16"
INDEX_DTYPE = "int32"
