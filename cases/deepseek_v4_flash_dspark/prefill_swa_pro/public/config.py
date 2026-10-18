# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed PRO contract for deepseek_v4_flash_dspark.prefill_swa_pro."""

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

CASE_CONFIG = PRO

# Fixed packed prefill ABI; source expressions remain visible.
CASE_ID = "deepseek_v4_flash_dspark.prefill_swa_pro"
FIXTURE_SEED = 0
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_DIM = CASE_CONFIG.qk_rope_head_dim
ROPE_HEAD_DIM = ROPE_DIM
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = HEAD_DIM - ROPE_DIM
Q_LORA = CASE_CONFIG.q_lora_rank
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
WIN = CASE_CONFIG.sliding_window
IDX_TOPK = CASE_CONFIG.index_topk
HC_MULT = CASE_CONFIG.hc_mult
HC_DIM = CASE_CONFIG.hc_dim
MIX_HC = CASE_CONFIG.mix_hc
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
EPS = CASE_CONFIG.rms_norm_eps
HC_EPS = CASE_CONFIG.hc_eps
HC_SINKHORN_ITER = CASE_CONFIG.hc_sinkhorn_iters
SOFTMAX_SCALE = CASE_CONFIG.softmax_scale
START_POS = 0

# Physical pools are source-global and independent of the selected model context.
BLOCK_SIZE = 128
ORI_KV_BLOCK_NUM = 128
DECODE_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
PREFILL_ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_KV_BLOCK_NUM = 64
DECODE_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM
PREFILL_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
KV_CMP_MAX_BLOCKS = 32
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS

# Historical imports use the original FLASH-context logical table ceiling.
# This operand is pool metadata, not another model preset or physical capacity.
SOURCE_FLASH_POOL_MAX_POSITION_EMBEDDINGS = 16384
KV_ORI_TABLE_MAX_BLOCKS = (SOURCE_FLASH_POOL_MAX_POSITION_EMBEDDINGS + BLOCK_SIZE - 1) // BLOCK_SIZE
PREFILL_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS

# Required by byte-exact helper imports/defaults/assertions only.
DECODE_BATCH = 64
DECODE_SEQ = 2
DECODE_START_POS = 8192
C4A_COMPRESSOR_BLOCK_SIZE = 4
C128_COMPRESSOR_BLOCK_SIZE = 8

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38

# Operator storage dtypes differ from the selected model's quantization metadata.
TENSOR_DTYPES = {
    "x_hc": "float32",
    "hc_attn_fn": "float32",
    "hc_attn_scale": "float32",
    "hc_attn_base": "float32",
    "attn_norm_w": "bfloat16",
    "wq_a": "bfloat16",
    "wq_b": "int8",
    "wq_b_scale": "float32",
    "wkv": "bfloat16",
    "gamma_cq": "bfloat16",
    "gamma_ckv": "bfloat16",
    "freqs_cos": "bfloat16",
    "freqs_sin": "bfloat16",
    "kv_cache": "bfloat16",
    "block_table": "int32",
    "ori_slot_mapping": "int64",
    "position_ids": "int32",
    "attn_sink": "float32",
    "wo_a": "bfloat16",
    "wo_b": "int8",
    "wo_b_scale": "float32",
    "x_out": "float32",
}
SCALAR_DTYPES = {"num_tokens": "int32"}
