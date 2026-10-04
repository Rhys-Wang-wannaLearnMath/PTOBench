# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixed contract for deepseek_v4.prefill_layer_ep4_csa_flash."""

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


# Fixed EP4 CSA invocation, deployment, and operator ABI.
LAYER_ID = 2
EP_WORLD_SIZE = 4
EP_RANK = 0
DECODE_BATCH = 8
DECODE_SEQ = 1
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
MOE_TOKENS = PREFILL_TOKENS
TOKEN_TILE = PREFILL_TOKENS
USER_BATCH = 2
CHUNK_LENS = (128, 192)
START_POSITIONS = None
LOGICAL_TOKENS = sum(CHUNK_LENS)
PADDED_TOKENS = sum((length + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE for length in CHUNK_LENS)
BLOCK_SIZE = 128
C4A_COMPRESSOR_BLOCK_SIZE = 4
C128_COMPRESSOR_BLOCK_SIZE = 8
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
KV_CMP_MAX_BLOCKS = 32
IDX_CACHE_MAX_BLOCKS = 64
DECODE_CMP_BLOCK_NUM = DECODE_BATCH * KV_CMP_MAX_BLOCKS
DECODE_IDX_BLOCK_NUM = DECODE_BATCH * IDX_CACHE_MAX_BLOCKS
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
PREFILL_IDX_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
PREFILL_ORI_MAX_BLOCKS = (PREFILL_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
PREFILL_ORI_BLOCK_NUM = PREFILL_BATCH * PREFILL_ORI_MAX_BLOCKS
B = PREFILL_BATCH
S = PREFILL_SEQ
T = PREFILL_TOKENS
EPS = CASE_CONFIG.rms_norm_eps
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
NOPE_HEAD_DIM = CASE_CONFIG.nope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = CASE_CONFIG.q_lora_rank
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
WIN = CASE_CONFIG.sliding_window
IDX_N_HEADS = CASE_CONFIG.index_n_heads
IDX_HEAD_DIM = CASE_CONFIG.index_head_dim
IDX_NOPE_HEAD_DIM = CASE_CONFIG.index_nope_head_dim
IDX_TOPK = CASE_CONFIG.index_topk
WEIGHTS_SCALE = CASE_CONFIG.index_weights_scale
SOFTMAX_SCALE = CASE_CONFIG.softmax_scale
HC_MULT = CASE_CONFIG.hc_mult
HC_DIM = HC_MULT * D
MIX_HC = (2 + HC_MULT) * HC_MULT
HC_EPS = CASE_CONFIG.hc_eps
HC_SINKHORN_ITERS = CASE_CONFIG.hc_sinkhorn_iters
SPARSE_TOPK = WIN + IDX_TOPK
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
PREFILL_SPARSE_TOPK = min(SPARSE_TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
HCA_COMPRESS_RATIO = 128
HCA_MAIN_OUT_DIM = HEAD_DIM
HCA_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = HCA_STATE_MAX_BLOCKS
HCA_ORI_BLOCK_NUM = PREFILL_ORI_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
CSA_COMPRESS_RATIO = 4
OVERLAP = CSA_COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
CSA_MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_LEN = COFF * CSA_COMPRESS_RATIO
CSA_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE
CSA_STATE_BLOCK_NUM = CSA_STATE_MAX_BLOCKS
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_LEN = COFF * CSA_COMPRESS_RATIO
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = INNER_STATE_MAX_BLOCKS
IDX_KV_LEN = MAX_SEQ_LEN // CSA_COMPRESS_RATIO
PREFILL_COMPRESSED_LEN = S // CSA_COMPRESS_RATIO
INDEXER_SCORE_MAX_BLOCKS = 2
INDEXER_SCORE_CAP = INDEXER_SCORE_MAX_BLOCKS * BLOCK_SIZE
INDEXER_TOPK_CAP = min(IDX_TOPK, INDEXER_SCORE_CAP)
INDEXER_OFFSET = WIN + T
MAX_CMP_WRITES = max(1, T // CSA_COMPRESS_RATIO)
TOPK = CASE_CONFIG.num_experts_per_tok
VOCAB = CASE_CONFIG.vocab_size
MOE_INTER = CASE_CONFIG.moe_intermediate_size
N_RANKS = EP_WORLD_SIZE
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts // 8 * EP_WORLD_SIZE
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
N_ROUTES = MOE_TOKENS * TOPK
RECV_MAX = EP_WORLD_SIZE * MOE_TOKENS
PREFILL_LAYER_RECV_MAX = RECV_MAX
ACTIVATION_DTYPE = OUTPUT_DTYPE = KV_CACHE_DTYPE = "bfloat16"
STATE_DTYPE = INDEX_SCALE_DTYPE = "float32"
INDEX_CACHE_DTYPE = "int8"
BLOCK_TABLE_DTYPE = LAYER_ID_DTYPE = "int32"
SLOT_MAPPING_DTYPE = "int64"
