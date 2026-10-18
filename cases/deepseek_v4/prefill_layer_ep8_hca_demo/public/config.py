"""Fixed contract for deepseek_v4.prefill_layer_ep8_hca_demo."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Follows HuggingFace config.json"""
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
    scoring_func: Literal['softmax', 'sigmoid', 'sqrtsoftplus']
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
    dtype: Literal['bf16', 'fp8']
    scale_fmt: Optional[Literal['ue8m0']]
    expert_dtype: Optional[Literal['fp4']]
    scale_dtype: Literal['fp32', 'fp8']
    max_batch_size: int

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** (-0.5)

    @property
    def index_nope_head_dim(self) -> int:
        return self.index_head_dim - self.qk_rope_head_dim

    @property
    def index_weights_scale(self) -> float:
        return self.index_head_dim ** (-0.5) * self.index_n_heads ** (-0.5)

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


DEMO = DeepSeekV4Config(name='demo', hidden_size=4096, num_attention_heads=64, head_dim=512, qk_rope_head_dim=64, q_lora_rank=1024, o_lora_rank=1024, o_groups=8, sliding_window=128, rms_norm_eps=1e-06, vocab_size=129280, moe_intermediate_size=4096, n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=2, scoring_func='sqrtsoftplus', routed_scaling_factor=1.0, swiglu_limit=0.0, num_hidden_layers=8, num_hash_layers=0, num_nextn_predict_layers=1, compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0), index_n_heads=64, index_head_dim=128, index_topk=512, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-06, max_position_embeddings=4096, rope_theta=10000.0, compress_rope_theta=40000.0, rope_factor=40.0, beta_fast=32, beta_slow=1, original_max_position_embeddings=0, dtype='fp8', scale_fmt='ue8m0', expert_dtype=None, scale_dtype='fp8', max_batch_size=4)

CASE_CONFIG = DEMO


LAYER_ID = 3
CHUNK_LENS = (128, 192)
START_POSITIONS = None
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
MOE_TOKENS = PREFILL_TOKENS
EP_WORLD_SIZE = 8
EP_RANK = 0
N_RANKS = EP_WORLD_SIZE
RECV_MAX = EP_WORLD_SIZE * MOE_TOKENS
PREFILL_RECV_MAX = EP_WORLD_SIZE * PREFILL_TOKENS
PREFILL_LAYER_RECV_MAX = PREFILL_RECV_MAX
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts // 8 * EP_WORLD_SIZE
N_LOCAL = N_EXPERTS_GLOBAL // EP_WORLD_SIZE
TOPK = CASE_CONFIG.num_experts_per_tok
N_ROUTES = MOE_TOKENS * TOPK
MOE_INTER = CASE_CONFIG.moe_intermediate_size
VOCAB = CASE_CONFIG.vocab_size
DECODE_BATCH = 8
DECODE_SEQ = 1
BLOCK_SIZE = 128
C128_COMPRESSOR_BLOCK_SIZE = 8
C4A_COMPRESSOR_BLOCK_SIZE = 4
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
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
NOPE_HEAD_DIM = CASE_CONFIG.nope_head_dim
Q_LORA = CASE_CONFIG.q_lora_rank
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
MAX_SEQ_LEN = CASE_CONFIG.max_position_embeddings
WIN = CASE_CONFIG.sliding_window
IDX_TOPK = CASE_CONFIG.index_topk
IDX_N_HEADS = CASE_CONFIG.index_n_heads
IDX_HEAD_DIM = CASE_CONFIG.index_head_dim
HC_MULT = CASE_CONFIG.hc_mult
HC_DIM = HC_MULT * D
MIX_HC = (2 + HC_MULT) * HC_MULT
SPARSE_TOPK = CASE_CONFIG.sliding_window + CASE_CONFIG.index_topk
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
HCA_COMPRESS_RATIO = 128
HCA_MAIN_OUT_DIM = HEAD_DIM
HCA_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = HCA_STATE_MAX_BLOCKS
HCA_ORI_BLOCK_NUM = SPARSE_ORI_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
CSA_COMPRESS_RATIO = 4
OVERLAP = CSA_COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
CSA_MAIN_OUT_DIM = COFF * HEAD_DIM
CSA_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE
CSA_STATE_BLOCK_NUM = CSA_STATE_MAX_BLOCKS
CSA_ORI_BLOCK_NUM = SPARSE_ORI_MAX_BLOCKS
CSA_CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = INNER_STATE_MAX_BLOCKS
ORI_CACHE_BLOCKS = CSA_ORI_BLOCK_NUM
CMP_CACHE_BLOCKS = max(HCA_CMP_BLOCK_NUM, CSA_CMP_BLOCK_NUM)
IDX_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
ORI_TABLE_BLOCKS = SPARSE_ORI_MAX_BLOCKS
CMP_TABLE_BLOCKS = SPARSE_CMP_MAX_BLOCKS
IDX_TABLE_BLOCKS = IDX_CACHE_MAX_BLOCKS
USER_BATCH = len(CHUNK_LENS)
TOK_TILE = PREFILL_TOKENS
TILE_COUNTS = tuple((chunk + TOK_TILE - 1) // TOK_TILE for chunk in CHUNK_LENS)
PADDED_LENS = tuple(count * TOK_TILE for count in TILE_COUNTS)
CHUNK_OFFSETS = tuple(sum(PADDED_LENS[:request]) for request in range(USER_BATCH))
CHUNK_TILE_OFFSETS = tuple(sum(TILE_COUNTS[:request]) for request in range(USER_BATCH))
TOTAL_TOKENS = sum(PADDED_LENS)
LOGICAL_TOKENS = sum(CHUNK_LENS)
SEQ_LENS = CHUNK_LENS
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
BF16_DTYPE = torch.bfloat16
FP32_DTYPE = torch.float32
INT8_DTYPE = torch.int8
INT32_DTYPE = torch.int32
INT64_DTYPE = torch.int64
TENSOR_LAYOUT = "ND"
