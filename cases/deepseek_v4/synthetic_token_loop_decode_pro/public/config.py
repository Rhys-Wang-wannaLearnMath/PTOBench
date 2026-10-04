"""Fixed contract for deepseek_v4.synthetic_token_loop_decode_pro.

Reduced seven-layer synthetic zero-weight Pro decode on EP2/TP2.
Original Pro attention widths; reduced MoE intermediate (1024), 192 effective experts.
Original routed-expert preset (384), scaled from EP4 deployment to EP2 (96 local experts).
"""

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
    name='pro',
    hidden_size=7168,
    num_attention_heads=128,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1536,
    o_lora_rank=1024,
    o_groups=16,
    sliding_window=128,
    rms_norm_eps=1e-06,
    vocab_size=129280,
    moe_intermediate_size=1024,
    n_routed_experts=384,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func='sqrtsoftplus',
    routed_scaling_factor=2.5,
    swiglu_limit=10.0,
    num_hidden_layers=7,
    num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(128, 128, 4, 128, 4, 128, 4, 0),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=1024,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-06,
    max_position_embeddings=16384,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    dtype='fp8',
    scale_fmt='ue8m0',
    expert_dtype='fp4',
    scale_dtype='fp8',
    max_batch_size=4,
)
CASE_CONFIG = PRO

CASE_ID = 'deepseek_v4.synthetic_token_loop_decode_pro'
ENTRY = 'l3_decode_fwd'
SOURCE_ENTRY_FILE = 'decode_fwd.py'


KERNEL_MAX_SEQ_LEN = 16384
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
DECODE_BATCH = 4
DECODE_SEQ = 2
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
DECODE_START_POS = PREFILL_TOKENS
BLOCK_SIZE = 128
C4A_COMPRESSOR_BLOCK_SIZE = 4
C128_COMPRESSOR_BLOCK_SIZE = 8
KV_ORI_TABLE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
KV_CMP_MAX_BLOCKS = 32
IDX_CACHE_MAX_BLOCKS = 64
ORI_KV_BLOCK_NUM = 128
CMP_KV_BLOCK_NUM = 32
IDX_KV_BLOCK_NUM = 64
HCA_STATE_PHYSICAL_BLOCKS = 64
CSA_STATE_PHYSICAL_BLOCKS = 65
CSA_INNER_STATE_PHYSICAL_BLOCKS = 65
DECODE_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
DECODE_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM
DECODE_IDX_BLOCK_NUM = IDX_KV_BLOCK_NUM
PREFILL_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
PREFILL_ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
PREFILL_IDX_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38
DEPLOYMENT_EP_WORLD_SIZE = 4
EP_WORLD_SIZE = 2
LM_HEAD_TP_SIZE = 2
N_RANKS = EP_WORLD_SIZE
N_EXPERTS_GLOBAL = CASE_CONFIG.n_routed_experts // DEPLOYMENT_EP_WORLD_SIZE * EP_WORLD_SIZE
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
Q_LORA = CASE_CONFIG.q_lora_rank
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
MAX_SEQ_LEN = KERNEL_MAX_SEQ_LEN
HC_MULT = CASE_CONFIG.hc_mult
HC_DIM = CASE_CONFIG.hc_dim
MIX_HC = CASE_CONFIG.mix_hc
MOE_INTER = CASE_CONFIG.moe_intermediate_size
TOPK = CASE_CONFIG.num_experts_per_tok
VOCAB = CASE_CONFIG.vocab_size
LM_HEAD_VOCAB = VOCAB
VOCAB_PER_TP = VOCAB // LM_HEAD_TP_SIZE
MAX_LOGIT_ROWS = DECODE_TOKENS
SAMPLED_IDS_PAD = 8
SWA_COMPRESS_RATIO = 0
CSA_COMPRESS_RATIO = 4
HCA_COMPRESS_RATIO = 128
MODEL_NUM_LAYERS = CASE_CONFIG.num_hidden_layers
FWD_NUM_LAYERS = MODEL_NUM_LAYERS
FWD_COMPRESS_RATIOS = CASE_CONFIG.compress_ratios[:MODEL_NUM_LAYERS]
SWA_NUM_LAYERS = FWD_COMPRESS_RATIOS.count(SWA_COMPRESS_RATIO)
CSA_NUM_LAYERS = FWD_COMPRESS_RATIOS.count(CSA_COMPRESS_RATIO)
HCA_NUM_LAYERS = FWD_COMPRESS_RATIOS.count(HCA_COMPRESS_RATIO)
CSA_OVERLAP = CSA_COMPRESS_RATIO == 4
CSA_COFF = 1 + int(CSA_OVERLAP)
HCA_MAIN_OUT_DIM = HEAD_DIM
CSA_MAIN_OUT_DIM = CSA_COFF * HEAD_DIM
IDX_HEAD_DIM = CASE_CONFIG.index_head_dim
IDX_N_HEADS = CASE_CONFIG.index_n_heads
INNER_OUT_DIM = CSA_COFF * IDX_HEAD_DIM
HCA_COMPRESS_STATE_DIM = 2 * HCA_MAIN_OUT_DIM
CSA_COMPRESS_STATE_DIM = 2 * CSA_MAIN_OUT_DIM
CSA_INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
HCA_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = HCA_STATE_PHYSICAL_BLOCKS
CSA_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE
CSA_STATE_BLOCK_NUM = CSA_STATE_PHYSICAL_BLOCKS
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = CSA_INNER_STATE_PHYSICAL_BLOCKS
CSA_ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CSA_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
CSA_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
ORI_TABLE_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
HCA_COMPRESS_STATE_BLOCK_NUM = HCA_STATE_BLOCK_NUM
HCA_COMPRESS_STATE_BLOCK_SIZE = HCA_STATE_BLOCK_SIZE
HCA_COMPRESS_STATE_MAX_BLOCKS = HCA_STATE_MAX_BLOCKS
CSA_MAIN_STATE_BLOCK_NUM = CSA_STATE_BLOCK_NUM
CSA_MAIN_STATE_BLOCK_SIZE = CSA_STATE_BLOCK_SIZE
CSA_MAIN_STATE_MAX_BLOCKS = CSA_STATE_MAX_BLOCKS
CSA_MAIN_STATE_DIM = CSA_COMPRESS_STATE_DIM
CSA_INNER_STATE_BLOCK_NUM = INNER_STATE_BLOCK_NUM
CSA_INNER_STATE_BLOCK_SIZE = INNER_STATE_BLOCK_SIZE
CSA_INNER_STATE_MAX_BLOCKS = INNER_STATE_MAX_BLOCKS
CSA_INNER_STATE_DIM = CSA_INNER_COMPRESS_STATE_DIM
CSA_INNER_OUT_DIM = INNER_OUT_DIM
CSA_IDX_HEAD_DIM = IDX_HEAD_DIM
CSA_IDX_N_HEADS = IDX_N_HEADS
CSA_IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
CSA_IDX_CACHE_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
SWA_WIN = CASE_CONFIG.sliding_window

B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
MOE_TOKENS = T
RECV_MAX = EP_WORLD_SIZE * MOE_TOKENS
START_POS = DECODE_START_POS
NUM_TOKENS = 1
INPUT_TOKEN = 0
