"""Fixed contract for deepseek_v4_flash_mtp.prefill_cp_csa_cp8_flash."""
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

# Selected-source deployment and cache relationships used by the official closure.
DECODE_BATCH = 4
DECODE_SEQ = 2
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
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

# Fixed CP8 host ABI and selected golden semantics.
CP_SIZE = 8
NUM_SEGMENTS = 2 * CP_SIZE
TAIL_ROWS = 128
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HC_DIM = CASE_CONFIG.hc_dim
HC_MULT = CASE_CONFIG.hc_mult
MIX_HC = CASE_CONFIG.mix_hc
O_GROUPS = CASE_CONFIG.o_groups
O_GROUP_IN = H * CASE_CONFIG.head_dim // O_GROUPS
O_LORA = CASE_CONFIG.o_lora_rank
Q_LORA = CASE_CONFIG.q_lora_rank
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
HEAD_DIM = CASE_CONFIG.head_dim
IDX_HEAD_DIM = CASE_CONFIG.index_head_dim
IDX_N_HEADS = CASE_CONFIG.index_n_heads
T = 128
COMPRESS_RATIO = 4
STATE_LEN = 8
WIN = CASE_CONFIG.sliding_window
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_SIZE
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2
MAX_COMPRESSED_ROWS_PER_TILE = T // COMPRESS_RATIO
MAX_COMPRESSED_ROWS_PER_SEGMENT = 2 * MAX_COMPRESSED_ROWS_PER_TILE
MAIN_STATE_BLOCK_SIZE = 4
INNER_STATE_BLOCK_SIZE = 4
MAIN_STATE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
MAIN_OUT_DIM = (1 + int(COMPRESS_RATIO == 4)) * HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
INNER_OUT_DIM = (1 + int(COMPRESS_RATIO == 4)) * IDX_HEAD_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
CP_CANDIDATE_CAPACITY = 1024
SPARSE_SELECTED_WIDTH = 256
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * T
MAX_COMPRESS_LEAVES = 1 + MAX_SEGMENT_TILES
LOCAL_LEAVES = LOCAL_PARTS * MAX_COMPRESS_LEAVES
QK_ROPE_HALF = ROPE_HEAD_DIM // 2

# Physical host tensor dtypes from original.py:prefill_cp_csa_test.
# These are distinct from the selected model preset's quantization metadata.
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
    "cmp_wkv": "bfloat16",
    "cmp_wgate": "bfloat16",
    "cmp_ape": "float32",
    "cmp_norm_w": "bfloat16",
    "hadamard_idx": "bfloat16",
    "idx_wq_b": "int8",
    "idx_wq_b_scale": "float32",
    "idx_weights_proj": "bfloat16",
    "inner_wkv": "bfloat16",
    "inner_wgate": "bfloat16",
    "inner_ape": "float32",
    "inner_norm_w": "bfloat16",
    "attn_sink": "float32",
    "wo_a": "bfloat16",
    "wo_b": "int8",
    "wo_b_scale": "float32",
    "compress_state": "float32",
    "compress_state_block_table": "int32",
    "inner_compress_state": "float32",
    "inner_compress_state_block_table": "int32",
    "kv_cache": "bfloat16",
    "cmp_kv": "bfloat16",
    "cmp_block_table": "int32",
    "idx_kv_cache": "int8",
    "idx_kv_scale": "float32",
    "idx_block_table": "int32",
    "segment_starts_t": "int32",
    "segment_lengths_t": "int32",
    "segment_active_lengths": "int32",
    "owner_segments_t": "int32",
    "predecessor_segments": "int32",
    "query_positions": "int32",
    "query_requests": "int32",
    "overlay_positions": "int32",
    "overlay_requests": "int32",
    "overlay_active_lengths": "int32",
    "swa_indices": "int32",
    "final_segment_t": "int32",
    "reverse_index": "int32",
    "owner_rank_table": "int32",
    "final_win_seg_src": "int32",
    "final_win_row_src": "int32",
    "final_slot_mapping": "int32",
    "leaf_positions_input": "int32",
    "leaf_main_slots_input": "int64",
    "leaf_idx_slots_input": "int64",
    "leaf_main_state_slots_input": "int64",
    "leaf_inner_state_slots_input": "int64",
    "leaf_num_tokens_input": "int32",
    "effective_x_workspace": "bfloat16",
    "main_state_workspace0": "float32",
    "inner_state_workspace0": "float32",
    "main_state_workspace1": "float32",
    "inner_state_workspace1": "float32",
    "x_out": "float32"
}
