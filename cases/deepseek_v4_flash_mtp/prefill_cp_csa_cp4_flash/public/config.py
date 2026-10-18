"""Fixed contract for deepseek_v4_flash_mtp.prefill_cp_csa_cp4_flash."""

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

# Fixed CP4 CSA operator contract.
D = CASE_CONFIG.hidden_size
H = CASE_CONFIG.num_attention_heads
HEAD_DIM = CASE_CONFIG.head_dim
ROPE_HEAD_DIM = CASE_CONFIG.qk_rope_head_dim
Q_LORA = CASE_CONFIG.q_lora_rank
O_LORA = CASE_CONFIG.o_lora_rank
O_GROUPS = CASE_CONFIG.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
HC_MULT = CASE_CONFIG.hc_mult
HC_DIM = CASE_CONFIG.hc_dim
MIX_HC = CASE_CONFIG.mix_hc
IDX_N_HEADS = CASE_CONFIG.index_n_heads
IDX_HEAD_DIM = CASE_CONFIG.index_head_dim
IDX_TOPK = CASE_CONFIG.index_topk
WIN = CASE_CONFIG.sliding_window
EPS = CASE_CONFIG.rms_norm_eps
HC_EPS = CASE_CONFIG.hc_eps
HC_SINKHORN_ITER = CASE_CONFIG.hc_sinkhorn_iters

CP_SIZE = 4
NUM_SEGMENTS = 2 * CP_SIZE
EPOCHS = 1
T = 128
TAIL_ROWS = 128
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * T
CP_TAIL_WINDOW_ROWS = NUM_SEGMENTS * TAIL_ROWS
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2
QK_ROPE_HALF = ROPE_HEAD_DIM // 2

PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM

BLOCK_SIZE = 128
C4A_COMPRESSOR_BLOCK_SIZE = 4
MAIN_STATE_BLOCK_SIZE = 4
INNER_STATE_BLOCK_SIZE = 4
CSA_STATE_PHYSICAL_BLOCKS = 65
CSA_INNER_STATE_PHYSICAL_BLOCKS = 65
MAIN_STATE_ROWS = CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE
INNER_STATE_ROWS = CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE
MAIN_STATE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE

KV_ORI_TABLE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
PREFILL_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_SIZE
KV_CMP_MAX_BLOCKS = 32
IDX_CACHE_MAX_BLOCKS = 64
ORI_KV_BLOCK_NUM = 128
CMP_KV_BLOCK_NUM = 32
IDX_KV_BLOCK_NUM = 64
# The child Fixture builder uses this physical pool for its unused spec draws.
PREFILL_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM
PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
PREFILL_IDX_BLOCK_NUM = IDX_KV_BLOCK_NUM
MAIN_LEAF_CACHE_BLOCKS = PREFILL_CMP_BLOCK_NUM
MAIN_LEAF_CACHE_ROWS = MAIN_LEAF_CACHE_BLOCKS * BLOCK_SIZE
IDX_LEAF_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
IDX_LEAF_CACHE_ROWS = IDX_LEAF_CACHE_BLOCKS * BLOCK_SIZE
MAIN_CACHE_ROWS = PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE
IDX_CACHE_ROWS = PREFILL_IDX_BLOCK_NUM * BLOCK_SIZE

MAX_SEED_ROWS = COMPRESS_RATIO + 3
MAX_COMPRESSED_ROWS_PER_TILE = T // COMPRESS_RATIO
MAX_COMPRESSED_ROWS_PER_SEGMENT = 2 * MAX_COMPRESSED_ROWS_PER_TILE
MAX_COMPRESS_LEAVES = 1 + MAX_SEGMENT_TILES
LOCAL_LEAVES = LOCAL_PARTS * MAX_COMPRESS_LEAVES
CP_CANDIDATE_CAPACITY = 1024
CP_INDEXER_SCORE_CAP = 1024
CP_INDEXER_SORT_LEN = 2048
SPARSE_SELECTED_WIDTH = 256

ROWS_PER_RANK = 128
STATE_ROWS_PER_RANK = 8
META_DIM = 8
STATE_META_DIM = 8
RECORDS_PER_WINDOW = CP_SIZE * ROWS_PER_RANK
STATE_RECORDS_PER_WINDOW = CP_SIZE * STATE_ROWS_PER_RANK
SCALE_TILE_COLS = 8

TOPK = WIN + IDX_TOPK
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
PREFILL_SPARSE_TOPK = min(TOPK, min(WIN, PREFILL_SEQ) + PREFILL_MAX_COMPRESSED)
# This padding fixes the sparse ABI; it is not an implementation tile choice.
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
MASK_ALIGN_ELEMS = 32 // 4
VALID_BLOCK_MASK_COLS = (PREFILL_ATTN_BLOCKS + MASK_ALIGN_ELEMS - 1) // MASK_ALIGN_ELEMS * MASK_ALIGN_ELEMS
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38

# Ordered host ABI: name -> (shape, dtype, direction), from the selected source.
# Dense contiguous ND tensors; all inactive output rows are defined.
TENSOR_ABI = {
    'x_hc': ([CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], 'float32', 'input'),
    'hc_attn_fn': ([MIX_HC, HC_DIM], 'float32', 'input'),
    'hc_attn_scale': ([3], 'float32', 'input'),
    'hc_attn_base': ([MIX_HC], 'float32', 'input'),
    'attn_norm_w': ([D], 'bfloat16', 'input'),
    'wq_a': ([D, Q_LORA], 'bfloat16', 'input'),
    'wq_b': ([Q_LORA, H * HEAD_DIM], 'int8', 'input'),
    'wq_b_scale': ([H * HEAD_DIM], 'float32', 'input'),
    'wkv': ([D, HEAD_DIM], 'bfloat16', 'input'),
    'gamma_cq': ([Q_LORA], 'bfloat16', 'input'),
    'gamma_ckv': ([HEAD_DIM], 'bfloat16', 'input'),
    'freqs_cos': ([CASE_CONFIG.max_position_embeddings, ROPE_HEAD_DIM], 'bfloat16', 'input'),
    'freqs_sin': ([CASE_CONFIG.max_position_embeddings, ROPE_HEAD_DIM], 'bfloat16', 'input'),
    'cmp_wkv': ([MAIN_OUT_DIM, D], 'bfloat16', 'input'),
    'cmp_wgate': ([MAIN_OUT_DIM, D], 'bfloat16', 'input'),
    'cmp_ape': ([COMPRESS_RATIO, MAIN_OUT_DIM], 'float32', 'input'),
    'cmp_norm_w': ([HEAD_DIM], 'bfloat16', 'input'),
    'hadamard_idx': ([IDX_HEAD_DIM, IDX_HEAD_DIM], 'bfloat16', 'input'),
    'idx_wq_b': ([Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], 'int8', 'input'),
    'idx_wq_b_scale': ([IDX_N_HEADS * IDX_HEAD_DIM], 'float32', 'input'),
    'idx_weights_proj': ([D, IDX_N_HEADS], 'bfloat16', 'input'),
    'inner_wkv': ([INNER_OUT_DIM, D], 'bfloat16', 'input'),
    'inner_wgate': ([INNER_OUT_DIM, D], 'bfloat16', 'input'),
    'inner_ape': ([COMPRESS_RATIO, INNER_OUT_DIM], 'float32', 'input'),
    'inner_norm_w': ([IDX_HEAD_DIM], 'bfloat16', 'input'),
    'attn_sink': ([H], 'float32', 'input'),
    'wo_a': ([O_GROUPS, O_LORA, O_GROUP_IN], 'bfloat16', 'input'),
    'wo_b': ([D, O_GROUPS * O_LORA], 'int8', 'input'),
    'wo_b_scale': ([D], 'float32', 'input'),
    'compress_state': ([CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], 'float32', 'inout'),
    'compress_state_block_table': ([CP_SIZE, MAIN_STATE_MAX_BLOCKS], 'int32', 'input'),
    'inner_compress_state': ([CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], 'float32', 'inout'),
    'inner_compress_state_block_table': ([CP_SIZE, INNER_STATE_MAX_BLOCKS], 'int32', 'input'),
    'kv_cache': ([CP_SIZE, ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], 'bfloat16', 'inout'),
    'cmp_kv': ([CP_SIZE, PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], 'bfloat16', 'inout'),
    'cmp_block_table': ([CP_SIZE, PREFILL_CMP_MAX_BLOCKS], 'int32', 'input'),
    'idx_kv_cache': ([CP_SIZE, PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], 'int8', 'inout'),
    'idx_kv_scale': ([CP_SIZE, PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], 'float32', 'inout'),
    'idx_block_table': ([CP_SIZE, IDX_CACHE_MAX_BLOCKS], 'int32', 'input'),
    'segment_starts_t': ([NUM_SEGMENTS], 'int32', 'input'),
    'segment_lengths_t': ([NUM_SEGMENTS], 'int32', 'input'),
    'segment_active_lengths': ([CP_SIZE, LOCAL_PARTS], 'int32', 'input'),
    'owner_segments_t': ([CP_SIZE, LOCAL_PARTS], 'int32', 'input'),
    'predecessor_segments': ([CP_SIZE, LOCAL_PARTS], 'int32', 'input'),
    'query_positions': ([CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], 'int32', 'input'),
    'query_requests': ([CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], 'int32', 'input'),
    'overlay_positions': ([CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], 'int32', 'input'),
    'overlay_requests': ([CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], 'int32', 'input'),
    'overlay_active_lengths': ([CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], 'int32', 'input'),
    'swa_indices': ([CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], 'int32', 'input'),
    'final_segment_t': ([1], 'int32', 'input'),
    'reverse_index': ([NUM_SEGMENTS], 'int32', 'input'),
    'owner_rank_table': ([NUM_SEGMENTS], 'int32', 'input'),
    'final_win_seg_src': ([T], 'int32', 'input'),
    'final_win_row_src': ([T], 'int32', 'input'),
    'final_slot_mapping': ([T], 'int32', 'input'),
    'leaf_positions_input': ([CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], 'int32', 'input'),
    'leaf_main_slots_input': ([CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], 'int64', 'input'),
    'leaf_idx_slots_input': ([CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], 'int64', 'input'),
    'leaf_main_state_slots_input': ([CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], 'int64', 'input'),
    'leaf_inner_state_slots_input': ([CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], 'int64', 'input'),
    'leaf_num_tokens_input': ([CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES], 'int32', 'input'),
    'effective_x_workspace': ([CP_SIZE, LOCAL_LEAVES * T, D], 'bfloat16', 'input'),
    'main_state_workspace0': ([CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], 'float32', 'input'),
    'inner_state_workspace0': ([CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], 'float32', 'input'),
    'main_state_workspace1': ([CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], 'float32', 'input'),
    'inner_state_workspace1': ([CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], 'float32', 'input'),
    'x_out': ([CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], 'float32', 'output'),
}
