from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Complete source-shaped DeepSeek V4 configuration for this Case preset."""
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

FLASH = DeepSeekV4Config(name="flash", hidden_size=4096, num_attention_heads=64, head_dim=512, qk_rope_head_dim=64, q_lora_rank=1024, o_lora_rank=1024, o_groups=8, sliding_window=128, rms_norm_eps=1e-6, vocab_size=129280, moe_intermediate_size=2048, n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=6, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=10.0, num_hidden_layers=43, num_hash_layers=3, num_nextn_predict_layers=1, compress_ratios=(0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0), index_n_heads=64, index_head_dim=128, index_topk=512, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, max_position_embeddings=16384, rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=16.0, beta_fast=32, beta_slow=1, original_max_position_embeddings=65536, dtype="fp8", scale_fmt="ue8m0", expert_dtype="fp4", scale_dtype="fp8", max_batch_size=4)
CASE_CONFIG = FLASH

# Fixed prefill CP-CSA operator and ABI bindings. Implementation tiles remain in Ref/Candidate.
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
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e+38

CP_SIZE = 2
EPOCHS = 1
TAIL_ROWS = 128
NUM_SEGMENTS = 2 * CP_SIZE
CP_TAIL_WINDOW_ROWS = NUM_SEGMENTS * TAIL_ROWS
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ
T = 128
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * T
MAX_COMPRESS_LEAVES = 1 + MAX_SEGMENT_TILES
LOCAL_LEAVES = LOCAL_PARTS * MAX_COMPRESS_LEAVES
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2

COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
STATE_LEN = COFF * COMPRESS_RATIO
MAIN_OUT_DIM = COFF * HEAD_DIM
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
MAIN_STATE_BLOCK_SIZE = 4
INNER_STATE_BLOCK_SIZE = 4
CSA_STATE_PHYSICAL_BLOCKS = 65
CSA_INNER_STATE_PHYSICAL_BLOCKS = 65
MAIN_STATE_ROWS = CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE
INNER_STATE_ROWS = CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE
MAIN_STATE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE

BLOCK_SIZE = 128
KV_ORI_TABLE_MAX_BLOCKS = (CASE_CONFIG.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
PREFILL_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_SIZE
KV_CMP_MAX_BLOCKS = 32
IDX_CACHE_MAX_BLOCKS = 64
CMP_KV_BLOCK_NUM = 32
IDX_KV_BLOCK_NUM = 64
PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
PREFILL_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM
PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
PREFILL_IDX_BLOCK_NUM = IDX_KV_BLOCK_NUM
ORI_KV_BLOCK_NUM = 128
PREFILL_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM

TENSOR_LAYOUT = "ND"
ABI_SHAPES = {
    "x_hc": (CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D),
    "hc_attn_fn": (MIX_HC, HC_DIM), "hc_attn_scale": (3,), "hc_attn_base": (MIX_HC,),
    "attn_norm_w": (D,), "wq_a": (D, Q_LORA), "wq_b": (Q_LORA, H * HEAD_DIM),
    "wq_b_scale": (H * HEAD_DIM,), "wkv": (D, HEAD_DIM), "gamma_cq": (Q_LORA,), "gamma_ckv": (HEAD_DIM,),
    "freqs_cos": (CASE_CONFIG.max_position_embeddings, ROPE_HEAD_DIM), "freqs_sin": (CASE_CONFIG.max_position_embeddings, ROPE_HEAD_DIM),
    "cmp_wkv": (MAIN_OUT_DIM, D), "cmp_wgate": (MAIN_OUT_DIM, D), "cmp_ape": (COMPRESS_RATIO, MAIN_OUT_DIM),
    "cmp_norm_w": (HEAD_DIM,), "hadamard_idx": (IDX_HEAD_DIM, IDX_HEAD_DIM),
    "idx_wq_b": (Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM), "idx_wq_b_scale": (IDX_N_HEADS * IDX_HEAD_DIM,),
    "idx_weights_proj": (D, IDX_N_HEADS), "inner_wkv": (INNER_OUT_DIM, D), "inner_wgate": (INNER_OUT_DIM, D),
    "inner_ape": (COMPRESS_RATIO, INNER_OUT_DIM), "inner_norm_w": (IDX_HEAD_DIM,), "attn_sink": (H,),
    "wo_a": (O_GROUPS, O_LORA, O_GROUP_IN), "wo_b": (D, O_GROUPS * O_LORA), "wo_b_scale": (D,),
    "compress_state": (CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM),
    "compress_state_block_table": (CP_SIZE, MAIN_STATE_MAX_BLOCKS),
    "inner_compress_state": (CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM),
    "inner_compress_state_block_table": (CP_SIZE, INNER_STATE_MAX_BLOCKS),
    "kv_cache": (CP_SIZE, ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
    "cmp_kv": (CP_SIZE, PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM),
    "cmp_block_table": (CP_SIZE, PREFILL_CMP_MAX_BLOCKS),
    "idx_kv_cache": (CP_SIZE, PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM),
    "idx_kv_scale": (CP_SIZE, PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1),
    "idx_block_table": (CP_SIZE, IDX_CACHE_MAX_BLOCKS),
    "segment_starts_t": (NUM_SEGMENTS,), "segment_lengths_t": (NUM_SEGMENTS,), "segment_active_lengths": (CP_SIZE, LOCAL_PARTS),
    "owner_segments_t": (CP_SIZE, LOCAL_PARTS), "predecessor_segments": (CP_SIZE, LOCAL_PARTS),
    "query_positions": (CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T), "query_requests": (CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T),
    "overlay_positions": (CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS), "overlay_requests": (CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS),
    "overlay_active_lengths": (CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES),
    "swa_indices": (CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN), "final_segment_t": (1,), "reverse_index": (NUM_SEGMENTS,),
    "owner_rank_table": (NUM_SEGMENTS,), "final_win_seg_src": (T,), "final_win_row_src": (T,), "final_slot_mapping": (T,),
    "leaf_positions_input": (CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T),
    "leaf_main_slots_input": (CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T), "leaf_idx_slots_input": (CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T),
    "leaf_main_state_slots_input": (CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T), "leaf_inner_state_slots_input": (CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T),
    "leaf_num_tokens_input": (CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES), "effective_x_workspace": (CP_SIZE, LOCAL_LEAVES * T, D),
    "main_state_workspace0": (CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM),
    "inner_state_workspace0": (CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM),
    "main_state_workspace1": (CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM),
    "inner_state_workspace1": (CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM),
    "x_out": (CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D),
}
ABI_DTYPES = {
    "x_hc":"float32", "hc_attn_fn":"float32", "hc_attn_scale":"float32", "hc_attn_base":"float32", "attn_norm_w":"bfloat16",
    "wq_a":"bfloat16", "wq_b":"int8", "wq_b_scale":"float32", "wkv":"bfloat16", "gamma_cq":"bfloat16", "gamma_ckv":"bfloat16",
    "freqs_cos":"bfloat16", "freqs_sin":"bfloat16", "cmp_wkv":"bfloat16", "cmp_wgate":"bfloat16", "cmp_ape":"float32", "cmp_norm_w":"bfloat16",
    "hadamard_idx":"bfloat16", "idx_wq_b":"int8", "idx_wq_b_scale":"float32", "idx_weights_proj":"bfloat16", "inner_wkv":"bfloat16", "inner_wgate":"bfloat16", "inner_ape":"float32", "inner_norm_w":"bfloat16", "attn_sink":"float32", "wo_a":"bfloat16", "wo_b":"int8", "wo_b_scale":"float32",
    "compress_state":"float32", "compress_state_block_table":"int32", "inner_compress_state":"float32", "inner_compress_state_block_table":"int32", "kv_cache":"bfloat16", "cmp_kv":"bfloat16", "cmp_block_table":"int32", "idx_kv_cache":"int8", "idx_kv_scale":"float32", "idx_block_table":"int32",
    "segment_starts_t":"int32", "segment_lengths_t":"int32", "segment_active_lengths":"int32", "owner_segments_t":"int32", "predecessor_segments":"int32", "query_positions":"int32", "query_requests":"int32", "overlay_positions":"int32", "overlay_requests":"int32", "overlay_active_lengths":"int32", "swa_indices":"int32", "final_segment_t":"int32", "reverse_index":"int32", "owner_rank_table":"int32", "final_win_seg_src":"int32", "final_win_row_src":"int32", "final_slot_mapping":"int32", "leaf_positions_input":"int32", "leaf_main_slots_input":"int64", "leaf_idx_slots_input":"int64", "leaf_main_state_slots_input":"int64", "leaf_inner_state_slots_input":"int64", "leaf_num_tokens_input":"int32", "effective_x_workspace":"bfloat16", "main_state_workspace0":"float32", "inner_state_workspace0":"float32", "main_state_workspace1":"float32", "inner_state_workspace1":"float32", "x_out":"float32",
}
