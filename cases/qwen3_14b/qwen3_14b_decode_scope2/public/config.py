"""Fixed evaluation contract for qwen3_14b.qwen3_14b_decode_scope2."""

from __future__ import annotations


# Model and ABI dimensions from the self-contained official source.
BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

# RMSNorm, RoPE, and attention semantics.
EPS = 1e-6
HEAD_DIM_INV = 1.0 / HEAD_DIM
HALF_DIM = HEAD_DIM // 2
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)

# Fixed grouped-query structure and numeric padding behavior.
Q_HEAD_BATCH = 5
Q_HEAD_PAD = 16
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS

# Paged-cache ABI. SEQ_TILE is the source operand that fixes block size.
SEQ_TILE = 64
BLOCK_SIZE = SEQ_TILE
MAX_CTX_BLOCKS = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
MAX_BLOCKS_PER_SEQ = MAX_CTX_BLOCKS
NUM_BLOCKS = BATCH * MAX_BLOCKS_PER_SEQ
CACHE_ROWS = NUM_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE

# Fixed tensor dtypes, grouped by their ABI roles.
PROJECTION_INPUT_DTYPE = "float32"  # q_proj, k_proj, v_proj
NORM_WEIGHT_DTYPE = "float32"  # q_norm_weight, k_norm_weight
INDEX_DTYPE = "int32"  # seq_lens, block_table, slot_mapping
ROPE_DTYPE = "float32"  # rope_cos, rope_sin
CACHE_DTYPE = "bfloat16"  # k_cache, v_cache
ATTENTION_OUTPUT_DTYPE = "bfloat16"  # attn_out
