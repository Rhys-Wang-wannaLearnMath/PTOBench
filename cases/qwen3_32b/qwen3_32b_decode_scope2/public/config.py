"""Fixed evaluation contract for qwen3_32b.qwen3_32b_decode_scope2."""

from __future__ import annotations


# Embedded source dimensions and derived attention dimensions.
BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
HALF_DIM = HEAD_DIM // 2
CACHE_ROWS = BATCH * NUM_KV_HEADS * MAX_SEQ
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)

# Fixed source contract-shaping layout values.
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ_TILE = 64
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE

# Tensor ABI dtypes and shapes.
Q_PROJ_DTYPE = "float32"
K_PROJ_DTYPE = "float32"
V_PROJ_DTYPE = "float32"
SEQ_LENS_DTYPE = "int32"
ROPE_DTYPE = "float32"
CACHE_DTYPE = "bfloat16"
OUTPUT_DTYPE = "bfloat16"
Q_PROJ_SHAPE = (BATCH, HIDDEN)
K_PROJ_SHAPE = (BATCH, KV_HIDDEN)
V_PROJ_SHAPE = (BATCH, KV_HIDDEN)
SEQ_LENS_SHAPE = (BATCH,)
ROPE_SHAPE = (MAX_SEQ, HEAD_DIM)
CACHE_SHAPE = (CACHE_ROWS, HEAD_DIM)
OUTPUT_SHAPE = (BATCH, HIDDEN)
