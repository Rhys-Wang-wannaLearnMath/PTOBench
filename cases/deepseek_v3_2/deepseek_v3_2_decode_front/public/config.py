"""Fixed fused-front workload contract (shared by Candidate, Ref, and evaluator).

The preserved official source is self-contained.  These values make its fixed
ABI and correctness semantics explicit for the v0.2 Case; implementation tile
choices remain in the editable Ref/Candidate sources.
"""
BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
NUM_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 128
KV_A_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM
CACHE_ROWS = BATCH * MAX_SEQ
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_Q_OUT = INDEX_HEADS * INDEX_HEAD_DIM
INDEX_TOPK = 2048
EP_NODES = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
INDEX_HEAD_DIM_INV = 1.0 / INDEX_HEAD_DIM
WEIGHT_SCALE = (INDEX_HEADS ** -0.5) * (INDEX_HEAD_DIM ** -0.5)
Q_LORA_INV = 1.0 / Q_LORA_RANK

# Scope 3/4 semantic constants.  SEQ_TILE is retained only because the source
# derives MAX_SEQ_BLOCKS from it; the tile itself remains implementation-owned.
SEQ_TILE = 64
MAX_SEQ_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
Q_VALID = 1
Q_PAD = 16
SORT_LEN = 8192
FP32_NEG_INF = -3.4028234663852886e38
ATTN_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

# Fixed tensor element types used by the official builder and ABI.
INPUT_DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"
INDEX_DTYPE = "int32"
STATE_DTYPE = "bfloat16"
