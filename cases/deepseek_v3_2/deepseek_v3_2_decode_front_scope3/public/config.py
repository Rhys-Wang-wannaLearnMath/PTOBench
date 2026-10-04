"""Fixed evaluation contract for DeepSeek V3.2 decode front Scope 3."""

# Fixed ABI dimensions and derived row counts from the official source.
BATCH = 16
MAX_SEQ = 4096
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_Q_ROWS = BATCH * INDEX_HEADS
INDEX_TOPK = 2048
CACHE_ROWS_IDX = BATCH * MAX_SEQ

# Fixed tiling and padding semantics used by the official implementation.
SEQ_TILE = 64
MAX_SEQ_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
Q_VALID = 1
Q_PAD = 16

# Fixed sort capacity and score sentinel are part of the observable contract.
SORT_LEN = 8192
MRGSORT_ITERS = 4
FP32_NEG_INF = -3.4028234663852886e38

# Input and output dtypes for the eight-tensor opaque entry.
Q_INDEX_DTYPE = "int8"
SCALE_DTYPE = "float32"
WEIGHTS_DTYPE = "float32"
SEQ_LENS_DTYPE = "int32"
TOPK_VALUES_DTYPE = "float32"
TOPK_INDICES_DTYPE = "int32"
