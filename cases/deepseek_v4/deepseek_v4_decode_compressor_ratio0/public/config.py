"""Fixed contract for the self-contained DeepSeek-V4 ratio-0 compressor."""

# Source constants that define the fixed workload ABI.
B = 16
S = 1
D = 4096
EPS = 1e-6
COMPRESS_RATIO = 0
HEAD_DIM = 512
ROTATE = False
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
OVERLAP = False
COFF = 1
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = 1
START_POS = 0

# Fixed tensor contract. HEAD_CHUNK and related launch tiling remain in the
# editable implementation and are intentionally not part of this config.
X_SHAPE = (B, S, D)
STATE_SHAPE = (B, 1, HEAD_DIM)
W_SHAPE = (HEAD_DIM, D)
APE_SHAPE = (1, HEAD_DIM)
ROPE_SHAPE = (1, ROPE_HEAD_DIM // 2)
HADAMARD_SHAPE = (HEAD_DIM, HEAD_DIM)
OUT_SHAPE = (B, HEAD_DIM)
X_DTYPE = "bfloat16"
STATE_DTYPE = "float32"
W_DTYPE = "bfloat16"
APE_DTYPE = "float32"
ROPE_DTYPE = "bfloat16"
HADAMARD_DTYPE = "bfloat16"
OUT_DTYPE = "bfloat16"
START_POS_DTYPE = "int32"
