"""Fixed contract for the self-contained legacy DeepSeek-V4 ratio-128 decoder."""

# Source constants and derived values defining the selected workload.
B = 16
S = 1
D = 4096
EPS = 1e-6
COMPRESS_RATIO = 128
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
ROTATE = False
OVERLAP = False
COFF = 1
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
START_POS = 127
SHOULD_COMPRESS = COMPRESS_RATIO != 0 and ((START_POS + 1) % COMPRESS_RATIO) == 0
APE_ROW = START_POS % COMPRESS_RATIO
SCATTER_SLOT = APE_ROW
HEAD_DIM_INV = 1.0 / HEAD_DIM

# Fixed tensor and scalar ABI. Launch tiling remains implementation-owned.
X_SHAPE = (B, S, D)
STATE_SHAPE = (B, STATE_LEN, OUT_DIM)
W_SHAPE = (OUT_DIM, D)
APE_SHAPE = (COMPRESS_RATIO, OUT_DIM)
NORM_W_SHAPE = (HEAD_DIM,)
ROPE_SHAPE = (1, ROPE_HEAD_DIM // 2)
HADAMARD_SHAPE = (HEAD_DIM, HEAD_DIM)
OUT_SHAPE = (B, HEAD_DIM)
X_DTYPE = "bfloat16"
STATE_DTYPE = "float32"
W_DTYPE = "bfloat16"
APE_DTYPE = "float32"
NORM_W_DTYPE = "bfloat16"
ROPE_DTYPE = "bfloat16"
HADAMARD_DTYPE = "bfloat16"
OUT_DTYPE = "bfloat16"
START_POS_DTYPE = "int32"
