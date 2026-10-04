"""Fixed contract for the DeepSeek-V4 legacy T=16 Q/KV projection Case."""

# Decode and hidden dimensions from the embedded legacy source contract.
B = 16
S = 1
T = B * S
D = 4096
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = HEAD_DIM - ROPE_DIM
Q_LORA = 1024
EPS = 1e-6

# The Q projection scale tensor keeps the source's [256, 128] ABI.
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK

# Fixed quantization semantics.
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4

# Input shapes and dtypes.
X_SHAPE = (B, S, D)
NORM_W_SHAPE = (D,)
WQ_A_SHAPE = (D, Q_LORA)
WQ_B_SHAPE = (Q_LORA, H * HEAD_DIM)
WQ_B_SCALE_SHAPE = (Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK)
WKV_SHAPE = (D, HEAD_DIM)
ROPE_SHAPE = (T, ROPE_DIM)
SELECTOR_SHAPE = (ROPE_HALF, ROPE_DIM)
GAMMA_CQ_SHAPE = (Q_LORA,)
GAMMA_CKV_SHAPE = (HEAD_DIM,)

Q_SHAPE = (T, H, HEAD_DIM)
KV_SHAPE = (T, HEAD_DIM)
QR_SHAPE = (T, Q_LORA)
QR_SCALE_SHAPE = (T, 1)

X_DTYPE = "bfloat16"
NORM_W_DTYPE = "float32"
WQ_A_DTYPE = "bfloat16"
WQ_B_DTYPE = "int8"
WQ_B_SCALE_DTYPE = "float32"
WKV_DTYPE = "bfloat16"
ROPE_DTYPE = "bfloat16"
SELECTOR_DTYPE = "bfloat16"
GAMMA_CQ_DTYPE = "bfloat16"
GAMMA_CKV_DTYPE = "bfloat16"
Q_DTYPE = "bfloat16"
KV_DTYPE = "bfloat16"
QR_DTYPE = "int8"
QR_SCALE_DTYPE = "float32"
