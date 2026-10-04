"""Fixed configuration for the frozen DeepSeek V3.2 decode-back workload."""
BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
INTERMEDIATE = 18432
NUM_HEADS = 128
V_HEAD_DIM = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
EP_NODES = 128
EPS = 1.0e-6
HIDDEN_INV = 1.0 / HIDDEN
