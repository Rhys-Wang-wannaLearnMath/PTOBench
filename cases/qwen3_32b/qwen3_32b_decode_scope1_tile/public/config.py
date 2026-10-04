"""Fixed evaluation contract for qwen3_32b.qwen3_32b_decode_scope1_tile."""

from __future__ import annotations


# Source-shaped decode dimensions and RMSNorm derivation.
BATCH = 16
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Fixed tensor ABI dtypes and shapes. Tile/chunk decomposition remains an
# implementation choice and is intentionally absent from this public contract.
HIDDEN_STATES_DTYPE = "bfloat16"
INPUT_RMS_WEIGHT_DTYPE = "float32"
PROJECTION_WEIGHT_DTYPE = "bfloat16"
PROJECTION_OUTPUT_DTYPE = "float32"
HIDDEN_STATES_SHAPE = (BATCH, HIDDEN)
INPUT_RMS_WEIGHT_SHAPE = (1, HIDDEN)
WQ_SHAPE = (HIDDEN, HIDDEN)
WK_SHAPE = (HIDDEN, KV_HIDDEN)
WV_SHAPE = (HIDDEN, KV_HIDDEN)
Q_PROJ_SHAPE = (BATCH, HIDDEN)
K_PROJ_SHAPE = (BATCH, KV_HIDDEN)
V_PROJ_SHAPE = (BATCH, KV_HIDDEN)
