"""Fixed evaluation contract for qwen3_14b.qwen3_14b_decode_scope1."""

from __future__ import annotations


# Model and ABI dimensions from the self-contained official source.
BATCH = 16
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

# RMSNorm semantics.
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Fixed tensor dtypes are kept readable for contract consumers.
HIDDEN_STATES_DTYPE = "bfloat16"
INPUT_RMS_WEIGHT_DTYPE = "float32"
PROJECTION_WEIGHT_DTYPE = "bfloat16"
PROJECTION_OUTPUT_DTYPE = "float32"
