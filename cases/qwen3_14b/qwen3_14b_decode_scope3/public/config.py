"""Fixed evaluation contract for qwen3_14b.qwen3_14b_decode_scope3."""

from __future__ import annotations


# Model and ABI dimensions from the self-contained official source.
BATCH = 16
HIDDEN = 5120
INTERMEDIATE = 17408

# Post-attention RMSNorm semantics.
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Fixed tensor dtypes, grouped by their ABI roles.
ACTIVATION_DTYPE = "bfloat16"  # attn_out, hidden_states
POST_RMS_WEIGHT_DTYPE = "float32"
PROJECTION_WEIGHT_DTYPE = "bfloat16"  # wo, w_gate, w_up, w_down
OUTPUT_DTYPE = "bfloat16"
