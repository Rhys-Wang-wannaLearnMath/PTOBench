"""Fixed evaluation contract for DeepSeek-V4 HC-pre decode, batch 16."""

B = 16
S = 1
D = 4096
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT
HC_DIM = HC_MULT * D
T = B * S
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6
HC_DIM_INV = 1.0 / HC_DIM

X_SHAPE = (B, S, HC_MULT, D)
HC_FN_SHAPE = (MIX_HC, HC_DIM)
HC_SCALE_SHAPE = (3,)
HC_BASE_SHAPE = (MIX_HC,)
X_MIXED_SHAPE = (B, S, D)
POST_SHAPE = (B, S, HC_MULT)
COMB_SHAPE = (B, S, HC_MULT, HC_MULT)

X_DTYPE = "bfloat16"
HC_FN_DTYPE = "float32"
HC_SCALE_DTYPE = "float32"
HC_BASE_DTYPE = "float32"
X_MIXED_DTYPE = "bfloat16"
POST_DTYPE = "float32"
COMB_DTYPE = "float32"
