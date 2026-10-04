"""Fixed evaluation contract for DeepSeek-V4 HC-post decode."""

B = 16
S = 1
D = 4096
HC_MULT = 4
T = B * S
D_CHUNK = 512
D_BLOCKS = D // D_CHUNK
HC_DIM = HC_MULT * D

X_SHAPE = (B, S, D)
RESIDUAL_SHAPE = (B, S, HC_MULT, D)
POST_SHAPE = (B, S, HC_MULT)
COMB_SHAPE = (B, S, HC_MULT, HC_MULT)
Y_SHAPE = (B, S, HC_MULT, D)

X_DTYPE = "bfloat16"
RESIDUAL_DTYPE = "bfloat16"
POST_DTYPE = "float32"
COMB_DTYPE = "float32"
Y_DTYPE = "bfloat16"
