"""Fixed evaluation contract for DeepSeek-V4 grouped output projection."""

B = 16
S = 1
T = B * S
D = 4096
H = 64
HEAD_DIM = 512
O_LORA = 1024
O_GROUPS = 8
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

O_SHAPE = (B, H, HEAD_DIM)
WO_A_SHAPE = (O_GROUPS, O_LORA, O_GROUP_IN)
WO_B_SHAPE = (D, O_GROUPS * O_LORA)
ATTN_OUT_SHAPE = (T, D)

O_DTYPE = "bfloat16"
WO_A_DTYPE = "bfloat16"
WO_B_DTYPE = "bfloat16"
ATTN_OUT_DTYPE = "bfloat16"
