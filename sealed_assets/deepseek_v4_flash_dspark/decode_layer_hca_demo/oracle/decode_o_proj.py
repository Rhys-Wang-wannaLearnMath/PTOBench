"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_hca_demo."""

from config import TP_SIZE
from config import CASE_CONFIG as M
from config import INT8_SCALE_MAX
from config import DECODE_TOKENS
from config import INT8_AMAX_EPS
import torch

D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
LOCAL_T = DECODE_TOKENS // TP_SIZE
TOKEN_TILE = 16
LOCAL_T_PAD = (LOCAL_T + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE
T_PAD = LOCAL_T_PAD

def golden_decode_o_proj_tp1(o_packed_heads, wo_a, wo_b, wo_b_scale, tokens):
    """Project full-group packed attention rows with the TP1 quantization path."""
    attention = o_packed_heads.reshape(O_GROUPS, T_PAD, O_GROUP_IN)[:, :tokens].float()
    o_a = torch.einsum('gti,gri->gtr', attention, wo_a.float())
    row_amax = o_a.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = INT8_SCALE_MAX / row_amax
    o_a_i8 = torch.round(o_a * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = 1.0 / scale_q
    wo_b_groups = wo_b.reshape(D, O_GROUPS, O_LORA)
    attn_out = torch.zeros(tokens, D, dtype=torch.float32)
    for group in range(O_GROUPS):
        group_i32 = o_a_i8[group].to(torch.int32)
        weight_i32 = wo_b_groups[:, group].to(torch.int32)
        group_partial = group_i32 @ weight_i32.T
        attn_out = attn_out + group_partial.float() * scale_dq[group]
    attn_out = attn_out * wo_b_scale.float().unsqueeze(0)
    return attn_out.to(torch.bfloat16)
