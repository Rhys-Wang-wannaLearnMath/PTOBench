"""DeepSeek-V4 packed prefill SWA attention.

The public contract is single-request token-major prefill: the
layer owns the per-request loop and feeds this op one contiguous run of <=T
tokens. SWA consumes lowered metadata such as position_ids, slot mappings, and
window-ring sparse indices.
"""
from config import BLOCK_SIZE, CASE_CONFIG as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_CMP_BLOCK_NUM, PREFILL_CMP_MAX_BLOCKS, PREFILL_ORI_BLOCK_NUM, PREFILL_ORI_MAX_BLOCKS, PREFILL_SEQ
from hc_post_v2 import golden_hc_post
from hc_pre_v2 import golden_hc_pre
from prefill_qkv_proj_rope_v2 import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from prefill_sparse_attn_v2 import golden_prefill_sparse_attn
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HEAD_DIM = ROPE_DIM
NOPE_DIM = M.nope_head_dim
NOPE_HEAD_DIM = NOPE_DIM
Q_LORA = M.q_lora_rank
ROPE_HALF = ROPE_DIM // 2
HALF_ROPE = ROPE_HALF
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
IDX_TOPK = M.index_topk
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
START_POS = 0
SPARSE_TOPK = WIN + IDX_TOPK
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_ORI_BLOCK_NUM = B * SPARSE_ORI_MAX_BLOCKS
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
WRITEBACK_GUARD_TILE = 16
MIX_PAD = 32
NEG_INF = -1e+20
T_TILE = 16
RMS_T_TILE = 16
LINEAR_T_TILE = 16
COMB_T_TILE = 16
RMS_K_CHUNK = 128
LINEAR_K_CHUNK = 512
D_CHUNK = 512
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK
RMS_PIPE_STAGE = 1 if T >= 64 else 4
assert WIN == BLOCK_SIZE, 'SWA prefill currently assumes one window page per batch'
assert S == WIN, 'SWA overlay raw-index contract maps current suffix rows as WIN+t'
assert SPARSE_ORI_BLOCK_NUM == B * SPARSE_ORI_MAX_BLOCKS
assert SPARSE_ORI_MAX_BLOCKS == BLOCK_NUM

def golden_prefill_attention_swa(tensors):
    """Torch reference for token-major packed SWA prefill."""
    import torch
    num_tokens = int(tensors['num_tokens'])
    x_hc_rect = tensors['x_hc'].view(B, S, HC_MULT, D)
    x_hc_flat = x_hc_rect.view(T, HC_MULT, D)
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({'x': x_hc_flat, 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': x_mixed, 'post': post, 'comb': comb})
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    rope_cos_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    rope_sin_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    x_normed = golden_rms_norm(x_mixed, tensors['attn_norm_w'])
    positions = tensors['position_ids'].to(torch.long)
    rope_cos_t = tensors['freqs_cos'].index_select(0, positions).contiguous()
    rope_sin_t = tensors['freqs_sin'].index_select(0, positions).contiguous()
    golden_qkv_proj_rope({'x': x_normed, 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': rope_cos_t, 'rope_sin': rope_sin_t, 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': qr, 'qr_scale': qr_scale})
    kv_cache_in = tensors['kv_cache'].clone()
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({'q': q, 'ori_kv': kv_cache_in, 'ori_block_table': tensors['block_table'], 'kv_overlay': kv, 'cmp_kv': torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16), 'cmp_block_table': torch.zeros(SPARSE_CMP_MAX_BLOCKS, dtype=torch.int32), 'cmp_sparse_indices': tensors['cmp_sparse_indices'], 'cmp_sparse_lens': tensors['cmp_sparse_lens'], 'attn_sink': tensors['attn_sink'], 'num_tokens': tensors['num_tokens'], 'freqs_cos': rope_cos_t, 'freqs_sin': rope_sin_t, 'wo_a': tensors['wo_a'], 'wo_b': tensors['wo_b'], 'wo_b_scale': tensors['wo_b_scale'], 'attn_out': attn_out})
    kv_cache_out = kv_cache_in.clone()
    kv_cache_flat = kv_cache_out.view(BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens):
        dst_row = int(tensors['ori_slot_mapping'][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]
    tensors['kv_cache'][:] = kv_cache_out
    y = torch.zeros(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({'x': attn_out.view(T, D), 'residual': x_hc_flat, 'post': post, 'comb': comb, 'y': y})
    tensors['x_out'][:] = y
