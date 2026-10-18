from __future__ import annotations
from config import T, HC_MULT, D
from config import BLOCK_SIZE, CASE_CONFIG as M, PREFILL_BATCH, PREFILL_ORI_MAX_BLOCKS, PREFILL_SEQ
from hc_post_v2 import golden_hc_post
from hc_pre_v2 import golden_hc_pre
from prefill_compressor_ratio128 import golden_prefill_compressor_ratio128
from prefill_qkv_proj_rope_v2 import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from prefill_sparse_attn_v2 import golden_prefill_sparse_attn
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
Q_LORA = M.q_lora_rank
HC_MULT = M.hc_mult
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
HCA_ORI_BLOCK_NUM = SPARSE_ORI_MAX_BLOCKS

def golden_prefill_attention_hca(tensors):
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
    x_normed = golden_rms_norm(x_mixed, tensors['attn_norm_w'])
    rope_cos_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    rope_sin_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    positions = tensors['position_ids'].to(torch.long)
    rope_cos_t = tensors['freqs_cos'].index_select(0, positions).contiguous()
    rope_sin_t = tensors['freqs_sin'].index_select(0, positions).contiguous()
    golden_qkv_proj_rope({'x': x_normed.view(T, D), 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': rope_cos_t, 'rope_sin': rope_sin_t, 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': qr, 'qr_scale': qr_scale})
    ori_kv = tensors['kv_cache']
    ori_kv_flat = ori_kv.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    cmp_kv = tensors['cmp_kv']
    golden_prefill_compressor_ratio128({'x': x_normed.view(T, D), 'kv_state': tensors['cmp_kv_state'], 'score_state': tensors['cmp_score_state'], 'compress_state_block_table': tensors['compress_state_block_table'], 'wkv': tensors['cmp_wkv'], 'wgate': tensors['cmp_wgate'], 'ape': tensors['cmp_ape'], 'norm_w': tensors['cmp_norm_w'], 'freqs_cos': tensors['freqs_cos'], 'freqs_sin': tensors['freqs_sin'], 'cmp_kv': cmp_kv, 'position_ids': tensors['position_ids'], 'num_tokens': tensors['num_tokens'], 'cmp_slot_mapping': tensors['cmp_slot_mapping'], 'state_slot_mapping': tensors['state_slot_mapping']})
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({'q': q, 'ori_kv': ori_kv, 'ori_block_table': tensors['ori_block_table'], 'kv_overlay': kv, 'cmp_kv': cmp_kv, 'cmp_block_table': tensors['cmp_block_table'], 'cmp_sparse_indices': tensors['cmp_sparse_indices'], 'cmp_sparse_lens': tensors['cmp_sparse_lens'], 'attn_sink': tensors['attn_sink'], 'num_tokens': tensors['num_tokens'], 'freqs_cos': rope_cos_t, 'freqs_sin': rope_sin_t, 'wo_a': tensors['wo_a'], 'wo_b': tensors['wo_b'], 'wo_b_scale': tensors['wo_b_scale'], 'attn_out': attn_out})
    for t in range(num_tokens):
        dst_row = int(tensors['ori_slot_mapping'][t].item())
        if dst_row >= 0:
            ori_kv_flat[dst_row, :] = kv[t]
    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({'x': attn_out.view(T, D), 'residual': x_hc_flat, 'post': post, 'comb': comb, 'y': y.view(T, HC_MULT, D)})
    tensors['x_out'][:] = y.view(T, HC_MULT, D)

def compute(inputs, scalars):
    values = {name: value.clone() if hasattr(value, 'clone') else value for (name, value) in inputs.items()}
    values.update(scalars)
    import torch
    values['x_out'] = torch.zeros((T, HC_MULT, D), dtype=torch.bfloat16)
    golden_prefill_attention_hca(values)
    return {name: values[name] for name in ['x_out', 'kv_cache', 'cmp_kv']}
