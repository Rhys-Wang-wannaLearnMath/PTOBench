# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent CPU golden closure."""

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE
from decode_sparse_attn_hca import CMP_TOPK as HCA_SPARSE_CMP_TOPK
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
HC_MULT = M.hc_mult
from config import HCA_MAX_SEQ_LEN as MAX_SEQ_LEN
from config import COMPRESS_RATIO
COMPRESS_TOPK = MAX_SEQ_LEN // COMPRESS_RATIO
HCA_TOPK_LIMIT = COMPRESS_TOPK
HCA_CMP_TOPK = HCA_SPARSE_CMP_TOPK

def golden_attention_hca(tensors):
    """End-to-end orchestration for the ratio=128 (HCA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==128 path: main compressor only,
    no indexer, compress_topk_idxs computed deterministically) + Block.hc_post."""
    import torch
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT)
    golden_hc_pre({'x': tensors['x_hc'], 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': x_mixed, 'post': post_t, 'comb': comb_t})
    position_ids = tensors['position_ids'].to(torch.int64)
    kv_seq_lens = tensors['kv_seq_lens'].to(torch.int64)
    win = WIN
    ratio = COMPRESS_RATIO
    rd = ROPE_HEAD_DIM
    freqs_cos = tensors['freqs_cos']
    freqs_sin = tensors['freqs_sin']
    rope_cos_T = torch.empty(T, rd, dtype=freqs_cos.dtype)
    rope_sin_T = torch.empty(T, rd, dtype=freqs_sin.dtype)
    for t in range(T):
        pos = int(position_ids[t].item())
        rope_cos_T[t] = freqs_cos[pos]
        rope_sin_T[t] = freqs_sin[pos]
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors['attn_norm_w'])
    golden_qkv_proj_rope({'x': x_normed, 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': rope_cos_T, 'rope_sin': rope_sin_T, 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': qr, 'qr_scale': qr_scale})
    kv_cache = tensors['kv_cache']
    window_swa_indices = tensors['window_swa_indices']
    window_swa_lens = tensors['window_swa_lens']
    cmp_kv = tensors['cmp_kv']
    cmp_block_table = tensors['cmp_block_table']
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    half_rd = rd // 2
    cmp_cos = torch.empty(B, half_rd, dtype=torch.float32)
    cmp_sin = torch.empty(B, half_rd, dtype=torch.float32)
    for b in range(B):
        first_pos_b = int(position_ids[b * S].item())
        cmp_offset_b = ratio - first_pos_b % ratio
        cmp_pos_b = first_pos_b + cmp_offset_b - ratio
        cmp_cos[b] = freqs_cos[cmp_pos_b, :half_rd].float()
        cmp_sin[b] = freqs_sin[cmp_pos_b, :half_rd].float()
    cmp_kv_proj = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    position_ids_bsd = position_ids.reshape(B, S).to(torch.int32).contiguous()
    cmp_slot_mapping_bsd = tensors['cmp_slot_mapping'].reshape(B, S).to(torch.int64).contiguous()
    state_slot_mapping_bsd = tensors['state_slot_mapping'].reshape(B, S).to(torch.int64).contiguous()
    golden_compressor({'x': x_normed.reshape(B, S, D), 'kv': cmp_kv_proj, 'compress_state': tensors['compress_state'], 'compress_state_block_table': tensors['compress_state_block_table'], 'wkv': tensors['cmp_wkv'], 'wgate': tensors['cmp_wgate'], 'ape': tensors['cmp_ape'], 'norm_w': tensors['cmp_norm_w'], 'cos': cmp_cos, 'sin': cmp_sin, 'cmp_kv_cache': cmp_kv, 'position_ids': position_ids_bsd, 'cmp_slot_mapping': cmp_slot_mapping_bsd, 'state_slot_mapping': state_slot_mapping_bsd})
    ori_slot_mapping = tensors['ori_slot_mapping'].to(torch.int64)
    for t in range(T):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]
    topk_all = torch.full((T, HCA_CMP_TOPK), -1, dtype=torch.int32)
    for t in range(T):
        b = t // S
        abs_pos = int(position_ids[t].item())
        cmp_valid = min(HCA_TOPK_LIMIT, (abs_pos + 1) // ratio, int(kv_seq_lens[b].item()) // ratio)
        if cmp_valid:
            topk_all[t, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32)
    golden_sparse_attn({'q': q, 'ori_kv': kv_cache, 'window_swa_indices': window_swa_indices, 'cmp_kv': cmp_kv, 'cmp_block_table': cmp_block_table, 'cmp_sparse_indices': topk_all, 'attn_sink': tensors['attn_sink'], 'freqs_cos': rope_cos_T, 'freqs_sin': rope_sin_T, 'wo_a': tensors['wo_a'], 'wo_b': tensors['wo_b'], 'wo_b_scale': tensors['wo_b_scale'], 'attn_out': attn_out})
    y = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({'x': attn_out, 'residual': tensors['x_hc'], 'post': post_t, 'comb': comb_t, 'y': y})
    tensors['x_out'][:] = y
from hc_pre import golden_hc_pre
from qkv_proj_rope import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from decode_compressor_ratio128 import golden_compressor
from decode_sparse_attn_hca import golden_sparse_attn
from hc_post import golden_hc_post
