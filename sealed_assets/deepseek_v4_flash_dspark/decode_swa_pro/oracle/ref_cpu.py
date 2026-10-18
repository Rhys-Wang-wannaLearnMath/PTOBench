# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent selected CPU golden closure for SWA decode."""

import torch
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE
from hc_pre import golden_hc_pre
from qkv_proj_rope import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from decode_sparse_attn_swa import golden_sparse_attn
from hc_post import golden_hc_post
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

def golden_attention_swa(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    import torch
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT)
    golden_hc_pre({'x': tensors['x_hc'], 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': x_mixed, 'post': post_t, 'comb': comb_t})
    position_ids = tensors['position_ids'].to(torch.int64)
    (bsz, seqlen) = (B, S)
    win = WIN
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
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    swa_slot_mapping = tensors['swa_slot_mapping'].to(torch.int64)
    for t in range(T):
        write_row = int(swa_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]
    golden_sparse_attn({'q': q, 'ori_kv': kv_cache, 'swa_indices': tensors['swa_indices'], 'swa_lens': tensors['swa_lens'], 'attn_sink': tensors['attn_sink'], 'freqs_cos': rope_cos_T, 'freqs_sin': rope_sin_T, 'wo_a': tensors['wo_a'], 'wo_b': tensors['wo_b'], 'wo_b_scale': tensors['wo_b_scale'], 'attn_out': attn_out})
    y = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({'x': attn_out, 'residual': tensors['x_hc'], 'post': post_t, 'comb': comb_t, 'y': y})
    tensors['x_out'][:] = y

def compute(values):
    tensors = {name: value.clone() if isinstance(value, torch.Tensor) else value for (name, value) in values.items()}
    golden_attention_swa(tensors)
    return {'x_out': tensors['x_out'], 'kv_cache': tensors['kv_cache']}
