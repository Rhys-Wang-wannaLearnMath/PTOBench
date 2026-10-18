# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num

"""ORACLE closure for deepseek_v4_flash_dspark.prefill_layer_hca_pro; fixed bindings from public config."""
from config import BLOCK_SIZE
from config import D
from config import H
from config import HCA_COMPRESS_RATIO as COMPRESS_RATIO
from config import HC_MULT
from config import HEAD_DIM
from config import IDX_TOPK
from config import KV_ORI_BLOCK_NUM
from config import Q_LORA
from config import SPARSE_CMP_MAX_BLOCKS
from config import WIN
from hc_post import golden_hc_post
from hc_pre import golden_hc_pre
from prefill_compressor_ratio128 import golden_prefill_compressor_ratio128
from prefill_sparse_attn import golden_prefill_sparse_attn
from qkv_proj_rope import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from utils import cache_row_from_table

HCA_ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM

def golden_prefill_attention_hca(tensors):
    import torch

    # Oracle-local dependency bound during isolated module load.

    token_count = tensors["x_hc"].shape[0]
    x_hc_flat = tensors["x_hc"].view(token_count, HC_MULT, D)
    x_mixed = torch.zeros(token_count, D, dtype=torch.bfloat16)
    post = torch.zeros(token_count, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(token_count, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre(
        {
            "x": x_hc_flat,
            "hc_fn": tensors["hc_attn_fn"],
            "hc_scale": tensors["hc_attn_scale"],
            "hc_base": tensors["hc_attn_base"],
            "x_mixed": x_mixed,
            "post": post,
            "comb": comb,
        }
    )

    q = torch.zeros(token_count, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(token_count, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(token_count, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(token_count, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()
    golden_qkv_proj_rope(
        {
            "x": x_normed.view(token_count, D),
            "wq_a": tensors["wq_a"],
            "wq_b": tensors["wq_b"],
            "wq_b_scale": tensors["wq_b_scale"],
            "wkv": tensors["wkv"],
            "rope_cos": rope_cos_t,
            "rope_sin": rope_sin_t,
            "gamma_cq": tensors["gamma_cq"],
            "gamma_ckv": tensors["gamma_ckv"],
            "q": q,
            "kv": kv,
            "qr": qr,
            "qr_scale": qr_scale,
        }
    )

    ori_kv = tensors["kv_cache"]
    ori_kv_flat = ori_kv.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(token_count):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            ori_kv_flat[dst_row, :] = kv[t]

    cmp_kv = tensors["cmp_kv"]
    golden_prefill_compressor_ratio128(
        {
            "x": x_normed.view(token_count, D),
            "compress_state": tensors["compress_state"],
            "compress_state_block_table": tensors["compress_state_block_table"],
            "wkv": tensors["cmp_wkv"],
            "wgate": tensors["cmp_wgate"],
            "ape": tensors["cmp_ape"],
            "norm_w": tensors["cmp_norm_w"],
            "freqs_cos": tensors["freqs_cos"],
            "freqs_sin": tensors["freqs_sin"],
            "cmp_kv": cmp_kv,
            "position_ids": tensors["position_ids"],
            "cmp_slot_mapping": tensors["cmp_slot_mapping"],
            "state_slot_mapping": tensors["state_slot_mapping"],
        }
    )

    def build_sparse_metadata():
        swa_idx = torch.full((token_count, WIN), -1, dtype=torch.int32)
        cmp_idx = torch.full((token_count, IDX_TOPK), -1, dtype=torch.int32)
        pos = tensors["position_ids"]
        ori_table = tensors["ori_block_table"]
        cmp_cap = SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE
        for t in range(token_count):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(ori_table, key_abs)
                if row >= 0:
                    swa_idx[t, k] = row
            visible_cmp = min((abs_pos + 1) // COMPRESS_RATIO, IDX_TOPK, cmp_cap)
            if visible_cmp > 0:
                cmp_idx[t, :visible_cmp] = torch.arange(visible_cmp, dtype=torch.int32)
        return swa_idx, cmp_idx

    swa_indices, cmp_indices = build_sparse_metadata()
    attn_out = torch.zeros(token_count, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn(
        {
            "q": q,
            "ori_kv": ori_kv,
            "swa_indices": swa_indices,
            "cmp_kv": cmp_kv,
            "cmp_block_table": tensors["cmp_block_table"],
            "cmp_indices": cmp_indices,
            "attn_sink": tensors["attn_sink"],
            "freqs_cos": rope_cos_t,
            "freqs_sin": rope_sin_t,
            "wo_a": tensors["wo_a"],
            "wo_b": tensors["wo_b"],
            "wo_b_scale": tensors["wo_b_scale"],
            "attn_out": attn_out,
        }
    )

    y = torch.zeros(token_count, HC_MULT, D, dtype=torch.float32)
    golden_hc_post(
        {
            "x": attn_out.view(token_count, D),
            "residual": x_hc_flat,
            "post": post,
            "comb": comb,
            "y": y,
        }
    )
    tensors["x_out"][:] = y


