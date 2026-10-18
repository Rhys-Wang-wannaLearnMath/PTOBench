# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num

"""ORACLE closure for deepseek_v4_flash_dspark.prefill_layer_hca_flash; fixed bindings from public config."""
from config import BLOCK_SIZE
from config import CSA_ORI_BLOCK_NUM
from config import D
from config import H
from config import HC_MULT
from config import HEAD_DIM
from config import Q_LORA
from config import ROPE_HEAD_DIM
from config import WIN
from hc_post import golden_hc_post
from hc_pre import golden_hc_pre
from prefill_compressor_ratio4 import golden_prefill_compressor_ratio4
from prefill_indexer import golden_prefill_indexer_core
from prefill_indexer import topk_prefix_contract_error
from prefill_sparse_attn import golden_prefill_sparse_attn
from qkv_proj_rope import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from utils import cache_row_from_table

HALF_ROPE = ROPE_HEAD_DIM // 2

def golden_prefill_attention_csa(tensors):
    """Torch reference for token-major packed CSA with overlay compressor/indexer."""
    import torch

    # Oracle-local dependency bound during isolated module load.

    token_count = int(tensors["x_hc"].shape[0])
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

    golden_prefill_compressor_ratio4(
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
            "cmp_kv": tensors["cmp_kv"],
            "position_ids": tensors["position_ids"],
            "cmp_slot_mapping": tensors["cmp_slot_mapping"],
            "state_slot_mapping": tensors["state_slot_mapping"],
        }
    )
    idx_cos = rope_cos_t[:, :HALF_ROPE].float().contiguous()
    idx_sin = rope_sin_t[:, :HALF_ROPE].float().contiguous()
    cmp_topk_indices, _idx_score = golden_prefill_indexer_core(
        {
            "x": x_normed.view(token_count, D),
            "qr": qr,
            "qr_scale": qr_scale,
            "wq_b": tensors["idx_wq_b"],
            "wq_b_scale": tensors["idx_wq_b_scale"],
            "weights_proj": tensors["idx_weights_proj"],
            "cos": idx_cos,
            "sin": idx_sin,
            "freqs_cos": tensors["freqs_cos"],
            "freqs_sin": tensors["freqs_sin"],
            "hadamard": tensors["hadamard_idx"],
            "inner_compress_state": tensors["inner_compress_state"],
            "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
            "inner_wkv": tensors["inner_wkv"],
            "inner_wgate": tensors["inner_wgate"],
            "inner_ape": tensors["inner_ape"],
            "inner_norm_w": tensors["inner_norm_w"],
            "idx_kv_cache": tensors["idx_kv_cache"],
            "idx_kv_scale": tensors["idx_kv_scale"],
            "idx_block_table": tensors["idx_block_table"],
            "position_ids": tensors["position_ids"],
            "idx_slot_mapping": tensors["idx_slot_mapping"],
            "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
        }
    )

    kv_cache_in = tensors["kv_cache"].clone()
    kv_cache_flat = kv_cache_in.view(CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(token_count):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]

    def assemble_swa_indices():
        swa_idx = torch.full((token_count, WIN), -1, dtype=torch.int32)
        pos = tensors["position_ids"]
        ori_table = tensors["ori_block_table"]
        for t in range(token_count):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(ori_table, key_abs)
                if row >= 0:
                    swa_idx[t, k] = row
        return swa_idx

    contract_error = topk_prefix_contract_error(
        cmp_topk_indices,
        tensors["position_ids"],
    )
    if contract_error:
        raise AssertionError(f"prefill indexer top-k contract failed: {contract_error}")
    swa_indices = assemble_swa_indices()
    cmp_indices = cmp_topk_indices.clone()
    attn_out = torch.zeros(token_count, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn(
        {
            "q": q,
            "ori_kv": kv_cache_in,
            "swa_indices": swa_indices,
            "cmp_kv": tensors["cmp_kv"],
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

    tensors["kv_cache"][:] = kv_cache_in

    y = torch.zeros(token_count, HC_MULT, D, dtype=torch.float32)
    golden_hc_post(
        {
            "x": attn_out,
            "residual": tensors["x_hc"],
            "post": post,
            "comb": comb,
            "y": y,
        }
    )
    tensors["x_out"][:] = y


