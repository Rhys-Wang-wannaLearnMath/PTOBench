# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: EP2/TP2 representative single-layer golden
# ci: no-sim    # CI marker: distributed communication oracle requires real devices

"""ORACLE closure for deepseek_v4_flash_dspark.prefill_layer_hca_demo; fixed bindings from public config."""
from config import D
from config import GROUP_TOKENS
from config import HC_MULT
from config import N_RANKS
from config import O_GROUPS
from config import O_GROUP_IN
from config import O_LORA
from config import TP_SIZE
from hc_pre import golden_hc_pre
from moe import T
from moe import golden_moe
from prefill_csa import golden_prefill_attention_csa
from prefill_hca import golden_prefill_attention_hca
from prefill_swa import golden_prefill_attention_swa

_CACHE_OUTPUTS = {
    0: {"kv_cache"},
    2: {
        "kv_cache", "csa_cmp_kv", "csa_compress_state",
        "csa_inner_compress_state", "idx_kv_cache", "idx_kv_scale",
    },
    3: {"kv_cache", "hca_cmp_kv", "hca_compress_state"},
}

def _attention_golden_tensors(tensors, rank, layer_id, x_out):
    wo_a_full = tensors["wo_a"][rank : rank + TP_SIZE].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    wo_b_full = tensors["wo_b"][rank : rank + TP_SIZE].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)
    common = {
        "x_hc": tensors["x_hc"][rank],
        "hc_attn_fn": tensors["hc_attn_fn"][rank],
        "hc_attn_scale": tensors["hc_attn_scale"][rank],
        "hc_attn_base": tensors["hc_attn_base"][rank],
        "attn_norm_w": tensors["attn_norm_w"][rank],
        "wq_a": tensors["wq_a"][rank],
        "wq_b": tensors["wq_b"][rank],
        "wq_b_scale": tensors["wq_b_scale"][rank],
        "wkv": tensors["wkv"][rank],
        "gamma_cq": tensors["gamma_cq"][rank],
        "gamma_ckv": tensors["gamma_ckv"][rank],
        "freqs_cos": tensors["freqs_cos"][rank, 1 if layer_id in (2, 3) else 0],
        "freqs_sin": tensors["freqs_sin"][rank, 1 if layer_id in (2, 3) else 0],
        "kv_cache": tensors["kv_cache"][rank],
        "ori_block_table": tensors["ori_block_table"][rank],
        "ori_slot_mapping": tensors["ori_slot_mapping_full"][rank],
        "position_ids": tensors["position_ids_full"][rank],
        "attn_sink": tensors["attn_sink"][rank],
        "wo_a": wo_a_full,
        "wo_b": wo_b_full,
        "wo_b_scale": tensors["wo_b_scale"][rank],
        "x_out": x_out,
        "num_tokens": GROUP_TOKENS,
    }
    if layer_id == 0:
        common["block_table"] = common.pop("ori_block_table")
        return common
    if layer_id == 3:
        common.update(
            {
                "cmp_wkv": tensors["hca_cmp_wkv"][rank],
                "cmp_wgate": tensors["hca_cmp_wgate"][rank],
                "cmp_ape": tensors["hca_cmp_ape"][rank],
                "cmp_norm_w": tensors["hca_cmp_norm_w"][rank],
                "compress_state": tensors["hca_compress_state"][rank],
                "compress_state_block_table": tensors["hca_compress_state_block_table"][rank],
                "cmp_kv": tensors["hca_cmp_kv"][rank],
                "cmp_block_table": tensors["hca_cmp_block_table"][rank],
                "cmp_slot_mapping": tensors["hca_cmp_slot_mapping_full"][rank],
                "state_slot_mapping": tensors["hca_state_slot_mapping_full"][rank],
            }
        )
        return common
    common.update(
        {
            "cmp_wkv": tensors["csa_cmp_wkv"][rank],
            "cmp_wgate": tensors["csa_cmp_wgate"][rank],
            "cmp_ape": tensors["csa_cmp_ape"][rank],
            "cmp_norm_w": tensors["csa_cmp_norm_w"][rank],
            "compress_state": tensors["csa_compress_state"][rank],
            "compress_state_block_table": tensors["csa_compress_state_block_table"][rank],
            "hadamard_idx": tensors["csa_hadamard_idx"][rank],
            "idx_wq_b": tensors["csa_idx_wq_b"][rank],
            "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"][rank],
            "idx_weights_proj": tensors["csa_weights_proj"][rank],
            "inner_wkv": tensors["csa_inner_wkv"][rank],
            "inner_wgate": tensors["csa_inner_wgate"][rank],
            "inner_ape": tensors["csa_inner_ape"][rank],
            "inner_norm_w": tensors["csa_inner_norm_w"][rank],
            "inner_compress_state": tensors["csa_inner_compress_state"][rank],
            "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"][rank],
            "cmp_kv": tensors["csa_cmp_kv"][rank],
            "cmp_block_table": tensors["csa_cmp_block_table"][rank],
            "idx_kv_cache": tensors["idx_kv_cache"][rank],
            "idx_kv_scale": tensors["idx_kv_scale"][rank],
            "idx_block_table": tensors["idx_block_table"][rank],
            "cmp_slot_mapping": tensors["csa_cmp_slot_mapping_full"][rank],
            "idx_slot_mapping": tensors["csa_idx_slot_mapping_full"][rank],
            "state_slot_mapping": tensors["csa_state_slot_mapping_full"][rank],
            "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping_full"][rank],
        }
    )
    return common


def golden_prefill_layer(tensors):
    """Compute the full-attention and distributed-MoE reference."""
    import torch

    # Oracle-local dependency bound during isolated module load.

    layer_id = int(tensors["layer_id"])
    attention_golden = {
        0: golden_prefill_attention_swa,
        2: golden_prefill_attention_csa,
        3: golden_prefill_attention_hca,
    }[layer_id]
    cache_outputs = _CACHE_OUTPUTS[layer_id]

    for group_base in range(0, N_RANKS, TP_SIZE):
        full_attn = torch.zeros(GROUP_TOKENS, HC_MULT, D, dtype=torch.float32)
        attention_tensors = _attention_golden_tensors(tensors, group_base, layer_id, full_attn)
        attention_golden(attention_tensors)
        full_attn = attention_tensors["x_out"]
        for rank in range(group_base, group_base + TP_SIZE):
            tensors["attn_stage"][rank].copy_(full_attn)
            for name in cache_outputs:
                tensors[name][rank].copy_(tensors[name][group_base])

    local_attn = torch.empty(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    for rank in range(N_RANKS):
        golden_hc_pre({
            "x": tensors["attn_stage"][rank],
            "hc_fn": tensors["hc_ffn_fn"][rank],
            "hc_scale": tensors["hc_ffn_scale"][rank],
            "hc_base": tensors["hc_ffn_base"][rank],
            "x_mixed": tensors["x_mixed"][rank],
            "post": tensors["post_ffn"][rank],
            "comb": tensors["comb_ffn"][rank],
        })
        tp_rank = rank % TP_SIZE
        local_attn[rank].copy_(tensors["attn_stage"][rank, tp_rank * T : (tp_rank + 1) * T])

    local_next = torch.zeros_like(local_attn)
    moe_tensors = dict(tensors)
    moe_tensors.update({"x_hc": local_attn, "x_next": local_next, "num_tokens": T, "layer_id": layer_id})
    golden_moe(moe_tensors)

    for group_base in range(0, N_RANKS, TP_SIZE):
        full_next = local_next[group_base : group_base + TP_SIZE].reshape(GROUP_TOKENS, HC_MULT, D)
        for rank in range(group_base, group_base + TP_SIZE):
            tensors["x_next"][rank].copy_(full_next)


