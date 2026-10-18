"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_csa_demo."""

from config import TP_GROUPS
from config import TP_SIZE
from config import N_RANKS
from decode_csa import HC_MULT as HC_MULT
from decode_csa import D as D
import decode_csa as csa
import moe as moe_module
import torch

MOE_TOKENS = moe_module.T
_CSA_INPUT_NAMES = ('x_hc', 'hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'attn_norm_w', 'wq_a', 'wq_b', 'wq_b_scale', 'wkv', 'gamma_cq', 'gamma_ckv', 'freqs_cos', 'freqs_sin', 'cmp_freqs_cos', 'cmp_freqs_sin', 'cmp_wkv', 'cmp_wgate', 'cmp_ape', 'cmp_norm_w', 'compress_state', 'compress_state_block_table', 'idx_wq_b', 'idx_wq_b_scale', 'weights_proj', 'hadamard_idx', 'inner_wkv', 'inner_wgate', 'inner_ape', 'inner_norm_w', 'inner_compress_state', 'inner_compress_state_block_table', 'kv_cache', 'cmp_kv', 'cmp_block_table', 'idx_kv_cache', 'idx_kv_scale', 'idx_block_table', 'ori_slot_mapping', 'window_swa_indices', 'window_swa_lens', 'cmp_slot_mapping', 'idx_slot_mapping', 'state_slot_mapping', 'inner_state_slot_mapping', 'position_ids', 'kv_seq_lens', 'attn_sink', 'wo_a', 'wo_b', 'wo_b_scale')

def golden_decode_layer_csa(tensors):
    """Compose the unchanged CSA and MoE goldens through the bounded bridge."""
    local_t = int(tensors['local_t'])
    tensors['x_attn_active'].zero_()
    for group in range(TP_GROUPS):
        group_begin = group * TP_SIZE
        group_end = group_begin + TP_SIZE
        group_tensors = {name: tensors[name][group_begin:group_end] for name in _CSA_INPUT_NAMES}
        group_tensors['x_out'] = tensors['x_attn_active'][group_begin:group_end]
        group_tensors['local_t'] = local_t
        csa.golden_decode_csa(group_tensors)
    x_attn_moe = torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    x_attn_moe[:, :local_t].copy_(tensors['x_attn_active'])
    moe_tensors = dict(tensors)
    moe_tensors['x_hc'] = x_attn_moe
    moe_tensors['x_next'] = tensors['x_moe_next']
    moe_tensors['num_tokens'] = local_t
    moe_module.golden_moe(moe_tensors)
    tensors['x_next'].copy_(tensors['x_moe_next'][:, :local_t])
