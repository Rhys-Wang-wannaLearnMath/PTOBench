"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_swa_flash."""

from config import TP_GROUPS
from config import TP_SIZE
from config import N_RANKS
import decode_swa as swa
import moe as moe_module
import torch

MOE_TOKENS = moe_module.T
D = swa.D
HC_MULT = swa.HC_MULT
_SWA_INPUT_NAMES = ('x_hc', 'hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'attn_norm_w', 'wq_a', 'wq_b', 'wq_b_scale', 'wkv', 'gamma_cq', 'gamma_ckv', 'freqs_cos', 'freqs_sin', 'kv_cache', 'swa_slot_mapping', 'swa_indices', 'swa_lens', 'position_ids', 'attn_sink', 'wo_a', 'wo_b', 'wo_b_scale')

def golden_decode_layer_swa(tensors):
    """Compose the unchanged SWA and MoE goldens through the bounded bridge."""
    local_t = int(tensors['local_t'])
    tensors['x_attn_active'].zero_()
    for group in range(TP_GROUPS):
        group_begin = group * TP_SIZE
        group_end = group_begin + TP_SIZE
        group_tensors = {name: tensors[name][group_begin:group_end] for name in _SWA_INPUT_NAMES}
        group_tensors['x_out'] = tensors['x_attn_active'][group_begin:group_end]
        group_tensors['local_t'] = local_t
        swa.golden_decode_swa(group_tensors)
    x_attn_moe = torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    x_attn_moe[:, :local_t].copy_(tensors['x_attn_active'])
    moe_tensors = dict(tensors)
    moe_tensors['x_hc'] = x_attn_moe
    moe_tensors['x_next'] = tensors['x_moe_next']
    moe_tensors['num_tokens'] = local_t
    moe_module.golden_moe(moe_tensors)
    tensors['x_next'].copy_(tensors['x_moe_next'][:, :local_t])
