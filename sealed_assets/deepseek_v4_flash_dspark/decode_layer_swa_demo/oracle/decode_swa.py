"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_swa_demo."""

from config import CASE_CONFIG as M
from config import TP_SIZE
from config import BLOCK_SIZE
from decode_sparse_attn_swa import T_PAD
import torch
from hc_pre import golden_hc_pre
from qkv_proj_rope import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from decode_o_proj import golden_decode_o_proj_tp1
from decode_sparse_attn_swa import golden_sparse_attn
from hc_post import golden_hc_post

D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
HC_MULT = M.hc_mult
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

def golden_decode_swa(tensors):
    """Run the complete TP1 SWA reference independently for each request shard."""
    full_wo_a = tensors['wo_a'].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    full_wo_b = tensors['wo_b'].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)
    for rank in range(TP_SIZE):
        rank_tensors = {name: value[rank] for (name, value) in tensors.items() if name != 'local_t'}
        rank_tensors['wo_a'] = full_wo_a
        rank_tensors['wo_b'] = full_wo_b
        rank_tensors['wo_b_scale'] = tensors['wo_b_scale'][0]
        golden_decode_swa_tp1(rank_tensors)

def golden_decode_swa_tp1(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    tokens = tensors['x_hc'].shape[0]
    x_mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    post_t = torch.zeros(tokens, HC_MULT)
    comb_t = torch.zeros(tokens, HC_MULT * HC_MULT)
    golden_hc_pre({'x': tensors['x_hc'], 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': x_mixed, 'post': post_t, 'comb': comb_t})
    q = torch.zeros(tokens, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(tokens, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(tokens, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors['attn_norm_w'])
    golden_qkv_proj_rope({'x': x_normed, 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': tensors['freqs_cos'], 'rope_sin': tensors['freqs_sin'], 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': qr, 'qr_scale': qr_scale})
    kv_cache = tensors['kv_cache']
    swa_slot_mapping = tensors['swa_slot_mapping'].to(torch.int64)
    for t in range(tokens):
        write_row = int(swa_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]
    o_packed_heads = torch.zeros(O_GROUPS, T_PAD * HEADS_PER_GROUP, HEAD_DIM, dtype=torch.bfloat16)
    golden_sparse_attn({'q': q, 'ori_kv': kv_cache, 'swa_indices': tensors['swa_indices'], 'swa_lens': tensors['swa_lens'], 'attn_sink': tensors['attn_sink'], 'freqs_cos': tensors['freqs_cos'], 'freqs_sin': tensors['freqs_sin'], 'o_packed_heads': o_packed_heads})
    attn_out = golden_decode_o_proj_tp1(o_packed_heads, tensors['wo_a'], tensors['wo_b'], tensors['wo_b_scale'], tokens)
    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({'x': attn_out, 'residual': tensors['x_hc'], 'post': post_t, 'comb': comb_t, 'y': y})
    tensors['x_out'][:] = y
