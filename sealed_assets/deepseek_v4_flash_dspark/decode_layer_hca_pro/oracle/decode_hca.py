"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_hca_pro."""

from config import CASE_CONFIG as M
from config import TP_SIZE
from config import BLOCK_SIZE
from config import DECODE_SEQ
from decode_sparse_attn_hca import T_PAD
import torch
from hc_pre import golden_hc_pre
from qkv_proj_rope import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from decode_compressor_ratio128 import golden_compressor
from decode_o_proj import golden_decode_o_proj_tp1
from decode_sparse_attn_hca import golden_sparse_attn
from hc_post import golden_hc_post

S = DECODE_SEQ
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
HC_MULT = M.hc_mult
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

def golden_decode_hca(tensors):
    """Run the TP1 full-layer reference independently for every token shard."""
    full_wo_a = tensors['wo_a'].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    full_wo_b = tensors['wo_b'].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)
    for rank in range(TP_SIZE):
        rank_tensors = {name: tensor[rank] for (name, tensor) in tensors.items() if name != 'local_t'}
        rank_tensors['wo_a'] = full_wo_a
        rank_tensors['wo_b'] = full_wo_b
        golden_decode_hca_tp1(rank_tensors)

def golden_decode_hca_tp1(tensors):
    """End-to-end orchestration for the ratio=128 (HCA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==128 path: main compressor only,
    no indexer) + Block.hc_post."""
    tokens = tensors['x_hc'].shape[0]
    batch = tokens // S
    x_mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    post_t = torch.zeros(tokens, HC_MULT)
    comb_t = torch.zeros(tokens, HC_MULT * HC_MULT)
    golden_hc_pre({'x': tensors['x_hc'], 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': x_mixed, 'post': post_t, 'comb': comb_t})
    position_ids = tensors['position_ids'].to(torch.int64)
    rope_cos_T = tensors['freqs_cos']
    rope_sin_T = tensors['freqs_sin']
    q = torch.zeros(tokens, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(tokens, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(tokens, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors['attn_norm_w'])
    golden_qkv_proj_rope({'x': x_normed, 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': rope_cos_T, 'rope_sin': rope_sin_T, 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': qr, 'qr_scale': qr_scale})
    kv_cache = tensors['kv_cache']
    window_swa_indices = tensors['window_swa_indices']
    cmp_kv = tensors['cmp_kv']
    cmp_block_table = tensors['cmp_block_table']
    cmp_cos = tensors['cmp_freqs_cos'][:batch]
    cmp_sin = tensors['cmp_freqs_sin'][:batch]
    cmp_kv_proj = torch.zeros(tokens, HEAD_DIM, dtype=torch.float32)
    position_ids_flat = position_ids.reshape(-1).to(torch.int32).contiguous()
    cmp_slot_mapping_flat = tensors['cmp_slot_mapping'].reshape(-1).to(torch.int64).contiguous()
    state_slot_mapping_flat = tensors['state_slot_mapping'].reshape(-1).to(torch.int64).contiguous()
    golden_compressor({'x': x_normed, 'kv': cmp_kv_proj, 'compress_state': tensors['compress_state'], 'compress_state_block_table': tensors['compress_state_block_table'], 'wkv': tensors['cmp_wkv'], 'wgate': tensors['cmp_wgate'], 'ape': tensors['cmp_ape'], 'norm_w': tensors['cmp_norm_w'], 'cos': cmp_cos, 'sin': cmp_sin, 'cmp_kv_cache': cmp_kv, 'position_ids': position_ids_flat, 'cmp_slot_mapping': cmp_slot_mapping_flat, 'state_slot_mapping': state_slot_mapping_flat})
    ori_slot_mapping = tensors['ori_slot_mapping'].to(torch.int64)
    for t in range(tokens):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]
    o_packed_heads = torch.zeros(O_GROUPS, T_PAD, O_GROUP_IN, dtype=torch.bfloat16)
    golden_sparse_attn({'q': q, 'ori_kv': kv_cache, 'window_swa_indices': window_swa_indices, 'cmp_kv': cmp_kv, 'cmp_block_table': cmp_block_table, 'position_ids': position_ids_flat, 'kv_seq_lens': tensors['kv_seq_lens'], 'attn_sink': tensors['attn_sink'], 'freqs_cos': rope_cos_T, 'freqs_sin': rope_sin_T, 'o_packed_heads': o_packed_heads})
    attn_out = golden_decode_o_proj_tp1(o_packed_heads, tensors['wo_a'], tensors['wo_b'], tensors['wo_b_scale'], tokens)
    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({'x': attn_out, 'residual': tensors['x_hc'], 'post': post_t, 'comb': comb_t, 'y': y})
    tensors['x_out'][:] = y
