"""Independent CPU golden for deepseek_v4_flash_dspark.decode_layer_csa_flash."""

from config import CASE_CONFIG as M
from config import TP_SIZE
from config import DECODE_SEQ
from config import BLOCK_SIZE
from decode_sparse_attn_csa import T_PAD
import torch
from decode_compressor_ratio4 import golden_compressor
from decode_o_proj import golden_decode_o_proj_tp1
from hc_pre import golden_hc_pre
from decode_indexer import golden_indexer
from qkv_proj_rope import golden_qkv_proj_rope
from rmsnorm import golden_rms_norm
from decode_sparse_attn_csa import golden_sparse_attn
from hc_post import golden_hc_post

S = DECODE_SEQ
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
HC_MULT = M.hc_mult
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

def golden_decode_csa(tensors):
    """Run the complete TP1 reference independently for each token rank."""
    full_wo_a = tensors['wo_a'].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    full_wo_b = tensors['wo_b'].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)
    full_wo_b_scale = tensors['wo_b_scale'][0]
    for rank in range(TP_SIZE):
        rank_tensors = {name: value[rank] for (name, value) in tensors.items() if name != 'local_t'}
        rank_tensors['wo_a'] = full_wo_a
        rank_tensors['wo_b'] = full_wo_b
        rank_tensors['wo_b_scale'] = full_wo_b_scale
        golden_decode_csa_tp1(rank_tensors)

def golden_decode_csa_tp1(tensors):
    """Torch reference for the ratio-4 compression-step CSA orchestration."""
    tokens = tensors['x_hc'].shape[0]
    batch = tokens // S
    x_mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    post_t = torch.zeros(tokens, HC_MULT, dtype=torch.float32)
    comb_t = torch.zeros(tokens, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({'x': tensors['x_hc'], 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': x_mixed, 'post': post_t, 'comb': comb_t})
    position_ids = tensors['position_ids'].to(torch.int64)
    position_ids_bsd = position_ids.reshape(batch, S).to(torch.int32).contiguous()
    cmp_slot_mapping_bsd = tensors['cmp_slot_mapping'].reshape(batch, S).to(torch.int64).contiguous()
    idx_slot_mapping_bsd = tensors['idx_slot_mapping'].reshape(batch, S).to(torch.int64).contiguous()
    state_slot_mapping_bsd = tensors['state_slot_mapping'].reshape(batch, S).to(torch.int64).contiguous()
    inner_state_slot_mapping_bsd = tensors['inner_state_slot_mapping'].reshape(batch, S).to(torch.int64).contiguous()
    rope_cos_t = tensors['freqs_cos']
    rope_sin_t = tensors['freqs_sin']

    def interleave_rope(cos, sin):
        cos_il = cos[:, :HALF_ROPE].float().repeat_interleave(2, dim=-1)
        sin_il = sin[:, :HALF_ROPE].float().repeat_interleave(2, dim=-1)
        sign = torch.ones(ROPE_HEAD_DIM, dtype=torch.float32)
        sign[0::2] = -1.0
        return (cos_il.contiguous(), (sin_il * sign).contiguous())
    (idx_cos, idx_sin) = interleave_rope(rope_cos_t, rope_sin_t)
    (cmp_cos, cmp_sin) = interleave_rope(tensors['cmp_freqs_cos'], tensors['cmp_freqs_sin'])
    q = torch.zeros(tokens, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    qr_i8 = torch.zeros(tokens, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(tokens, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors['attn_norm_w'])
    golden_qkv_proj_rope({'x': x_normed, 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': rope_cos_t, 'rope_sin': rope_sin_t, 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': qr_i8, 'qr_scale': qr_scale})
    kv_cache = tensors['kv_cache']
    window_swa_indices = tensors['window_swa_indices']
    cmp_kv = tensors['cmp_kv']
    cmp_block_table = tensors['cmp_block_table']
    cmp_out = torch.zeros(tokens, HEAD_DIM, dtype=torch.float32)
    golden_compressor({'x': x_normed, 'kv': cmp_out, 'compress_state': tensors['compress_state'], 'compress_state_block_table': tensors['compress_state_block_table'], 'wkv': tensors['cmp_wkv'], 'wgate': tensors['cmp_wgate'], 'ape': tensors['cmp_ape'], 'norm_w': tensors['cmp_norm_w'], 'cos': cmp_cos, 'sin': cmp_sin, 'cmp_kv_cache': cmp_kv, 'position_ids': position_ids_bsd.reshape(-1), 'cmp_slot_mapping': cmp_slot_mapping_bsd.reshape(-1), 'state_slot_mapping': state_slot_mapping_bsd.reshape(-1)})
    idx_kv = torch.zeros(tokens, IDX_HEAD_DIM, dtype=torch.float32)
    idx_topk_scores = torch.full((tokens, IDX_TOPK), float('-inf'), dtype=torch.float32)
    idx_topk = torch.full((tokens, IDX_TOPK), -1, dtype=torch.int32)
    golden_indexer({'x': x_normed, 'qr': qr_i8, 'qr_scale': qr_scale, 'wq_b': tensors['idx_wq_b'], 'wq_b_scale': tensors['idx_wq_b_scale'], 'weights_proj': tensors['weights_proj'], 'cos': idx_cos, 'sin': idx_sin, 'cmp_cos': cmp_cos, 'cmp_sin': cmp_sin, 'hadamard': tensors['hadamard_idx'], 'inner_kv': idx_kv, 'inner_compress_state': tensors['inner_compress_state'], 'inner_compress_state_block_table': tensors['inner_compress_state_block_table'], 'inner_wkv': tensors['inner_wkv'], 'inner_wgate': tensors['inner_wgate'], 'inner_ape': tensors['inner_ape'], 'inner_norm_w': tensors['inner_norm_w'], 'idx_kv_cache': tensors['idx_kv_cache'], 'idx_kv_scale': tensors['idx_kv_scale'], 'idx_block_table': tensors['idx_block_table'], 'topk_scores': idx_topk_scores, 'topk_idxs': idx_topk, 'position_ids': position_ids_bsd.reshape(-1), 'idx_slot_mapping': idx_slot_mapping_bsd.reshape(-1), 'inner_state_slot_mapping': inner_state_slot_mapping_bsd.reshape(-1), 'kv_seq_lens': tensors['kv_seq_lens']})
    ori_slot_mapping = tensors['ori_slot_mapping'].to(torch.int64)
    for t in range(tokens):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            blk_id = write_row // BLOCK_SIZE
            intra = write_row % BLOCK_SIZE
            kv_cache[blk_id, intra, 0] = kv[t]
    o_packed_heads = torch.zeros(O_GROUPS, T_PAD, O_GROUP_IN, dtype=torch.bfloat16)
    golden_sparse_attn({'q': q, 'ori_kv': kv_cache, 'window_swa_indices': window_swa_indices, 'cmp_kv': cmp_kv, 'cmp_block_table': cmp_block_table, 'idx_topk': idx_topk, 'position_ids': position_ids.view(tokens, 1), 'attn_sink': tensors['attn_sink'], 'freqs_cos': rope_cos_t, 'freqs_sin': rope_sin_t, 'o_packed_heads': o_packed_heads})
    attn_out = golden_decode_o_proj_tp1(o_packed_heads, tensors['wo_a'], tensors['wo_b'], tensors['wo_b_scale'], tokens)
    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({'x': attn_out, 'residual': tensors['x_hc'], 'post': post_t, 'comb': comb_t, 'y': y})
    tensors['x_out'][:] = y
