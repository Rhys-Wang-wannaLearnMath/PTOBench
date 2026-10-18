"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_csa_flash."""

from config import CASE_CONFIG as M
from config import COMPRESS_RATIO
from config import MAX_SEQ_LEN
from config import TP_SIZE
from config import BLOCK_SIZE
from config import C4A_COMPRESSOR_BLOCK_SIZE
import pypto.language as pl
import pypto.language.distributed as pld
from decode_compressor_ratio4 import compressor_ratio4
from hc_post import hc_post
from hc_pre import hc_pre
from decode_indexer import indexer
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from decode_o_proj import LOCAL_O_GROUPS
from decode_o_proj import decode_sharded_o_projection_reduce_scatter
from decode_o_proj import GROUP_T_PAD
from decode_o_proj import LOCAL_T_PAD
from decode_o_proj import o_group_a2a
from decode_o_proj import ATTENTION_WINDOW_ROWS
from decode_o_proj import O_WINDOW_ROWS
from decode_o_proj import LOCAL_O_WIDTH
from decode_sparse_attn_csa import sparse_attn_csa

B_DYN = pl.dynamic('B_DYN')
T_DYN = pl.dynamic('T_DYN')
ORI_BLOCK_NUM_DYN = pl.dynamic('ORI_BLOCK_NUM_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('CMP_BLOCK_NUM_DYN')
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic('IDX_CACHE_BLOCK_NUM_DYN')
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic('CSA_STATE_BLOCK_NUM_DYN')
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic('INNER_STATE_BLOCK_NUM_DYN')
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
MAIN_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
MAIN_STATE_MAX_BLOCKS = (MAIN_STATE_LEN + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_LEN = COFF * COMPRESS_RATIO
INNER_STATE_MAX_BLOCKS = (INNER_STATE_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
CMP_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_MAX_BLOCKS = (CMP_MAX_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_MAX_BLOCKS = CMP_MAX_BLOCKS
CSA_WB_TOKEN_TILE = 8

@pl.jit.inline(auto_scope=False)
def decode_csa(x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], compress_state_block_table: pl.Tensor[[B_DYN, MAIN_STATE_MAX_BLOCKS], pl.INT32], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32], idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32], idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32], ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], window_swa_lens: pl.Tensor[[T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16], attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32], o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32]):
    """Run one rank of the complete tensor-parallel CSA layer."""
    t_dim = pl.tensor.dim(x_hc, 0)
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    idx_topk_scores = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.FP32)
    idx_topk = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.INT32)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    wb_blocks = t_dim // CSA_WB_TOKEN_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.scope():
        hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)
    idx_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    idx_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='csa_rope_interleave') as rope_tid:
        il_ones = pl.full([4, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(il_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        for rope_t0 in pl.range(0, t_dim, 4):
            idx_cos_il[rope_t0:rope_t0 + 4, :] = pl.gather(pl.cast(freqs_cos[rope_t0:rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32), dim=-1, index=il_dup_idx)
            idx_sin_signed[rope_t0:rope_t0 + 4, :] = pl.mul(pl.gather(pl.cast(freqs_sin[rope_t0:rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32), dim=-1, index=il_dup_idx), il_sign)
            cmp_cos_il[rope_t0:rope_t0 + 4, :] = pl.gather(pl.cast(cmp_freqs_cos[rope_t0:rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32), dim=-1, index=il_dup_idx)
            cmp_sin_signed[rope_t0:rope_t0 + 4, :] = pl.mul(pl.gather(pl.cast(cmp_freqs_sin[rope_t0:rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32), dim=-1, index=il_dup_idx), il_sign)
    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.scope():
        rms_norm(x_mixed, attn_norm_w, x_normed_t)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])
    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    with pl.scope():
        late_dep = pl.system.task_dummy(deps=[rope_tid])
        qkv_proj_rope(x_normed_t, wq_a, wq_b, wq_b_scale, wkv, freqs_cos, freqs_sin, gamma_cq, gamma_ckv, q, kv, qr, qr_scale, late_dep)
        ori_block_num = pl.tensor.dim(kv_cache, 0)
        kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
        with pl.spmd(wb_blocks, name_hint='csa_cache_writeback') as ori_cache_write_tid:
            wb_blk = pl.tile.get_block_idx()
            wb_t0 = wb_blk * CSA_WB_TOKEN_TILE
            for write_dt in pl.range(CSA_WB_TOKEN_TILE):
                write_t = wb_t0 + write_dt
                write_row_i64 = pl.read(ori_slot_mapping, [write_t])
                if write_row_i64 >= 0:
                    write_row = pl.cast(write_row_i64, pl.INDEX)
                    kv_cache_flat[write_row:write_row + 1, 0:HEAD_DIM] = kv[write_t:write_t + 1, 0:HEAD_DIM]
        cmp_out = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
        (cmp_out, cmp_cache_write_tid) = compressor_ratio4(x_normed_t, cmp_out, compress_state, compress_state_block_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, cmp_cos_il, cmp_sin_signed, cmp_kv, position_ids, cmp_slot_mapping, state_slot_mapping, late_dep)
        cache_ready_dep = pl.system.task_dummy(deps=[ori_cache_write_tid, cmp_cache_write_tid])
        idx_kv_unused = pl.create_tensor([t_dim, IDX_HEAD_DIM], dtype=pl.FP32)
        (idx_topk_scores, idx_topk) = indexer(x_normed_t, qr, qr_scale, idx_wq_b, idx_wq_b_scale, weights_proj, idx_cos_il, idx_sin_signed, cmp_cos_il, cmp_sin_signed, hadamard_idx, idx_kv_unused, inner_compress_state, inner_compress_state_block_table, inner_wkv, inner_wgate, inner_ape, inner_norm_w, idx_kv_cache, idx_kv_scale, idx_block_table, idx_topk_scores, idx_topk, position_ids, idx_slot_mapping, inner_state_slot_mapping, kv_seq_lens, late_dep)
        attention_grouped = pl.create_tensor([O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], dtype=pl.BF16)
        (attention_grouped, _heads_tid) = sparse_attn_csa(q, kv_cache, window_swa_indices, cmp_kv, cmp_block_table, idx_topk, position_ids_t1, attn_sink, freqs_cos, freqs_sin, attention_grouped, cache_ready_dep)
        (attention_local_flat, attention_signal) = o_group_a2a(attention_grouped, attention_local_flat, attention_window, attention_signal, group_base, tp_rank, local_t)
    attention_local_groups = pl.reshape(attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN])
    o_local = pl.create_tensor([LOCAL_T_PAD, D], dtype=pl.BF16)
    (o_local, o_signal) = decode_sharded_o_projection_reduce_scatter(attention_local_groups, wo_a, wo_b, wo_b_scale, local_t, o_local, o_window, o_signal, group_base, tp_rank)
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    for block in pl.spmd(t_dim // CSA_WB_TOKEN_TILE, name_hint='csa_o_local_bridge'):
        token_start = block * CSA_WB_TOKEN_TILE
        attn_out[token_start:token_start + CSA_WB_TOKEN_TILE, 0:D] = o_local[token_start:token_start + CSA_WB_TOKEN_TILE, 0:D]
    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out
