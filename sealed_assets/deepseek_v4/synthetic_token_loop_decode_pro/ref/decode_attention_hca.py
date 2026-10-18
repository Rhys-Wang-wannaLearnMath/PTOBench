import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C128_COMPRESSOR_BLOCK_SIZE, DECODE_CMP_BLOCK_NUM, DECODE_ORI_BLOCK_NUM, KV_CMP_MAX_BLOCKS
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from decode_compressor_ratio128 import compressor_ratio128
from decode_sparse_attn_hca import sparse_attn_hca, CMP_TOPK as HCA_SPARSE_CMP_TOPK
from config import B
from config import S
from config import T
from config import D
from config import H
from config import HEAD_DIM
from config import ROPE_HEAD_DIM
from config import Q_LORA
WIN = M.sliding_window
from config import HC_MULT
from config import MIX_HC
from config import HC_DIM
from config import MAX_SEQ_LEN
from config import O_LORA
from config import O_GROUPS
from config import O_GROUP_IN
COMPRESS_RATIO = 128
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
from config import ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = 64
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
COMPRESS_TOPK = MAX_SEQ_LEN // COMPRESS_RATIO
SPARSE_IDX_TOPK = M.index_topk
HCA_TOPK_LIMIT = min(COMPRESS_TOPK, SPARSE_IDX_TOPK)
HCA_CMP_TOPK = HCA_SPARSE_CMP_TOPK
HCA_TOPK_TOKEN_TILE = 8
HCA_WB_TOKEN_TILE = 8

@pl.jit.inline
def attention_hca(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32], compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32], ori_slot_mapping: pl.Tensor[[T], pl.INT64], window_swa_indices: pl.Tensor[[T, WIN], pl.INT32], window_swa_lens: pl.Tensor[[T], pl.INT32], cmp_slot_mapping: pl.Tensor[[T], pl.INT64], state_slot_mapping: pl.Tensor[[T], pl.INT64], position_ids: pl.Tensor[[T], pl.INT32], kv_seq_lens: pl.Tensor[[B], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Tensor[[T, HC_MULT, D], pl.FP32]):
    """HCA decode orchestration for compress_ratio=128."""
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    cmp_cos = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='hca_rope'):
        for b in pl.range(B):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            cmp_offset_b = COMPRESS_RATIO - first_pos_b % COMPRESS_RATIO
            cmp_pos_b = pl.cast(first_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos_row = freqs_cos[cmp_pos_b:cmp_pos_b + 1, 0:ROPE_HEAD_DIM // 2]
            cmp_sin_row = freqs_sin[cmp_pos_b:cmp_pos_b + 1, 0:ROPE_HEAD_DIM // 2]
            cmp_cos[b:b + 1, 0:ROPE_HEAD_DIM // 2] = pl.cast(cmp_cos_row, target_type=pl.FP32)
            cmp_sin[b:b + 1, 0:ROPE_HEAD_DIM // 2] = pl.cast(cmp_sin_row, target_type=pl.FP32)
            for s in pl.range(S):
                t = b * S + s
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                step_cos_row = pl.cast(freqs_cos[pos_b:pos_b + 1, 0:ROPE_HEAD_DIM], target_type=pl.FP32)
                step_sin_row = pl.cast(freqs_sin[pos_b:pos_b + 1, 0:ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t:t + 1, 0:ROPE_HEAD_DIM] = pl.cast(step_cos_row, target_type=pl.BF16, mode='rint')
                rope_sin_t[t:t + 1, 0:ROPE_HEAD_DIM] = pl.cast(step_sin_row, target_type=pl.BF16, mode='rint')
    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(x_normed, wq_a, wq_b, wq_b_scale, wkv, rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv, q, kv, qr, qr_scale, late_dep)
    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for wb_blk in pl.spmd(T // HCA_WB_TOKEN_TILE, name_hint='hca_cache_writeback'):
        wb_t0 = wb_blk * HCA_WB_TOKEN_TILE
        for write_dt in pl.range(HCA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            write_row_i64 = pl.read(ori_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row:write_row + 1, 0:HEAD_DIM] = kv[write_t:write_t + 1, 0:HEAD_DIM]
    x_normed_bsd = pl.reshape(x_normed, [B, S, D])
    cmp_kv_proj = pl.create_tensor([B, S, HEAD_DIM], dtype=pl.FP32)
    position_ids_bsd = pl.reshape(position_ids, [B, S])
    cmp_slot_mapping_bsd = pl.reshape(cmp_slot_mapping, [B, S])
    state_slot_mapping_bsd = pl.reshape(state_slot_mapping, [B, S])
    compressor_ratio128(x_normed_bsd, cmp_kv_proj, compress_state, compress_state_block_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, cmp_cos, cmp_sin, cmp_kv, position_ids_bsd, cmp_slot_mapping_bsd, state_slot_mapping_bsd, late_dep)
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    topk_all = pl.create_tensor([T, HCA_CMP_TOPK], dtype=pl.INT32)
    for topk_block in pl.spmd(T // HCA_TOPK_TOKEN_TILE, name_hint='hca_cache_topk'):
        topk_t0 = topk_block * HCA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(HCA_TOPK_TOKEN_TILE):
            topk_t = topk_t0 + topk_dt
            if topk_t < T:
                topk_b = topk_t // S
                topk_abs_pos = pl.read(position_ids, [topk_t])
                topk_cmp_valid = pl.min(HCA_TOPK_LIMIT, pl.min((topk_abs_pos + 1) // COMPRESS_RATIO, pl.read(kv_seq_lens, [topk_b]) // COMPRESS_RATIO))
                for topk_ck in pl.range(HCA_CMP_TOPK):
                    if topk_ck < topk_cmp_valid:
                        pl.write(topk_all, [topk_t, topk_ck], pl.cast(topk_ck, pl.INT32))
                    else:
                        pl.write(topk_all, [topk_t, topk_ck], pl.cast(-1, pl.INT32))
    sparse_attn_hca(q, kv_cache, window_swa_indices, cmp_kv, cmp_block_table, topk_all, attn_sink, rope_cos_t, rope_sin_t, wo_a, wo_b, wo_b_scale, attn_out)
    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out
