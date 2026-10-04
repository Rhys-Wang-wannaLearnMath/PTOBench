import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_ORI_BLOCK_NUM, DECODE_SEQ, BLOCK_SIZE, KV_ORI_TABLE_MAX_BLOCKS
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from decode_sparse_attn_swa import sparse_attn_swa
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
from config import ORI_TABLE_MAX_BLOCKS
from config import ORI_BLOCK_NUM
NEG_INF = -1e+20

@pl.jit.inline
def attention_swa(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], swa_slot_mapping: pl.Tensor[[T], pl.INT64], swa_indices: pl.Tensor[[T, WIN], pl.INT32], swa_lens: pl.Tensor[[T], pl.INT32], position_ids: pl.Tensor[[T], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Tensor[[T, HC_MULT, D], pl.FP32]):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='swa_rope_step'):
        for b in pl.range(B):
            for s_idx in pl.range(S):
                t = b * S + s_idx
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                cos_row = pl.cast(freqs_cos[pos_b:pos_b + 1, 0:ROPE_HEAD_DIM], target_type=pl.FP32)
                sin_row = pl.cast(freqs_sin[pos_b:pos_b + 1, 0:ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t:t + 1, 0:ROPE_HEAD_DIM] = pl.cast(cos_row, target_type=pl.BF16, mode='rint')
                rope_sin_t[t:t + 1, 0:ROPE_HEAD_DIM] = pl.cast(sin_row, target_type=pl.BF16, mode='rint')
    x_normed_t = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(x_normed_t, wq_a, wq_b, wq_b_scale, wkv, rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv, q, kv, qr, qr_scale, late_dep)
    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([T, WIN], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='swa_cache_insert_valid_bias'):
        for write_t in pl.range(T):
            write_row_i64 = pl.read(swa_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row:write_row + 1, 0:HEAD_DIM] = kv[write_t:write_t + 1, 0:HEAD_DIM]
        v_col = pl.cast(pl.arange(0, [1, WIN], dtype=pl.INT32), target_type=pl.FP32)
        v_col_m = pl.col_expand(pl.full([T, WIN], dtype=pl.FP32, value=0.0), v_col)
        v_lens = pl.cast(pl.reshape(swa_lens[0:T], [T, 1]), target_type=pl.FP32)
        v_valid = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0), 1.0)
        sparse_bias[0:T, 0:WIN] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn_swa(q, kv_cache_flat, swa_indices, sparse_bias, attn_sink, rope_cos_t, rope_sin_t, wo_a, wo_b, wo_b_scale, attn_out)
    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out
