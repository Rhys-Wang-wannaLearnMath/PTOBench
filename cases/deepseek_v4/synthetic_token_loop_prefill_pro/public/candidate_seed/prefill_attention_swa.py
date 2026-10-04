import pypto.language as pl
from config import BLOCK_SIZE, CASE_CONFIG as M, PREFILL_BATCH, PREFILL_CMP_BLOCK_NUM, PREFILL_CMP_MAX_BLOCKS, PREFILL_ORI_BLOCK_NUM, PREFILL_SEQ
from hc_post import hc_post_prefill
from hc_pre import hc_pre
from qkv_proj_rope import materialize_rope_rows, qkv_proj_rope
from rmsnorm import rms_norm
from prefill_sparse_attn import prefill_sparse_attn
BLOCK_NUM_DYN = pl.dynamic('PREFILL_ORI_BLOCK_NUM_DYN')
from config import B
from config import S
from config import T
from config import D
from config import H
from config import HEAD_DIM
ROPE_DIM = M.qk_rope_head_dim
from config import ROPE_HEAD_DIM
from config import Q_LORA
from config import MAX_SEQ_LEN
WIN = M.sliding_window
IDX_TOPK = M.index_topk
from config import HC_MULT
from config import MIX_HC
from config import HC_DIM
from config import O_LORA
from config import O_GROUPS
HEADS_PER_GROUP = H // O_GROUPS
from config import O_GROUP_IN
BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
from config import SPARSE_CMP_MAX_BLOCKS

@pl.jit.inline
def prefill_attention_swa(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], block_table: pl.Tensor[[BLOCK_NUM], pl.INT32], ori_slot_mapping: pl.Tensor[[T], pl.INT64], position_ids: pl.Tensor[[T], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]], num_tokens: pl.Scalar[pl.INT32]):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)
    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(freqs_cos, freqs_sin, position_ids, num_tokens, rope_cos_t, rope_sin_t)
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(x_normed, wq_a, wq_b, wq_b_scale, wkv, rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv, q, kv, qr, qr_scale, late_dep)
    block_num = pl.tensor.dim(kv_cache, 0)
    cache_rows = block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [cache_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_swa_cache_write'):
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row:write_row + 1, :] = kv[write_t:write_t + 1, :]
    swa_indices = pl.create_tensor([T, WIN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_swa_window_indices'):
        for idx_t in pl.range(T):
            idx_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            if idx_t < num_tokens:
                abs_pos = pl.read(position_ids, [idx_t])
                window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for win_col in pl.range(WIN):
                    win_col_i32 = pl.cast(win_col, pl.INT32)
                    if win_col_i32 < window_valid:
                        key_abs = key_start_abs + win_col_i32
                        blk_slot = key_abs // BLOCK_SIZE
                        blk = pl.read(block_table, [pl.cast(blk_slot, pl.INDEX)])
                        if blk >= 0:
                            row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                            pl.write(idx_row, [0, win_col], row)
            swa_indices = pl.assemble(swa_indices, idx_row, [idx_t, 0])
    cmp_block_table_dummy = pl.create_tensor([SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_swa_dummy_cmp_table'):
        for dummy_blk in pl.range(SPARSE_CMP_MAX_BLOCKS):
            pl.write(cmp_block_table_dummy, [dummy_blk], pl.cast(0, pl.INT32))
    cmp_kv_dummy = pl.create_tensor([CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_indices_dummy = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_swa_empty_cmp_meta'):
        for cmp_t in pl.range(T):
            cmp_indices_dummy[cmp_t:cmp_t + 1, 0:IDX_TOPK] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    prefill_sparse_attn(q, kv_cache, swa_indices, cmp_kv_dummy, cmp_block_table_dummy, cmp_indices_dummy, attn_sink, num_tokens, rope_cos_t, rope_sin_t, wo_a, wo_b, wo_b_scale, attn_out)
    hc_post_prefill(attn_out, x_hc, post, comb, x_out, num_tokens)
    return (kv_cache, x_out)
