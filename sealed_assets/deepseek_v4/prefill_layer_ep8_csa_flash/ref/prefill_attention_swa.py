from config import D, H, HC_DIM, HC_MULT, HEAD_DIM, MAX_SEQ_LEN, MIX_HC, O_GROUPS, O_GROUP_IN, O_LORA, Q_LORA, ROPE_HEAD_DIM, SPARSE_CMP_MAX_BLOCKS, SPARSE_TOPK

import pypto.language as pl

from config import BLOCK_SIZE, PREFILL_BATCH, PREFILL_CMP_BLOCK_NUM, PREFILL_ORI_BLOCK_NUM, PREFILL_SEQ

from hc_post_v2 import hc_post

from hc_pre_v2 import hc_pre

from prefill_qkv_proj_rope_v2 import materialize_rope_rows, qkv_proj_rope

from rmsnorm import rms_norm

from prefill_sparse_attn_v2 import prefill_sparse_attn

B = PREFILL_BATCH

S = PREFILL_SEQ

T = B * S

BLOCK_NUM = PREFILL_ORI_BLOCK_NUM

CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM

WRITEBACK_GUARD_TILE = 16

@pl.jit.inline
def prefill_attention_swa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[BLOCK_NUM], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_norm(x_mixed, attn_norm_w, x_normed)

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        position_ids,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
    )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale,
    )

    cmp_block_table_dummy = pl.create_tensor([SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_dummy_cmp_table"):
        for dummy_blk in pl.range(SPARSE_CMP_MAX_BLOCKS):
            pl.write(cmp_block_table_dummy, [dummy_blk], pl.cast(0, pl.INT32))
    cmp_kv_dummy = pl.create_tensor([CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    prefill_sparse_attn(
        q, kv_cache, block_table, kv,
        cmp_kv_dummy, cmp_block_table_dummy,
        cmp_sparse_indices, cmp_sparse_lens,
        attn_sink, num_tokens,
        rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )
    kv_cache_flat = pl.reshape(kv_cache, [BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cache_writeback"):
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    write_guard = pl.cast(attn_out[write_t : write_t + 1, 0:WRITEBACK_GUARD_TILE], target_type=pl.FP32)
                    write_zero = pl.mul(write_guard, 0.0)
                    write_kv_head = pl.cast(kv[write_t : write_t + 1, 0:WRITEBACK_GUARD_TILE], target_type=pl.FP32)
                    kv_cache_flat[write_row : write_row + 1, 0:WRITEBACK_GUARD_TILE] = pl.cast(
                        pl.add(write_kv_head, write_zero), target_type=pl.BF16)
                    kv_cache_flat[write_row : write_row + 1, WRITEBACK_GUARD_TILE:HEAD_DIM] = kv[
                        write_t : write_t + 1, WRITEBACK_GUARD_TILE:HEAD_DIM]

    hc_post(attn_out, x_hc, post, comb, x_out)
    return kv_cache, x_out
