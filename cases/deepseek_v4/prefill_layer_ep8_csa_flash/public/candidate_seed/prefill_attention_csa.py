from config import COFF, CSA_CMP_BLOCK_NUM, CSA_ORI_BLOCK_NUM, D, H, HC_DIM, HC_MULT, HEAD_DIM, IDX_HEAD_DIM, IDX_N_HEADS, IDX_TOPK, INNER_OUT_DIM, MAX_SEQ_LEN, MIX_HC, O_GROUPS, O_GROUP_IN, O_LORA, Q_LORA, ROPE_HEAD_DIM, SPARSE_CMP_MAX_BLOCKS, SPARSE_ORI_MAX_BLOCKS, SPARSE_TOPK, WIN

import pypto.language as pl

from config import BLOCK_SIZE, PREFILL_BATCH, PREFILL_IDX_BLOCK_NUM, PREFILL_SEQ

from prefill_compressor_ratio4 import CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_STATE_MAX_BLOCKS, prefill_compressor_ratio4

from prefill_hc_post import prefill_hc_post

from prefill_hc_pre import prefill_hc_pre

from prefill_indexer import IDX_CACHE_MAX_BLOCKS, INDEXER_SCORE_CAP, INDEXER_TOPK_CAP, prefill_indexer

from prefill_indexer_compressor import INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_MAX_BLOCKS

from prefill_qkv_proj_rope_v2 import materialize_rope_rows, qkv_proj_rope

from rmsnorm import rms_norm

from prefill_sparse_attn_v2 import prefill_sparse_attn

B = PREFILL_BATCH

S = PREFILL_SEQ

T = B * S

HALF_ROPE = ROPE_HEAD_DIM // 2

COMPRESS_RATIO = 4

MAIN_OUT_DIM = COFF * HEAD_DIM

CSA_TOPK_TOKEN_TILE = 2

WRITEBACK_GUARD_TILE = 16

@pl.jit.inline
def prefill_attention_csa(
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
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.Out[pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc_4d = pl.reshape(x_hc, [B, S, HC_MULT, D])
    x_mixed_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post_3d = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb_4d = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    prefill_hc_pre(x_hc_4d, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed_3d, post_3d, comb_4d)
    x_mixed = pl.reshape(x_mixed_3d, [T, D])
    post = pl.reshape(post_3d, [T, HC_MULT])
    comb = pl.reshape(comb_4d, [T, HC_MULT * HC_MULT])

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

    prefill_compressor_ratio4(
        x_normed, cmp_kv_state, cmp_score_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        freqs_cos, freqs_sin, cmp_kv,
        position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )
    idx_cos = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    idx_sin = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_idx_halfrope"):
        for idx_t in pl.range(T):
            idx_pos = pl.cast(pl.read(position_ids, [idx_t]), pl.INDEX)
            idx_cos = pl.assemble(idx_cos, pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [idx_pos, 0]), target_type=pl.FP32), [idx_t, 0])
            idx_sin = pl.assemble(idx_sin, pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [idx_pos, 0]), target_type=pl.FP32), [idx_t, 0])

    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    idx_score_unused = pl.create_tensor([T, INDEXER_SCORE_CAP], dtype=pl.FP32)
    prefill_indexer(
        x_normed, qr, qr_scale,
        idx_wq_b, idx_wq_b_scale, idx_weights_proj,
        idx_cos, idx_sin, freqs_cos, freqs_sin, hadamard_idx,
        inner_kv_state, inner_score_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        idx_score_unused, cmp_topk_indices,
        position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping,
    )

    cmp_sparse_lens_2d = pl.create_tensor([1, T], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_sparse_lens"):
        cmp_sparse_lens_2d = pl.assemble(cmp_sparse_lens_2d, pl.full([1, T], dtype=pl.INT32, value=SPARSE_TOPK), [0, 0])
    cmp_sparse_lens = pl.reshape(cmp_sparse_lens_2d, [T])
    cmp_sparse_work = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)

    pos_to_token = pl.create_tensor([1, MAX_SEQ_LEN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_pos_map"):
        pos_to_token[0:1, 0:MAX_SEQ_LEN] = pl.full([1, MAX_SEQ_LEN], dtype=pl.INT32, value=-1)
        for map_t in pl.range(T):
            if map_t < num_tokens:
                map_pos = pl.read(position_ids, [map_t])
                pl.write(pos_to_token, [0, pl.cast(map_pos, pl.INDEX)], pl.cast(map_t, pl.INT32))

    for topk_block in pl.spmd((T + CSA_TOPK_TOKEN_TILE - 1) // CSA_TOPK_TOKEN_TILE,
                              name_hint="prefill_csa_sparse_idx_tile"):
        topk_t0 = topk_block * CSA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(CSA_TOPK_TOKEN_TILE):
            t_idx = topk_t0 + topk_dt
            sparse_row = pl.full([1, SPARSE_TOPK], dtype=pl.INT32, value=-1)
            if t_idx < num_tokens:
                abs_pos = pl.read(position_ids, [t_idx])
                window_valid = pl.min(WIN, abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for sparse_col in pl.range(SPARSE_TOPK):
                    sparse_raw = pl.cast(-1, pl.INT32)
                    sparse_col_i32 = pl.cast(sparse_col, pl.INT32)
                    if sparse_col_i32 < window_valid:
                        key_abs = key_start_abs + sparse_col_i32
                        sparse_raw = pl.cast(key_abs % WIN, pl.INT32)
                        overlay_t = pl.read(pos_to_token, [0, pl.cast(key_abs, pl.INDEX)])
                        if overlay_t >= 0:
                            if overlay_t <= t_idx:
                                sparse_raw = pl.cast(WIN + overlay_t, pl.INT32)
                    else:
                        comp_start = window_valid
                        if sparse_col_i32 >= comp_start:
                            comp_col = sparse_col_i32 - comp_start
                            visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
                            if comp_col < visible_cmp:
                                if comp_col < pl.cast(INDEXER_TOPK_CAP, pl.INT32):
                                    topk_raw = pl.read(cmp_topk_indices, [t_idx, comp_col])
                                    if topk_raw >= WIN + T:
                                        sparse_raw = topk_raw
                    pl.write(sparse_row, [0, sparse_col], sparse_raw)
            cmp_sparse_work = pl.assemble(cmp_sparse_work, sparse_row, [t_idx, 0])

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    prefill_sparse_attn(
        q, kv_cache, ori_block_table, kv,
        cmp_kv, cmp_block_table,
        cmp_sparse_work, cmp_sparse_lens,
        attn_sink, num_tokens,
        rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )
    kv_cache_flat = pl.reshape(kv_cache, [CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_cache_writeback"):
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

    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out_4d = pl.reshape(x_out, [B, S, HC_MULT, D])
    prefill_hc_post(attn_out_3d, x_hc_4d, post_3d, comb_4d, x_out_4d)
    x_out = pl.reshape(x_out_4d, [T, HC_MULT, D])
    return kv_cache, cmp_kv, cmp_kv_state, cmp_score_state, idx_kv_cache, idx_kv_scale, inner_kv_state, inner_score_state, x_out
