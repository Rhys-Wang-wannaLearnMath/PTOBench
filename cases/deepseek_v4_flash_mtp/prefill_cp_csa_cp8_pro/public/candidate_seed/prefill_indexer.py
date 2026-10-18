from config import COMPRESS_RATIO, CP_CANDIDATE_CAPACITY as CP_INDEXER_SCORE_CAP, SPARSE_SELECTED_WIDTH as CP_INDEXER_SELECTED_WIDTH, PREFILL_BATCH as B, PREFILL_SEQ as S, T

import pypto.language as pl

from config import CASE_CONFIG as M, BLOCK_SIZE, FP32_NEG_INF, IDX_CACHE_MAX_BLOCKS, INT8_SCALE_MAX, INT8_AMAX_EPS

D = M.hidden_size

ROPE_HEAD_DIM = M.qk_rope_head_dim

IDX_N_HEADS = M.index_n_heads

IDX_HEAD_DIM = M.index_head_dim

IDX_NOPE_HEAD_DIM = M.index_nope_head_dim

Q_LORA = M.q_lora_rank

WEIGHTS_SCALE = M.index_weights_scale


IDX_TOPK = M.index_topk

CACHE_TILE = 32


CP_INDEXER_SORT_LEN = 2048





SCORE_TOKEN_TILE = 8

TOPK_TILE = 4

IDX_BLOCK_NUM_DYN = pl.dynamic('PREFILL_IDX_BLOCK_NUM_DYN')

Q_TILE = 128

Q_OUT_TILE = 256

QR_PROJ_ROW_TILE = 16

HEAD_DIM_TILE = 32

D_TILE = 32

WEIGHTS_ROW_TILE = 32

QH_QUANT_ROW_TILE = 64

ROPE_ROW_BLOCK = IDX_N_HEADS

ROPE_PREP_TOKEN_TILE = 16

QH_MM_TILE = 64

SCORE_INIT_TILE = 16

@pl.jit.inline
def _prefill_indexer_cp_score_topk(x: pl.Tensor[[T, D], pl.BF16], qr: pl.Tensor[[T, Q_LORA], pl.INT8], qr_scale: pl.Tensor[[T, 1], pl.FP32], wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32], sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32], hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32], idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32], position_ids: pl.Tensor[[T], pl.INT32], num_tokens: pl.Scalar[pl.INT32], score: pl.Out[pl.Tensor[[T, CP_INDEXER_SCORE_CAP], pl.FP32]], cmp_topk_indices: pl.Out[pl.Tensor[[T, IDX_TOPK], pl.INT32]]):
    """Compute CP indexer query scores and top-k selections."""
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint='prefill_cp_idx_qr_proj'):
        o0 = idx * Q_OUT_TILE
        qr_acc = pl.create_tensor([T, Q_OUT_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
            q0 = kb * Q_TILE
            qr_tile = qr[:, q0:q0 + Q_TILE]
            wq_tile = wq_b[q0:q0 + Q_TILE, o0:o0 + Q_OUT_TILE]
            if q0 == 0:
                qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
            else:
                qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
        wq_scale = pl.reshape(wq_b_scale[o0:o0 + Q_OUT_TILE], [1, Q_OUT_TILE])
        for r0 in pl.range(0, T, QR_PROJ_ROW_TILE):
            qr_acc_tile = qr_acc[r0:r0 + QR_PROJ_ROW_TILE, :]
            acc_fp32 = pl.cast(qr_acc_tile, target_type=pl.FP32, mode='none')
            scale_dq = qr_scale[r0:r0 + QR_PROJ_ROW_TILE, :]
            qr_scaled = pl.row_expand_mul(acc_fp32, scale_dq)
            qr_dequant = pl.col_expand_mul(qr_scaled, wq_scale)
            qr_proj[r0:r0 + QR_PROJ_ROW_TILE, o0:o0 + Q_OUT_TILE] = qr_dequant
    rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    for prep_idx in pl.spmd(T // ROPE_PREP_TOKEN_TILE, name_hint='prefill_cp_idx_rope_prepare', allow_early_resolve=True):
        t0 = prep_idx * ROPE_PREP_TOKEN_TILE
        rope_ones = pl.full([ROPE_PREP_TOKEN_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col_i32 = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        rope_col_fp32 = pl.cast(rope_col_i32, target_type=pl.FP32)
        rope_col = pl.col_expand_mul(rope_ones, rope_col_fp32)
        rope_dup_i32 = pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode='trunc')
        rope_dup_f = pl.cast(rope_dup_i32, target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_tile = cos[t0:t0 + ROPE_PREP_TOKEN_TILE, 0:ROPE_HEAD_DIM // 2]
        sin_tile = sin[t0:t0 + ROPE_PREP_TOKEN_TILE, 0:ROPE_HEAD_DIM // 2]
        cos_il = pl.gather(cos_tile, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_tile, dim=-1, index=rope_dup_idx)
        rope_cos_il[t0:t0 + ROPE_PREP_TOKEN_TILE, :] = cos_il
        rope_sin_signed[t0:t0 + ROPE_PREP_TOKEN_TILE, :] = pl.mul(sin_il, rope_sign)
    rope_swap_idx = pl.create_tensor([ROPE_ROW_BLOCK, ROPE_HEAD_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_cp_idx_rope_swap_idx', allow_early_resolve=True):
        swap_ones = pl.full([ROPE_ROW_BLOCK, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        swap_col_i32 = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        swap_col_fp32 = pl.cast(swap_col_i32, target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_col_fp32)
        swap_dup_i32 = pl.cast(pl.mul(swap_col, 0.5), target_type=pl.INT32, mode='trunc')
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))
        swap_index = pl.cast(pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0)), target_type=pl.INT32)
        rope_swap_idx[0:ROPE_ROW_BLOCK, 0:ROPE_HEAD_DIM] = swap_index
    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    for token_idx in pl.spmd(T, name_hint='prefill_cp_idx_qr_rope', allow_early_resolve=True):
        r0 = token_idx * ROPE_ROW_BLOCK
        qr_nope_fp32 = qr_proj_flat[r0:r0 + ROPE_ROW_BLOCK, 0:IDX_NOPE_HEAD_DIM]
        qr_nope = pl.cast(qr_nope_fp32, target_type=pl.BF16, mode='rint')
        qr_rope = qr_proj_flat[r0:r0 + ROPE_ROW_BLOCK, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
        rope_swap_tile = rope_swap_idx[0:ROPE_ROW_BLOCK, 0:ROPE_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope, dim=-1, index=rope_swap_tile)
        rope_cos_tile = rope_cos_il[token_idx:token_idx + 1, :]
        rope_sin_tile = rope_sin_signed[token_idx:token_idx + 1, :]
        rope_main = pl.col_expand_mul(qr_rope, rope_cos_tile)
        rope_swapped = pl.col_expand_mul(qr_swapped, rope_sin_tile)
        rope_rot = pl.add(rope_main, rope_swapped)
        rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode='rint')
        qr_bf16[r0:r0 + ROPE_ROW_BLOCK, :] = pl.concat(qr_nope, rope_bf16)
    qh_acc_gm = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for mm_idx in pl.spmd(T * IDX_N_HEADS // QH_MM_TILE, name_hint='prefill_cp_idx_qr_hadamard', allow_early_resolve=True):
        r0 = mm_idx * QH_MM_TILE
        qr_bf16_tile = qr_bf16[r0:r0 + QH_MM_TILE, :]
        qh_acc = pl.matmul(qr_bf16_tile, hadamard, out_dtype=pl.FP32)
        qh_acc_gm[r0:r0 + QH_MM_TILE, :] = qh_acc
    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for quant_idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_ROW_TILE, name_hint='prefill_cp_idx_qr_quant', allow_early_resolve=True):
        r0 = quant_idx * QH_QUANT_ROW_TILE
        qh_amax = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_tile = qh_acc_gm[r0:r0 + QH_QUANT_ROW_TILE, h0:h0 + HEAD_DIM_TILE]
            qh_abs = pl.maximum(qh_tile, pl.neg(qh_tile))
            qh_row_max = pl.reshape(pl.row_max(qh_abs), [1, QH_QUANT_ROW_TILE])
            qh_amax = pl.maximum(qh_amax, qh_row_max)
        scale_max = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        scale_quant_row = pl.div(scale_max, qh_amax)
        qr_scale_tile = pl.reshape(pl.recip(scale_quant_row), [QH_QUANT_ROW_TILE, 1])
        qr_hadamard_scale_dq[r0:r0 + QH_QUANT_ROW_TILE, :] = qr_scale_tile
        scale_quant = pl.reshape(scale_quant_row, [QH_QUANT_ROW_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_quant_tile = qh_acc_gm[r0:r0 + QH_QUANT_ROW_TILE, h1:h1 + HEAD_DIM_TILE]
            qh_scaled = pl.row_expand_mul(qh_quant_tile, scale_quant)
            qh_i32 = pl.cast(qh_scaled, target_type=pl.INT32, mode='rint')
            qh_half = pl.cast(qh_i32, target_type=pl.FP16, mode='round')
            qh_i8 = pl.cast(qh_half, target_type=pl.INT8, mode='trunc')
            qr_hadamard_i8[r0:r0 + QH_QUANT_ROW_TILE, h1:h1 + HEAD_DIM_TILE] = qh_i8
    weights = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    for idx in pl.spmd(T // WEIGHTS_ROW_TILE, name_hint='prefill_cp_idx_weights_proj'):
        wrow0 = idx * WEIGHTS_ROW_TILE
        weights_acc = pl.create_tensor([WEIGHTS_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.pipeline(0, D // D_TILE, stage=2):
            d0 = db * D_TILE
            x_tile = x[wrow0:wrow0 + WEIGHTS_ROW_TILE, d0:d0 + D_TILE]
            wp_tile = weights_proj[d0:d0 + D_TILE, :]
            if d0 == 0:
                weights_acc = pl.matmul(x_tile, wp_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, wp_tile)
        weights[wrow0:wrow0 + WEIGHTS_ROW_TILE, :] = pl.mul(weights_acc, WEIGHTS_SCALE)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_block_num * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * BLOCK_SIZE, 1])
    score_wide = pl.create_tensor([T, CP_INDEXER_SORT_LEN], dtype=pl.FP32)
    for si in pl.parallel(0, T, SCORE_INIT_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_cp_idx_score_init'):
            score_init_tile = pl.full([SCORE_INIT_TILE, CP_INDEXER_SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
            score_wide[si:si + SCORE_INIT_TILE, :] = score_init_tile
    for score_idx in pl.spmd(T // SCORE_TOKEN_TILE, name_hint='prefill_cp_idx_score'):
        token0 = score_idx * SCORE_TOKEN_TILE
        last_pos = pl.read(position_ids, [num_tokens - 1])
        visible_limit = pl.min((last_pos + 1) // COMPRESS_RATIO, CP_INDEXER_SCORE_CAP)
        for cb in pl.range(CP_INDEXER_SCORE_CAP // CACHE_TILE):
            cache0 = cb * CACHE_TILE
            logical_block = cache0 // BLOCK_SIZE
            if visible_limit > cache0 and logical_block < IDX_CACHE_MAX_BLOCKS:
                physical_block_raw = pl.read(idx_block_table, [logical_block])
                if physical_block_raw >= 0 and physical_block_raw < idx_block_num:
                    physical_block = pl.cast(physical_block_raw, pl.INDEX)
                    kv_row0 = physical_block * BLOCK_SIZE + cache0 % BLOCK_SIZE
                    kv_q_i8_full = kv_cache_i8_flat[kv_row0:kv_row0 + CACHE_TILE, 0:IDX_HEAD_DIM]
                    kv_cache_scale_dq = kv_scale_flat[kv_row0:kv_row0 + CACHE_TILE, :]
                    for token_offset in pl.range(SCORE_TOKEN_TILE):
                        t = token0 + token_offset
                        if t < num_tokens:
                            q_s0 = t * IDX_N_HEADS
                            qr_hadamard_i8_tile = qr_hadamard_i8[q_s0:q_s0 + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                            score_acc_s = pl.matmul(kv_q_i8_full, qr_hadamard_i8_tile, out_dtype=pl.INT32, b_trans=True)
                            qh_scale_source = qr_hadamard_scale_dq[q_s0:q_s0 + IDX_N_HEADS, :]
                            qh_scale_s = pl.reshape(qh_scale_source, [1, IDX_N_HEADS])
                            score_acc_fp32 = pl.cast(score_acc_s, target_type=pl.FP32, mode='none')
                            score_row_scaled = pl.row_expand_mul(score_acc_fp32, kv_cache_scale_dq)
                            score_scaled = pl.col_expand_mul(score_row_scaled, qh_scale_s)
                            score_zero = pl.mul(score_scaled, 0.0)
                            relu_score_s = pl.maximum(score_scaled, score_zero)
                            weighted_heads = pl.col_expand_mul(relu_score_s, weights[t:t + 1, :])
                            weighted_sum = pl.row_sum(weighted_heads)
                            weighted_score_s = pl.reshape(weighted_sum, [1, CACHE_TILE])
                            pos = pl.read(position_ids, [t])
                            visible_t = pl.min((pos + 1) // COMPRESS_RATIO, CP_INDEXER_SCORE_CAP)
                            if visible_t > cache0:
                                valid_len_t = pl.min(CACHE_TILE, visible_t - cache0)
                            else:
                                valid_len_t = 0
                            weighted_visible = pl.set_validshape(weighted_score_s, 1, valid_len_t)
                            weighted_valid_t = pl.fillpad(weighted_visible, pad_value=pl.PadValue.min)
                            neg_inf_tile = pl.full([1, CACHE_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            weighted_valid_t = pl.maximum(weighted_valid_t, neg_inf_tile)
                            score_wide[t:t + 1, cache0:cache0 + CACHE_TILE] = weighted_valid_t
    for score_out0 in pl.parallel(0, T, SCORE_INIT_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_cp_idx_score_out'):
            score_tile = score_wide[score_out0:score_out0 + SCORE_INIT_TILE, 0:CP_INDEXER_SCORE_CAP]
            score[score_out0:score_out0 + SCORE_INIT_TILE, :] = score_tile
    for topk_idx in pl.spmd(T // TOPK_TILE, name_hint='prefill_cp_idx_topk'):
        t0 = topk_idx * TOPK_TILE
        for ti in pl.range(TOPK_TILE):
            t = t0 + ti
            topk_init = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            cmp_topk_indices[t:t + 1, 0:IDX_TOPK] = topk_init
            if t < num_tokens:
                pos = pl.read(position_ids, [t])
                visible_t = pl.min((pos + 1) // COMPRESS_RATIO, CP_INDEXER_SCORE_CAP)
                if visible_t > 0:
                    score_row = score_wide[t:t + 1, :]
                    idx_init = pl.arange(0, [1, CP_INDEXER_SORT_LEN], dtype=pl.UINT32)
                    sorted_tile = pl.sort32(score_row, idx_init)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=64)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=256)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=1024)
                    topk_pairs = sorted_tile[:, 0:2 * CP_INDEXER_SELECTED_WIDTH]
                    topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    valid_topk = pl.min(CP_INDEXER_SELECTED_WIDTH, visible_t)
                    topk_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
                    cmp_topk_indices[t:t + 1, 0:CP_INDEXER_SELECTED_WIDTH] = topk_valid
    return (score, cmp_topk_indices)
