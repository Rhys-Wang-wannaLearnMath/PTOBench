import pypto.language as pl
from config import CASE_CONFIG as M, BLOCK_SIZE, FP32_NEG_INF, IDX_CACHE_MAX_BLOCKS, INT8_SCALE_MAX, INT8_AMAX_EPS
from prefill_indexer_compressor import INNER_STATE_BLOCK_SIZE, INNER_STATE_MAX_BLOCKS, prefill_indexer_compressor
IDX_BLOCK_NUM_DYN = pl.dynamic('PREFILL_IDX_BLOCK_NUM_DYN')
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic('PREFILL_INNER_STATE_BLOCK_NUM_DYN')
from config import D
from config import ROPE_HEAD_DIM
from config import IDX_N_HEADS
from config import IDX_HEAD_DIM
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
from config import Q_LORA
WEIGHTS_SCALE = M.index_weights_scale
from config import MAX_SEQ_LEN
COMPRESS_RATIO = 4
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
from config import INNER_OUT_DIM
INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
CACHE_TILE = 32
INDEXER_SCORE_MAX_BLOCKS = 2
from config import B
from config import S
from config import T
TOPK_TILE = 16
INDEXER_SCORE_CAP = INDEXER_SCORE_MAX_BLOCKS * BLOCK_SIZE
INDEXER_SCORE_BLOCKS = max(1, (INDEXER_SCORE_CAP + CACHE_TILE - 1) // CACHE_TILE)
Q_TILE = 128
Q_OUT_TILE = 256
QR_PROJ_ROW_TILE = 16
HEAD_DIM_TILE = 32
D_TILE = 32
WEIGHTS_ROW_TILE = 32
QH_QUANT_BLOCK = 256
QH_QUANT_ROW_TILE = 64
ROPE_ROW_BLOCK = IDX_N_HEADS
ROPE_ROW_TILE = 32
SORT_LEN = 2048
MRG_TOPK_RUN = 1024
PREFILL_TOPK_CAP = IDX_TOPK
SCORE_INIT_TILE = 16

@pl.jit.inline
def prefill_indexer(x: pl.Tensor[[T, D], pl.BF16], qr: pl.Tensor[[T, Q_LORA], pl.INT8], qr_scale: pl.Tensor[[T, 1], pl.FP32], wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32], sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16], idx_kv_cache: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32], score: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.FP32]], cmp_topk_indices: pl.Out[pl.Tensor[[T, IDX_TOPK], pl.INT32]], position_ids: pl.Tensor[[T], pl.INT32], num_tokens: pl.Scalar[pl.INT32], idx_slot_mapping: pl.Tensor[[T], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T], pl.INT64]):
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint='prefill_idx_qr_proj'):
        o0 = idx * Q_OUT_TILE
        wq_scale = pl.reshape(wq_b_scale[o0:o0 + Q_OUT_TILE], [1, Q_OUT_TILE])
        for r0 in pl.range(0, T, QR_PROJ_ROW_TILE):
            qr_acc = pl.create_tensor([QR_PROJ_ROW_TILE, Q_OUT_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = qr[r0:r0 + QR_PROJ_ROW_TILE, q0:q0 + Q_TILE]
                wq_tile = wq_b[q0:q0 + Q_TILE, o0:o0 + Q_OUT_TILE]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            acc_fp32 = pl.cast(qr_acc, target_type=pl.FP32, mode='none')
            scale_dq = qr_scale[r0:r0 + QR_PROJ_ROW_TILE, :]
            qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, scale_dq), wq_scale)
            qr_proj[r0:r0 + QR_PROJ_ROW_TILE, o0:o0 + Q_OUT_TILE] = qr_dequant
    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_rope_out = pl.create_tensor([T * IDX_N_HEADS, ROPE_HEAD_DIM], dtype=pl.BF16)
    for idx in pl.spmd(T * IDX_N_HEADS // ROPE_ROW_BLOCK, name_hint='prefill_idx_qr_rope'):
        o0 = idx * ROPE_ROW_BLOCK
        token_idx = idx
        cos_b = cos[token_idx:token_idx + 1, 0:ROPE_HEAD_DIM // 2]
        sin_b = sin[token_idx:token_idx + 1, 0:ROPE_HEAD_DIM // 2]
        rope_ones = pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_b32 = pl.col_expand_mul(pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=1.0), cos_b)
        sin_b32 = pl.col_expand_mul(pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=1.0), sin_b)
        cos_il = pl.gather(cos_b32, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b32, dim=-1, index=rope_dup_idx)
        for ro in pl.range(0, ROPE_ROW_BLOCK, ROPE_ROW_TILE):
            r0 = o0 + ro
            qr_rope_slice = qr_proj_flat[r0:r0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
            qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
            rope_rot = pl.add(pl.mul(qr_rope_slice, cos_il), pl.mul(pl.mul(qr_swapped, rope_sign), sin_il))
            qr_rope_out[r0:r0 + ROPE_ROW_TILE, :] = pl.cast(rope_rot, target_type=pl.BF16, mode='rint')
    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    qh_acc_buffer = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_BLOCK, name_hint='prefill_idx_qr_hadamard_quant'):
        o0 = idx * QH_QUANT_BLOCK
        for ro in pl.range(0, QH_QUANT_BLOCK, QH_QUANT_ROW_TILE):
            qh_nope = pl.cast(qr_proj_flat[o0 + ro:o0 + ro + QH_QUANT_ROW_TILE, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode='rint')
            qh_rope = qr_rope_out[o0 + ro:o0 + ro + QH_QUANT_ROW_TILE, :]
            qh_acc = pl.matmul(qh_nope, hadamard[0:IDX_NOPE_HEAD_DIM, :], out_dtype=pl.FP32)
            qh_acc = pl.matmul_acc(qh_acc, qh_rope, hadamard[IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM, :])
            qh_acc_buffer[o0 + ro:o0 + ro + QH_QUANT_ROW_TILE, :] = qh_acc
    for idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_BLOCK, name_hint="prefill_idx_qr_quant_materialized"):
        o0 = idx * QH_QUANT_BLOCK
        for ro in pl.range(0, QH_QUANT_BLOCK, QH_QUANT_ROW_TILE):
            qh_acc = qh_acc_buffer[o0 + ro:o0 + ro + QH_QUANT_ROW_TILE, :]
            qh_amax = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
                qh_a_f32 = qh_acc[0:QH_QUANT_ROW_TILE, h0:h0 + HEAD_DIM_TILE]
                qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
                qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_ROW_TILE])
                qh_amax = pl.maximum(qh_amax, qh_a_max)
            qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
            qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_ROW_TILE, 1])
            qr_hadamard_scale_dq[o0 + ro:o0 + ro + QH_QUANT_ROW_TILE, :] = qh_scale_dq
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_ROW_TILE, 1])
            for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
                qh_q_f32 = qh_acc[0:QH_QUANT_ROW_TILE, h1:h1 + HEAD_DIM_TILE]
                qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
                qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode='rint')
                qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode='round')
                qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode='trunc')
                qr_hadamard_i8[o0 + ro:o0 + ro + QH_QUANT_ROW_TILE, h1:h1 + HEAD_DIM_TILE] = qh_i8
    weights = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    for idx in pl.spmd(T // WEIGHTS_ROW_TILE, name_hint='prefill_idx_weights_proj'):
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
    compressor_completion = pl.array.create(1, pl.TASK_ID)
    prefill_indexer_compressor(x, inner_compress_state, inner_compress_state_block_table, inner_wkv, inner_wgate, inner_ape, inner_norm_w, freqs_cos, freqs_sin, hadamard, idx_kv_cache, idx_kv_scale, idx_block_table, position_ids, num_tokens, idx_slot_mapping, inner_state_slot_mapping, compressor_completion)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_block_num * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * BLOCK_SIZE, 1])
    score_wide = pl.create_tensor([T, SORT_LEN], dtype=pl.FP32)
    for si in pl.parallel(0, T, SCORE_INIT_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_idx_score_init'):
            score_wide[si:si + SCORE_INIT_TILE, :] = pl.full([SCORE_INIT_TILE, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_idx_score', deps=[compressor_completion[0]]):
        last_pos = pl.read(position_ids, [num_tokens - 1])
        max_visible = pl.min((last_pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
        for cb in pl.range(INDEXER_SCORE_BLOCKS):
            cache0 = cb * CACHE_TILE
            if max_visible > cache0:
                idx_blk_id = pl.cast(pl.read(idx_block_table, [cache0 // BLOCK_SIZE]), pl.INDEX)
                kv_row0 = idx_blk_id * BLOCK_SIZE + cache0 % BLOCK_SIZE
                kv_q_i8_full = kv_cache_i8_flat[kv_row0:kv_row0 + CACHE_TILE, 0:IDX_HEAD_DIM]
                kv_cache_scale_dq = kv_scale_flat[kv_row0:kv_row0 + CACHE_TILE, :]
                for t in pl.range(T):
                    if t < num_tokens:
                        q_s0 = t * IDX_N_HEADS
                        qr_hadamard_tile = qr_hadamard_i8[q_s0:q_s0 + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                        score_acc_s = pl.matmul(kv_q_i8_full, qr_hadamard_tile, out_dtype=pl.INT32, b_trans=True)
                        qh_scale_s = pl.reshape(qr_hadamard_scale_dq[q_s0:q_s0 + IDX_N_HEADS, :], [1, IDX_N_HEADS])
                        score_tile_s = pl.cast(score_acc_s, target_type=pl.FP32, mode='none')
                        score_tile_s = pl.col_expand_mul(pl.row_expand_mul(score_tile_s, kv_cache_scale_dq), qh_scale_s)
                        relu_score_s = pl.maximum(score_tile_s, pl.mul(score_tile_s, 0.0))
                        weighted_score_s = pl.reshape(pl.row_sum(pl.col_expand_mul(relu_score_s, weights[t:t + 1, :])), [1, CACHE_TILE])
                        pos = pl.read(position_ids, [t])
                        visible_t = pl.min((pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
                        if visible_t > cache0:
                            valid_len_t = pl.min(CACHE_TILE, visible_t - cache0)
                        else:
                            valid_len_t = 0
                        weighted_valid_t = pl.fillpad(pl.set_validshape(weighted_score_s, 1, valid_len_t), pad_value=pl.PadValue.min)
                        weighted_valid_t = pl.maximum(weighted_valid_t, pl.full([1, CACHE_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
                        score_wide[t:t + 1, cache0:cache0 + CACHE_TILE] = weighted_valid_t
    score_out_flat = pl.reshape(score, [T, INDEXER_SCORE_CAP])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_idx_score_out'):
        score_out_flat[0:T, :] = score_wide[0:T, 0:INDEXER_SCORE_CAP]
    for topk_idx in pl.spmd(T // TOPK_TILE, name_hint='prefill_idx_topk'):
        t0 = topk_idx * TOPK_TILE
        for ti in pl.range(TOPK_TILE):
            t = t0 + ti
            cmp_topk_indices[t:t + 1, 0:IDX_TOPK] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            if t < num_tokens:
                pos = pl.read(position_ids, [t])
                visible_t = pl.min((pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
                if visible_t > 0:
                    score_row = score_wide[t:t + 1, :]
                    idx_init = pl.arange(0, [1, SORT_LEN], dtype=pl.UINT32)
                    sorted_tile = pl.sort32(score_row, idx_init)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=64)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=256)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=MRG_TOPK_RUN)
                    topk_pairs = sorted_tile[:, 0:2 * PREFILL_TOPK_CAP]
                    topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    valid_topk = pl.min(PREFILL_TOPK_CAP, visible_t)
                    cmp_topk_indices[t:t + 1, 0:PREFILL_TOPK_CAP] = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
    return (idx_kv_cache, idx_kv_scale, inner_compress_state, score, cmp_topk_indices)
