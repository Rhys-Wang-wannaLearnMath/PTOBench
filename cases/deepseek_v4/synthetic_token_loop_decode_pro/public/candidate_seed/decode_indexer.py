import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, DECODE_IDX_BLOCK_NUM, IDX_CACHE_MAX_BLOCKS, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS
from decode_indexer_compressor import indexer_compressor
from config import B
from config import S
from config import T
from config import D
from config import Q_LORA
from config import ROPE_HEAD_DIM
from config import IDX_N_HEADS
from config import IDX_HEAD_DIM
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
from config import MAX_SEQ_LEN
COMPRESS_RATIO = 4
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
from config import INNER_OUT_DIM
from config import INNER_STATE_BLOCK_SIZE
INNER_STATE_PHYSICAL_BLOCKS = 65
from config import INNER_STATE_MAX_BLOCKS
from config import INNER_STATE_BLOCK_NUM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
SCORE_LEN = IDX_KV_LEN
REDUCE_TILE = 128
REDUCE_NSPLIT = 4
Q_TILE = 256
Q_OUT_TILE = 1024
MM_N_TILE = 512
MM_ROW_TILE = 16
T_PAD = (T + MM_ROW_TILE - 1) // MM_ROW_TILE * MM_ROW_TILE
D_TILE = 512
WEIGHTS_OK = 4 if M.name == 'flash' else 7
WEIGHTS_K_SLICE = D // WEIGHTS_OK
QH_QUANT_TILE = 64
QH_MM_TILE = 64
QH_HEAD_DIM_TILE = 64
ROPE_ROW_BLOCK = S * IDX_N_HEADS
ROPE_ROW_TILE = 32
ROPE_SWAP_FLAT_LEN = ROPE_ROW_TILE * ROPE_HEAD_DIM
ROPE_HEAD_DIM_F = float(ROPE_HEAD_DIM)
ROPE_HEAD_DIM_INV = 1.0 / ROPE_HEAD_DIM
TOPK_HALF_LEN = SCORE_LEN // 2
TOPK_HALF_PAIR_OFFSET = 2 * TOPK_HALF_LEN
TOPK_PAIR_WIDTH = 2 * IDX_TOPK

@pl.jit.inline
def indexer(x: pl.Tensor[[B, S, D], pl.BF16], qr: pl.Tensor[[T, Q_LORA], pl.INT8], qr_scale: pl.Tensor[[T, 1], pl.FP32], wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32], sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32], hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32], inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16], idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32], score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32], topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32], position_ids: pl.Tensor[[B, S], pl.INT32], idx_slot_mapping: pl.Tensor[[B, S], pl.INT64], inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64], kv_seq_lens: pl.Tensor[[B], pl.INT32], offset: pl.Scalar[pl.INT32], late_dep: pl.Scalar[pl.TASK_ID]):
    qr_acc_pad = pl.create_tensor([T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
    for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint='idx_qr_proj_matmul', allow_early_resolve=True):
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(qr, [T_PAD, Q_TILE], [0, q0], valid_shape=[T, Q_TILE])
                wq_tile = wq_b[q0:q0 + Q_TILE, o_base + ns:o_base + ns + MM_N_TILE]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            qr_acc_pad[0:T_PAD, o_base + ns:o_base + ns + MM_N_TILE] = qr_acc
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint='idx_qr_proj_dequant', allow_early_resolve=True):
        o_base = ot * Q_OUT_TILE
        wq_scale = pl.reshape(wq_b_scale[o_base:o_base + Q_OUT_TILE], [1, Q_OUT_TILE])
        acc_fp32 = pl.cast(qr_acc_pad[0:T, o_base:o_base + Q_OUT_TILE], target_type=pl.FP32, mode='none')
        qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale[0:T, :]), wq_scale)
        qr_proj[0:T, o_base:o_base + Q_OUT_TILE] = qr_dequant
    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    cos_il_t = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed_t = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_swap_idx_t = pl.create_tensor([1, ROPE_SWAP_FLAT_LEN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='qr_rope_tables', allow_early_resolve=True):
        il_col = pl.col_expand_mul(pl.full([B, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0), pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        cos_il_t[0:B, 0:ROPE_HEAD_DIM] = pl.gather(cos[0:B, 0:ROPE_HEAD_DIM // 2], dim=-1, index=il_dup_idx)
        sin_signed_t[0:B, 0:ROPE_HEAD_DIM] = pl.mul(pl.gather(sin[0:B, 0:ROPE_HEAD_DIM // 2], dim=-1, index=il_dup_idx), il_sign)
        sw_col = pl.cast(pl.arange(0, [1, ROPE_SWAP_FLAT_LEN], dtype=pl.INT32), target_type=pl.FP32)
        sw_rowbase = pl.mul(pl.cast(pl.cast(pl.mul(sw_col, ROPE_HEAD_DIM_INV), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32), ROPE_HEAD_DIM_F)
        sw_j = pl.sub(sw_col, sw_rowbase)
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_j, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        sw_lane = pl.sub(sw_j, pl.mul(sw_dup_f, 2.0))
        sw_jx = pl.sub(pl.add(sw_j, 1.0), pl.mul(sw_lane, 2.0))
        rope_swap_idx_t[0:1, 0:ROPE_SWAP_FLAT_LEN] = pl.cast(pl.add(sw_rowbase, sw_jx), target_type=pl.INT32)
    for idx in pl.spmd(T * IDX_N_HEADS // ROPE_ROW_TILE, name_hint='qr_rope', allow_early_resolve=True):
        o0 = idx * ROPE_ROW_TILE
        batch_idx = o0 // ROPE_ROW_BLOCK
        rope_swap_flat = rope_swap_idx_t[0:1, 0:ROPE_SWAP_FLAT_LEN]
        cos_row = cos_il_t[batch_idx:batch_idx + 1, 0:ROPE_HEAD_DIM]
        sin_row = sin_signed_t[batch_idx:batch_idx + 1, 0:ROPE_HEAD_DIM]
        qr_nope_slice = qr_proj_flat[o0:o0 + ROPE_ROW_TILE, 0:IDX_NOPE_HEAD_DIM]
        qr_rope_slice = qr_proj_flat[o0:o0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
        qr_rope_flat = pl.reshape(qr_rope_slice, [1, ROPE_SWAP_FLAT_LEN])
        qr_swapped = pl.reshape(pl.gather(qr_rope_flat, dim=-1, index=rope_swap_flat), [ROPE_ROW_TILE, ROPE_HEAD_DIM])
        rope_rot = pl.add(pl.col_expand_mul(qr_rope_slice, cos_row), pl.col_expand_mul(qr_swapped, sin_row))
        qr_vec = pl.concat(pl.cast(qr_nope_slice, target_type=pl.BF16, mode='rint'), pl.cast(rope_rot, target_type=pl.BF16, mode='rint'))
        qr_bf16[o0:o0 + ROPE_ROW_TILE, :] = qr_vec
    qh_acc_gm = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_MM_TILE, name_hint='qr_hadamard_matmul', allow_early_resolve=True):
        o0 = idx * QH_MM_TILE
        qh_acc = pl.matmul(qr_bf16[o0:o0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32)
        qh_acc_gm[o0:o0 + QH_MM_TILE, :] = qh_acc
    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_TILE, name_hint='qr_hadamard_quant', allow_early_resolve=True):
        o0 = idx * QH_QUANT_TILE
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[o0:o0 + QH_QUANT_TILE, h0:h0 + QH_HEAD_DIM_TILE]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0:o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[o0:o0 + QH_QUANT_TILE, h1:h1 + QH_HEAD_DIM_TILE]
            qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode='rint')
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode='round')
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode='trunc')
            qr_hadamard_i8[o0:o0 + QH_QUANT_TILE, h1:h1 + QH_HEAD_DIM_TILE] = qh_i8
    x_flat = pl.reshape(x, [T, D])
    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    weights_partial = pl.create_tensor([WEIGHTS_OK * MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
    with pl.spmd(WEIGHTS_OK, name_hint='weights_proj', deps=[late_dep]) as _weights_tid:
        kb = pl.tile.get_block_idx()
        k_base = kb * WEIGHTS_K_SLICE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [0, d0], valid_shape=[pl.min(MM_ROW_TILE, T), D_TILE])
            weights_proj_tile = weights_proj[d0:d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
        weights_partial[kb * MM_ROW_TILE:kb * MM_ROW_TILE + MM_ROW_TILE, :] = weights_acc
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='weights_proj_reduce'):
        w_sum = weights_partial[0:MM_ROW_TILE, :]
        for kb in pl.unroll(1, WEIGHTS_OK):
            w_sum = pl.add(w_sum, weights_partial[kb * MM_ROW_TILE:kb * MM_ROW_TILE + MM_ROW_TILE, :])
        weights[0:MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)
    indexer_compressor(x, inner_kv, inner_compress_state, inner_compress_state_block_table, inner_wkv, inner_wgate, inner_ape, inner_norm_w, cos, sin, hadamard, idx_kv_cache, idx_kv_scale, position_ids, idx_slot_mapping, inner_state_slot_mapping, late_dep)
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, 1])
    idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])
    score_flat = pl.reshape(score, [T, SCORE_LEN])
    score_acc_gm = pl.create_tensor([T * IDX_KV_LEN, IDX_N_HEADS], dtype=pl.INT32)
    for tg in pl.spmd(T, name_hint='score_mat', allow_early_resolve=True):
        b = tg // S
        s = tg - b * S
        clen_b = pl.read(kv_seq_lens, [b]) // COMPRESS_RATIO
        cblk_b = (clen_b + BLOCK_SIZE - 1) // BLOCK_SIZE
        qb = b * S * IDX_N_HEADS
        qr_full = qr_hadamard_i8[qb + s * IDX_N_HEADS:qb + (s + 1) * IDX_N_HEADS, 0:IDX_HEAD_DIM]
        for cb in pl.pipeline(0, cblk_b, stage=2):
            cache0 = cb * BLOCK_SIZE
            idx_blk_id = pl.cast(pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + cb]), pl.INDEX)
            kv0 = idx_blk_id * BLOCK_SIZE
            base = tg * IDX_KV_LEN + cache0
            kv_i8_mat = kv_cache_i8_flat[kv0:kv0 + BLOCK_SIZE, :]
            score_acc_mat = pl.matmul(kv_i8_mat, qr_full, out_dtype=pl.INT32, b_trans=True)
            score_acc_gm[base:base + BLOCK_SIZE, :] = score_acc_mat
    for unit in pl.spmd(T * REDUCE_NSPLIT, name_hint='score_reduce', allow_early_resolve=True):
        tg = unit // REDUCE_NSPLIT
        split = unit - tg * REDUCE_NSPLIT
        b = tg // S
        s = tg - b * S
        clen_b = pl.read(kv_seq_lens, [b]) // COMPRESS_RATIO
        pos_t = pl.read(position_ids, [b, s])
        visible_len_t = pl.min(pl.min(clen_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN)
        cblk_t = (visible_len_t + REDUCE_TILE - 1) // REDUCE_TILE
        tb = b * S
        qb = b * S * IDX_N_HEADS
        qh_scale_s = pl.reshape(qr_hadamard_scale_dq[qb + s * IDX_N_HEADS:qb + (s + 1) * IDX_N_HEADS, :], [1, IDX_N_HEADS])
        weights_row_s = pl.reshape(weights[tb + s:tb + s + 1, :], [1, IDX_N_HEADS])
        lane_iters = (cblk_t - split + REDUCE_NSPLIT - 1) // REDUCE_NSPLIT
        for cb_local in pl.pipeline(0, lane_iters, stage=2):
            cb = split + cb_local * REDUCE_NSPLIT
            cache0 = cb * REDUCE_TILE
            valid_len = pl.min(REDUCE_TILE, visible_len_t - cache0)
            base = tg * IDX_KV_LEN + cache0
            idx_blk_id = pl.cast(pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + cb]), pl.INDEX)
            kv0 = idx_blk_id * BLOCK_SIZE
            score_acc_red = score_acc_gm[base:base + REDUCE_TILE, :]
            kv_dq_red = kv_scale_flat[kv0:kv0 + REDUCE_TILE, :]
            score_tile_red = pl.cast(score_acc_red, target_type=pl.FP32, mode='none')
            score_tile_red = pl.col_expand_mul(score_tile_red, qh_scale_s)
            relu_score_red = pl.maximum(score_tile_red, pl.full([REDUCE_TILE, IDX_N_HEADS], dtype=pl.FP32, value=0.0))
            weighted_score_red = pl.col_expand_mul(relu_score_red, weights_row_s)
            weighted_score_row = pl.mul(pl.row_sum(weighted_score_red), kv_dq_red)
            weighted_score_s = pl.reshape(weighted_score_row, [1, REDUCE_TILE])
            weighted_score_valid_s = pl.fillpad(pl.set_validshape(weighted_score_s, 1, valid_len), pad_value=pl.PadValue.min)
            weighted_score_valid_s = pl.maximum(weighted_score_valid_s, pl.full([1, REDUCE_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
            score_flat[tb + s:tb + s + 1, cache0:cache0 + REDUCE_TILE] = weighted_score_valid_s
    topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])
    for t in pl.spmd(T, name_hint='topk', allow_early_resolve=True):
        invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
        topk_idxs_flat[t:t + 1, :] = invalid_idxs
        batch_idx = t // S
        token_s = t - batch_idx * S
        cache_len_b = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
        pos_t = pl.read(position_ids, [batch_idx, token_s])
        visible_len_t = pl.min(pl.min(cache_len_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN)
        if visible_len_t > 0:
            offset_i32 = pl.cast(offset, target_type=pl.INT32)
            score_full_raw = score_flat[t:t + 1, 0:SCORE_LEN]
            score_full = pl.fillpad(pl.set_validshape(score_full_raw, 1, visible_len_t), pad_value=pl.PadValue.min)
            score_full = pl.maximum(score_full, pl.full([1, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF))
            idx_init = pl.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
            sorted_full = pl.sort32(score_full, idx_init)
            sorted_full = pl.mrgsort(sorted_full, block_len=64)
            sorted_full = pl.mrgsort(sorted_full, block_len=256)
            sorted_full = pl.mrgsort(sorted_full, block_len=1024)
            half0_candidates = sorted_full[:, 0:TOPK_PAIR_WIDTH]
            half1_candidates = sorted_full[:, TOPK_HALF_PAIR_OFFSET:TOPK_HALF_PAIR_OFFSET + TOPK_PAIR_WIDTH]
            merged_candidates = pl.mrgsort(half0_candidates, half1_candidates)
            topk_pairs = merged_candidates[:, 0:TOPK_PAIR_WIDTH]
            topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid_topk = pl.min(IDX_TOPK, visible_len_t)
            topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
            topk_idxs_flat[t:t + 1, 0:IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)
    return (idx_kv_cache, idx_kv_scale, inner_compress_state, score, topk_idxs)
