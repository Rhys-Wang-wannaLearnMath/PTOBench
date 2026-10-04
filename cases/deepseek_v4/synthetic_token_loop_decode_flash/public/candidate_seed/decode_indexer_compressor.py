import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, DECODE_IDX_BLOCK_NUM, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS
from config import B
from config import S
EPS = M.rms_norm_eps
from config import D
HEAD_DIM = M.index_head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
from config import ROPE_HEAD_DIM
NOPE_HEAD_DIM = M.index_nope_head_dim
from config import MAX_SEQ_LEN
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = 65
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
K_TILE = 512
OUT_TILE = 64
PROJ_OUT_TILE = 32
MM_B_TILE = 16
BS_PAD = (B * S + MM_B_TILE - 1) // MM_B_TILE * MM_B_TILE
HEAD_TILE = 64
RMS_PAD_TILE = 16
CMP_SWAP_FLAT_LEN = RMS_PAD_TILE * ROPE_HEAD_DIM
ROPE_HEAD_DIM_F = float(ROPE_HEAD_DIM)
ROPE_HEAD_DIM_INV = 1.0 / ROPE_HEAD_DIM
RMS_PAD_ROWS = RMS_PAD_TILE

@pl.jit.inline
def indexer_compressor(x: pl.Tensor[[B, S, D], pl.BF16], kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32], compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32], wkv: pl.Tensor[[OUT_DIM, D], pl.BF16], wgate: pl.Tensor[[OUT_DIM, D], pl.BF16], ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32], norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32], sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32], hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16], idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]], position_ids: pl.Tensor[[B, S], pl.INT32], idx_slot_mapping: pl.Tensor[[B, S], pl.INT64], inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64], late_dep: pl.Scalar[pl.TASK_ID]):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(compress_state, [COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, 1])
    with pl.spmd(BS_PAD * OUT_DIM // (MM_B_TILE * PROJ_OUT_TILE), name_hint='kv_score_proj', deps=[late_dep]) as _kv_score_tid:
        idx = pl.tile.get_block_idx()
        global_row0 = idx // (OUT_DIM // PROJ_OUT_TILE) * MM_B_TILE
        o0 = idx % (OUT_DIM // PROJ_OUT_TILE) * PROJ_OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            wkv_tile = wkv[o0:o0 + PROJ_OUT_TILE, k0:k0 + K_TILE]
            wgate_tile = wgate[o0:o0 + PROJ_OUT_TILE, k0:k0 + K_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
        kv_proj_pad[global_row0:global_row0 + MM_B_TILE, o0:o0 + PROJ_OUT_TILE] = kv_acc
        score_proj_pad[global_row0:global_row0 + MM_B_TILE, o0:o0 + PROJ_OUT_TILE] = score_acc
    pooled_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='scatter_softmax_pool'):
        for c_idx in pl.range(B):
            for s_sc in pl.pipeline(S, stage=2):
                token_pos = pl.read(position_ids, [c_idx, s_sc])
                state_row_i64 = pl.read(inner_state_slot_mapping, [c_idx, s_sc])
                proj_row = c_idx * S + s_sc
                token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                if state_row_i64 >= 0:
                    state_row = pl.cast(state_row_i64, pl.INDEX)
                    kv_tile = kv_proj_pad[proj_row:proj_row + 1, 0:OUT_DIM]
                    score_tile = score_proj_pad[proj_row:proj_row + 1, 0:OUT_DIM]
                    ape_tile = ape[token_ape_row:token_ape_row + 1, 0:OUT_DIM]
                    score_tile = pl.add(score_tile, ape_tile)
                    compress_state_flat[state_row:state_row + 1, 0:OUT_DIM] = kv_tile
                    compress_state_flat[state_row:state_row + 1, OUT_DIM:2 * OUT_DIM] = score_tile
            pad_idx = c_idx
            first_pos_b = pl.read(position_ids, [c_idx, 0])
            pos_b = first_pos_b % COMPRESS_RATIO
            pre_tokens_b = COMPRESS_RATIO - pos_b
            boundary_end_b = first_pos_b + pre_tokens_b - 1
            cur_window_start_b = boundary_end_b - COMPRESS_RATIO + 1
            prev_window_start_b = cur_window_start_b - COMPRESS_RATIO
            if pos_b + S >= COMPRESS_RATIO:
                for hb in pl.range(HEAD_DIM // HEAD_TILE):
                    h0 = hb * HEAD_TILE
                    last_abs = cur_window_start_b + COMPRESS_RATIO - 1
                    last_blk_off = last_abs // COMPRESS_STATE_BLOCK_SIZE
                    last_intra = last_abs % COMPRESS_STATE_BLOCK_SIZE
                    last_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, last_blk_off]), pl.INDEX)
                    last_row = last_blk_id * COMPRESS_STATE_BLOCK_SIZE + last_intra
                    last_col0 = OUT_DIM + HEAD_DIM + h0
                    mi = compress_state_flat[last_row:last_row + 1, last_col0:last_col0 + HEAD_TILE]
                    li = pl.exp(pl.sub(mi, mi))
                    oi = compress_state_flat[last_row:last_row + 1, HEAD_DIM + h0:HEAD_DIM + h0 + HEAD_TILE]
                    for s in pl.range(0, COMPRESS_RATIO):
                        prev_abs = prev_window_start_b + s
                        front_score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                        front_kv = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                        if first_pos_b >= COMPRESS_RATIO:
                            prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                            prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                            prev_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, prev_blk_off]), pl.INDEX)
                            prev_row = prev_blk_id * COMPRESS_STATE_BLOCK_SIZE + prev_intra
                            front_score = compress_state_flat[prev_row:prev_row + 1, OUT_DIM + h0:OUT_DIM + h0 + HEAD_TILE]
                            front_kv = compress_state_flat[prev_row:prev_row + 1, h0:h0 + HEAD_TILE]
                        mi_next_front = pl.maximum(mi, front_score)
                        alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                        beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                        li = pl.add(pl.mul(alpha_front, li), beta_front)
                        oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                        mi = mi_next_front
                    for s in pl.range(0, COMPRESS_RATIO - 1):
                        cur_abs = cur_window_start_b + s
                        cur_blk_off = cur_abs // COMPRESS_STATE_BLOCK_SIZE
                        cur_intra = cur_abs % COMPRESS_STATE_BLOCK_SIZE
                        cur_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, cur_blk_off]), pl.INDEX)
                        cur_row = cur_blk_id * COMPRESS_STATE_BLOCK_SIZE + cur_intra
                        back_col0 = OUT_DIM + HEAD_DIM + h0
                        back_score = compress_state_flat[cur_row:cur_row + 1, back_col0:back_col0 + HEAD_TILE]
                        back_kv = compress_state_flat[cur_row:cur_row + 1, HEAD_DIM + h0:HEAD_DIM + h0 + HEAD_TILE]
                        mi_next_back = pl.maximum(mi, back_score)
                        alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                        beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                        li = pl.add(pl.mul(alpha_back, li), beta_back)
                        oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                        mi = mi_next_back
                    pooled_chunk = pl.div(oi, li)
                    pooled_kv[pad_idx:pad_idx + 1, h0:h0 + HEAD_TILE] = pooled_chunk
    normed_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    cmp_cos_il = pl.create_tensor([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_swap_flat = pl.create_tensor([1, CMP_SWAP_FLAT_LEN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cmp_rope_tables', allow_early_resolve=True):
        cos_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        cos_b[0:B, 0:ROPE_HEAD_DIM // 2] = cos[0:B, 0:ROPE_HEAD_DIM // 2]
        sin_b[0:B, 0:ROPE_HEAD_DIM // 2] = sin[0:B, 0:ROPE_HEAD_DIM // 2]
        t_col = pl.col_expand_mul(pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0), pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        t_dup_f = pl.cast(pl.cast(pl.mul(t_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        t_dup_idx = pl.cast(t_dup_f, target_type=pl.INT32)
        t_lane = pl.sub(t_col, pl.mul(t_dup_f, 2.0))
        t_sign = pl.sub(pl.mul(t_lane, 2.0), 1.0)
        cmp_cos_il[0:RMS_PAD_TILE, 0:ROPE_HEAD_DIM] = pl.gather(cos_b, dim=-1, index=t_dup_idx)
        cmp_sin_signed[0:RMS_PAD_TILE, 0:ROPE_HEAD_DIM] = pl.mul(pl.gather(sin_b, dim=-1, index=t_dup_idx), t_sign)
        f_col = pl.cast(pl.arange(0, [1, CMP_SWAP_FLAT_LEN], dtype=pl.INT32), target_type=pl.FP32)
        f_rowbase = pl.mul(pl.cast(pl.cast(pl.mul(f_col, ROPE_HEAD_DIM_INV), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32), ROPE_HEAD_DIM_F)
        f_j = pl.sub(f_col, f_rowbase)
        f_dup = pl.cast(pl.cast(pl.mul(f_j, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        f_lane = pl.sub(f_j, pl.mul(f_dup, 2.0))
        f_jx = pl.sub(pl.add(f_j, 1.0), pl.mul(f_lane, 2.0))
        cmp_swap_flat[0:1, 0:CMP_SWAP_FLAT_LEN] = pl.cast(pl.add(f_rowbase, f_jx), target_type=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='rmsnorm_rope'):
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[0:RMS_PAD_TILE, k0:k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[0:RMS_PAD_TILE, k0:k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0:k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[0:RMS_PAD_TILE, k0:k0 + HEAD_TILE] = pl.cast(normed_chunk, target_type=pl.BF16, mode='rint')
        kv_rope_norm = pooled_kv[0:RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_flat = pl.reshape(rope_normed, [1, CMP_SWAP_FLAT_LEN])
        swapped = pl.reshape(pl.gather(rope_flat, dim=-1, index=cmp_swap_flat[0:1, 0:CMP_SWAP_FLAT_LEN]), [RMS_PAD_TILE, ROPE_HEAD_DIM])
        rope_rot = pl.add(pl.mul(rope_normed, cmp_cos_il[0:RMS_PAD_TILE, 0:ROPE_HEAD_DIM]), pl.mul(swapped, cmp_sin_signed[0:RMS_PAD_TILE, 0:ROPE_HEAD_DIM]))
        normed_kv[0:RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM] = pl.cast(rope_rot, target_type=pl.BF16, mode='rint')
    kv_final = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='kv_hadamard'):
        kv_proj_tile = normed_kv[0:RMS_PAD_TILE, 0:HEAD_DIM]
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0:HEAD_DIM, o0:o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[0:RMS_PAD_TILE, o0:o0 + OUT_TILE] = kv_hadamard_acc
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='kv_and_cache_write'):
        kv_blk_f32 = pl.cast(pl.cast(kv_final[0:RMS_PAD_TILE, 0:HEAD_DIM], target_type=pl.BF16, mode='rint'), target_type=pl.FP32)
        kv_amax = pl.reshape(pl.row_max(pl.abs(kv_blk_f32)), [1, RMS_PAD_TILE])
        kv_amax = pl.maximum(kv_amax, pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS))
        kv_scale_q_row = pl.div(pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax)
        kv_scale_dq_col = pl.reshape(pl.recip(kv_scale_q_row), [RMS_PAD_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [RMS_PAD_TILE, 1])
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode='rint')
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode='round')
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode='trunc')
        for inner in pl.range(B):
            c_idx = inner
            first_pos_b = pl.read(position_ids, [c_idx, 0])
            pos_b = first_pos_b % COMPRESS_RATIO
            if pos_b + S >= COMPRESS_RATIO:
                boundary_s = COMPRESS_RATIO - 1 - pos_b
                kv_row_fp32 = kv_final[inner:inner + 1, 0:HEAD_DIM]
                cache_row_i64 = pl.read(idx_slot_mapping, [c_idx, boundary_s])
                if cache_row_i64 >= 0:
                    cache_row = pl.cast(cache_row_i64, pl.INDEX)
                    kv_flat[c_idx * S:c_idx * S + 1, :] = kv_row_fp32
                    idx_kv_cache_flat[cache_row:cache_row + 1, :] = kv_i8_blk[inner:inner + 1, :]
                    pl.write(idx_kv_scale_flat, [cache_row, 0], pl.read(kv_scale_dq_col, [inner, 0]))
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return (kv, compress_state, idx_kv_cache, idx_kv_scale)
