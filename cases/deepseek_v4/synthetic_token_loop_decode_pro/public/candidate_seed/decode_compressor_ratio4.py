import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, DECODE_CMP_BLOCK_NUM, FP32_NEG_INF
from config import B
from config import S
EPS = M.rms_norm_eps
from config import D
from config import HEAD_DIM
HEAD_DIM_INV = 1.0 / HEAD_DIM
from config import ROPE_HEAD_DIM
NOPE_HEAD_DIM = M.nope_head_dim
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
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
K_TILE = 512
OUT_TILE = 64
MM_B_TILE = 16
BS_PAD = (B * S + MM_B_TILE - 1) // MM_B_TILE * MM_B_TILE
HEAD_TILE = 64
RMS_PAD_TILE = 16
RMS_PAD_ROWS = RMS_PAD_TILE

@pl.jit.inline
def compressor_ratio4(x: pl.Tensor[[B, S, D], pl.BF16], kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32], compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32], compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32], wkv: pl.Tensor[[OUT_DIM, D], pl.BF16], wgate: pl.Tensor[[OUT_DIM, D], pl.BF16], ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32], norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32], sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32], cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], position_ids: pl.Tensor[[B, S], pl.INT32], cmp_slot_mapping: pl.Tensor[[B, S], pl.INT64], state_slot_mapping: pl.Tensor[[B, S], pl.INT64], late_dep: pl.Scalar[pl.TASK_ID]):
    x_flat = pl.reshape(x, [B * S, D])
    cmp4_kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(compress_state, [COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.spmd(BS_PAD * OUT_DIM // (MM_B_TILE * OUT_TILE), name_hint='kv_score_proj', deps=[late_dep]) as _kv_score_tid:
        idx = pl.tile.get_block_idx()
        global_row0 = idx // (OUT_DIM // OUT_TILE) * MM_B_TILE
        o0 = idx % (OUT_DIM // OUT_TILE) * OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            wkv_tile = wkv[o0:o0 + OUT_TILE, k0:k0 + K_TILE]
            wgate_tile = wgate[o0:o0 + OUT_TILE, k0:k0 + K_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
        cmp4_kv_proj_pad[global_row0:global_row0 + MM_B_TILE, o0:o0 + OUT_TILE] = kv_acc
        cmp4_score_proj_pad[global_row0:global_row0 + MM_B_TILE, o0:o0 + OUT_TILE] = score_acc
    pooled_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='scatter_softmax_pool'):
        for c_idx in pl.range(B):
            for s_sc in pl.pipeline(S, stage=2):
                token_pos = pl.read(position_ids, [c_idx, s_sc])
                state_row_i64 = pl.read(state_slot_mapping, [c_idx, s_sc])
                proj_row = c_idx * S + s_sc
                token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                if state_row_i64 >= 0:
                    state_row = pl.cast(state_row_i64, pl.INDEX)
                    kv_tile = cmp4_kv_proj_pad[proj_row:proj_row + 1, 0:OUT_DIM]
                    score_tile = cmp4_score_proj_pad[proj_row:proj_row + 1, 0:OUT_DIM]
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
                last_abs = cur_window_start_b + COMPRESS_RATIO - 1
                last_blk_off = last_abs // COMPRESS_STATE_BLOCK_SIZE
                last_intra = last_abs % COMPRESS_STATE_BLOCK_SIZE
                last_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, last_blk_off]), pl.INDEX)
                last_row = last_blk_id * COMPRESS_STATE_BLOCK_SIZE + last_intra
                mi = compress_state_flat[last_row:last_row + 1, OUT_DIM + HEAD_DIM:COMPRESS_STATE_DIM]
                li = pl.exp(pl.sub(mi, mi))
                oi = compress_state_flat[last_row:last_row + 1, HEAD_DIM:OUT_DIM]
                for s in pl.range(0, COMPRESS_RATIO):
                    prev_abs = prev_window_start_b + s
                    front_score = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=FP32_NEG_INF)
                    front_kv = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                    if first_pos_b >= COMPRESS_RATIO:
                        prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                        prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                        prev_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, prev_blk_off]), pl.INDEX)
                        prev_row = prev_blk_id * COMPRESS_STATE_BLOCK_SIZE + prev_intra
                        front_score = compress_state_flat[prev_row:prev_row + 1, OUT_DIM:OUT_DIM + HEAD_DIM]
                        front_kv = compress_state_flat[prev_row:prev_row + 1, 0:HEAD_DIM]
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
                    back_score = compress_state_flat[cur_row:cur_row + 1, OUT_DIM + HEAD_DIM:COMPRESS_STATE_DIM]
                    back_kv = compress_state_flat[cur_row:cur_row + 1, HEAD_DIM:OUT_DIM]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back
                pooled_chunk = pl.div(oi, li)
                pooled_kv[pad_idx:pad_idx + 1, 0:HEAD_DIM] = pooled_chunk
    normed_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='rmsnorm_rope_cache_write'):
        cos_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        cos_b[0:B, 0:ROPE_HEAD_DIM // 2] = cos[0:B, 0:ROPE_HEAD_DIM // 2]
        sin_b[0:B, 0:ROPE_HEAD_DIM // 2] = sin[0:B, 0:ROPE_HEAD_DIM // 2]
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
            normed_kv[0:RMS_PAD_TILE, k0:k0 + HEAD_TILE] = normed_chunk
        kv_rope_norm = pooled_kv[0:RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_kv[0:RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_rot
        for inner in pl.range(B):
            c_idx = inner
            first_pos_b = pl.read(position_ids, [c_idx, 0])
            pos_b = first_pos_b % COMPRESS_RATIO
            if pos_b + S >= COMPRESS_RATIO:
                boundary_s = COMPRESS_RATIO - 1 - pos_b
                kv_row_fp32 = normed_kv[inner:inner + 1, 0:HEAD_DIM]
                cache_row_i64 = pl.read(cmp_slot_mapping, [c_idx, boundary_s])
                if cache_row_i64 >= 0:
                    cache_row = pl.cast(cache_row_i64, pl.INDEX)
                    kv_flat[c_idx * S:c_idx * S + 1, :] = kv_row_fp32
                    cmp_kv_cache_flat[cache_row:cache_row + 1, :] = pl.cast(kv_row_fp32, target_type=pl.BF16, mode='rint')
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv
