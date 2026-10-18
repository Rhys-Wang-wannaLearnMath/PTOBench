"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_csa_flash."""

from config import TP
from config import CASE_CONFIG as M
from config import COMPRESS_RATIO
from config import DECODE_BATCH
from config import DECODE_SEQ
from config import FP32_NEG_INF
from config import INT8_SCALE_MAX
from config import C4A_COMPRESSOR_BLOCK_SIZE
from config import INT8_AMAX_EPS
from config import BLOCK_SIZE
import pypto.language as pl

B_DYN = pl.dynamic('B_DYN')
T_DYN = pl.dynamic('T_DYN')
B = DECODE_BATCH // TP
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.index_head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.index_nope_head_dim
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_MAX_BLOCKS = (STATE_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * OUT_DIM
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic('IDX_CACHE_BLOCK_NUM_DYN')
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic('INNER_STATE_BLOCK_NUM_DYN')
K_TILE = 512
OUT_TILE = 64
PROJ_OUT_TILE = 32
MM_B_TILE = 16
BS_PAD = (B * S + MM_B_TILE - 1) // MM_B_TILE * MM_B_TILE
HEAD_TILE = 64
RMS_PAD_TILE = 16

@pl.jit.inline
def indexer_compressor(x: pl.Tensor[[T_DYN, D], pl.BF16], kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32], compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32], compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32], wkv: pl.Tensor[[OUT_DIM, D], pl.BF16], wgate: pl.Tensor[[OUT_DIM, D], pl.BF16], ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32], norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32], sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32], hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16], idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32], position_ids: pl.Tensor[[T_DYN], pl.INT32], idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], late_dep: pl.Scalar[pl.TASK_ID]):
    b_dim = pl.tensor.dim(compress_state_block_table, 0)
    bs = pl.tensor.dim(x, 0)
    s_dim = bs // b_dim
    t_matmul = (bs + MM_B_TILE - 1) // MM_B_TILE * MM_B_TILE
    rms_blocks = (bs + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    x_flat = x
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    compress_state_flat = pl.reshape(compress_state, [compress_state_block_num * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv_flat = kv
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [idx_block_num * BLOCK_SIZE, HEAD_DIM])
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * BLOCK_SIZE, 1])
    with pl.spmd(t_matmul * OUT_DIM // (MM_B_TILE * PROJ_OUT_TILE), name_hint='kv_score_proj', deps=[late_dep]) as _kv_score_tid:
        idx = pl.tile.get_block_idx()
        global_row0 = idx // (OUT_DIM // PROJ_OUT_TILE) * MM_B_TILE
        o0 = idx % (OUT_DIM // PROJ_OUT_TILE) * PROJ_OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, bs - global_row0)
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
    pooled_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='scatter_softmax_pool', deps=[_kv_score_tid]) as pool_tid:
        for c_idx in pl.range(b_dim):
            first_pos_b = pl.read(position_ids, [c_idx * s_dim])
            for s_idx in pl.range(s_dim):
                token = c_idx * s_dim + s_idx
                token_pos = pl.read(position_ids, [token])
                pooled_kv[token:token + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                if (token_pos + 1) % COMPRESS_RATIO == 0:
                    window_start = token_pos - STATE_LEN + 1
                    for h0 in pl.range(0, HEAD_DIM, HEAD_TILE):
                        last_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                        mi = pl.add(score_proj_pad[token:token + 1, HEAD_DIM + h0:HEAD_DIM + h0 + HEAD_TILE], ape[last_ape_row:last_ape_row + 1, HEAD_DIM + h0:HEAD_DIM + h0 + HEAD_TILE])
                        li = pl.exp(pl.sub(mi, mi))
                        oi = kv_proj_pad[token:token + 1, HEAD_DIM + h0:HEAD_DIM + h0 + HEAD_TILE]
                        for state_idx in pl.range(STATE_LEN - 1):
                            logical_pos = window_start + state_idx
                            value = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                            score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            state_half = 0
                            if state_idx >= COMPRESS_RATIO:
                                state_half = HEAD_DIM
                            if logical_pos >= 0 and logical_pos < first_pos_b:
                                ring_row = logical_pos % STATE_LEN
                                state_page_off = ring_row // COMPRESS_STATE_BLOCK_SIZE
                                state_blk_id_i32 = pl.read(compress_state_block_table, [c_idx, state_page_off])
                                if state_blk_id_i32 >= 0:
                                    state_blk_id = pl.cast(state_blk_id_i32, pl.INDEX)
                                    state_row = state_blk_id * COMPRESS_STATE_BLOCK_SIZE + ring_row % COMPRESS_STATE_BLOCK_SIZE
                                    value = compress_state_flat[state_row:state_row + 1, state_half + h0:state_half + h0 + HEAD_TILE]
                                    score = compress_state_flat[state_row:state_row + 1, OUT_DIM + state_half + h0:OUT_DIM + state_half + h0 + HEAD_TILE]
                            if logical_pos >= first_pos_b:
                                if logical_pos <= token_pos:
                                    overlay_token = c_idx * s_dim + logical_pos - first_pos_b
                                    ape_row = pl.cast(logical_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                                    value = kv_proj_pad[overlay_token:overlay_token + 1, state_half + h0:state_half + h0 + HEAD_TILE]
                                    score = pl.add(score_proj_pad[overlay_token:overlay_token + 1, state_half + h0:state_half + h0 + HEAD_TILE], ape[ape_row:ape_row + 1, state_half + h0:state_half + h0 + HEAD_TILE])
                            mi_next = pl.maximum(mi, score)
                            alpha = pl.exp(pl.sub(mi, mi_next))
                            beta = pl.exp(pl.sub(score, mi_next))
                            li = pl.add(pl.mul(alpha, li), beta)
                            oi = pl.add(pl.mul(oi, alpha), pl.mul(value, beta))
                            mi = mi_next
                        pooled_kv[token:token + 1, h0:h0 + HEAD_TILE] = pl.div(oi, li)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='compress_state_commit', deps=[pool_tid]):
        for token in pl.range(bs):
            state_row_i64 = pl.read(inner_state_slot_mapping, [token])
            if state_row_i64 >= 0:
                state_row = pl.cast(state_row_i64, pl.INDEX)
                token_pos = pl.read(position_ids, [token])
                ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                compress_state_flat[state_row:state_row + 1, 0:OUT_DIM] = kv_proj_pad[token:token + 1, 0:OUT_DIM]
                compress_state_flat[state_row:state_row + 1, OUT_DIM:COMPRESS_STATE_DIM] = pl.add(score_proj_pad[token:token + 1, 0:OUT_DIM], ape[ape_row:ape_row + 1, 0:OUT_DIM])
    normed_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.spmd(rms_blocks, name_hint='rmsnorm_rope', deps=[pool_tid]) as rms_tid:
        rms_blk = pl.tile.get_block_idx()
        b0 = rms_blk * RMS_PAD_TILE
        rms_blk_rows = pl.min(RMS_PAD_TILE, bs - b0)
        cos_b = pl.slice(cos, [RMS_PAD_TILE, ROPE_HEAD_DIM], [b0, 0], valid_shape=[rms_blk_rows, ROPE_HEAD_DIM])
        sin_b = pl.slice(sin, [RMS_PAD_TILE, ROPE_HEAD_DIM], [b0, 0], valid_shape=[rms_blk_rows, ROPE_HEAD_DIM])
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(0, HEAD_DIM, HEAD_TILE, stage=2):
            kv_rms_chunk = pooled_kv[b0:b0 + RMS_PAD_TILE, k0:k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.pipeline(0, NOPE_HEAD_DIM, HEAD_TILE, stage=2):
            kv_norm_chunk = pooled_kv[b0:b0 + RMS_PAD_TILE, k0:k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0:k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[b0:b0 + RMS_PAD_TILE, k0:k0 + HEAD_TILE] = pl.cast(normed_chunk, target_type=pl.BF16, mode='rint')
        kv_rope_norm = pooled_kv[b0:b0 + RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_b))
        normed_kv[b0:b0 + RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM] = pl.cast(rope_rot, target_type=pl.BF16, mode='rint')
    kv_final = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(rms_blocks, name_hint='kv_hadamard', deps=[rms_tid]) as hadamard_tid:
        had_blk = pl.tile.get_block_idx()
        had_b0 = had_blk * RMS_PAD_TILE
        kv_proj_tile = normed_kv[had_b0:had_b0 + RMS_PAD_TILE, 0:HEAD_DIM]
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0:HEAD_DIM, o0:o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[had_b0:had_b0 + RMS_PAD_TILE, o0:o0 + OUT_TILE] = kv_hadamard_acc
    with pl.spmd(rms_blocks, name_hint='kv_and_cache_write', deps=[hadamard_tid]) as _write_tid:
        wr_blk = pl.tile.get_block_idx()
        wr_b0 = wr_blk * RMS_PAD_TILE
        wr_blk_rows = pl.min(RMS_PAD_TILE, bs - wr_b0)
        kv_blk_f32 = pl.cast(pl.cast(kv_final[wr_b0:wr_b0 + RMS_PAD_TILE, 0:HEAD_DIM], target_type=pl.BF16, mode='rint'), target_type=pl.FP32)
        kv_amax = pl.reshape(pl.row_max(pl.abs(kv_blk_f32)), [1, RMS_PAD_TILE])
        kv_amax = pl.maximum(kv_amax, pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS))
        kv_scale_q_row = pl.div(pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax)
        kv_scale_dq_col = pl.reshape(pl.recip(kv_scale_q_row), [RMS_PAD_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [RMS_PAD_TILE, 1])
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode='rint')
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode='round')
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode='trunc')
        for inner in pl.range(wr_blk_rows):
            token = wr_b0 + inner
            cache_row_i64 = pl.read(idx_slot_mapping, [token])
            if cache_row_i64 >= 0:
                cache_row = pl.cast(cache_row_i64, pl.INDEX)
                kv_flat[token:token + 1, :] = kv_final[token:token + 1, 0:HEAD_DIM]
                idx_kv_cache_flat[cache_row:cache_row + 1, :] = kv_i8_blk[inner:inner + 1, :]
                pl.write(idx_kv_scale_flat, [cache_row, 0], pl.read(kv_scale_dq_col, [inner, 0]))
    return _write_tid
