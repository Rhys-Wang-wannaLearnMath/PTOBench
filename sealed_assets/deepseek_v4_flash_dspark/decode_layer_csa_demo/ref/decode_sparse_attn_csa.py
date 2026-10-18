"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_csa_demo."""

from config import TP
from config import CASE_CONFIG as M
from config import COMPRESS_RATIO
from config import MAX_SEQ_LEN
from config import DECODE_BATCH
from config import DECODE_SEQ
from config import BLOCK_SIZE
import pypto.language as pl

B_DYN = pl.dynamic('B_DYN')
T_DYN = pl.dynamic('T_DYN')
ORI_BLOCK_NUM_DYN = pl.dynamic('ORI_BLOCK_NUM_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('CMP_BLOCK_NUM_DYN')
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
IDX_TOPK = M.index_topk
CMP_TOPK = IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
COMPRESS_RATIO_INV = 1.0 / COMPRESS_RATIO
CSA_CMP_GE_BIAS = 1.0
NEG_INF = -1e+20
CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
H_TILE = 16
QK_M_TILE = 32
ATTN_K_TILE = 128
NUM_QK_CORES = 24
T_PAD = (T + 16 - 1) // 16 * 16
ROPE_CS_T_TILE = 8
TOPK = WIN + CMP_TOPK
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
QK_ITEMS = T * SPARSE_BLOCKS
SWA_TILE_WIN_ROWS = min(ATTN_K_TILE, WIN)
SWA_RUNS = (SWA_TILE_WIN_ROWS + 2 * (BLOCK_SIZE - 1)) // BLOCK_SIZE
BIAS_T_TILE = min(T, 8)

@pl.jit.inline
def sparse_attn_csa(q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16], ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32], idx_topk: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32], position_ids: pl.Tensor[[T_DYN, 1], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16], cache_ready_dep: pl.Scalar[pl.TASK_ID]) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write CSA heads as ``[group, T_PAD, O_GROUP_IN]`` slabs.

    Only the first runtime ``t_dim`` rows in each group are valid. The
    returned task ID covers every write to the packed output tensor.
    """
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    t_dim = pl.tensor.dim(q, 0)
    t_heads = t_dim * H
    t_blk = t_dim * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
    t_hblocks = t_dim * (H // H_TILE)
    qk_items = t_dim * SPARSE_BLOCKS
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='kv_touch', allow_early_resolve=True):
        ori_kv_flat[0:1, 0:HEAD_DIM] = ori_kv_flat[0:1, 0:HEAD_DIM]
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    cmp_sparse_indices = pl.create_tensor([t_dim, CMP_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([t_dim, SPARSE_BLOCKS], dtype=pl.INT32)
    qk_order = pl.create_tensor([QK_ITEMS], dtype=pl.INT32)
    qk_wcur = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='csa_slots_build_valid_qk_plan') as qk_plan_tid:
        for bias_t0 in pl.range(0, t_dim, BIAS_T_TILE):
            c_raw = pl.cast(idx_topk[bias_t0:bias_t0 + BIAS_T_TILE, 0:IDX_TOPK], target_type=pl.FP32)
            c_pos = pl.cast(position_ids[bias_t0:bias_t0 + BIAS_T_TILE, 0:1], target_type=pl.FP32)
            c_pos_scaled = pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV)
            c_pos_i32 = pl.cast(c_pos_scaled, target_type=pl.INT32, mode='trunc')
            c_pos_q = pl.cast(c_pos_i32, target_type=pl.FP32)
            c_upper_b = pl.row_expand_mul(pl.full([BIAS_T_TILE, IDX_TOPK], dtype=pl.FP32, value=1.0), c_pos_q)
            c_ge = pl.minimum(pl.maximum(pl.add(c_raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
            c_lt = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw), 0.0), 1.0)
            c_mask = pl.mul(c_ge, c_lt)
            c_out = pl.sub(pl.mul(c_mask, pl.add(c_raw, 1.0)), 1.0)
            cmp_sparse_indices[bias_t0:bias_t0 + BIAS_T_TILE, 0:IDX_TOPK] = pl.cast(c_out, target_type=pl.INT32)
            for c_t0 in pl.range(BIAS_T_TILE):
                pl.write(valid_block_mask, [bias_t0 + c_t0, 0], pl.cast(1, pl.INT32))
            for c_sb in pl.range(1, SPARSE_BLOCKS):
                c_s0 = (c_sb - 1) * ATTN_K_TILE
                c_blk_valid = pl.row_max(c_mask[:, c_s0:c_s0 + ATTN_K_TILE])
                for c_dt in pl.range(BIAS_T_TILE):
                    c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), target_type=pl.INT32)
                    pl.write(valid_block_mask, [bias_t0 + c_dt, c_sb], c_valid)
            v_win_f = pl.cast(window_swa_indices[bias_t0:bias_t0 + BIAS_T_TILE, 0:WIN], target_type=pl.FP32)
            v_win_valid = pl.minimum(pl.maximum(pl.add(v_win_f, 1.0), 0.0), 1.0)
            sparse_bias[bias_t0:bias_t0 + BIAS_T_TILE, 0:WIN] = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
            sparse_bias[bias_t0:bias_t0 + BIAS_T_TILE, WIN:TOPK] = pl.mul(pl.minimum(c_out, 0.0), -NEG_INF)
            if PADDED_TOPK > TOPK:
                sparse_bias[bias_t0:bias_t0 + BIAS_T_TILE, TOPK:PADDED_TOPK] = pl.full([BIAS_T_TILE, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)
        pl.write(qk_wcur, [0], pl.cast(0, pl.INT32))
        for plan_t in pl.range(t_dim):
            for plan_sb in pl.range(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [plan_t, plan_sb]) > 0:
                    plan_w = pl.read(qk_wcur, [0])
                    pl.write(qk_order, [plan_w], pl.cast(plan_t * SPARSE_BLOCKS + plan_sb, pl.INT32))
                    pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
        for plan_t in pl.range(t_dim):
            for plan_sb in pl.range(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [plan_t, plan_sb]) <= 0:
                    plan_w = pl.read(qk_wcur, [0])
                    pl.write(qk_order, [plan_w], pl.cast(plan_t * SPARSE_BLOCKS + plan_sb, pl.INT32))
                    pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_heads, HEAD_DIM])
    sparse_blk_mi = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([t_blk, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(NUM_QK_CORES, name_hint='qk_pv', deps=[qk_plan_tid, cache_ready_dep], allow_early_resolve=True) as _qk_tid:
        qk_core = pl.tile.get_block_idx()
        qk_lane_iters = (qk_items - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES
        for qk_it in pl.range(qk_lane_iters):
            qk_flat = qk_core + qk_it * NUM_QK_CORES
            qk_item = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)
            qk_t = qk_item // SPARSE_BLOCKS
            qk_sb = qk_item - qk_t * SPARSE_BLOCKS
            qk_b = qk_t // S
            qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_s0:qk_s0 + ATTN_K_TILE]
            qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])
            if qk_block_valid > 0:
                qk_kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
                qk_win_rows = pl.min(pl.max(WIN - qk_s0, 0), ATTN_K_TILE)
                if qk_win_rows > 0:
                    qk_pos = pl.cast(pl.read(position_ids, [qk_t, 0]), pl.INDEX)
                    qk_win_len = pl.min(qk_pos + 1, WIN)
                    qk_win_start = qk_pos - qk_win_len + 1
                    qk_run_rows = pl.min(pl.max(qk_win_len - qk_s0, 0), qk_win_rows)
                    qk_head = (qk_win_start + qk_s0) % BLOCK_SIZE
                    for qk_run in pl.unroll(SWA_RUNS):
                        qk_run_lo = pl.max(qk_run * BLOCK_SIZE - qk_head, 0)
                        qk_run_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head, qk_run_rows)
                        if qk_run_hi > qk_run_lo:
                            qk_run_raw = pl.read(window_swa_indices, [qk_t, qk_s0 + qk_run_lo])
                            qk_run_src = pl.cast(pl.max(qk_run_raw, 0), pl.INDEX)
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_run_lo, 0], [qk_run_src, 0], [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_run_hi - qk_run_lo, HEAD_DIM])
                    qk_tail_n = qk_win_rows - qk_run_rows
                    if qk_tail_n > 0:
                        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_run_rows, 0], [0, 0], [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_tail_n, HEAD_DIM])
                for qk_r in pl.range(qk_win_rows, ATTN_K_TILE):
                    qk_cmp_k = qk_s0 + qk_r - WIN
                    if qk_cmp_k < CMP_TOPK:
                        qk_ridx = pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k])
                        if qk_ridx >= 0:
                            qk_slot = qk_ridx
                            qk_cblk = pl.cast(pl.read(cmp_block_table, [qk_b, qk_slot // BLOCK_SIZE]), pl.INDEX)
                            qk_csrc = qk_cblk * BLOCK_SIZE + qk_slot % BLOCK_SIZE
                            qk_kv = pl.gather_row(qk_kv, cmp_kv_flat, [qk_r, 0], [qk_csrc, 0], [1, HEAD_DIM])
                        else:
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])
                    else:
                        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])
                for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                    qk_h0 = qk_hb * QK_M_TILE
                    qk_head_row = qk_t * H + qk_h0
                    qk_q_tile = q_flat[qk_head_row:qk_head_row + QK_M_TILE, 0:HEAD_DIM]
                    qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                    qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                    qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                    qk_mi = pl.row_max(qk_scores)
                    qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                    qk_li = pl.row_sum(qk_exp)
                    qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode='rint')
                    qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                    for qk_sub in pl.unroll(QK_M_TILE // H_TILE):
                        qk_h_idx = qk_hb * (QK_M_TILE // H_TILE) + qk_sub
                        qk_r0 = qk_sub * H_TILE
                        qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                        qk_row = qk_blk_base + qk_sb * H_TILE
                        sparse_blk_mi[qk_row:qk_row + H_TILE, 0:1] = qk_mi[qk_r0:qk_r0 + H_TILE, 0:1]
                        sparse_blk_li[qk_row:qk_row + H_TILE, 0:1] = qk_li[qk_r0:qk_r0 + H_TILE, 0:1]
                        sparse_blk_oi[qk_row:qk_row + H_TILE, 0:HEAD_DIM] = qk_oi[qk_r0:qk_r0 + H_TILE, 0:HEAD_DIM]
            else:
                qk_oi_zero = pl.full([H_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                for qk_h_idx in pl.range(H // H_TILE):
                    qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                    qk_row = qk_blk_base + qk_sb * H_TILE
                    for qk_hr in pl.range(H_TILE):
                        pl.write(sparse_blk_mi, [qk_row + qk_hr, 0], -3e+38)
                        pl.write(sparse_blk_li, [qk_row + qk_hr, 0], 0.0)
                    sparse_blk_oi[qk_row:qk_row + H_TILE, 0:HEAD_DIM] = qk_oi_zero
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='rope_cs', allow_early_resolve=True):
        sw_ones = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        sw_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_idx_f)
        sw_dup_i32 = pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode='trunc')
        sw_dup_f = pl.cast(sw_dup_i32, target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        sw_swap_f = pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0))
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(sw_swap_f, target_type=pl.INT32)
        cs_ones = pl.full([ROPE_CS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        cs_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_idx_f)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode='trunc')
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        for cs_rb in pl.range(rope_cs_blocks):
            cs_t0 = cs_rb * ROPE_CS_T_TILE
            cs_cos = pl.cast(freqs_cos[cs_t0:cs_t0 + ROPE_CS_T_TILE, 0:HALF_ROPE], target_type=pl.FP32)
            cs_sin = pl.cast(freqs_sin[cs_t0:cs_t0 + ROPE_CS_T_TILE, 0:HALF_ROPE], target_type=pl.FP32)
            rope_cos_il[cs_t0:cs_t0 + ROPE_CS_T_TILE, 0:ROPE_DIM] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
            cs_sin_il = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
            rope_sin_signed[cs_t0:cs_t0 + ROPE_CS_T_TILE, 0:ROPE_DIM] = pl.mul(cs_sin_il, cs_sign)
    with pl.spmd(t_hblocks, name_hint='merge_norm') as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_idx * SPARSE_BLOCKS * H_TILE
        m_mi = sparse_blk_mi[m_blk_base:m_blk_base + H_TILE, 0:1]
        m_li = sparse_blk_li[m_blk_base:m_blk_base + H_TILE, 0:1]
        m_oi = sparse_blk_oi[m_blk_base:m_blk_base + H_TILE, 0:HEAD_DIM]
        for m_sb in pl.pipeline(1, SPARSE_BLOCKS, stage=2):
            m_row = m_blk_base + m_sb * H_TILE
            m_cur_mi = sparse_blk_mi[m_row:m_row + H_TILE, 0:1]
            m_cur_li = sparse_blk_li[m_row:m_row + H_TILE, 0:1]
            m_cur_oi = sparse_blk_oi[m_row:m_row + H_TILE, 0:HEAD_DIM]
            m_mi_new = pl.maximum(m_mi, m_cur_mi)
            m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
            m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
            m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
            m_oi = pl.add(pl.row_expand_mul(m_oi, m_alpha), pl.row_expand_mul(m_cur_oi, m_beta))
            m_mi = m_mi_new
        n_sink_bias = pl.reshape(attn_sink[m_h0:m_h0 + H_TILE], [H_TILE, 1])
        n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
        n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
        n_full = pl.row_expand_div(m_oi, n_denom)[0:H_TILE, 0:HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode='rint')
        m_rope = n_full[0:H_TILE, NOPE_DIM:HEAD_DIM]
        m_cos_il = rope_cos_il[m_t:m_t + 1, 0:ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t:m_t + 1, 0:ROPE_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[0:H_TILE, 0:ROPE_DIM])
        m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode='rint')
        n_full_bf16 = pl.concat(n_bf16[:, :NOPE_DIM], n_rope_bf16)
        for n_hi in pl.unroll(H_TILE):
            n_pack_row = (m_h0 + n_hi) // HEADS_PER_GROUP * T_PAD + m_t
            n_col = (m_h0 + n_hi) % HEADS_PER_GROUP * HEAD_DIM
            o_packed_heads[n_pack_row:n_pack_row + 1, n_col:n_col + HEAD_DIM] = n_full_bf16[n_hi:n_hi + 1, :]
    return (o_packed_heads, merge_tid)
