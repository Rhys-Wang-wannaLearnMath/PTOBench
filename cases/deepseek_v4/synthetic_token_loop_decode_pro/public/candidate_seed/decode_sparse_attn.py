import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, DECODE_CMP_BLOCK_NUM, DECODE_ORI_BLOCK_NUM, KV_CMP_MAX_BLOCKS, INT8_SCALE_MAX, INT8_AMAX_EPS
from config import B
from config import S
from config import T
from config import D
from config import H
from config import HEAD_DIM
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
IDX_TOPK = M.index_topk
CMP_TOPK = IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
from config import O_LORA
from config import O_GROUPS
HEADS_PER_GROUP = H // O_GROUPS
from config import O_GROUP_IN
O_GROUPS_PER_SPMD = 2
DEFAULT_COMPRESS_RATIO = 4
from config import MAX_SEQ_LEN
INDEXER_SCORE_LEN = MAX_SEQ_LEN // 4
COMPRESS_RATIO_INV = 1.0 / DEFAULT_COMPRESS_RATIO
CSA_CMP_GE_BIAS = 1.0
from config import ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
H_TILE = 16
MERGE_SWAP_FLAT_LEN = H_TILE * ROPE_DIM
ROPE_DIM_F = float(ROPE_DIM)
ROPE_DIM_INV = 1.0 / ROPE_DIM
QK_M_TILE = 32
ATTN_K_TILE = 128
NUM_QK_CORES = 24
A_K_TILE = 256
PROJ_A_MM_N_TILE = 128
MM_T_TILE = 16
T_PAD = (T + MM_T_TILE - 1) // MM_T_TILE * MM_T_TILE
B_K_TILE = 256
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512
QUANT_TOKEN_TILE = 8
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE
PA_NF_PER_BLOCK = 2
PA_BLOCKS = PA_NFRAGS // PA_NF_PER_BLOCK
PROJ_B_D_CHUNK = 512 if M.name == 'flash' else 1792
PB_DCHUNKS = D // PROJ_B_D_CHUNK
PROJ_B_ACT_T_TILE = 8
PROJ_B_ACT_TBLK = 8
PB_ACT_NREG = D // PROJ_B_ACT_N_TILE
PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK
NEG_INF = -1e+20
TOPK = WIN + CMP_TOPK
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
QK_ITEMS = T * SPARSE_BLOCKS

@pl.jit.inline
def sparse_attn(q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16], ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], window_swa_indices: pl.Tensor[[T, WIN], pl.INT32], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32], idx_topk: pl.Tensor[[T, INDEXER_SCORE_LEN], pl.INT32], position_ids: pl.Tensor[[T, 1], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], attn_out: pl.Tensor[[T, D], pl.BF16]):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='kv_touch'):
        ori_kv_flat[0:T, 0:HEAD_DIM] = ori_kv_flat[0:T, 0:HEAD_DIM]
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)
    cmp_sparse_indices = pl.create_tensor([T, CMP_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([T, SPARSE_BLOCKS], dtype=pl.INT32)
    qk_order = pl.create_tensor([QK_ITEMS], dtype=pl.INT32)
    qk_wcur = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='csa_slots_build_valid_qk_plan', allow_early_resolve=True) as qk_plan_tid:
        c_raw = pl.cast(idx_topk[0:T, 0:IDX_TOPK], target_type=pl.FP32)
        c_pos = pl.cast(position_ids[0:T, 0:1], target_type=pl.FP32)
        c_pos_q = pl.cast(pl.cast(pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        c_upper_b = pl.row_expand_mul(pl.full([T, IDX_TOPK], dtype=pl.FP32, value=1.0), c_pos_q)
        c_ge = pl.minimum(pl.maximum(pl.add(c_raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
        c_lt = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw), 0.0), 1.0)
        c_mask = pl.mul(c_ge, c_lt)
        c_out = pl.sub(pl.mul(c_mask, pl.add(c_raw, 1.0)), 1.0)
        cmp_sparse_indices[0:T, 0:IDX_TOPK] = pl.cast(c_out, target_type=pl.INT32)
        for c_t0 in pl.range(T):
            pl.write(valid_block_mask, [c_t0, 0], pl.cast(1, pl.INT32))
        for c_sb in pl.range(1, SPARSE_BLOCKS):
            c_s0 = (c_sb - 1) * ATTN_K_TILE
            c_blk_valid = pl.row_max(c_mask[:, c_s0:c_s0 + ATTN_K_TILE])
            for c_dt in pl.range(T):
                c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), target_type=pl.INT32)
                pl.write(valid_block_mask, [c_dt, c_sb], c_valid)
        v_win_f = pl.cast(window_swa_indices[0:T, 0:WIN], target_type=pl.FP32)
        v_win_valid = pl.minimum(pl.maximum(pl.add(v_win_f, 1.0), 0.0), 1.0)
        sparse_bias[0:T, 0:WIN] = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
        sparse_bias[0:T, WIN:TOPK] = pl.mul(pl.minimum(c_out, 0.0), -NEG_INF)
        if PADDED_TOPK > TOPK:
            sparse_bias[0:T, TOPK:PADDED_TOPK] = pl.full([T, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)
        pl.write(qk_wcur, [0], pl.cast(0, pl.INT32))
        for plan_t in pl.unroll(T):
            for plan_sb in pl.unroll(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [plan_t, plan_sb]) > 0:
                    plan_w = pl.read(qk_wcur, [0])
                    pl.write(qk_order, [plan_w], pl.cast(plan_t * SPARSE_BLOCKS + plan_sb, pl.INT32))
                    pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
        for plan_t in pl.unroll(T):
            for plan_sb in pl.unroll(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [plan_t, plan_sb]) <= 0:
                    plan_w = pl.read(qk_wcur, [0])
                    pl.write(qk_order, [plan_w], pl.cast(plan_t * SPARSE_BLOCKS + plan_sb, pl.INT32))
                    pl.write(qk_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
    with pl.spmd(NUM_QK_CORES, name_hint='qk_pv', deps=[qk_plan_tid], allow_early_resolve=True) as qk_tid:
        qk_core = pl.tile.get_block_idx()
        qk_lane_iters = (QK_ITEMS - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES
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
                for qk_r in pl.range(ATTN_K_TILE):
                    qk_k = qk_s0 + qk_r
                    if qk_k < WIN:
                        qk_win_slot_i32 = pl.read(window_swa_indices, [qk_t, qk_k])
                        if qk_win_slot_i32 >= 0:
                            qk_win_slot = pl.cast(qk_win_slot_i32, pl.INDEX)
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [qk_win_slot, 0], [1, HEAD_DIM])
                        else:
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])
                    else:
                        qk_cmp_k = qk_k - WIN
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
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx_m = pl.create_tensor([1, MERGE_SWAP_FLAT_LEN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='rope_cs', allow_early_resolve=True):
        sw_col_m = pl.cast(pl.arange(0, [1, MERGE_SWAP_FLAT_LEN], dtype=pl.INT32), target_type=pl.FP32)
        sw_rowbase_m = pl.mul(pl.cast(pl.cast(pl.mul(sw_col_m, ROPE_DIM_INV), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32), ROPE_DIM_F)
        sw_j_m = pl.sub(sw_col_m, sw_rowbase_m)
        sw_dup_f_m = pl.cast(pl.cast(pl.mul(sw_j_m, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        sw_lane_m = pl.sub(sw_j_m, pl.mul(sw_dup_f_m, 2.0))
        sw_jx_m = pl.sub(pl.add(sw_j_m, 1.0), pl.mul(sw_lane_m, 2.0))
        rope_swap_idx_m[0:1, 0:MERGE_SWAP_FLAT_LEN] = pl.cast(pl.add(sw_rowbase_m, sw_jx_m), target_type=pl.INT32)
        cs_col = pl.col_expand_mul(pl.full([T, ROPE_DIM], dtype=pl.FP32, value=1.0), pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        cs_cos = pl.cast(freqs_cos[0:T, 0:HALF_ROPE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, 0:HALF_ROPE], target_type=pl.FP32)
        rope_cos_il[0:T, 0:ROPE_DIM] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, 0:ROPE_DIM] = pl.mul(pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)
    merge_tids = pl.array.create(O_GROUPS // O_GROUPS_PER_SPMD, pl.TASK_ID)
    for merge_group in pl.parallel(O_GROUPS // O_GROUPS_PER_SPMD):
        with pl.spmd(T, name_hint='merge_norm') as merge_tid:
            m_t = pl.tile.get_block_idx()
            m_h_idx = merge_group
            m_h0 = m_h_idx * H_TILE
            m_idx = m_t * (H // H_TILE) + m_h_idx
            m_blk_base = m_idx * SPARSE_BLOCKS * H_TILE
            m_mi = sparse_blk_mi[m_blk_base:m_blk_base + H_TILE, 0:1]
            m_li = sparse_blk_li[m_blk_base:m_blk_base + H_TILE, 0:1]
            m_oi = sparse_blk_oi[m_blk_base:m_blk_base + H_TILE, 0:HEAD_DIM]
            if SPARSE_BLOCKS > 1:
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
            n_nope = pl.row_expand_div(m_oi[0:H_TILE, 0:NOPE_DIM], n_denom)
            n_nope_bf16 = pl.cast(n_nope, target_type=pl.BF16, mode='rint')
            m_swap_flat = rope_swap_idx_m[0:1, 0:MERGE_SWAP_FLAT_LEN]
            m_rope = pl.row_expand_div(m_oi[0:H_TILE, NOPE_DIM:HEAD_DIM], n_denom)
            m_cos_il = rope_cos_il[m_t:m_t + 1, 0:ROPE_DIM]
            m_sin_signed = rope_sin_signed[m_t:m_t + 1, 0:ROPE_DIM]
            m_rope_flat = pl.reshape(m_rope, [1, MERGE_SWAP_FLAT_LEN])
            m_swapped = pl.reshape(pl.gather(m_rope_flat, dim=-1, index=m_swap_flat), [H_TILE, ROPE_DIM])
            m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
            n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode='rint')
            n_full_bf16 = pl.concat(n_nope_bf16, n_rope_bf16)
            for n_hi in pl.range(H_TILE):
                n_gh = m_h0 + n_hi
                n_g = n_gh // HEADS_PER_GROUP
                n_hh = n_gh - n_g * HEADS_PER_GROUP
                n_pack_row = n_g * T + m_t
                n_col = n_hh * HEAD_DIM
                o_packed[n_pack_row:n_pack_row + 1, n_col:n_col + HEAD_DIM] = n_full_bf16[n_hi:n_hi + 1, 0:HEAD_DIM]
        merge_tids[merge_group] = merge_tid
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS // O_GROUPS_PER_SPMD, pl.TASK_ID)
    with pl.manual_scope():
        for group_bundle in pl.parallel(O_GROUPS // O_GROUPS_PER_SPMD):
            with pl.spmd(O_GROUPS_PER_SPMD * PA_BLOCKS, name_hint='proj_a_mm', deps=[merge_tids[group_bundle]], allow_early_resolve=True) as pa_tid:
                pa_idx = pl.tile.get_block_idx()
                pa_local_group = pa_idx // PA_BLOCKS
                pa_blk = pa_idx - pa_local_group * PA_BLOCKS
                g = group_bundle * O_GROUPS_PER_SPMD + pa_local_group
                row_base_o = g * T
                out_col_g = g * O_LORA
                for anf in pl.range(PA_NF_PER_BLOCK):
                    n0 = pa_blk * (PA_NF_PER_BLOCK * PROJ_A_MM_N_TILE) + anf * PROJ_A_MM_N_TILE
                    xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, 0], valid_shape=[T, A_K_TILE])
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, k0], valid_shape=[T, A_K_TILE])
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, k0:k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    o_r_pad = pl.assemble(o_r_pad, acc_a, [0, out_col_g + n0])
            with pl.spmd(O_GROUPS_PER_SPMD, name_hint='quant', deps=[pa_tid], allow_early_resolve=True) as q_tid:
                q_local_group = pl.tile.get_block_idx()
                g = group_bundle * O_GROUPS_PER_SPMD + q_local_group
                col_g = g * O_LORA
                for qt in pl.pipeline(0, T, QUANT_TOKEN_TILE, stage=2):
                    oc_amax = o_r_pad[qt:qt + QUANT_TOKEN_TILE, col_g:col_g + O_LORA]
                    g_abs = pl.abs(oc_amax)
                    g_row_max = pl.row_max(g_abs)
                    g_row_max = pl.reshape(g_row_max, [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    act_scale_dq = pl.assemble(act_scale_dq, pl.recip(g_sq_row), [g, qt])
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oq_scaled = pl.row_expand_mul(oc_amax, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode='rint')
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode='round')
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode='trunc')
                    o_r_i8_pad = pl.assemble(o_r_i8_pad, oq_i8, [qt, col_g])
            with pl.spmd(O_GROUPS_PER_SPMD * PB_DCHUNKS, name_hint='proj_b_mm', deps=[q_tid], allow_early_resolve=True) as pb_tid:
                pb_idx = pl.tile.get_block_idx()
                pb_local_group = pb_idx // PB_DCHUNKS
                dc = pb_idx - pb_local_group * PB_DCHUNKS
                g = group_bundle * O_GROUPS_PER_SPMD + pb_local_group
                col_g = g * O_LORA
                d0 = dc * PROJ_B_D_CHUNK
                for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.create_tensor([MM_T_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                        k0 = col_g + kb * B_K_TILE
                        if kb == 0:
                            b_act = o_r_i8_pad[:, col_g:col_g + B_K_TILE]
                            b_weight = wo_b[n0:n0 + PROJ_B_MM_N_TILE, col_g:col_g + B_K_TILE]
                            acc_b = pl.matmul(b_act, b_weight, b_trans=True, out_dtype=pl.INT32)
                        else:
                            b_act = o_r_i8_pad[:, k0:k0 + B_K_TILE]
                            b_weight = wo_b[n0:n0 + PROJ_B_MM_N_TILE, k0:k0 + B_K_TILE]
                            acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True)
                    partials = pl.assemble(partials, acc_b, [0, g * D + n0])
            proj_b_tids[group_bundle] = pb_tid
    with pl.spmd(PB_ACT_NREG * PB_ACT_TBLKS, name_hint='proj_b_act', deps=[proj_b_tids[i] for i in range(O_GROUPS // O_GROUPS_PER_SPMD)], allow_early_resolve=True) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // PB_ACT_TBLKS
        tblk = act_idx - nreg * PB_ACT_TBLKS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TBLK
        wb_scale = wo_b_scale[ob_n0:ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TBLK, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb:b_tb + PROJ_B_ACT_T_TILE, p_col0:p_col0 + PROJ_B_ACT_N_TILE]
                g_scale_row = act_scale_dq[act_g:act_g + 1, b_tb:b_tb + PROJ_B_ACT_T_TILE]
                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode='none')
                p_g_scaled = pl.row_expand_mul(p_g_f32, g_scale)
                acc = pl.add(acc, p_g_scaled)
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode='rint')
            attn_out[b_tb:b_tb + PROJ_B_ACT_T_TILE, ob_n0:ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16
    return attn_out
