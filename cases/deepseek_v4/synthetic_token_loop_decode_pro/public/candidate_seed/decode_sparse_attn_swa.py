import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, DECODE_ORI_BLOCK_NUM, INT8_SCALE_MAX, INT8_AMAX_EPS
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
SOFTMAX_SCALE = M.softmax_scale
from config import O_LORA
from config import O_GROUPS
HEADS_PER_GROUP = H // O_GROUPS
from config import O_GROUP_IN
from config import ORI_BLOCK_NUM
GATHER_SPLITS = 4
GATHER_ROWS_PER_TASK = WIN // GATHER_SPLITS
GATHER_RUN = 16
H_TILE = 16
QK_M_TILE = 32
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 256
PROJ_A_MM_N_TILE = 128
MM_T_TILE = 16
T_PAD = (T + MM_T_TILE - 1) // MM_T_TILE * MM_T_TILE
B_K_TILE = 256
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512
QUANT_TOKEN_TILE = 8
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE
PROJ_B_D_CHUNK = 512
PB_DCHUNKS = D // PROJ_B_D_CHUNK
PROJ_B_ACT_T_TILE = 8
PROJ_B_ACT_TBLK = 8
PB_ACT_NREG = D // PROJ_B_ACT_N_TILE
PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK
SPARSE_BLOCKS = 1
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

@pl.jit.inline
def sparse_attn_swa(q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16], ori_kv_flat: pl.Tensor[[ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM], pl.BF16], swa_indices: pl.Tensor[[T, WIN], pl.INT32], sparse_bias: pl.Tensor[[T, PADDED_TOPK], pl.FP32], attn_sink: pl.Tensor[[H], pl.FP32], freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], attn_out: pl.Tensor[[T, D], pl.BF16]):
    """Standalone sparse attention with a BF16 projected output."""
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    swa_kv_flat = pl.create_tensor([T * WIN, HEAD_DIM], dtype=pl.BF16)
    gather_tids = pl.array.create(1, pl.TASK_ID)
    with pl.spmd(T * GATHER_SPLITS, name_hint='swa_gather_kv') as gather_tid:
        g_task = pl.tile.get_block_idx()
        g_t = g_task // GATHER_SPLITS
        g_split = g_task - g_t * GATHER_SPLITS
        g_r0 = g_split * GATHER_ROWS_PER_TASK
        g_base = g_t * WIN
        for g_sub in pl.range(GATHER_ROWS_PER_TASK // GATHER_RUN):
            g_sr0 = g_r0 + g_sub * GATHER_RUN
            g_sdst = g_base + g_sr0
            g_first = pl.read(swa_indices, [g_t, g_sr0])
            g_last = pl.read(swa_indices, [g_t, g_sr0 + GATHER_RUN - 1])
            g_run_ok = g_last - g_first + pl.min(g_first, 0) * GATHER_RUN
            if g_run_ok == GATHER_RUN - 1:
                g_run_src = pl.cast(g_first, pl.INDEX)
                swa_kv_flat[g_sdst:g_sdst + GATHER_RUN, 0:HEAD_DIM] = ori_kv_flat[g_run_src:g_run_src + GATHER_RUN, 0:HEAD_DIM]
            else:
                for g_dr in pl.range(GATHER_RUN):
                    g_dst = g_sdst + g_dr
                    g_slot_i32 = pl.read(swa_indices, [g_t, g_sr0 + g_dr])
                    if g_slot_i32 >= 0:
                        g_slot = pl.cast(g_slot_i32, pl.INDEX)
                        swa_kv_flat[g_dst:g_dst + 1, 0:HEAD_DIM] = ori_kv_flat[g_slot:g_slot + 1, 0:HEAD_DIM]
                    else:
                        swa_kv_flat[g_dst:g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    gather_tids[0] = gather_tid
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    o_packed_heads = pl.create_tensor([O_GROUPS * T * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T, O_GROUP_IN])
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(T, name_hint='qk_pv', deps=[gather_tids[0]], allow_early_resolve=True) as qk_tid:
        qk_t = pl.tile.get_block_idx()
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        for qk_sb in pl.unroll(SPARSE_BLOCKS):
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_s0:qk_s0 + ATTN_K_TILE]
            qk_base = qk_t * WIN + qk_s0
            qk_kv = swa_kv_flat[qk_base:qk_base + ATTN_K_TILE, 0:HEAD_DIM]
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
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='rope_cs') as rope_tid:
        swap_ones = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        swap_range_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        swap_range = pl.cast(swap_range_i32, target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_range)
        swap_half = pl.mul(swap_col, 0.5)
        swap_dup_i32 = pl.cast(swap_half, target_type=pl.INT32, mode='trunc')
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))
        swap_next = pl.add(swap_col, 1.0)
        swap_stride = pl.mul(swap_lane, 2.0)
        swap_idx_f = pl.sub(swap_next, swap_stride)
        rope_swap_idx[:, :] = pl.cast(swap_idx_f, target_type=pl.INT32)
        cs_ones = pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
        cs_range_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
        cs_range = pl.cast(cs_range_i32, target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_range)
        cs_half = pl.mul(cs_col, 0.5)
        cs_dup_i32 = pl.cast(cs_half, target_type=pl.INT32, mode='trunc')
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign_base = pl.sub(pl.mul(cs_lane, 2.0), 1.0)
        cs_sign = pl.neg(cs_sign_base)
        for cp in pl.range(HALF_ROPE // ROPE_TILE):
            cp_r0 = cp * ROPE_TILE
            cp_c0 = 2 * cp_r0
            cs_cos = pl.cast(freqs_cos[0:T, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
            cs_sin = pl.cast(freqs_sin[0:T, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
            cs_cos_dup = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
            cs_sin_dup = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
            cs_sin_signed = pl.mul(cs_sin_dup, cs_sign)
            rope_cos_il[0:T, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_dup
            rope_sin_signed[0:T, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = cs_sin_signed
    with pl.spmd(T * (H // H_TILE), name_hint='merge_norm', deps=[qk_tid, rope_tid], allow_early_resolve=True) as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_t * H + m_h0
        m_mi = sparse_blk_mi[m_blk_base:m_blk_base + H_TILE, 0:1]
        m_li = sparse_blk_li[m_blk_base:m_blk_base + H_TILE, 0:1]
        m_oi = sparse_blk_oi[m_blk_base:m_blk_base + H_TILE, 0:HEAD_DIM]
        n_sink = pl.reshape(attn_sink[m_h0:m_h0 + H_TILE], [H_TILE, 1])
        n_sink_delta = pl.sub(n_sink, m_mi)
        n_sink_exp = pl.exp(n_sink_delta)
        n_denom = pl.add(m_li, n_sink_exp)
        n_normalized = pl.row_expand_div(m_oi, n_denom)
        n_full = n_normalized[0:H_TILE, 0:HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode='rint')
        m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[:, :])
        m_cos_il = rope_cos_il[m_t:m_t + 1, 0:ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t:m_t + 1, 0:ROPE_DIM]
        m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
        m_swap_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
        m_rot = pl.add(m_rope_cos, m_swap_sin)
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode='rint')
        m_g0 = m_h0 // HEADS_PER_GROUP
        for m_sg in pl.unroll(H_TILE // HEADS_PER_GROUP):
            m_src_h0 = m_sg * HEADS_PER_GROUP
            n_pack_row = (m_g0 + m_sg) * T + m_t
            n_dst_head = n_pack_row * HEADS_PER_GROUP
            o_packed_heads[n_dst_head:n_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM] = n_bf16[m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM]
            o_packed_heads[n_dst_head:n_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM] = n_rope_bf16[m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM]
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            with pl.spmd(PA_NFRAGS, name_hint='proj_a_mm', deps=[merge_tid], allow_early_resolve=True) as pa_tid:
                nf = pl.tile.get_block_idx()
                n0 = nf * PROJ_A_MM_N_TILE
                xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, 0], valid_shape=[T, A_K_TILE])
                wa0_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    xa_k_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, k0], valid_shape=[T, A_K_TILE])
                    wa_k_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, k0:k0 + A_K_TILE]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                o_r_pad = pl.assemble(o_r_pad, acc_a, [0, out_col_g + n0])
            col_g = g * O_LORA
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='quant', deps=[pa_tid], allow_early_resolve=True) as q_tid:
                for qt in pl.pipeline(0, T, QUANT_TOKEN_TILE, stage=2):
                    oc_amax = o_r_pad[qt:qt + QUANT_TOKEN_TILE, col_g:col_g + O_LORA]
                    g_abs = pl.abs(oc_amax)
                    g_row_max = pl.row_max(g_abs)
                    g_row_max = pl.reshape(g_row_max, [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    g_scale_dq = pl.mul(g_amax, 1.0 / INT8_SCALE_MAX)
                    act_scale_dq = pl.assemble(act_scale_dq, g_scale_dq, [g, qt])
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt:qt + QUANT_TOKEN_TILE, col_g:col_g + O_LORA]
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode='rint')
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode='round')
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode='trunc')
                    o_r_i8_pad = pl.assemble(o_r_i8_pad, oq_i8, [qt, col_g])
                    if T_PAD > T:
                        zero_half = pl.full([T_PAD - T, O_LORA], dtype=pl.FP16, value=0.0)
                        zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode='trunc')
                        o_r_i8_pad = pl.assemble(o_r_i8_pad, zero_i8, [T, col_g])
            with pl.spmd(PB_DCHUNKS, name_hint='proj_b_mm', deps=[q_tid], allow_early_resolve=True) as pb_tid:
                dc = pl.tile.get_block_idx()
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
            proj_b_tids[g] = pb_tid
    with pl.spmd(PB_ACT_NREG * PB_ACT_TBLKS, name_hint='proj_b_act', deps=[proj_b_tids[i] for i in range(O_GROUPS)], allow_early_resolve=True) as _act_tid:
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
