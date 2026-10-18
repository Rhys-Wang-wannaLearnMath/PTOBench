import pypto.language as pl

from config import CASE_CONFIG as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ

B = PREFILL_BATCH

S = PREFILL_SEQ

T = B * S

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

HALF_ROPE = ROPE_DIM // 2

ROPE_HALF = HALF_ROPE

NOPE_DIM = M.nope_head_dim

IDX_TOPK = M.index_topk

WIN = M.sliding_window

TOPK = WIN + IDX_TOPK

SOFTMAX_SCALE = M.softmax_scale

O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))

PREFILL_SPARSE_TOPK = min(TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)

HEAD_TILE = 16

QK_M_TILE = 32

BIAS_TOKEN_TILE = 16

QUANT_TOKEN_TILE = 8

ROPE_TILE = 16

ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE

A_K_TILE = 256

B_K_TILE = 256

QUANT_TILE = 512

PROJ_A_MM_N_TILE = 128

PROJ_B_MM_N_TILE = 256

PROJ_B_D_CHUNK = 1024

PROJ_B_ACT_N_TILE = 512

PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE

PB_DCHUNKS = D // PROJ_B_D_CHUNK

NUM_QUANT_T_CHUNKS = 4

QUANT_T_CHUNK = T // NUM_QUANT_T_CHUNKS

PROJ_B_ACT_T_TILE = 16

PROJ_B_ACT_TBLK = 32

PB_ACT_NREG = D // PROJ_B_ACT_N_TILE

PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK

PREFILL_ATTN_TILE = 128

PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE

PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE

MASK_ALIGN_ELEMS = 32 // 4

VALID_BLOCK_MASK_COLS = (PREFILL_ATTN_BLOCKS + MASK_ALIGN_ELEMS - 1) // MASK_ALIGN_ELEMS * MASK_ALIGN_ELEMS

SPARSE_BIAS_COLS = min(TOPK, PREFILL_SPARSE_PAD)

@pl.jit.inline
def _prefill_sparse_attn_math(q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16], sparse_kv: pl.Tensor[[T * PREFILL_SPARSE_PAD, HEAD_DIM], pl.BF16], sparse_bias: pl.Tensor[[T, PREFILL_SPARSE_PAD], pl.FP32], valid_block_mask: pl.Tensor[[T, VALID_BLOCK_MASK_COLS], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]], num_tokens: pl.Scalar[pl.INT32]):
    """Run source-independent sparse QK/PV, merge, inverse RoPE, and projection."""
    blk_rows = T * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
    sparse_blk_mi = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([blk_rows, HEAD_DIM], dtype=pl.FP32)
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.spmd(ROPE_HALF // ROPE_TILE, name_hint='rope_cs') as rope_cs_tid:
        cp = pl.tile.get_block_idx()
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        swap_ones = pl.full([HEAD_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        swap_col = pl.col_expand_mul(swap_ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        swap_dup_f = pl.cast(pl.cast(pl.mul(swap_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))
        swap_idx = pl.cast(pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0)), target_type=pl.INT32)
        rope_swap_idx[:, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = swap_idx[:, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE]
        cs_col = pl.col_expand_mul(pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0), pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:T, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)
    with pl.spmd(T, name_hint='qk_pv') as qk_tid:
        qk_t = pl.tile.get_block_idx()
        if qk_t < num_tokens:
            qk_kv_base = qk_t * PREFILL_SPARSE_PAD
            qk_token_base = qk_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
            for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                qk_s0 = qk_kv_base + qk_sb * PREFILL_ATTN_TILE
                qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_sb * PREFILL_ATTN_TILE:qk_sb * PREFILL_ATTN_TILE + PREFILL_ATTN_TILE]
                qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])
                if qk_sb == 0:
                    qk_block_valid = pl.cast(1, pl.INT32)
                if qk_block_valid > 0:
                    qk_kv_k = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                    qk_kv_v = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                    for qk_hb in pl.range(H // QK_M_TILE):
                        qk_head_row = qk_t * H + qk_hb * QK_M_TILE
                        qk_q_tile = q_flat[qk_head_row:qk_head_row + QK_M_TILE, :]
                        qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                        qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                        qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                        qk_mi = pl.row_max(qk_scores)
                        qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                        qk_li = pl.row_sum(qk_exp)
                        qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode='rint')
                        qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                        for qk_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                            qk_h_idx = qk_hb * (QK_M_TILE // HEAD_TILE) + qk_sub
                            qk_r0 = qk_sub * HEAD_TILE
                            qk_row = qk_token_base + qk_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE + qk_sb * HEAD_TILE
                            sparse_blk_mi[qk_row:qk_row + HEAD_TILE, :] = qk_mi[qk_r0:qk_r0 + HEAD_TILE, :]
                            sparse_blk_li[qk_row:qk_row + HEAD_TILE, :] = qk_li[qk_r0:qk_r0 + HEAD_TILE, :]
                            sparse_blk_oi[qk_row:qk_row + HEAD_TILE, :] = qk_oi[qk_r0:qk_r0 + HEAD_TILE, :]
    o_packed_heads = pl.create_tensor([O_GROUPS * T * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T, O_GROUP_IN])
    with pl.spmd(T, name_hint='merge_rope_pack', deps=[qk_tid, rope_cs_tid]) as merge_tid:
        m_t = pl.tile.get_block_idx()
        m_token_base = m_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
        m_mask_row = valid_block_mask[m_t:m_t + 1, :]
        m_swap_idx = rope_swap_idx[:, :]
        m_cos_il = rope_cos_il[m_t:m_t + 1, :]
        m_sin_signed = rope_sin_signed[m_t:m_t + 1, :]
        for m_h_idx in pl.range(H // HEAD_TILE):
            m_h0 = m_h_idx * HEAD_TILE
            if m_t < num_tokens:
                m_blk_base = m_token_base + m_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
                m_mi = sparse_blk_mi[m_blk_base:m_blk_base + HEAD_TILE, :]
                m_li = sparse_blk_li[m_blk_base:m_blk_base + HEAD_TILE, :]
                m_oi = sparse_blk_oi[m_blk_base:m_blk_base + HEAD_TILE, :]
                for m_sb in pl.unroll(1, PREFILL_ATTN_BLOCKS):
                    m_block_valid = pl.read(m_mask_row, [0, m_sb])
                    if m_block_valid > 0:
                        m_row = m_blk_base + m_sb * HEAD_TILE
                        cur_mi = sparse_blk_mi[m_row:m_row + HEAD_TILE, :]
                        cur_li = sparse_blk_li[m_row:m_row + HEAD_TILE, :]
                        cur_oi = sparse_blk_oi[m_row:m_row + HEAD_TILE, :]
                        mi_new = pl.maximum(m_mi, cur_mi)
                        alpha = pl.exp(pl.sub(m_mi, mi_new))
                        beta = pl.exp(pl.sub(cur_mi, mi_new))
                        m_li = pl.add(pl.mul(alpha, m_li), pl.mul(beta, cur_li))
                        m_oi = pl.add(pl.row_expand_mul(m_oi, alpha), pl.row_expand_mul(cur_oi, beta))
                        m_mi = mi_new
                sink_bias = attn_sink_col[m_h0:m_h0 + HEAD_TILE, :]
                sink_tile = pl.add(pl.sub(m_mi, m_mi), sink_bias)
                denom = pl.add(m_li, pl.exp(pl.sub(sink_tile, m_mi)))
                n_full = pl.row_expand_div(m_oi, denom)[0:HEAD_TILE, :]
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode='rint')
            else:
                n_full = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                n_bf16 = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
            m_swapped = pl.gather(m_rope, dim=-1, index=m_swap_idx)
            m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
            n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode='rint')
            if HEAD_TILE % HEADS_PER_GROUP == 0:
                m_g0 = m_h0 // HEADS_PER_GROUP
                for m_sg in pl.unroll(HEAD_TILE // HEADS_PER_GROUP):
                    m_src_h0 = m_sg * HEADS_PER_GROUP
                    m_pack_row = (m_g0 + m_sg) * T + m_t
                    m_dst_head = m_pack_row * HEADS_PER_GROUP
                    o_packed_heads[m_dst_head:m_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM] = n_bf16[m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM]
                    o_packed_heads[m_dst_head:m_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM] = n_rope_bf16[m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM]
            else:
                for m_hi in pl.range(HEAD_TILE):
                    m_gh = m_h0 + m_hi
                    m_g = m_gh // HEADS_PER_GROUP
                    m_pack_row = m_g * T + m_t
                    m_col = (m_gh - m_g * HEADS_PER_GROUP) * HEAD_DIM
                    o_packed[m_pack_row:m_pack_row + 1, m_col:m_col + NOPE_DIM] = n_bf16[m_hi:m_hi + 1, 0:NOPE_DIM]
                    o_packed[m_pack_row:m_pack_row + 1, m_col + NOPE_DIM:m_col + HEAD_DIM] = n_rope_bf16[m_hi:m_hi + 1, 0:ROPE_DIM]
    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)
    partials = pl.create_tensor([T, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    quant_tids = pl.array.create(O_GROUPS * NUM_QUANT_T_CHUNKS, pl.TASK_ID)
    proj_b_tids = pl.array.create(PB_DCHUNKS * O_GROUPS, pl.TASK_ID)
    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='proj_a_mm', deps=[merge_tid]) as pa_tid:
                    xa0_chunk = o_packed[row_base_o:row_base_o + T, 0:A_K_TILE]
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = o_packed[row_base_o:row_base_o + T, k0:k0 + A_K_TILE]
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, k0:k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    o_r = pl.assemble(o_r, pl.reshape(acc_a, [T, PROJ_A_MM_N_TILE]), [0, out_col_g + n0])
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid
        for tc in pl.parallel(NUM_QUANT_T_CHUNKS):
            t_base = tc * QUANT_T_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='quant', deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)]) as q_tid:
                    for qt in pl.range(t_base, t_base + QUANT_T_CHUNK, QUANT_TOKEN_TILE):
                        g_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            g_amax = pl.maximum(g_amax, pl.reshape(pl.row_max(pl.maximum(oc, pl.neg(oc))), [1, QUANT_TOKEN_TILE]))
                        g_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), g_amax)
                        act_scale_dq = pl.assemble(act_scale_dq, pl.recip(g_sq_row), [g, qt])
                        g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            oq_i32 = pl.cast(pl.row_expand_mul(oc, g_sq_col), target_type=pl.INT32, mode='rint')
                            oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode='round')
                            o_r_i8 = pl.assemble(o_r_i8, pl.cast(oq_half, target_type=pl.INT8, mode='trunc'), [qt, col_g + k1])
                quant_tids[g * NUM_QUANT_T_CHUNKS + tc] = q_tid
        for dc in pl.parallel(PB_DCHUNKS):
            d0 = dc * PROJ_B_D_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='proj_b_mm', deps=[quant_tids[g * NUM_QUANT_T_CHUNKS + tc] for tc in range(NUM_QUANT_T_CHUNKS)]) as pb_tid:
                    for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        acc_b = pl.matmul(o_r_i8[:, col_g:col_g + B_K_TILE], wo_b[n0:n0 + PROJ_B_MM_N_TILE, col_g:col_g + B_K_TILE], b_trans=True, out_dtype=pl.INT32)
                        for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                            k0 = col_g + kb * B_K_TILE
                            acc_b = pl.matmul_acc(acc_b, o_r_i8[:, k0:k0 + B_K_TILE], wo_b[n0:n0 + PROJ_B_MM_N_TILE, k0:k0 + B_K_TILE], b_trans=True)
                        partials = pl.assemble(partials, acc_b, [0, g * D + n0])
                proj_b_tids[dc * O_GROUPS + g] = pb_tid
    with pl.spmd(PB_ACT_NREG * PB_ACT_TBLKS, name_hint='proj_b_act', deps=[proj_b_tids[i] for i in range(PB_DCHUNKS * O_GROUPS)]) as act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // PB_ACT_TBLKS
        tblk = act_idx - nreg * PB_ACT_TBLKS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TBLK
        wb_scale_chunk = pl.reshape(wo_b_scale[ob_n0:ob_n0 + PROJ_B_ACT_N_TILE], [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TBLK, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.range(O_GROUPS):
                p_g = partials[b_tb:b_tb + PROJ_B_ACT_T_TILE, g * D + ob_n0:g * D + ob_n0 + PROJ_B_ACT_N_TILE]
                g_scale = pl.reshape(act_scale_dq[g:g + 1, b_tb:b_tb + PROJ_B_ACT_T_TILE], [PROJ_B_ACT_T_TILE, 1])
                acc = pl.add(acc, pl.row_expand_mul(pl.cast(p_g, target_type=pl.FP32, mode='none'), g_scale))
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            attn_out[b_tb:b_tb + PROJ_B_ACT_T_TILE, ob_n0:ob_n0 + PROJ_B_ACT_N_TILE] = pl.cast(out_t, target_type=pl.BF16, mode='rint')
    return attn_out
