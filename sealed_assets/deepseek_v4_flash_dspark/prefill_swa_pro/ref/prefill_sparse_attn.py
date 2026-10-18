'NPU implementation for packed SWA prefill.'

import pypto.language as pl

from config import BLOCK_SIZE, CASE_CONFIG as M, FP32_NEG_INF, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_CMP_MAX_BLOCKS, PREFILL_SEQ

ORI_BLOCK_NUM_DYN = pl.dynamic('PREFILL_ORI_BLOCK_NUM_DYN')

CMP_BLOCK_NUM_DYN = pl.dynamic('PREFILL_CMP_BLOCK_NUM_DYN')

B = PREFILL_BATCH

S = PREFILL_SEQ

T = B * S

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

ROPE_HALF = ROPE_DIM // 2

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

CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS

HEAD_TILE = 16

QK_M_TILE = 32

GATHER_TOKEN_TILE = 2

BIAS_TOKEN_TILE = 16

QUANT_TOKEN_TILE = 8

ROPE_TILE = 16

ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE

A_K_TILE = 256

B_K_TILE = 256

QUANT_TILE = 512

PROJ_A_MM_N_TILE = 128

PROJ_B_MM_N_TILE = 256

PROJ_B_D_TILE = 1024

PROJ_B_ACT_N_TILE = 512

QUANT_CHUNKS = 4

QUANT_T_TILE = T // QUANT_CHUNKS

PROJ_B_ACT_T_TILE = 16

PROJ_B_ACT_TASK_T_TILE = 32

PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE

PB_DSLABS = D // PROJ_B_D_TILE

PREFILL_ATTN_TILE = 128

PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE

PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE

MASK_ALIGN_ELEMS = 32 // 4

VALID_BLOCK_MASK_COLS = (PREFILL_ATTN_BLOCKS + MASK_ALIGN_ELEMS - 1) // MASK_ALIGN_ELEMS * MASK_ALIGN_ELEMS

SPARSE_BIAS_COLS = min(TOPK, PREFILL_SPARSE_PAD)

SPARSE_CMP_BIAS_COLS = max(0, SPARSE_BIAS_COLS - WIN)

@pl.jit.inline
def sparse_attn_math(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    sparse_kv: pl.Tensor[[T * PREFILL_SPARSE_PAD, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[T, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[T, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run source-independent sparse QK/PV, merge, inverse RoPE, and projection."""

    # Block 0 stores all-masked fallback statistics.
    blk_rows = T * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
    sparse_blk_mi = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([blk_rows, HEAD_DIM], dtype=pl.FP32)
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    # Reshape sink bias for head-tile broadcast.
    attn_sink_col = pl.reshape(attn_sink, [H, 1])

    # Build head-invariant inverse-RoPE tables in disjoint column tiles.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.spmd(ROPE_HALF // ROPE_TILE, name_hint="rope_cs") as rope_cs_tid:
        cp = pl.tile.get_block_idx()
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0

        swap_ones = pl.full([HEAD_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        swap_ramp = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_ramp)
        swap_dup_i32 = pl.cast(pl.mul(swap_col, 0.5), target_type=pl.INT32, mode="trunc")
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))                                   # j%2
        swap_pair_f = pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0))                     # j^1
        swap_idx = pl.cast(swap_pair_f, target_type=pl.INT32)
        rope_swap_idx[:, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = swap_idx[:, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE]

        cs_ones = pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
        cs_ramp_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
        cs_ramp = pl.cast(cs_ramp_i32, target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_ramp)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                    # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                         # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                     # [+1,-1,...]
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_cos_il = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        cs_sin_il = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
        rope_cos_il[0:T, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_il
        rope_sin_signed[0:T, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(cs_sin_il, cs_sign)

    # Consume staged sparse sources for QK/PV.
    with pl.spmd(T, name_hint="qk_pv") as qk_tid:
        qk_t = pl.tile.get_block_idx()
        if qk_t < num_tokens:
            qk_kv_base = qk_t * PREFILL_SPARSE_PAD
            qk_token_base = qk_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
            for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                qk_s0 = qk_kv_base + qk_sb * PREFILL_ATTN_TILE
                qk_b0 = qk_sb * PREFILL_ATTN_TILE
                qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_b0:qk_b0 + PREFILL_ATTN_TILE]
                qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])
                if qk_sb == 0:
                    qk_block_valid = pl.cast(1, pl.INT32)
                if qk_block_valid > 0:
                    # Separate QK and PV cache views over the same rows.
                    qk_kv_k = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                    qk_kv_v = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                    for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                        qk_head_row = qk_t * H + qk_hb * QK_M_TILE
                        qk_q_tile = q_flat[qk_head_row:qk_head_row + QK_M_TILE, :]
                        qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                        qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                        qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                        qk_mi = pl.row_max(qk_scores)
                        qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                        qk_li = pl.row_sum(qk_exp)
                        qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                        qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                        for qk_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                            qk_h_idx = qk_hb * (QK_M_TILE // HEAD_TILE) + qk_sub
                            qk_r0 = qk_sub * HEAD_TILE
                            qk_blk_base = qk_token_base + qk_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
                            qk_row = qk_blk_base + qk_sb * HEAD_TILE
                            sparse_blk_mi[qk_row:qk_row + HEAD_TILE, :] = qk_mi[qk_r0:qk_r0 + HEAD_TILE, :]
                            sparse_blk_li[qk_row:qk_row + HEAD_TILE, :] = qk_li[qk_r0:qk_r0 + HEAD_TILE, :]
                            sparse_blk_oi[qk_row:qk_row + HEAD_TILE, :] = qk_oi[qk_r0:qk_r0 + HEAD_TILE, :]

    # Merge in FP32 and apply inverse RoPE before BF16 rounding.
    # o_packed aliases grouped-head storage in projection layout.
    o_packed_heads = pl.create_tensor([O_GROUPS * T * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T, O_GROUP_IN])
    # merge_tid orders proj_a after writes to o_packed.
    with pl.spmd(T, name_hint="merge_rope_pack", deps=[qk_tid, rope_cs_tid]) as merge_tid:
        m_t = pl.tile.get_block_idx()
        m_token_base = m_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
        # Load the aligned mask row; only set blocks have statistics.
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
                        m_oi_alpha = pl.row_expand_mul(m_oi, alpha)
                        cur_oi_beta = pl.row_expand_mul(cur_oi, beta)
                        m_oi = pl.add(m_oi_alpha, cur_oi_beta)
                        m_mi = mi_new
                sink_bias = attn_sink_col[m_h0:m_h0 + HEAD_TILE, :]
                sink_tile = pl.add(pl.sub(m_mi, m_mi), sink_bias)
                denom = pl.add(m_li, pl.exp(pl.sub(sink_tile, m_mi)))
                n_full = pl.row_expand_div(m_oi, denom)[0:HEAD_TILE, :]
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
            else:
                n_full = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                n_bf16 = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)

            m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
            m_swapped = pl.gather(m_rope, dim=-1, index=m_swap_idx)
            m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
            m_swapped_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
            m_rot = pl.add(m_rope_cos, m_swapped_sin)
            n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

            if HEAD_TILE % HEADS_PER_GROUP == 0:
                m_g0 = m_h0 // HEADS_PER_GROUP
                for m_sg in pl.unroll(HEAD_TILE // HEADS_PER_GROUP):
                    m_src_h0 = m_sg * HEADS_PER_GROUP
                    m_pack_row = (m_g0 + m_sg) * T + m_t
                    m_dst_head = m_pack_row * HEADS_PER_GROUP
                    # Slice bounds stay inline: hoisting the end offset loses the static dim,
                    # and naming the source materializes an extra tile.
                    o_packed_heads[
                        m_dst_head:m_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM
                    ] = n_bf16[
                        m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM
                    ]
                    o_packed_heads[
                        m_dst_head:m_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM
                    ] = n_rope_bf16[
                        m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM
                    ]
            else:
                # Store groups crossing a head tile one head at a time.
                for m_hi in pl.range(HEAD_TILE):
                    m_gh = m_h0 + m_hi
                    m_g = m_gh // HEADS_PER_GROUP
                    m_pack_row = m_g * T + m_t
                    m_col = (m_gh - m_g * HEADS_PER_GROUP) * HEAD_DIM
                    o_packed[m_pack_row:m_pack_row + 1, m_col:m_col + NOPE_DIM] = n_bf16[m_hi:m_hi + 1, 0:NOPE_DIM]
                    m_rope_row = n_rope_bf16[m_hi:m_hi + 1, 0:ROPE_DIM]
                    o_packed[m_pack_row:m_pack_row + 1, m_col + NOPE_DIM:m_col + HEAD_DIM] = m_rope_row

    # Chain merge, quantization, and projection through explicit task dependencies.
    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)
    partials = pl.create_tensor([T, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    quant_tids = pl.array.create(O_GROUPS * QUANT_CHUNKS, pl.TASK_ID)
    proj_b_tids = pl.array.create(PB_DSLABS * O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        # proj_a[g, nf]: BF16 grouped GEMM -> o_r[:, group g], peel-first-iter form.
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mm", deps=[merge_tid]) as pa_tid:
                    xa0_chunk = o_packed[row_base_o:row_base_o + T, 0:A_K_TILE]
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = o_packed[row_base_o:row_base_o + T, k0:k0 + A_K_TILE]
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, k0:k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    acc_a_2d = pl.reshape(acc_a, [T, PROJ_A_MM_N_TILE])
                    o_r[0:T, out_col_g + n0:out_col_g + n0 + PROJ_A_MM_N_TILE] = acc_a_2d
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid

        # quant[g, tc]: per-group amax + symmetric INT8 quant of o_r[:, group g] over a
        # token-chunk, storing the per-row group dequant scale into act_scale_dq[g, :].
        for tc in pl.parallel(QUANT_CHUNKS):
            t_base = tc * QUANT_T_TILE
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant",
                           deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)]) as q_tid:
                    for qt in pl.range(t_base, t_base + QUANT_T_TILE, QUANT_TOKEN_TILE):
                        g_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            oc_abs = pl.maximum(oc, pl.neg(oc))
                            oc_amax = pl.reshape(pl.row_max(oc_abs), [1, QUANT_TOKEN_TILE])
                            g_amax = pl.maximum(g_amax, oc_amax)
                        g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                        g_sq_row = pl.div(g_scale_num, g_amax)
                        act_scale_dq[g:g + 1, qt:qt + QUANT_TOKEN_TILE] = pl.recip(g_sq_row)
                        g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            oq_scaled = pl.row_expand_mul(oc, g_sq_col)
                            oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                            oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                            oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                            o_r_i8[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE] = oq_i8
                quant_tids[g * QUANT_CHUNKS + tc] = q_tid

        # proj_b_mm[dc, g]: INT8 GEMM of group g's contribution to a PROJ_B_D_TILE-wide slab
        # of D, written as INT32 partials[:, g*D+n]. Peel-first matmul: matmul_acc from a zero
        # carry trips TLOAD DN->NZ (pypto#1540).
        for dc in pl.parallel(PB_DSLABS):
            d0 = dc * PROJ_B_D_TILE
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mm",
                           deps=[quant_tids[g * QUANT_CHUNKS + tc] for tc in range(QUANT_CHUNKS)]) as pb_tid:
                    for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        b_act0 = o_r_i8[:, col_g:col_g + B_K_TILE]
                        b_weight0 = wo_b[n0:n0 + PROJ_B_MM_N_TILE, col_g:col_g + B_K_TILE]
                        acc_b = pl.matmul(b_act0, b_weight0, b_trans=True, out_dtype=pl.INT32)
                        for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                            k0 = col_g + kb * B_K_TILE
                            b_act = o_r_i8[:, k0:k0 + B_K_TILE]
                            b_weight = wo_b[n0:n0 + PROJ_B_MM_N_TILE, k0:k0 + B_K_TILE]
                            acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True)
                        partials[0:T, g * D + n0:g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
                proj_b_tids[dc * O_GROUPS + g] = pb_tid

    # Dequantize and sum per-group INT32 partials into the BF16 output.
    act_blocks = (D // PROJ_B_ACT_N_TILE) * (T // PROJ_B_ACT_TASK_T_TILE)
    with pl.spmd(act_blocks, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(PB_DSLABS * O_GROUPS)]) as act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // (T // PROJ_B_ACT_TASK_T_TILE)
        tblk = act_idx - nreg * (T // PROJ_B_ACT_TASK_T_TILE)
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0:ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.range(O_GROUPS):
                p_g = partials[b_tb:b_tb + PROJ_B_ACT_T_TILE, g * D + ob_n0:g * D + ob_n0 + PROJ_B_ACT_N_TILE]
                g_scale_row = act_scale_dq[g:g + 1, b_tb:b_tb + PROJ_B_ACT_T_TILE]
                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
                p_g_scaled = pl.row_expand_mul(p_g_f32, g_scale)
                acc = pl.add(acc, p_g_scaled)
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            attn_out[b_tb:b_tb + PROJ_B_ACT_T_TILE, ob_n0:ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16

    return attn_out

@pl.jit.inline
def sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[T, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Stage sparse sources and per-token bias, then run the sparse-attention math."""
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    cmp_cache_rows = cmp_block_num * BLOCK_SIZE
    ori_kv_flat = pl.reshape(ori_kv, [ori_cache_rows, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_cache_rows, HEAD_DIM])
    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.FP32)

    # Gather sparse sources in reverse token-tile order.
    with pl.spmd((T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE, name_hint="gather_ori_kv") as gather_ori_tid:
        gather_schedule_block = pl.tile.get_block_idx()
        gather_token_block = (T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE - 1 - gather_schedule_block
        gather_t0 = gather_token_block * GATHER_TOKEN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                if gather_t < num_tokens:
                    block_base = gather_t * PREFILL_SPARSE_PAD
                    stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    for gather_ki in pl.range(PREFILL_ATTN_TILE):
                        gather_raw = pl.read(swa_indices, [gather_t, gather_ki])
                        if gather_raw >= 0:
                            src = pl.cast(gather_raw, pl.INDEX)
                            stage[gather_ki:gather_ki + 1, :] = ori_kv_flat[src:src + 1, :]
                    sparse_kv[block_base:block_base + PREFILL_ATTN_TILE, :] = stage

    gather_cmp_blocks = ((T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE) * (PREFILL_ATTN_BLOCKS - 1)
    with pl.spmd(gather_cmp_blocks, name_hint="gather_cmp_kv") as gather_cmp_tid:
        gather_block = pl.tile.get_block_idx()
        gather_schedule_block = gather_block // (PREFILL_ATTN_BLOCKS - 1)
        gather_token_block = (T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE - 1 - gather_schedule_block
        gather_sb = gather_block - gather_schedule_block * (PREFILL_ATTN_BLOCKS - 1) + 1
        gather_t0 = gather_token_block * GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                if gather_t < num_tokens:
                    gather_block_valid = pl.read(valid_block_mask, [gather_t, gather_sb])
                    if gather_block_valid > 0:
                        block_base = gather_t * PREFILL_SPARSE_PAD + gather_k0
                        stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        for gather_ki in pl.range(PREFILL_ATTN_TILE):
                            gather_cmp_k = gather_k0 + gather_ki - WIN
                            if gather_cmp_k < IDX_TOPK:
                                gather_raw = pl.read(cmp_indices, [gather_t, gather_cmp_k])
                                if gather_raw >= 0:
                                    cmp_slot = gather_raw
                                    blk_slot = cmp_slot // BLOCK_SIZE
                                    blk = pl.cast(pl.read(cmp_block_table, [blk_slot]), pl.INDEX)
                                    src = blk * BLOCK_SIZE + (cmp_slot - blk_slot * BLOCK_SIZE)
                                    stage[gather_ki:gather_ki + 1, :] = cmp_kv_flat[src:src + 1, :]
                        sparse_kv[block_base:block_base + PREFILL_ATTN_TILE, :] = stage

    with pl.spmd(T // BIAS_TOKEN_TILE, name_hint="build_bias") as bias_tid:
        bias_blk = pl.tile.get_block_idx()
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_win_rows = swa_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN]
        bias_win_idx = pl.cast(bias_win_rows, target_type=pl.FP32)
        bias_win_raw_flag = pl.minimum(pl.maximum(pl.add(bias_win_idx, 1.0), 0.0), 1.0)
        bias_win = pl.mul(pl.sub(bias_win_raw_flag, 1.0), -FP32_NEG_INF)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN] = bias_win
        if SPARSE_CMP_BIAS_COLS > 0:
            bias_cmp_rows = cmp_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:SPARSE_CMP_BIAS_COLS]
            bias_cmp_idx = pl.cast(bias_cmp_rows, target_type=pl.FP32)
            bias_cmp_raw_flag = pl.minimum(pl.maximum(pl.add(bias_cmp_idx, 1.0), 0.0), 1.0)
            bias_cmp = pl.mul(pl.sub(bias_cmp_raw_flag, 1.0), -FP32_NEG_INF)
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, WIN:SPARSE_BIAS_COLS] = bias_cmp
        if PREFILL_SPARSE_PAD > SPARSE_BIAS_COLS:
            bias_pad_cols = PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS
            bias_pad = pl.full([BIAS_TOKEN_TILE, bias_pad_cols], dtype=pl.FP32, value=FP32_NEG_INF)
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, SPARSE_BIAS_COLS:PREFILL_SPARSE_PAD] = bias_pad

    return sparse_attn_math(
        q,
        sparse_kv, sparse_bias,
        valid_block_mask, attn_sink,
        freqs_cos, freqs_sin,
        wo_a, wo_b, wo_b_scale,
        attn_out, num_tokens,
    )
