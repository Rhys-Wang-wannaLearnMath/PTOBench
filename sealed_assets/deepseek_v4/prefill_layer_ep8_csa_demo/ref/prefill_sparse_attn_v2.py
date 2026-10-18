from config import D, H, HEADS_PER_GROUP, HEAD_DIM, IDX_TOPK, O_GROUPS, O_GROUP_IN, O_LORA, SPARSE_TOPK as TOPK, WIN

import pypto.language as pl

from config import BLOCK_SIZE, CASE_CONFIG as M, FP32_NEG_INF, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_CMP_BLOCK_NUM, PREFILL_CMP_MAX_BLOCKS, PREFILL_ORI_MAX_BLOCKS, PREFILL_SEQ

B = PREFILL_BATCH

S = PREFILL_SEQ

T = B * S

ROPE_DIM = M.qk_rope_head_dim

HALF_ROPE = ROPE_DIM // 2

ROPE_HALF = HALF_ROPE

NOPE_DIM = M.nope_head_dim

SOFTMAX_SCALE = M.softmax_scale

PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))

PREFILL_SPARSE_TOPK = min(TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)

ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS

CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS

CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM

HEAD_TILE = 16                       # head-tile granularity for storage / merge

QK_M_TILE = 32                       # head rows cube-batched per QK/PV matmul

GATHER_TOKEN_TILE = 4

BIAS_TOKEN_TILE = 16

QUANT_TOKEN_TILE = 8

ROPE_OUT_TOK_TILE = T // 2

ROPE_TILE = 16

ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE

A_K_TILE = 256                       # proj_a cube K-frag: K*2B = 512B = one a2a3 L2 line (128 wastes half)

B_K_TILE = 256

QUANT_TILE = 512

PROJ_A_MM_N_TILE = 128                    # proj_a cube N frag (L0A/L0B wall); PA_NFRAGS per group

PROJ_B_MM_N_TILE = 256                    # proj_b_mm cube N frag (L0C wall)

PROJ_B_D_CHUNK = 1024                     # proj_b D-chunk per task (keeps proj_b at O_GROUPS*PB_DCHUNKS tasks)

PROJ_B_ACT_N_TILE = 512                   # proj_b_act vector N frag (UB wall)

PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE

PB_DCHUNKS = D // PROJ_B_D_CHUNK

NUM_QUANT_T_CHUNKS = 4                    # quant split per (group, token-chunk) over vector cores

QUANT_T_CHUNK = T // NUM_QUANT_T_CHUNKS

PROJ_B_ACT_T_TILE = 16                    # inner token tile for the O_GROUPS-way INT32->FP32 accumulate

PROJ_B_ACT_TBLK = 32                      # proj_b_act token block per task

PB_ACT_NREG = D // PROJ_B_ACT_N_TILE

PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK

PREFILL_ATTN_TILE = 128

PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE

PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE

SPARSE_BIAS_COLS = min(TOPK, PREFILL_SPARSE_PAD)
BIAS_WORK_COLS = max(8, ((SPARSE_BIAS_COLS + 7) // 8) * 8)  # Full logical sparse extent, 32-byte FP32 alignment.

@pl.jit.inline
def prefill_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Token-major sparse attention with current-suffix KV overlay: gather the
    sliding-window / overlay / compressed KV rows, then run causal sparse attention,
    inverse RoPE, and the grouped INT8 output projection -- all in one scope set,
    mirroring decode_sparse_attn.

    Raw index contract: -1 invalid; [0, WIN) ring KV; [WIN, WIN + T) overlay row;
    [WIN + T, ...) compressed slot. HCA/CSA use all three sources; SWA passes dummy
    compressed tensors with compressed raw indices unreachable. cmp_sparse_lens is the
    usable prefix per row (pass TOPK when the rows are already -1 padded, e.g. CSA)."""
    ori_kv_flat = pl.reshape(ori_kv, [ORI_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    for gather_block in pl.spmd(((T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE) * PREFILL_ATTN_BLOCKS, name_hint="gather_kv"):
        gather_token_block = gather_block // PREFILL_ATTN_BLOCKS
        gather_sb = gather_block - gather_token_block * PREFILL_ATTN_BLOCKS
        gather_t0 = gather_token_block * GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                block_base = gather_t * PREFILL_SPARSE_PAD + gather_k0
                stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                if gather_t < num_tokens:
                    gather_len = pl.read(cmp_sparse_lens, [gather_t])
                    gather_len_eff = pl.cast(0, pl.INT32)
                    if gather_len > 0:
                        gather_len_eff = gather_len
                    for gather_ki in pl.range(PREFILL_ATTN_TILE):
                        gather_k = gather_k0 + gather_ki
                        gather_raw = pl.cast(-1, pl.INT32)
                        if gather_k < TOPK:
                            if gather_k < gather_len_eff:
                                gather_raw = pl.read(cmp_sparse_indices, [gather_t, gather_k])
                        if gather_raw >= 0:
                            if gather_raw < WIN:
                                blk_slot = gather_raw // BLOCK_SIZE
                                blk = pl.cast(pl.read(ori_block_table, [blk_slot]), pl.INDEX)
                                src = blk * BLOCK_SIZE + (gather_raw - blk_slot * BLOCK_SIZE)
                                stage[gather_ki:gather_ki + 1, :] = ori_kv_flat[src:src + 1, :]
                            elif gather_raw < WIN + T:
                                ov = pl.cast(gather_raw - WIN, pl.INDEX)
                                stage[gather_ki:gather_ki + 1, :] = kv_overlay[ov:ov + 1, :]
                            else:
                                cmp_slot = gather_raw - (WIN + T)
                                blk_slot = cmp_slot // BLOCK_SIZE
                                blk = pl.cast(pl.read(cmp_block_table, [blk_slot]), pl.INDEX)
                                src = blk * BLOCK_SIZE + (cmp_slot - blk_slot * BLOCK_SIZE)
                                stage[gather_ki:gather_ki + 1, :] = cmp_kv_flat[src:src + 1, :]
                sparse_kv[block_base:block_base + PREFILL_ATTN_TILE, :] = stage

    cmp_sparse_lens_2d = pl.reshape(cmp_sparse_lens, [T, 1])
    sparse_bias = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    for bias_blk in pl.spmd(T // BIAS_TOKEN_TILE, name_hint="build_bias"):
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_idx = pl.full([BIAS_TOKEN_TILE, BIAS_WORK_COLS], dtype=pl.FP32, value=-1.0)
        for br in pl.range(BIAS_TOKEN_TILE):
            for bc in pl.range(SPARSE_BIAS_COLS):
                pl.write(bias_idx, [br, bc], pl.cast(pl.read(cmp_sparse_indices, [bias_t0 + br, bc]), pl.FP32))
        bias_raw_flag = pl.minimum(pl.maximum(pl.add(bias_idx, 1.0), 0.0), 1.0)
        bias_col = pl.col_expand(
            pl.full([BIAS_TOKEN_TILE, BIAS_WORK_COLS], dtype=pl.FP32, value=0.0),
            pl.cast(pl.arange(0, [1, BIAS_WORK_COLS], dtype=pl.INT32), target_type=pl.FP32))
        bias_len = pl.full([BIAS_TOKEN_TILE, BIAS_WORK_COLS], dtype=pl.FP32, value=0.0)
        for br in pl.range(BIAS_TOKEN_TILE):
            lv = pl.cast(pl.read(cmp_sparse_lens, [bias_t0 + br]), pl.FP32)
            for bc in pl.range(BIAS_WORK_COLS):
                pl.write(bias_len, [br, bc], lv)
        bias_len_flag = pl.minimum(pl.maximum(pl.neg(pl.sub(bias_col, bias_len)), 0.0), 1.0)
        bias_valid = pl.mul(bias_raw_flag, bias_len_flag)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:BIAS_WORK_COLS] = pl.mul(pl.sub(bias_valid, 1.0), -FP32_NEG_INF)
        if PREFILL_SPARSE_PAD > BIAS_WORK_COLS:
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, BIAS_WORK_COLS:PREFILL_SPARSE_PAD] = pl.full(
                [BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD - BIAS_WORK_COLS], dtype=pl.FP32, value=FP32_NEG_INF)

    blk_rows = T * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
    sparse_blk_mi = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([blk_rows, HEAD_DIM], dtype=pl.FP32)
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    for qk_t in pl.spmd(T, name_hint="qk_pv"):
        if qk_t < num_tokens:
            qk_kv_base = qk_t * PREFILL_SPARSE_PAD
            qk_token_base = qk_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
            for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                qk_s0 = qk_kv_base + qk_sb * PREFILL_ATTN_TILE
                qk_kv_k = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                qk_kv_v = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_sb * PREFILL_ATTN_TILE:qk_sb * PREFILL_ATTN_TILE + PREFILL_ATTN_TILE]
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
                        qk_row = qk_token_base + qk_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE + qk_sb * HEAD_TILE
                        sparse_blk_mi[qk_row:qk_row + HEAD_TILE, :] = qk_mi[qk_r0:qk_r0 + HEAD_TILE, :]
                        sparse_blk_li[qk_row:qk_row + HEAD_TILE, :] = qk_li[qk_r0:qk_r0 + HEAD_TILE, :]
                        sparse_blk_oi[qk_row:qk_row + HEAD_TILE, :] = qk_oi[qk_r0:qk_r0 + HEAD_TILE, :]

    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    with pl.spmd(T, name_hint="merge_norm") as merge_tid:
        m_t = pl.tile.get_block_idx()
        m_token_base = m_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
        for m_h_idx in pl.range(H // HEAD_TILE):
            m_h0 = m_h_idx * HEAD_TILE
            m_rope_row = m_t * H + m_h0
            if m_t < num_tokens:
                m_blk_base = m_token_base + m_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
                m_mi = sparse_blk_mi[m_blk_base:m_blk_base + HEAD_TILE, :]
                m_li = sparse_blk_li[m_blk_base:m_blk_base + HEAD_TILE, :]
                m_oi = sparse_blk_oi[m_blk_base:m_blk_base + HEAD_TILE, :]
                for m_sb in pl.range(1, PREFILL_ATTN_BLOCKS):
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
                sink_bias = pl.reshape(attn_sink[m_h0:m_h0 + HEAD_TILE], [HEAD_TILE, 1])
                sink_tile = pl.add(pl.sub(m_mi, m_mi), sink_bias)
                denom = pl.add(m_li, pl.exp(pl.sub(sink_tile, m_mi)))
                n_full = pl.row_expand_div(m_oi, denom)[0:HEAD_TILE, :]
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
            else:
                n_full = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                n_bf16 = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)

            attn_rope_stage[m_rope_row:m_rope_row + HEAD_TILE, :] = n_full[0:HEAD_TILE, NOPE_DIM:HEAD_DIM]
            for n_hi in pl.range(HEAD_TILE):
                gh = m_h0 + n_hi
                g = gh // HEADS_PER_GROUP
                pack_row = g * T + m_t
                col = (gh - g * HEADS_PER_GROUP) * HEAD_DIM
                o_packed[pack_row:pack_row + 1, col:col + NOPE_DIM] = n_bf16[n_hi:n_hi + 1, 0:NOPE_DIM]

    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    for cp in pl.spmd(ROPE_HALF // ROPE_TILE, name_hint="rope_cs"):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...]
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(
            pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

    attn_rope_stage_3d = pl.reshape(attn_rope_stage, [T, H, ROPE_DIM])
    with pl.spmd((H // 4) * (T // ROPE_OUT_TOK_TILE), name_hint="rope") as rope_tid:
        rp_idx = pl.tile.get_block_idx()
        rp_hg = rp_idx // (T // ROPE_OUT_TOK_TILE)
        rp_tt = rp_idx - rp_hg * (T // ROPE_OUT_TOK_TILE)
        rp_t0 = rp_tt * ROPE_OUT_TOK_TILE
        sp_col = pl.col_expand_mul(
            pl.full([ROPE_OUT_TOK_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        sp_dup_f = pl.cast(pl.cast(pl.mul(sp_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sp_lane = pl.sub(sp_col, pl.mul(sp_dup_f, 2.0))                                           # j%2
        sp_swap_idx = pl.cast(pl.sub(pl.add(sp_col, 1.0), pl.mul(sp_lane, 2.0)), target_type=pl.INT32)  # j^1
        for rp_hl in pl.range(0, 4):
            rp_gh = rp_hg * 4 + rp_hl
            rp_g = rp_gh // HEADS_PER_GROUP
            rp_hh = rp_gh - rp_g * HEADS_PER_GROUP
            rp_col = rp_hh * HEAD_DIM + NOPE_DIM
            rp_o0 = rp_g * T + rp_t0
            for r_r0 in pl.range(0, ROPE_HALF, ROPE_TILE):
                c0 = 2 * r_r0
                r_tile_fp32 = pl.reshape(
                    attn_rope_stage_3d[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, rp_gh : rp_gh + 1, c0 : c0 + ROPE_INTERLEAVE_TILE],
                    [ROPE_OUT_TOK_TILE, ROPE_INTERLEAVE_TILE])
                r_cos_il = rope_cos_il[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_sin_signed = rope_sin_signed[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_swapped = pl.gather(r_tile_fp32, dim=-1, index=sp_swap_idx)
                r_rot = pl.add(pl.mul(r_tile_fp32, r_cos_il), pl.mul(r_swapped, r_sin_signed))
                r_rot = pl.cast(r_rot, target_type=pl.BF16, mode="rint")
                o_packed[rp_o0 : rp_o0 + ROPE_OUT_TOK_TILE, rp_col + c0 : rp_col + c0 + ROPE_INTERLEAVE_TILE] = r_rot

    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)   # [G, T]: each group's
    partials = pl.create_tensor([T, O_GROUPS * D], dtype=pl.INT32)  # per-group INT32 partials
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    quant_tids = pl.array.create(O_GROUPS * NUM_QUANT_T_CHUNKS, pl.TASK_ID)
    proj_b_tids = pl.array.create(PB_DCHUNKS * O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mm", deps=[merge_tid, rope_tid]) as pa_tid:
                    xa0_chunk = o_packed[row_base_o : row_base_o + T, 0:A_K_TILE]
                    wa0_chunk = wo_a[g, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = o_packed[row_base_o : row_base_o + T, k0 : k0 + A_K_TILE]
                        wa_k_chunk = wo_a[g, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    o_r = pl.assemble(o_r, pl.reshape(acc_a, [T, PROJ_A_MM_N_TILE]), [0, out_col_g + n0])
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid

        for tc in pl.parallel(NUM_QUANT_T_CHUNKS):
            t_base = tc * QUANT_T_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant",
                           deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)]) as q_tid:
                    for qt in pl.range(t_base, t_base + QUANT_T_CHUNK, QUANT_TOKEN_TILE):
                        g_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            g_amax = pl.maximum(g_amax, pl.reshape(pl.row_max(pl.maximum(oc, pl.neg(oc))), [1, QUANT_TOKEN_TILE]))
                        g_sq_row = pl.mul(pl.recip(g_amax), INT8_SCALE_MAX)
                        act_scale_dq = pl.assemble(act_scale_dq, pl.recip(g_sq_row), [g, qt])
                        g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            oq_i32 = pl.cast(pl.row_expand_mul(oc, g_sq_col), target_type=pl.INT32, mode="rint")
                            oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                            o_r_i8 = pl.assemble(o_r_i8, pl.cast(oq_half, target_type=pl.INT8, mode="trunc"), [qt, col_g + k1])
                quant_tids[g * NUM_QUANT_T_CHUNKS + tc] = q_tid

        for dc in pl.parallel(PB_DCHUNKS):
            d0 = dc * PROJ_B_D_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mm",
                           deps=[quant_tids[g * NUM_QUANT_T_CHUNKS + tc] for tc in range(NUM_QUANT_T_CHUNKS)]) as pb_tid:
                    for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        acc_b = pl.matmul(o_r_i8[:, col_g:col_g + B_K_TILE],
                                          wo_b[n0:n0 + PROJ_B_MM_N_TILE, col_g:col_g + B_K_TILE],
                                          b_trans=True, out_dtype=pl.INT32)
                        for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                            k0 = col_g + kb * B_K_TILE
                            acc_b = pl.matmul_acc(acc_b, o_r_i8[:, k0:k0 + B_K_TILE],
                                                  wo_b[n0:n0 + PROJ_B_MM_N_TILE, k0:k0 + B_K_TILE], b_trans=True)
                        partials = pl.assemble(partials, acc_b, [0, g * D + n0])
                proj_b_tids[dc * O_GROUPS + g] = pb_tid

    with pl.spmd(PB_ACT_NREG * PB_ACT_TBLKS, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(PB_DCHUNKS * O_GROUPS)]) as act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // PB_ACT_TBLKS
        tblk = act_idx - nreg * PB_ACT_TBLKS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TBLK
        wb_scale_chunk = pl.reshape(wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE], [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TBLK, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.range(O_GROUPS):
                p_g = partials[b_tb:b_tb + PROJ_B_ACT_T_TILE, g * D + ob_n0:g * D + ob_n0 + PROJ_B_ACT_N_TILE]
                g_scale = pl.reshape(act_scale_dq[g:g + 1, b_tb:b_tb + PROJ_B_ACT_T_TILE], [PROJ_B_ACT_T_TILE, 1])
                acc = pl.add(acc, pl.row_expand_mul(pl.cast(p_g, target_type=pl.FP32, mode="none"), g_scale))
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            attn_out[b_tb:b_tb + PROJ_B_ACT_T_TILE, ob_n0:ob_n0 + PROJ_B_ACT_N_TILE] = pl.cast(out_t, target_type=pl.BF16, mode="rint")

    return attn_out
