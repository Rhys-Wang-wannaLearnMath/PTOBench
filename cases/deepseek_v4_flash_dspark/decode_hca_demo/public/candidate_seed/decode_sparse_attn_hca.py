"""Runtime-only HCA implementation closure."""
import pypto.language as pl
from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, DECODE_CMP_BLOCK_NUM, KV_CMP_MAX_BLOCKS, INT8_SCALE_MAX, INT8_AMAX_EPS
ORI_BLOCK_NUM_DYN = pl.dynamic('ORI_BLOCK_NUM_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('CMP_BLOCK_NUM_DYN')
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
from config import HCA_MAX_SEQ_LEN as MAX_SEQ_LEN
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
from config import COMPRESS_RATIO
from config import SPARSE_NEG_INF as NEG_INF
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
VALID_TOKEN_TILE = 1
GATHER_SEGS = 4
GATHER_RUN = 16
H_TILE = 16
QK_M_TILE = 32
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 256
PROJ_A_MM_N_TILE = 128
HCA_TOKEN_GROUP = 8
assert T % HCA_TOKEN_GROUP == 0
T_PAD = (HCA_TOKEN_GROUP + 16 - 1) // 16 * 16
MM_T_TILE = T_PAD
B_K_TILE = 256
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512
QUANT_TOKEN_TILE = 8
PROJ_B_D_TILE = 512
PROJ_B_ACT_T_TILE = 8
PROJ_B_ACT_TASK_T_TILE = 8
CMP_CAPACITY = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_TOPK = (CMP_CAPACITY + ATTN_K_TILE - 1) // ATTN_K_TILE * ATTN_K_TILE
MAX_SUPPORTED_SEQ = CMP_TOPK * COMPRESS_RATIO
CMP_BLOCKS_PER_REQ = (CMP_TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
TOPK = WIN + CMP_TOPK
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
GATHER_WIN_ROWS = WIN // GATHER_SEGS
GATHER_CMP_ROWS = (PADDED_TOPK - WIN) // GATHER_SEGS
assert CMP_BLOCKS_PER_REQ <= CMP_MAX_BLOCKS, f'compressed block table ({CMP_MAX_BLOCKS} blocks) must index the whole {CMP_TOPK}-slot tail; MAX_SUPPORTED_SEQ={MAX_SUPPORTED_SEQ}'
assert B * CMP_BLOCKS_PER_REQ <= CMP_BLOCK_NUM, f'compressed KV pool ({CMP_BLOCK_NUM} blocks) must hold B={B} requests x {CMP_BLOCKS_PER_REQ} blocks; MAX_SUPPORTED_SEQ={MAX_SUPPORTED_SEQ}'
assert WIN == ATTN_K_TILE, f'HCA window tile requires WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})'
assert BLOCK_SIZE % GATHER_RUN == 0, 'a contiguous run must not straddle two paged blocks by construction'

@pl.jit.inline
def _sparse_attn_hca_chunk(q: pl.Tensor[[HCA_TOKEN_GROUP, H, HEAD_DIM], pl.BF16], ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], window_swa_indices: pl.Tensor[[HCA_TOKEN_GROUP, WIN], pl.INT32], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32], cmp_sparse_indices: pl.Tensor[[HCA_TOKEN_GROUP, CMP_TOPK], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], freqs_cos: pl.Tensor[[HCA_TOKEN_GROUP, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[HCA_TOKEN_GROUP, ROPE_DIM], pl.BF16], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], attn_out: pl.Tensor[[HCA_TOKEN_GROUP, D], pl.BF16], token_start: pl.Scalar[pl.INDEX]):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([HCA_TOKEN_GROUP, PADDED_TOPK], dtype=pl.FP32)
    for v_blk in pl.spmd(HCA_TOKEN_GROUP // VALID_TOKEN_TILE, name_hint='build_valid', allow_early_resolve=True):
        v_t0 = v_blk * VALID_TOKEN_TILE
        v_win_f = pl.cast(window_swa_indices[v_t0:v_t0 + VALID_TOKEN_TILE, 0:WIN], target_type=pl.FP32)
        v_idx_f = pl.cast(cmp_sparse_indices[v_t0:v_t0 + VALID_TOKEN_TILE, 0:CMP_TOPK], target_type=pl.FP32)
        v_win_valid = pl.minimum(pl.maximum(pl.add(v_win_f, 1.0), 0.0), 1.0)
        v_cmp_valid = pl.minimum(pl.maximum(pl.add(v_idx_f, 1.0), 0.0), 1.0)
        sparse_bias[v_t0:v_t0 + VALID_TOKEN_TILE, 0:WIN] = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
        sparse_bias[v_t0:v_t0 + VALID_TOKEN_TILE, WIN:TOPK] = pl.mul(pl.sub(v_cmp_valid, 1.0), -NEG_INF)
        if PADDED_TOPK > TOPK:
            sparse_bias[v_t0:v_t0 + VALID_TOKEN_TILE, TOPK:PADDED_TOPK] = pl.full([VALID_TOKEN_TILE, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)
    hca_kv_flat = pl.create_tensor([HCA_TOKEN_GROUP * PADDED_TOPK, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(HCA_TOKEN_GROUP * GATHER_SEGS, name_hint='hca_gather_kv') as gather_tid:
        g_task = pl.tile.get_block_idx()
        g_t = g_task // GATHER_SEGS
        g_seg = g_task - g_t * GATHER_SEGS
        g_b = (token_start + g_t) // S
        g_row0 = g_t * PADDED_TOPK
        g_wk0 = g_seg * GATHER_WIN_ROWS
        for g_sub in pl.range(GATHER_WIN_ROWS // GATHER_RUN):
            g_sk0 = g_wk0 + g_sub * GATHER_RUN
            g_sdst = g_row0 + g_sk0
            g_first = pl.read(window_swa_indices, [g_t, g_sk0])
            g_last = pl.read(window_swa_indices, [g_t, g_sk0 + GATHER_RUN - 1])
            g_run_ok = g_last - g_first + pl.min(g_first, 0) * GATHER_RUN
            if g_run_ok == GATHER_RUN - 1:
                g_run_src = pl.cast(g_first, pl.INDEX)
                hca_kv_flat[g_sdst:g_sdst + GATHER_RUN, 0:HEAD_DIM] = ori_kv_flat[g_run_src:g_run_src + GATHER_RUN, 0:HEAD_DIM]
            else:
                for g_dr in pl.range(GATHER_RUN):
                    g_wdst = g_sdst + g_dr
                    g_win_slot_i32 = pl.read(window_swa_indices, [g_t, g_sk0 + g_dr])
                    if g_win_slot_i32 >= 0:
                        g_win_slot = pl.cast(g_win_slot_i32, pl.INDEX)
                        hca_kv_flat[g_wdst:g_wdst + 1, 0:HEAD_DIM] = ori_kv_flat[g_win_slot:g_win_slot + 1, 0:HEAD_DIM]
                    else:
                        hca_kv_flat[g_wdst:g_wdst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        g_ck0 = g_seg * GATHER_CMP_ROWS
        g_cdst0 = g_row0 + WIN + g_ck0
        for g_dr in pl.range(GATHER_CMP_ROWS):
            g_dst = g_cdst0 + g_dr
            g_cmp_k = g_ck0 + g_dr
            if g_cmp_k < CMP_TOPK:
                g_ridx = pl.read(cmp_sparse_indices, [g_t, g_cmp_k])
                if g_ridx >= 0:
                    g_cblk = pl.cast(pl.read(cmp_block_table, [g_b, g_ridx // BLOCK_SIZE]), pl.INDEX)
                    g_csrc = g_cblk * BLOCK_SIZE + g_ridx % BLOCK_SIZE
                    hca_kv_flat[g_dst:g_dst + 1, 0:HEAD_DIM] = cmp_kv_flat[g_csrc:g_csrc + 1, 0:HEAD_DIM]
                else:
                    hca_kv_flat[g_dst:g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
            else:
                hca_kv_flat[g_dst:g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    q_flat = pl.reshape(q, [HCA_TOKEN_GROUP * H, HEAD_DIM])
    o_packed = pl.create_tensor([O_GROUPS * HCA_TOKEN_GROUP, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([HCA_TOKEN_GROUP * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([HCA_TOKEN_GROUP * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([HCA_TOKEN_GROUP * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(HCA_TOKEN_GROUP * SPARSE_BLOCKS, name_hint='qk_pv', deps=[gather_tid], allow_early_resolve=True) as qk_tid:
        qk_item = pl.tile.get_block_idx()
        qk_t = qk_item // SPARSE_BLOCKS
        qk_sb = qk_item - qk_t * SPARSE_BLOCKS
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        qk_s0 = qk_sb * ATTN_K_TILE
        qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_s0:qk_s0 + ATTN_K_TILE]
        qk_base = qk_t * PADDED_TOPK + qk_s0
        qk_kv = hca_kv_flat[qk_base:qk_base + ATTN_K_TILE, 0:HEAD_DIM]
        for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
            qk_h0 = qk_hb * QK_M_TILE
            qk_head_row = qk_t * H + qk_h0
            qk_q_tile = q_flat[qk_head_row:qk_head_row + QK_M_TILE, 0:HEAD_DIM]
            qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
            qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
            qk_scores = pl.add(qk_scaled, pl.col_expand(pl.full([QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0), qk_bias_row))
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
    rope_cos_il = pl.create_tensor([HCA_TOKEN_GROUP, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([HCA_TOKEN_GROUP, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='rope_swap'):
        sw_col = pl.col_expand_mul(pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0), pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)
    for cp in pl.spmd(HALF_ROPE // ROPE_TILE, name_hint='rope_cs'):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(pl.full([HCA_TOKEN_GROUP, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0), pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        cs_cos = pl.cast(freqs_cos[0:HCA_TOKEN_GROUP, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:HCA_TOKEN_GROUP, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:HCA_TOKEN_GROUP, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:HCA_TOKEN_GROUP, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)
    with pl.spmd(HCA_TOKEN_GROUP * (H // H_TILE), name_hint='merge_norm') as merge_tid:
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
            n_pack_row = (m_h0 + n_hi) // HEADS_PER_GROUP * HCA_TOKEN_GROUP + m_t
            n_col = (m_h0 + n_hi) % HEADS_PER_GROUP * HEAD_DIM
            o_packed[n_pack_row:n_pack_row + 1, n_col:n_col + HEAD_DIM] = n_full_bf16[n_hi:n_hi + 1, :]
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, HCA_TOKEN_GROUP], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * HCA_TOKEN_GROUP
            out_col_g = g * O_LORA
            with pl.spmd(O_LORA // PROJ_A_MM_N_TILE, name_hint='proj_a_mm', deps=[merge_tid], allow_early_resolve=True) as pa_tid:
                nf = pl.tile.get_block_idx()
                n0 = nf * PROJ_A_MM_N_TILE
                xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, 0], valid_shape=[HCA_TOKEN_GROUP, A_K_TILE])
                wa0_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    xa_k_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, k0], valid_shape=[HCA_TOKEN_GROUP, A_K_TILE])
                    wa_k_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, k0:k0 + A_K_TILE]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                o_r_pad = pl.assemble(o_r_pad, acc_a, [0, out_col_g + n0])
            col_g = g * O_LORA
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='quant', deps=[pa_tid], allow_early_resolve=True) as q_tid:
                for qt in pl.pipeline(0, HCA_TOKEN_GROUP, QUANT_TOKEN_TILE, stage=2):
                    oc_amax = o_r_pad[qt:qt + QUANT_TOKEN_TILE, col_g:col_g + O_LORA]
                    g_abs = pl.abs(oc_amax)
                    g_row_max = pl.row_max(g_abs)
                    g_row_max = pl.reshape(g_row_max, [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    act_scale_dq[g:g + 1, qt:qt + QUANT_TOKEN_TILE] = pl.recip(g_sq_row)
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt:qt + QUANT_TOKEN_TILE, col_g:col_g + O_LORA]
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode='rint')
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode='round')
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode='trunc')
                    o_r_i8_pad[qt:qt + QUANT_TOKEN_TILE, col_g:col_g + O_LORA] = oq_i8
                    if T_PAD > HCA_TOKEN_GROUP:
                        zero_half = pl.full([T_PAD - HCA_TOKEN_GROUP, O_LORA], dtype=pl.FP16, value=0.0)
                        zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode='trunc')
                        o_r_i8_pad[HCA_TOKEN_GROUP:T_PAD, col_g:col_g + O_LORA] = zero_i8
            with pl.spmd(D // PROJ_B_D_TILE, name_hint='proj_b_mm', deps=[q_tid], allow_early_resolve=True) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_TILE
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.matmul(o_r_i8_pad[:, col_g:col_g + B_K_TILE], wo_b[n0:n0 + PROJ_B_MM_N_TILE, col_g:col_g + B_K_TILE], b_trans=True, out_dtype=pl.INT32)
                    for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                        k0 = col_g + kb * B_K_TILE
                        acc_b = pl.matmul_acc(acc_b, o_r_i8_pad[:, k0:k0 + B_K_TILE], wo_b[n0:n0 + PROJ_B_MM_N_TILE, k0:k0 + B_K_TILE], b_trans=True)
                    partials[0:MM_T_TILE, g * D + n0:g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
            proj_b_tids[g] = pb_tid
    with pl.spmd(D // PROJ_B_ACT_N_TILE * (HCA_TOKEN_GROUP // PROJ_B_ACT_TASK_T_TILE), name_hint='proj_b_act', deps=[proj_b_tids[i] for i in range(O_GROUPS)], allow_early_resolve=True) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // (HCA_TOKEN_GROUP // PROJ_B_ACT_TASK_T_TILE)
        tblk = act_idx - nreg * (HCA_TOKEN_GROUP // PROJ_B_ACT_TASK_T_TILE)
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0:ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
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

@pl.jit.inline
def sparse_attn_hca(q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16], ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], window_swa_indices: pl.Tensor[[T, WIN], pl.INT32], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32], cmp_sparse_indices: pl.Tensor[[T, CMP_TOPK], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], attn_out: pl.Tensor[[T, D], pl.BF16]):
    with pl.at(level=pl.Level.HOST, name_hint='hca_token_groups'):
        for token_start in pl.range(0, T, HCA_TOKEN_GROUP):
            q_chunk = pl.slice(q, [HCA_TOKEN_GROUP, H, HEAD_DIM], [token_start, 0, 0])
            window_swa_indices_chunk = pl.slice(window_swa_indices, [HCA_TOKEN_GROUP, WIN], [token_start, 0])
            cmp_sparse_indices_chunk = pl.slice(cmp_sparse_indices, [HCA_TOKEN_GROUP, CMP_TOPK], [token_start, 0])
            freqs_cos_chunk = pl.slice(freqs_cos, [HCA_TOKEN_GROUP, ROPE_DIM], [token_start, 0])
            freqs_sin_chunk = pl.slice(freqs_sin, [HCA_TOKEN_GROUP, ROPE_DIM], [token_start, 0])
            attn_out_chunk = pl.slice(attn_out, [HCA_TOKEN_GROUP, D], [token_start, 0])
            _sparse_attn_hca_chunk(q_chunk, ori_kv, window_swa_indices_chunk, cmp_kv, cmp_block_table, cmp_sparse_indices_chunk, attn_sink, freqs_cos_chunk, freqs_sin_chunk, wo_a, wo_b, wo_b_scale, attn_out_chunk, token_start)
