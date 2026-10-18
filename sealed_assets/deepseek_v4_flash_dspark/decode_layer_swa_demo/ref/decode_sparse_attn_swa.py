"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_swa_demo."""

from config import TP
from config import CASE_CONFIG as M
from config import DECODE_BATCH
from config import DECODE_SEQ
from config import BLOCK_SIZE
import pypto.language as pl

T_DYN = pl.dynamic('T_DYN')
ORI_BLOCK_NUM_DYN = pl.dynamic('ORI_BLOCK_NUM_DYN')
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
AIC_CORES = 24
AIV_CORES = 48
GATHER_TASKS = 32
GATHER_T = max(1, T // GATHER_TASKS)
QK_TASKS = AIC_CORES
MERGE_TASKS = AIV_CORES
GATHER_RUN = 16
H_TILE = 32
QK_M_TILE = 32
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
T_PAD = (T + 16 - 1) // 16 * 16
ROPE_CS_T_TILE = 8
SPARSE_BLOCKS = 1
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

@pl.jit.inline
def sparse_attn_swa(q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16], ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], sparse_bias: pl.Tensor[[T_DYN, PADDED_TOPK], pl.FP32], attn_sink: pl.Tensor[[H], pl.FP32], freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16]) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write SWA heads as ``[group, T_PAD, head-in-group, dim]`` slabs.

    Only the first runtime ``t_dim`` rows in each group are valid. The
    returned task ID covers every write to the packed output tensor.
    """
    t_dim = pl.tensor.dim(q, 0)
    t_heads = t_dim * H
    t_win = t_dim * WIN
    t_blk = t_dim * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
    t_hblocks = t_dim * (H // H_TILE)
    t_gather = (t_dim + GATHER_T - 1) // GATHER_T
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    swa_kv_flat = pl.create_tensor([t_win, HEAD_DIM], dtype=pl.BF16)
    gather_tids = pl.array.create(1, pl.TASK_ID)
    with pl.spmd(t_gather, name_hint='swa_gather_kv') as gather_tid:
        g_grp = pl.tile.get_block_idx()
        for g_tok in pl.unroll(GATHER_T):
            g_t = g_grp * GATHER_T + g_tok
            if g_t < t_dim:
                g_base = g_t * WIN
                for g_sub in pl.range(WIN // GATHER_RUN):
                    g_sr0 = g_sub * GATHER_RUN
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
    q_flat = pl.reshape(q, [t_heads, HEAD_DIM])
    sparse_blk_mi = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([t_blk, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(QK_TASKS, name_hint='qk_pv', deps=[gather_tids[0]], allow_early_resolve=True) as qk_tid:
        qk_task = pl.tile.get_block_idx()
        for qk_t in pl.range(qk_task, t_dim, QK_TASKS):
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
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
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
        cs_ones = pl.full([ROPE_CS_T_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
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
            for cs_rb in pl.range(rope_cs_blocks):
                cs_t0 = cs_rb * ROPE_CS_T_TILE
                cs_cos = pl.cast(freqs_cos[cs_t0:cs_t0 + ROPE_CS_T_TILE, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
                cs_sin = pl.cast(freqs_sin[cs_t0:cs_t0 + ROPE_CS_T_TILE, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
                cs_cos_dup = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
                cs_sin_dup = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
                cs_sin_signed = pl.mul(cs_sin_dup, cs_sign)
                rope_cos_il[cs_t0:cs_t0 + ROPE_CS_T_TILE, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_dup
                rope_sin_signed[cs_t0:cs_t0 + ROPE_CS_T_TILE, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = cs_sin_signed
    with pl.spmd(MERGE_TASKS, name_hint='merge_norm', deps=[qk_tid, rope_tid], allow_early_resolve=True) as merge_tid:
        m_task = pl.tile.get_block_idx()
        for m_idx in pl.range(m_task, t_hblocks, MERGE_TASKS):
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
                n_pack_row = (m_g0 + m_sg) * T_PAD + m_t
                n_dst_head = n_pack_row * HEADS_PER_GROUP
                o_packed_heads[n_dst_head:n_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM] = n_bf16[m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM]
                o_packed_heads[n_dst_head:n_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM] = n_rope_bf16[m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM]
    return (o_packed_heads, merge_tid)
