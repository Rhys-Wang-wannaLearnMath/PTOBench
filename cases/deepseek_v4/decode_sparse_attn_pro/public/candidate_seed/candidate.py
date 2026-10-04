# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped output projection (decode)."""


import pypto.language as pl

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, INT8_SCALE_MAX, INT8_AMAX_EPS


# model config
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
IDX_TOPK = M.index_topk
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# kernel-local
ORI_MAX_BLOCKS = 1  # paged-KV pool: ori (sliding-window) blocks per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 64  # paged-KV pool: compressed blocks per batch
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# tiling
ROPE_TOKEN_TILE = 1
ROPE_PACK_TOKEN_TILE = 8   # decode T=8; avoid zero-core T//32 rope_pack dispatch
H_TILE = 16
ATTN_K_TILE = 32
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_T_TILE = 16
MM_T_TILE = 16          # cube matmul rows must be 16-aligned on current ptoas
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
A_K_TILE = 128
A_N_TILE = 128
B_T_TILE = 16
B_K_TILE = 128
B_N_TILE = 128
QUANT_TILE = 32
QUANT_TOKEN_TILE = 8
QUANT_K_TILE = O_GROUPS * O_LORA // 2

@pl.jit.inline
def sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    # Gather the sliding-window + compressed-cache rows into a per-token packed KV list.
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_kv = pl.create_tensor([T * TOPK, HEAD_DIM], dtype=pl.BF16)
    for g_t in pl.spmd(T, name_hint="gather_kv"):
        g_b = g_t // S
        g_s = g_t - g_b * S
        g_seq_end = pl.read(seqused_kv, [g_b])
        g_seq_len = g_seq_end - S + 1 + g_s
        g_win_v = pl.min(WIN, g_seq_len)
        g_cmp_v = g_seq_len - g_win_v
        g_tk_v = pl.min(IDX_TOPK, g_cmp_v)
        g_sparse_k = g_win_v + g_tk_v
        g_kv_base = g_t * TOPK

        # Window prefix: contiguous, copy as one row block.
        g_ori_blk = pl.cast(pl.read(ori_block_table, [g_b, 0]), pl.INDEX)
        g_ori_row = g_ori_blk * BLOCK_SIZE
        window_rows = pl.set_validshape(ori_kv_flat[g_ori_row : g_ori_row + WIN, 0 : HEAD_DIM], g_win_v, HEAD_DIM)
        sparse_kv[g_kv_base : g_kv_base + WIN, 0 : HEAD_DIM] = window_rows

        # Compressed-cache hits after the window prefix (sparse row-gather).
        for g_kk in pl.range(g_tk_v):
            g_raw = pl.read(cmp_sparse_indices, [g_t, g_win_v + g_kk])
            g_slot = g_raw - WIN
            g_blk = pl.cast(pl.read(cmp_block_table, [g_b, g_slot // BLOCK_SIZE]), pl.INDEX)
            g_src_row = g_blk * BLOCK_SIZE + g_slot % BLOCK_SIZE
            g_dst_row = g_kv_base + g_win_v + g_kk
            sparse_kv[g_dst_row : g_dst_row + 1, 0 : HEAD_DIM] = cmp_kv_flat[g_src_row : g_src_row + 1, 0 : HEAD_DIM]

        # Zero-pad the tail so ratio-0/128 sanity modes stay deterministic.
        zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for g_pad_kk in pl.range(g_sparse_k, TOPK):
            g_pad_row = g_kv_base + g_pad_kk
            sparse_kv[g_pad_row : g_pad_row + 1, 0 : HEAD_DIM] = zero_kv_row

    # Sparse-K attention: qk_pv writes per-tile (mi, li, oi) into GM scratch,
    # merge_norm reads them back. ATTN_K_TILE keeps K and V right-buffer
    # copies together under the 64KB L1B limit.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.BF16)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE, HEAD_DIM], dtype=pl.FP32)

    for qk_t in pl.spmd(T, name_hint="qk_pv"):
        qk_b = qk_t // S
        qk_s = qk_t - qk_b * S
        qk_seq_end = pl.read(seqused_kv, [qk_b])
        qk_seq_len = qk_seq_end - S + 1 + qk_s
        qk_win_v = pl.min(WIN, qk_seq_len)
        qk_tk_v = pl.min(IDX_TOPK, qk_seq_len - qk_win_v)
        qk_sparse_k = qk_win_v + qk_tk_v
        qk_kv_base = qk_t * TOPK
        qk_token_base = qk_t * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE
        for qk_h_idx in pl.range((H // H_TILE)):
            qk_h0 = qk_h_idx * H_TILE
            qk_head_row = qk_t * H + qk_h0
            qk_q_tile = q_flat[qk_head_row : qk_head_row + H_TILE, 0 : HEAD_DIM]
            qk_blk_base = qk_token_base + qk_h_idx * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE

            for qk_sb in pl.range(((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)):
                qk_s0 = qk_sb * ATTN_K_TILE
                if qk_s0 < qk_sparse_k:
                    qk_s_v = pl.min(ATTN_K_TILE, qk_sparse_k - qk_s0)
                    qk_kv_k = sparse_kv[qk_kv_base + qk_s0 : qk_kv_base + qk_s0 + ATTN_K_TILE, 0 : HEAD_DIM]
                    qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                    qk_scores_v = pl.set_validshape(pl.mul(qk_raw, SOFTMAX_SCALE), H_TILE, qk_s_v)
                    qk_scores = pl.fillpad(qk_scores_v, pad_value=pl.PadValue.min)
                    qk_mi = pl.row_max(qk_scores)
                    qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                    qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16)
                    qk_li = pl.row_sum(pl.cast(qk_exp_bf16, target_type=pl.FP32))
                    qk_kv_v = sparse_kv[qk_kv_base + qk_s0 : qk_kv_base + qk_s0 + ATTN_K_TILE, 0 : HEAD_DIM]
                    qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                    qk_row = qk_blk_base + qk_sb * H_TILE
                    sparse_blk_mi[qk_row : qk_row + H_TILE, 0 : 1] = qk_mi
                    sparse_blk_li[qk_row : qk_row + H_TILE, 0 : 1] = qk_li
                    sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi

    # Online-softmax merge across sparse-K tiles, then sink-norm.
    for m_t in pl.spmd(T, name_hint="merge_norm"):
        m_b = m_t // S
        m_s = m_t - m_b * S
        m_seq_end = pl.read(seqused_kv, [m_b])
        m_seq_len = m_seq_end - S + 1 + m_s
        m_win_v = pl.min(WIN, m_seq_len)
        m_tk_v = pl.min(IDX_TOPK, m_seq_len - m_win_v)
        m_sparse_k = m_win_v + m_tk_v
        m_token_base = m_t * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE

        for m_h_idx in pl.range((H // H_TILE)):
            m_h0 = m_h_idx * H_TILE
            m_blk_base = m_token_base + m_h_idx * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE
            m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

            for m_sb in pl.range(1, ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)):
                m_s0 = m_sb * ATTN_K_TILE
                if m_s0 < m_sparse_k:
                    m_row = m_blk_base + m_sb * H_TILE
                    m_cur_mi = sparse_blk_mi[m_row : m_row + H_TILE, 0 : 1]
                    m_cur_li = sparse_blk_li[m_row : m_row + H_TILE, 0 : 1]
                    m_cur_oi = sparse_blk_oi[m_row : m_row + H_TILE, 0 : HEAD_DIM]
                    m_mi_new = pl.maximum(m_mi, m_cur_mi)
                    m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
                    m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
                    m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
                    m_oi = pl.add(pl.row_expand_mul(m_oi, m_alpha), pl.row_expand_mul(m_cur_oi, m_beta))
                    m_mi = m_mi_new

            n_sink_bias = pl.reshape(attn_sink[m_h0 : m_h0 + H_TILE], [H_TILE, 1])
            n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
            n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
            n_out = pl.cast(pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM], target_type=pl.BF16)
            n_rope_row = m_t * H + m_h0
            attn_rope_stage[n_rope_row : n_rope_row + H_TILE, 0 : ROPE_DIM] = n_out[0 : H_TILE, NOPE_DIM : HEAD_DIM]

            for n_hi in pl.range(H_TILE):
                n_gh = m_h0 + n_hi
                n_g = n_gh // HEADS_PER_GROUP
                n_hh = n_gh - n_g * HEADS_PER_GROUP
                n_pack_row = n_g * T + m_t
                n_col = n_hh * HEAD_DIM
                o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + NOPE_DIM] = n_out[n_hi : n_hi + 1, 0 : NOPE_DIM]

    # Inverse RoPE: gather even/odd lanes (HW native TGATHER<mask>) for de-interleave,
    # rotate with cos/sin, then mask-form scatter (TSCATTER<mask>) re-interleaves back
    # to interleaved layout — the exact inverse of the gathers, mirroring
    # decode_qkv_proj_rope's gather-rotate-scatter. An earlier *index-form* scatter
    # (hand-built even/odd index tiles) caused ~11% attn_out precision FAIL; mask-form
    # drops those index tiles and the intermediate BF16 round-trip entirely.
    # Note: mask-form scatter is A3/sim only — A5 rejects it.
    rope_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    for r_idx in pl.spmd(T // ROPE_TOKEN_TILE, name_hint="rope"):
        r_t0 = r_idx * ROPE_TOKEN_TILE
        for r_dt in pl.range(ROPE_TOKEN_TILE):
            r_t = r_t0 + r_dt
            r_row = r_t * H

            for r_r0 in pl.range(0, HALF_ROPE, ROPE_TILE):
                # gather_mask requires FP/INT (not BF16): cast BF16 tile to FP32 first.
                r_tile_fp32 = pl.cast(attn_rope_stage[r_row : r_row + H, 2 * r_r0 : 2 * r_r0 + ROPE_INTERLEAVE_TILE], target_type=pl.FP32)
                r_even = pl.tensor.gather(r_tile_fp32, mask_pattern=pl.tile.MaskPattern.P0101)
                r_odd = pl.tensor.gather(r_tile_fp32, mask_pattern=pl.tile.MaskPattern.P1010)

                r_cos = pl.cast(freqs_cos[r_t : r_t + 1, r_r0 : r_r0 + ROPE_TILE], target_type=pl.FP32)
                r_sin = pl.cast(freqs_sin[r_t : r_t + 1, r_r0 : r_r0 + ROPE_TILE], target_type=pl.FP32)
                r_even_rot = pl.add(pl.col_expand_mul(r_even, r_cos), pl.col_expand_mul(r_odd, r_sin))
                r_odd_rot = pl.sub(pl.col_expand_mul(r_odd, r_cos), pl.col_expand_mul(r_even, r_sin))
                # Match golden's BF16 round-trip on the rotated values (golden does
                # `.to(bfloat16).float()` after the rotation, which the original NPU
                # matmul reassemble matched by casting BF16 before the matmul).
                # Without this, FP32-only output exceeds the precision tolerance.
                r_even_rot = pl.cast(pl.cast(r_even_rot, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
                r_odd_rot = pl.cast(pl.cast(r_odd_rot, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)

                # Re-interleave via mask-form scatter: write the FP32 rotated even/odd
                # halves into the interleaved columns selected by the P0101/P1010 mask
                # (inverse of the gathers above). One buffer, no add, no BF16 round-trip.
                r_rope_buf = pl.full([H, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=0.0)
                r_rope_buf = pl.tensor.scatter(r_even_rot, mask_pattern=pl.tile.MaskPattern.P0101, dst=r_rope_buf)
                r_rope_buf = pl.tensor.scatter(r_odd_rot, mask_pattern=pl.tile.MaskPattern.P1010, dst=r_rope_buf)
                rope_buf[r_row : r_row + H, 2 * r_r0 : 2 * r_r0 + ROPE_INTERLEAVE_TILE] = r_rope_buf

    for rp_block in pl.spmd((T // ROPE_PACK_TOKEN_TILE) * O_GROUPS, name_hint="rope_pack"):
        rp_tb = rp_block // O_GROUPS
        rp_g = rp_block - rp_tb * O_GROUPS
        rp_t0 = rp_tb * ROPE_PACK_TOKEN_TILE

        for rp_dt in pl.range(ROPE_PACK_TOKEN_TILE):
            rp_t = rp_t0 + rp_dt
            rp_row = rp_t * H + rp_g * HEADS_PER_GROUP

            # Write only this group's inverse-RoPE tail of o_packed: cast the single
            # scatter-merged FP32 buffer to BF16.
            rp_rope = rope_buf[rp_row : rp_row + HEADS_PER_GROUP, 0 : ROPE_DIM]
            rp_full = pl.cast(rp_rope, target_type=pl.BF16)
            rp_pack_row = rp_g * T + rp_t
            for rp_hh in pl.range(HEADS_PER_GROUP):
                rp_col = rp_hh * HEAD_DIM + NOPE_DIM
                o_packed[rp_pack_row : rp_pack_row + 1, rp_col : rp_col + ROPE_DIM] = rp_full[rp_hh : rp_hh + 1, 0 : ROPE_DIM]

    # Grouped BF16 projection `o_packed @ wo_a^T` -> `o_r`. Vec post-process
    # (BF16 store + per-row partial amax) is T-tiled inside the scope as a
    # pypto#1472 workaround — without it the fused proj_a AIV side oversizes
    # UB and AllocateMemoryAddr rejects the kernel.
    o_r = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_amax_parts = pl.create_tensor([O_GROUPS * (O_LORA // A_N_TILE), T_PAD], dtype=pl.FP32)
    for proj_a_block in pl.spmd(O_GROUPS * (O_LORA // A_N_TILE), name_hint="proj_a"):
        # K-split BF16 matmul for one wo_a output tile. Stays in
        # peel-first-iter form: the `pl.create_tensor` + `if k0 == 0`
        # carry hits pypto#1540 on the 3D wo_a slice.
        g = proj_a_block // (O_LORA // A_N_TILE)
        nb = proj_a_block - g * (O_LORA // A_N_TILE)
        row_base_o = g * T
        out_col_g = g * O_LORA
        n0 = nb * A_N_TILE
        amax_part_row = g * (O_LORA // A_N_TILE) + nb

        xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, 0], valid_shape=[T, A_K_TILE])
        wa0_chunk = wo_a[g:g + 1, n0:n0 + A_N_TILE, 0:A_K_TILE]
        acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, k0], valid_shape=[T, A_K_TILE])
            wa_k_chunk = wo_a[g:g + 1, n0:n0 + A_N_TILE, k0:k0 + A_K_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

        acc_a_2d = pl.reshape(acc_a, [MM_T_TILE, A_N_TILE])
        for tb in pl.range(0, T, A_T_TILE):
            acc_t = acc_a_2d[tb:tb + A_T_TILE, 0:A_N_TILE]
            acc_t_bf16 = pl.cast(acc_t, target_type=pl.BF16)
            o_r[tb:tb + A_T_TILE, out_col_g + n0:out_col_g + n0 + A_N_TILE] = acc_t_bf16
            acc_t_f32 = pl.cast(acc_t_bf16, target_type=pl.FP32)
            acc_t_abs = pl.maximum(acc_t_f32, pl.neg(acc_t_f32))
            acc_t_amax = pl.reshape(pl.row_max(acc_t_abs), [1, A_T_TILE])
            o_r_amax_parts[amax_part_row:amax_part_row + 1, tb:tb + A_T_TILE] = acc_t_amax

    # Per-row symmetric INT8 quant of `o_r`, K-tiled as a second parallel axis.
    o_r_i8 = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    o_r_scale_dq = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    for q_block in pl.spmd((T // QUANT_TOKEN_TILE) * ((O_GROUPS * O_LORA) // QUANT_K_TILE), name_hint="quant"):
        qt_idx = q_block // ((O_GROUPS * O_LORA) // QUANT_K_TILE)
        qk_idx = q_block - qt_idx * ((O_GROUPS * O_LORA) // QUANT_K_TILE)
        quant_t0 = qt_idx * QUANT_TOKEN_TILE
        k0 = qk_idx * QUANT_K_TILE

        or_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for ab in pl.range(0, O_GROUPS * (O_LORA // A_N_TILE), 1):
            or_a_part = o_r_amax_parts[ab:ab + 1, quant_t0:quant_t0 + QUANT_TOKEN_TILE]
            or_amax = pl.maximum(or_amax, or_a_part)
        or_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
        or_scale_dq = pl.reshape(pl.recip(or_sq_row), [QUANT_TOKEN_TILE, 1])
        o_r_scale_dq[quant_t0:quant_t0 + QUANT_TOKEN_TILE, 0:1] = or_scale_dq
        or_sq_col = pl.reshape(or_sq_row, [QUANT_TOKEN_TILE, 1])
        for k1 in pl.range(k0, k0 + QUANT_K_TILE, QUANT_TILE):
            or_q_f32 = pl.cast(o_r[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE], target_type=pl.FP32)
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE] = pl.cast(or_q_half, target_type=pl.INT8, mode="trunc")

    # INT8 projection `o_r_i8 @ wo_b^T`, then dequantize -> final BF16 output.
    for nb in pl.spmd(D // B_N_TILE, name_hint="proj_b"):
        # K-split INT8 GEMM + dequant in one scope. T-tiled vec post-process
        # is the pypto#1472 workaround (same as proj_a).
        n0 = nb * B_N_TILE
        acc_b = pl.create_tensor([T_PAD, B_N_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(0, (O_GROUPS * O_LORA) // B_K_TILE, stage=2):
            k0 = kb * B_K_TILE
            xb_k_chunk = o_r_i8[0:T_PAD, k0:k0 + B_K_TILE]
            wb_k_chunk = wo_b[n0:n0 + B_N_TILE, k0:k0 + B_K_TILE]
            if k0 == 0:
                acc_b = pl.matmul(xb_k_chunk, wb_k_chunk, b_trans=True, out_dtype=pl.INT32)
            else:
                acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

        wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + B_N_TILE], [1, B_N_TILE])
        for b_tb in pl.range(0, T, B_T_TILE):
            acc_b_t = acc_b[b_tb:b_tb + B_T_TILE, 0:B_N_TILE]
            b_scale_t = o_r_scale_dq[b_tb:b_tb + B_T_TILE, 0:1]
            attn_t = pl.cast(acc_b_t, target_type=pl.FP32, mode="none")
            attn_t = pl.col_expand_mul(pl.row_expand_mul(attn_t, b_scale_t), wb_scale_chunk)
            attn_out[b_tb:b_tb + T, n0:n0 + B_N_TILE] = pl.cast(attn_t[0:T, 0:B_N_TILE], target_type=pl.BF16, mode="rint")

    return attn_out


@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    attn_out = sparse_attn(
        q,
        ori_kv,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        seqused_kv,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    return attn_out


