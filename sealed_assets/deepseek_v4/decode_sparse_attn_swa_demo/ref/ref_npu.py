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

from config import (
    CASE_CONFIG as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    DECODE_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)


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
MAX_SEQ_LEN = M.max_position_embeddings
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# kernel-local
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM

# tiling
VALID_TOKEN_TILE = 8
GATHER_FILL_TILE = 128
ROPE_OUT_TOK_TILE = 8
H_TILE = 16
# qk_pv cube-batch tile (M for the QK/PV matmuls). Batching QK_M_TILE head rows
# per matmul extracts the shared KV tile L1->L0 once per QK_M_TILE/H_TILE
# head-tiles (2x reuse at 32) instead of per H_TILE head-tile, then slices the
# [QK_M_TILE, ...] result back into H_TILE-row stores so the sparse_blk_* layout
# and merge_norm stay unchanged. 32 keeps the [32,128] softmax inside the 192KB
# Vec budget without a cross-core split. 64 is infeasible without further work
# (its [64,128] softmax and co-resident QK+PV L0C accumulators overflow Vec/L0C).
QK_M_TILE = 32
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
# proj_a cube K-frag. 256 (not 128) keeps the B-cache-line floor: B is K-contiguous
# under b_trans, so K*2B(bf16) = 512B == the a2a3 L2 line (K=128 was 256B, half a
# line -> wasted MTE2 DMA). At 256 the cube's L0A/L0B operand staging hits 100%
# (the wall); 512 would spill it for no gain (swept: K=512 net-negative).
A_K_TILE = 256
# proj_a is a pure-cube matmul scope (proj_a_mm) writing the fp32 GM intermediate
# o_r (cf. expert_routed w2 decouple), consumed directly by the fused amax+quant
# scope below; the decouple frees the cube N-frag from any vector-side UB constraint.
PROJ_A_MM_N_TILE = 128   # cube N frag; Mat/L0C have room but L0A/L0B are the wall at
                         # K=256, and a wider N raised cube exec without a TTT win (swept).
B_T_TILE = 8
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_TILE = 256  # proj_b_mm cube K frag; the GEMM is not the proj_b bottleneck (see below),
                # and a device sweep found growing it (512/1024) only re-streamed more weight
                # for no TTT gain, so it stays at the cache-line-safe 256 (256 B per INT8 row).
# proj_b is decoupled into a pure-cube GEMM scope (proj_b_mm) and a pure-vector dequant
# scope (proj_b_act) meeting through the attn_acc_i32 GM intermediate, so each sizes its
# own N-fragment to its own wall (cf. the expert_routed w2 decouple). A device tile sweep
# showed the cube is NOT the bottleneck (proj_b_mm ~32us, ~78% Exec, with L0C/L1 headroom)
# while the vector dequant dominated: growing PROJ_B_ACT_N_TILE 128->1024 cut the per-N-block
# vector-setup count 4x and dropped standalone TTT ~835->~730us. The vector frag goes to
# the UB wall; the cube frag sizes to its own L0C wall (below).
PROJ_B_MM_N_TILE = 256    # cube N frag. A later device-bias-controlled sweep (paired
                          # within-device, post vector-retune) found 128->256 worth ~1.5-2%
                          # (fewer matmul setups: per-D-chunk inner N-frags 8->4; proj_b_mm
                          # exec ~33->31us). Acc = M*N*4 = 128*256*4 = 128KB rides the L0C
                          # wall exactly (fits a2a3); Mat ~192KB has room. proj_a N stayed
                          # 128: growing it was TTT-neutral (64->32 task drop offset by 2x
                          # per-task cube exec). B_K_TILE stayed 256: the K=512 cache-line
                          # fix was ambiguous (raised proj_b dispatch-prop for no clear gain).
PROJ_B_ACT_N_TILE = 512   # vector N frag for the decoupled per-group dequant+sum (proj_b_act
                          # now sums O_GROUPS INT32 partials, each x its group act scale, then
                          # x the per-channel weight scale -> BF16). 512 (not 1024) keeps the
                          # O_GROUPS-way accumulate inside UB and gives D/512 = 8 vector tasks.
# Fused amax+quant token tile. The fused scope streams o_r twice (amax pass + quant
# pass) per token-tile rather than holding the whole row, so UB stays small; 8 keeps
# the [1, QUANT_TOKEN_TILE] fp32 amax tile 32-byte aligned (8*4=32B, the alloc-tile
# row floor that a [QUANT_TOKEN_TILE, 1] column accumulator would violate). The
# full-row hold (read o_r once) needs QUANT_TOKEN_TILE<=4 for UB but >=8 for that
# alignment -- mutually exclusive -- so we stream; the 2nd pass mostly hits L2.
QUANT_TOKEN_TILE = 8
# Per-group back-to-back o_proj (manual-scope, qwen3-style fine-grained deps):
# proj_a[g] -> quant[g] (PER-GROUP amax, no global barrier) -> proj_b[g] pipeline.
# Each proj_b group-slab dequantizes its INT32 partial (per-row group act scale x
# per-channel weight scale) and FP32 atomic-adds into a zero-seeded accumulator.
SEED_N_CHUNK = 128   # attn_acc_f32 zero-seed column chunk ([T, 128] FP32 x2 pipeline = 128KB UB)
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE   # proj_a cube N-frags per group
PB_NFRAGS = D // PROJ_B_MM_N_TILE         # proj_b total cube N-frags over D (module-level
                                          # so pl.array.create / range() get static ints)
# proj_b is one task per (D-chunk, group): the D-chunk's N-frags loop INSIDE the task,
# so the per-group split does NOT multiply the task count by N-frags. PROJ_B_D_CHUNK=1024
# keeps proj_b at O_GROUPS*(D//PROJ_B_D_CHUNK) = 32 tasks (= the baseline proj_b count),
# each doing the same total work, instead of 256 tiny tasks (more dispatch overhead).
PROJ_B_D_CHUNK = 1024
PB_DCHUNKS = D // PROJ_B_D_CHUNK
# quant is split per (group, token-chunk) so the per-group amax+quant spreads over more
# vector cores: O_GROUPS*NUM_QUANT_T_CHUNKS = 8*4 = 32 tasks.
NUM_QUANT_T_CHUNKS = 1
QUANT_T_CHUNK = T // NUM_QUANT_T_CHUNKS
# proj_b_act is split per (D-region, token-block) so the O_GROUPS-way dequant+sum spreads
# over more vector cores: (D//PROJ_B_ACT_N_TILE)*(T//PROJ_B_ACT_TBLK) = 8*4 = 32 tasks.
PROJ_B_ACT_T_TILE = 8    # inner token tile for the proj_b_act O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TBLK = 8      # proj_b_act token block per task
PB_ACT_NREG = D // PROJ_B_ACT_N_TILE
PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK
NEG_INF = -1.0e20
CACHE_INSERT_BLOCKS = 1

assert T % VALID_TOKEN_TILE == 0
assert T % 2 == 0
assert H % 4 == 0
assert QK_M_TILE % H_TILE == 0
assert H % QK_M_TILE == 0
assert T % QUANT_TOKEN_TILE == 0
assert H % O_GROUPS == 0
assert (O_GROUPS * O_LORA) % B_K_TILE == 0
assert D % PROJ_B_MM_N_TILE == 0, "proj_b_mm cube N-loop must cover D"
assert D % PROJ_B_D_CHUNK == 0, "proj_b D-chunk loop must cover D"
assert PROJ_B_D_CHUNK % PROJ_B_MM_N_TILE == 0, "proj_b inner N-frag loop must cover the D-chunk"
assert T % NUM_QUANT_T_CHUNKS == 0 and QUANT_T_CHUNK % QUANT_TOKEN_TILE == 0
assert T % PROJ_B_ACT_TBLK == 0 and PROJ_B_ACT_TBLK % PROJ_B_ACT_T_TILE == 0
assert D % PROJ_B_ACT_N_TILE == 0, "proj_b_act vector N-loop must cover D"
assert O_LORA % B_K_TILE == 0, "proj_b group K-loop covers O_LORA in B_K_TILE iters"
assert D % SEED_N_CHUNK == 0


# SWA sparse-K width: sliding window only.
TOPK = WIN
# Decode SWA consumes metadata-expanded physical KV-cache slots. The current
# kernel shape keeps the SWA window in one attention K tile.
SPARSE_BLOCKS = 1
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
assert TOPK == WIN, f"SWA decode expects TOPK ({TOPK}) == WIN ({WIN})"
assert WIN == ATTN_K_TILE, f"SWA decode expects WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"


@pl.jit.inline
def sparse_attn_swa(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv_flat: pl.Tensor[[ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    # SWA metadata already lowered each logical window row to a physical cache
    # slot. Current decode tokens must be inserted into ori_kv by the caller
    # before this function runs; there is no MTP overlay path here.
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)
    for vb_blk in pl.spmd(T // VALID_TOKEN_TILE, name_hint="swa_valid_bias"):
        vb = vb_blk * VALID_TOKEN_TILE
        v_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        v_col_m = pl.col_expand(pl.full([VALID_TOKEN_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0), v_col)
        v_lens = pl.cast(pl.reshape(swa_lens[vb : vb + VALID_TOKEN_TILE], [VALID_TOKEN_TILE, 1]), target_type=pl.FP32)
        v_valid = pl.minimum(
            pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0),
            1.0,
        )
        sparse_bias[vb : vb + VALID_TOKEN_TILE, 0 : ATTN_K_TILE] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)

    swa_kv_flat = pl.create_tensor([T * WIN, HEAD_DIM], dtype=pl.BF16)
    gather_tids = pl.array.create(1, pl.TASK_ID)
    with pl.spmd(T, name_hint="swa_gather_kv") as gather_tid:
        g_t = pl.tile.get_block_idx()
        g_base = g_t * WIN
        for g_r in pl.range(WIN):
            g_slot_i32 = pl.read(swa_indices, [g_t, g_r])
            g_dst = g_base + g_r
            if g_slot_i32 >= 0:
                g_slot = pl.cast(g_slot_i32, pl.INDEX)
                swa_kv_flat[g_dst : g_dst + 1, 0 : HEAD_DIM] = ori_kv_flat[g_slot : g_slot + 1, 0 : HEAD_DIM]
            else:
                swa_kv_flat[g_dst : g_dst + 1, 0 : HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    gather_tids[0] = gather_tid

    # qk_pv writes per-tile (mi, li, oi) to GM; merge_norm reads them back. Not
    # fused on a2a3: the PV output (Acc) -> online rescale (Vec) needs an
    # unsupported tmov, and a [H_TILE, HEAD_DIM] carry overflows the Vec buffer.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(T, name_hint="qk_pv", deps=[gather_tids[0]]) as qk_tid:
        qk_t = pl.tile.get_block_idx()
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        for qk_sb in pl.unroll(SPARSE_BLOCKS):
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            qk_base = qk_t * WIN + qk_s0
            qk_kv = swa_kv_flat[qk_base : qk_base + ATTN_K_TILE, 0 : HEAD_DIM]

            # Cube-batch QK_M_TILE head rows per QK/PV matmul so the shared KV
            # tile is extracted L1->L0 once per QK_M_TILE/H_TILE head-tiles
            # (2x reuse at QK_M_TILE=32) instead of per head-tile. The
            # [QK_M_TILE, ...] softmax result is sliced back into H_TILE-row
            # stores at the SAME offsets as the per-head-tile path
            # (qk_h_idx == qk_hb * (QK_M_TILE // H_TILE) + qk_sub), so the
            # sparse_blk_* layout and merge_norm are bit-identical.
            for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                qk_h0 = qk_hb * QK_M_TILE
                qk_head_row = qk_t * H + qk_h0
                qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]
                qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                qk_scores = pl.add(qk_scaled, pl.col_expand(pl.full([QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0), qk_bias_row))
                qk_mi = pl.row_max(qk_scores)
                # Invalid lanes (NEG_INF bias, zero kv rows) exp to ~0; all-invalid
                # blocks die in the merge alpha/beta -- no mask multiply needed.
                qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                qk_li = pl.row_sum(qk_exp)
                qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                for qk_sub in pl.unroll(QK_M_TILE // H_TILE):
                    qk_h_idx = qk_hb * (QK_M_TILE // H_TILE) + qk_sub
                    qk_r0 = qk_sub * H_TILE
                    qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                    qk_row = qk_blk_base + qk_sb * H_TILE
                    sparse_blk_mi[qk_row : qk_row + H_TILE, 0 : 1] = qk_mi[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                    sparse_blk_li[qk_row : qk_row + H_TILE, 0 : 1] = qk_li[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                    sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi[qk_r0 : qk_r0 + H_TILE, 0 : HEAD_DIM]

    # Precompute the head-invariant interleaved cos and sign*sin once: they depend
    # only on (token, column), not head, so building them per head would repeat the
    # same dup-gather H times on the bottleneck Vec engine. sign is folded into sin
    # (multiply by +/-1). The conjugate (inverse) rotation is:
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]
    # Hoisted ABOVE merge_norm (which now fuses the rotation): independent of qk_pv,
    # so it overlaps it and is off merge_norm's critical path.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    for cp in pl.spmd(HALF_ROPE // ROPE_TILE, name_hint="rope_cs"):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...] (conjugate)
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(
            pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

    # Online-softmax merge across sparse-K tiles, sink-norm, then fused inverse RoPE.
    # One spmd block per (token, head-tile) -- T*(H//H_TILE) blocks -- so the merge
    # fans out over that many AIVs instead of T blocks each running a serial head-tile
    # loop. The inverse-RoPE rotation + rope-column pack is fused in (was a separate
    # "rope" spmd reading an attn_rope_stage GM round-trip): the head-tile's fp32 rope
    # segment is rotated in UB and packed straight into o_packed's rope columns.
    # with-form spmd so the dispatch TaskId (merge_tid) can be an explicit dep of
    # the manual-scope proj_a tasks below (which read merge_norm's o_packed cols).
    with pl.spmd(T * (H // H_TILE), name_hint="merge_norm") as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_idx * SPARSE_BLOCKS * H_TILE
        m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

        # Online-merge any additional window tiles if the SWA window grows past
        # one ATTN_K_TILE.
        if SPARSE_BLOCKS > 1:
            for m_sb in pl.range(1, SPARSE_BLOCKS):
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
        n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

        # Inverse RoPE on this head-tile's fp32 rope segment. cos_il / sign*sin are
        # head-invariant for token m_t, so col_expand them over the H_TILE head rows;
        # swap_idx (j^1) pairs the interleaved real/imag lanes. Rounded to bf16 (golden
        # also rounds inverse-RoPE to bf16) and packed into o_packed's rope columns.
        m_col = pl.col_expand_mul(
            pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        m_dup_f = pl.cast(pl.cast(pl.mul(m_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        m_lane = pl.sub(m_col, pl.mul(m_dup_f, 2.0))                                              # j%2
        m_swap_idx = pl.cast(pl.sub(pl.add(m_col, 1.0), pl.mul(m_lane, 2.0)), target_type=pl.INT32)  # j^1
        m_rope = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0 : ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0 : ROPE_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=m_swap_idx)
        m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

        for n_hi in pl.unroll(H_TILE):
            n_gh = m_h0 + n_hi
            n_g = n_gh // HEADS_PER_GROUP
            n_hh = n_gh - n_g * HEADS_PER_GROUP
            n_pack_row = n_g * T + m_t
            n_col = n_hh * HEAD_DIM
            o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + NOPE_DIM] = n_bf16[n_hi : n_hi + 1, 0 : NOPE_DIM]
            o_packed[n_pack_row : n_pack_row + 1, n_col + NOPE_DIM : n_col + HEAD_DIM] = n_rope_bf16[n_hi : n_hi + 1, 0 : ROPE_DIM]

    # ========================================================================
    # Back-to-back grouped output projection (manual scope, PER-GROUP INT8 quant).
    #
    # Per-GROUP amax localizes the quant reduction to each O_LORA group (vs the
    # per-ROW-amax form, where a full-8192-row reduction is a hard barrier between
    # proj_a and proj_b), so the three stages PIPELINE per group with qwen3-style
    # fine-grained deps: proj_b[*, g] waits only on quant[g], which waits only on
    # proj_a[g, *] -- so proj_b's cube for group g runs while proj_a/quant of later
    # groups are still in flight (a genuine proj_a<->proj_b back-to-back GEMM).
    #
    # manual_scope SUPPRESSES auto-dep, so every edge is explicit: proj_a reads
    # o_packed (auto region) -> deps=[merge_tid] (merge_norm now writes both the NOPE
    # and the fused-RoPE columns); quant[g] deps on group
    # g's proj_a tasks; proj_b deps on [seed, quant[g]]. Each proj_b group-slab
    # dequantizes its INT32 partial by the group's per-row act scale (the per-group
    # scale cannot factor out of the K-sum) and FP32 atomic-adds into a zero-seeded
    # accumulator; proj_b_act (auto region) applies the per-channel weight scale and
    # is the consolidated writer that registers attn_out's return tensormap edge.
    # ========================================================================
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)   # [G, T] so each group's
                                                                     # per-row scale is a contiguous
                                                                     # row (column reads would be a
                                                                     # strided GM->VecTile load)
    # Per-group INT32 partials: proj_b_mm (pure cube) writes group g's contribution to
    # output channel n at partials[:, g*D + n]; proj_b_act (pure vector) sums the
    # O_GROUPS partials with their per-group act scales. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    quant_tids = pl.array.create(O_GROUPS * NUM_QUANT_T_CHUNKS, pl.TASK_ID)
    proj_b_tids = pl.array.create(PB_DCHUNKS * O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        # proj_a[g, nf]: BF16 grouped GEMM -> o_r[:, group g], peel-first-iter form.
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mm", deps=[merge_tid]) as pa_tid:
                    xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, 0], valid_shape=[T, A_K_TILE])
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, k0], valid_shape=[T, A_K_TILE])
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, k0:k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    o_r_pad = pl.assemble(o_r_pad, acc_a, [0, out_col_g + n0])
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid

        # quant[g, tc]: PER-GROUP amax + symmetric INT8 quant of o_r[:, group g] over a
        # token-chunk (no cross-group barrier -- what unlocks the per-group pipeline).
        # Deps on ONLY group g's proj_a tasks; stores the per-row group dequant scale
        # into act_scale_dq[g, :]. The token-chunk split spreads it over more vector
        # cores. Cast chain (rint INT32 -> round FP16 -> trunc INT8) = per-row form per group.
        for tc in pl.parallel(NUM_QUANT_T_CHUNKS):
            t_base = tc * QUANT_T_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant",
                           deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)]) as q_tid:
                    for qt in pl.pipeline(t_base, t_base + QUANT_T_CHUNK, QUANT_TOKEN_TILE, stage=2):
                        # amax pass: O_LORA fits one tile, so the whole group row is read at once.
                        oc_amax = o_r_pad[qt:qt + QUANT_TOKEN_TILE, col_g:col_g + O_LORA]
                        g_amax = pl.maximum(
                            pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS),
                            pl.reshape(pl.row_max(pl.maximum(oc_amax, pl.neg(oc_amax))), [1, QUANT_TOKEN_TILE]))
                        g_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), g_amax)
                        act_scale_dq = pl.assemble(act_scale_dq, pl.recip(g_sq_row), [g, qt])
                        g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                        # quant pass: re-read o_r (the 2nd stream mostly hits L2) to keep UB small.
                        oc_q = o_r_pad[qt:qt + QUANT_TOKEN_TILE, col_g:col_g + O_LORA]
                        oq_i32 = pl.cast(pl.row_expand_mul(oc_q, g_sq_col), target_type=pl.INT32, mode="rint")
                        oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                        oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                        o_r_i8 = pl.assemble(o_r_i8, oq_i8, [qt, col_g])
                        o_r_i8_pad = pl.assemble(o_r_i8_pad, oq_i8, [qt, col_g])
                        if T_PAD > T:
                            zero_i32 = pl.full([T_PAD - T, O_LORA], dtype=pl.INT32, value=0)
                            zero_half = pl.cast(zero_i32, target_type=pl.FP16, mode="round")
                            o_r_i8_pad = pl.assemble(
                                o_r_i8_pad,
                                pl.cast(zero_half, target_type=pl.INT8, mode="trunc"),
                                [T, col_g],
                            )
                quant_tids[g * NUM_QUANT_T_CHUNKS + tc] = q_tid

        # proj_b_mm[dc, g]: PURE-CUBE INT8 GEMM of group g's contribution to a
        # PROJ_B_D_CHUNK-wide slab of D, written as INT32 partials to partials[:, g*D+n]
        # (NO vector work here -- the cube never stalls on a dequant; the dequant/sum
        # moves to proj_b_act). The slab's N-frags loop INSIDE the task, so proj_b stays
        # at PB_DCHUNKS*O_GROUPS = 32 tasks. Deps = quant[g] (all token-chunks) ONLY ->
        # group g's proj_b fires as soon as quant[g] lands, overlapping proj_a[g+1] (the
        # back-to-back). Peel-first matmul (matmul_acc from a zero carry trips the TLOAD
        # DN->NZ assertion, pypto#1540).
        for dc in pl.parallel(PB_DCHUNKS):
            d0 = dc * PROJ_B_D_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mm",
                           deps=[quant_tids[g * NUM_QUANT_T_CHUNKS + tc] for tc in range(NUM_QUANT_T_CHUNKS)]) as pb_tid:
                    for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        acc_b = pl.create_tensor([MM_T_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
                        for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                            k0 = col_g + kb * B_K_TILE
                            if kb == 0:
                                acc_b = pl.matmul(o_r_i8_pad[:, col_g:col_g + B_K_TILE],
                                          wo_b[n0:n0 + PROJ_B_MM_N_TILE, col_g:col_g + B_K_TILE],
                                          b_trans=True, out_dtype=pl.INT32)
                            else:
                                acc_b = pl.matmul_acc(acc_b, o_r_i8_pad[:, k0:k0 + B_K_TILE],
                                                  wo_b[n0:n0 + PROJ_B_MM_N_TILE, k0:k0 + B_K_TILE], b_trans=True)
                        partials = pl.assemble(partials, acc_b, [0, g * D + n0])
                proj_b_tids[dc * O_GROUPS + g] = pb_tid

    # proj_b_act (PURE-VECTOR consolidated writer, auto region): sum the O_GROUPS INT32
    # partials -- each dequantized by its group's per-row act scale -- then apply the
    # per-channel weight scale -> BF16. Explicit deps on all proj_b_mm tasks bridge
    # manual_scope -> the return's auto-dep (this auto-region write registers the edge).
    with pl.spmd(PB_ACT_NREG * PB_ACT_TBLKS, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(PB_DCHUNKS * O_GROUPS)]) as act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // PB_ACT_TBLKS
        tblk = act_idx - nreg * PB_ACT_TBLKS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TBLK
        wb_scale_chunk = pl.reshape(wo_b_scale[ob_n0:ob_n0 + PROJ_B_ACT_N_TILE], [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TBLK, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.pipeline(O_GROUPS, stage=2):
                p_g = partials[b_tb:b_tb + PROJ_B_ACT_T_TILE, g * D + ob_n0:g * D + ob_n0 + PROJ_B_ACT_N_TILE]
                g_scale = pl.reshape(act_scale_dq[g:g + 1, b_tb:b_tb + PROJ_B_ACT_T_TILE], [PROJ_B_ACT_T_TILE, 1])
                acc = pl.add(acc, pl.row_expand_mul(pl.cast(p_g, target_type=pl.FP32, mode="none"), g_scale))
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            attn_out[b_tb:b_tb + PROJ_B_ACT_T_TILE, ob_n0:ob_n0 + PROJ_B_ACT_N_TILE] = pl.cast(out_t, target_type=pl.BF16, mode="rint")

    return attn_out

@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_attn_swa(
        q,
        ori_kv_flat,
        swa_indices,
        swa_lens,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    return attn_out


