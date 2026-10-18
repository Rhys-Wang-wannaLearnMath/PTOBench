# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped output projection (decode).

Corresponds to model.py Attention.forward decode branch lines 533-542:
    o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, softmax_scale)
    apply_rotary_emb(o[..., -rope_dim:], freqs_cis, inverse=True)
    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    x = self.wo_b(o.flatten(2))

Inputs:
- q                 : query tensor from MLA prolog (RoPE already applied)
- ori_kv / cmp_kv   : paged sliding-window and paged compressed KV pools
- cmp_sparse_indices: per-token absolute indices (window + compressed concat)
                      computed by orchestrator (window topk + indexer/HCA topk)
- attn_sink         : per-head sink term added inside softmax
- freqs_cos/sin     : split-half inverse-RoPE tables for the sparse-attn output
- wo_a              : grouped first-stage output-projection weights from model.py:537-541
- wo_b / wo_b_scale : grouped second-stage output-projection W8 per-channel weights

Standalone contract:
- `cmp_sparse_indices[t, :]` may contain `-1` pads.
- entries in `[0, WIN)` address the logical sliding-window ring slots.
- entries in `[WIN, WIN + cmp_valid)` address compressed cache slots, where
    `seq_used = seqused_kv[b] - S + 1 + s`
    and `cmp_valid = max(seq_used - min(WIN, seq_used), 0)`.
- the grouped projection layout matches:
    `o.view(bsz, seqlen, self.n_local_groups, -1)` with
    `self.n_local_groups == O_GROUPS` and `-1 == O_GROUP_IN`.

The standalone harness exposes `--compress-ratio {0,4,128}` for testing.
"""


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
MAX_SEQ_LEN = M.max_position_embeddings
IDX_TOPK = M.index_topk
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# kernel-local
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 128
ORI_MAX_BLOCKS = 1                 # paged-KV pool: ori (sliding-window) blocks per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 64                # paged-KV pool: compressed blocks per batch
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# tiling
GATHER_TOKEN_TILE = 4
ATTN_TOKEN_TILE = 8
ROPE_TOKEN_TILE = 4
ROPE_PACK_TOKEN_TILE = 32
ROPE_PACK_GROUP_TILE = 1
MATMUL_ROW_PAD = 16
SPARSE_ATTN_TILE = 64
SPARSE_ATTN_BLOCKS = (TOPK + SPARSE_ATTN_TILE - 1) // SPARSE_ATTN_TILE
ROPE_CHUNK = 16
ROPE_INTERLEAVE_CHUNK = 2 * ROPE_CHUNK
A_K_CHUNK = 128
A_N_CHUNK = 128
MM_T_TILE = 16          # cube matmul rows must be 16-aligned on current ptoas
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_CHUNK = 128
B_N_CHUNK = 128 if T >= 128 else 256
QUANT_CHUNK = 32 if T >= 128 else (128 if T >= 64 else 256)


def get_standalone_cmp_valid(compress_ratio: int) -> int:
    """Map demo compress-ratio modes to the valid compressed-cache tail length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return IDX_TOPK
    if compress_ratio == 128:
        return MAX_SEQ_LEN // compress_ratio
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")


@pl.jit.inline
def sparse_attn(
    q:                  pl.Tensor[[T, H, HEAD_DIM],                               pl.BF16],
    ori_kv:             pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    ori_block_table:    pl.Tensor[[B, ORI_MAX_BLOCKS],                            pl.INT32],
    cmp_kv:             pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    cmp_block_table:    pl.Tensor[[B, CMP_MAX_BLOCKS],                            pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK],                                      pl.INT32],
    attn_sink:          pl.Tensor[[H],                                            pl.FP32],
    seqused_kv:         pl.Tensor[[B],                                            pl.INT32],
    freqs_cos:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    freqs_sin:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    even_select_local:  pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],            pl.BF16],
    odd_select_local:   pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],            pl.BF16],
    wo_a:               pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],                 pl.BF16],
    wo_b:               pl.Tensor[[D, O_GROUPS * O_LORA],                         pl.INT8],
    wo_b_scale:         pl.Tensor[[D],                                            pl.FP32],
    attn_out:           pl.Tensor[[T, D],                                         pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    A_K_BLOCKS = O_GROUP_IN // A_K_CHUNK
    A_N_BLOCKS = O_LORA // A_N_CHUNK
    B_K_BLOCKS = (O_GROUPS * O_LORA) // B_K_CHUNK
    B_N_BLOCKS = D // B_N_CHUNK

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [B * CMP_MAX_BLOCKS])
    cmp_sparse_indices_flat = pl.reshape(cmp_sparse_indices, [T * TOPK])
    sparse_kv = pl.create_tensor([T * TOPK, HEAD_DIM], dtype=pl.BF16)
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.BF16)
    sparse_exp = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, SPARSE_ATTN_TILE], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, HEAD_DIM], dtype=pl.FP32)
    sparse_mi = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    sparse_li = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    sparse_oi = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.FP32)
    o_proj_even = pl.create_tensor([T * H, HALF_ROPE], dtype=pl.FP32)
    o_proj_odd = pl.create_tensor([T * H, HALF_ROPE], dtype=pl.FP32)
    rope_even = pl.create_tensor([T * H, HALF_ROPE], dtype=pl.BF16)
    rope_odd = pl.create_tensor([T * H, HALF_ROPE], dtype=pl.BF16)
    rope_even_interleave_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    rope_odd_interleave_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)

    # Stage 1: gather the sparse KV rows selected by the sliding-window path and
    # the compressed-cache path into one per-token packed KV list.
    for gather_t0 in pl.parallel(0, T, GATHER_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_gather_kv_topk_tile"):
            for gather_dt in pl.range(GATHER_TOKEN_TILE):
                gather_t = gather_t0 + gather_dt
                gather_b = gather_t // S
                gather_s = gather_t - gather_b * S
                gather_seq_final = pl.read(seqused_kv, [gather_b])
                gather_seq_used = gather_seq_final - S + 1 + gather_s
                gather_window_valid = pl.min(WIN, gather_seq_used)
                gather_cmp_valid = gather_seq_used - gather_window_valid
                gather_cmp_topk_valid = pl.min(IDX_TOPK, gather_cmp_valid)
                gather_sparse_k = gather_window_valid + gather_cmp_topk_valid
                gather_ori_block_base = gather_b * ORI_MAX_BLOCKS
                gather_cmp_block_base = gather_b * CMP_MAX_BLOCKS
                gather_sparse_idx_base = gather_t * TOPK
                gather_sparse_kv_base = gather_t * TOPK

                # The standalone decode contract uses a contiguous full-window
                # prefix, so copy that prefix as one row block and leave the
                # truly sparse compressed tail on the dynamic row-gather path.
                gather_ori_blk = pl.cast(pl.read(ori_block_table_flat, [gather_ori_block_base]), pl.INDEX)
                gather_ori_row = gather_ori_blk * BLOCK_SIZE
                window_rows = pl.slice(
                    ori_kv_flat,
                    [WIN, HEAD_DIM],
                    [gather_ori_row, 0],
                    valid_shape=[gather_window_valid, HEAD_DIM],
                )
                sparse_kv = pl.assemble(sparse_kv, window_rows, [gather_sparse_kv_base, 0])

                # Append compressed-cache hits after the window prefix.
                for gather_cmp_kk in pl.range(gather_cmp_topk_valid):
                    gather_cmp_idx_pos = gather_sparse_idx_base + gather_window_valid + gather_cmp_kk
                    gather_cmp_raw_idx = pl.read(cmp_sparse_indices_flat, [gather_cmp_idx_pos])
                    gather_cmp_slot = gather_cmp_raw_idx - WIN
                    gather_cmp_block_slot = gather_cmp_slot // BLOCK_SIZE
                    gather_cmp_block_pos = gather_cmp_block_base + gather_cmp_block_slot
                    gather_cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [gather_cmp_block_pos]), pl.INDEX)
                    gather_cmp_intra = gather_cmp_slot % BLOCK_SIZE
                    gather_cmp_row = gather_cmp_blk * BLOCK_SIZE + gather_cmp_intra
                    sparse_kv = pl.assemble(
                        sparse_kv,
                        cmp_kv_flat[gather_cmp_row : gather_cmp_row + 1, 0 : HEAD_DIM],
                        [gather_sparse_kv_base + gather_window_valid + gather_cmp_kk, 0],
                    )

                # Keep padded rows deterministic for ratio-0/128 sanity modes.
                zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for gather_pad_kk in pl.range(gather_sparse_k, TOPK):
                    sparse_kv = pl.assemble(sparse_kv, zero_kv_row, [gather_sparse_kv_base + gather_pad_kk, 0])

    for attn_t0 in pl.parallel(0, T, ATTN_TOKEN_TILE):
        for h0 in pl.parallel(0, H, MATMUL_ROW_PAD):
            # Stage 2a: QK + tile-local softmax for every sparse-K tile.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_qk_softmax_tile"):
                for qk_dt in pl.range(ATTN_TOKEN_TILE):
                    qk_t = attn_t0 + qk_dt
                    qk_b = qk_t // S
                    qk_s = qk_t - qk_b * S
                    qk_seq_final = pl.read(seqused_kv, [qk_b])
                    qk_seq_used = qk_seq_final - S + 1 + qk_s
                    qk_window_valid = pl.min(WIN, qk_seq_used)
                    qk_cmp_valid = qk_seq_used - qk_window_valid
                    qk_cmp_topk_valid = pl.min(IDX_TOPK, qk_cmp_valid)
                    qk_sparse_k = qk_window_valid + qk_cmp_topk_valid
                    qk_sparse_kv_base = qk_t * TOPK
                    qk_head_row = qk_t * H + h0
                    qk_q_batch = q_flat[qk_head_row : qk_head_row + MATMUL_ROW_PAD, 0 : HEAD_DIM]

                    for qk_sb in pl.range(SPARSE_ATTN_BLOCKS):
                        qk_tile_start = qk_sb * SPARSE_ATTN_TILE
                        if qk_tile_start < qk_sparse_k:
                            qk_tile_valid = pl.min(SPARSE_ATTN_TILE, qk_sparse_k - qk_tile_start)
                            qk_kv_tile = sparse_kv[
                                qk_sparse_kv_base + qk_tile_start : qk_sparse_kv_base + qk_tile_start + SPARSE_ATTN_TILE,
                                0 : HEAD_DIM,
                            ]
                            qk_raw_scores = pl.matmul(qk_q_batch, qk_kv_tile, b_trans=True, out_dtype=pl.FP32)
                            qk_scores_valid = pl.slice(
                                pl.mul(qk_raw_scores, SOFTMAX_SCALE),
                                [MATMUL_ROW_PAD, SPARSE_ATTN_TILE],
                                [0, 0],
                                valid_shape=[MATMUL_ROW_PAD, qk_tile_valid],
                            )
                            qk_scores = pl.fillpad(qk_scores_valid, pad_value=pl.PadValue.min)
                            qk_mi = pl.row_max(qk_scores)
                            qk_exp_scores = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                            qk_exp_scores_bf16 = pl.cast(qk_exp_scores, target_type=pl.BF16)
                            qk_li = pl.row_sum(pl.cast(qk_exp_scores_bf16, target_type=pl.FP32))
                            qk_block_row = qk_t * H * SPARSE_ATTN_BLOCKS + qk_sb * H + h0
                            sparse_exp = pl.assemble(sparse_exp, qk_exp_scores_bf16, [qk_block_row, 0])
                            sparse_blk_mi = pl.assemble(sparse_blk_mi, qk_mi, [qk_block_row, 0])
                            sparse_blk_li = pl.assemble(sparse_blk_li, qk_li, [qk_block_row, 0])

            # Stage 2b: PV for each sparse-K tile. Keep the online merge in a
            # separate scope under FLASH so the AIV live set stays bounded.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_pv_tile"):
                for pv_dt in pl.range(ATTN_TOKEN_TILE):
                    pv_t = attn_t0 + pv_dt
                    pv_b = pv_t // S
                    pv_s = pv_t - pv_b * S
                    pv_seq_final = pl.read(seqused_kv, [pv_b])
                    pv_seq_used = pv_seq_final - S + 1 + pv_s
                    pv_window_valid = pl.min(WIN, pv_seq_used)
                    pv_cmp_valid = pv_seq_used - pv_window_valid
                    pv_cmp_topk_valid = pl.min(IDX_TOPK, pv_cmp_valid)
                    pv_sparse_k = pv_window_valid + pv_cmp_topk_valid
                    pv_sparse_kv_base = pv_t * TOPK
                    for pv_sb in pl.range(SPARSE_ATTN_BLOCKS):
                        pv_tile_start = pv_sb * SPARSE_ATTN_TILE
                        if pv_tile_start < pv_sparse_k:
                            pv_block_row = pv_t * H * SPARSE_ATTN_BLOCKS + pv_sb * H + h0
                            pv_exp = sparse_exp[pv_block_row : pv_block_row + MATMUL_ROW_PAD, 0 : SPARSE_ATTN_TILE]
                            pv_kv_tile = sparse_kv[
                                pv_sparse_kv_base + pv_tile_start : pv_sparse_kv_base + pv_tile_start + SPARSE_ATTN_TILE,
                                0 : HEAD_DIM,
                            ]
                            pv_oi_tmp = pl.matmul(pv_exp, pv_kv_tile, out_dtype=pl.FP32)
                            sparse_blk_oi = pl.assemble(sparse_blk_oi, pv_oi_tmp, [pv_block_row, 0])

            # Stage 2c: online-softmax merge across sparse-K tiles.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_merge_tile"):
                for merge_dt in pl.range(ATTN_TOKEN_TILE):
                    merge_t = attn_t0 + merge_dt
                    merge_b = merge_t // S
                    merge_s = merge_t - merge_b * S
                    merge_seq_final = pl.read(seqused_kv, [merge_b])
                    merge_seq_used = merge_seq_final - S + 1 + merge_s
                    merge_window_valid = pl.min(WIN, merge_seq_used)
                    merge_cmp_valid = merge_seq_used - merge_window_valid
                    merge_cmp_topk_valid = pl.min(IDX_TOPK, merge_cmp_valid)
                    merge_sparse_k = merge_window_valid + merge_cmp_topk_valid
                    merge_head_row = merge_t * H + h0
                    merge_block_row0 = merge_t * H * SPARSE_ATTN_BLOCKS + h0
                    merge_mi = sparse_blk_mi[merge_block_row0 : merge_block_row0 + MATMUL_ROW_PAD, 0 : 1]
                    merge_li = sparse_blk_li[merge_block_row0 : merge_block_row0 + MATMUL_ROW_PAD, 0 : 1]
                    merge_oi = sparse_blk_oi[merge_block_row0 : merge_block_row0 + MATMUL_ROW_PAD, 0 : HEAD_DIM]

                    for merge_sb in pl.range(1, SPARSE_ATTN_BLOCKS):
                        merge_tile_start = merge_sb * SPARSE_ATTN_TILE
                        if merge_tile_start < merge_sparse_k:
                            merge_block_row = merge_t * H * SPARSE_ATTN_BLOCKS + merge_sb * H + h0
                            merge_cur_mi = sparse_blk_mi[merge_block_row : merge_block_row + MATMUL_ROW_PAD, 0 : 1]
                            merge_cur_li = sparse_blk_li[merge_block_row : merge_block_row + MATMUL_ROW_PAD, 0 : 1]
                            merge_cur_oi = sparse_blk_oi[merge_block_row : merge_block_row + MATMUL_ROW_PAD, 0 : HEAD_DIM]
                            merge_mi_new = pl.maximum(merge_mi, merge_cur_mi)
                            merge_alpha = pl.exp(pl.sub(merge_mi, merge_mi_new))
                            merge_beta = pl.exp(pl.sub(merge_cur_mi, merge_mi_new))
                            merge_li = pl.add(pl.mul(merge_alpha, merge_li), pl.mul(merge_beta, merge_cur_li))
                            merge_oi = pl.add(
                                pl.row_expand_mul(merge_oi, merge_alpha),
                                pl.row_expand_mul(merge_cur_oi, merge_beta),
                            )
                            merge_mi = merge_mi_new

                    sparse_mi = pl.assemble(sparse_mi, merge_mi, [merge_head_row, 0])
                    sparse_li = pl.assemble(sparse_li, merge_li, [merge_head_row, 0])
                    sparse_oi = pl.assemble(sparse_oi, merge_oi, [merge_head_row, 0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_norm_tile"):
                for norm_dt in pl.range(ATTN_TOKEN_TILE):
                    norm_t = attn_t0 + norm_dt
                    norm_attn_head_row = norm_t * H + h0
                    norm_oi = sparse_oi[norm_attn_head_row : norm_attn_head_row + MATMUL_ROW_PAD, 0 : HEAD_DIM]
                    norm_mi = sparse_mi[norm_attn_head_row : norm_attn_head_row + MATMUL_ROW_PAD, 0 : 1]
                    norm_li = sparse_li[norm_attn_head_row : norm_attn_head_row + MATMUL_ROW_PAD, 0 : 1]
                    norm_sink_bias = pl.reshape(attn_sink[h0 : h0 + MATMUL_ROW_PAD], [MATMUL_ROW_PAD, 1])
                    norm_sink_tile = pl.add(pl.sub(norm_mi, norm_mi), norm_sink_bias)
                    norm_denom = pl.add(norm_li, pl.exp(pl.sub(norm_sink_tile, norm_mi)))
                    oi_out = pl.row_expand_div(norm_oi, norm_denom)
                    attn_stage_row = pl.cast(
                        oi_out[0 : MATMUL_ROW_PAD, 0 : HEAD_DIM],
                        target_type=pl.BF16,
                    )
                    attn_rope_stage = pl.assemble(
                        attn_rope_stage,
                        attn_stage_row[0 : MATMUL_ROW_PAD, NOPE_DIM:HEAD_DIM],
                        [norm_attn_head_row, 0],
                    )

                    for norm_head_i in pl.range(MATMUL_ROW_PAD):
                        norm_global_head = h0 + norm_head_i
                        norm_g = norm_global_head // HEADS_PER_GROUP
                        norm_hh = norm_global_head - norm_g * HEADS_PER_GROUP
                        norm_pack_row = norm_g * T + norm_t
                        norm_head_col = norm_hh * HEAD_DIM
                        o_packed = pl.assemble(
                            o_packed,
                            attn_stage_row[norm_head_i : norm_head_i + 1, 0:NOPE_DIM],
                            [norm_pack_row, norm_head_col],
                        )

    # Stage 3: inverse RoPE on the rope slice of the attention output by
    # deinterleaving even/odd lanes, rotating them, then reinterleaving.
    for rope_t0 in pl.parallel(0, T, ROPE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_rope_slice_tile"):
            for rope_dt in pl.range(ROPE_TOKEN_TILE):
                rope_slice_t = rope_t0 + rope_dt
                rope_slice_head_row = rope_slice_t * H

                # Split interleaved rope lanes into even and odd halves.
                for rope_slice_r0 in pl.range(0, HALF_ROPE, ROPE_CHUNK):
                    rope_tile = attn_rope_stage[
                        rope_slice_head_row : rope_slice_head_row + H,
                        2 * rope_slice_r0 : 2 * rope_slice_r0 + ROPE_INTERLEAVE_CHUNK,
                    ]
                    even_chunk = pl.matmul(rope_tile, even_select_local, out_dtype=pl.FP32)
                    odd_chunk = pl.matmul(rope_tile, odd_select_local, out_dtype=pl.FP32)
                    o_proj_even = pl.assemble(o_proj_even, even_chunk, [rope_slice_head_row, rope_slice_r0])
                    o_proj_odd = pl.assemble(o_proj_odd, odd_chunk, [rope_slice_head_row, rope_slice_r0])

    # Stage 3: inverse RoPE on the rope slice of the attention output by
    # rotating the even/odd halves, then reinterleaving into packed rows.
    for rope_apply_t0 in pl.parallel(0, T, ROPE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_rope_apply_tile"):
            for rope_apply_dt in pl.range(ROPE_TOKEN_TILE):
                rope_apply_t = rope_apply_t0 + rope_apply_dt
                rope_apply_head_row = rope_apply_t * H

                # Apply inverse rotary mixing with this token's cos/sin tables.
                cos_tile = pl.cast(freqs_cos[rope_apply_t : rope_apply_t + 1, 0 : HALF_ROPE], target_type=pl.FP32)
                sin_tile = pl.cast(freqs_sin[rope_apply_t : rope_apply_t + 1, 0 : HALF_ROPE], target_type=pl.FP32)
                even_tile = o_proj_even[rope_apply_head_row : rope_apply_head_row + H, :]
                odd_tile = o_proj_odd[rope_apply_head_row : rope_apply_head_row + H, :]
                rope_even_acc = pl.add(
                    pl.col_expand_mul(even_tile, cos_tile),
                    pl.col_expand_mul(odd_tile, sin_tile),
                )
                rope_odd_acc = pl.sub(
                    pl.col_expand_mul(odd_tile, cos_tile),
                    pl.col_expand_mul(even_tile, sin_tile),
                )
                rope_even = pl.assemble(
                    rope_even,
                    pl.cast(rope_even_acc, target_type=pl.BF16, mode="rint"),
                    [rope_apply_head_row, 0],
                )
                rope_odd = pl.assemble(
                    rope_odd,
                    pl.cast(rope_odd_acc, target_type=pl.BF16, mode="rint"),
                    [rope_apply_head_row, 0],
                )

    for rope_asm_t0 in pl.parallel(0, T, ROPE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_rope_assemble_matmul_tile"):
            for rope_asm_dt in pl.range(ROPE_TOKEN_TILE):
                rope_asm_t = rope_asm_t0 + rope_asm_dt
                rope_asm_head_row = rope_asm_t * H

                # Reinterleave the rotated even/odd halves back to rope lane order.
                for rope_asm_r0 in pl.range(0, HALF_ROPE, ROPE_CHUNK):
                    rope_even_chunk = rope_even[rope_asm_head_row : rope_asm_head_row + H, rope_asm_r0 : rope_asm_r0 + ROPE_CHUNK]
                    rope_odd_chunk = rope_odd[rope_asm_head_row : rope_asm_head_row + H, rope_asm_r0 : rope_asm_r0 + ROPE_CHUNK]
                    rope_even_interleave = pl.matmul(
                        rope_even_chunk,
                        even_select_local,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    rope_odd_interleave = pl.matmul(
                        rope_odd_chunk,
                        odd_select_local,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    rope_even_interleave_buf = pl.assemble(
                        rope_even_interleave_buf,
                        rope_even_interleave,
                        [rope_asm_head_row, 2 * rope_asm_r0],
                    )
                    rope_odd_interleave_buf = pl.assemble(
                        rope_odd_interleave_buf,
                        rope_odd_interleave,
                        [rope_asm_head_row, 2 * rope_asm_r0],
                    )

    for rope_combine_t0 in pl.parallel(0, T, ROPE_PACK_TOKEN_TILE):
        for rope_pack_g0 in pl.parallel(0, O_GROUPS, ROPE_PACK_GROUP_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_rope_pack_group_tile"):
                for rope_combine_dt in pl.range(ROPE_PACK_TOKEN_TILE):
                    rope_combine_t = rope_combine_t0 + rope_combine_dt
                    for rope_pack_dg in pl.range(ROPE_PACK_GROUP_TILE):
                        rope_pack_g = rope_pack_g0 + rope_pack_dg
                        rope_pack_head_row = rope_combine_t * H + rope_pack_g * HEADS_PER_GROUP

                        # Merge and write only this group's inverse-RoPE tail.
                        rope_even_tile = rope_even_interleave_buf[
                            rope_pack_head_row : rope_pack_head_row + HEADS_PER_GROUP,
                            0 : ROPE_DIM,
                        ]
                        rope_odd_tile = rope_odd_interleave_buf[
                            rope_pack_head_row : rope_pack_head_row + HEADS_PER_GROUP,
                            0 : ROPE_DIM,
                        ]
                        rope_full = pl.cast(
                            pl.add(rope_even_tile, rope_odd_tile),
                            target_type=pl.BF16,
                        )
                        rope_pack_row = rope_pack_g * T + rope_combine_t
                        for rope_pack_hh in pl.range(HEADS_PER_GROUP):
                            rope_pack_head_col = rope_pack_hh * HEAD_DIM + NOPE_DIM
                            o_packed = pl.assemble(
                                o_packed,
                                rope_full[rope_pack_hh : rope_pack_hh + 1, 0:ROPE_DIM],
                                [rope_pack_row, rope_pack_head_col],
                            )

    o_r = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_i8 = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)

    # Stage 5: grouped BF16 projection `o_packed @ wo_a^T`, producing the
    # low-rank intermediate activation `o_r`.
    for g in pl.parallel(0, O_GROUPS, 1):
        row_base_o = g * T
        out_col_g = g * O_LORA

        for nb in pl.parallel(0, A_N_BLOCKS, 1):
            n0 = nb * A_N_CHUNK

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_a_accum"):
                # K-split BF16 matmul for one wo_a output tile.
                xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_CHUNK], [row_base_o, 0], valid_shape=[T, A_K_CHUNK])
                wa0_chunk = wo_a[g:g + 1, n0:n0 + A_N_CHUNK, 0:A_K_CHUNK]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, A_K_BLOCKS, stage=2):
                    k0 = kb * A_K_CHUNK
                    xa_k_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_CHUNK], [row_base_o, k0], valid_shape=[T, A_K_CHUNK])
                    wa_k_chunk = wo_a[g:g + 1, n0:n0 + A_N_CHUNK, k0:k0 + A_K_CHUNK]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_a_store"):
                # Store this projection tile back as BF16 activations.
                acc_a_2d = pl.reshape(acc_a, [MM_T_TILE, A_N_CHUNK])
                o_r[0:T_PAD, out_col_g + n0:out_col_g + n0 + A_N_CHUNK] = pl.cast(
                    acc_a_2d,
                    target_type=pl.BF16,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_quant"):
        # Stage 6: per-row symmetric INT8 quantization of `o_r` for the W8A8C16
        # second projection stage.
        or_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
            or_a_f32 = pl.cast(o_r[0:T, k0:k0 + QUANT_CHUNK], target_type=pl.FP32)
            or_a_abs = pl.maximum(or_a_f32, pl.neg(or_a_f32))
            or_a_max = pl.reshape(pl.row_max(or_a_abs), [1, T])
            or_amax = pl.maximum(or_amax, or_a_max)
        or_sq_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
        o_r_scale_dq = pl.reshape(pl.recip(or_sq_row), [T, 1])
        or_sq_col = pl.reshape(or_sq_row, [T, 1])
        for k1 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
            or_q_f32 = pl.cast(o_r[0:T, k1:k1 + QUANT_CHUNK], target_type=pl.FP32)
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[0:T, k1:k1 + QUANT_CHUNK] = pl.cast(or_q_half, target_type=pl.INT8, mode="trunc")

    # Stage 7: INT8 projection `o_r_i8 @ wo_b^T`, then dequantize with the
    # activation and weight scales into the final BF16 output.
    for nb in pl.parallel(0, B_N_BLOCKS, 1):
        n0 = nb * B_N_CHUNK

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_accum"):
            # K-split INT8 GEMM for one output-channel tile.
            xb0_chunk = o_r_i8[0:T_PAD, 0:B_K_CHUNK]
            wb0_chunk = wo_b[n0:n0 + B_N_CHUNK, 0:B_K_CHUNK]
            acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
            for kb in pl.pipeline(1, B_K_BLOCKS, stage=2):
                k0 = kb * B_K_CHUNK
                xb_k_chunk = o_r_i8[0:T_PAD, k0:k0 + B_K_CHUNK]
                wb_k_chunk = wo_b[n0:n0 + B_N_CHUNK, k0:k0 + B_K_CHUNK]
                acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_store"):
            # Apply the per-row and per-channel dequant scales before casting to BF16.
            wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + B_N_CHUNK], [1, B_N_CHUNK])
            attn_chunk = pl.cast(acc_b, target_type=pl.FP32, mode="none")
            o_r_scale_pad = pl.slice(o_r_scale_dq, [MM_T_TILE, 1], [0, 0], valid_shape=[T, 1])
            attn_chunk = pl.col_expand_mul(pl.row_expand_mul(attn_chunk, o_r_scale_pad), wb_scale_chunk)
            attn_out[0:T, n0:n0 + B_N_CHUNK] = pl.cast(attn_chunk[0:T, 0:B_N_CHUNK], target_type=pl.BF16, mode="rint")

    return attn_out


@pl.jit
def sparse_attn_test(
    q:                  pl.Tensor[[T, H, HEAD_DIM],                               pl.BF16],
    ori_kv:             pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    ori_block_table:    pl.Tensor[[B, ORI_MAX_BLOCKS],                            pl.INT32],
    cmp_kv:             pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    cmp_block_table:    pl.Tensor[[B, CMP_MAX_BLOCKS],                            pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK],                                      pl.INT32],
    attn_sink:          pl.Tensor[[H],                                            pl.FP32],
    seqused_kv:         pl.Tensor[[B],                                            pl.INT32],
    freqs_cos:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    freqs_sin:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    even_select_local:  pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],            pl.BF16],
    odd_select_local:   pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],            pl.BF16],
    wo_a:               pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],                 pl.BF16],
    wo_b:               pl.Tensor[[D, O_GROUPS * O_LORA],                         pl.INT8],
    wo_b_scale:         pl.Tensor[[D],                                            pl.FP32],
    attn_out:           pl.Out[pl.Tensor[[T, D],                                  pl.BF16]],
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
        even_select_local,
        odd_select_local,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    return attn_out

