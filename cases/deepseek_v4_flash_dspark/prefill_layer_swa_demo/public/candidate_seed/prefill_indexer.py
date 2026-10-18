# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""REF closure for deepseek_v4_flash_dspark.prefill_layer_swa_demo; fixed bindings from public config."""
from config import BLOCK_SIZE
from config import D
from config import FP32_NEG_INF
from config import IDX_CACHE_MAX_BLOCKS
from config import IDX_HEAD_DIM
from config import IDX_NOPE_HEAD_DIM
from config import IDX_N_HEADS
from config import IDX_TOPK
from config import INT8_AMAX_EPS
from config import INT8_SCALE_MAX
from config import Q_LORA
from config import ROPE_HEAD_DIM
from config import WEIGHTS_SCALE
import pypto.language as pl

PREFILL_DENSE_TILE = 512
T_DYN = pl.dynamic("PREFILL_INDEXER_T_DYN")
Q_T_DYN = pl.dynamic("PREFILL_INDEXER_Q_T_DYN")
IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
COMPRESS_RATIO = 4
RAW_TOKEN_CAP = 8192
INDEXER_SCORE_CAP = RAW_TOKEN_CAP // COMPRESS_RATIO
INDEXER_TOPK_CAP = min(IDX_TOPK, INDEXER_SCORE_CAP)
CACHE_TILE = 32
INDEXER_SCORE_BLOCKS = max(1, (INDEXER_SCORE_CAP + CACHE_TILE - 1) // CACHE_TILE)
SCORE_TOKEN_TILE = 8
Q_TILE = 128
Q_OUT_TILE = 256
QR_PROJ_MM_ROW_TILE = min(128, PREFILL_DENSE_TILE)
QR_PROJ_TAIL_ROW_TILE = 16
QR_PROJ_ROW_TILE = 16
HEAD_DIM_TILE = 32
D_TILE = 32
WEIGHTS_ROW_TILE = 32
QH_QUANT_ROW_TILE = 64
ROPE_ROW_TILE = IDX_N_HEADS
QH_MM_TILE = 64
WEIGHTS_TAIL_ROW_TILE = 16
SORT_LEN = 2048
PREFILL_TOPK_CAP = INDEXER_TOPK_CAP
TOPK_PAIR_WIDTH = 2 * PREFILL_TOPK_CAP

@pl.jit.inline(auto_scope=False)
def _prefill_indexer_dense_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[T_DYN, INDEXER_SCORE_CAP], pl.FP32]],
    cmp_topk_indices: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    rope_dup_idx_template: pl.Tensor[[1, ROPE_HEAD_DIM], pl.INT32],
    rope_swap_idx_template: pl.Tensor[[ROPE_ROW_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_sign_template: pl.Tensor[[1, ROPE_HEAD_DIM], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    max_visible: pl.Scalar[pl.INDEX],
):
    """Project and select one bounded PREFILL_DENSE_TILE-row dense tile."""
    t_dim = pl.tensor.dim(x, 0)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    kv_cache_i8_flat = pl.reshape(
        idx_kv_cache,
        [idx_block_num * BLOCK_SIZE, IDX_HEAD_DIM],
    )
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * BLOCK_SIZE, 1])
    x_view = pl.reshape(x, [t_dim, D])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])

    # Q projection scratch is bounded by PREFILL_DENSE_TILE. Full 128-row cube
    # blocks keep the dense path; the final incomplete block uses the proven
    # 16-row partial-M shape.
    qr_proj = pl.create_tensor([PREFILL_DENSE_TILE, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    qr_full_rows = (tile_rows // QR_PROJ_MM_ROW_TILE) * QR_PROJ_MM_ROW_TILE
    qr_padded_rows = (
        (tile_rows + QR_PROJ_TAIL_ROW_TILE - 1) // QR_PROJ_TAIL_ROW_TILE
    ) * QR_PROJ_TAIL_ROW_TILE
    qr_full_row_blocks = PREFILL_DENSE_TILE // QR_PROJ_MM_ROW_TILE
    if qr_full_rows > 0:
        for idx in pl.spmd(
            (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE) * qr_full_row_blocks,
            name_hint="prefill_idx_qr_proj_full",
        ):
            qr_n = idx // qr_full_row_blocks
            qr_row_block = idx - qr_n * qr_full_row_blocks
            o0 = qr_n * Q_OUT_TILE
            local_t0 = qr_row_block * QR_PROJ_MM_ROW_TILE
            if local_t0 < qr_full_rows:
                global_t0 = tile_base + local_t0
                qr_acc = pl.create_tensor([QR_PROJ_MM_ROW_TILE, Q_OUT_TILE], dtype=pl.INT32)
                for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                    q0 = kb * Q_TILE
                    qr_tile = qr_view[
                        global_t0 : global_t0 + QR_PROJ_MM_ROW_TILE,
                        q0 : q0 + Q_TILE,
                    ]
                    wq_tile = wq_b[q0 : q0 + Q_TILE, o0 : o0 + Q_OUT_TILE]
                    if q0 == 0:
                        qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                    else:
                        qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
                wq_scale = pl.reshape(wq_b_scale[o0 : o0 + Q_OUT_TILE], [1, Q_OUT_TILE])
                for rl in pl.range(0, QR_PROJ_MM_ROW_TILE, QR_PROJ_ROW_TILE):
                    acc_fp32 = pl.cast(
                        qr_acc[rl : rl + QR_PROJ_ROW_TILE, :],
                        target_type=pl.FP32,
                        mode="none",
                    )
                    scale_dq = qr_scale_view[
                        global_t0 + rl : global_t0 + rl + QR_PROJ_ROW_TILE,
                        :,
                    ]
                    qr_dequant = pl.col_expand_mul(
                        pl.row_expand_mul(acc_fp32, scale_dq),
                        wq_scale,
                    )
                    qr_proj[
                        local_t0 + rl : local_t0 + rl + QR_PROJ_ROW_TILE,
                        o0 : o0 + Q_OUT_TILE,
                    ] = qr_dequant

    if qr_full_rows < qr_padded_rows:
        qr_tail_row_blocks = QR_PROJ_MM_ROW_TILE // QR_PROJ_TAIL_ROW_TILE
        for tail_idx in pl.spmd(
            (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE) * qr_tail_row_blocks,
            name_hint="prefill_idx_qr_proj_tail",
        ):
            tail_n = tail_idx // qr_tail_row_blocks
            tail_row_block = tail_idx - tail_n * qr_tail_row_blocks
            tail_o0 = tail_n * Q_OUT_TILE
            tail_t0 = qr_full_rows + tail_row_block * QR_PROJ_TAIL_ROW_TILE
            if tail_t0 < qr_padded_rows:
                tail_valid_rows = pl.min(QR_PROJ_TAIL_ROW_TILE, tile_rows - tail_t0)
                tail_global_t0 = tile_base + tail_t0
                qr_tail_acc = pl.create_tensor([QR_PROJ_TAIL_ROW_TILE, Q_OUT_TILE], dtype=pl.INT32)
                for tail_kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                    tail_q0 = tail_kb * Q_TILE
                    qr_tail = pl.slice(
                        qr_view,
                        [QR_PROJ_TAIL_ROW_TILE, Q_TILE],
                        [tail_global_t0, tail_q0],
                        valid_shape=[tail_valid_rows, Q_TILE],
                    )
                    wq_tail = wq_b[
                        tail_q0 : tail_q0 + Q_TILE,
                        tail_o0 : tail_o0 + Q_OUT_TILE,
                    ]
                    if tail_q0 == 0:
                        qr_tail_acc = pl.matmul(qr_tail, wq_tail, out_dtype=pl.INT32)
                    else:
                        qr_tail_acc = pl.matmul_acc(qr_tail_acc, qr_tail, wq_tail)
                tail_acc_fp32 = pl.cast(qr_tail_acc, target_type=pl.FP32, mode="none")
                tail_scale = pl.slice(
                    qr_scale_view,
                    [QR_PROJ_TAIL_ROW_TILE, 1],
                    [tail_global_t0, 0],
                    valid_shape=[tail_valid_rows, 1],
                )
                tail_wq_scale = pl.reshape(
                    wq_b_scale[tail_o0 : tail_o0 + Q_OUT_TILE],
                    [1, Q_OUT_TILE],
                )
                qr_tail_dequant = pl.col_expand_mul(
                    pl.row_expand_mul(tail_acc_fp32, tail_scale),
                    tail_wq_scale,
                )
                qr_proj[
                    tail_t0 : tail_t0 + QR_PROJ_TAIL_ROW_TILE,
                    tail_o0 : tail_o0 + Q_OUT_TILE,
                ] = qr_tail_dequant

    # Q RoPE -> Hadamard -> per-row INT8. All intermediates are local to this
    # 2048-token tile instead of scaling with the physical request length.
    qr_proj_flat = pl.reshape(qr_proj, [PREFILL_DENSE_TILE * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([PREFILL_DENSE_TILE * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    for token_idx in pl.spmd(tile_rows, name_hint="prefill_idx_qr_rope", allow_early_resolve=True):
        rope_global_t = tile_base + token_idx
        r0 = token_idx * ROPE_ROW_TILE
        qr_nope_fp32 = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, 0:IDX_NOPE_HEAD_DIM]
        qr_nope = pl.cast(qr_nope_fp32, target_type=pl.BF16, mode="rint")
        qr_rope = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope, dim=-1, index=rope_swap_idx_template)
        cos_il = pl.gather(
            cos[rope_global_t : rope_global_t + 1, 0 : ROPE_HEAD_DIM // 2],
            dim=-1,
            index=rope_dup_idx_template,
        )
        sin_il = pl.gather(
            sin[rope_global_t : rope_global_t + 1, 0 : ROPE_HEAD_DIM // 2],
            dim=-1,
            index=rope_dup_idx_template,
        )
        rope_rot = pl.add(
            pl.col_expand_mul(qr_rope, cos_il),
            pl.col_expand_mul(qr_swapped, pl.mul(sin_il, rope_sign_template)),
        )
        rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
        qr_bf16[r0 : r0 + ROPE_ROW_TILE, :] = pl.concat(qr_nope, rope_bf16)

    qh_acc_gm = pl.create_tensor([PREFILL_DENSE_TILE * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for mm_idx in pl.spmd(tile_rows, name_hint="prefill_idx_qr_hadamard", allow_early_resolve=True):
        r0 = mm_idx * QH_MM_TILE
        qh_acc = pl.matmul(
            qr_bf16[r0 : r0 + QH_MM_TILE, :],
            hadamard,
            out_dtype=pl.FP32,
        )
        qh_acc_gm[r0 : r0 + QH_MM_TILE, :] = qh_acc

    qr_hadamard_i8 = pl.create_tensor(
        [PREFILL_DENSE_TILE * IDX_N_HEADS, IDX_HEAD_DIM],
        dtype=pl.INT8,
    )
    qr_hadamard_scale_dq = pl.create_tensor(
        [PREFILL_DENSE_TILE * IDX_N_HEADS, 1],
        dtype=pl.FP32,
    )
    for quant_idx in pl.spmd(tile_rows, name_hint="prefill_idx_qr_quant", allow_early_resolve=True):
        r0 = quant_idx * QH_QUANT_ROW_TILE
        qh_amax = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_tile = qh_acc_gm[
                r0 : r0 + QH_QUANT_ROW_TILE,
                h0 : h0 + HEAD_DIM_TILE,
            ]
            qh_abs = pl.maximum(qh_tile, pl.neg(qh_tile))
            qh_amax = pl.maximum(
                qh_amax,
                pl.reshape(pl.row_max(qh_abs), [1, QH_QUANT_ROW_TILE]),
            )
        scale_quant_row = pl.div(
            pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX),
            qh_amax,
        )
        qr_hadamard_scale_dq[r0 : r0 + QH_QUANT_ROW_TILE, :] = pl.reshape(
            pl.recip(scale_quant_row),
            [QH_QUANT_ROW_TILE, 1],
        )
        scale_quant = pl.reshape(scale_quant_row, [QH_QUANT_ROW_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_quant_tile = qh_acc_gm[
                r0 : r0 + QH_QUANT_ROW_TILE,
                h1 : h1 + HEAD_DIM_TILE,
            ]
            qh_scaled = pl.row_expand_mul(qh_quant_tile, scale_quant)
            qh_i32 = pl.cast(qh_scaled, target_type=pl.INT32, mode="rint")
            qh_half = pl.cast(qh_i32, target_type=pl.FP16, mode="round")
            qr_hadamard_i8[
                r0 : r0 + QH_QUANT_ROW_TILE,
                h1 : h1 + HEAD_DIM_TILE,
            ] = pl.cast(qh_half, target_type=pl.INT8, mode="trunc")

    # Per-head weight projection: dense M32, with only the final incomplete
    # block lowered through M16.
    weights = pl.create_tensor([PREFILL_DENSE_TILE, IDX_N_HEADS], dtype=pl.FP32)
    weights_full_rows = (tile_rows // WEIGHTS_ROW_TILE) * WEIGHTS_ROW_TILE
    weights_padded_rows = (
        (tile_rows + WEIGHTS_TAIL_ROW_TILE - 1) // WEIGHTS_TAIL_ROW_TILE
    ) * WEIGHTS_TAIL_ROW_TILE
    for weights_idx in pl.spmd(
        PREFILL_DENSE_TILE // WEIGHTS_ROW_TILE,
        name_hint="prefill_idx_weights_proj_full",
    ):
        weights_t0 = weights_idx * WEIGHTS_ROW_TILE
        if weights_t0 < weights_full_rows:
            weights_global_t0 = tile_base + weights_t0
            weights_acc = pl.create_tensor([WEIGHTS_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
            for db in pl.pipeline(0, D // D_TILE, stage=2):
                d0 = db * D_TILE
                x_tile = x_view[
                    weights_global_t0 : weights_global_t0 + WEIGHTS_ROW_TILE,
                    d0 : d0 + D_TILE,
                ]
                wp_tile = weights_proj[d0 : d0 + D_TILE, :]
                if d0 == 0:
                    weights_acc = pl.matmul(x_tile, wp_tile, out_dtype=pl.FP32)
                else:
                    weights_acc = pl.matmul_acc(weights_acc, x_tile, wp_tile)
            weights[weights_t0 : weights_t0 + WEIGHTS_ROW_TILE, :] = pl.mul(
                weights_acc,
                WEIGHTS_SCALE,
            )

    if weights_full_rows < weights_padded_rows:
        for weights_tail_idx in pl.spmd(
            WEIGHTS_ROW_TILE // WEIGHTS_TAIL_ROW_TILE,
            name_hint="prefill_idx_weights_proj_tail",
        ):
            weights_tail_t0 = weights_full_rows + weights_tail_idx * WEIGHTS_TAIL_ROW_TILE
            if weights_tail_t0 < weights_padded_rows:
                weights_tail_valid = pl.min(
                    WEIGHTS_TAIL_ROW_TILE,
                    tile_rows - weights_tail_t0,
                )
                weights_tail_global_t0 = tile_base + weights_tail_t0
                weights_tail_acc = pl.create_tensor(
                    [WEIGHTS_TAIL_ROW_TILE, IDX_N_HEADS],
                    dtype=pl.FP32,
                )
                for tail_db in pl.pipeline(0, D // D_TILE, stage=2):
                    tail_d0 = tail_db * D_TILE
                    x_tail = pl.slice(
                        x_view,
                        [WEIGHTS_TAIL_ROW_TILE, D_TILE],
                        [weights_tail_global_t0, tail_d0],
                        valid_shape=[weights_tail_valid, D_TILE],
                    )
                    wp_tail = weights_proj[tail_d0 : tail_d0 + D_TILE, :]
                    if tail_d0 == 0:
                        weights_tail_acc = pl.matmul(x_tail, wp_tail, out_dtype=pl.FP32)
                    else:
                        weights_tail_acc = pl.matmul_acc(weights_tail_acc, x_tail, wp_tail)
                weights[
                    weights_tail_t0 : weights_tail_t0 + WEIGHTS_TAIL_ROW_TILE,
                    :,
                ] = pl.mul(weights_tail_acc, WEIGHTS_SCALE)

    # Query selection is explicitly wave-bounded. Score is initialized and
    # written directly in the public output, so there is no duplicate
    # [physical_T, 2048] sort scratch.
    wave_count = (tile_rows + Q_TILE - 1) // Q_TILE
    wave_blocks = wave_count * Q_TILE
    for init_idx in pl.spmd(
        wave_blocks,
        name_hint="prefill_idx_score_init",
        allow_early_resolve=True,
    ):
        wave_idx = init_idx // Q_TILE
        wave_local = init_idx - wave_idx * Q_TILE
        wave_base = wave_idx * Q_TILE
        wave_rows = pl.min(Q_TILE, tile_rows - wave_base)
        if wave_local < wave_rows:
            init_global_t = tile_base + wave_base + wave_local
            score[init_global_t : init_global_t + 1, 0:SORT_LEN] = pl.full(
                [1, SORT_LEN],
                dtype=pl.FP32,
                value=FP32_NEG_INF,
            )

    score_groups_per_wave = Q_TILE // SCORE_TOKEN_TILE
    for score_idx in pl.spmd(
        wave_count * score_groups_per_wave,
        name_hint="prefill_idx_score",
    ):
        score_wave_idx = score_idx // score_groups_per_wave
        score_group_in_wave = score_idx - score_wave_idx * score_groups_per_wave
        score_wave_base = score_wave_idx * Q_TILE
        score_wave_rows = pl.min(Q_TILE, tile_rows - score_wave_base)
        wave_last_t = tile_base + score_wave_base + score_wave_rows - 1
        wave_last_pos = pl.read(position_ids, [wave_last_t])
        wave_max_visible = pl.min(
            max_visible,
            pl.cast((wave_last_pos + 1) // COMPRESS_RATIO, pl.INDEX),
        )
        wave_token0 = score_group_in_wave * SCORE_TOKEN_TILE
        if wave_token0 < score_wave_rows:
            for cb in pl.range(INDEXER_SCORE_BLOCKS):
                cache0 = cb * CACHE_TILE
                logical_block = cache0 // BLOCK_SIZE
                if wave_max_visible > cache0 and logical_block < IDX_CACHE_MAX_BLOCKS:
                    idx_blk_id_raw = pl.read(idx_block_table, [logical_block])
                    if idx_blk_id_raw >= 0 and idx_blk_id_raw < idx_block_num:
                        idx_blk_id = pl.cast(idx_blk_id_raw, pl.INDEX)
                        kv_row0 = idx_blk_id * BLOCK_SIZE + (cache0 % BLOCK_SIZE)
                        kv_q_i8_full = kv_cache_i8_flat[
                            kv_row0 : kv_row0 + CACHE_TILE,
                            0:IDX_HEAD_DIM,
                        ]
                        kv_cache_scale_dq = kv_scale_flat[
                            kv_row0 : kv_row0 + CACHE_TILE,
                            :,
                        ]
                        for token_offset in pl.range(SCORE_TOKEN_TILE):
                            wave_token = wave_token0 + token_offset
                            if wave_token < score_wave_rows:
                                local_t = score_wave_base + wave_token
                                score_global_t = tile_base + local_t
                                q_s0 = local_t * IDX_N_HEADS
                                qr_hadamard_tile = qr_hadamard_i8[
                                    q_s0 : q_s0 + IDX_N_HEADS,
                                    0:IDX_HEAD_DIM,
                                ]
                                score_acc = pl.matmul(
                                    kv_q_i8_full,
                                    qr_hadamard_tile,
                                    out_dtype=pl.INT32,
                                    b_trans=True,
                                )
                                qh_scale = pl.reshape(
                                    qr_hadamard_scale_dq[
                                        q_s0 : q_s0 + IDX_N_HEADS,
                                        :,
                                    ],
                                    [1, IDX_N_HEADS],
                                )
                                score_tile = pl.cast(
                                    score_acc,
                                    target_type=pl.FP32,
                                    mode="none",
                                )
                                score_tile = pl.col_expand_mul(
                                    pl.row_expand_mul(score_tile, kv_cache_scale_dq),
                                    qh_scale,
                                )
                                relu_score = pl.maximum(
                                    score_tile,
                                    pl.mul(score_tile, 0.0),
                                )
                                weighted_score = pl.reshape(
                                    pl.row_sum(
                                        pl.col_expand_mul(
                                            relu_score,
                                            weights[local_t : local_t + 1, :],
                                        )
                                    ),
                                    [1, CACHE_TILE],
                                )
                                pos = pl.read(position_ids, [score_global_t])
                                visible_t = pl.min(
                                    (pos + 1) // COMPRESS_RATIO,
                                    INDEXER_SCORE_CAP,
                                )
                                if visible_t > cache0:
                                    valid_len = pl.min(CACHE_TILE, visible_t - cache0)
                                    weighted_valid = pl.fillpad(
                                        pl.set_validshape(weighted_score, 1, valid_len),
                                        pad_value=pl.PadValue.min,
                                    )
                                    score[
                                        score_global_t : score_global_t + 1,
                                        cache0 : cache0 + CACHE_TILE,
                                    ] = pl.maximum(
                                        weighted_valid,
                                        pl.full(
                                            [1, CACHE_TILE],
                                            dtype=pl.FP32,
                                            value=FP32_NEG_INF,
                                        ),
                                    )

    for topk_idx in pl.spmd(
        wave_blocks,
        name_hint="prefill_idx_topk",
        allow_early_resolve=True,
    ):
        topk_wave_idx = topk_idx // Q_TILE
        topk_wave_local = topk_idx - topk_wave_idx * Q_TILE
        topk_wave_base = topk_wave_idx * Q_TILE
        topk_wave_rows = pl.min(Q_TILE, tile_rows - topk_wave_base)
        if topk_wave_local < topk_wave_rows:
            local_t = topk_wave_base + topk_wave_local
            topk_global_t = tile_base + local_t
            cmp_topk_indices[topk_global_t : topk_global_t + 1, 0:IDX_TOPK] = pl.full(
                [1, IDX_TOPK],
                dtype=pl.INT32,
                value=-1,
            )
            pos = pl.read(position_ids, [topk_global_t])
            visible_t = pl.min((pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
            if visible_t > 0:
                score_row = score[topk_global_t : topk_global_t + 1, 0:SORT_LEN]
                idx_init = pl.arange(0, [1, SORT_LEN], dtype=pl.UINT32)
                sorted_tile = pl.sort32(score_row, idx_init)
                sorted_tile = pl.mrgsort(sorted_tile, block_len=64)
                sorted_tile = pl.mrgsort(sorted_tile, block_len=256)
                sorted_tile = pl.mrgsort(sorted_tile, block_len=1024)
                topk_pairs = sorted_tile[:, 0:TOPK_PAIR_WIDTH]
                topk_idxs_tile = pl.gather(
                    topk_pairs,
                    mask_pattern=pl.tile.MaskPattern.P1010,
                    output_dtype=pl.INT32,
                )
                valid_topk = pl.min(PREFILL_TOPK_CAP, visible_t)
                cmp_topk_indices[
                    topk_global_t : topk_global_t + 1,
                    0:PREFILL_TOPK_CAP,
                ] = pl.set_validshape(topk_idxs_tile, 1, valid_topk)


@pl.jit.inline(auto_scope=False)
def prefill_indexer_query(
    x: pl.Tensor[[Q_T_DYN, D], pl.BF16],
    qr: pl.Tensor[[Q_T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[Q_T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[Q_T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[Q_T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[Q_T_DYN, INDEXER_SCORE_CAP], pl.FP32]],
    cmp_topk_indices: pl.Out[pl.Tensor[[Q_T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[Q_T_DYN], pl.INT32],
):
    """Score this rank's queries against the indexer cache and select their top-k.

    Query-side half of the indexer: it reads the cache but never writes it, so
    under context parallelism it runs on the local slice while
    `prefill_indexer_compressor` fills the same cache from the gathered stream.
    Its token axis is a separate symbol for exactly that reason -- the two
    halves carry different extents in one program.
    """
    t_dim = pl.tensor.dim(x, 0)

    last_pos = pl.read(position_ids, [t_dim - 1])
    max_visible = pl.cast(
        pl.min((last_pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP),
        pl.INDEX,
    )

    # Head-invariant RoPE templates are created once per invocation, outside
    # the dense tile loop.
    rope_dup_idx_template = pl.create_tensor([1, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_swap_idx_template = pl.create_tensor([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_sign_template = pl.create_tensor([1, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_rope_index_prepare"):
        for rope_c in pl.range(ROPE_HEAD_DIM):
            rope_lane = rope_c % 2
            pl.write(rope_dup_idx_template, [0, rope_c], pl.cast(rope_c // 2, pl.INT32))
            pl.write(
                rope_sign_template,
                [0, rope_c],
                pl.cast(pl.cast(rope_lane * 2 - 1, pl.INT32), pl.FP32),
            )
            for rope_r in pl.range(ROPE_ROW_TILE):
                pl.write(
                    rope_swap_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c + 1 - rope_lane * 2, pl.INT32),
                )

    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        # The full-forward layer entries place scopes explicitly.  Keep each
        # bounded dense tile independently reclaimable there while preserving
        # the compiler-owned loop scope of standalone auto-scoped entries.
        with pl.scope():
            tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
            _prefill_indexer_dense_tile(
                x,
                qr,
                qr_scale,
                wq_b,
                wq_b_scale,
                weights_proj,
                cos,
                sin,
                hadamard,
                idx_kv_cache,
                idx_kv_scale,
                idx_block_table,
                score,
                cmp_topk_indices,
                position_ids,
                rope_dup_idx_template,
                rope_swap_idx_template,
                rope_sign_template,
                tile_base,
                tile_rows,
                max_visible,
            )

    return score, cmp_topk_indices


