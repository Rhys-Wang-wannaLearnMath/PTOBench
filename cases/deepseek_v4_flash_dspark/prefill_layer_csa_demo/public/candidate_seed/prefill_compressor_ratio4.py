# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""REF closure for deepseek_v4_flash_dspark.prefill_layer_csa_demo; fixed bindings from public config."""
from config import BLOCK_SIZE
from config import CSA_COMPRESS_RATIO as COMPRESS_RATIO
from config import CSA_COMPRESS_STATE_DIM as COMPRESS_STATE_DIM
from config import CSA_MAIN_OUT_DIM as OUT_DIM
from config import CSA_STATE_BLOCK_SIZE
from config import CSA_STATE_LEN as STATE_LEN
from config import CSA_STATE_MAX_BLOCKS
from config import D
from config import EPS
from config import FP32_NEG_INF
from config import HEAD_DIM
from config import MAX_SEQ_LEN
from config import NOPE_HEAD_DIM
from config import ROPE_HEAD_DIM
import pypto.language as pl

PREFILL_STATE_TILE = 512
T_DYN = pl.dynamic("PREFILL_CSA_C4_T_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
HEAD_DIM_INV = 1.0 / HEAD_DIM
MAX_CMP_WRITES = PREFILL_STATE_TILE // COMPRESS_RATIO
K_TILE = 512
OUT_TILE = 64
PROJ_ROW_TILE = 128
PROJ_TAIL_ROW_TILE = 16
PROJ_FULL_ROW_BLOCKS = PREFILL_STATE_TILE // PROJ_ROW_TILE
PROJ_TAIL_ROW_BLOCKS = PROJ_ROW_TILE // PROJ_TAIL_ROW_TILE
HEAD_D_TILE = 512
HEAD_TILE = 64
PACKED_RMS_TILE = 16

@pl.jit.inline(auto_scope=False)
def _prefill_compressor_ratio4_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    rope_dup_idx_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_swap_idx_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_sign_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.FP32],
    state_order_fence: pl.InOut[pl.Tensor[[1], pl.INT32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Compress one ordered physical token tile, retaining only 512-row scratch."""
    state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    kv_proj_scratch = pl.create_tensor([PREFILL_STATE_TILE, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([PREFILL_STATE_TILE, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(
        compress_state,
        [state_block_num * CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
    )
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])

    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, D])
    proj_full_rows = (tile_rows // PROJ_ROW_TILE) * PROJ_ROW_TILE
    proj_padded_rows = ((tile_rows + PROJ_TAIL_ROW_TILE - 1) // PROJ_TAIL_ROW_TILE) * PROJ_TAIL_ROW_TILE

    # Complete 128-row cube blocks; a partially valid M=128 is not reliable on A3.
    if proj_full_rows > 0:
        for proj_idx in pl.spmd(
            (OUT_DIM // OUT_TILE) * PROJ_FULL_ROW_BLOCKS,
            name_hint="prefill_c4_kv_score_proj_full",
        ):
            proj_n = proj_idx // PROJ_FULL_ROW_BLOCKS
            proj_row_block = proj_idx - proj_n * PROJ_FULL_ROW_BLOCKS
            o0 = proj_n * OUT_TILE
            local_t0 = proj_row_block * PROJ_ROW_TILE
            if local_t0 < proj_full_rows:
                global_t0 = tile_base + local_t0
                kv_acc = pl.create_tensor([PROJ_ROW_TILE, OUT_TILE], dtype=pl.FP32)
                score_acc = pl.create_tensor([PROJ_ROW_TILE, OUT_TILE], dtype=pl.FP32)
                for kb in pl.pipeline(0, D // K_TILE, stage=2):
                    k0 = kb * K_TILE
                    x_tile = x_flat[
                        global_t0 : global_t0 + PROJ_ROW_TILE,
                        k0 : k0 + K_TILE,
                    ]
                    wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                    wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                    if k0 == 0:
                        kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                        score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
                    else:
                        kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                        score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
                kv_proj_scratch[
                    local_t0 : local_t0 + PROJ_ROW_TILE,
                    o0 : o0 + OUT_TILE,
                ] = kv_acc
                score_proj_scratch[
                    local_t0 : local_t0 + PROJ_ROW_TILE,
                    o0 : o0 + OUT_TILE,
                ] = score_acc

    # Pad only through the proven M=16 cube shape. Downstream stages consume
    # physical rows only, so the last block's padded lanes remain internal.
    if proj_full_rows < proj_padded_rows:
        for tail_idx in pl.spmd(
            (OUT_DIM // OUT_TILE) * PROJ_TAIL_ROW_BLOCKS,
            name_hint="prefill_c4_kv_score_proj_tail",
        ):
            tail_n = tail_idx // PROJ_TAIL_ROW_BLOCKS
            tail_row_block = tail_idx - tail_n * PROJ_TAIL_ROW_BLOCKS
            tail_o0 = tail_n * OUT_TILE
            tail_t0 = proj_full_rows + tail_row_block * PROJ_TAIL_ROW_TILE
            if tail_t0 < proj_padded_rows:
                tail_valid_rows = pl.min(PROJ_TAIL_ROW_TILE, tile_rows - tail_t0)
                tail_global_t0 = tile_base + tail_t0
                kv_tail_acc = pl.create_tensor([PROJ_TAIL_ROW_TILE, OUT_TILE], dtype=pl.FP32)
                score_tail_acc = pl.create_tensor([PROJ_TAIL_ROW_TILE, OUT_TILE], dtype=pl.FP32)
                for tail_kb in pl.pipeline(0, D // K_TILE, stage=2):
                    tail_k0 = tail_kb * K_TILE
                    x_tail = pl.slice(
                        x_flat,
                        [PROJ_TAIL_ROW_TILE, K_TILE],
                        [tail_global_t0, tail_k0],
                        valid_shape=[tail_valid_rows, K_TILE],
                    )
                    wkv_tail = wkv[
                        tail_o0 : tail_o0 + OUT_TILE,
                        tail_k0 : tail_k0 + K_TILE,
                    ]
                    wgate_tail = wgate[
                        tail_o0 : tail_o0 + OUT_TILE,
                        tail_k0 : tail_k0 + K_TILE,
                    ]
                    if tail_k0 == 0:
                        kv_tail_acc = pl.matmul(
                            x_tail,
                            wkv_tail,
                            out_dtype=pl.FP32,
                            b_trans=True,
                        )
                        score_tail_acc = pl.matmul(
                            x_tail,
                            wgate_tail,
                            out_dtype=pl.FP32,
                            b_trans=True,
                        )
                    else:
                        kv_tail_acc = pl.matmul_acc(
                            kv_tail_acc,
                            x_tail,
                            wkv_tail,
                            b_trans=True,
                        )
                        score_tail_acc = pl.matmul_acc(
                            score_tail_acc,
                            x_tail,
                            wgate_tail,
                            b_trans=True,
                        )
                kv_proj_scratch[
                    tail_t0 : tail_t0 + PROJ_TAIL_ROW_TILE,
                    tail_o0 : tail_o0 + OUT_TILE,
                ] = kv_tail_acc
                score_proj_scratch[
                    tail_t0 : tail_t0 + PROJ_TAIL_ROW_TILE,
                    tail_o0 : tail_o0 + OUT_TILE,
                ] = score_tail_acc

    write_pos_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_write_map"):
        write_pos_map[0:1, 0:MAX_CMP_WRITES] = pl.full([1, MAX_CMP_WRITES], dtype=pl.INT32, value=0)
        write_dst_map[0:1, 0:MAX_CMP_WRITES] = pl.full([1, MAX_CMP_WRITES], dtype=pl.INT32, value=-1)
        map_seen = pl.cast(0, pl.INDEX)
        for map_local in pl.range(tile_rows):
            map_global = tile_base + map_local
            map_slot_raw = pl.read(cmp_slot_mapping, [map_global])
            if map_slot_raw >= 0:
                pl.write(write_pos_map, [0, map_seen], pl.read(position_ids, [map_global]))
                pl.write(write_dst_map, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                map_seen = map_seen + 1

    # Scatter the whole tile before pooling. The production state ring has 520
    # physical rows, so a 512-row tile plus the ratio-4 eight-row read window
    # cannot overwrite a live row within this tile. The fence orders reuse by
    # the following tile.
    with pl.spmd(tile_rows, name_hint="prefill_c4_state_scatter_pre"):
        scatter_local = pl.tile.get_block_idx()
        scatter_order = pl.read(state_order_fence, [0])
        if scatter_order >= 0:
            scatter_global = tile_base + scatter_local
            scatter_row_raw = pl.read(state_slot_mapping, [scatter_global])
            if scatter_row_raw >= 0:
                scatter_row = pl.cast(scatter_row_raw, pl.INDEX)
                scatter_pos = pl.read(position_ids, [scatter_global])
                scatter_ape_slot = pl.cast(scatter_pos % COMPRESS_RATIO, pl.INDEX)
                for scatter_ob in pl.range(OUT_DIM // OUT_TILE):
                    scatter_o0 = scatter_ob * OUT_TILE
                    compress_state_flat[
                        scatter_row : scatter_row + 1,
                        scatter_o0 : scatter_o0 + OUT_TILE,
                    ] = kv_proj_scratch[
                        scatter_local : scatter_local + 1,
                        scatter_o0 : scatter_o0 + OUT_TILE,
                    ]
                    compress_state_flat[
                        scatter_row : scatter_row + 1,
                        OUT_DIM + scatter_o0 : OUT_DIM + scatter_o0 + OUT_TILE,
                    ] = pl.add(
                        score_proj_scratch[
                            scatter_local : scatter_local + 1,
                            scatter_o0 : scatter_o0 + OUT_TILE,
                        ],
                        ape[
                            scatter_ape_slot : scatter_ape_slot + 1,
                            scatter_o0 : scatter_o0 + OUT_TILE,
                        ],
                    )

    pooled_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)
    for write_i in pl.spmd(MAX_CMP_WRITES, name_hint="prefill_c4_softmax_pool"):
        pool_kv_tile = pl.create_tensor([STATE_LEN, HEAD_D_TILE], dtype=pl.FP32)
        pool_score_tile = pl.create_tensor([STATE_LEN, HEAD_D_TILE], dtype=pl.FP32)
        write_slot_raw = pl.read(write_dst_map, [0, write_i])
        if write_slot_raw >= 0:
            write_pos = pl.read(write_pos_map, [0, write_i])
            cur_start = write_pos + 1 - COMPRESS_RATIO
            prev_start = cur_start - COMPRESS_RATIO
            for pool_s in pl.range(COMPRESS_RATIO):
                pool_kv_tile[pool_s : pool_s + 1, 0:HEAD_D_TILE] = pl.full(
                    [1, HEAD_D_TILE], dtype=pl.FP32, value=0.0
                )
                pool_score_tile[pool_s : pool_s + 1, 0:HEAD_D_TILE] = pl.full(
                    [1, HEAD_D_TILE], dtype=pl.FP32, value=FP32_NEG_INF
                )
                if write_pos >= STATE_LEN - 1:
                    prev_abs = prev_start + pool_s
                    prev_state_block = pl.cast(prev_abs // CSA_STATE_BLOCK_SIZE, pl.INDEX)
                    prev_state_intra = pl.cast(
                        prev_abs - prev_state_block * CSA_STATE_BLOCK_SIZE,
                        pl.INDEX,
                    )
                    prev_phys_block_raw = pl.read(
                        compress_state_block_table,
                        [prev_state_block],
                    )
                    if prev_phys_block_raw >= 0:
                        prev_state_row = (
                            pl.cast(prev_phys_block_raw, pl.INDEX) * CSA_STATE_BLOCK_SIZE + prev_state_intra
                        )
                        pool_kv_tile[pool_s : pool_s + 1, 0:HEAD_D_TILE] = compress_state_flat[
                            prev_state_row : prev_state_row + 1,
                            0:HEAD_DIM,
                        ]
                        pool_score_tile[pool_s : pool_s + 1, 0:HEAD_D_TILE] = compress_state_flat[
                            prev_state_row : prev_state_row + 1,
                            OUT_DIM : OUT_DIM + HEAD_DIM,
                        ]

                cur_slot = COMPRESS_RATIO + pool_s
                cur_abs = cur_start + pool_s
                pool_kv_tile[cur_slot : cur_slot + 1, 0:HEAD_D_TILE] = pl.full(
                    [1, HEAD_D_TILE], dtype=pl.FP32, value=0.0
                )
                pool_score_tile[cur_slot : cur_slot + 1, 0:HEAD_D_TILE] = pl.full(
                    [1, HEAD_D_TILE], dtype=pl.FP32, value=FP32_NEG_INF
                )
                cur_state_block = pl.cast(cur_abs // CSA_STATE_BLOCK_SIZE, pl.INDEX)
                cur_state_intra = pl.cast(
                    cur_abs - cur_state_block * CSA_STATE_BLOCK_SIZE,
                    pl.INDEX,
                )
                cur_phys_block_raw = pl.read(compress_state_block_table, [cur_state_block])
                if cur_phys_block_raw >= 0:
                    cur_state_row = (
                        pl.cast(cur_phys_block_raw, pl.INDEX) * CSA_STATE_BLOCK_SIZE + cur_state_intra
                    )
                    pool_kv_tile[cur_slot : cur_slot + 1, 0:HEAD_D_TILE] = compress_state_flat[
                        cur_state_row : cur_state_row + 1,
                        HEAD_DIM:OUT_DIM,
                    ]
                    pool_score_tile[cur_slot : cur_slot + 1, 0:HEAD_D_TILE] = compress_state_flat[
                        cur_state_row : cur_state_row + 1,
                        OUT_DIM + HEAD_DIM : COMPRESS_STATE_DIM,
                    ]

            init_slot = STATE_LEN - 1
            mi_buf = pl.create_tensor([1, HEAD_D_TILE], dtype=pl.FP32)
            li_buf = pl.create_tensor([1, HEAD_D_TILE], dtype=pl.FP32)
            oi_buf = pl.create_tensor([1, HEAD_D_TILE], dtype=pl.FP32)
            mi_buf[0:1, 0:HEAD_D_TILE] = pool_score_tile[init_slot : init_slot + 1, 0:HEAD_D_TILE]
            li_buf[0:1, 0:HEAD_D_TILE] = pl.exp(
                pl.sub(
                    mi_buf[0:1, 0:HEAD_D_TILE],
                    mi_buf[0:1, 0:HEAD_D_TILE],
                )
            )
            oi_buf[0:1, 0:HEAD_D_TILE] = pool_kv_tile[init_slot : init_slot + 1, 0:HEAD_D_TILE]
            for pool_slot_i in pl.range(STATE_LEN - 1):
                if pool_slot_i >= COMPRESS_RATIO or write_pos >= STATE_LEN - 1:
                    mi = mi_buf[0:1, 0:HEAD_D_TILE]
                    li = li_buf[0:1, 0:HEAD_D_TILE]
                    oi = oi_buf[0:1, 0:HEAD_D_TILE]
                    slot_score = pool_score_tile[pool_slot_i : pool_slot_i + 1, 0:HEAD_D_TILE]
                    slot_kv = pool_kv_tile[pool_slot_i : pool_slot_i + 1, 0:HEAD_D_TILE]
                    mi_next = pl.maximum(mi, slot_score)
                    alpha = pl.exp(pl.sub(mi, mi_next))
                    beta = pl.exp(pl.sub(slot_score, mi_next))
                    li_buf[0:1, 0:HEAD_D_TILE] = pl.add(pl.mul(alpha, li), beta)
                    oi_buf[0:1, 0:HEAD_D_TILE] = pl.add(
                        pl.mul(oi, alpha),
                        pl.mul(slot_kv, beta),
                    )
                    mi_buf[0:1, 0:HEAD_D_TILE] = mi_next
            pooled_kv[write_i : write_i + 1, 0:HEAD_DIM] = pl.div(
                oi_buf[0:1, 0:HEAD_D_TILE],
                li_buf[0:1, 0:HEAD_D_TILE],
            )
        else:
            pooled_zero = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
            pooled_kv[write_i : write_i + 1, 0:HEAD_DIM] = pooled_zero

    # A single-output GM task closes the state RAW/WAR chain before the next
    # dynamic tile reuses the 520-row ring.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_state_order_commit"):
        pool_sample = pl.read(pooled_kv, [0, 0])
        fence_bit = pl.cast(pool_sample == pool_sample, pl.INT32)
        pl.write(state_order_fence, [0], fence_bit * fence_bit)

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    normed_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)
    for final_block in pl.spmd(
        MAX_CMP_WRITES // PACKED_RMS_TILE,
        name_hint="prefill_c4_rmsnorm_rope",
    ):
        final_base = final_block * PACKED_RMS_TILE
        cos_b = pl.full(
            [PACKED_RMS_TILE, ROPE_HEAD_DIM // 2],
            dtype=pl.FP32,
            value=0.0,
        )
        sin_b = pl.full(
            [PACKED_RMS_TILE, ROPE_HEAD_DIM // 2],
            dtype=pl.FP32,
            value=0.0,
        )
        for final_dt in pl.range(PACKED_RMS_TILE):
            final_i = final_base + final_dt
            write_slot_raw = pl.read(write_dst_map, [0, final_i])
            if write_slot_raw >= 0:
                write_pos = pl.read(write_pos_map, [0, final_i])
                cmp_pos = pl.cast(write_pos + 1 - COMPRESS_RATIO, pl.INDEX)
                cos_b[final_dt : final_dt + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(
                    freqs_cos[cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2],
                    target_type=pl.FP32,
                )
                sin_b[final_dt : final_dt + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(
                    freqs_sin[cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2],
                    target_type=pl.FP32,
                )

        partial_sq = pl.full([1, PACKED_RMS_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[
                final_base : final_base + PACKED_RMS_TILE,
                k0 : k0 + HEAD_TILE,
            ]
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(
                    pl.row_sum(pl.mul(kv_rms_chunk, kv_rms_chunk)),
                    [1, PACKED_RMS_TILE],
                ),
            )
        variance = pl.reshape(
            pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS),
            [PACKED_RMS_TILE, 1],
        )
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[
                final_base : final_base + PACKED_RMS_TILE,
                k0 : k0 + HEAD_TILE,
            ]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_kv[
                final_base : final_base + PACKED_RMS_TILE,
                k0 : k0 + HEAD_TILE,
            ] = pl.col_expand_mul(
                pl.row_expand_mul(kv_norm_chunk, inv_rms),
                gamma,
            )
        kv_rope_norm = pooled_kv[
            final_base : final_base + PACKED_RMS_TILE,
            NOPE_HEAD_DIM:HEAD_DIM,
        ]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(
            pl.row_expand_mul(kv_rope_norm, inv_rms),
            gamma_rope,
        )
        rope_dup_idx = rope_dup_idx_template[:, :]
        rope_swap_idx = rope_swap_idx_template[:, :]
        rope_sign = rope_sign_template[:, :]
        cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        normed_kv[
            final_base : final_base + PACKED_RMS_TILE,
            NOPE_HEAD_DIM:HEAD_DIM,
        ] = pl.add(
            pl.mul(rope_normed, cos_il),
            pl.mul(pl.mul(swapped, rope_sign), sin_il),
        )

    with pl.spmd(
        MAX_CMP_WRITES // PACKED_RMS_TILE,
        name_hint="prefill_c4_cache_write",
    ):
        final_base = pl.tile.get_block_idx() * PACKED_RMS_TILE
        for final_dt in pl.range(PACKED_RMS_TILE):
            final_i = final_base + final_dt
            dst_row_raw = pl.read(write_dst_map, [0, final_i])
            if dst_row_raw >= 0:
                dst_row = pl.cast(dst_row_raw, pl.INDEX)
                cmp_kv_flat[dst_row : dst_row + 1, 0:HEAD_DIM] = pl.cast(
                    normed_kv[final_i : final_i + 1, 0:HEAD_DIM],
                    target_type=pl.BF16,
                    mode="rint",
                )

    return cmp_kv, compress_state


@pl.jit.inline(auto_scope=False)
def compressor_ratio4(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    """Compress one physical dynamic token run through ordered 512-row state tiles."""
    t_dim = pl.tensor.dim(x, 0)
    rope_dup_idx_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_swap_idx_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_sign_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_rope_index_prepare"):
        for rope_r in pl.range(PACKED_RMS_TILE):
            for rope_c in pl.range(ROPE_HEAD_DIM):
                rope_lane = rope_c % 2
                pl.write(
                    rope_dup_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c // 2, pl.INT32),
                )
                pl.write(
                    rope_swap_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c + 1 - rope_lane * 2, pl.INT32),
                )
                pl.write(
                    rope_sign_template,
                    [rope_r, rope_c],
                    pl.cast(pl.cast(rope_lane * 2 - 1, pl.INT32), pl.FP32),
                )

    state_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_state_order_init"):
        pl.write(state_order_fence, [0], pl.cast(0, pl.INT32))
    for tile_base in pl.range(0, t_dim, PREFILL_STATE_TILE):
        tile_rows = pl.min(PREFILL_STATE_TILE, t_dim - tile_base)
        with pl.scope():
            _prefill_compressor_ratio4_tile(
                x,
                compress_state,
                compress_state_block_table,
                wkv,
                wgate,
                ape,
                norm_w,
                freqs_cos,
                freqs_sin,
                cmp_kv,
                position_ids,
                cmp_slot_mapping,
                state_slot_mapping,
                rope_dup_idx_template,
                rope_swap_idx_template,
                rope_sign_template,
                state_order_fence,
                tile_base,
                tile_rows,
            )
    return cmp_kv, compress_state


