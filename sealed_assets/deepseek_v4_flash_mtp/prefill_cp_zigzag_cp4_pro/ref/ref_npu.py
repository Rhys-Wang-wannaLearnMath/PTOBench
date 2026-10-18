# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek V4 context-parallel zigzag ownership and projected-KV tail exchange."""

import pypto.language as pl
import pypto.language.distributed as pld
from config import CASE_CONFIG as M, CP_SIZE as FIXED_CP_SIZE, EPOCHS as FIXED_EPOCHS, TAIL_ROWS as FIXED_TAIL_ROWS

# model config
TAIL_ROWS = FIXED_TAIL_ROWS
HEAD_DIM = M.head_dim

# CP layout
EPOCHS = FIXED_EPOCHS

# tiling
ROW_TILE = 8


CP_SIZE = FIXED_CP_SIZE
NUM_SEGMENTS = 2 * CP_SIZE

# Rank-major tail-window rows.
CP_TAIL_WINDOW_ROWS = NUM_SEGMENTS * TAIL_ROWS


@pl.jit.incore
def prefill_cp_zigzag_kv_tail_exchange_core(
    local_kv_tail: pl.Tensor[[EPOCHS * 2 * TAIL_ROWS, HEAD_DIM], pl.BF16],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    logical_tails_out: pl.Out[pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]],
    decode_raw_window_out: pl.Out[pl.Tensor[[TAIL_ROWS, HEAD_DIM], pl.BF16]],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]:
    """Exchange two local tails and assemble the final decode window."""
    for epoch in pl.range(EPOCHS):
        epoch_value = pl.cast(epoch + 1, pl.INT32)
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.wait(
                    signal=consumed,
                    offsets=[peer, 0],
                    expected=epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        for peer in pl.range(CP_SIZE):
            for part in pl.range(2):
                rank_major_pos = my_rank * 2 + part
                pld.tensor.put(
                    dst=kv_tail_window,
                    peer=peer,
                    src=local_kv_tail,
                    dst_offsets=[rank_major_pos * TAIL_ROWS, 0],
                    src_offsets=[(epoch * 2 + part) * TAIL_ROWS, 0],
                    shape=[TAIL_ROWS, HEAD_DIM],
                    chunk_rows=ROW_TILE,
                    chunk_cols=HEAD_DIM,
                    pipeline=True,
                )
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=ready,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        for seg in pl.range(NUM_SEGMENTS):
            rm_pos = reverse_index[seg]
            owner = owner_rank_table[seg]
            if owner != my_rank:
                pld.system.wait(
                    signal=ready,
                    offsets=[owner, 0],
                    expected=epoch_value,
                    cmp=pld.WaitCmp.Ge,
                )
            source_row = rm_pos * TAIL_ROWS
            destination_row = epoch * CP_TAIL_WINDOW_ROWS + seg * TAIL_ROWS
            for t0 in pl.range(0, TAIL_ROWS, ROW_TILE):
                win_tile = pl.load(
                    kv_tail_window,
                    [source_row + t0, 0],
                    [ROW_TILE, HEAD_DIM],
                )
                pl.store(
                    win_tile,
                    [destination_row + t0, 0],
                    logical_tails_out,
                )

        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=consumed,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    last_epoch_base = (EPOCHS - 1) * CP_TAIL_WINDOW_ROWS
    for j in pl.range(TAIL_ROWS):
        seg_src = final_win_seg_src[j]
        row_src = final_win_row_src[j]
        if seg_src >= 0:
            source_row = last_epoch_base + seg_src * TAIL_ROWS + row_src
            win_row = pl.load(logical_tails_out, [source_row, 0], [1, HEAD_DIM])
            pl.store(win_row, [j, 0], decode_raw_window_out)

    return logical_tails_out


@pl.jit
def prefill_cp_zigzag_kv_tail_exchange(
    local_kv_tail: pl.Tensor[[EPOCHS * 2 * TAIL_ROWS, HEAD_DIM], pl.BF16],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    logical_tails_out: pl.Out[pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]],
    decode_raw_window_out: pl.Out[pl.Tensor[[TAIL_ROWS, HEAD_DIM], pl.BF16]],
    my_rank: pl.Scalar[pl.INT32],
):
    return prefill_cp_zigzag_kv_tail_exchange_core(
        local_kv_tail, reverse_index, owner_rank_table,
        final_win_seg_src, final_win_row_src,
        kv_tail_window, ready, consumed,
        logical_tails_out, decode_raw_window_out, my_rank,
    )


@pl.jit.host
def prefill_cp_zigzag_kv_tail_exchange_test(
    local_kv_tail: pl.Tensor[[CP_SIZE, EPOCHS * 2 * TAIL_ROWS, HEAD_DIM], pl.BF16],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    logical_tails_out: pl.Out[pl.Tensor[[CP_SIZE, EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]],
    decode_raw_window_out: pl.Out[pl.Tensor[[CP_SIZE, TAIL_ROWS, HEAD_DIM], pl.BF16]],
):
    """Launch one zigzag exchange child per rank."""
    kv_tail_window_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        kv_tail_window = pld.window(kv_tail_window_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
        ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        consumed = pld.window(consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        prefill_cp_zigzag_kv_tail_exchange(
            local_kv_tail[rank], reverse_index, owner_rank_table,
            final_win_seg_src, final_win_row_src,
            kv_tail_window, ready, consumed,
            logical_tails_out[rank], decode_raw_window_out[rank], rank,
            device=rank,
        )
