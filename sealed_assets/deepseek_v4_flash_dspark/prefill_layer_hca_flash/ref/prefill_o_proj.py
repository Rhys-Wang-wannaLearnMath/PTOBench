# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""REF closure for deepseek_v4_flash_dspark.prefill_layer_hca_flash; fixed bindings from public config."""
from config import D
from config import O_GROUPS
from config import O_GROUP_IN
from config import O_LORA
from config import O_PROJ_LOCAL_COLS
from config import O_PROJ_LOCAL_GROUPS
from config import TP_SIZE
import pypto.language as pl
import pypto.language.distributed as pld

O_PROJ_FULL_ROWS = O_GROUPS * O_LORA
O_PROJ_SCRATCH_GROUPS = O_GROUPS
O_PROJ_SCRATCH_RANK = O_LORA
O_PROJ_SCRATCH_INPUT = O_GROUP_IN
O_PROJ_SCRATCH_D = D
O_PROJ_SCRATCH_COLS = O_PROJ_FULL_ROWS
O_PROJ_WO_A_WINDOW_ROWS = O_PROJ_FULL_ROWS if TP_SIZE > 1 else 1
O_PROJ_WO_A_WINDOW_COLS = O_GROUP_IN if TP_SIZE > 1 else 1
O_PROJ_WO_B_WINDOW_ROWS = D if TP_SIZE > 1 else 1
O_PROJ_WO_B_WINDOW_COLS = O_PROJ_FULL_ROWS if TP_SIZE > 1 else 1
O_PROJ_WEIGHT_COPY_TILE = 16

@pl.jit.inline(auto_scope=False)
def gather_o_proj_full_weights(
    wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_local: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    order_fence: pl.Tensor[[1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    weight_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Materialize one layer's resident TP shards in reusable full-weight scratch."""
    previous_epoch = weight_epoch - pl.const(1, pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_reuse_wait") as reuse_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=weight_consumed, offsets=[source_tp, 0],
                    expected=previous_epoch, cmp=pld.WaitCmp.Ge,
                )

    wo_a_local_flat = pl.reshape(wo_a_local, [O_PROJ_LOCAL_COLS, O_GROUP_IN])
    with pl.spmd(TP_SIZE, name_hint="o_proj_weight_push", deps=[reuse_wait_tid], allow_early_resolve=True) as push_tid:
        peer_tp = pl.tile.get_block_idx()
        peer = group_base + pl.cast(peer_tp, pl.INT32)
        pld.tensor.put(
            dst=wo_a_window, peer=peer, src=wo_a_local_flat,
            dst_offsets=[tp_rank * O_PROJ_LOCAL_COLS, 0], src_offsets=[0, 0],
            shape=[O_PROJ_LOCAL_COLS, O_GROUP_IN],
            chunk_rows=O_PROJ_WEIGHT_COPY_TILE, chunk_cols=O_GROUP_IN,
        )
        pld.tensor.put(
            dst=wo_b_window, peer=peer, src=wo_b_local,
            dst_offsets=[0, tp_rank * O_PROJ_LOCAL_COLS], src_offsets=[0, 0],
            shape=[D, O_PROJ_LOCAL_COLS],
            chunk_rows=O_PROJ_WEIGHT_COPY_TILE, chunk_cols=O_PROJ_LOCAL_COLS,
        )
        if peer_tp != tp_rank:
            pld.system.notify(
                target=weight_ready, peer=peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_ready_wait") as ready_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=weight_ready, offsets=[source_tp, 0],
                    expected=weight_epoch, cmp=pld.WaitCmp.Ge,
                )

    wo_a_full_flat = pl.reshape(wo_a_full, [O_PROJ_FULL_ROWS, O_GROUP_IN])
    with pl.spmd(
        O_PROJ_FULL_ROWS // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_wo_a_readback",
        deps=[push_tid, ready_wait_tid],
    ) as wo_a_readback_tid:
        order = pl.read(order_fence, [0])
        if order >= 0:
            row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
            tile = pl.load(wo_a_window, [row, 0], [O_PROJ_WEIGHT_COPY_TILE, O_GROUP_IN], target_memory=pl.MemorySpace.Vec)
            pl.store(tile, [row, 0], wo_a_full_flat)

    with pl.spmd(
        D // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_wo_b_readback",
        deps=[push_tid, ready_wait_tid],
    ) as wo_b_readback_tid:
        order = pl.read(order_fence, [0])
        if order >= 0:
            row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
            tile = pl.load(
                wo_b_window, [row, 0], [O_PROJ_WEIGHT_COPY_TILE, O_PROJ_FULL_ROWS],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(tile, [row, 0], wo_b_full)

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_consumed",
        deps=[wo_a_readback_tid, wo_b_readback_tid],
    ) as weights_ready_tid:
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=weight_consumed, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    return weights_ready_tid


