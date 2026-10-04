# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE packed dispatch (decode, single-card EP).

Pure scatter from token-major router outputs to the per-local-expert layout
consumed by ``expert_routed``. The INT8 quant of ``x_norm`` already happened in
``gate``; this kernel only moves the pre-quantized rows and dequant
scales into the recv buffers.
"""


import pypto.language as pl
from config import (CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ,
                    EP_WORLD_SIZE, EP_RANK, RECV_MAX)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS


@pl.jit.inline
def dispatch(
    x_norm_i8:       pl.Tensor[[T, D],    pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1],    pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq:     pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_weights:      pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_token:        pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1],           pl.INT32],
):
    # recv_x still uses a flat row view because its destination row is
    # data-dependent. Small route metadata uses the natural 2-D layout.
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch"):
        # Zero-init tail slots so downstream consumers (expert_routed / combine)
        # can rely on slot >= recv_expert_count[e] being neutral:
        #   - recv_scale_dq = 0 -> dequant of any recv_x tail row yields 0
        #   - recv_weights  = 0 -> combine's weighted reduction skips tail rows
        #   - recv_token    = 0 -> safe scatter target (combine ignores via count)
        # The recv_x INT8 tail is left uninitialized; pairing it with scale_dq=0
        # is sufficient to neutralize its contribution after dequant.
        for e in pl.range(N_LOCAL_EXPERTS):
            pl.write(recv_expert_count, [e, 0], pl.cast(0, pl.INT32))
            for s in pl.range(RECV_MAX):
                pl.write(recv_scale_dq, [e, s], 0.0)
                pl.write(recv_weights, [e, s], 0.0)
                pl.write(recv_token, [e, s], pl.cast(0, pl.INT32))

        for t in pl.range(T):
            for k in pl.range(TOPK):
                e_global = pl.read(indices, [t, k])
                e = pl.cast(e_global - EXPERTS_START_IDX, pl.INDEX)
                slot_i32 = pl.read(recv_expert_count, [e, 0])
                slot = pl.cast(slot_i32, pl.INDEX)
                dst = e * RECV_MAX + slot

                recv_x_flat = pl.assemble(recv_x_flat, pl.slice(x_norm_i8, [1, D], [t, 0]), [dst, 0])
                pl.write(recv_scale_dq, [e, slot], pl.read(x_norm_scale, [t, 0]))
                pl.write(recv_weights, [e, slot], pl.read(weights, [t, k]))
                pl.write(recv_token, [e, slot], pl.cast(t, pl.INT32))
                pl.write(recv_expert_count, [e, 0], pl.cast(slot_i32 + 1, pl.INT32))


@pl.jit
def dispatch_test(
    x_norm_i8:       pl.Tensor[[T, D],    pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1],    pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8]],
    recv_scale_dq:     pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_weights:      pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_token:        pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32]],
    recv_expert_count: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]],
):
    dispatch(
        x_norm_i8, x_norm_scale, indices, weights,
        recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count,
    )
    return recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count
