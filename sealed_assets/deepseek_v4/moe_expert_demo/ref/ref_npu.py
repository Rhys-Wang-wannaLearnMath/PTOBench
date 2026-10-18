"""Combined local routed and shared INT8 experts."""

import pypto.language as pl

from config import (
    CASE_CONFIG as M, T, D, MOE_INTER, SWIGLU_LIMIT, N_LOCAL_EXPERTS,
    RECV_MAX, INT8_SCALE_MAX, INT8_AMAX_EPS,
)

RECV_TILE = 32
K_CHUNK = 512
INTER_K = 512
INTER_CHUNK = 128
D_OUT_CHUNK = 128
QUANT_CHUNK = 256
T_TILE = 16


@pl.jit.inline
def moe_expert(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    x_local: pl.Tensor[[T, D], pl.BF16],
    expert_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    expert_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])



    for local_i in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(recv_expert_count, [local_i, 0])
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        flat_base = local_i * RECV_MAX

        for t in pl.parallel(n_tiles):
            t0 = t * RECV_TILE
            flat_t0 = flat_base + t0

            valid_rows = pl.min(RECV_TILE, n_rows - t0)


            h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)

            for n0 in pl.parallel(0, MOE_INTER, INTER_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_gate_up_matmul"):
                    x_init = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, 0 : K_CHUNK]
                    w1_init = expert_w1[local_i : local_i + 1, n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                    w3_init = expert_w3[local_i : local_i + 1, n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                    gate_acc = pl.matmul(x_init, w1_init, b_trans=True, out_dtype=pl.INT32)
                    up_acc = pl.matmul(x_init, w3_init, b_trans=True, out_dtype=pl.INT32)
                    for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                        x_k = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, k0 : k0 + K_CHUNK]
                        w1_k = expert_w1[local_i : local_i + 1, n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                        w3_k = expert_w3[local_i : local_i + 1, n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                        gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                        up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_dequant_swiglu"):
                    gate_2d_i32 = pl.reshape(gate_acc, [RECV_TILE, INTER_CHUNK])
                    up_2d_i32 = pl.reshape(up_acc, [RECV_TILE, INTER_CHUNK])
                    recv_x_scale_dq = pl.reshape(recv_scale_dq[local_i : local_i + 1, t0 : t0 + RECV_TILE], [RECV_TILE, 1])
                    w1_scale_chunk = expert_w1_scale[local_i : local_i + 1, n0 : n0 + INTER_CHUNK]
                    w3_scale_chunk = expert_w3_scale[local_i : local_i + 1, n0 : n0 + INTER_CHUNK]
                    gate_2d = pl.cast(gate_2d_i32, target_type=pl.FP32, mode="none")
                    up_2d = pl.cast(up_2d_i32, target_type=pl.FP32, mode="none")
                    gate_2d = pl.col_expand_mul(pl.row_expand_mul(gate_2d, recv_x_scale_dq), w1_scale_chunk)
                    up_2d = pl.col_expand_mul(pl.row_expand_mul(up_2d, recv_x_scale_dq), w3_scale_chunk)
                    if SWIGLU_LIMIT > 0.0:
                        gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                        up_2d = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                    silu = pl.mul(gate_2d, sigmoid)
                    gated = pl.mul(silu, up_2d)

                    gated_valid = pl.tensor.set_validshape(gated, valid_rows, INTER_CHUNK)
                    gated_masked = pl.fillpad(gated_valid, pad_value=pl.PadValue.zero)
                    h_tile_fp32[:, n0 : n0 + INTER_CHUNK] = gated_masked


            h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q"):
                eh_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for k0 in pl.range(0, MOE_INTER, QUANT_CHUNK):
                    eh_a_f32 = h_tile_fp32[:, k0 : k0 + QUANT_CHUNK]
                    eh_a_abs = pl.maximum(eh_a_f32, pl.neg(eh_a_f32))
                    eh_a_max = pl.reshape(pl.row_max(eh_a_abs), [1, RECV_TILE])
                    eh_amax = pl.maximum(eh_amax, eh_a_max)
                eh_sq_row = pl.div(
                    pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), eh_amax
                )
                h_tile_scale_dq = pl.reshape(pl.recip(eh_sq_row), [RECV_TILE, 1])
                eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                for k1 in pl.range(0, MOE_INTER, QUANT_CHUNK):
                    eh_q_f32 = h_tile_fp32[:, k1 : k1 + QUANT_CHUNK]
                    eh_q_scaled = pl.row_expand_mul(eh_q_f32, eh_sq_col)
                    eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode="rint")
                    eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode="round")
                    h_tile_i8[:, k1 : k1 + QUANT_CHUNK] = pl.cast(eh_q_half, target_type=pl.INT8, mode="trunc")


            for d0 in pl.parallel(0, D, D_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_w2_matmul"):
                    h_init = h_tile_i8[:, 0 : INTER_K]
                    w2_init = expert_w2[local_i : local_i + 1, d0 : d0 + D_OUT_CHUNK, 0 : INTER_K]
                    y_acc = pl.matmul(h_init, w2_init, b_trans=True, out_dtype=pl.INT32)
                    for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                        h_k = h_tile_i8[:, k0 : k0 + INTER_K]
                        w2_k = expert_w2[local_i : local_i + 1, d0 : d0 + D_OUT_CHUNK, k0 : k0 + INTER_K]
                        y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_w2_dequant_write"):
                    y_2d_i32 = pl.reshape(y_acc, [RECV_TILE, D_OUT_CHUNK])
                    w2_scale_chunk = expert_w2_scale[local_i : local_i + 1, d0 : d0 + D_OUT_CHUNK]
                    y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                    y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_tile_scale_dq), w2_scale_chunk)
                    recv_y_flat[flat_t0 : flat_t0 + RECV_TILE, d0 : d0 + D_OUT_CHUNK] = pl.cast(
                        y_2d, target_type=pl.BF16, mode="rint"
                    )


    for ts0 in pl.parallel(0, T, T_TILE):
        x_local_i8_tile = pl.create_tensor([T_TILE, D], dtype=pl.INT8)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_local_q"):
            xl_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.range(0, D, QUANT_CHUNK):
                xl_a_f32 = pl.full([T_TILE, QUANT_CHUNK], dtype=pl.FP32, value=0.0)
                xl_a_f32[0:T, :] = pl.cast(x_local[ts0 : ts0 + T, k0 : k0 + QUANT_CHUNK], target_type=pl.FP32)
                xl_a_abs = pl.maximum(xl_a_f32, pl.neg(xl_a_f32))
                xl_a_max = pl.reshape(pl.row_max(xl_a_abs), [1, T_TILE])
                xl_amax = pl.maximum(xl_amax, xl_a_max)
            xl_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), xl_amax)
            x_local_scale_dq = pl.reshape(pl.recip(xl_sq_row), [T_TILE, 1])
            xl_sq_col = pl.reshape(xl_sq_row, [T_TILE, 1])
            for k1 in pl.range(0, D, QUANT_CHUNK):
                xl_q_f32 = pl.full([T_TILE, QUANT_CHUNK], dtype=pl.FP32, value=0.0)
                xl_q_f32[0:T, :] = pl.cast(x_local[ts0 : ts0 + T, k1 : k1 + QUANT_CHUNK], target_type=pl.FP32)
                xl_q_scaled = pl.row_expand_mul(xl_q_f32, xl_sq_col)
                xl_q_i32 = pl.cast(xl_q_scaled, target_type=pl.INT32, mode="rint")
                xl_q_half = pl.cast(xl_q_i32, target_type=pl.FP16, mode="round")
                x_local_i8_tile[:, k1 : k1 + QUANT_CHUNK] = pl.cast(xl_q_half, target_type=pl.INT8, mode="trunc")

        sh_tile_fp32 = pl.create_tensor([T_TILE, MOE_INTER], dtype=pl.FP32)
        for n0 in pl.parallel(0, MOE_INTER, INTER_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up_matmul"):
                xs_init = x_local_i8_tile[:, 0 : K_CHUNK]
                sw1_init = shared_w1[n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                sw3_init = shared_w3[n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                sh_gate_acc = pl.matmul(xs_init, sw1_init, b_trans=True, out_dtype=pl.INT32)
                sh_up_acc = pl.matmul(xs_init, sw3_init, b_trans=True, out_dtype=pl.INT32)
                for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                    xs_k = x_local_i8_tile[:, k0 : k0 + K_CHUNK]
                    sw1_k = shared_w1[n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                    sw3_k = shared_w3[n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                    sh_gate_acc = pl.matmul_acc(sh_gate_acc, xs_k, sw1_k, b_trans=True)
                    sh_up_acc = pl.matmul_acc(sh_up_acc, xs_k, sw3_k, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_dequant_swiglu"):
                sw1_scale_chunk = pl.reshape(shared_w1_scale[n0 : n0 + INTER_CHUNK], [1, INTER_CHUNK])
                sw3_scale_chunk = pl.reshape(shared_w3_scale[n0 : n0 + INTER_CHUNK], [1, INTER_CHUNK])
                sh_gate = pl.cast(sh_gate_acc, target_type=pl.FP32, mode="none")
                sh_up = pl.cast(sh_up_acc, target_type=pl.FP32, mode="none")
                sh_gate = pl.col_expand_mul(pl.row_expand_mul(sh_gate, x_local_scale_dq), sw1_scale_chunk)
                sh_up = pl.col_expand_mul(pl.row_expand_mul(sh_up, x_local_scale_dq), sw3_scale_chunk)
                sh_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(sh_gate)), 1.0))
                sh_silu = pl.mul(sh_gate, sh_sigmoid)
                sh_gated = pl.mul(sh_silu, sh_up)
                sh_tile_fp32[:, n0 : n0 + INTER_CHUNK] = sh_gated

        sh_tile_i8 = pl.create_tensor([T_TILE, MOE_INTER], dtype=pl.INT8)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_h_q"):
            shq_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.range(0, MOE_INTER, QUANT_CHUNK):
                shq_a_f32 = sh_tile_fp32[:, k0 : k0 + QUANT_CHUNK]
                shq_a_abs = pl.maximum(shq_a_f32, pl.neg(shq_a_f32))
                shq_a_max = pl.reshape(pl.row_max(shq_a_abs), [1, T_TILE])
                shq_amax = pl.maximum(shq_amax, shq_a_max)
            shq_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), shq_amax)
            sh_tile_scale_dq = pl.reshape(pl.recip(shq_sq_row), [T_TILE, 1])
            shq_sq_col = pl.reshape(shq_sq_row, [T_TILE, 1])
            for k1 in pl.range(0, MOE_INTER, QUANT_CHUNK):
                shq_q_f32 = sh_tile_fp32[:, k1 : k1 + QUANT_CHUNK]
                shq_q_scaled = pl.row_expand_mul(shq_q_f32, shq_sq_col)
                shq_q_i32 = pl.cast(shq_q_scaled, target_type=pl.INT32, mode="rint")
                shq_q_half = pl.cast(shq_q_i32, target_type=pl.FP16, mode="round")
                sh_tile_i8[:, k1 : k1 + QUANT_CHUNK] = pl.cast(shq_q_half, target_type=pl.INT8, mode="trunc")

        for d0 in pl.parallel(0, D, D_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_w2_matmul"):
                sh_y_acc = pl.create_tensor([T_TILE, D_OUT_CHUNK], dtype=pl.INT32)
                for kb in pl.pipeline(0, MOE_INTER // INTER_K, stage=2):
                    k0 = kb * INTER_K
                    hs_k = sh_tile_i8[:, k0 : k0 + INTER_K]
                    sw2_k = shared_w2[d0 : d0 + D_OUT_CHUNK, k0 : k0 + INTER_K]
                    if kb == 0:
                        sh_y_acc = pl.matmul(hs_k, sw2_k, b_trans=True, out_dtype=pl.INT32)
                    else:
                        sh_y_acc = pl.matmul_acc(sh_y_acc, hs_k, sw2_k, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_w2_dequant_write"):
                sw2_scale_chunk = pl.reshape(shared_w2_scale[d0 : d0 + D_OUT_CHUNK], [1, D_OUT_CHUNK])
                sh_y = pl.cast(sh_y_acc, target_type=pl.FP32, mode="none")
                sh_y = pl.col_expand_mul(pl.row_expand_mul(sh_y, sh_tile_scale_dq), sw2_scale_chunk)
                sh[ts0 : ts0 + T, d0 : d0 + D_OUT_CHUNK] = pl.cast(sh_y[0:T, :], target_type=pl.BF16, mode="rint")


@pl.jit
def moe_expert_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    x_local: pl.Tensor[[T, D], pl.BF16],
    expert_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    expert_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    moe_expert(
        recv_x, recv_scale_dq, recv_expert_count, x_local,
        expert_w1, expert_w1_scale, expert_w3, expert_w3_scale,
        expert_w2, expert_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        recv_y, sh,
    )
    return recv_y, sh
