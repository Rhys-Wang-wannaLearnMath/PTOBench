from __future__ import annotations

import pypto.language as pl
from config import CASE_CONFIG as M, MOE_TOKENS, INT8_SCALE_MAX, INT8_AMAX_EPS
T = MOE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
SH_M_TILE = 16
T_PAD = (T + SH_M_TILE - 1) // SH_M_TILE * SH_M_TILE
SH_VALID_M = T if T < SH_M_TILE else SH_M_TILE
N_MTILES = T_PAD // SH_M_TILE
K_TILE = 512
INTER_K = 512
MM_INTER_TILE = 256
ACT_INTER_TILE = 128
ACT_GATE_INNER = 4
D_OUT_TILE = 256
QUANT_TILE = 256
D_OUT_TILE_ACT = 512
W2_ACT_INNER = 4

@pl.jit.inline
def expert_shared(x_local_i8: pl.Tensor[[T, D], pl.INT8], x_local_scale_dq: pl.Tensor[[T, 1], pl.FP32], shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8], shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32], shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8], shared_w2_scale: pl.Tensor[[D], pl.FP32], sh: pl.Tensor[[T, D], pl.BF16]):
    for mt in pl.parallel(N_MTILES):
        ts0 = mt * SH_M_TILE
        h_tile_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        gate_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)
        up_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)
        for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint='sh_gate_mm'):
            n0 = nb_idx * MM_INTER_TILE
            gate_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE])
                sw1_k = shared_w1[n0:n0 + MM_INTER_TILE, k0:k0 + K_TILE]
                if k0 == 0:
                    gate_acc = pl.matmul(xs_k, sw1_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    gate_acc = pl.matmul_acc(gate_acc, xs_k, sw1_k, b_trans=True)
            gate_i32[:, n0:n0 + MM_INTER_TILE] = gate_acc
        for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint='sh_up_mm'):
            n0 = nb_idx * MM_INTER_TILE
            up_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE])
                sw3_k = shared_w3[n0:n0 + MM_INTER_TILE, k0:k0 + K_TILE]
                if k0 == 0:
                    up_acc = pl.matmul(xs_k, sw3_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    up_acc = pl.matmul_acc(up_acc, xs_k, sw3_k, b_trans=True)
            up_i32[:, n0:n0 + MM_INTER_TILE] = up_acc
        for nb_idx in pl.spmd(MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE), name_hint='sh_gate_up_act'):
            n_base = nb_idx * (ACT_GATE_INNER * ACT_INTER_TILE)
            for ng in pl.pipeline(ACT_GATE_INNER, stage=2):
                n0 = n_base + ng * ACT_INTER_TILE
                gate_2d_i32 = gate_i32[:, n0:n0 + ACT_INTER_TILE]
                up_2d_i32 = up_i32[:, n0:n0 + ACT_INTER_TILE]
                x_local_scale_dq_tile = pl.slice(x_local_scale_dq, [SH_M_TILE, 1], [ts0, 0], valid_shape=[SH_VALID_M, 1])
                w1_scale_chunk = pl.reshape(shared_w1_scale[n0:n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                w3_scale_chunk = pl.reshape(shared_w3_scale[n0:n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                gate_2d = pl.cast(gate_2d_i32, target_type=pl.FP32, mode='none')
                up_2d = pl.cast(up_2d_i32, target_type=pl.FP32, mode='none')
                gate_2d = pl.col_expand_mul(pl.row_expand_mul(gate_2d, x_local_scale_dq_tile), w1_scale_chunk)
                up_2d = pl.col_expand_mul(pl.row_expand_mul(up_2d, x_local_scale_dq_tile), w3_scale_chunk)
                if SWIGLU_LIMIT > 0.0:
                    gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                    up_2d = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                silu = pl.mul(gate_2d, sigmoid)
                gated = pl.mul(silu, up_2d)
                h_tile_fp32[:, n0:n0 + ACT_INTER_TILE] = gated
        h_tile_i8 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT8)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='sh_h_q'):
            eh_amax = pl.full([1, SH_M_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                eh_a_f32 = h_tile_fp32[:, k0:k0 + QUANT_TILE]
                eh_a_abs = pl.maximum(eh_a_f32, pl.neg(eh_a_f32))
                eh_a_max = pl.reshape(pl.row_max(eh_a_abs), [1, SH_M_TILE])
                eh_amax = pl.maximum(eh_amax, eh_a_max)
            eh_sq_row = pl.div(pl.full([1, SH_M_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), eh_amax)
            h_tile_scale_dq = pl.reshape(pl.recip(eh_sq_row), [SH_M_TILE, 1])
            eh_sq_col = pl.reshape(eh_sq_row, [SH_M_TILE, 1])
            for k1 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                eh_q_f32 = h_tile_fp32[:, k1:k1 + QUANT_TILE]
                eh_q_scaled = pl.row_expand_mul(eh_q_f32, eh_sq_col)
                eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode='rint')
                eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode='round')
                h_tile_i8[:, k1:k1 + QUANT_TILE] = pl.cast(eh_q_half, target_type=pl.INT8, mode='trunc')
        y_i32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.INT32)
        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint='sh_w2_mm'):
            d0 = db_idx * D_OUT_TILE
            y_acc = pl.create_tensor([SH_M_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                hs_k = h_tile_i8[:, k0:k0 + INTER_K]
                sw2_k = shared_w2[d0:d0 + D_OUT_TILE, k0:k0 + INTER_K]
                if k0 == 0:
                    y_acc = pl.matmul(hs_k, sw2_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True)
            y_i32[:, d0:d0 + D_OUT_TILE] = y_acc
        for db_idx in pl.spmd(D // (W2_ACT_INNER * D_OUT_TILE_ACT), name_hint='sh_w2_act'):
            d_base = db_idx * (W2_ACT_INNER * D_OUT_TILE_ACT)
            for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                d0 = d_base + dg * D_OUT_TILE_ACT
                y_2d_i32 = y_i32[:, d0:d0 + D_OUT_TILE_ACT]
                w2_scale_chunk = pl.reshape(shared_w2_scale[d0:d0 + D_OUT_TILE_ACT], [1, D_OUT_TILE_ACT])
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode='none')
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_tile_scale_dq), w2_scale_chunk)
                y_bf16 = pl.cast(y_2d, target_type=pl.BF16, mode='rint')
                sh[ts0:ts0 + SH_VALID_M, d0:d0 + D_OUT_TILE_ACT] = y_bf16[0:SH_VALID_M, :]
    return sh
