import pypto.language as pl

from config import CASE_CONFIG as M, INT8_SCALE_MAX, INT8_AMAX_EPS, EP_WORLD_SIZE, RECV_MAX

D = M.hidden_size

MOE_INTER = M.moe_intermediate_size

SWIGLU_LIMIT = M.swiglu_limit

N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

RECV_TILE = 16

K_TILE = 512

INTER_K = 512

MM_INTER_TILE = 256

MM_GATE_INNER = 4

ACT_INTER_TILE = 128

ACT_GATE_INNER = 4

D_OUT_TILE = 256

QUANT_TILE = 256

D_OUT_TILE_ACT = 512

W2_INNER = 4

# Assemble complete rows: the lowered output tile has a packed row stride.
# A narrower block would alias a parent view whose row stride is still D.
W2_ACT_INNER = D // D_OUT_TILE_ACT

TILES_PER_EXPERT = RECV_MAX // RECV_TILE

@pl.jit.inline
def expert_routed(recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8], recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32], recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32], recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32], routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8], routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32], routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8], routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32], routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8], routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32], recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    w2_tids = pl.array.create(N_LOCAL_EXPERTS * TILES_PER_EXPERT, pl.TASK_ID)
    for local_i in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(recv_expert_count, [local_i, 0])
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        flat_base = local_i * RECV_MAX
        for t in pl.parallel(n_tiles):
            tile_idx = local_i * TILES_PER_EXPERT + t
            t0 = t * RECV_TILE
            flat_t0 = flat_base + t0
            valid_rows = pl.min(RECV_TILE, n_rows - t0)
            with pl.manual_scope():
                h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32, manual_dep=True)
                gate_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32, manual_dep=True)
                up_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32, manual_dep=True)
                with pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint='exp_gate_mm') as gate_tid:
                    nb_idx = pl.tile.get_block_idx()
                    n_base = nb_idx * (MM_GATE_INNER * MM_INTER_TILE)
                    for ng in pl.range(MM_GATE_INNER):
                        n0 = n_base + ng * MM_INTER_TILE
                        gate_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                        for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                            x_k = recv_x[local_i:local_i + 1, t0:t0 + RECV_TILE, k0:k0 + K_TILE]
                            w1_k = routed_w1[local_i:local_i + 1, n0:n0 + MM_INTER_TILE, k0:k0 + K_TILE]
                            if k0 == 0:
                                gate_acc = pl.matmul(x_k, w1_k, b_trans=True, out_dtype=pl.INT32)
                            else:
                                gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                        gate_i32[:, n0:n0 + MM_INTER_TILE] = pl.reshape(gate_acc, [RECV_TILE, MM_INTER_TILE])
                with pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint='exp_up_mm') as up_tid:
                    nb_idx = pl.tile.get_block_idx()
                    n_base = nb_idx * (MM_GATE_INNER * MM_INTER_TILE)
                    for ng in pl.range(MM_GATE_INNER):
                        n0 = n_base + ng * MM_INTER_TILE
                        up_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                        for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                            x_k = recv_x[local_i:local_i + 1, t0:t0 + RECV_TILE, k0:k0 + K_TILE]
                            w3_k = routed_w3[local_i:local_i + 1, n0:n0 + MM_INTER_TILE, k0:k0 + K_TILE]
                            if k0 == 0:
                                up_acc = pl.matmul(x_k, w3_k, b_trans=True, out_dtype=pl.INT32)
                            else:
                                up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)
                        up_i32[:, n0:n0 + MM_INTER_TILE] = pl.reshape(up_acc, [RECV_TILE, MM_INTER_TILE])
                with pl.spmd(MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE), name_hint='exp_gate_up_act', deps=[gate_tid, up_tid]) as act_tid:
                    nb_idx = pl.tile.get_block_idx()
                    n_base = nb_idx * (ACT_GATE_INNER * ACT_INTER_TILE)
                    for ng in pl.pipeline(ACT_GATE_INNER, stage=2):
                        n0 = n_base + ng * ACT_INTER_TILE
                        gate_2d_i32 = gate_i32[:, n0:n0 + ACT_INTER_TILE]
                        up_2d_i32 = up_i32[:, n0:n0 + ACT_INTER_TILE]
                        recv_x_scale_dq = pl.reshape(recv_scale_dq[local_i:local_i + 1, t0:t0 + RECV_TILE], [RECV_TILE, 1])
                        w1_scale_chunk = pl.reshape(routed_w1_scale[local_i:local_i + 1, n0:n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                        w3_scale_chunk = pl.reshape(routed_w3_scale[local_i:local_i + 1, n0:n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                        gate_2d = pl.cast(gate_2d_i32, target_type=pl.FP32, mode='none')
                        up_2d = pl.cast(up_2d_i32, target_type=pl.FP32, mode='none')
                        gate_2d = pl.col_expand_mul(pl.row_expand_mul(gate_2d, recv_x_scale_dq), w1_scale_chunk)
                        up_2d = pl.col_expand_mul(pl.row_expand_mul(up_2d, recv_x_scale_dq), w3_scale_chunk)
                        if SWIGLU_LIMIT > 0.0:
                            gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                            up_2d = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                        silu = pl.mul(gate_2d, sigmoid)
                        gated = pl.mul(silu, up_2d)
                        gated_valid = pl.set_validshape(gated, valid_rows, ACT_INTER_TILE)
                        gated_masked = pl.fillpad(gated_valid, pad_value=pl.PadValue.zero)
                        h_tile_fp32[:, n0:n0 + ACT_INTER_TILE] = gated_masked
                h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8, manual_dep=True)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='exp_h_q', deps=[act_tid]) as hq_tid:
                    eh_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                        eh_a_f32 = h_tile_fp32[:, k0:k0 + QUANT_TILE]
                        eh_a_abs = pl.maximum(eh_a_f32, pl.neg(eh_a_f32))
                        eh_a_max = pl.reshape(pl.row_max(eh_a_abs), [1, RECV_TILE])
                        eh_amax = pl.maximum(eh_amax, eh_a_max)
                    eh_sq_row = pl.div(pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), eh_amax)
                    h_tile_scale_dq = pl.reshape(pl.recip(eh_sq_row), [RECV_TILE, 1])
                    eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                    for k1 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                        eh_q_f32 = h_tile_fp32[:, k1:k1 + QUANT_TILE]
                        eh_q_scaled = pl.row_expand_mul(eh_q_f32, eh_sq_col)
                        eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode='rint')
                        eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode='round')
                        h_tile_i8[:, k1:k1 + QUANT_TILE] = pl.cast(eh_q_half, target_type=pl.INT8, mode='trunc')
                y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32, manual_dep=True)
                with pl.spmd(D // (W2_INNER * D_OUT_TILE), name_hint='exp_w2_mm', deps=[hq_tid]) as w2_tid:
                    db_idx = pl.tile.get_block_idx()
                    d_base = db_idx * (W2_INNER * D_OUT_TILE)
                    for dg in pl.range(W2_INNER):
                        d0 = d_base + dg * D_OUT_TILE
                        y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
                        for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                            h_k = h_tile_i8[:, k0:k0 + INTER_K]
                            w2_k = routed_w2[local_i:local_i + 1, d0:d0 + D_OUT_TILE, k0:k0 + INTER_K]
                            if k0 == 0:
                                y_acc = pl.matmul(h_k, w2_k, b_trans=True, out_dtype=pl.INT32)
                            else:
                                y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)
                        y_i32[:, d0:d0 + D_OUT_TILE] = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])
                w2_tids[tile_idx] = w2_tid
            for db_idx in pl.parallel(D // (W2_ACT_INNER * D_OUT_TILE_ACT)):
                act_d_base = db_idx * (W2_ACT_INNER * D_OUT_TILE_ACT)
                recv_y_block = pl.create_tensor([RECV_TILE, W2_ACT_INNER * D_OUT_TILE_ACT], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint='exp_w2_act', deps=[w2_tids[tile_idx]]):
                    w_col_blk = pl.reshape(recv_weights[local_i:local_i + 1, t0:t0 + RECV_TILE], [RECV_TILE, 1])
                    row_scale_blk = pl.mul(h_tile_scale_dq, w_col_blk)
                    for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                        act_d0 = act_d_base + dg * D_OUT_TILE_ACT
                        block_d0 = dg * D_OUT_TILE_ACT
                        y_2d_i32 = y_i32[:, act_d0:act_d0 + D_OUT_TILE_ACT]
                        w2_scale_chunk = pl.reshape(routed_w2_scale[local_i:local_i + 1, act_d0:act_d0 + D_OUT_TILE_ACT], [1, D_OUT_TILE_ACT])
                        y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode='none')
                        y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, row_scale_blk), w2_scale_chunk)
                        recv_y_block[:, block_d0:block_d0 + D_OUT_TILE_ACT] = pl.cast(y_2d, target_type=pl.BF16, mode='rint')
                recv_y_flat = pl.assemble(recv_y_flat, recv_y_block, [flat_t0, act_d_base])
    return recv_y
