"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_hca_flash."""

from config import CASE_CONFIG as M
from config import TP_SIZE
from config import INT8_SCALE_MAX
from config import DECODE_TOKENS
from config import INT8_AMAX_EPS
import pypto.language as pl
import pypto.language.distributed as pld

D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
LOCAL_T = DECODE_TOKENS // TP_SIZE
LOCAL_O_GROUPS = O_GROUPS // TP_SIZE
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA
TOKEN_TILE = 16
COMM_ROW_TILE = 8
O_RS_REDUCE_WORKERS = 8
O_RS_PUBLISH_WORKERS = 24
O_RS_D_TILE = 4096
LOCAL_T_PAD = (LOCAL_T + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE
GROUP_T_PAD = TP_SIZE * LOCAL_T_PAD
ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD
O_WINDOW_ROWS = TP_SIZE * LOCAL_T_PAD
O_A_T_TILE = 16
O_A_K_TILE = 256
O_A_N_TILE = 128
QUANT_T_TILE = 8
O_B_T_TILE = 128
O_B_K_TILE = 256
O_B_N_TILE = 256
O_B_D_TILE = 512
ACT_T_TILE = 8
ACT_N_TILE = 512

@pl.jit.inline
def decode_sharded_o_projection_reduce_scatter(attention_local_groups: pl.Tensor[[LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN], pl.BF16], wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], local_t: pl.Scalar[pl.INT32], local_out: pl.Tensor[[LOCAL_T_PAD, D], pl.BF16], reduce_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32], reduce_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32]):
    """Project O-B tiles directly into their ReduceScatter owner windows."""
    group_t = TP_SIZE * local_t
    o_a_rows = (group_t + O_A_T_TILE - 1) // O_A_T_TILE
    o_b_rows = (group_t + O_B_T_TILE - 1) // O_B_T_TILE
    o_b_group_t = o_b_rows * O_B_T_TILE
    owner_rows = (local_t + ACT_T_TILE - 1) // ACT_T_TILE
    attn_2d = pl.reshape(attention_local_groups, [LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN])
    wo_a_flat = pl.reshape(wo_a, [LOCAL_O_WIDTH, O_GROUP_IN])
    o_a_fp32 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_WIDTH], dtype=pl.FP32)
    o_a_i8 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_WIDTH], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([LOCAL_O_GROUPS, GROUP_T_PAD], dtype=pl.FP32)
    o_b_i32 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(LOCAL_O_GROUPS, pl.TASK_ID)
    for local_group in pl.parallel(LOCAL_O_GROUPS):
        attention_row = local_group * GROUP_T_PAD
        o_a_col = local_group * O_LORA
        with pl.spmd(o_a_rows * (O_LORA // O_A_N_TILE), name_hint='tp_o_a') as proj_a_tid:
            proj_a_unit = pl.tile.get_block_idx()
            row_block = proj_a_unit // (O_LORA // O_A_N_TILE)
            n_block = proj_a_unit - row_block * (O_LORA // O_A_N_TILE)
            t0 = row_block * O_A_T_TILE
            n0 = n_block * O_A_N_TILE
            a_rows = pl.min(O_A_T_TILE, group_t - t0)
            src_row = attention_row + t0
            weight_row = o_a_col + n0
            o_a_x0 = pl.slice(attn_2d, [O_A_T_TILE, O_A_K_TILE], [src_row, 0], valid_shape=[a_rows, O_A_K_TILE])
            o_a_w0 = wo_a_flat[weight_row:weight_row + O_A_N_TILE, 0:O_A_K_TILE]
            o_a_acc = pl.matmul(o_a_x0, o_a_w0, b_trans=True, out_dtype=pl.FP32)
            for k0 in pl.pipeline(O_A_K_TILE, O_GROUP_IN, O_A_K_TILE, stage=2):
                o_a_xk = pl.slice(attn_2d, [O_A_T_TILE, O_A_K_TILE], [src_row, k0], valid_shape=[a_rows, O_A_K_TILE])
                o_a_wk = wo_a_flat[weight_row:weight_row + O_A_N_TILE, k0:k0 + O_A_K_TILE]
                o_a_acc = pl.matmul_acc(o_a_acc, o_a_xk, o_a_wk, b_trans=True)
            o_a_valid = pl.set_validshape(o_a_acc, a_rows, O_A_N_TILE)
            o_a_fp32[t0:t0 + O_A_T_TILE, weight_row:weight_row + O_A_N_TILE] = o_a_valid
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='tp_o_a_quant', deps=[proj_a_tid]) as quant_tid:
            for qt in pl.pipeline(0, group_t, QUANT_T_TILE, stage=2):
                quant_rows = pl.min(QUANT_T_TILE, group_t - qt)
                o_a_tile = pl.slice(o_a_fp32, [QUANT_T_TILE, O_LORA], [qt, o_a_col], valid_shape=[quant_rows, O_LORA])
                o_a_abs = pl.abs(o_a_tile)
                row_amax = pl.reshape(pl.row_max(o_a_abs), [1, QUANT_T_TILE])
                amax_floor = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                row_amax = pl.maximum(amax_floor, row_amax)
                scale_max = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                scale_q_row = pl.div(scale_max, row_amax)
                scale_dq_row = pl.recip(scale_q_row)
                scale_dq_valid = pl.set_validshape(scale_dq_row, 1, quant_rows)
                act_scale_dq[local_group:local_group + 1, qt:qt + QUANT_T_TILE] = scale_dq_valid
                scale_q_col = pl.reshape(scale_q_row, [QUANT_T_TILE, 1])
                o_a_scaled = pl.row_expand_mul(o_a_tile, scale_q_col)
                o_a_i32 = pl.cast(o_a_scaled, target_type=pl.INT32, mode='rint')
                o_a_fp16 = pl.cast(o_a_i32, target_type=pl.FP16, mode='round')
                o_a_fp16_full = pl.set_validshape(o_a_fp16, QUANT_T_TILE, O_LORA)
                o_a_quant = pl.cast(o_a_fp16_full, target_type=pl.INT8, mode='trunc')
                o_a_quant_valid = pl.set_validshape(o_a_quant, quant_rows, O_LORA)
                o_a_i8[qt:qt + QUANT_T_TILE, o_a_col:o_a_col + O_LORA] = o_a_quant_valid
            for qt in pl.range(group_t, o_b_group_t, QUANT_T_TILE):
                pad_rows = pl.min(QUANT_T_TILE, o_b_group_t - qt)
                o_a_padding_fp16 = pl.full([QUANT_T_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                o_a_padding = pl.cast(o_a_padding_fp16, target_type=pl.INT8, mode='trunc')
                o_a_padding_valid = pl.set_validshape(o_a_padding, pad_rows, O_LORA)
                o_a_i8[qt:qt + QUANT_T_TILE, o_a_col:o_a_col + O_LORA] = o_a_padding_valid
        with pl.spmd(o_b_rows * (D // O_B_D_TILE), name_hint='tp_o_b', deps=[quant_tid]) as proj_b_tid:
            proj_b_unit = pl.tile.get_block_idx()
            row_block = proj_b_unit // (D // O_B_D_TILE)
            d_block = proj_b_unit - row_block * (D // O_B_D_TILE)
            t0 = row_block * O_B_T_TILE
            d0 = d_block * O_B_D_TILE
            for n0 in pl.range(d0, d0 + O_B_D_TILE, O_B_N_TILE):
                o_b_x0 = o_a_i8[t0:t0 + O_B_T_TILE, o_a_col:o_a_col + O_B_K_TILE]
                o_b_w0 = wo_b[n0:n0 + O_B_N_TILE, o_a_col:o_a_col + O_B_K_TILE]
                o_b_acc = pl.matmul(o_b_x0, o_b_w0, b_trans=True, out_dtype=pl.INT32)
                for k0 in pl.pipeline(O_B_K_TILE, O_LORA, O_B_K_TILE, stage=2):
                    b_k0 = o_a_col + k0
                    o_b_xk = o_a_i8[t0:t0 + O_B_T_TILE, b_k0:b_k0 + O_B_K_TILE]
                    o_b_wk = wo_b[n0:n0 + O_B_N_TILE, b_k0:b_k0 + O_B_K_TILE]
                    o_b_acc = pl.matmul_acc(o_b_acc, o_b_xk, o_b_wk, b_trans=True)
                partial_col = local_group * D + n0
                o_b_i32[t0:t0 + O_B_T_TILE, partial_col:partial_col + O_B_N_TILE] = o_b_acc
        proj_b_tids[local_group] = proj_b_tid
    publish_stage = pl.create_tensor([O_RS_PUBLISH_WORKERS * ACT_T_TILE, ACT_N_TILE], dtype=pl.FP32)
    with pl.spmd(O_RS_PUBLISH_WORKERS, name_hint='tp_o_b_dequant_publish', deps=[proj_b_tids[group] for group in range(LOCAL_O_GROUPS)], optimizations=[pl.cross_core_slot(slot_num=2)]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for owner_rank in pl.range(TP_SIZE):
            owner_base = owner_rank * local_t
            for block in pl.range(worker, owner_rows * (D // ACT_N_TILE), O_RS_PUBLISH_WORKERS):
                row_block = block // (D // ACT_N_TILE)
                n_block = block - row_block * (D // ACT_N_TILE)
                owner_row = row_block * ACT_T_TILE
                t0 = owner_base + owner_row
                n0 = n_block * ACT_N_TILE
                out_rows = pl.min(ACT_T_TILE, local_t - owner_row)
                dequant_acc = pl.full([ACT_T_TILE, ACT_N_TILE], dtype=pl.FP32, value=0.0)
                for local_group in pl.pipeline(LOCAL_O_GROUPS, stage=2):
                    part_col = local_group * D + n0
                    group_i32 = pl.slice(o_b_i32, [ACT_T_TILE, ACT_N_TILE], [t0, part_col], valid_shape=[out_rows, ACT_N_TILE])
                    group_partial_fp32 = pl.cast(group_i32, target_type=pl.FP32, mode='none')
                    scale_row = pl.slice(act_scale_dq, [1, ACT_T_TILE], [local_group, t0], valid_shape=[1, out_rows])
                    scale_col = pl.reshape(scale_row, [ACT_T_TILE, 1])
                    group_dequant = pl.row_expand_mul(group_partial_fp32, scale_col)
                    dequant_acc = pl.add(dequant_acc, group_dequant)
                weight_scale = pl.reshape(wo_b_scale[n0:n0 + ACT_N_TILE], [1, ACT_N_TILE])
                o_b_fp32 = pl.col_expand_mul(dequant_acc, weight_scale)
                publish_value = pl.set_validshape(o_b_fp32, out_rows, ACT_N_TILE)
                stage_row = worker * ACT_T_TILE
                publish_stage[stage_row:stage_row + ACT_T_TILE, 0:ACT_N_TILE] = publish_value
                target_row = tp_rank * LOCAL_T_PAD + owner_row
                pld.tensor.put(dst=reduce_window, peer=group_base + owner_rank, src=publish_stage, dst_offsets=[target_row, n0], src_offsets=[stage_row, 0], shape=[out_rows, ACT_N_TILE], chunk_rows=ACT_T_TILE, chunk_cols=ACT_N_TILE)
        for owner_rank in pl.range(TP_SIZE):
            if owner_rank != tp_rank:
                pld.system.notify(target=reduce_signal, peer=group_base + owner_rank, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='tp_o_rs_wait', deps=[publish_tid]) as wait_tid:
        expected = pl.cast(O_RS_PUBLISH_WORKERS, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(signal=reduce_signal, offsets=[source_tp, 0], expected=expected, cmp=pld.WaitCmp.Ge)
    with pl.spmd(O_RS_REDUCE_WORKERS, name_hint='tp_o_rs_reduce', deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, local_t * (D // O_RS_D_TILE), O_RS_REDUCE_WORKERS):
            local_row = block // (D // O_RS_D_TILE)
            d_block = block - local_row * (D // O_RS_D_TILE)
            d0 = d_block * O_RS_D_TILE
            reduce_acc = pl.load(reduce_window, [local_row, d0], [1, O_RS_D_TILE])
            for source_tp in pl.range(1, TP_SIZE):
                source_row = source_tp * LOCAL_T_PAD + local_row
                source_partial = pl.load(reduce_window, [source_row, d0], [1, O_RS_D_TILE])
                reduce_acc = pl.add(reduce_acc, source_partial)
            reduced = pl.cast(reduce_acc, target_type=pl.BF16, mode='rint')
            pl.store(reduced, [local_row, d0], local_out)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='tp_o_rs_complete', deps=[reduce_tid]):
        completion_anchor = pl.read(local_out, [0, 0])
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(target=reduce_signal, peer=group_base + peer_tp, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        completion_expected = pl.cast(O_RS_PUBLISH_WORKERS + 1, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(signal=reduce_signal, offsets=[source_tp, 0], expected=completion_expected, cmp=pld.WaitCmp.Ge)
        reset_value = pl.cast(-(O_RS_PUBLISH_WORKERS + 1), pl.INT32)
        self_rank = group_base + tp_rank
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(target=reduce_signal, peer=self_rank, offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd)
        pl.write(local_out, [0, 0], completion_anchor)
    return (local_out, reduce_signal)

@pl.jit.incore
def o_group_a2a(attention_grouped: pl.Tensor[[O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], pl.BF16], local_groups_out: pl.InOut[pl.Tensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16]], exchange_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16], exchange_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32]):
    """Exchange valid output-group rows into local-group, group-token order."""
    group_t = TP_SIZE * local_t
    for destination_rank in pl.range(TP_SIZE):
        for local_group in pl.range(LOCAL_O_GROUPS):
            global_group = destination_rank * LOCAL_O_GROUPS + local_group
            source_row = global_group * LOCAL_T_PAD
            target_row = local_group * GROUP_T_PAD + tp_rank * local_t
            pld.tensor.put(dst=exchange_window, peer=group_base + destination_rank, src=attention_grouped, dst_offsets=[target_row, 0], src_offsets=[source_row, 0], shape=[local_t, O_GROUP_IN], chunk_rows=COMM_ROW_TILE, chunk_cols=O_GROUP_IN)
    expected_one = pl.cast(1, pl.INT32)
    exchange_signal = tp_group_barrier(exchange_signal, group_base, tp_rank, expected_one)
    for local_group in pl.range(LOCAL_O_GROUPS):
        group_base_row = local_group * GROUP_T_PAD
        for group_row in pl.range(group_t):
            copy_row = group_base_row + group_row
            window_row = exchange_window[copy_row:copy_row + 1, 0:O_GROUP_IN]
            local_groups_out[copy_row:copy_row + 1, 0:O_GROUP_IN] = window_row
    expected_two = pl.cast(2, pl.INT32)
    exchange_signal = tp_group_barrier(exchange_signal, group_base, tp_rank, expected_two)
    exchange_signal = reset_tp_group_signal(exchange_signal, group_base, tp_rank)
    return (local_groups_out, exchange_signal)

@pl.jit.inline
def reset_tp_group_signal(signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32]):
    """Clear this rank's peer-credit cells after a collective completes."""
    reset_value = pl.cast(-2, pl.INT32)
    self_rank = group_base + tp_rank
    for source_tp in pl.range(TP_SIZE):
        if source_tp != tp_rank:
            pld.system.notify(target=signal, peer=self_rank, offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd)
    return signal

@pl.jit.inline
def tp_group_barrier(signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], expected: pl.Scalar[pl.INT32]):
    """Synchronize one contiguous physical TP group."""
    for peer_tp in pl.range(TP_SIZE):
        if peer_tp != tp_rank:
            pld.system.notify(target=signal, peer=group_base + peer_tp, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    for source_tp in pl.range(TP_SIZE):
        if source_tp != tp_rank:
            pld.system.wait(signal=signal, offsets=[source_tp, 0], expected=expected, cmp=pld.WaitCmp.Ge)
    return signal
