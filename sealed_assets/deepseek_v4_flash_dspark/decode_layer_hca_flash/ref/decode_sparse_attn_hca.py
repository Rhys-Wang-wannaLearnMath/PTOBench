"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_hca_flash."""

from config import TP
from config import CASE_CONFIG as M
from config import COMPRESS_RATIO
from config import DECODE_BATCH
from config import DECODE_SEQ
from config import BLOCK_SIZE
import pypto.language as pl

B_DYN = pl.dynamic('B_DYN')
T_DYN = pl.dynamic('T_DYN')
ORI_BLOCK_NUM_DYN = pl.dynamic('ORI_BLOCK_NUM_DYN')
CMP_BLOCK_NUM_DYN = pl.dynamic('CMP_BLOCK_NUM_DYN')
CMP_TABLE_BLOCKS_DYN = pl.dynamic('CMP_TABLE_BLOCKS_DYN')
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
NEG_INF = -1e+20
HCA_MAX_COMPRESSED_ROWS = 1048576 // COMPRESS_RATIO
GATHER_SEGMENTS = 4
GATHER_RUN_TILE = 16
GATHER_WINDOW_TILE = WIN // GATHER_SEGMENTS
RAW_K_TILE = BLOCK_SIZE
ATTN_K_TILE = 128
ATTN_D_TILE = 256
H_TILE = 16
QK_M_TILE = 32
CMP_PAGES_PER_WORK = ATTN_K_TILE // BLOCK_SIZE
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
ROPE_CS_T_TILE = 8
T_PAD = (T + 16 - 1) // 16 * 16

@pl.jit.inline(auto_scope=False)
def sparse_attn_hca(q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16], ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32], position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16], o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16], cache_ready_dep: pl.Scalar[pl.TASK_ID]) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write HCA heads as grouped ``[T_PAD, O_GROUP_IN]`` slabs."""
    t_dim = pl.tensor.dim(q, 0)
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_table_blocks = pl.tensor.dim(cmp_block_table, 1)
    cmp_work_count = (cmp_table_blocks + CMP_PAGES_PER_WORK - 1) // CMP_PAGES_PER_WORK
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    request_count = pl.tensor.dim(cmp_block_table, 0)
    raw_gather_count = t_dim * GATHER_SEGMENTS
    raw_tile_count = t_dim * (WIN // RAW_K_TILE)
    stream_block_count = t_dim * (H // H_TILE)
    cmp_gather_count = request_count * cmp_work_count
    cmp_qk_block_count = t_dim * cmp_work_count
    cmp_partial_rows = t_dim * (H // H_TILE) * cmp_work_count * H_TILE
    stream_state_m = pl.create_tensor([t_dim * H, 1], dtype=pl.FP32)
    stream_state_l = pl.create_tensor([t_dim * H, 1], dtype=pl.FP32)
    stream_heads = pl.create_tensor([t_dim * H, HEAD_DIM], dtype=pl.FP32)
    cmp_partial_m = pl.create_tensor([cmp_partial_rows, 8], dtype=pl.FP32)
    cmp_partial_l = pl.create_tensor([cmp_partial_rows, 8], dtype=pl.FP32)
    cmp_partial_o = pl.create_tensor([cmp_partial_rows, HEAD_DIM], dtype=pl.FP32)
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([1, ROPE_DIM], dtype=pl.INT32)
    raw_branch_tids = pl.array.create(1, pl.TASK_ID)
    cmp_branch_tids = pl.array.create(1, pl.TASK_ID)
    rope_swap_tids = pl.array.create(1, pl.TASK_ID)
    rope_cs_tids = pl.array.create(1, pl.TASK_ID)
    with pl.scope():
        raw_kv = pl.create_tensor([t_dim * WIN, HEAD_DIM], dtype=pl.BF16)
        raw_kv_t = pl.create_tensor([raw_tile_count * HEAD_DIM, RAW_K_TILE], dtype=pl.BF16)
        raw_valid = pl.create_tensor([t_dim, WIN], dtype=pl.FP32)
        with pl.spmd(raw_gather_count, name_hint='hca_gather_kv', deps=[cache_ready_dep]) as raw_gather_tid:
            g_task = pl.tile.get_block_idx()
            g_t = g_task // GATHER_SEGMENTS
            g_seg = g_task - g_t * GATHER_SEGMENTS
            g_wk0 = g_seg * GATHER_WINDOW_TILE
            g_row0 = g_t * WIN
            for g_sub in pl.range(GATHER_WINDOW_TILE // GATHER_RUN_TILE):
                g_sk0 = g_wk0 + g_sub * GATHER_RUN_TILE
                g_sdst = g_row0 + g_sk0
                g_first = pl.read(window_swa_indices, [g_t, g_sk0])
                g_run_matches = pl.cast(g_first >= 0, pl.INT32)
                for g_dr in pl.unroll(GATHER_RUN_TILE):
                    g_slot_i32 = pl.read(window_swa_indices, [g_t, g_sk0 + g_dr])
                    g_run_matches = g_run_matches * pl.cast(g_slot_i32 == g_first + g_dr, pl.INT32)
                if g_run_matches == 1:
                    g_run_src = pl.cast(g_first, pl.INDEX)
                    raw_kv[g_sdst:g_sdst + GATHER_RUN_TILE, 0:HEAD_DIM] = ori_kv_flat[g_run_src:g_run_src + GATHER_RUN_TILE, 0:HEAD_DIM]
                    for g_dr in pl.unroll(GATHER_RUN_TILE):
                        pl.write(raw_valid, [g_t, g_sk0 + g_dr], 1.0)
                else:
                    for g_dr in pl.range(GATHER_RUN_TILE):
                        g_lane = g_sk0 + g_dr
                        g_dst = g_row0 + g_lane
                        g_slot_i32 = pl.read(window_swa_indices, [g_t, g_lane])
                        if g_slot_i32 >= 0:
                            g_slot = pl.cast(g_slot_i32, pl.INDEX)
                            raw_kv[g_dst:g_dst + 1, 0:HEAD_DIM] = ori_kv_flat[g_slot:g_slot + 1, 0:HEAD_DIM]
                            pl.write(raw_valid, [g_t, g_lane], 1.0)
                        else:
                            raw_kv[g_dst:g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                            pl.write(raw_valid, [g_t, g_lane], 0.0)
        with pl.spmd(raw_tile_count, name_hint='hca_raw_transpose', deps=[raw_gather_tid]) as raw_transpose_tid:
            raw_tile = pl.tile.get_block_idx()
            raw_row = raw_tile * RAW_K_TILE
            raw_t_row = raw_tile * HEAD_DIM
            raw_kv_t[raw_t_row:raw_t_row + HEAD_DIM, 0:RAW_K_TILE] = pl.transpose(raw_kv[raw_row:raw_row + RAW_K_TILE, 0:HEAD_DIM], axis1=0, axis2=1)
        with pl.spmd(1, name_hint='rope_swap') as rope_swap_tid:
            sw_block = pl.tile.get_block_idx()
            sw_one = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=1.0)
            sw_index_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
            sw_index = pl.cast(sw_index_i32, target_type=pl.FP32)
            sw_col = pl.col_expand_mul(sw_one, sw_index)
            sw_dup = pl.mul(sw_col, 0.5)
            sw_dup_i32 = pl.cast(sw_dup, target_type=pl.INT32, mode='trunc')
            sw_dup_f = pl.cast(sw_dup_i32, target_type=pl.FP32)
            sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
            sw_next = pl.add(sw_col, 1.0)
            sw_lane_offset = pl.mul(sw_lane, 2.0)
            sw_swap = pl.sub(sw_next, sw_lane_offset)
            rope_swap_idx[sw_block:sw_block + 1, 0:ROPE_DIM] = pl.cast(sw_swap, target_type=pl.INT32)
        with pl.spmd(HALF_ROPE // ROPE_TILE, name_hint='rope_cs') as rope_cs_tid:
            cp = pl.tile.get_block_idx()
            cp_r0 = cp * ROPE_TILE
            cp_c0 = 2 * cp_r0
            cs_one = pl.full([ROPE_CS_T_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
            cs_index_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
            cs_index = pl.cast(cs_index_i32, target_type=pl.FP32)
            cs_col = pl.col_expand_mul(cs_one, cs_index)
            cs_dup = pl.mul(cs_col, 0.5)
            cs_dup_idx = pl.cast(cs_dup, target_type=pl.INT32, mode='trunc')
            cs_dup_f = pl.cast(cs_dup_idx, target_type=pl.FP32)
            cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
            cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
            for cs_rb in pl.range(rope_cs_blocks):
                cs_t0 = cs_rb * ROPE_CS_T_TILE
                cs_cos_bf16 = freqs_cos[cs_t0:cs_t0 + ROPE_CS_T_TILE, cp_r0:cp_r0 + ROPE_TILE]
                cs_sin_bf16 = freqs_sin[cs_t0:cs_t0 + ROPE_CS_T_TILE, cp_r0:cp_r0 + ROPE_TILE]
                cs_cos = pl.cast(cs_cos_bf16, target_type=pl.FP32)
                cs_sin = pl.cast(cs_sin_bf16, target_type=pl.FP32)
                cs_cos_dup = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
                cs_sin_dup = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
                cs_sin_signed = pl.mul(cs_sin_dup, cs_sign)
                rope_cos_il[cs_t0:cs_t0 + ROPE_CS_T_TILE, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_dup
                rope_sin_signed[cs_t0:cs_t0 + ROPE_CS_T_TILE, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = cs_sin_signed
        with pl.spmd(stream_block_count, name_hint='hca_raw_attn', deps=[raw_transpose_tid]) as raw_heads_tid:
            stream_idx = pl.tile.get_block_idx()
            stream_t = stream_idx // (H // H_TILE)
            stream_h_tile = stream_idx - stream_t * (H // H_TILE)
            stream_h0 = stream_h_tile * H_TILE
            stream_state_row = stream_t * H + stream_h0
            stream_q = pl.load(q_flat, [stream_state_row, 0], [H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
            stream_row_max_tmp = pl.create_tile([H_TILE, RAW_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            stream_row_sum_tmp = pl.create_tile([H_TILE, RAW_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            for stream_raw_item in pl.unroll(WIN // RAW_K_TILE):
                stream_raw_begin = stream_t * WIN + stream_raw_item * RAW_K_TILE
                stream_valid_begin = stream_raw_item * RAW_K_TILE
                stream_raw_kv = pl.load(raw_kv, [stream_raw_begin, 0], [RAW_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
                stream_raw_t_begin = (stream_t * (WIN // RAW_K_TILE) + stream_raw_item) * HEAD_DIM
                stream_raw_kv_t_left = pl.load(raw_kv_t, [stream_raw_t_begin, 0], [ATTN_D_TILE, RAW_K_TILE], target_memory=pl.MemorySpace.Vec)
                stream_raw_kv_t_right = pl.load(raw_kv_t, [stream_raw_t_begin + ATTN_D_TILE, 0], [ATTN_D_TILE, RAW_K_TILE], target_memory=pl.MemorySpace.Vec)
                stream_raw_valid_row = pl.load(raw_valid, [stream_t, stream_valid_begin], [1, RAW_K_TILE], target_memory=pl.MemorySpace.Vec)
                stream_raw_valid_zero = pl.tile.full([H_TILE, RAW_K_TILE], dtype=pl.FP32, value=0.0)
                stream_raw_valid = pl.col_expand_add(stream_raw_valid_zero, stream_raw_valid_row)
                stream_raw_bias = pl.mul(pl.sub(stream_raw_valid, 1.0), -NEG_INF)
                stream_raw_scores = pl.matmul(stream_q[:, 0:ATTN_D_TILE], stream_raw_kv_t_left, out_dtype=pl.FP32)
                stream_raw_scores = pl.matmul_acc(stream_raw_scores, stream_q[:, ATTN_D_TILE:HEAD_DIM], stream_raw_kv_t_right)
                stream_raw_scores = pl.add(pl.mul(stream_raw_scores, SOFTMAX_SCALE), stream_raw_bias)
                stream_raw_mi_col = pl.row_max(stream_raw_scores, stream_row_max_tmp)
                stream_raw_exp = pl.exp(pl.row_expand_sub(stream_raw_scores, stream_raw_mi_col))
                stream_raw_exp = pl.mul(stream_raw_exp, stream_raw_valid)
                stream_raw_li_col = pl.row_sum(stream_raw_exp, stream_row_sum_tmp)
                stream_raw_exp_bf16 = pl.cast(stream_raw_exp, target_type=pl.BF16, mode='rint')
                stream_raw_oi_left = pl.matmul(stream_raw_exp_bf16, stream_raw_kv[:, 0:ATTN_D_TILE], out_dtype=pl.FP32)
                stream_raw_oi_right = pl.matmul(stream_raw_exp_bf16, stream_raw_kv[:, ATTN_D_TILE:2 * ATTN_D_TILE], out_dtype=pl.FP32)
                stream_raw_oi = pl.concat(stream_raw_oi_left, stream_raw_oi_right)
                if stream_raw_item == 0:
                    pl.store(stream_raw_mi_col, [stream_state_row, 0], stream_state_m)
                    pl.store(stream_raw_li_col, [stream_state_row, 0], stream_state_l)
                    pl.store(stream_raw_oi, [stream_state_row, 0], stream_heads)
                else:
                    stream_m = pl.load(stream_state_m, [stream_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
                    stream_l = pl.load(stream_state_l, [stream_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
                    stream_o = pl.load(stream_heads, [stream_state_row, 0], [H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
                    stream_m_new = pl.maximum(stream_m, stream_raw_mi_col)
                    stream_alpha = pl.exp(pl.sub(stream_m, stream_m_new))
                    stream_beta = pl.exp(pl.sub(stream_raw_mi_col, stream_m_new))
                    stream_l_new = pl.add(pl.mul(stream_alpha, stream_l), pl.mul(stream_beta, stream_raw_li_col))
                    stream_o_new = pl.add(pl.row_expand_mul(stream_o, stream_alpha), pl.row_expand_mul(stream_raw_oi, stream_beta))
                    pl.store(stream_m_new, [stream_state_row, 0], stream_state_m)
                    pl.store(stream_l_new, [stream_state_row, 0], stream_state_l)
                    pl.store(stream_o_new, [stream_state_row, 0], stream_heads)
        raw_branch_tids[0] = raw_heads_tid
        rope_swap_tids[0] = rope_swap_tid
        rope_cs_tids[0] = rope_cs_tid
    with pl.scope():
        cmp_work_kv = pl.create_tensor([cmp_gather_count * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16)
        with pl.spmd(cmp_gather_count, name_hint='hca_cmp_work_gather', deps=[cache_ready_dep]) as cmp_gather_tid:
            gather_item = pl.tile.get_block_idx()
            gather_request = gather_item // cmp_work_count
            gather_work = gather_item - gather_request * cmp_work_count
            gather_first_col = gather_work * CMP_PAGES_PER_WORK
            gather_dst0 = gather_item * ATTN_K_TILE
            for gather_page in pl.unroll(CMP_PAGES_PER_WORK):
                gather_page_col = gather_first_col + gather_page
                gather_dst = gather_dst0 + gather_page * BLOCK_SIZE
                cmp_work_kv[gather_dst:gather_dst + BLOCK_SIZE, 0:HEAD_DIM] = pl.full([BLOCK_SIZE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                if gather_page_col < cmp_table_blocks:
                    gather_page_i32 = pl.read(cmp_block_table, [gather_request, gather_page_col])
                    if gather_page_i32 >= 0:
                        if gather_page_i32 < cmp_block_num:
                            gather_page_id = pl.cast(gather_page_i32, pl.INDEX)
                            gather_src = gather_page_id * BLOCK_SIZE
                            cmp_work_kv[gather_dst:gather_dst + BLOCK_SIZE, 0:HEAD_DIM] = cmp_kv_flat[gather_src:gather_src + BLOCK_SIZE, 0:HEAD_DIM]
        with pl.spmd(cmp_qk_block_count, name_hint='hca_cmp_qk_pv', deps=[cmp_gather_tid]) as cmp_qk_tid:
            qk_item = pl.tile.get_block_idx()
            qk_t = qk_item // cmp_work_count
            qk_work = qk_item - qk_t * cmp_work_count
            qk_request = qk_t // S
            qk_work_row = qk_work * ATTN_K_TILE
            qk_token_base = qk_t * (H // H_TILE) * cmp_work_count * H_TILE
            qk_neutral_m = pl.tile.full([H_TILE, 8], dtype=pl.FP32, value=NEG_INF)
            qk_neutral_l = pl.tile.full([H_TILE, 8], dtype=pl.FP32, value=0.0)
            qk_neutral_o = pl.tile.full([H_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            for qk_h_idx in pl.unroll(H // H_TILE):
                qk_row = qk_token_base + qk_h_idx * cmp_work_count * H_TILE + qk_work * H_TILE
                pl.store(qk_neutral_m, [qk_row, 0], cmp_partial_m)
                pl.store(qk_neutral_l, [qk_row, 0], cmp_partial_l)
                pl.store(qk_neutral_o, [qk_row, 0], cmp_partial_o)
            qk_rows = pl.cast(0, pl.INDEX)
            qk_position_i32 = pl.read(position_ids, [qk_t])
            if qk_request < request_count:
                qk_kv_len_i32 = pl.read(kv_seq_lens, [qk_request])
                if qk_position_i32 >= 0:
                    if qk_kv_len_i32 >= 0:
                        qk_position = pl.cast(qk_position_i32, pl.INDEX)
                        qk_kv_len = pl.cast(qk_kv_len_i32, pl.INDEX)
                        qk_position_rows = (qk_position + 1) // COMPRESS_RATIO
                        qk_kv_rows = qk_kv_len // COMPRESS_RATIO
                        qk_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(qk_position_rows, qk_kv_rows))
            if qk_work_row < qk_rows:
                qk_first_col = qk_work * CMP_PAGES_PER_WORK
                qk_first_page_i32 = pl.read(cmp_block_table, [qk_request, qk_first_col])
                if qk_first_page_i32 >= 0:
                    if qk_first_page_i32 < cmp_block_num:
                        qk_work_src = (qk_request * cmp_work_count + qk_work) * ATTN_K_TILE
                        qk_kv = pl.load(cmp_work_kv, [qk_work_src, 0], [ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                        qk_valid_rows = pl.min(ATTN_K_TILE, qk_rows - qk_work_row)
                        qk_kv_t = pl.tile.transpose_view(qk_kv)
                        qk_row_max_tmp = pl.create_tile([QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                        qk_row_sum_tmp = pl.create_tile([QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                        for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                            qk_h0 = qk_hb * QK_M_TILE
                            qk_head_row = qk_t * H + qk_h0
                            qk_q = pl.load(q_flat, [qk_head_row, 0], [QK_M_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                            qk_scores = pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)
                            qk_scores = pl.mul(qk_scores, SOFTMAX_SCALE)
                            qk_scores = pl.set_validshape(qk_scores, QK_M_TILE, qk_valid_rows)
                            qk_scores = pl.fillpad(qk_scores, pad_value=pl.PadValue.min)
                            qk_mi = pl.row_max(qk_scores, qk_row_max_tmp)
                            qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                            qk_li = pl.row_sum(qk_exp, qk_row_sum_tmp)
                            qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode='rint')
                            qk_oi_left = pl.matmul(qk_exp_bf16, qk_kv[:, 0:ATTN_D_TILE], out_dtype=pl.FP32)
                            qk_oi_right = pl.matmul(qk_exp_bf16, qk_kv[:, ATTN_D_TILE:HEAD_DIM], out_dtype=pl.FP32)
                            qk_oi = pl.concat(qk_oi_left, qk_oi_right)
                            qk_h_idx0 = qk_hb * (QK_M_TILE // H_TILE)
                            qk_row0 = qk_token_base + qk_h_idx0 * cmp_work_count * H_TILE + qk_work * H_TILE
                            qk_row1 = qk_row0 + cmp_work_count * H_TILE
                            pl.store(qk_mi[0:H_TILE, 0:1], [qk_row0, 0], cmp_partial_m)
                            pl.store(qk_li[0:H_TILE, 0:1], [qk_row0, 0], cmp_partial_l)
                            pl.store(qk_oi[0:H_TILE, 0:HEAD_DIM], [qk_row0, 0], cmp_partial_o)
                            pl.store(qk_mi[H_TILE:QK_M_TILE, 0:1], [qk_row1, 0], cmp_partial_m)
                            pl.store(qk_li[H_TILE:QK_M_TILE, 0:1], [qk_row1, 0], cmp_partial_l)
                            pl.store(qk_oi[H_TILE:QK_M_TILE, 0:HEAD_DIM], [qk_row1, 0], cmp_partial_o)
        cmp_branch_tids[0] = cmp_qk_tid
    with pl.spmd(stream_block_count, name_hint='hca_stream_merge', deps=[raw_branch_tids[0], cmp_branch_tids[0]]) as stream_heads_tid:
        stream_idx = pl.tile.get_block_idx()
        stream_t = stream_idx // (H // H_TILE)
        stream_h_tile = stream_idx - stream_t * (H // H_TILE)
        stream_h0 = stream_h_tile * H_TILE
        stream_state_row = stream_t * H + stream_h0
        stream_m = pl.load(stream_state_m, [stream_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
        stream_l = pl.load(stream_state_l, [stream_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
        stream_o = pl.load(stream_heads, [stream_state_row, 0], [H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
        stream_token_base = stream_t * (H // H_TILE) * cmp_work_count * H_TILE
        for stream_work in pl.range(cmp_work_count):
            stream_partial_row = stream_token_base + stream_h_tile * cmp_work_count * H_TILE + stream_work * H_TILE
            stream_cmp_m_aligned = pl.load(cmp_partial_m, [stream_partial_row, 0], [H_TILE, 8], target_memory=pl.MemorySpace.Vec)
            stream_cmp_l_aligned = pl.load(cmp_partial_l, [stream_partial_row, 0], [H_TILE, 8], target_memory=pl.MemorySpace.Vec)
            stream_cmp_m = stream_cmp_m_aligned[0:H_TILE, 0:1]
            stream_cmp_l = stream_cmp_l_aligned[0:H_TILE, 0:1]
            stream_cmp_o = pl.load(cmp_partial_o, [stream_partial_row, 0], [H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
            stream_m_new = pl.maximum(stream_m, stream_cmp_m)
            stream_alpha = pl.exp(pl.sub(stream_m, stream_m_new))
            stream_beta = pl.exp(pl.sub(stream_cmp_m, stream_m_new))
            stream_l = pl.add(pl.mul(stream_alpha, stream_l), pl.mul(stream_beta, stream_cmp_l))
            stream_o = pl.add(pl.row_expand_mul(stream_o, stream_alpha), pl.row_expand_mul(stream_cmp_o, stream_beta))
            stream_m = stream_m_new
        stream_sink = pl.load(attn_sink_col, [stream_h0, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
        stream_sink_tile = pl.add(pl.sub(stream_m, stream_m), stream_sink)
        stream_denom = pl.add(stream_l, pl.exp(pl.sub(stream_sink_tile, stream_m)))
        stream_output = pl.row_expand_div(stream_o, stream_denom)
        pl.store(stream_output, [stream_state_row, 0], stream_heads)
    with pl.spmd(stream_block_count, name_hint='hca_stream_pack', deps=[stream_heads_tid, rope_swap_tids[0], rope_cs_tids[0]]) as heads_tid:
        stream_idx = pl.tile.get_block_idx()
        stream_t = stream_idx // (H // H_TILE)
        stream_h_tile = stream_idx - stream_t * (H // H_TILE)
        stream_h0 = stream_h_tile * H_TILE
        stream_state_row = stream_t * H + stream_h0
        stream_output = stream_heads[stream_state_row:stream_state_row + H_TILE, 0:HEAD_DIM]
        stream_bf16 = pl.cast(stream_output, target_type=pl.BF16, mode='rint')
        stream_rope = stream_output[0:H_TILE, NOPE_DIM:HEAD_DIM]
        stream_cos_il = rope_cos_il[stream_t:stream_t + 1, 0:ROPE_DIM]
        stream_sin_signed = rope_sin_signed[stream_t:stream_t + 1, 0:ROPE_DIM]
        stream_swap_zero = pl.full([H_TILE, ROPE_DIM], dtype=pl.INT32, value=0)
        stream_swap_idx = pl.col_expand_add(stream_swap_zero, rope_swap_idx[0:1, 0:ROPE_DIM])
        stream_swapped = pl.gather(stream_rope, dim=-1, index=stream_swap_idx)
        stream_rot = pl.add(pl.col_expand_mul(stream_rope, stream_cos_il), pl.col_expand_mul(stream_swapped, stream_sin_signed))
        n_rope_bf16 = pl.cast(stream_rot, target_type=pl.BF16, mode='rint')
        n_full_bf16 = pl.concat(stream_bf16[0:H_TILE, 0:NOPE_DIM], n_rope_bf16)
        for n_hi in pl.unroll(H_TILE):
            n_head = stream_h0 + n_hi
            n_pack_row = n_head // HEADS_PER_GROUP * T_PAD + stream_t
            n_col = n_head % HEADS_PER_GROUP * HEAD_DIM
            o_packed_heads[n_pack_row:n_pack_row + 1, n_col:n_col + HEAD_DIM] = n_full_bf16[n_hi:n_hi + 1, 0:HEAD_DIM]
    return (o_packed_heads, heads_tid)
