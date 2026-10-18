"""Evaluated NPU runtime closure for deepseek_v4_flash_dspark.decode_layer_csa_demo."""

from config import CASE_CONFIG as M
from config import TP
from config import COMPRESS_RATIO
from config import MAX_SEQ_LEN
from config import DECODE_BATCH
from config import DECODE_SEQ
from config import FP32_NEG_INF
from config import BLOCK_SIZE
from config import INT8_SCALE_MAX
from config import C4A_COMPRESSOR_BLOCK_SIZE
from config import INT8_AMAX_EPS
import pypto.language as pl
from decode_indexer_compressor import indexer_compressor

B_DYN = pl.dynamic('B_DYN')
T_DYN = pl.dynamic('T_DYN')
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_LEN = INNER_COFF * COMPRESS_RATIO
INNER_STATE_MAX_BLOCKS = (INNER_STATE_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic('INNER_STATE_BLOCK_NUM_DYN')
INNER_STATE_DIM = 2 * INNER_OUT_DIM
IDX_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_MAX_BLOCKS = (IDX_MAX_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic('IDX_CACHE_BLOCK_NUM_DYN')
Q_TILE = 256
Q_OUT_TILE = 1024
T_PAD = (T + 16 - 1) // 16 * 16
MM_ROW_TILE = 16
MM_N_TILE = min(512, 128 * 1024 // (MM_ROW_TILE * 4))
QR_OT_COUNT = IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE
DEQUANT_T_TILE = min(T, 8)
D_TILE = 512
WEIGHTS_OK = 4
WEIGHTS_K_SLICE = D // WEIGHTS_OK
QH_QUANT_TILE = 64
QH_MM_TILE = 64
QH_HEAD_DIM_TILE = 64
ROPE_ROW_BLOCK = IDX_N_HEADS
ROPE_ROW_TILE = 32
TOPK_PAIR_WIDTH = 2 * IDX_TOPK
TOPK_CANDIDATES_PER_LEAF = 2048
TOPK_MAX_CANDIDATES = IDX_MAX_ROWS
TOPK_MAX_LEAVES = TOPK_MAX_CANDIDATES // TOPK_CANDIDATES_PER_LEAF
TOPK_MAX_NODES = 2 * TOPK_MAX_LEAVES - 1
TOPK_ARENA_ROWS = T_PAD * TOPK_MAX_NODES
TOPK_LEAF_WORKERS = 24
TOPK_LEVEL1_BASE = TOPK_MAX_LEAVES
TOPK_LEVEL2_BASE = TOPK_LEVEL1_BASE + 64
TOPK_LEVEL3_BASE = TOPK_LEVEL2_BASE + 32
TOPK_LEVEL4_BASE = TOPK_LEVEL3_BASE + 16
TOPK_LEVEL5_BASE = TOPK_LEVEL4_BASE + 8
TOPK_LEVEL6_BASE = TOPK_LEVEL5_BASE + 4
TOPK_ROOT_BASE = TOPK_LEVEL6_BASE + 2

@pl.jit.inline
def indexer(x: pl.Tensor[[T_DYN, D], pl.BF16], qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8], qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32], wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16], cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32], sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32], cmp_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32], cmp_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32], hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], inner_kv: pl.Tensor[[T_DYN, INNER_HEAD_DIM], pl.FP32], inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16], idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32], topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]], topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]], position_ids: pl.Tensor[[T_DYN], pl.INT32], idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], late_dep: pl.Scalar[pl.TASK_ID]):
    bs = pl.tensor.dim(x, 0)
    bs_heads = bs * IDX_N_HEADS
    row_blocks = (bs + MM_ROW_TILE - 1) // MM_ROW_TILE
    qr_acc_pad = pl.create_tensor([T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
    for qr_unit in pl.spmd(QR_OT_COUNT * row_blocks, name_hint='idx_qr_proj_matmul', allow_early_resolve=True):
        qr_rb = qr_unit // QR_OT_COUNT
        ot = qr_unit - qr_rb * QR_OT_COUNT
        qr_r0 = qr_rb * MM_ROW_TILE
        qr_rows = pl.min(MM_ROW_TILE, bs - qr_r0)
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(qr, [MM_ROW_TILE, Q_TILE], [qr_r0, q0], valid_shape=[qr_rows, Q_TILE])
                wq_tile = wq_b[q0:q0 + Q_TILE, o_base + ns:o_base + ns + MM_N_TILE]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            qr_acc_pad[qr_r0:qr_r0 + MM_ROW_TILE, o_base + ns:o_base + ns + MM_N_TILE] = qr_acc
    qr_proj = pl.create_tensor([bs, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint='idx_qr_proj_dequant', allow_early_resolve=True):
        o_base = ot * Q_OUT_TILE
        wq_scale = pl.reshape(wq_b_scale[o_base:o_base + Q_OUT_TILE], [1, Q_OUT_TILE])
        for dq_t0 in pl.range(0, bs, DEQUANT_T_TILE):
            acc_fp32 = pl.cast(qr_acc_pad[dq_t0:dq_t0 + DEQUANT_T_TILE, o_base:o_base + Q_OUT_TILE], target_type=pl.FP32, mode='none')
            qr_scale_tile = qr_scale[dq_t0:dq_t0 + DEQUANT_T_TILE, :]
            qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
            qr_proj[dq_t0:dq_t0 + DEQUANT_T_TILE, o_base:o_base + Q_OUT_TILE] = qr_dequant
    qr_proj_flat = pl.reshape(qr_proj, [bs_heads, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([bs_heads, IDX_HEAD_DIM], dtype=pl.BF16)
    rope_swap_idx_t = pl.create_tensor([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='qr_rope_swap_idx', allow_early_resolve=True):
        sw_col = pl.col_expand_mul(pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0), pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM] = pl.cast(pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)
    for idx in pl.spmd(bs_heads // ROPE_ROW_TILE, name_hint='qr_rope', allow_early_resolve=True):
        o0 = idx * ROPE_ROW_TILE
        token_idx = o0 // ROPE_ROW_BLOCK
        rope_swap_idx = rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM]
        cos_row = cos[token_idx:token_idx + 1, 0:ROPE_HEAD_DIM]
        sin_row = sin[token_idx:token_idx + 1, 0:ROPE_HEAD_DIM]
        qr_nope_slice = qr_proj_flat[o0:o0 + ROPE_ROW_TILE, 0:IDX_NOPE_HEAD_DIM]
        qr_rope_slice = qr_proj_flat[o0:o0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.col_expand_mul(qr_rope_slice, cos_row), pl.col_expand_mul(qr_swapped, sin_row))
        qr_vec = pl.concat(pl.cast(qr_nope_slice, target_type=pl.BF16, mode='rint'), pl.cast(rope_rot, target_type=pl.BF16, mode='rint'))
        qr_bf16[o0:o0 + ROPE_ROW_TILE, :] = qr_vec
    qh_acc_gm = pl.create_tensor([bs_heads, IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(bs_heads // QH_MM_TILE, name_hint='qr_hadamard_matmul', allow_early_resolve=True):
        o0 = idx * QH_MM_TILE
        qh_acc = pl.matmul(qr_bf16[o0:o0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32)
        qh_acc_gm[o0:o0 + QH_MM_TILE, :] = qh_acc
    qr_hadamard_i8 = pl.create_tensor([T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T_PAD * IDX_N_HEADS, 1], dtype=pl.FP32)
    with pl.spmd(bs_heads // QH_QUANT_TILE, name_hint='qr_hadamard_quant', allow_early_resolve=True) as qh_quant_tid:
        idx = pl.tile.get_block_idx()
        o0 = idx * QH_QUANT_TILE
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[o0:o0 + QH_QUANT_TILE, h0:h0 + QH_HEAD_DIM_TILE]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0:o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[o0:o0 + QH_QUANT_TILE, h1:h1 + QH_HEAD_DIM_TILE]
            qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode='rint')
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode='round')
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode='trunc')
            qr_hadamard_i8[o0:o0 + QH_QUANT_TILE, h1:h1 + QH_HEAD_DIM_TILE] = qh_i8
    x_flat = x
    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    weights_partial = pl.create_tensor([WEIGHTS_OK * T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    with pl.spmd(WEIGHTS_OK * row_blocks, name_hint='weights_proj', deps=[late_dep]) as _weights_tid:
        w_unit = pl.tile.get_block_idx()
        w_rb = w_unit // WEIGHTS_OK
        kb = w_unit - w_rb * WEIGHTS_OK
        w_r0 = w_rb * MM_ROW_TILE
        w_rows = pl.min(MM_ROW_TILE, bs - w_r0)
        k_base = kb * WEIGHTS_K_SLICE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [w_r0, d0], valid_shape=[w_rows, D_TILE])
            weights_proj_tile = weights_proj[d0:d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
        weights_partial[kb * T_PAD + w_r0:kb * T_PAD + w_r0 + MM_ROW_TILE, :] = weights_acc
    with pl.spmd(row_blocks, name_hint='weights_proj_reduce', allow_early_resolve=True) as weights_tid:
        w_rb = pl.tile.get_block_idx()
        w_r0 = w_rb * MM_ROW_TILE
        w_sum = weights_partial[w_r0:w_r0 + MM_ROW_TILE, :]
        for kb in pl.unroll(1, WEIGHTS_OK):
            partial_r0 = kb * T_PAD + w_r0
            w_sum = pl.add(w_sum, weights_partial[partial_r0:partial_r0 + MM_ROW_TILE, :])
        weights[w_r0:w_r0 + MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)
    cache_write_tid = indexer_compressor(x, inner_kv, inner_compress_state, inner_compress_state_block_table, inner_wkv, inner_wgate, inner_ape, inner_norm_w, cmp_cos, cmp_sin, hadamard, idx_kv_cache, idx_kv_scale, position_ids, idx_slot_mapping, inner_state_slot_mapping, late_dep)
    (topk_scores, topk_idxs) = indexer_score_topk_forest(qr_hadamard_i8, qr_hadamard_scale_dq, weights, idx_kv_cache, idx_kv_scale, idx_block_table, position_ids, kv_seq_lens, topk_scores, topk_idxs, qh_quant_tid, weights_tid, cache_write_tid)
    return (topk_scores, topk_idxs)

@pl.jit.inline(auto_scope=False)
def indexer_score_topk_forest(qr_hadamard_i8: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8], qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32], weights: pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32], idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32], idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32], position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]], topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]], qh_quant_tid: pl.Scalar[pl.TASK_ID], weights_tid: pl.Scalar[pl.TASK_ID], cache_write_tid: pl.Scalar[pl.TASK_ID]):
    """Run exact Top-K with the score and pair arenas on separate rings."""
    bs = pl.tensor.dim(position_ids, 0)
    b_dim = pl.tensor.dim(idx_block_table, 0)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    idx_table_len = b_dim * IDX_MAX_BLOCKS
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_block_num * BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * BLOCK_SIZE, 1])
    idx_block_table_flat = pl.reshape(idx_block_table, [idx_table_len])
    pair_arena = pl.create_tensor([TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], dtype=pl.FP32)
    with pl.scope():
        score_arena = pl.create_tensor([bs, TOPK_MAX_CANDIDATES], dtype=pl.FP32)
        with pl.spmd(TOPK_LEAF_WORKERS, name_hint='indexer_score_leaf_wave', deps=[qh_quant_tid, weights_tid, cache_write_tid], optimizations=[pl.split(pl.SplitMode.NONE, slot_num=2)]) as score_tid:
            worker = pl.tile.get_block_idx()
            query_count = pl.tensor.dim(position_ids, 0)
            global_leaf_base = 0
            for query in pl.range(query_count):
                batch_idx = query // S
                position = pl.read(position_ids, [query])
                cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
                visible_count = pl.max(pl.min(pl.min(cache_len, (position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES), 0)
                leaf_count = (visible_count + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF
                base_mod = global_leaf_base % TOPK_LEAF_WORKERS
                first_leaf = (worker + base_mod) % TOPK_LEAF_WORKERS
                for leaf in pl.range(first_leaf, leaf_count, TOPK_LEAF_WORKERS):
                    logical_begin = leaf * TOPK_CANDIDATES_PER_LEAF
                    valid_count = pl.min(TOPK_CANDIDATES_PER_LEAF, visible_count - logical_begin)
                    query_head_begin = query * IDX_N_HEADS
                    query_vector = qr_hadamard_i8[query_head_begin:query_head_begin + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                    query_scale = pl.reshape(qr_hadamard_scale_dq[query_head_begin:query_head_begin + IDX_N_HEADS, 0:1], [1, IDX_N_HEADS])
                    query_weight = weights[query:query + 1, 0:IDX_N_HEADS]
                    page_count = (valid_count + BLOCK_SIZE - 1) // BLOCK_SIZE
                    for page in pl.pipeline(0, page_count, stage=2):
                        page_begin = page * BLOCK_SIZE
                        logical_row = logical_begin + page_begin
                        logical_page = logical_row // BLOCK_SIZE
                        physical_block = pl.cast(pl.read(idx_block_table_flat, [batch_idx * IDX_MAX_BLOCKS + logical_page]), pl.INDEX)
                        physical_row = physical_block * BLOCK_SIZE
                        kv_i8 = kv_cache_i8_flat[physical_row:physical_row + BLOCK_SIZE, 0:IDX_HEAD_DIM]
                        score_i32 = pl.matmul(kv_i8, query_vector, out_dtype=pl.INT32, b_trans=True)
                        score_fp32 = pl.cast(score_i32, target_type=pl.FP32, mode='none')
                        score_fp32 = pl.col_expand_mul(score_fp32, query_scale)
                        score_fp32 = pl.maximum(score_fp32, 0.0)
                        score_fp32 = pl.col_expand_mul(score_fp32, query_weight)
                        kv_scale = kv_scale_flat[physical_row:physical_row + BLOCK_SIZE, 0:1]
                        score = pl.mul(pl.row_sum(score_fp32), kv_scale)
                        score_row = pl.reshape(score, [1, BLOCK_SIZE])
                        valid_rows = pl.min(BLOCK_SIZE, valid_count - page_begin)
                        score_valid = pl.fillpad(pl.set_validshape(score_row, 1, valid_rows), pad_value=pl.PadValue.min)
                        score_arena[query:query + 1, logical_row:logical_row + BLOCK_SIZE] = score_valid
                global_leaf_base = global_leaf_base + leaf_count
        with pl.spmd(TOPK_LEAF_WORKERS, name_hint='indexer_topk_leaf_wave', deps=[score_tid]) as topk_tid:
            indexer_topk_leaf_wave(position_ids, kv_seq_lens, score_arena, pair_arena)
        with pl.spmd(bs, name_hint='indexer_topk_query_merge', deps=[topk_tid]) as _score_tid:
            indexer_topk_query_merge(position_ids, kv_seq_lens, pair_arena, topk_scores, topk_idxs)
    return (topk_scores, topk_idxs)

@pl.jit.inline
def indexer_topk_leaf(score_arena: pl.Tensor[[T_DYN, TOPK_MAX_CANDIDATES], pl.FP32], pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32], query: pl.Scalar[pl.INDEX], logical_begin: pl.Scalar[pl.INDEX], valid_count: pl.Scalar[pl.INDEX], output_slot: pl.Scalar[pl.INDEX]) -> None:
    """Sort one scored 2K leaf and store its exact Top-512 pair row."""
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    leaf_indices = pl.add(pl.tile.arange(0, [1, TOPK_CANDIDATES_PER_LEAF], dtype=pl.INT32), logical_begin_i32)
    leaf_scores_raw = pl.load(score_arena, [query, logical_begin], [1, TOPK_CANDIDATES_PER_LEAF], valid_shape=[1, valid_count])
    leaf_scores = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
    leaf_scores = pl.maximum(leaf_scores, pl.tile.full([1, TOPK_CANDIDATES_PER_LEAF], dtype=pl.FP32, value=FP32_NEG_INF))
    pairs = pl.sort32(leaf_scores, pl.reinterpret_view(leaf_indices, pl.UINT32))
    pairs = pl.mrgsort(pairs, block_len=64)
    pairs = pl.mrgsort(pairs, block_len=256)
    pairs = pl.mrgsort(pairs, block_len=1024)
    pl.store(pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)

@pl.jit.incore
def indexer_topk_leaf_wave(position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], score_arena: pl.Tensor[[T_DYN, TOPK_MAX_CANDIDATES], pl.FP32], pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32]):
    """Sort a globally striped sequence of scored leaves."""
    worker = pl.tile.get_block_idx()
    query_count = pl.tensor.dim(position_ids, 0)
    global_leaf_base = 0
    for query in pl.range(query_count):
        batch_idx = query // S
        position = pl.read(position_ids, [query])
        cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
        visible_count = pl.max(pl.min(pl.min(cache_len, (position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES), 0)
        leaf_count = (visible_count + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF
        arena_base = query * TOPK_MAX_NODES
        base_mod = global_leaf_base % TOPK_LEAF_WORKERS
        first_leaf = (worker + base_mod) % TOPK_LEAF_WORKERS
        for leaf in pl.range(first_leaf, leaf_count, TOPK_LEAF_WORKERS):
            logical_begin = leaf * TOPK_CANDIDATES_PER_LEAF
            valid_count = pl.min(TOPK_CANDIDATES_PER_LEAF, visible_count - logical_begin)
            indexer_topk_leaf(score_arena, pair_arena, query, logical_begin, valid_count, arena_base + leaf)
        global_leaf_base = global_leaf_base + leaf_count

@pl.jit.incore
def indexer_topk_query_merge(position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32], topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32], topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]):
    """Merge each query's completed leaf rows and materialize Top-512 outputs."""
    query = pl.tile.get_block_idx()
    batch_idx = query // S
    position = pl.read(position_ids, [query])
    cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
    visible_count = pl.min(pl.min(cache_len, (position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES)
    pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [query, 0], topk_scores)
    pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [query, 0], topk_indices)
    if visible_count > 0:
        leaf_count = (visible_count + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF
        arena_base = query * TOPK_MAX_NODES
        level1_count = (leaf_count + 1) // 2
        merge_topk_level_pairs(pair_arena, arena_base, leaf_count, 0, TOPK_LEVEL1_BASE)
        level2_count = (level1_count + 1) // 2
        merge_topk_level_pairs(pair_arena, arena_base, level1_count, TOPK_LEVEL1_BASE, TOPK_LEVEL2_BASE)
        level3_count = (level2_count + 1) // 2
        merge_topk_level_pairs(pair_arena, arena_base, level2_count, TOPK_LEVEL2_BASE, TOPK_LEVEL3_BASE)
        level4_count = (level3_count + 1) // 2
        merge_topk_level_pairs(pair_arena, arena_base, level3_count, TOPK_LEVEL3_BASE, TOPK_LEVEL4_BASE)
        level5_count = (level4_count + 1) // 2
        merge_topk_level_pairs(pair_arena, arena_base, level4_count, TOPK_LEVEL4_BASE, TOPK_LEVEL5_BASE)
        level6_count = (level5_count + 1) // 2
        merge_topk_level_pairs(pair_arena, arena_base, level5_count, TOPK_LEVEL5_BASE, TOPK_LEVEL6_BASE)
        merge_topk_level_pairs(pair_arena, arena_base, level6_count, TOPK_LEVEL6_BASE, TOPK_ROOT_BASE)
        root_slot = arena_base + TOPK_ROOT_BASE
        root_pairs = pl.load(pair_arena, [root_slot, 0], [1, TOPK_PAIR_WIDTH])
        pl.store(pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32), [query, 0], topk_scores)
        root_indices = pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        output_indices = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
        valid_topk = pl.min(visible_count, IDX_TOPK)
        for lane in pl.range(valid_topk):
            pl.tile.write(output_indices, [0, lane], pl.tile.read(root_indices, [0, lane]))
        pl.store(output_indices, [query, 0], topk_indices)

@pl.jit.inline
def merge2_top512_pairs(pair_arena: pl.Tensor, left_slot: pl.Scalar[pl.INDEX], right_slot: pl.Scalar[pl.INDEX], output_slot: pl.Scalar[pl.INDEX]) -> None:
    """Merge two arena rows and store their exact Top-512 pair row."""
    left = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
    right = pl.load(pair_arena, [right_slot, 0], [1, TOPK_PAIR_WIDTH])
    merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left, right, tmp=merge_tmp)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output_slot, 0], pair_arena)

@pl.jit.inline
def merge_topk_level_pairs(pair_arena: pl.Tensor, arena_base: pl.Scalar[pl.INDEX], input_count: pl.Scalar[pl.INDEX], input_base: pl.Scalar[pl.INDEX], output_base: pl.Scalar[pl.INDEX]) -> None:
    """Reduce one exact-Top-K forest level, forwarding an odd final node."""
    output_count = (input_count + 1) // 2
    for output in pl.range(output_count):
        left_slot = arena_base + input_base + 2 * output
        right_slot = left_slot + 1
        output_slot = arena_base + output_base + output
        if right_slot < arena_base + input_base + input_count:
            merge2_top512_pairs(pair_arena, left_slot, right_slot, output_slot)
        else:
            forwarded = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
            pl.store(forwarded, [output_slot, 0], pair_arena)
