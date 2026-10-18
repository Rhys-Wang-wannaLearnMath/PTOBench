from config import D, H, HC_DIM, HC_MULT, IDX_TOPK, LEAF_CMP_BLOCKS, LEAF_CMP_ROWS, LOCAL_PARTS, LOCAL_ROWS, MAX_COMPRESS_LEAVES, MAX_SEGMENT_TILES, MIX_HC, NUM_LOCAL_TILES, ORI_CACHE_ROWS, ORI_MAX_BLOCKS, OVERLAY_ROWS, OVERLAY_SOURCES, O_GROUPS, O_GROUP_IN, O_LORA, Q_LORA, ROPE_HEAD_DIM, STATE_ROWS, WIN
import pypto.language as pl

import pypto.language.distributed as pld

from config import BLOCK_SIZE, CASE_CONFIG as M, FP32_NEG_INF, HCA_STATE_PHYSICAL_BLOCKS, PREFILL_CMP_BLOCK_NUM, PREFILL_CMP_MAX_BLOCKS, PREFILL_ORI_MAX_BLOCKS

from prefill_compressor_ratio128 import COMPRESS_RATIO, COMPRESS_STATE_DIM, HCA_CMP_BLOCK_NUM, HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_STATE_MAX_BLOCKS, MAX_SEQ_LEN, prefill_compressor_ratio128

from prefill_cp_exchange import CMP_META_DIM, CMP_ROWS_PER_RANK, CMP_ROWS_PER_SEGMENT, CMP_WINDOW_ROWS, STATE_META_DIM, STATE_WINDOW_ROWS, _prefill_cp_dual_tail_exchange_wave, _prefill_cp_hca_compact_exchange_commit_wave, _prefill_cp_sparse_stage

from prefill_cp_zigzag import CP_SIZE, CP_TAIL_WINDOW_ROWS, EPOCHS, NUM_SEGMENTS, TAIL_ROWS

from hc_post import hc_post_prefill

from hc_pre import hc_pre

from prefill_sparse_attn import HEAD_DIM, PREFILL_SPARSE_PAD, ROPE_DIM, VALID_BLOCK_MASK_COLS, _prefill_sparse_attn_math

from qkv_proj_rope import materialize_rope_rows, qkv_proj_rope

from rmsnorm import rms_norm

LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD

@pl.jit.inline
def prefill_cp_hca_core(x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.InOut[pl.Tensor[[HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32], owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32], predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32], query_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], query_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], overlay_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32], cmp_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32], segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32], snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32], snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32], final_segment_t: pl.Tensor[[1], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32], hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16], kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16], tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], cmp_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16], cmp_meta_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32], state_window: pld.DistributedTensor[[STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32], state_meta_window: pld.DistributedTensor[[CP_SIZE, STATE_META_DIM], pl.INT32], compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]], my_rank: pl.Scalar[pl.INT32]):
    """CP-HCA attention math (inline). Shared by the standalone rank child
    and the layer composition child. Inlining avoids child-in-child nesting
    (@pl.jit cannot call another @pl.jit)."""
    q = pl.create_tensor([LOCAL_ROWS, H, HEAD_DIM], dtype=pl.BF16)
    q_initial_flat = pl.reshape(q, [LOCAL_ROWS * H * HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_q'):
        for q_initial_block in pl.range(LOCAL_ROWS * H * HEAD_DIM // 256):
            q_initial_flat[q_initial_block:q_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    post = pl.create_tensor([LOCAL_ROWS, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([LOCAL_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    rope_cos_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_cos_flat_initial_flat = pl.reshape(rope_cos_flat, [LOCAL_ROWS * ROPE_HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_rope_cos_flat'):
        for rope_cos_flat_initial_block in pl.range(LOCAL_ROWS * ROPE_HEAD_DIM // 256):
            rope_cos_flat_initial_flat[rope_cos_flat_initial_block:rope_cos_flat_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    rope_sin_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_flat_initial_flat = pl.reshape(rope_sin_flat, [LOCAL_ROWS * ROPE_HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_rope_sin_flat'):
        for rope_sin_flat_initial_block in pl.range(LOCAL_ROWS * ROPE_HEAD_DIM // 256):
            rope_sin_flat_initial_flat[rope_sin_flat_initial_block:rope_sin_flat_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    local_kv = pl.create_tensor([LOCAL_ROWS, HEAD_DIM], dtype=pl.BF16)
    local_kv_initial_flat = pl.reshape(local_kv, [LOCAL_ROWS * HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_local_kv'):
        for local_kv_initial_block in pl.range(LOCAL_ROWS * HEAD_DIM // 256):
            local_kv_initial_flat[local_kv_initial_block:local_kv_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    normed = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
    normed_initial_flat = pl.reshape(normed, [LOCAL_ROWS * D // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_normed'):
        for normed_initial_block in pl.range(LOCAL_ROWS * D // 256):
            normed_initial_flat[normed_initial_block:normed_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    mixed = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
    qr = pl.create_tensor([LOCAL_ROWS, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([LOCAL_ROWS, 1], dtype=pl.FP32)
    x_flat = pl.reshape(x_hc, [LOCAL_ROWS, HC_MULT, D])
    query_positions_flat = pl.reshape(query_positions, [LOCAL_ROWS])
    query_requests_flat = pl.reshape(query_requests, [LOCAL_ROWS])
    overlay_positions_flat = pl.reshape(overlay_positions, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    overlay_requests_flat = pl.reshape(overlay_requests, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    overlay_active_flat = pl.reshape(overlay_active_lengths, [NUM_LOCAL_TILES, OVERLAY_SOURCES])
    swa_indices_flat = pl.reshape(swa_indices, [LOCAL_ROWS, WIN])
    cmp_indices_flat = pl.reshape(cmp_indices, [LOCAL_ROWS, IDX_TOPK])
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * TAIL_ROWS
        x_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [row0, 0, 0])
        mixed_tile = pl.slice(mixed, [TAIL_ROWS, D], [row0, 0])
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [row0, 0])
        position_tile = pl.slice(query_positions_flat, [TAIL_ROWS], [row0])
        cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [row0, 0])
        normed_tile = pl.slice(normed, [TAIL_ROWS, D], [row0, 0])
        q_tile = pl.slice(q, [TAIL_ROWS, H, HEAD_DIM], [row0, 0, 0])
        kv_tile = pl.slice(local_kv, [TAIL_ROWS, HEAD_DIM], [row0, 0])
        qr_tile = pl.slice(qr, [TAIL_ROWS, Q_LORA], [row0, 0])
        qr_scale_tile = pl.slice(qr_scale, [TAIL_ROWS, 1], [row0, 0])
        hc_pre(x_tile, hc_attn_fn, hc_attn_scale, hc_attn_base, mixed_tile, post_tile, comb_tile)
        rms_tid = rms_norm(mixed_tile, attn_norm_w, normed_tile)
        late_dep = pl.system.task_dummy(deps=[rms_tid])
        active = pl.read(overlay_active_lengths, [tile // MAX_SEGMENT_TILES, tile % MAX_SEGMENT_TILES, 1])
        materialize_rope_rows(freqs_cos, freqs_sin, position_tile, active, cos_tile, sin_tile)
        qkv_proj_rope(normed_tile, wq_a, wq_b, wq_b_scale, wkv, cos_tile, sin_tile, gamma_cq, gamma_ckv, q_tile, kv_tile, qr_tile, qr_scale_tile, late_dep)
    local_hidden_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * TAIL_ROWS, D], dtype=pl.BF16)
    local_hidden_tail_initial_flat = pl.reshape(local_hidden_tail, [EPOCHS * LOCAL_PARTS * TAIL_ROWS * D // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_local_hidden_tail'):
        for local_hidden_tail_initial_block in pl.range(EPOCHS * LOCAL_PARTS * TAIL_ROWS * D // 256):
            local_hidden_tail_initial_flat[local_hidden_tail_initial_block:local_hidden_tail_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    local_kv_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    local_kv_tail_initial_flat = pl.reshape(local_kv_tail, [EPOCHS * LOCAL_PARTS * TAIL_ROWS * HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_local_kv_tail'):
        for local_kv_tail_initial_block in pl.range(EPOCHS * LOCAL_PARTS * TAIL_ROWS * HEAD_DIM // 256):
            local_kv_tail_initial_flat[local_kv_tail_initial_block:local_kv_tail_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_tail_assemble'):
        for part in pl.range(LOCAL_PARTS):
            first_len = pl.read(overlay_active_lengths, [part, 0, 1])
            second_len = pl.read(overlay_active_lengths, [part, 1, 1])
            total = first_len + second_len
            tail_offset0 = pl.max(total - TAIL_ROWS, 0)
            for row in pl.range(TAIL_ROWS):
                tail_offset = tail_offset0 + row
                if tail_offset < total:
                    if tail_offset < TAIL_ROWS:
                        source = part * MAX_SEGMENT_TILES * TAIL_ROWS + tail_offset
                    else:
                        source = part * MAX_SEGMENT_TILES * TAIL_ROWS + tail_offset
                    destination = part * TAIL_ROWS + row
                    local_hidden_tail[destination:destination + 1, :] = normed[source:source + 1, :]
                    local_kv_tail[destination:destination + 1, :] = local_kv[source:source + 1, :]
    logical_hidden = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    logical_kv = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_tail_exchange'):
        _prefill_cp_dual_tail_exchange_wave(local_hidden_tail, local_kv_tail, reverse_index, owner_rank_table, hidden_tail_window, kv_tail_window, tail_ready, tail_consumed, logical_hidden, logical_kv, my_rank, pl.cast(0, pl.INT32))
    effective_x = pl.create_tensor([LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS, D], dtype=pl.BF16)
    effective_x_initial_flat = pl.reshape(effective_x, [LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS * D // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_effective_x'):
        for effective_x_initial_block in pl.range(LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS * D // 256):
            effective_x_initial_flat[effective_x_initial_block:effective_x_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    leaf_positions = pl.create_tensor([LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS], dtype=pl.INT32)
    leaf_positions_initial_flat = pl.reshape(leaf_positions, [LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_leaf_positions'):
        for leaf_positions_initial_block in pl.range(LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS // 256):
            leaf_positions_initial_flat[leaf_positions_initial_block:leaf_positions_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.INT32, value=0)
    leaf_num_tokens = pl.create_tensor([LOCAL_PARTS, MAX_COMPRESS_LEAVES], dtype=pl.INT32)
    leaf_num_tokens_initial_flat = pl.reshape(leaf_num_tokens, [LOCAL_PARTS * MAX_COMPRESS_LEAVES])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_leaf_num_tokens'):
        for leaf_num_tokens_initial_element in pl.range(LOCAL_PARTS * MAX_COMPRESS_LEAVES):
            pl.write(leaf_num_tokens_initial_flat, [leaf_num_tokens_initial_element], pl.cast(0, pl.INT32))
    leaf_cmp_slots = pl.create_tensor([LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS], dtype=pl.INT64)
    leaf_cmp_slots_initial_flat = pl.reshape(leaf_cmp_slots, [LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_leaf_cmp_slots'):
        for leaf_cmp_slots_initial_element in pl.range(LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS):
            pl.write(leaf_cmp_slots_initial_flat, [leaf_cmp_slots_initial_element], pl.cast(-1, pl.INT64))
    leaf_state_slots = pl.create_tensor([LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS], dtype=pl.INT64)
    leaf_state_slots_initial_flat = pl.reshape(leaf_state_slots, [LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_leaf_state_slots'):
        for leaf_state_slots_initial_element in pl.range(LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS):
            pl.write(leaf_state_slots_initial_flat, [leaf_state_slots_initial_element], pl.cast(-1, pl.INT64))
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_leaf_lowering'):
        for part in pl.range(LOCAL_PARTS):
            predecessor = pl.read(predecessor_segments, [part])
            predecessor_valid = pl.read(overlay_active_lengths, [part, 0, 0])
            pl.write(leaf_num_tokens, [part, 0], predecessor_valid)
            for leaf_row in pl.range(TAIL_ROWS):
                leaf_index = part * MAX_COMPRESS_LEAVES * TAIL_ROWS + leaf_row
                if predecessor >= 0 and leaf_row < predecessor_valid:
                    source = predecessor * TAIL_ROWS + leaf_row
                    effective_x[leaf_index:leaf_index + 1, :] = logical_hidden[source:source + 1, :]
                    position = pl.read(segment_tail_positions, [predecessor, leaf_row])
                    pl.write(leaf_positions, [leaf_index], position)
                    if position >= 0:
                        logical_block = position // HCA_STATE_BLOCK_SIZE
                        physical_block = pl.read(compress_state_block_table, [logical_block])
                        if physical_block >= 0:
                            predecessor_state_row = pl.cast(physical_block, pl.INT64) * HCA_STATE_BLOCK_SIZE + position % HCA_STATE_BLOCK_SIZE
                            pl.write(leaf_state_slots, [leaf_index], predecessor_state_row)
            for tile in pl.range(MAX_SEGMENT_TILES):
                leaf = 1 + tile
                active = pl.read(overlay_active_lengths, [part, tile, 1])
                pl.write(leaf_num_tokens, [part, leaf], active)
                local_row0 = (part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS
                leaf_row0 = (part * MAX_COMPRESS_LEAVES + leaf) * TAIL_ROWS
                for leaf_row in pl.range(TAIL_ROWS):
                    destination = leaf_row0 + leaf_row
                    if leaf_row < active:
                        source = local_row0 + leaf_row
                        effective_x[destination:destination + 1, :] = normed[source:source + 1, :]
                        position = pl.read(query_positions_flat, [source])
                        pl.write(leaf_positions, [destination], position)
                        logical_block = position // HCA_STATE_BLOCK_SIZE
                        physical_block = pl.read(compress_state_block_table, [logical_block])
                        if physical_block >= 0:
                            local_state_row = pl.cast(physical_block, pl.INT64) * HCA_STATE_BLOCK_SIZE + position % HCA_STATE_BLOCK_SIZE
                            pl.write(leaf_state_slots, [destination], local_state_row)
                        if (position + 1) % COMPRESS_RATIO == 0:
                            pl.write(leaf_cmp_slots, [destination], pl.cast(0, pl.INT64))
    scratch_state = pl.create_tensor([LOCAL_PARTS * HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], dtype=pl.FP32)
    scratch_state_initial_flat = pl.reshape(scratch_state, [LOCAL_PARTS * HCA_STATE_PHYSICAL_BLOCKS * HCA_STATE_BLOCK_SIZE * COMPRESS_STATE_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_scratch_state'):
        for scratch_state_initial_block in pl.range(LOCAL_PARTS * HCA_STATE_PHYSICAL_BLOCKS * HCA_STATE_BLOCK_SIZE * COMPRESS_STATE_DIM // 256):
            scratch_state_initial_flat[scratch_state_initial_block:scratch_state_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
    persistent_state_flat = pl.reshape(compress_state, [STATE_ROWS, COMPRESS_STATE_DIM])
    scratch_state_flat = pl.reshape(scratch_state, [LOCAL_PARTS * STATE_ROWS, COMPRESS_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_seed_state'):
        for part in pl.range(LOCAL_PARTS):
            segment = pl.read(owner_segments_t, [part])
            if segment == 0:
                for state_row in pl.range(STATE_ROWS):
                    destination = part * STATE_ROWS + state_row
                    persistent_state_row = persistent_state_flat[state_row:state_row + 1, :]
                    scratch_state_flat[destination:destination + 1, :] = persistent_state_row
    leaf_cmp = pl.create_tensor([LOCAL_PARTS * MAX_COMPRESS_LEAVES * LEAF_CMP_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    leaf_cmp_initial_flat = pl.reshape(leaf_cmp, [LOCAL_PARTS * MAX_COMPRESS_LEAVES * LEAF_CMP_BLOCKS * BLOCK_SIZE * 1 * HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_leaf_cmp'):
        for leaf_cmp_initial_block in pl.range(LOCAL_PARTS * MAX_COMPRESS_LEAVES * LEAF_CMP_BLOCKS * BLOCK_SIZE * 1 * HEAD_DIM // 256):
            leaf_cmp_initial_flat[leaf_cmp_initial_block:leaf_cmp_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    for part in pl.range(LOCAL_PARTS):
        state_base = part * HCA_STATE_PHYSICAL_BLOCKS
        state_part = pl.slice(scratch_state, [HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], [state_base, 0, 0])
        for leaf in pl.range(MAX_COMPRESS_LEAVES):
            leaf_index = part * MAX_COMPRESS_LEAVES + leaf
            token0 = leaf_index * TAIL_ROWS
            cmp_block0 = leaf_index * LEAF_CMP_BLOCKS
            x_leaf = pl.slice(effective_x, [TAIL_ROWS, D], [token0, 0])
            position_leaf = pl.slice(leaf_positions, [TAIL_ROWS], [token0])
            cmp_slots_leaf = pl.slice(leaf_cmp_slots, [TAIL_ROWS], [token0])
            state_slots_leaf = pl.slice(leaf_state_slots, [TAIL_ROWS], [token0])
            cmp_leaf = pl.slice(leaf_cmp, [LEAF_CMP_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], [cmp_block0, 0, 0, 0])
            active = pl.read(leaf_num_tokens, [part, leaf])
            (cmp_leaf, state_part) = prefill_compressor_ratio128(x_leaf, state_part, compress_state_block_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, freqs_cos, freqs_sin, cmp_leaf, position_leaf, active, cmp_slots_leaf, state_slots_leaf)
            leaf_cmp = pl.assemble(leaf_cmp, cmp_leaf, [cmp_block0, 0, 0, 0])
        scratch_state = pl.assemble(scratch_state, state_part, [state_base, 0, 0])
    local_cmp_payload = pl.create_tensor([EPOCHS * CMP_ROWS_PER_RANK, HEAD_DIM], dtype=pl.BF16)
    local_cmp_payload_initial_flat = pl.reshape(local_cmp_payload, [EPOCHS * CMP_ROWS_PER_RANK * HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_local_cmp_payload') as local_cmp_payload_init_tid:
        for local_cmp_payload_initial_block in pl.range(EPOCHS * CMP_ROWS_PER_RANK * HEAD_DIM // 256):
            local_cmp_payload_initial_flat[local_cmp_payload_initial_block:local_cmp_payload_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    local_cmp_meta = pl.create_tensor([EPOCHS * CMP_ROWS_PER_RANK, CMP_META_DIM], dtype=pl.INT32)
    local_cmp_meta_initial_flat = pl.reshape(local_cmp_meta, [EPOCHS * CMP_ROWS_PER_RANK * CMP_META_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_local_cmp_meta') as local_cmp_meta_init_tid:
        for local_cmp_meta_initial_element in pl.range(EPOCHS * CMP_ROWS_PER_RANK * CMP_META_DIM):
            pl.write(local_cmp_meta_initial_flat, [local_cmp_meta_initial_element], pl.cast(-1, pl.INT32))
    local_state_payload = pl.create_tensor([EPOCHS * TAIL_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32)
    local_state_payload_initial_flat = pl.reshape(local_state_payload, [EPOCHS * TAIL_ROWS * COMPRESS_STATE_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_local_state_payload'):
        for local_state_payload_initial_block in pl.range(EPOCHS * TAIL_ROWS * COMPRESS_STATE_DIM // 256):
            local_state_payload_initial_flat[local_state_payload_initial_block:local_state_payload_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
    local_state_meta = pl.create_tensor([EPOCHS, STATE_META_DIM], dtype=pl.INT32)
    local_state_meta_initial_flat = pl.reshape(local_state_meta, [EPOCHS * STATE_META_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_local_state_meta'):
        for local_state_meta_initial_element in pl.range(EPOCHS * STATE_META_DIM):
            pl.write(local_state_meta_initial_flat, [local_state_meta_initial_element], pl.cast(-1, pl.INT32))
    leaf_cmp_flat = pl.reshape(leaf_cmp, [LEAF_CMP_ROWS, HEAD_DIM])
    scratch_state_flat = pl.reshape(scratch_state, [LOCAL_PARTS * STATE_ROWS, COMPRESS_STATE_DIM])
    final_segment = pl.read(final_segment_t, [0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_pack_compact', deps=[local_cmp_payload_init_tid, local_cmp_meta_init_tid]):
        for part in pl.range(LOCAL_PARTS):
            segment = pl.read(owner_segments_t, [part])
            for tile in pl.range(MAX_SEGMENT_TILES):
                active = pl.read(overlay_active_lengths, [part, tile, 1])
                query_row0 = (part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS
                for row in pl.range(TAIL_ROWS):
                    if row < active:
                        position = pl.read(query_positions_flat, [query_row0 + row])
                        if (position + 1) % COMPRESS_RATIO == 0:
                            destination = part * CMP_ROWS_PER_SEGMENT + tile
                            leaf_index = part * MAX_COMPRESS_LEAVES + 1 + tile
                            source = leaf_index * LEAF_CMP_BLOCKS * BLOCK_SIZE
                            local_cmp_payload[destination:destination + 1, :] = leaf_cmp_flat[source:source + 1, :]
                            pl.write(local_cmp_meta, [destination, 0], pl.cast(1, pl.INT32))
                            pl.write(local_cmp_meta, [destination, 1], segment)
                            pl.write(local_cmp_meta, [destination, 2], position)
                            pl.write(local_cmp_meta, [destination, 3], pl.cast((position + 1) // COMPRESS_RATIO - 1, pl.INT32))
            if segment == final_segment:
                valid = pl.read(snapshot_valid, [part])
                end_position = pl.read(segment_starts_t, [segment]) + pl.read(segment_active_lengths, [part])
                pl.write(local_state_meta, [0, 0], pl.cast(1, pl.INT32))
                pl.write(local_state_meta, [0, 1], segment)
                pl.write(local_state_meta, [0, 2], valid)
                pl.write(local_state_meta, [0, 3], end_position)
                for row in pl.range(TAIL_ROWS):
                    if row < valid:
                        position = pl.read(snapshot_positions, [part, row])
                        logical_block = position // HCA_STATE_BLOCK_SIZE
                        physical_block = pl.read(compress_state_block_table, [logical_block])
                        if physical_block >= 0:
                            source = part * STATE_ROWS + pl.cast(physical_block, pl.INDEX) * HCA_STATE_BLOCK_SIZE + position % HCA_STATE_BLOCK_SIZE
                            local_state_payload[row:row + 1, :] = scratch_state_flat[source:source + 1, :]
    cmp_kv_flat = pl.reshape(cmp_kv, [PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    compress_state_flat = pl.reshape(compress_state, [STATE_ROWS, COMPRESS_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_compact_commit'):
        cmp_kv_flat = _prefill_cp_hca_compact_exchange_commit_wave(local_cmp_payload, local_cmp_meta, local_state_payload, local_state_meta, owner_rank_table, owner_part_table, cmp_block_table, compress_state_block_table, cmp_window, cmp_meta_window, state_window, state_meta_window, compact_ready, compact_consumed, cmp_kv_flat, compress_state_flat, my_rank, pl.cast(0, pl.INT32))
    cache_flat = pl.reshape(kv_cache, [ORI_CACHE_ROWS, HEAD_DIM])
    sparse_kv = pl.create_tensor([LOCAL_SPARSE_ROWS, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([LOCAL_ROWS, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    valid_mask = pl.create_tensor([LOCAL_ROWS, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    _prefill_cp_sparse_stage(cache_flat, local_kv, logical_kv, cmp_kv_flat, cmp_block_table, query_positions_flat, query_requests_flat, overlay_positions_flat, overlay_requests_flat, predecessor_segments, segment_starts_t, swa_indices_flat, cmp_indices_flat, sparse_kv, sparse_bias, valid_mask, overlay_active_flat)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_raw_commit'):
        for row in pl.range(TAIL_ROWS):
            raw_segment = pl.read(final_win_seg_src, [row])
            raw_source_row = pl.read(final_win_row_src, [row])
            raw_destination = pl.read(final_slot_mapping, [row])
            if raw_segment >= 0 and raw_source_row >= 0 and (raw_destination >= 0):
                raw_source = raw_segment * TAIL_ROWS + raw_source_row
                cache_flat[raw_destination:raw_destination + 1, :] = logical_kv[raw_source:raw_source + 1, :]
    x_out_flat = pl.reshape(x_out, [LOCAL_ROWS, HC_MULT, D])
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * TAIL_ROWS
        sparse_row0 = row0 * PREFILL_SPARSE_PAD
        q_tile = pl.slice(q, [TAIL_ROWS, H, HEAD_DIM], [row0, 0, 0])
        sparse_tile = pl.slice(sparse_kv, [TAIL_ROWS * PREFILL_SPARSE_PAD, HEAD_DIM], [sparse_row0, 0])
        bias_tile = pl.slice(sparse_bias, [TAIL_ROWS, PREFILL_SPARSE_PAD], [row0, 0])
        mask_tile = pl.slice(valid_mask, [TAIL_ROWS, VALID_BLOCK_MASK_COLS], [row0, 0])
        cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_DIM], [row0, 0])
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [row0, 0])
        residual_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [row0, 0, 0])
        active = pl.read(overlay_active_lengths, [tile // MAX_SEGMENT_TILES, tile % MAX_SEGMENT_TILES, 1])
        attn_out_tile = pl.create_tensor([TAIL_ROWS, D], dtype=pl.BF16)
        y_tile = pl.create_tensor([TAIL_ROWS, HC_MULT, D], dtype=pl.FP32)
        y_tile_initial_flat = pl.reshape(y_tile, [TAIL_ROWS * HC_MULT * D // 256, 256])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_initialize_y_tile'):
            for y_tile_initial_block in pl.range(TAIL_ROWS * HC_MULT * D // 256):
                y_tile_initial_flat[y_tile_initial_block:y_tile_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        _prefill_sparse_attn_math(q_tile, sparse_tile, bias_tile, mask_tile, attn_sink, cos_tile, sin_tile, wo_a, wo_b, wo_b_scale, attn_out_tile, active)
        hc_post_prefill(attn_out_tile, residual_tile, post_tile, comb_tile, y_tile, active)
        x_out_flat[row0:row0 + TAIL_ROWS, 0:HC_MULT, 0:D] = y_tile
    # Persistent benchmark windows retain one ready/consumed credit per
    # remote rank; this entry restarts its single exchange epoch at zero.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_hca_signal_retire'):
        _x_out_anchor = pl.read(x_out_flat, [0, 0, 0])
        for signal_peer in pl.range(CP_SIZE):
            if signal_peer != my_rank:
                pld.system.wait(signal=tail_consumed, offsets=[signal_peer, 0], expected=EPOCHS, cmp=pld.WaitCmp.Ge)
                pld.system.wait(signal=compact_consumed, offsets=[signal_peer, 0], expected=EPOCHS, cmp=pld.WaitCmp.Ge)
                pld.system.notify(target=tail_ready, peer=my_rank, offsets=[signal_peer, 0], value=pl.cast(0 - EPOCHS, pl.INT32), op=pld.NotifyOp.AtomicAdd)
                pld.system.notify(target=tail_consumed, peer=my_rank, offsets=[signal_peer, 0], value=pl.cast(0 - EPOCHS, pl.INT32), op=pld.NotifyOp.AtomicAdd)
                pld.system.notify(target=compact_ready, peer=my_rank, offsets=[signal_peer, 0], value=pl.cast(0 - EPOCHS, pl.INT32), op=pld.NotifyOp.AtomicAdd)
                pld.system.notify(target=compact_consumed, peer=my_rank, offsets=[signal_peer, 0], value=pl.cast(0 - EPOCHS, pl.INT32), op=pld.NotifyOp.AtomicAdd)
    return pl.reshape(x_out_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D])

@pl.jit
def prefill_cp_hca_rank(x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.InOut[pl.Tensor[[HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32], owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32], predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32], query_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], query_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], overlay_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32], cmp_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32], segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32], snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32], snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32], final_segment_t: pl.Tensor[[1], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32], hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16], kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16], tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], cmp_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16], cmp_meta_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32], state_window: pld.DistributedTensor[[STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32], state_meta_window: pld.DistributedTensor[[CP_SIZE, STATE_META_DIM], pl.INT32], compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]], my_rank: pl.Scalar[pl.INT32]):
    """Standalone CP-HCA rank child. Delegates to the inline core so the
    standalone test preserves the original @pl.jit entry point."""
    return prefill_cp_hca_core(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, segment_starts_t, segment_active_lengths, owner_segments_t, predecessor_segments, query_positions, query_requests, overlay_positions, overlay_requests, overlay_active_lengths, swa_indices, cmp_indices, segment_tail_positions, snapshot_positions, snapshot_valid, final_segment_t, reverse_index, owner_rank_table, owner_part_table, final_win_seg_src, final_win_row_src, final_slot_mapping, hidden_tail_window, kv_tail_window, tail_ready, tail_consumed, cmp_window, cmp_meta_window, state_window, state_meta_window, compact_ready, compact_consumed, attn_sink, wo_a, wo_b, wo_b_scale, x_out, my_rank)

@pl.jit.host
def prefill_cp_hca_test(x_hc: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.InOut[pl.Tensor[[CP_SIZE, HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]], compress_state_block_table: pl.Tensor[[CP_SIZE, HCA_STATE_MAX_BLOCKS], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[CP_SIZE, ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[CP_SIZE, PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_block_table: pl.Tensor[[CP_SIZE, PREFILL_CMP_MAX_BLOCKS], pl.INT32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], segment_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32], owner_segments_t: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32], predecessor_segments: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32], query_positions: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], query_requests: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], overlay_positions: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32], cmp_indices: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32], segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32], snapshot_positions: pl.Tensor[[CP_SIZE, LOCAL_PARTS, TAIL_ROWS], pl.INT32], snapshot_valid: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32], final_segment_t: pl.Tensor[[1], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]]):
    hidden_tail_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    kv_tail_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    tail_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    tail_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    cmp_window_buf = pld.alloc_window_buffer([CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    cmp_meta_window_buf = pld.alloc_window_buffer([CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32)
    state_window_buf = pld.alloc_window_buffer([STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32)
    state_meta_window_buf = pld.alloc_window_buffer([CP_SIZE, STATE_META_DIM], dtype=pl.INT32)
    compact_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    compact_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_tail_window = pld.window(hidden_tail_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
        kv_tail_window = pld.window(kv_tail_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
        tail_ready = pld.window(tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        tail_consumed = pld.window(tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        cmp_window = pld.window(cmp_window_buf, [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
        cmp_meta_window = pld.window(cmp_meta_window_buf, [CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32)
        state_window = pld.window(state_window_buf, [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32)
        state_meta_window = pld.window(state_meta_window_buf, [CP_SIZE, STATE_META_DIM], dtype=pl.INT32)
        compact_ready = pld.window(compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        compact_consumed = pld.window(compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        prefill_cp_hca_rank(x_hc[rank], hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state[rank], compress_state_block_table[rank], kv_cache[rank], cmp_kv[rank], cmp_block_table[rank], segment_starts_t, segment_active_lengths[rank], owner_segments_t[rank], predecessor_segments[rank], query_positions[rank], query_requests[rank], overlay_positions[rank], overlay_requests[rank], overlay_active_lengths[rank], swa_indices[rank], cmp_indices[rank], segment_tail_positions, snapshot_positions[rank], snapshot_valid[rank], final_segment_t, reverse_index, owner_rank_table, owner_part_table, final_win_seg_src, final_win_row_src, final_slot_mapping, hidden_tail_window, kv_tail_window, tail_ready, tail_consumed, cmp_window, cmp_meta_window, state_window, state_meta_window, compact_ready, compact_consumed, attn_sink, wo_a, wo_b, wo_b_scale, x_out[rank], rank, device=rank)
