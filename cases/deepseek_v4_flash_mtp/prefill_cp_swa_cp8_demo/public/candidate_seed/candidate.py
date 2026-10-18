import pypto.language as pl

import pypto.language.distributed as pld

from config import BLOCK_SIZE, CASE_CONFIG as M, CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, FP32_NEG_INF, PREFILL_ORI_MAX_BLOCKS

from prefill_cp_zigzag import CP_TAIL_WINDOW_ROWS, HEAD_DIM, _prefill_cp_zigzag_kv_tail_exchange_wave

from hc_post import hc_post_prefill

from hc_pre import hc_pre

from qkv_proj_rope import materialize_rope_rows, qkv_proj_rope

from rmsnorm import rms_norm

from prefill_sparse_attn import BIAS_TOKEN_TILE, PREFILL_ATTN_BLOCKS, PREFILL_ATTN_TILE, PREFILL_SPARSE_PAD, SPARSE_BIAS_COLS, VALID_BLOCK_MASK_COLS, _prefill_sparse_attn_math

D = M.hidden_size

H = M.num_attention_heads

HC_DIM = M.hc_dim

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

O_GROUPS = M.o_groups

O_GROUP_IN = H * M.head_dim // O_GROUPS

O_LORA = M.o_lora_rank

Q_LORA = M.q_lora_rank

ROPE_HEAD_DIM = M.qk_rope_head_dim

ROPE_DIM = ROPE_HEAD_DIM

MAX_SEQ_LEN = M.max_position_embeddings

NUM_SEGMENTS = 2 * CP_SIZE

WIN = M.sliding_window

TAIL_ROWS = WIN

BLOCK_ROWS = BLOCK_SIZE

NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES

LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS

LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD

ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS

ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_ROWS

OVERLAY_BASE = ORI_CACHE_ROWS

PRED_OVERLAY_ROWS = TAIL_ROWS

CUR_OVERLAY_ROWS = TAIL_ROWS

OVERLAY_ROWS = PRED_OVERLAY_ROWS + CUR_OVERLAY_ROWS

OVERLAY_SOURCES = 2

@pl.jit.inline
def _cp_swa_stage_sources(cache_flat: pl.Tensor[[ORI_CACHE_ROWS, HEAD_DIM], pl.BF16], local_kv: pl.Tensor[[LOCAL_ROWS, HEAD_DIM], pl.BF16], logical_tails: pl.Tensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16], query_positions: pl.Tensor[[LOCAL_ROWS], pl.INT32], query_requests: pl.Tensor[[LOCAL_ROWS], pl.INT32], overlay_positions: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_ROWS], pl.INT32], predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], swa_indices: pl.Tensor[[LOCAL_ROWS, WIN], pl.INT32], sparse_kv: pl.Tensor[[LOCAL_SPARSE_ROWS, HEAD_DIM], pl.BF16], sparse_bias: pl.Tensor[[NUM_LOCAL_TILES * TAIL_ROWS, PREFILL_SPARSE_PAD], pl.FP32], valid_block_mask: pl.Tensor[[NUM_LOCAL_TILES * TAIL_ROWS, VALID_BLOCK_MASK_COLS], pl.INT32], overlay_active_lengths: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_SOURCES], pl.INT32]):
    """Stage the accepted persistent/predecessor/current source ABI."""
    prefix = pl.read(segment_starts_t, [0])
    with pl.spmd(LOCAL_ROWS // 2 * PREFILL_ATTN_BLOCKS, name_hint='gather_kv') as gather_tid:
        block = pl.tile.get_block_idx()
        schedule = block // PREFILL_ATTN_BLOCKS
        sb = block - schedule * PREFILL_ATTN_BLOCKS
        token_block = LOCAL_ROWS // 2 - 1 - schedule
        t0 = token_block * 2
        k0 = sb * PREFILL_ATTN_TILE
        for dt in pl.range(2):
            row = t0 + dt
            if row < LOCAL_ROWS:
                stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                out_base = row * PREFILL_SPARSE_PAD + k0
                for ki in pl.range(PREFILL_ATTN_TILE):
                    col = k0 + ki
                    if col < WIN:
                        raw = pl.read(swa_indices, [row, col])
                        if raw >= 0:
                            q_abs = pl.read(query_positions, [row])
                            q_req = pl.read(query_requests, [row])
                            key_abs = q_abs - WIN + 1 + col
                            if raw < ORI_CACHE_ROWS:
                                if key_abs < prefix and key_abs <= q_abs and (q_req >= 0):
                                    src = pl.cast(raw, pl.INDEX)
                                    stage[ki:ki + 1, :] = cache_flat[src:src + 1, :]
                            elif raw < OVERLAY_BASE + OVERLAY_ROWS:
                                tile = row // TAIL_ROWS
                                ov_row = raw - OVERLAY_BASE
                                if ov_row >= PRED_OVERLAY_ROWS:
                                    source_kind = 1
                                    src_row = ov_row - PRED_OVERLAY_ROWS
                                else:
                                    source_kind = 0
                                    src_row = ov_row
                                ov_idx = src_row
                                if source_kind == 1:
                                    ov_idx = PRED_OVERLAY_ROWS + src_row
                                ov_active = pl.read(overlay_active_lengths, [tile, source_kind])
                                ov_abs = pl.read(overlay_positions, [tile, ov_idx])
                                ov_req = pl.read(overlay_requests, [tile, ov_idx])
                                if src_row >= 0 and src_row < ov_active:
                                    if ov_abs == key_abs and ov_abs <= q_abs and (ov_req == q_req) and (ov_req >= 0):
                                        if source_kind == 1:
                                            src = tile * TAIL_ROWS + src_row
                                            stage[ki:ki + 1, :] = local_kv[src:src + 1, :]
                                        elif tile % MAX_SEGMENT_TILES == 0:
                                            part = tile // MAX_SEGMENT_TILES
                                            pred = pl.read(predecessor_segments, [part])
                                            if pred >= 0:
                                                src = pred * TAIL_ROWS + src_row
                                                stage[ki:ki + 1, :] = logical_tails[src:src + 1, :]
                                        else:
                                            src = (tile - 1) * TAIL_ROWS + src_row
                                            stage[ki:ki + 1, :] = local_kv[src:src + 1, :]
                sparse_kv[out_base:out_base + PREFILL_ATTN_TILE, :] = stage
    with pl.spmd(LOCAL_ROWS // BIAS_TOKEN_TILE, name_hint='build_bias') as bias_tid:
        bias_blk = pl.tile.get_block_idx()
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_idx = pl.cast(swa_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN], target_type=pl.FP32)
        flags = pl.minimum(pl.maximum(pl.add(bias_idx, 1.0), 0.0), 1.0)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN] = pl.mul(pl.sub(flags, 1.0), -FP32_NEG_INF)
        if WIN < PREFILL_SPARSE_PAD:
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, WIN:PREFILL_SPARSE_PAD] = pl.full([BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD - WIN], dtype=pl.FP32, value=FP32_NEG_INF)
    return (gather_tid, bias_tid)

@pl.jit.inline
def prefill_cp_swa_core(x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16]], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32], query_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], query_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], overlay_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32], kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16], ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], x_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]], my_rank: pl.Scalar[pl.INT32]):
    """CP-SWA attention math (inline). Shared by the standalone rank child
    and the layer composition child. Inlining avoids child-in-child nesting
    (@pl.jit cannot call another @pl.jit)."""
    q = pl.create_tensor([LOCAL_ROWS, H, HEAD_DIM], dtype=pl.BF16)
    q_initial_flat = pl.reshape(q, [LOCAL_ROWS * H * HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_swa_initialize_q'):
        for q_initial_block in pl.range(LOCAL_ROWS * H * HEAD_DIM // 256):
            q_initial_flat[q_initial_block:q_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    post = pl.create_tensor([LOCAL_ROWS, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([LOCAL_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    rope_cos_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_cos_flat_initial_flat = pl.reshape(rope_cos_flat, [LOCAL_ROWS * ROPE_HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_swa_initialize_rope_cos_flat'):
        for rope_cos_flat_initial_block in pl.range(LOCAL_ROWS * ROPE_HEAD_DIM // 256):
            rope_cos_flat_initial_flat[rope_cos_flat_initial_block:rope_cos_flat_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    rope_sin_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_flat_initial_flat = pl.reshape(rope_sin_flat, [LOCAL_ROWS * ROPE_HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_swa_initialize_rope_sin_flat'):
        for rope_sin_flat_initial_block in pl.range(LOCAL_ROWS * ROPE_HEAD_DIM // 256):
            rope_sin_flat_initial_flat[rope_sin_flat_initial_block:rope_sin_flat_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    logical_tails = pl.create_tensor([CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    sparse_kv = pl.create_tensor([LOCAL_SPARSE_ROWS, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([LOCAL_ROWS, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    x_flat = pl.reshape(x_hc, [NUM_LOCAL_TILES * TAIL_ROWS, HC_MULT, D])
    qr = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, 1], dtype=pl.FP32)
    local_kv = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    x_mixed = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, D], dtype=pl.BF16)
    normed = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, D], dtype=pl.BF16)
    q_pos_flat = pl.reshape(query_positions, [NUM_LOCAL_TILES * TAIL_ROWS])
    q_req_flat = pl.reshape(query_requests, [NUM_LOCAL_TILES * TAIL_ROWS])
    ov_pos_flat = pl.reshape(overlay_positions, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    ov_req_flat = pl.reshape(overlay_requests, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    ov_active_flat = pl.reshape(overlay_active_lengths, [NUM_LOCAL_TILES, OVERLAY_SOURCES])
    swa_flat = pl.reshape(swa_indices, [NUM_LOCAL_TILES * TAIL_ROWS, WIN])
    for tile in pl.range(NUM_LOCAL_TILES):
        t0 = tile * TAIL_ROWS
        x_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [t0, 0, 0])
        mixed_tile = pl.slice(x_mixed, [TAIL_ROWS, D], [t0, 0])
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [t0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [t0, 0])
        position_tile = pl.slice(q_pos_flat, [TAIL_ROWS], [t0])
        rope_cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [t0, 0])
        rope_sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [t0, 0])
        normed_tile = pl.slice(normed, [TAIL_ROWS, D], [t0, 0])
        qr_tile = pl.slice(qr, [TAIL_ROWS, Q_LORA], [t0, 0])
        qr_scale_tile = pl.slice(qr_scale, [TAIL_ROWS, 1], [t0, 0])
        q_tile = pl.slice(q, [TAIL_ROWS, H, HEAD_DIM], [t0, 0, 0])
        kv_tile = pl.slice(local_kv, [TAIL_ROWS, HEAD_DIM], [t0, 0])
        hc_pre(x_tile, hc_attn_fn, hc_attn_scale, hc_attn_base, mixed_tile, post_tile, comb_tile)
        rms_tid = rms_norm(mixed_tile, attn_norm_w, normed_tile)
        late_dep = pl.system.task_dummy(deps=[rms_tid])
        active = pl.read(overlay_active_lengths, [tile // MAX_SEGMENT_TILES, tile % MAX_SEGMENT_TILES, 1])
        materialize_rope_rows(freqs_cos, freqs_sin, position_tile, active, rope_cos_tile, rope_sin_tile)
        qkv_proj_rope(normed_tile, wq_a, wq_b, wq_b_scale, wkv, rope_cos_tile, rope_sin_tile, gamma_cq, gamma_ckv, q_tile, kv_tile, qr_tile, qr_scale_tile, late_dep)
    local_tail = pl.create_tensor([2 * TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    local_tail_initial_flat = pl.reshape(local_tail, [2 * TAIL_ROWS * HEAD_DIM // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_swa_initialize_local_tail'):
        for local_tail_initial_block in pl.range(2 * TAIL_ROWS * HEAD_DIM // 256):
            local_tail_initial_flat[local_tail_initial_block:local_tail_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.BF16, value=0.0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_tail_assemble'):
        for part in pl.range(LOCAL_PARTS):
            tile0_len = pl.read(overlay_active_lengths, [part, 0, 1])
            tile1_len = pl.read(overlay_active_lengths, [part, 1, 1])
            segment_total = tile0_len + tile1_len
            tail_start = pl.max(segment_total - TAIL_ROWS, 0)
            for row in pl.range(TAIL_ROWS):
                tail_offset = tail_start + row
                if tail_offset < segment_total:
                    if tail_offset < TAIL_ROWS:
                        src = part * MAX_SEGMENT_TILES * TAIL_ROWS + tail_offset
                    else:
                        src = part * MAX_SEGMENT_TILES * TAIL_ROWS + TAIL_ROWS + tail_offset - TAIL_ROWS
                    local_tail[part * TAIL_ROWS + row:part * TAIL_ROWS + row + 1] = local_kv[src:src + 1]
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_swa_tail_exchange'):
        _prefill_cp_zigzag_kv_tail_exchange_wave(local_tail, reverse_index, owner_rank_table, kv_tail_window, ready, consumed, logical_tails, my_rank, pl.cast(0, pl.INT32))
    cache_flat = pl.reshape(kv_cache, [ORI_CACHE_ROWS, HEAD_DIM])
    valid_mask = pl.create_tensor([LOCAL_ROWS, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    valid_mask_initial_flat = pl.reshape(valid_mask, [LOCAL_ROWS * VALID_BLOCK_MASK_COLS // 256, 256])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_swa_initialize_valid_mask'):
        for valid_mask_initial_block in pl.range(LOCAL_ROWS * VALID_BLOCK_MASK_COLS // 256):
            valid_mask_initial_flat[valid_mask_initial_block:valid_mask_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.INT32, value=0)
    gather_tid, bias_tid = _cp_swa_stage_sources(cache_flat, local_kv, logical_tails, q_pos_flat, q_req_flat, ov_pos_flat, ov_req_flat, predecessor_segments, segment_starts_t, swa_flat, sparse_kv, sparse_bias, valid_mask, ov_active_flat)
    cache_commit_flat = pl.reshape(kv_cache, [ORI_CACHE_ROWS, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_swa_cache_commit', deps=[gather_tid]):
        for row in pl.range(TAIL_ROWS):
            seg = final_win_seg_src[row]
            src_row = final_win_row_src[row]
            dst = final_slot_mapping[row]
            if seg >= 0 and src_row >= 0 and (dst >= 0):
                src = seg * TAIL_ROWS + src_row
                cache_commit_flat[dst:dst + 1] = logical_tails[src:src + 1]
    x_flat = pl.reshape(x_hc, [LOCAL_ROWS, HC_MULT, D])
    x_out_flat = pl.reshape(x_out, [LOCAL_ROWS, HC_MULT, D])
    for tile in pl.range(NUM_LOCAL_TILES):
        t0 = tile * TAIL_ROWS
        sparse0 = tile * TAIL_ROWS * PREFILL_SPARSE_PAD
        part = tile // MAX_SEGMENT_TILES
        part_tile = tile % MAX_SEGMENT_TILES
        q_tile = pl.slice(q, [TAIL_ROWS, H, HEAD_DIM], [t0, 0, 0])
        sparse_kv_tile = pl.slice(sparse_kv, [TAIL_ROWS * PREFILL_SPARSE_PAD, HEAD_DIM], [sparse0, 0])
        bias_tile = pl.slice(sparse_bias, [TAIL_ROWS, PREFILL_SPARSE_PAD], [t0, 0])
        mask_tile = pl.slice(valid_mask, [TAIL_ROWS, VALID_BLOCK_MASK_COLS], [t0, 0])
        cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_DIM], [t0, 0])
        sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_DIM], [t0, 0])
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [t0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [t0, 0])
        x_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [t0, 0, 0])
        active = pl.read(overlay_active_lengths, [part, part_tile, 1])
        attn_out_tile = pl.create_tensor([TAIL_ROWS, D], dtype=pl.BF16)
        y_tile = pl.create_tensor([TAIL_ROWS, HC_MULT, D], dtype=pl.FP32)
        y_tile_initial_flat = pl.reshape(y_tile, [TAIL_ROWS * HC_MULT * D // 256, 256])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='cp_swa_initialize_y_tile'):
            for y_tile_initial_block in pl.range(TAIL_ROWS * HC_MULT * D // 256):
                y_tile_initial_flat[y_tile_initial_block:y_tile_initial_block + 1, 0:256] = pl.full([1, 256], dtype=pl.FP32, value=0.0)
        _prefill_sparse_attn_math(q=q_tile, sparse_kv=sparse_kv_tile, sparse_bias=bias_tile, valid_block_mask=mask_tile, attn_sink=attn_sink, freqs_cos=cos_tile, freqs_sin=sin_tile, wo_a=wo_a, wo_b=wo_b, wo_b_scale=wo_b_scale, attn_out=attn_out_tile, num_tokens=active)
        hc_post_prefill(attn_out_tile, x_tile, post_tile, comb_tile, y_tile, active)
        x_out_flat[t0:t0 + TAIL_ROWS, 0:HC_MULT, 0:D] = y_tile
    x_out = pl.reshape(x_out_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D])
    return x_out

@pl.jit
def prefill_cp_swa_rank(x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16]], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32], query_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], query_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], overlay_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32], kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16], ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], x_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]], my_rank: pl.Scalar[pl.INT32]):
    """Standalone CP-SWA rank child. Delegates to the inline core so the
    standalone test preserves the original @pl.jit entry point."""
    return prefill_cp_swa_core(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, kv_cache, attn_sink, wo_a, wo_b, wo_b_scale, segment_starts_t, predecessor_segments, query_positions, query_requests, overlay_positions, overlay_requests, overlay_active_lengths, swa_indices, reverse_index, owner_rank_table, final_win_seg_src, final_win_row_src, final_slot_mapping, kv_tail_window, ready, consumed, x_out, my_rank)

@pl.jit.host
def prefill_cp_swa_test(x_hc: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[CP_SIZE, ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16]], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16], wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8], wo_b_scale: pl.Tensor[[D], pl.FP32], segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32], predecessor_segments: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32], query_position_ids: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], query_token_to_request: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32], overlay_position_ids: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_token_to_request: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32], overlay_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32], swa_indices: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32], final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32], x_out: pl.Out[pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]]):
    """Launch one CP-SWA child per rank."""
    window_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        window = pld.window(window_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
        ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        consumed = pld.window(consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        prefill_cp_swa_rank(x_hc[rank], hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, kv_cache[rank], attn_sink, wo_a, wo_b, wo_b_scale, segment_starts_t, predecessor_segments[rank], query_position_ids[rank], query_token_to_request[rank], overlay_position_ids[rank], overlay_token_to_request[rank], overlay_active_lengths[rank], swa_indices[rank], reverse_index, owner_rank_table, final_win_seg_src, final_win_row_src, final_slot_mapping, window, ready, consumed, x_out[rank], rank, device=rank)
