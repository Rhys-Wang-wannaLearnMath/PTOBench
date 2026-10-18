from config import T, COMPRESS_RATIO, STATE_LEN, LOCAL_PARTS, MAX_SEGMENT_TILES, ORI_MAX_BLOCKS, ORI_CACHE_ROWS, OVERLAY_ROWS, MAIN_STATE_BLOCK_SIZE, INNER_STATE_BLOCK_SIZE, CP_CANDIDATE_CAPACITY, SPARSE_SELECTED_WIDTH, NUM_LOCAL_TILES, LOCAL_ROWS, QK_ROPE_HALF

from prefill_indexer import _int8_quant_per_row

from prefill_cp_zigzag import CP_SIZE, NUM_SEGMENTS, cp_owner_rank, cp_owner_part

import torch

from config import BLOCK_SIZE, CASE_CONFIG as M, PREFILL_CMP_BLOCK_NUM, PREFILL_IDX_BLOCK_NUM, PREFILL_ORI_MAX_BLOCKS

from hc_post import golden_hc_post_prefill

from hc_pre import golden_hc_pre

from prefill_compressor_ratio4 import HEAD_DIM, golden_prefill_compressor_ratio4

from prefill_cp_exchange import INNER_STATE_DIM, MAIN_STATE_DIM

from prefill_cp_zigzag import NUM_SEGMENTS, TAIL_ROWS

from prefill_indexer import IDX_HEAD_DIM, IDX_N_HEADS, IDX_TOPK

from prefill_indexer_compressor import golden_prefill_indexer_compressor

from prefill_sparse_attn import golden_prefill_sparse_attn

from qkv_proj_rope import golden_qkv_proj_rope

from rmsnorm import golden_rms_norm

D = M.hidden_size

H = M.num_attention_heads

HC_MULT = M.hc_mult

Q_LORA = M.q_lora_rank

ROPE_HEAD_DIM = M.qk_rope_head_dim















MAIN_LEAF_CACHE_BLOCKS = PREFILL_CMP_BLOCK_NUM

IDX_LEAF_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM


def _active_tile(length: int, tile: int) -> int:
    begin = tile * T
    return max(0, min(T, length - begin))

def _lower_row(table: torch.Tensor, logical_slot: int, block_size: int) -> int:
    if logical_slot < 0:
        return -1
    logical_block = logical_slot // block_size
    if logical_block >= table.numel():
        return -1
    physical_block = int(table[logical_block].item())
    if physical_block < 0:
        return -1
    return physical_block * block_size + logical_slot % block_size

def golden_prefill_cp_csa(tensors):
    """Compose CP-CSA golden outputs in logical-segment commit order."""
    ctx = getattr(golden_prefill_cp_csa, '_ctx', None)
    if ctx is None:
        raise RuntimeError('CP-CSA golden context was not installed')
    cp_size = int(ctx['cp_size'])
    starts = [int(value) for value in ctx['starts']]
    lengths = [int(value) for value in ctx['lengths']]
    owners = ctx['owners']
    reverse = tensors['reverse_index'].tolist()
    local_norm = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, D, dtype=torch.bfloat16)
    local_q = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, H, HEAD_DIM, dtype=torch.bfloat16)
    local_kv = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HEAD_DIM, dtype=torch.bfloat16)
    local_qr = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, Q_LORA, dtype=torch.int8)
    local_qr_scale = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, 1, dtype=torch.float32)
    local_post = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT)
    local_comb = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT * HC_MULT)
    logical_norm: dict[int, torch.Tensor] = {}
    logical_kv: dict[int, torch.Tensor] = {}
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = int(owners[rank][part])
            active_segment = lengths[segment]
            segment_norm_rows = []
            segment_kv_rows = []
            for tile in range(MAX_SEGMENT_TILES):
                active = _active_tile(active_segment, tile)
                x_tile = tensors['x_hc'][rank, part, tile]
                mixed = torch.zeros(T, D, dtype=torch.bfloat16)
                post = torch.zeros(T, HC_MULT)
                comb = torch.zeros(T, HC_MULT * HC_MULT)
                golden_hc_pre({'x': x_tile, 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': mixed, 'post': post, 'comb': comb})
                normed = golden_rms_norm(mixed, tensors['attn_norm_w'])
                positions = tensors['query_positions'][rank, part, tile]
                rope_rows = positions.clamp_min(0).long()
                q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
                kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
                qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
                qr_scale = torch.zeros(T, 1)
                golden_qkv_proj_rope({'x': normed, 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': tensors['freqs_cos'].index_select(0, rope_rows), 'rope_sin': tensors['freqs_sin'].index_select(0, rope_rows), 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': qr, 'qr_scale': qr_scale})
                local_norm[rank, part, tile] = normed
                local_q[rank, part, tile] = q
                local_kv[rank, part, tile] = kv
                local_qr[rank, part, tile] = qr
                local_qr_scale[rank, part, tile] = qr_scale
                local_post[rank, part, tile] = post
                local_comb[rank, part, tile] = comb
                if active:
                    segment_norm_rows.append(normed[:active])
                    segment_kv_rows.append(kv[:active])
            logical_norm[segment] = torch.cat(segment_norm_rows) if segment_norm_rows else torch.zeros(0, D, dtype=torch.bfloat16)
            logical_kv[segment] = torch.cat(segment_kv_rows) if segment_kv_rows else torch.zeros(0, HEAD_DIM, dtype=torch.bfloat16)
    cmp_published: dict[int, torch.Tensor] = {}
    idx_published: dict[int, torch.Tensor] = {}
    scale_published: dict[int, torch.Tensor] = {}
    final_main_state = None
    final_inner_state = None
    final_owner_rank = -1
    for segment in range(NUM_SEGMENTS):
        source_slot = int(reverse[segment])
        rank = source_slot // LOCAL_PARTS
        part = source_slot % LOCAL_PARTS
        main_state = torch.zeros_like(tensors['compress_state'][rank])
        inner_state = torch.zeros_like(tensors['inner_compress_state'][rank])
        if segment == 0:
            main_state.copy_(tensors['compress_state'][rank])
            inner_state.copy_(tensors['inner_compress_state'][rank])
        leaves: list[tuple[torch.Tensor, torch.Tensor]] = []
        if segment > 0 and lengths[segment] > 0:
            predecessor = segment - 1
            predecessor_rows = logical_norm[predecessor]
            seed_len = min(starts[segment] % COMPRESS_RATIO + COMPRESS_RATIO, min(T, lengths[predecessor]))
            seed = torch.zeros(T, D, dtype=torch.bfloat16)
            positions = torch.zeros(T, dtype=torch.int32)
            if seed_len:
                seed[:seed_len] = predecessor_rows[-seed_len:]
                positions[:seed_len] = torch.arange(starts[segment] - seed_len, starts[segment], dtype=torch.int32)
            leaves.append((seed, positions))
        else:
            leaves.append((torch.zeros(T, D, dtype=torch.bfloat16), torch.zeros(T, dtype=torch.int32)))
        for tile in range(MAX_SEGMENT_TILES):
            leaves.append((local_norm[rank, part, tile], tensors['query_positions'][rank, part, tile]))
        for (leaf, (x_leaf, positions)) in enumerate(leaves):
            if leaf == 0:
                active = min(starts[segment] % COMPRESS_RATIO + COMPRESS_RATIO, min(T, lengths[segment - 1])) if segment > 0 and lengths[segment] > 0 else 0
            else:
                active = _active_tile(lengths[segment], leaf - 1)
            if active <= 0:
                continue
            main_cache = torch.zeros(MAIN_LEAF_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
            idx_cache = torch.zeros(IDX_LEAF_CACHE_BLOCKS, BLOCK_SIZE, 1, IDX_HEAD_DIM, dtype=torch.int8)
            idx_scale = torch.zeros(IDX_LEAF_CACHE_BLOCKS, BLOCK_SIZE, 1, 1)
            main_map = torch.full((T,), -1, dtype=torch.int64)
            idx_map = torch.full((T,), -1, dtype=torch.int64)
            main_state_map = torch.full((T,), -1, dtype=torch.int64)
            inner_state_map = torch.full((T,), -1, dtype=torch.int64)
            write_count = 0
            for row in range(active):
                position = int(positions[row])
                main_state_map[row] = _lower_row(tensors['compress_state_block_table'][rank], position, MAIN_STATE_BLOCK_SIZE)
                inner_state_map[row] = _lower_row(tensors['inner_compress_state_block_table'][rank], position, INNER_STATE_BLOCK_SIZE)
                if leaf > 0 and (position + 1) % COMPRESS_RATIO == 0:
                    main_map[row] = write_count
                    idx_map[row] = write_count
                    write_count += 1
            golden_prefill_compressor_ratio4({'x': x_leaf, 'compress_state': main_state, 'compress_state_block_table': tensors['compress_state_block_table'][rank], 'wkv': tensors['cmp_wkv'], 'wgate': tensors['cmp_wgate'], 'ape': tensors['cmp_ape'], 'norm_w': tensors['cmp_norm_w'], 'freqs_cos': tensors['freqs_cos'], 'freqs_sin': tensors['freqs_sin'], 'cmp_kv': main_cache, 'position_ids': positions, 'num_tokens': active, 'cmp_slot_mapping': main_map, 'state_slot_mapping': main_state_map})
            golden_prefill_indexer_compressor({'x': x_leaf, 'compress_state': inner_state, 'inner_compress_state_block_table': tensors['inner_compress_state_block_table'][rank], 'wkv': tensors['inner_wkv'], 'wgate': tensors['inner_wgate'], 'ape': tensors['inner_ape'], 'norm_w': tensors['inner_norm_w'], 'freqs_cos': tensors['freqs_cos'], 'freqs_sin': tensors['freqs_sin'], 'hadamard': tensors['hadamard_idx'], 'kv': torch.zeros(T // COMPRESS_RATIO, IDX_HEAD_DIM, dtype=torch.int8), 'idx_kv_cache': idx_cache, 'idx_kv_scale': idx_scale, 'idx_block_table': tensors['idx_block_table'][rank], 'position_ids': positions, 'num_tokens': active, 'idx_slot_mapping': idx_map, 'inner_state_slot_mapping': inner_state_map})
            for row in range(active):
                local_row = int(main_map[row])
                if local_row >= 0:
                    logical_slot = (int(positions[row]) + 1) // COMPRESS_RATIO - 1
                    cmp_published[logical_slot] = main_cache.view(-1, HEAD_DIM)[local_row].clone()
                    idx_published[logical_slot] = idx_cache.view(-1, IDX_HEAD_DIM)[local_row].clone()
                    scale_published[logical_slot] = idx_scale.view(-1, 1)[local_row].clone()
        if segment == int(ctx['final_segment']):
            final_main_state = main_state
            final_inner_state = inner_state
            final_owner_rank = rank
    cmp_out = tensors['cmp_kv'].clone()
    idx_out = tensors['idx_kv_cache'].clone()
    scale_out = tensors['idx_kv_scale'].clone()
    for rank in range(cp_size):
        cmp_flat = cmp_out[rank].view(-1, HEAD_DIM)
        idx_flat = idx_out[rank].view(-1, IDX_HEAD_DIM)
        scale_flat = scale_out[rank].view(-1, 1)
        for (logical_slot, value) in cmp_published.items():
            row = _lower_row(tensors['cmp_block_table'][rank], logical_slot, BLOCK_SIZE)
            if row >= 0:
                cmp_flat[row] = value
        for (logical_slot, value) in idx_published.items():
            row = _lower_row(tensors['idx_block_table'][rank], logical_slot, BLOCK_SIZE)
            if row >= 0:
                idx_flat[row] = value
                scale_flat[row] = scale_published[logical_slot]
    main_state_out = tensors['compress_state'].clone()
    inner_state_out = tensors['inner_compress_state'].clone()
    final_end = starts[int(ctx['final_segment'])] + lengths[int(ctx['final_segment'])]
    final_positions = range(max(0, final_end - STATE_LEN), final_end)
    for receiver in range(cp_size):
        for position in final_positions:
            source = _lower_row(tensors['compress_state_block_table'][final_owner_rank], position, MAIN_STATE_BLOCK_SIZE)
            destination = _lower_row(tensors['compress_state_block_table'][receiver], position, MAIN_STATE_BLOCK_SIZE)
            if source >= 0 and destination >= 0:
                main_state_out[receiver].view(-1, MAIN_STATE_DIM)[destination] = final_main_state.view(-1, MAIN_STATE_DIM)[source]
            source = _lower_row(tensors['inner_compress_state_block_table'][final_owner_rank], position, INNER_STATE_BLOCK_SIZE)
            destination = _lower_row(tensors['inner_compress_state_block_table'][receiver], position, INNER_STATE_BLOCK_SIZE)
            if source >= 0 and destination >= 0:
                inner_state_out[receiver].view(-1, INNER_STATE_DIM)[destination] = final_inner_state.view(-1, INNER_STATE_DIM)[source]
    topk_by_rank = []
    for rank in range(cp_size):
        norm_flat = local_norm[rank].reshape(LOCAL_ROWS, D)
        qr_flat = local_qr[rank].reshape(LOCAL_ROWS, Q_LORA)
        qr_scale_flat = local_qr_scale[rank].reshape(LOCAL_ROWS, 1)
        positions = tensors['query_positions'][rank].reshape(LOCAL_ROWS)
        q_i32 = qr_flat.to(torch.int32) @ tensors['idx_wq_b'].to(torch.int32)
        query = (q_i32.float() * qr_scale_flat * tensors['idx_wq_b_scale'].float().view(1, -1)).view(LOCAL_ROWS, IDX_N_HEADS, IDX_HEAD_DIM)
        rope_pair = query[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        rope_rows = positions.clamp_min(0).long()
        cos = tensors['freqs_cos'].index_select(0, rope_rows)[:, :QK_ROPE_HALF].float().unsqueeze(1)
        sin = tensors['freqs_sin'].index_select(0, rope_rows)[:, :QK_ROPE_HALF].float().unsqueeze(1)
        query = torch.cat([query[..., :-ROPE_HEAD_DIM], torch.stack([rope_pair[..., 0] * cos - rope_pair[..., 1] * sin, rope_pair[..., 0] * sin + rope_pair[..., 1] * cos], dim=-1).flatten(-2).to(torch.bfloat16).float()], dim=-1)
        query = query @ tensors['hadamard_idx'].float()
        weights = norm_flat.float() @ tensors['idx_weights_proj'].float() * M.index_weights_scale
        (query_i8, query_scale) = _int8_quant_per_row(query.reshape(LOCAL_ROWS * IDX_N_HEADS, IDX_HEAD_DIM))
        max_visible = min(CP_CANDIDATE_CAPACITY, max(0, (int(positions.max()) + 1) // COMPRESS_RATIO))
        cache_rows = []
        cache_scales = []
        idx_flat = idx_out[rank].view(-1, IDX_HEAD_DIM)
        scale_flat = scale_out[rank].view(-1)
        for logical_slot in range(max_visible):
            row = _lower_row(tensors['idx_block_table'][rank], logical_slot, BLOCK_SIZE)
            cache_rows.append(idx_flat[row] if row >= 0 else torch.zeros(IDX_HEAD_DIM, dtype=torch.int8))
            cache_scales.append(float(scale_flat[row]) if row >= 0 else 0.0)
        topk = torch.full((LOCAL_ROWS, IDX_TOPK), -1, dtype=torch.int32)
        if max_visible:
            cache_i8 = torch.stack(cache_rows).to(torch.int32)
            cache_scale = torch.tensor(cache_scales).view(1, 1, -1)
            query_i32 = query_i8.view(LOCAL_ROWS, IDX_N_HEADS, IDX_HEAD_DIM).to(torch.int32)
            query_scale = query_scale.view(LOCAL_ROWS, IDX_N_HEADS, 1)
            score = torch.einsum('thd,cd->thc', query_i32, cache_i8).float() * query_scale * cache_scale
            score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)
            for row in range(LOCAL_ROWS):
                visible = min(CP_CANDIDATE_CAPACITY, max(0, (int(positions[row]) + 1) // COMPRESS_RATIO))
                selected = min(SPARSE_SELECTED_WIDTH, visible)
                if selected:
                    topk[row, :selected] = score[row, :visible].topk(selected).indices.to(torch.int32)
        topk_by_rank.append(topk)
    kv_before_commit = tensors['kv_cache'].clone()
    x_out = torch.zeros_like(tensors['x_out'])
    for rank in range(cp_size):
        topk = topk_by_rank[rank].view(LOCAL_PARTS, MAX_SEGMENT_TILES, T, IDX_TOPK)
        for part in range(LOCAL_PARTS):
            segment = int(owners[rank][part])
            for tile in range(MAX_SEGMENT_TILES):
                active = _active_tile(lengths[segment], tile)
                overlay = torch.zeros(OVERLAY_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                predecessor_len = int(tensors['overlay_active_lengths'][rank, part, tile, 0])
                current_len = int(tensors['overlay_active_lengths'][rank, part, tile, 1])
                if predecessor_len:
                    if tile == 0:
                        predecessor = segment - 1
                        overlay[:predecessor_len] = logical_kv[predecessor][-predecessor_len:]
                    else:
                        overlay[:predecessor_len] = local_kv[rank, part, tile - 1, :predecessor_len]
                overlay[T:T + current_len] = local_kv[rank, part, tile, :current_len]
                sparse_source = torch.zeros(ORI_CACHE_ROWS + OVERLAY_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                sparse_source[:ORI_CACHE_ROWS] = kv_before_commit[rank].view(-1, HEAD_DIM)
                sparse_source[ORI_CACHE_ROWS:] = overlay
                sparse_source_cache = sparse_source.view((ORI_CACHE_ROWS + OVERLAY_ROWS) // BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM)
                attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
                positions = tensors['query_positions'][rank, part, tile]
                golden_prefill_sparse_attn({'q': local_q[rank, part, tile], 'ori_kv': sparse_source_cache, 'swa_indices': tensors['swa_indices'][rank, part, tile], 'cmp_kv': cmp_out[rank], 'cmp_block_table': tensors['cmp_block_table'][rank], 'cmp_indices': topk[part, tile], 'attn_sink': tensors['attn_sink'], 'num_tokens': active, 'freqs_cos': tensors['freqs_cos'].index_select(0, positions.clamp_min(0).long()), 'freqs_sin': tensors['freqs_sin'].index_select(0, positions.clamp_min(0).long()), 'wo_a': tensors['wo_a'], 'wo_b': tensors['wo_b'], 'wo_b_scale': tensors['wo_b_scale'], 'overlay_kv': overlay, 'overlay_position_ids': tensors['overlay_positions'][rank, part, tile], 'overlay_token_to_request': tensors['overlay_requests'][rank, part, tile], 'overlay_active_lengths': tensors['overlay_active_lengths'][rank, part, tile], 'query_position_ids': positions, 'query_token_to_request': tensors['query_requests'][rank, part, tile], 'attn_out': attn_out})
                y = torch.zeros(T, HC_MULT, D)
                golden_hc_post_prefill({'x': attn_out, 'residual': tensors['x_hc'][rank, part, tile], 'post': local_post[rank, part, tile], 'comb': local_comb[rank, part, tile], 'y': y, 'num_tokens': active})
                x_out[rank, part, tile] = y
    kv_out = kv_before_commit.clone()
    for row in range(T):
        segment = int(tensors['final_win_seg_src'][row])
        source_row = int(tensors['final_win_row_src'][row])
        destination = int(tensors['final_slot_mapping'][row])
        if segment >= 0 and source_row >= 0 and (destination >= 0):
            value = logical_kv[segment][-min(T, lengths[segment]):][source_row]
            for rank in range(cp_size):
                kv_out[rank].view(-1, HEAD_DIM)[destination] = value
    tensors['compress_state'][:] = main_state_out
    tensors['inner_compress_state'][:] = inner_state_out
    tensors['cmp_kv'][:] = cmp_out
    tensors['idx_kv_cache'][:] = idx_out
    tensors['idx_kv_scale'][:] = scale_out
    tensors['kv_cache'][:] = kv_out
    tensors['x_out'][:] = x_out


def compute(inputs, scalars):
    assert not scalars
    values = {n: v.clone() for n, v in inputs.items()}
    values['x_out'] = torch.zeros_like(values['x_hc'])
    golden_prefill_cp_csa._ctx = {
        'cp_size': CP_SIZE,
        'starts': values['segment_starts_t'].tolist(),
        'lengths': [int(values['segment_active_lengths'][cp_owner_rank(s, CP_SIZE), cp_owner_part(s, CP_SIZE)]) for s in range(NUM_SEGMENTS)],
        'owners': values['owner_segments_t'].tolist(),
        'final_segment': int(values['final_segment_t'].item()),
    }
    golden_prefill_cp_csa(values)
    return {n: values[n] for n in ('compress_state', 'inner_compress_state', 'kv_cache', 'cmp_kv', 'idx_kv_cache', 'idx_kv_scale', 'x_out')}
