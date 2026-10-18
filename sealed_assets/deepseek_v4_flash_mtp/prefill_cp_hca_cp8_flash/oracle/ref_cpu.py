from config import D, H, HC_MULT, IDX_TOPK, LEAF_CMP_BLOCKS, LOCAL_PARTS, MAX_COMPRESSED_ROWS_PER_SEGMENT, MAX_SEGMENT_TILES, ORI_CACHE_ROWS, ORI_MAX_BLOCKS, OVERLAY_BASE, OVERLAY_ROWS, PRED_OVERLAY_ROWS, Q_LORA
import torch

from config import BLOCK_SIZE, CASE_CONFIG as M, HCA_STATE_PHYSICAL_BLOCKS, PREFILL_CMP_BLOCK_NUM, PREFILL_CMP_MAX_BLOCKS, PREFILL_ORI_MAX_BLOCKS

from prefill_compressor_ratio128 import COMPRESS_RATIO, COMPRESS_STATE_DIM, HCA_CMP_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_STATE_MAX_BLOCKS, MAX_SEQ_LEN, golden_prefill_compressor_ratio128

from prefill_cp_zigzag import CP_SIZE, NUM_SEGMENTS, TAIL_ROWS, cp_owner_part, cp_owner_rank

from hc_post import golden_hc_post_prefill

from hc_pre import golden_hc_pre

from prefill_sparse_attn import HEAD_DIM, golden_prefill_sparse_attn

from qkv_proj_rope import golden_qkv_proj_rope

from rmsnorm import golden_rms_norm

def _state_physical_row(table, absolute_position: int) -> int:
    if absolute_position < 0 or absolute_position >= MAX_SEQ_LEN:
        return -1
    logical_block = absolute_position // HCA_STATE_BLOCK_SIZE
    physical_block = int(table[logical_block].item())
    if physical_block < 0:
        return -1
    return physical_block * HCA_STATE_BLOCK_SIZE + absolute_position % HCA_STATE_BLOCK_SIZE

def _cmp_physical_row(table, logical_slot: int) -> int:
    if logical_slot < 0:
        return -1
    logical_block = logical_slot // BLOCK_SIZE
    if logical_block >= table.numel():
        return -1
    physical_block = int(table[logical_block].item())
    if physical_block < 0:
        return -1
    return physical_block * BLOCK_SIZE + logical_slot % BLOCK_SIZE

def golden_prefill_cp_hca(tensors):
    """Compose CP-HCA golden outputs in logical-segment order."""
    import torch
    ctx = getattr(golden_prefill_cp_hca, '_ctx', None)
    if ctx is None:
        raise RuntimeError('CP-HCA golden context was not installed')
    cp_size = ctx['cp_size']
    starts = ctx['starts']
    lengths = ctx['lengths']
    owners = ctx['owners']
    local_q = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, H, HEAD_DIM, dtype=torch.bfloat16)
    local_kv = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    local_norm = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, D, dtype=torch.bfloat16)
    local_post = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT)
    local_comb = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT * HC_MULT)
    logical_hidden = torch.zeros(NUM_SEGMENTS, TAIL_ROWS, D, dtype=torch.bfloat16)
    logical_kv = torch.zeros(NUM_SEGMENTS, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            for tile in range(MAX_SEGMENT_TILES):
                active = int(tensors['overlay_active_lengths'][rank, part, tile, 1])
                x_tile = tensors['x_hc'][rank, part, tile]
                mixed = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                post = torch.zeros(TAIL_ROWS, HC_MULT)
                comb = torch.zeros(TAIL_ROWS, HC_MULT * HC_MULT)
                golden_hc_pre({'x': x_tile, 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': mixed, 'post': post, 'comb': comb})
                normed = golden_rms_norm(mixed, tensors['attn_norm_w'])
                positions = tensors['query_positions'][rank, part, tile]
                rope_positions = positions.clamp_min(0).to(torch.long)
                q = torch.zeros(TAIL_ROWS, H, HEAD_DIM, dtype=torch.bfloat16)
                kv = torch.zeros(TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                golden_qkv_proj_rope({'x': normed, 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': tensors['freqs_cos'].index_select(0, rope_positions), 'rope_sin': tensors['freqs_sin'].index_select(0, rope_positions), 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': torch.zeros(TAIL_ROWS, Q_LORA, dtype=torch.int8), 'qr_scale': torch.zeros(TAIL_ROWS, 1)})
                local_q[rank, part, tile] = q
                local_kv[rank, part, tile] = kv
                local_norm[rank, part, tile] = normed
                local_post[rank, part, tile] = post
                local_comb[rank, part, tile] = comb
            first = int(tensors['overlay_active_lengths'][rank, part, 0, 1])
            second = int(tensors['overlay_active_lengths'][rank, part, 1, 1])
            total = first + second
            if total:
                hidden_rows = torch.cat([local_norm[rank, part, 0, :first], local_norm[rank, part, 1, :second]], dim=0)
                kv_rows = torch.cat([local_kv[rank, part, 0, :first], local_kv[rank, part, 1, :second]], dim=0)
                valid = min(TAIL_ROWS, total)
                logical_hidden[segment, :valid] = hidden_rows[-valid:]
                logical_kv[segment, :valid] = kv_rows[-valid:]
    compressed_rows = []
    segment_snapshots = {}
    initial_state = tensors['compress_state'].clone()
    for segment in range(NUM_SEGMENTS):
        owner = cp_owner_rank(segment, cp_size)
        part = cp_owner_part(segment, cp_size)
        state_table = tensors['compress_state_block_table'][owner]
        scratch = torch.zeros_like(initial_state[owner])
        if segment == 0:
            scratch.copy_(initial_state[owner])
        leaves = []
        predecessor = segment - 1
        if predecessor >= 0:
            pred_valid = min(TAIL_ROWS, lengths[predecessor])
            pred_x = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
            pred_positions = torch.zeros(TAIL_ROWS, dtype=torch.int32)
            if pred_valid:
                pred_x[:pred_valid] = logical_hidden[predecessor, :pred_valid]
                pred_positions[:pred_valid] = tensors['segment_tail_positions'][predecessor, :pred_valid]
            leaves.append((pred_x, pred_positions, pred_valid, False))
        else:
            leaves.append((torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16), torch.zeros(TAIL_ROWS, dtype=torch.int32), 0, False))
        for tile in range(MAX_SEGMENT_TILES):
            active = int(tensors['overlay_active_lengths'][owner, part, tile, 1])
            leaves.append((local_norm[owner, part, tile], tensors['query_positions'][owner, part, tile], active, True))
        for (leaf, (leaf_x, positions, active, publish)) in enumerate(leaves):
            cmp_slots = torch.full((TAIL_ROWS,), -1, dtype=torch.int64)
            state_slots = torch.full_like(cmp_slots, -1)
            logical_slot = -1
            for row in range(active):
                position = int(positions[row])
                state_slots[row] = _state_physical_row(state_table, position)
                if publish and (position + 1) % COMPRESS_RATIO == 0:
                    cmp_slots[row] = 0
                    logical_slot = (position + 1) // COMPRESS_RATIO - 1
            leaf_cmp = torch.zeros(LEAF_CMP_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
            golden_prefill_compressor_ratio128({'x': leaf_x, 'compress_state': scratch, 'compress_state_block_table': state_table, 'wkv': tensors['cmp_wkv'], 'wgate': tensors['cmp_wgate'], 'ape': tensors['cmp_ape'], 'norm_w': tensors['cmp_norm_w'], 'freqs_cos': tensors['freqs_cos'], 'freqs_sin': tensors['freqs_sin'], 'cmp_kv': leaf_cmp, 'position_ids': positions, 'num_tokens': active, 'cmp_slot_mapping': cmp_slots, 'state_slot_mapping': state_slots})
            if logical_slot >= 0:
                compressed_rows.append((logical_slot, leaf_cmp.view(-1, HEAD_DIM)[0].clone()))
        valid = int(tensors['snapshot_valid'][owner, part])
        snapshot = torch.zeros(TAIL_ROWS, COMPRESS_STATE_DIM)
        scratch_flat = scratch.view(-1, COMPRESS_STATE_DIM)
        for row in range(valid):
            position = int(tensors['snapshot_positions'][owner, part, row])
            source = _state_physical_row(state_table, position)
            if source >= 0:
                snapshot[row] = scratch_flat[source]
        segment_snapshots[segment] = snapshot
    cmp_result = tensors['cmp_kv'].clone()
    for (logical_slot, value) in compressed_rows:
        for receiver in range(cp_size):
            destination = _cmp_physical_row(tensors['cmp_block_table'][receiver], logical_slot)
            if destination >= 0:
                cmp_result[receiver].view(-1, HEAD_DIM)[destination] = value
    state_result = initial_state.clone()
    final_segment = ctx['final_segment']
    final_snapshot = segment_snapshots[final_segment]
    final_owner = cp_owner_rank(final_segment, cp_size)
    final_part = cp_owner_part(final_segment, cp_size)
    final_valid = int(tensors['snapshot_valid'][final_owner, final_part])
    for receiver in range(cp_size):
        state_flat = state_result[receiver].view(-1, COMPRESS_STATE_DIM)
        table = tensors['compress_state_block_table'][receiver]
        for row in range(final_valid):
            position = int(tensors['snapshot_positions'][final_owner, final_part, row])
            destination = _state_physical_row(table, position)
            if destination >= 0:
                state_flat[destination] = final_snapshot[row]
    raw_initial = tensors['kv_cache'].clone()
    output = torch.zeros_like(tensors['x_out'])
    for rank in range(cp_size):
        persistent = raw_initial[rank].view(-1, HEAD_DIM)
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            for tile in range(MAX_SEGMENT_TILES):
                active = int(tensors['overlay_active_lengths'][rank, part, tile, 1])
                fake = torch.zeros(ORI_CACHE_ROWS + OVERLAY_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                fake[:ORI_CACHE_ROWS] = persistent
                predecessor = int(tensors['predecessor_segments'][rank, part])
                pred_valid = int(tensors['overlay_active_lengths'][rank, part, tile, 0])
                if pred_valid:
                    if tile == 0 and predecessor >= 0:
                        fake[OVERLAY_BASE:OVERLAY_BASE + pred_valid] = logical_kv[predecessor, :pred_valid]
                    elif tile > 0:
                        fake[OVERLAY_BASE:OVERLAY_BASE + pred_valid] = local_kv[rank, part, tile - 1, :pred_valid]
                fake[OVERLAY_BASE + PRED_OVERLAY_ROWS:OVERLAY_BASE + PRED_OVERLAY_ROWS + active] = local_kv[rank, part, tile, :active]
                fake_cache = fake.view(-1, BLOCK_SIZE, 1, HEAD_DIM)
                positions = tensors['query_positions'][rank, part, tile]
                rope_positions = positions.clamp_min(0).to(torch.long)
                attn = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                golden_prefill_sparse_attn({'q': local_q[rank, part, tile], 'ori_kv': fake_cache, 'swa_indices': tensors['swa_indices'][rank, part, tile], 'cmp_kv': cmp_result[rank], 'cmp_block_table': tensors['cmp_block_table'][rank], 'cmp_indices': tensors['cmp_indices'][rank, part, tile], 'attn_sink': tensors['attn_sink'], 'num_tokens': active, 'freqs_cos': tensors['freqs_cos'].index_select(0, rope_positions), 'freqs_sin': tensors['freqs_sin'].index_select(0, rope_positions), 'wo_a': tensors['wo_a'], 'wo_b': tensors['wo_b'], 'wo_b_scale': tensors['wo_b_scale'], 'attn_out': attn})
                y = torch.zeros(TAIL_ROWS, HC_MULT, D)
                golden_hc_post_prefill({'x': attn, 'residual': tensors['x_hc'][rank, part, tile], 'post': local_post[rank, part, tile], 'comb': local_comb[rank, part, tile], 'y': y, 'num_tokens': active})
                output[rank, part, tile] = y
    raw_result = raw_initial.clone().view(cp_size, -1, HEAD_DIM)
    for row in range(TAIL_ROWS):
        segment = int(tensors['final_win_seg_src'][row])
        source_row = int(tensors['final_win_row_src'][row])
        destination = int(tensors['final_slot_mapping'][row])
        if segment >= 0 and source_row >= 0 and (destination >= 0):
            raw_result[:, destination] = logical_kv[segment, source_row]
    tensors['compress_state'][:] = state_result
    tensors['cmp_kv'][:] = cmp_result
    tensors['kv_cache'][:] = raw_result.view_as(tensors['kv_cache'])
    tensors['x_out'][:] = output

def compute(inputs, scalars):
    if scalars:
        raise ValueError("CP8 HCA has no scalar inputs")
    tensors = {name: value.detach().cpu().clone() for name, value in inputs.items()}
    owners = tensors['owner_segments_t'].tolist()
    lengths = [0] * NUM_SEGMENTS
    for rank, segments in enumerate(owners):
        for part, segment in enumerate(segments):
            lengths[segment] = int(tensors['segment_active_lengths'][rank, part])
    golden_prefill_cp_hca._ctx = {
        'cp_size': CP_SIZE, 'prefix': int(tensors['segment_starts_t'][0]),
        'lengths': lengths, 'starts': tensors['segment_starts_t'].tolist(),
        'owners': owners, 'final_segment': int(tensors['final_segment_t'][0]),
    }
    tensors['x_out'] = torch.zeros_like(tensors['x_hc'])
    golden_prefill_cp_hca(tensors)
    return {name: tensors[name].detach().cpu().contiguous()
            for name in ('compress_state', 'kv_cache', 'cmp_kv', 'x_out')}
