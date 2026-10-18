import torch

from config import BLOCK_SIZE, CASE_CONFIG as M, PREFILL_ORI_MAX_BLOCKS

from prefill_cp_zigzag import HEAD_DIM, cp_final_window_sources, cp_owner_part, cp_owner_rank, cp_reverse_index

from hc_post import golden_hc_post_prefill

from hc_pre import golden_hc_pre

from qkv_proj_rope import golden_qkv_proj_rope

from rmsnorm import golden_rms_norm

from prefill_sparse_attn import golden_prefill_sparse_attn

D = M.hidden_size

H = M.num_attention_heads

HC_MULT = M.hc_mult

Q_LORA = M.q_lora_rank

CP_SIZE = 4

WIN = M.sliding_window

TAIL_ROWS = WIN

BLOCK_ROWS = BLOCK_SIZE

MAX_SEGMENT_TILES = 2

LOCAL_PARTS = 2

ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS

ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_ROWS

OVERLAY_BASE = ORI_CACHE_ROWS

PRED_OVERLAY_ROWS = TAIL_ROWS

CUR_OVERLAY_ROWS = TAIL_ROWS

OVERLAY_ROWS = PRED_OVERLAY_ROWS + CUR_OVERLAY_ROWS

NUM_RING_BLOCKS = ORI_CACHE_ROWS // BLOCK_ROWS

IDENTITY_BLOCK_TABLE = torch.arange(NUM_RING_BLOCKS, dtype=torch.int32)

def ring_phys_row(abs_pos: int) -> int:
    ring_row = abs_pos % ORI_CACHE_ROWS
    block = ring_row // BLOCK_ROWS
    intra = ring_row % BLOCK_ROWS
    return int(IDENTITY_BLOCK_TABLE[block].item()) * BLOCK_ROWS + intra

def owner_segments(cp_size: int):
    """Return each rank's two logical segments."""
    table = [[-1, -1] for _ in range(cp_size)]
    for seg in range(2 * cp_size):
        rank = cp_owner_rank(seg, cp_size)
        part = cp_owner_part(seg, cp_size)
        table[rank][part] = seg
    return table

def pred_segment(segment: int) -> int:
    return segment - 1 if segment > 0 else -1

def active_tile(segment_len: int, tile: int) -> int:
    if tile == 0:
        return min(segment_len, TAIL_ROWS)
    return max(0, min(segment_len, 2 * TAIL_ROWS) - TAIL_ROWS)

def tail_start(seg_start: int, seg_len: int) -> int:
    return seg_start + max(0, seg_len - TAIL_ROWS)

def segment_starts(prefix: int, segment_span: int, nseg: int):
    return [prefix + s * segment_span for s in range(nseg)]

def lower_key_build(key_abs, segment, tile, starts, lengths, prefix):
    """Lower an absolute key position into the persistent or overlay row."""
    seg_start = starts[segment]
    seg_len = lengths[segment]
    cur_start = seg_start + tile * TAIL_ROWS
    cur_len = active_tile(seg_len, tile)
    if key_abs < prefix:
        return ring_phys_row(key_abs)
    if cur_start <= key_abs < cur_start + cur_len:
        return OVERLAY_BASE + CUR_OVERLAY_ROWS + (key_abs - cur_start)
    if key_abs < cur_start:
        if tile == 0:
            pred_seg = pred_segment(segment)
            if pred_seg < 0:
                return -1
            pred_len = min(TAIL_ROWS, lengths[pred_seg])
            pred_tail = tail_start(starts[pred_seg], lengths[pred_seg])
            if pred_tail <= key_abs < pred_tail + pred_len:
                return OVERLAY_BASE + (key_abs - pred_tail)
            return -1
        prev_start = cur_start - TAIL_ROWS
        prev_len = active_tile(seg_len, 0)
        if prev_start <= key_abs < prev_start + prev_len:
            return OVERLAY_BASE + (key_abs - prev_start)
        return -1
    return -1

def build_metadata(cp_size: int=CP_SIZE):
    """Build the canonical zero-history CP metadata."""
    prefix = 0
    segment_span = TAIL_ROWS
    lengths = [TAIL_ROWS] * (2 * cp_size)
    nseg = 2 * cp_size
    starts = segment_starts(prefix, segment_span, nseg)
    parts = owner_segments(cp_size)
    seg_starts_t = torch.tensor(starts, dtype=torch.int32)
    seg_lens_t = torch.tensor(lengths, dtype=torch.int32)
    owner_segs_t = torch.tensor(parts, dtype=torch.int32)
    reverse_index_t = cp_reverse_index(cp_size).to(torch.int32)
    seg_active_lengths = torch.full((cp_size, LOCAL_PARTS), -1, dtype=torch.int32)
    predecessor_segments = torch.full((cp_size, LOCAL_PARTS), -1, dtype=torch.int32)
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            seg = parts[rank][part]
            seg_active_lengths[rank, part] = lengths[seg]
            predecessor_segments[rank, part] = pred_segment(seg)
    q_pos = torch.zeros((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS), dtype=torch.int32)
    q_req = torch.full((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS), -1, dtype=torch.int32)
    ov_pos = torch.full((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS), -1, dtype=torch.int32)
    ov_req = torch.full_like(ov_pos, -1)
    ov_len = torch.full((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, 2), -1, dtype=torch.int32)
    swa = torch.full((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN), -1, dtype=torch.int32)
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = parts[rank][part]
            seg_len = lengths[segment]
            for tile in range(MAX_SEGMENT_TILES):
                active = active_tile(seg_len, tile)
                tile_start = starts[segment] + tile * TAIL_ROWS
                if active > 0:
                    q_pos[rank, part, tile, :active] = torch.arange(tile_start, tile_start + active, dtype=torch.int32)
                    q_req[rank, part, tile, :active] = 0
                if tile == 0:
                    pred_seg = pred_segment(segment)
                    if pred_seg >= 0:
                        pred_len = min(TAIL_ROWS, lengths[pred_seg])
                        pred_start = tail_start(starts[pred_seg], lengths[pred_seg])
                    else:
                        pred_len = 0
                        pred_start = 0
                else:
                    pred_len = active_tile(seg_len, 0)
                    pred_start = starts[segment] + (tile - 1) * TAIL_ROWS
                if pred_len > 0:
                    ov_pos[rank, part, tile, :pred_len] = torch.arange(pred_start, pred_start + pred_len, dtype=torch.int32)
                    ov_req[rank, part, tile, :pred_len] = 0
                if active > 0:
                    ov_pos[rank, part, tile, PRED_OVERLAY_ROWS:PRED_OVERLAY_ROWS + active] = torch.arange(tile_start, tile_start + active, dtype=torch.int32)
                    ov_req[rank, part, tile, PRED_OVERLAY_ROWS:PRED_OVERLAY_ROWS + active] = 0
                ov_len[rank, part, tile, 0] = pred_len
                ov_len[rank, part, tile, 1] = active
                for query_row in range(active):
                    query_abs = tile_start + query_row
                    for col in range(WIN):
                        key_abs = query_abs - WIN + 1 + col
                        if key_abs < 0 or key_abs > query_abs:
                            continue
                        raw = lower_key_build(key_abs, segment, tile, starts, lengths, prefix)
                        swa[rank, part, tile, query_row, col] = raw
    (final_seg_src, final_row_src) = cp_final_window_sources(lengths)
    final_seg_src = final_seg_src.to(torch.int32)
    final_row_src = final_row_src.to(torch.int32)
    total = sum(lengths)
    final_slot = torch.full((TAIL_ROWS,), -1, dtype=torch.int32)
    for row in range(TAIL_ROWS):
        abs_pos = prefix + total - TAIL_ROWS + row
        if abs_pos < prefix:
            continue
        final_slot[row] = ring_phys_row(abs_pos)
    tensors = {'segment_lens': seg_lens_t, 'segment_starts': seg_starts_t, 'owner_segments': owner_segs_t, 'reverse_index': reverse_index_t, 'segment_active_lengths': seg_active_lengths, 'predecessor_segments': predecessor_segments, 'query_position_ids': q_pos, 'query_token_to_request': q_req, 'overlay_position_ids': ov_pos, 'overlay_token_to_request': ov_req, 'overlay_active_lengths': ov_len, 'swa_indices': swa, 'final_win_seg_src': final_seg_src, 'final_win_row_src': final_row_src, 'final_slot_mapping': final_slot}
    ctx = {'cp_size': cp_size, 'prefix': prefix, 'segment_span': segment_span, 'lengths': lengths, 'starts': starts, 'owner_segments': parts, 'block_table': IDENTITY_BLOCK_TABLE}
    return (tensors, ctx)

def golden_prefill_cp_swa(tensors):
    """Compose CP-SWA golden outputs in logical-segment order."""
    import torch
    cp = tensors['x_hc'].shape[0]
    metadata_names = ('predecessor_segments', 'query_position_ids', 'query_token_to_request', 'overlay_position_ids', 'overlay_token_to_request', 'overlay_active_lengths', 'swa_indices')
    meta = {name: tensors[name] for name in metadata_names}
    ctx_case = getattr(golden_prefill_cp_swa, '_ctx', None)
    if ctx_case is None:
        raise RuntimeError('CP-SWA golden context was not installed by the fixture')
    lengths = ctx_case['lengths']
    starts = ctx_case['starts']
    parts = ctx_case['owner_segments']
    prefix = ctx_case['prefix']
    initial_cache = tensors['kv_cache'].clone()
    local_kvs = torch.zeros(cp, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    local_q = torch.zeros(cp, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, H, HEAD_DIM, dtype=torch.bfloat16)
    local_post = torch.zeros(cp, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT)
    local_comb = torch.zeros(cp, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT * HC_MULT)
    logical = torch.zeros(2 * cp, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    for rank in range(cp):
        for part in range(LOCAL_PARTS):
            seg = parts[rank][part]
            for tile in range(MAX_SEGMENT_TILES):
                active = int(meta['overlay_active_lengths'][rank, part, tile, 1])
                x_tile = tensors['x_hc'][rank, part, tile]
                xm = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                post = torch.zeros(TAIL_ROWS, HC_MULT)
                comb = torch.zeros(TAIL_ROWS, HC_MULT * HC_MULT)
                golden_hc_pre({'x': x_tile, 'hc_fn': tensors['hc_attn_fn'], 'hc_scale': tensors['hc_attn_scale'], 'hc_base': tensors['hc_attn_base'], 'x_mixed': xm, 'post': post, 'comb': comb})
                positions = meta['query_position_ids'][rank, part, tile].long()
                norm = golden_rms_norm(xm, tensors['attn_norm_w'])
                q = torch.zeros(TAIL_ROWS, H, HEAD_DIM, dtype=torch.bfloat16)
                kv = torch.zeros(TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                qr = torch.zeros(TAIL_ROWS, Q_LORA, dtype=torch.int8)
                qrs = torch.zeros(TAIL_ROWS, 1)
                golden_qkv_proj_rope({'x': norm, 'wq_a': tensors['wq_a'], 'wq_b': tensors['wq_b'], 'wq_b_scale': tensors['wq_b_scale'], 'wkv': tensors['wkv'], 'rope_cos': tensors['freqs_cos'].index_select(0, positions), 'rope_sin': tensors['freqs_sin'].index_select(0, positions), 'gamma_cq': tensors['gamma_cq'], 'gamma_ckv': tensors['gamma_ckv'], 'q': q, 'kv': kv, 'qr': qr, 'qr_scale': qrs})
                local_q[rank, part, tile] = q
                local_kvs[rank, part, tile] = kv
                local_post[rank, part, tile] = post
                local_comb[rank, part, tile] = comb
    for rank in range(cp):
        for part in range(LOCAL_PARTS):
            seg = parts[rank][part]
            tile0_len = int(meta['overlay_active_lengths'][rank, part, 0, 1])
            tile1_len = int(meta['overlay_active_lengths'][rank, part, 1, 1])
            total = tile0_len + tile1_len
            tail_start = max(0, total - TAIL_ROWS)
            for row in range(TAIL_ROWS):
                tail_offset = tail_start + row
                if tail_offset >= total:
                    continue
                if tail_offset < TAIL_ROWS:
                    logical[seg, row] = local_kvs[rank, part, 0, tail_offset]
                else:
                    logical[seg, row] = local_kvs[rank, part, 1, tail_offset - TAIL_ROWS]
    out = torch.zeros_like(tensors['x_out'])
    for rank in range(cp):
        cache_flat = initial_cache[rank].reshape(-1, HEAD_DIM)
        for part in range(LOCAL_PARTS):
            for tile in range(MAX_SEGMENT_TILES):
                active = int(meta['overlay_active_lengths'][rank, part, tile, 1])
                fake = torch.zeros(ORI_CACHE_ROWS + OVERLAY_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                fake[:ORI_CACHE_ROWS] = cache_flat
                seg = parts[rank][part]
                pred = int(meta['predecessor_segments'][rank, part])
                if tile == 0 and pred >= 0:
                    fake[OVERLAY_BASE:OVERLAY_BASE + TAIL_ROWS] = logical[pred]
                elif tile == 1:
                    fake[OVERLAY_BASE:OVERLAY_BASE + TAIL_ROWS] = local_kvs[rank, part, 0]
                fake[OVERLAY_BASE + TAIL_ROWS:OVERLAY_BASE + OVERLAY_ROWS] = local_kvs[rank, part, tile]
                fake_cache = fake.view((ORI_CACHE_ROWS + OVERLAY_ROWS) // BLOCK_ROWS, BLOCK_ROWS, 1, HEAD_DIM)
                attn = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                positions = meta['query_position_ids'][rank, part, tile].long()
                golden_prefill_sparse_attn({'q': local_q[rank, part, tile], 'ori_kv': fake_cache, 'swa_indices': meta['swa_indices'][rank, part, tile], 'cmp_kv': torch.zeros(1, BLOCK_ROWS, 1, HEAD_DIM, dtype=torch.bfloat16), 'cmp_block_table': torch.zeros(1, dtype=torch.int32), 'cmp_indices': torch.full((TAIL_ROWS, 1), -1, dtype=torch.int32), 'attn_sink': tensors['attn_sink'], 'num_tokens': active, 'freqs_cos': tensors['freqs_cos'].index_select(0, positions), 'freqs_sin': tensors['freqs_sin'].index_select(0, positions), 'wo_a': tensors['wo_a'], 'wo_b': tensors['wo_b'], 'wo_b_scale': tensors['wo_b_scale'], 'attn_out': attn})
                y = torch.zeros(TAIL_ROWS, HC_MULT, D)
                golden_hc_post_prefill({'x': attn, 'residual': tensors['x_hc'][rank, part, tile], 'post': local_post[rank, part, tile], 'comb': local_comb[rank, part, tile], 'y': y, 'num_tokens': active})
                out[rank, part, tile] = y
    tensors['x_out'][:] = out
    final_cache = initial_cache.clone().reshape(cp, -1, HEAD_DIM)
    for row in range(TAIL_ROWS):
        seg = int(tensors['final_win_seg_src'][row])
        src_row = int(tensors['final_win_row_src'][row])
        dst = int(tensors['final_slot_mapping'][row])
        if seg >= 0 and src_row >= 0 and (dst >= 0):
            final_cache[:, dst] = logical[seg, src_row]
    tensors['kv_cache'][:] = final_cache.reshape_as(tensors['kv_cache'])


def compute(inputs, scalars=None):
    del scalars
    values = {name: value.detach().clone() for name, value in inputs.items()}
    values['x_out'] = torch.zeros_like(values['x_hc'])
    golden_prefill_cp_swa._ctx = build_metadata(4)[1]
    golden_prefill_cp_swa(values)
    return {'x_out': values['x_out'].detach().cpu().contiguous(),
            'kv_cache': values['kv_cache'].detach().cpu().contiguous()}
