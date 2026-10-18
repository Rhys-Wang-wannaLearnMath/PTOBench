TAIL_ROWS = 128

CP_CHOICES = (2, 4, 8)

CP_SIZE = 2

NUM_SEGMENTS = 2 * CP_SIZE

def cp_owner_rank(segment: int, cp_size: int=CP_SIZE) -> int:
    """Return the physical zigzag owner rank for a logical segment."""
    if segment < cp_size:
        return segment
    return 2 * cp_size - 1 - segment

def cp_owner_part(segment: int, cp_size: int=CP_SIZE) -> int:
    """Return the owner-local part for a logical segment."""
    return 0 if segment < cp_size else 1

def cp_reverse_index(cp_size: int=CP_SIZE):
    """Return rank-major payload indices in logical segment order."""
    import torch
    idx = torch.empty(2 * cp_size, dtype=torch.int64)
    for s in range(2 * cp_size):
        if s < cp_size:
            idx[s] = 2 * s
        else:
            idx[s] = 2 * (2 * cp_size - 1 - s) + 1
    return idx

def cp_final_window_sources(segment_lens):
    """Return final raw-window logical-segment and tail-row sources."""
    import torch
    total = sum(segment_lens)
    seg_src = torch.full((TAIL_ROWS,), -1, dtype=torch.int64)
    row_src = torch.full((TAIL_ROWS,), -1, dtype=torch.int64)
    cum = 0
    boundaries = [0]
    for length in segment_lens:
        cum += length
        boundaries.append(cum)
    for j in range(TAIL_ROWS):
        abs_pos = total - TAIL_ROWS + j
        if abs_pos < 0:
            continue
        for s in range(len(segment_lens)):
            if boundaries[s] <= abs_pos < boundaries[s + 1]:
                seg_src[j] = s
                seg_local = abs_pos - boundaries[s]
                tail_start = max(0, segment_lens[s] - TAIL_ROWS)
                row_src[j] = seg_local - tail_start
                break
    return (seg_src, row_src)

def cp_owner_tables(cp_size: int=CP_SIZE):
    """Build logical-segment owner-rank and owner-part tables."""
    import torch
    owner_rank = torch.tensor([cp_owner_rank(s, cp_size) for s in range(2 * cp_size)], dtype=torch.int32)
    owner_part = torch.tensor([cp_owner_part(s, cp_size) for s in range(2 * cp_size)], dtype=torch.int32)
    return (owner_rank, owner_part)
