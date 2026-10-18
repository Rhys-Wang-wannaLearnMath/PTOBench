TAIL_ROWS = 128

CP_SIZE = 4

NUM_SEGMENTS = 2 * CP_SIZE

def cp_owner_rank(segment: int, cp_size: int=CP_SIZE) -> int:
    """Return the physical zigzag owner rank for a logical segment."""
    if segment < cp_size:
        return segment
    return 2 * cp_size - 1 - segment

def cp_owner_part(segment: int, cp_size: int=CP_SIZE) -> int:
    """Return the owner-local part for a logical segment."""
    return 0 if segment < cp_size else 1
