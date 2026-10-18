import pypto.language as pl

import pypto.language.distributed as pld

from config import CASE_CONFIG as M, CP_SIZE, TAIL_ROWS

HEAD_DIM = M.head_dim

ROW_TILE = 8

NUM_SEGMENTS = 2 * CP_SIZE

CP_TAIL_WINDOW_ROWS = NUM_SEGMENTS * TAIL_ROWS

@pl.jit.inline
def _prefill_cp_zigzag_kv_tail_exchange_wave(local_kv_tail: pl.Tensor[[2 * TAIL_ROWS, HEAD_DIM], pl.BF16], reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32], owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32], kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16], ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32], logical_tails_out: pl.Out[pl.Tensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]], my_rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32]) -> pl.Tensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]:
    """Exchange one local tail wave into logical segment order."""
    epoch_value = pl.cast(epoch + 1, pl.INT32)
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(signal=consumed, offsets=[peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    for peer in pl.range(CP_SIZE):
        for part in pl.range(2):
            rank_major_pos = my_rank * 2 + part
            dst_row_base = rank_major_pos * TAIL_ROWS
            src_row_base = part * TAIL_ROWS
            pld.tensor.put(dst=kv_tail_window, peer=peer, src=local_kv_tail, dst_offsets=[dst_row_base, 0], src_offsets=[src_row_base, 0], shape=[TAIL_ROWS, HEAD_DIM], chunk_rows=ROW_TILE, chunk_cols=HEAD_DIM, pipeline=True)
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(target=ready, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    for seg in pl.range(NUM_SEGMENTS):
        rm_pos = reverse_index[seg]
        owner = owner_rank_table[seg]
        if owner != my_rank:
            pld.system.wait(signal=ready, offsets=[owner, 0], expected=epoch_value, cmp=pld.WaitCmp.Ge)
        rm_row_base = rm_pos * TAIL_ROWS
        out_row_base = seg * TAIL_ROWS
        for t0 in pl.range(0, TAIL_ROWS, ROW_TILE):
            win_tile = pl.load(kv_tail_window, [rm_row_base + t0, 0], [ROW_TILE, HEAD_DIM])
            pl.store(win_tile, [out_row_base + t0, 0], logical_tails_out)
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(target=consumed, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    return logical_tails_out
