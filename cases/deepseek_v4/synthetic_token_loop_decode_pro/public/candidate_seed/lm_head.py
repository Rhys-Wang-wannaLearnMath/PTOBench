import pypto.language as pl
import pypto.language.distributed as pld
from config import CASE_CONFIG as M, DECODE_TOKENS
T_DYN = pl.dynamic('LM_HEAD_T_DYN')
from config import D
from config import VOCAB
from config import LM_HEAD_TP_SIZE as TP_SIZE
from config import VOCAB_PER_TP
from config import MAX_LOGIT_ROWS
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS
FUSED_K_TILE = 256
FUSED_VOCAB_TILE = 128
HIDDEN_GATHER_TILE = 512
LOGITS_COMM_TILE = 2048
VOCAB_TAIL = VOCAB_PER_TP % FUSED_VOCAB_TILE
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
FUSED_LM_HEAD_CORES = 24
GREEDY_VOCAB_CHUNK = 256
GREEDY_NUM_VOCAB_CHUNKS = VOCAB // GREEDY_VOCAB_CHUNK
GREEDY_CHUNK_PAD = 512
GREEDY_TOPK = 16
from config import SAMPLED_IDS_PAD
N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE
LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))
LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS

@pl.jit.inline(auto_scope=False)
def lm_head(hidden_states: pl.Tensor, lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16], logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32], logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32], hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16], hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32], logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], done_epoch: pl.Scalar[pl.INT32]) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)
    owner_hiddens = pl.create_tensor([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint='lm_head_dispatch_push'):
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        source_row_raw = pl.read(logit_row_indices, [row])
        safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
        selected_hidden[row:row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
        if source_row_raw >= 0:
            source_row = pl.cast(safe_raw, target_type=pl.INDEX)
            selected_hidden[row:row + 1, :] = hidden_states[source_row:source_row + 1, :]
        for peer_tp in pl.range(TP_SIZE):
            pld.tensor.put(dst=hidden_window, peer=group_base + peer_tp, src=selected_hidden, dst_offsets=[tp_rank * MAX_LOGIT_ROWS + row, 0], src_offsets=[row, 0], shape=[1, D])
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(target=hidden_done, peer=group_base + peer_tp, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='lm_head_dispatch_wait') as _dwait_tid:
        _hidden_anchor = pl.read(hidden_states, [0, 0])
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.wait(signal=hidden_done, offsets=[owner_tp, 0], expected=pl.cast(done_epoch * MAX_LOGIT_ROWS, pl.INT32), cmp=pld.WaitCmp.Ge)
    with pl.spmd(D // HIDDEN_GATHER_TILE, name_hint='lm_head_dispatch_gather', deps=[_dwait_tid]) as _dgather_tid:
        gkb = pl.tile.get_block_idx()
        gk0 = gkb * HIDDEN_GATHER_TILE
        owner_hiddens[:, gk0:gk0 + HIDDEN_GATHER_TILE] = hidden_window[:, gk0:gk0 + HIDDEN_GATHER_TILE]
    logits_shards = pl.create_tensor([GROUP_LOGIT_ROWS, VOCAB_PER_TP], dtype=pl.FP32)
    for lm_core in pl.spmd(FUSED_LM_HEAD_CORES, name_hint='lm_head_matmul'):
        for mm_ob in pl.range(lm_core, VOCAB_PER_TP // FUSED_VOCAB_TILE, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            mm_hidden0 = owner_hiddens[:, 0:FUSED_K_TILE]
            mm_weight0 = lm_head_weight[mm_o0:mm_o0 + FUSED_VOCAB_TILE, 0:FUSED_K_TILE]
            mm_acc = pl.matmul(mm_hidden0, mm_weight0, b_trans=True, out_dtype=pl.FP32)
            for mm_kb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                mm_k0 = mm_kb * FUSED_K_TILE
                mm_hidden_tile = owner_hiddens[:, mm_k0:mm_k0 + FUSED_K_TILE]
                mm_weight_tile = lm_head_weight[mm_o0:mm_o0 + FUSED_VOCAB_TILE, mm_k0:mm_k0 + FUSED_K_TILE]
                mm_acc = pl.matmul_acc(mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True)
            logits_shards[:, mm_o0:mm_o0 + FUSED_VOCAB_TILE] = mm_acc
        if VOCAB_TAIL != 0:
            if lm_core == VOCAB_PER_TP // FUSED_VOCAB_TILE % FUSED_LM_HEAD_CORES:
                mm_tail_o0 = VOCAB_PER_TP // FUSED_VOCAB_TILE * FUSED_VOCAB_TILE
                mm_hidden_t0 = owner_hiddens[:, 0:FUSED_K_TILE]
                mm_weight_t0 = lm_head_weight[mm_tail_o0:mm_tail_o0 + VOCAB_TAIL, 0:FUSED_K_TILE]
                mm_acc_tail = pl.matmul(mm_hidden_t0, mm_weight_t0, b_trans=True, out_dtype=pl.FP32)
                for mm_tail_kb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                    mm_tail_k0 = mm_tail_kb * FUSED_K_TILE
                    mm_hidden_tk = owner_hiddens[:, mm_tail_k0:mm_tail_k0 + FUSED_K_TILE]
                    mm_weight_tk = lm_head_weight[mm_tail_o0:mm_tail_o0 + VOCAB_TAIL, mm_tail_k0:mm_tail_k0 + FUSED_K_TILE]
                    mm_acc_tail = pl.matmul_acc(mm_acc_tail, mm_hidden_tk, mm_weight_tk, b_trans=True)
                logits_shards[:, mm_tail_o0:mm_tail_o0 + VOCAB_TAIL] = mm_acc_tail
    for blk in pl.spmd(LOGITS_COMM_BLOCKS, name_hint='lm_head_combine_push'):
        vocab_base = tp_rank * VOCAB_PER_TP
        for owner_tp in pl.range(TP_SIZE):
            source_row_base = owner_tp * MAX_LOGIT_ROWS
            for ob in pl.range(blk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                pld.tensor.put(dst=logits_window, peer=group_base + owner_tp, src=logits_shards, dst_offsets=[0, vocab_base + o0], src_offsets=[source_row_base, o0], shape=[MAX_LOGIT_ROWS, LOGITS_COMM_TILE])
            if LOGITS_COMM_TAIL != 0:
                if blk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    pld.tensor.put(dst=logits_window, peer=group_base + owner_tp, src=logits_shards, dst_offsets=[0, vocab_base + tail_o0], src_offsets=[source_row_base, tail_o0], shape=[MAX_LOGIT_ROWS, LOGITS_COMM_TAIL])
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.notify(target=logits_done, peer=group_base + owner_tp, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='lm_head_combine_wait') as _cwait_tid:
        _shard_anchor = pl.read(logits_shards, [0, 0])
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(signal=logits_done, offsets=[src_tp, 0], expected=pl.cast(done_epoch * LOGITS_COMM_BLOCKS, pl.INT32), cmp=pld.WaitCmp.Ge)
    with pl.spmd(LOGITS_COMM_BLOCKS, name_hint='lm_head_combine_gather', deps=[_cwait_tid]) as _gather_tid:
        gblk = pl.tile.get_block_idx()
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                logits[:, lo:lo + LOGITS_COMM_TILE] = logits_window[:, lo:lo + LOGITS_COMM_TILE]
            if LOGITS_COMM_TAIL != 0:
                if gblk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    tl = src_vocab_base + tail_o0
                    logits[:, tl:tl + LOGITS_COMM_TAIL] = logits_window[:, tl:tl + LOGITS_COMM_TAIL]
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='lm_head_signal_clear'):
        _completion_anchor = pl.read(logits, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return logits

@pl.jit.inline
def greedy_sample(logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32], sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]):
    """Select the first maximum token id from each full-vocabulary logits row."""
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint='lm_head_greedy_sample'):
        chunk_idx_init = pl.arange(0, [1, GREEDY_VOCAB_CHUNK], dtype=pl.UINT32)
        chunk_maxima = pl.create_tensor([1, GREEDY_CHUNK_PAD], dtype=pl.FP32)
        chunk_maxima[:, :] = pl.full([1, GREEDY_CHUNK_PAD], dtype=pl.FP32, value=-3.402823e+38)
        for chunk in pl.range(GREEDY_NUM_VOCAB_CHUNKS):
            chunk_start = chunk * GREEDY_VOCAB_CHUNK
            scores = logits[row:row + 1, chunk_start:chunk_start + GREEDY_VOCAB_CHUNK]
            sorted_pairs = pl.sort32(scores, chunk_idx_init)
            sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
            sorted_pairs = pl.mrgsort(sorted_pairs[:, 0:GREEDY_VOCAB_CHUNK], sorted_pairs[:, GREEDY_VOCAB_CHUNK:2 * GREEDY_VOCAB_CHUNK])
            top_pair = sorted_pairs[:, 0:2 * GREEDY_TOPK]
            top_values = pl.gather(top_pair, mask_pattern=pl.tile.MaskPattern.P0101)
            pl.write(chunk_maxima, [0, chunk], pl.read(top_values, [0, 0]))
        maxima_idx_init = pl.arange(0, [1, GREEDY_CHUNK_PAD], dtype=pl.UINT32)
        sorted_maxima = pl.sort32(chunk_maxima, maxima_idx_init)
        sorted_maxima = pl.mrgsort(sorted_maxima, block_len=64)
        sorted_maxima = pl.mrgsort(sorted_maxima, block_len=256)
        top_maximum_pair = sorted_maxima[:, 0:2 * GREEDY_TOPK]
        top_maximum_values = pl.gather(top_maximum_pair, mask_pattern=pl.tile.MaskPattern.P0101)
        best_value = pl.read(top_maximum_values, [0, 0])
        winning_chunk = pl.cast(0, pl.INT32)
        for chunk in pl.range(GREEDY_NUM_VOCAB_CHUNKS):
            scan_chunk = GREEDY_NUM_VOCAB_CHUNKS - 1 - chunk
            if pl.read(chunk_maxima, [0, scan_chunk]) == best_value:
                winning_chunk = pl.cast(scan_chunk, pl.INT32)
        chunk_base = winning_chunk * pl.cast(GREEDY_VOCAB_CHUNK, pl.INT32)
        winning_scores = pl.slice(logits, [1, GREEDY_VOCAB_CHUNK], [pl.cast(row, pl.INDEX), pl.cast(chunk_base, pl.INDEX)])
        winning_offset = pl.cast(0, pl.INT32)
        for offset in pl.range(GREEDY_VOCAB_CHUNK):
            scan_offset = GREEDY_VOCAB_CHUNK - 1 - offset
            if pl.read(winning_scores, [0, scan_offset]) == best_value:
                winning_offset = pl.cast(scan_offset, pl.INT32)
        sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
        sampled_row[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
        pl.write(sampled_row, [0, 0], chunk_base + winning_offset)
        sampled_ids[row:row + 1, :] = sampled_row
    return sampled_ids

@pl.jit.inline(auto_scope=False)
def lm_head_with_sampling(hidden_states: pl.Tensor[[T_DYN, D], pl.BF16], lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16], logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32], logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]], sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]], hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16], hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32], logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], done_epoch: pl.Scalar[pl.INT32]):
    """Project logits and sample top-1 tokens in one opaque L2 entry."""
    lm_head(hidden_states, lm_head_weight, logit_row_indices, logits, hidden_window, hidden_done, logits_window, logits_done, group_base, tp_rank, done_epoch)
    greedy_sample(logits, sampled_ids)
    return (logits, sampled_ids)
