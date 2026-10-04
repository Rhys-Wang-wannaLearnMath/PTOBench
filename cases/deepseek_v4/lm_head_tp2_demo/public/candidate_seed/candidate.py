import pypto.language as pl
import pypto.language.distributed as pld

from config import CASE_CONFIG as M, DECODE_TOKENS, LM_HEAD_TP_SIZE, TP_SIZE


T = DECODE_TOKENS
D = M.hidden_size
VOCAB = M.vocab_size
LM_HEAD_K_CHUNK = 128
VOCAB_CHUNK = 160
T_TILE = 16
T_PAD = ((T + T_TILE - 1) // T_TILE) * T_TILE
K_BLOCKS = D // LM_HEAD_K_CHUNK
VOCAB_PER_TP = VOCAB // TP_SIZE
VOCAB_BLOCKS_PER_TP = VOCAB_PER_TP // VOCAB_CHUNK

assert D % LM_HEAD_K_CHUNK == 0
assert VOCAB % VOCAB_CHUNK == 0
assert VOCAB % TP_SIZE == 0
assert TP_SIZE <= LM_HEAD_TP_SIZE
assert VOCAB_PER_TP % VOCAB_CHUNK == 0


@pl.jit.inline
def lm_head(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits_shard: pl.Out[pl.Tensor[[T_PAD, VOCAB_PER_TP], pl.FP32]],
) -> pl.Tensor[[T_PAD, VOCAB_PER_TP], pl.FP32]:
    for t0 in pl.parallel(0, T_PAD, T_TILE):
        valid_t = pl.min(T_TILE, T - t0)
        for ob in pl.parallel(VOCAB_BLOCKS_PER_TP):
            o0 = ob * VOCAB_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head"):
                hidden_chunk = pl.slice(
                    hidden_states,
                    [T_TILE, LM_HEAD_K_CHUNK],
                    [t0, 0],
                    valid_shape=[valid_t, LM_HEAD_K_CHUNK],
                )
                weight_chunk = pl.slice(
                    lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [o0, 0]
                )
                acc = pl.matmul(hidden_chunk, weight_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, K_BLOCKS):
                    k0 = kb * LM_HEAD_K_CHUNK
                    hidden_chunk = pl.slice(
                        hidden_states,
                        [T_TILE, LM_HEAD_K_CHUNK],
                        [t0, k0],
                        valid_shape=[valid_t, LM_HEAD_K_CHUNK],
                    )
                    weight_chunk = pl.slice(
                        lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [o0, k0]
                    )
                    acc = pl.matmul_acc(acc, hidden_chunk, weight_chunk, b_trans=True)
                logits_shard = pl.assemble(logits_shard, acc, [t0, o0])
    return logits_shard


@pl.jit.incore
def gather_step(
    logits_shard: pl.Tensor[[T_PAD, VOCAB_PER_TP], pl.FP32],
    logits: pl.Out[pl.Tensor[[T, VOCAB], pl.FP32]],
    logits_window: pld.DistributedTensor[[T, VOCAB], pl.FP32],
    gather_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, VOCAB], pl.FP32]:
    vocab_base = my_rank * VOCAB_PER_TP
    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            for t0 in pl.range(0, T_PAD, T_TILE):
                valid_t = pl.min(T_TILE, T - t0)
                for ob in pl.range(VOCAB_BLOCKS_PER_TP):
                    o0 = ob * VOCAB_CHUNK
                    tile = pl.load(
                        logits_shard,
                        [t0, o0],
                        [T_TILE, VOCAB_CHUNK],
                        valid_shape=[valid_t, VOCAB_CHUNK],
                    )
                    pld.tile.remote_store(
                        tile,
                        target=logits_window,
                        peer=peer,
                        offsets=[t0, vocab_base + o0],
                    )

    for peer in pl.range(TP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=gather_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.Set,
            )

    for src in pl.range(TP_SIZE):
        if src != my_rank:
            pld.system.wait(
                signal=gather_done,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    for t0 in pl.range(0, T_PAD, T_TILE):
        valid_t = pl.min(T_TILE, T - t0)
        for src in pl.range(TP_SIZE):
            src_vocab_base = src * VOCAB_PER_TP
            for ob in pl.range(VOCAB_BLOCKS_PER_TP):
                o0 = ob * VOCAB_CHUNK
                if src == my_rank:
                    tile = pl.load(
                        logits_shard,
                        [t0, o0],
                        [T_TILE, VOCAB_CHUNK],
                        valid_shape=[valid_t, VOCAB_CHUNK],
                    )
                    pl.store(tile, [t0, src_vocab_base + o0], logits)
                else:
                    tile = pl.load(
                        logits_window,
                        [t0, src_vocab_base + o0],
                        [T_TILE, VOCAB_CHUNK],
                        valid_shape=[valid_t, VOCAB_CHUNK],
                    )
                    pl.store(tile, [t0, src_vocab_base + o0], logits)
    return logits


@pl.jit
def lm_head_tp(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[T, VOCAB], pl.FP32]],
    logits_window: pld.DistributedTensor[[T, VOCAB], pl.FP32],
    gather_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, VOCAB], pl.FP32]:
    logits_shard = pl.create_tensor([T_PAD, VOCAB_PER_TP], dtype=pl.FP32)
    logits_shard = lm_head(hidden_states, lm_head_weight, logits_shard)
    logits = gather_step(logits_shard, logits, logits_window, gather_done, my_rank)
    return logits


@pl.jit.host
def host_orch(
    hidden_states: pl.Tensor[[TP_SIZE, T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[TP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[TP_SIZE, T, VOCAB], pl.FP32]],
):
    logits_window_buf = pld.alloc_window_buffer(T * VOCAB * 4)
    gather_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        logits_window = pld.window(logits_window_buf, [T, VOCAB], dtype=pl.FP32)
        gather_done = pld.window(gather_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        lm_head_tp(
            hidden_states[r],
            lm_head_weight[r],
            logits[r],
            logits_window,
            gather_done,
            r,
            device=r,
        )
