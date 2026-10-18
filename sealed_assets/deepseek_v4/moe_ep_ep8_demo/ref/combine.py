import pypto.language as pl

import pypto.language.distributed as pld

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

D = M.hidden_size

TOPK = M.num_experts_per_tok

N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

@pl.jit.inline
def combine_ep(recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16], recv_r_route_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32], sh: pl.Tensor[[T, D], pl.BF16], ffn_out: pl.Tensor[[T, D], pl.BF16], pub_counts: pld.DistributedTensor[[EP_WORLD_SIZE * EP_WORLD_SIZE, N_LOCAL_EXPERTS], pl.INT32], routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16], combine_done: pld.DistributedTensor[[EP_WORLD_SIZE, 1], pl.INT32], num_tokens: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32]):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='combine_push'):
        for dst in pl.range(EP_WORLD_SIZE):
            for e in pl.range(N_LOCAL_EXPERTS):
                n = pl.cast(pl.read(pub_counts, [dst * EP_WORLD_SIZE + my_rank, e]), pl.INDEX)
                src_off = pl.const(0, pl.INT32)
                for s in pl.range(EP_WORLD_SIZE):
                    if s < dst:
                        src_off = src_off + pl.read(pub_counts, [s * EP_WORLD_SIZE + my_rank, e])
                src_off_idx = pl.cast(src_off, pl.INDEX)
                for row in pl.range(n):
                    slot = src_off_idx + row
                    r_route = pl.cast(pl.read(recv_r_route_out, [e, slot]), pl.INDEX)
                    payload = pl.load(recv_y_flat, [e * RECV_MAX + slot, 0], [1, D])
                    pld.tile.remote_store(payload, target=routed_y_buf, peer=dst, offsets=[r_route, 0])
        for peer in pl.range(EP_WORLD_SIZE):
            if peer != my_rank:
                pld.system.notify(target=combine_done, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        for src in pl.range(EP_WORLD_SIZE):
            if src != my_rank:
                pld.system.wait(signal=combine_done, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    for tb in pl.spmd(T // 4, name_hint='combine_reduce'):
        for tt in pl.range(4):
            t = tb * 4 + tt
            if t < active_tokens:
                acc = pl.cast(sh[t:t + 1, :], target_type=pl.FP32)
                for k in pl.range(TOPK):
                    r = t * TOPK + k
                    acc = pl.add(acc, pl.cast(routed_y_buf[r:r + 1, :], target_type=pl.FP32))
                ffn_out[t:t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode='rint')
            else:
                ffn_out[t:t + 1, :] = sh[t:t + 1, :]
