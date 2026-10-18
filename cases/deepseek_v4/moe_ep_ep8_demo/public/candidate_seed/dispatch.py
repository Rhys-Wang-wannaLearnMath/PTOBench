import pypto.language as pl

import pypto.language.distributed as pld

from config import CASE_CONFIG as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

D = M.hidden_size

TOPK = M.num_experts_per_tok

N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

W_PAD = 8

IDX_PAD = 8

X_STAGE_ROWS = 8

@pl.jit.inline
def dispatch_ep(indices: pl.Tensor[[T, TOPK], pl.INT32], x_norm_i8: pl.Tensor[[T, D], pl.INT8], x_norm_scale: pl.Tensor[[T, 1], pl.FP32], weights: pl.Tensor[[T, TOPK], pl.FP32], recv_x_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8], recv_scale_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32], recv_w_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32], recv_r_route_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32], recv_count_out: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32], pub_counts: pld.DistributedTensor[[EP_WORLD_SIZE * EP_WORLD_SIZE, N_LOCAL_EXPERTS], pl.INT32], count_done: pld.DistributedTensor[[EP_WORLD_SIZE, 1], pl.INT32], data_done: pld.DistributedTensor[[EP_WORLD_SIZE, 1], pl.INT32], recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8], recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, W_PAD], pl.FP32], recv_w: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, W_PAD], pl.FP32], recv_r_route: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, IDX_PAD], pl.INT32], num_tokens: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32]):
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='dispatch_ep'):
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        send_counts = pl.array.create(EP_WORLD_SIZE * N_LOCAL_EXPERTS, pl.INT32)
        for d in pl.range(EP_WORLD_SIZE):
            for e in pl.range(N_LOCAL_EXPERTS):
                send_counts[d * N_LOCAL_EXPERTS + e] = 0
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                d = eid // N_LOCAL_EXPERTS
                e = eid - d * N_LOCAL_EXPERTS
                cur = send_counts[d * N_LOCAL_EXPERTS + e]
                send_counts[d * N_LOCAL_EXPERTS + e] = cur + 1
        for peer in pl.range(EP_WORLD_SIZE):
            for d in pl.range(EP_WORLD_SIZE):
                for e in pl.range(N_LOCAL_EXPERTS):
                    v = send_counts[d * N_LOCAL_EXPERTS + e]
                    if v != 0:
                        pld.system.notify(target=pub_counts, peer=peer, offsets=[my_rank * EP_WORLD_SIZE + d, e], value=v, op=pld.NotifyOp.AtomicAdd)
        for peer in pl.range(EP_WORLD_SIZE):
            if peer != my_rank:
                pld.system.notify(target=count_done, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        for src in pl.range(EP_WORLD_SIZE):
            if src != my_rank:
                pld.system.wait(signal=count_done, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        my_slot_at_dst = pl.array.create(EP_WORLD_SIZE * N_LOCAL_EXPERTS, pl.INT32)
        for d in pl.range(EP_WORLD_SIZE):
            for e in pl.range(N_LOCAL_EXPERTS):
                acc = pl.const(0, pl.INT32)
                for s in pl.range(EP_WORLD_SIZE):
                    if s < my_rank:
                        acc = acc + pl.read(pub_counts, [s * EP_WORLD_SIZE + d, e])
                my_slot_at_dst[d * N_LOCAL_EXPERTS + e] = acc
        for e in pl.range(N_LOCAL_EXPERTS):
            acc = pl.const(0, pl.INT32)
            for s in pl.range(EP_WORLD_SIZE):
                acc = acc + pl.read(pub_counts, [s * EP_WORLD_SIZE + my_rank, e])
            pl.write(recv_count_out, [e, 0], acc)
        cursor = pl.array.create(EP_WORLD_SIZE * N_LOCAL_EXPERTS, pl.INT32)
        for d in pl.range(EP_WORLD_SIZE):
            for e in pl.range(N_LOCAL_EXPERTS):
                cursor[d * N_LOCAL_EXPERTS + e] = 0
        scale_tile = pl.tile.full([1, W_PAD], dtype=pl.FP32, value=0.0)
        w_tile = pl.tile.full([1, W_PAD], dtype=pl.FP32, value=0.0)
        idx_tile = pl.tile.full([1, IDX_PAD], dtype=pl.INT32, value=0)
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL_EXPERTS
                loc_e = eid - dst * N_LOCAL_EXPERTS
                bucket = dst * N_LOCAL_EXPERTS + loc_e
                cur_val = cursor[bucket]
                slot_off = my_slot_at_dst[bucket]
                slot = slot_off + cur_val
                row = loc_e * RECV_MAX + slot
                cursor[bucket] = cur_val + 1
                r_route = t * TOPK + k
                payload = pl.load(x_norm_i8, [t, 0], [1, D])
                pld.tile.remote_store(payload, target=recv_x, peer=dst, offsets=[row, 0])
                pl.tile.write(scale_tile, [0, 0], pl.read(x_norm_scale, [t, 0]))
                pld.tile.remote_store(scale_tile, target=recv_scale, peer=dst, offsets=[row, 0])
                pl.tile.write(w_tile, [0, 0], pl.read(weights, [t, k]))
                pld.tile.remote_store(w_tile, target=recv_w, peer=dst, offsets=[row, 0])
                pl.tile.write(idx_tile, [0, 0], pl.cast(r_route, pl.INT32))
                pld.tile.remote_store(idx_tile, target=recv_r_route, peer=dst, offsets=[row, 0])
        for peer in pl.range(EP_WORLD_SIZE):
            if peer != my_rank:
                pld.system.notify(target=data_done, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        for src in pl.range(EP_WORLD_SIZE):
            if src != my_rank:
                pld.system.wait(signal=data_done, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        for row in pl.range(0, N_LOCAL_EXPERTS * RECV_MAX, X_STAGE_ROWS):
            recv_x_out_flat[row:row + X_STAGE_ROWS, :] = recv_x[row:row + X_STAGE_ROWS, :]
        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
            for slot in pl.range(n_rows):
                row = e * RECV_MAX + slot
                pl.write(recv_scale_out, [e, slot], pl.read(recv_scale, [row, 0]))
                pl.write(recv_w_out, [e, slot], pl.read(recv_w, [row, 0]))
                pl.write(recv_r_route_out, [e, slot], pl.read(recv_r_route, [row, 0]))
