# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num

"""ORACLE closure for deepseek_v4_flash_dspark.prefill_layer_csa_flash; fixed bindings from public config."""
from config import D
from config import HC_MULT
from config import N_LOCAL
from config import N_RANKS
from config import N_ROUTES
from config import RECV_MAX
from config import T
from config import TOPK
from expert_routed import golden_expert_routed
from expert_shared import golden_expert_shared
from gate import golden_gate_core
from hc_post import golden_hc_post
from hc_pre import golden_hc_pre


def _golden_moe_single(tensors):
    """Per-rank torch reference. Replays the 4 stages on host. Each rank's
    output depends only on its own inputs because the dispatch+combine round-
    trip is r_route-keyed and shape-preserving (test_l3 pattern).

    The per-route result is invariant to the packing layout (each recv row's
    SwiGLU output depends only on that row's own input), so this src-major host
    packing matches the device's per-source-lane cumsum layout by construction."""
    import torch

    # Oracle-local dependency bound during isolated module load.
    # Oracle-local dependency bound during isolated module load.
    # Oracle-local dependency bound during isolated module load.
    # Oracle-local dependency bound during isolated module load.
    # Oracle-local dependency bound during isolated module load.

    x_next_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # Stages 1-2: hc_pre + gate per rank. Rank-independent, so compute once and
    # reuse for both the dispatch replay and each rank's local stages.
    all_post = []
    all_comb = []
    all_indices = []
    all_x_i8 = []
    all_scale = []
    all_weights = []
    for src in range(N_RANKS):
        src_x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
        src_post = torch.zeros(T, HC_MULT, dtype=torch.float32)
        src_comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
        golden_hc_pre({
            "x":        tensors["x_hc"][src],
            "hc_fn":    tensors["hc_ffn_fn"][src],
            "hc_scale": tensors["hc_ffn_scale"][src],
            "hc_base":  tensors["hc_ffn_base"][src],
            "x_mixed":  src_x_mixed,
            "post":     src_post,
            "comb":     src_comb,
        })
        src_x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
        src_x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
        src_indices = torch.zeros(T, TOPK, dtype=torch.int32)
        src_weights = torch.zeros(T, TOPK, dtype=torch.float32)
        golden_gate_core({
            "x_mixed":      src_x_mixed,
            "norm_w":       tensors["norm_w"][src],
            "gate_w":       tensors["gate_w"][src],
            "gate_bias":    tensors["gate_bias"][src],
            "layer_id":     tensors["layer_id"],
            "num_tokens":   tensors["num_tokens"],
            "tid2eid":      tensors["tid2eid"][src],
            "input_ids":    tensors["input_ids"][src],
            "x_norm_i8":    src_x_norm_i8,
            "x_norm_scale": src_x_norm_scale,
            "indices":      src_indices,
            "weights":      src_weights,
        })
        all_post.append(src_post)
        all_comb.append(src_comb)
        all_indices.append(src_indices)
        all_x_i8.append(src_x_norm_i8)
        all_scale.append(src_x_norm_scale)
        all_weights.append(src_weights)

    # Route counts per (src, dst, local expert); drives the per-source lane cumsum.
    send_counts = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(num_tokens):
            for k in range(TOPK):
                eid = int(all_indices[src][t, k].item())
                send_counts[src, eid // N_LOCAL, eid % N_LOCAL] += 1

    # Stages 4-5: dispatch replay + routed expert per dst. Also rank-independent
    # (each recv row's SwiGLU output depends only on that row), so compute once.
    dst_recv_y = {}
    for dst in range(N_RANKS):
        # Pack onto rank dst in src-major order within each local expert — same
        # convention as dispatch's per-source lane cumsum.
        d_recv_x = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.int8)
        d_recv_scale = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
        d_recv_w = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
        d_recv_count = torch.zeros(N_LOCAL, 1, dtype=torch.int32)
        d_slot_offsets = torch.zeros(N_RANKS, N_LOCAL, dtype=torch.int32)
        d_running = torch.zeros(N_LOCAL, dtype=torch.int32)
        for src in range(N_RANKS):
            d_slot_offsets[src] = d_running.clone()
            d_running = d_running + send_counts[src, dst]
        for e in range(N_LOCAL):
            d_recv_count[e, 0] = int(d_running[e].item())
        for src in range(N_RANKS):
            cursor = torch.zeros(N_LOCAL, dtype=torch.int32)
            for t in range(num_tokens):
                for k in range(TOPK):
                    eid = int(all_indices[src][t, k].item())
                    if eid // N_LOCAL != dst:
                        continue
                    loc_e = eid % N_LOCAL
                    slot = int(d_slot_offsets[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    d_recv_x[loc_e, slot, :] = all_x_i8[src][t, :]
                    d_recv_scale[loc_e, slot] = float(all_scale[src][t, 0].item())
                    d_recv_w[loc_e, slot] = float(all_weights[src][t, k].item())
        d_recv_y = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.bfloat16)
        golden_expert_routed({
            "recv_x":            d_recv_x,
            "recv_scale_dq":     d_recv_scale,
            "recv_weights":      d_recv_w,
            "recv_expert_count": d_recv_count,
            "routed_w1":         tensors["routed_w1"][dst],
            "routed_w1_scale":   tensors["routed_w1_scale"][dst],
            "routed_w3":         tensors["routed_w3"][dst],
            "routed_w3_scale":   tensors["routed_w3_scale"][dst],
            "routed_w2":         tensors["routed_w2"][dst],
            "routed_w2_scale":   tensors["routed_w2_scale"][dst],
            "recv_y":            d_recv_y,
        })
        dst_recv_y[dst] = d_recv_y

    for r in range(N_RANKS):
        x_norm_i8 = all_x_i8[r]
        x_norm_scale = all_scale[r]
        post_t = all_post[r]
        comb_t = all_comb[r]

        # Stage 3: expert_shared (local)
        sh = torch.zeros(T, D, dtype=torch.bfloat16)
        golden_expert_shared({
            "x_local_i8":       x_norm_i8,
            "x_local_scale_dq": x_norm_scale,
            "num_tokens":       tensors["num_tokens"],
            "shared_w1":        tensors["shared_w1"][r],
            "shared_w1_scale":  tensors["shared_w1_scale"][r],
            "shared_w3":        tensors["shared_w3"][r],
            "shared_w3_scale":  tensors["shared_w3_scale"][r],
            "shared_w2":        tensors["shared_w2"][r],
            "shared_w2_scale":  tensors["shared_w2_scale"][r],
            "sh":               sh,
        })

        # Stage 6: combine — for each (src, t, k) that originated on this
        # rank, find the (loc_e, slot) on rank dst where the SwiGLU result
        # landed, then accumulate by r_route = t*TOPK+k.
        my_routes = []
        for t in range(num_tokens):
            for k in range(TOPK):
                eid = int(all_indices[r][t, k].item())
                dst = eid // N_LOCAL
                loc_e = eid % N_LOCAL
                my_routes.append((t, k, dst, loc_e))

        # Rank r's contribution to dst sits at slot offset
        # Sigma_{s<r} send_counts[s, dst, loc_e] plus a running per-(dst, loc_e)
        # cursor over r's own routes in (t, k) order.
        routed_y_buf_r = torch.zeros(N_ROUTES, D, dtype=torch.bfloat16)
        cursors = {}
        for (t, k, dst, loc_e) in my_routes:
            src_off = int(send_counts[:r, dst, loc_e].sum().item())
            cursor = cursors.get((dst, loc_e), 0)
            cursors[(dst, loc_e)] = cursor + 1
            r_route = t * TOPK + k
            routed_y_buf_r[r_route, :] = dst_recv_y[dst][loc_e, src_off + cursor, :]

        # Stage 7: reduce + sh + hc_post
        acc = sh.float().clone()
        for k in range(TOPK):
            for t in range(num_tokens):
                acc[t, :] += routed_y_buf_r[t * TOPK + k, :].float()
        ffn_out = acc.to(torch.bfloat16)
        if "ffn_out" in tensors:
            tensors["ffn_out"][r].copy_(ffn_out)
        x_next_r = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
        golden_hc_post({
            "x":        ffn_out,
            "residual": tensors["x_hc"][r],
            "post":     post_t,
            "comb":     comb_t,
            "y":        x_next_r,
        })
        x_next_out[r] = x_next_r

    tensors["x_next"][:] = x_next_out


def golden_moe(tensors):
    """Evaluate one MoE layer."""
    _golden_moe_single(tensors)


