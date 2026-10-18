import torch
from config import CASE_CONFIG as M, T, D, HC_MULT, MIX_HC, HC_DIM, N_RANKS, N_LOCAL, RECV_MAX, TOPK, N_ROUTES, INT8_AMAX_EPS, INT8_SCALE_MAX
N_LOCAL_EXPERTS = N_LOCAL
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps
N_HASH_LAYERS = M.num_hash_layers
ROUTE_SCALE = M.routed_scaling_factor
SWIGLU_LIMIT = M.swiglu_limit
RMS_K_CHUNK = 512
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK

def int8_quant_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path.

    Rounds to int8 through fp16 to match the device rounding; returns the
    per-row dequant scale (``1 / scale_quant``).
    """
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return (out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1))

def _golden_a2a3_cube_linear(x_flat_2d, hc_fn):
    """Emulate the A2/A3 f32 Cube MAD reduction used by the HC projection."""
    import torch
    cube_acc_k = 4
    group_block = 64
    assert LINEAR_K_PER_SPLIT % cube_acc_k == 0
    groups_per_split = LINEAR_K_PER_SPLIT // cube_acc_k
    t_dim = x_flat_2d.shape[0]
    x_groups = x_flat_2d.reshape(t_dim, 1, LINEAR_OK, groups_per_split, cube_acc_k)
    w_groups = hc_fn.reshape(1, MIX_HC, LINEAR_OK, groups_per_split, cube_acc_k)
    split_mixes = torch.zeros(t_dim, MIX_HC, LINEAR_OK, dtype=torch.float32)
    for group0 in range(0, groups_per_split, group_block):
        group_dots = (x_groups[..., group0:group0 + group_block, :].double() * w_groups[..., group0:group0 + group_block, :].double()).sum(dim=-1)
        for group in range(group_dots.shape[-1]):
            split_mixes = (split_mixes.double() + group_dots[..., group]).float()
    mixes = torch.zeros(t_dim, MIX_HC, dtype=torch.float32)
    for split in range(LINEAR_OK):
        mixes += split_mixes[..., split]
    return mixes

def golden_hc_pre(tensors):
    """Torch reference matching the target HC-pre reduction and nonlinearities."""
    import torch
    x = tensors['x'].float()
    hc_fn = tensors['hc_fn'].float()
    hc_scale = tensors['hc_scale'].float()
    hc_base = tensors['hc_base'].float()
    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)
    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)
    mixes = _golden_a2a3_cube_linear(x_flat_2d, hc_fn)
    mixes *= rsqrt
    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1] + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]).view(t_dim, HC_MULT, HC_MULT)
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = y0 + y1 + (y2 + y3)
    tensors['x_mixed'][:] = y.to(torch.bfloat16).reshape(t_dim, D)
    tensors['post'][:] = post_t.reshape(t_dim, HC_MULT)
    tensors['comb'][:] = comb_t.reshape(t_dim, HC_MULT * HC_MULT)

def golden_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    import torch
    x = tensors['x'].float()
    residual = tensors['residual'].float()
    post = tensors['post'].float()
    comb = tensors['comb'].float().reshape(-1, HC_MULT, HC_MULT)
    T = x.shape[0]
    y_fp32 = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, in_h, :] * comb[:, in_h, out_h:out_h + 1]
        y_fp32[:, out_h, :] = y_row
    y = y_fp32
    tensors['y'][:] = y

def _per_token_int8_quant(x_bf16):
    import torch
    x_f32 = x_bf16.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = INT8_SCALE_MAX / amax
    scaled = x_f32 * scale_q
    x_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = (1.0 / scale_q).reshape(-1)
    return (x_i8, scale_dq)

def golden_gate_core(tensors):
    import torch
    num_tokens = max(0, min(T, int(tensors.get('num_tokens', T))))
    x_f = tensors['x_mixed'].float().view(T, D)
    norm_w = tensors['norm_w'].float()
    sq_sum = (x_f * x_f).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)
    xg = x_f * norm_w.view(1, D)
    (x_norm_i8, scale_dq_g) = _per_token_int8_quant(xg)
    x_norm_scale = scale_dq_g.reshape(T, 1) * inv_rms
    gate_w = tensors['gate_w'].float()
    gate_bias = tensors['gate_bias'].float()
    logits = inv_rms * (xg.float() @ gate_w.T)
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.view(1, -1)
    layer_id = int(tensors['layer_id'])
    if layer_id < N_HASH_LAYERS:
        tid2eid = tensors['tid2eid']
        input_ids = tensors['input_ids']
        indices = tid2eid[input_ids.flatten().long()]
    else:
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]
    topk_vals = torch.gather(scores, dim=-1, index=indices.long())
    denom = topk_vals.sum(dim=-1, keepdim=True)
    weights = topk_vals / denom * ROUTE_SCALE
    if num_tokens < T:
        x_norm_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0
    tensors['x_norm_i8'][:] = x_norm_i8
    tensors['x_norm_scale'][:] = x_norm_scale.reshape(T, 1)
    tensors['indices'][:] = indices.to(torch.int32)
    tensors['weights'][:] = weights.to(torch.float32)

def golden_expert_shared(tensors):
    """Torch reference for the shared expert.

    Input is the per-token INT8 quant produced by gate (shared with
    dispatch / routed expert); we dequant inside to match the kernel's
    dequant-then-matmul pattern."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)
    x_local_i8 = tensors['x_local_i8']
    x_local_scale_dq = tensors['x_local_scale_dq'].float()
    x_local = x_local_i8.float() * x_local_scale_dq
    sw1 = dequant_w(tensors['shared_w1'], tensors['shared_w1_scale'].float())
    sw3 = dequant_w(tensors['shared_w3'], tensors['shared_w3_scale'].float())
    sw2 = dequant_w(tensors['shared_w2'], tensors['shared_w2_scale'].float())
    sh_gate = x_local @ sw1.T
    sh_up = x_local @ sw3.T
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up
    (sh_h_i8, sh_h_sd) = int8_quant_per_row(sh_h)
    sh_h = sh_h_i8.float() * sh_h_sd
    sh = sh_h @ sw2.T
    tensors['sh'][:] = sh.to(torch.bfloat16)

def golden_expert_routed(tensors):
    """Torch reference for the routed expert. recv_y is the per-row routing-
    weight-scaled SwiGLU output, ready for combine reduce to simply sum.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)
    recv_x_i8 = tensors['recv_x']
    recv_scale_dq = tensors['recv_scale_dq'].float()
    recv_weights = tensors['recv_weights'].float()
    recv_expert_count = tensors['recv_expert_count']
    w1 = dequant_w(tensors['routed_w1'], tensors['routed_w1_scale'].float())
    w3 = dequant_w(tensors['routed_w3'], tensors['routed_w3_scale'].float())
    w2 = dequant_w(tensors['routed_w2'], tensors['routed_w2_scale'].float())
    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub_i8 = recv_x_i8[e, :n_rows, :]
        x_sub_sd = recv_scale_dq[e, :n_rows].reshape(-1, 1)
        x_sub_q = x_sub_i8.float() * x_sub_sd
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)
        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        (h_i8, h_sd) = int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_per_row)
        recv_y[e, :n_rows, :] = h @ w2[e].T
    tensors['recv_y'][:] = recv_y.to(torch.bfloat16)

def golden_moe(tensors):
    """Per-rank torch reference. Replays the 4 stages on host. Each rank's
    output depends only on its own inputs because the dispatch+combine round-
    trip is r_route-keyed and shape-preserving (test_l3 pattern).

    The per-route result is invariant to the packing layout (each recv row's
    SwiGLU output depends only on that row's own input), so this src-major host
    packing matches the device's per-source-lane cumsum layout by construction."""
    import torch
    x_next_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    num_tokens = max(0, min(T, int(tensors.get('num_tokens', T))))
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
        golden_hc_pre({'x': tensors['x_hc'][src], 'hc_fn': tensors['hc_ffn_fn'][src], 'hc_scale': tensors['hc_ffn_scale'][src], 'hc_base': tensors['hc_ffn_base'][src], 'x_mixed': src_x_mixed, 'post': src_post, 'comb': src_comb})
        src_x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
        src_x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
        src_indices = torch.zeros(T, TOPK, dtype=torch.int32)
        src_weights = torch.zeros(T, TOPK, dtype=torch.float32)
        golden_gate_core({'x_mixed': src_x_mixed, 'norm_w': tensors['norm_w'][src], 'gate_w': tensors['gate_w'][src], 'gate_bias': tensors['gate_bias'][src], 'layer_id': tensors['layer_id'], 'num_tokens': tensors['num_tokens'], 'tid2eid': tensors['tid2eid'][src], 'input_ids': tensors['input_ids'][src], 'x_norm_i8': src_x_norm_i8, 'x_norm_scale': src_x_norm_scale, 'indices': src_indices, 'weights': src_weights})
        all_post.append(src_post)
        all_comb.append(src_comb)
        all_indices.append(src_indices)
        all_x_i8.append(src_x_norm_i8)
        all_scale.append(src_x_norm_scale)
        all_weights.append(src_weights)
    send_counts = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(num_tokens):
            for k in range(TOPK):
                eid = int(all_indices[src][t, k].item())
                send_counts[src, eid // N_LOCAL, eid % N_LOCAL] += 1
    dst_recv_y = {}
    for dst in range(N_RANKS):
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
        golden_expert_routed({'recv_x': d_recv_x, 'recv_scale_dq': d_recv_scale, 'recv_weights': d_recv_w, 'recv_expert_count': d_recv_count, 'routed_w1': tensors['routed_w1'][dst], 'routed_w1_scale': tensors['routed_w1_scale'][dst], 'routed_w3': tensors['routed_w3'][dst], 'routed_w3_scale': tensors['routed_w3_scale'][dst], 'routed_w2': tensors['routed_w2'][dst], 'routed_w2_scale': tensors['routed_w2_scale'][dst], 'recv_y': d_recv_y})
        dst_recv_y[dst] = d_recv_y
    for r in range(N_RANKS):
        x_norm_i8 = all_x_i8[r]
        x_norm_scale = all_scale[r]
        post_t = all_post[r]
        comb_t = all_comb[r]
        sh = torch.zeros(T, D, dtype=torch.bfloat16)
        golden_expert_shared({'x_local_i8': x_norm_i8, 'x_local_scale_dq': x_norm_scale, 'num_tokens': tensors['num_tokens'], 'shared_w1': tensors['shared_w1'][r], 'shared_w1_scale': tensors['shared_w1_scale'][r], 'shared_w3': tensors['shared_w3'][r], 'shared_w3_scale': tensors['shared_w3_scale'][r], 'shared_w2': tensors['shared_w2'][r], 'shared_w2_scale': tensors['shared_w2_scale'][r], 'sh': sh})
        my_routes = []
        for t in range(num_tokens):
            for k in range(TOPK):
                eid = int(all_indices[r][t, k].item())
                dst = eid // N_LOCAL
                loc_e = eid % N_LOCAL
                my_routes.append((t, k, dst, loc_e))
        routed_y_buf_r = torch.zeros(N_ROUTES, D, dtype=torch.bfloat16)
        cursors = {}
        for (t, k, dst, loc_e) in my_routes:
            src_off = int(send_counts[:r, dst, loc_e].sum().item())
            cursor = cursors.get((dst, loc_e), 0)
            cursors[dst, loc_e] = cursor + 1
            r_route = t * TOPK + k
            routed_y_buf_r[r_route, :] = dst_recv_y[dst][loc_e, src_off + cursor, :]
        acc = sh.float().clone()
        for k in range(TOPK):
            for t in range(num_tokens):
                acc[t, :] += routed_y_buf_r[t * TOPK + k, :].float()
        ffn_out = acc.to(torch.bfloat16)
        x_next_r = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
        golden_hc_post({'x': ffn_out, 'residual': tensors['x_hc'][r], 'post': post_t, 'comb': comb_t, 'y': x_next_r})
        x_next_out[r] = x_next_r
    tensors['x_next'][:] = x_next_out

def compute(tensors):
    values = {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
              for name, value in tensors.items()}
    values['x_next'] = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    golden_moe(values)
    return {'x_next': values['x_next']}
