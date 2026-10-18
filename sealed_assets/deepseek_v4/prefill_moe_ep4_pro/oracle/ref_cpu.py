import torch
import torch.nn.functional as F
from config import T, HC_MULT as LANES, D, MIX_HC as MIX, HC_DIM, MOE_INTER as INTER, TOPK, N_LOCAL as LOCAL, NORM_EPS, HC_EPS, HC_SINKHORN_ITER, N_HASH_LAYERS, INT8_SCALE_MAX as INT8_MAX, INT8_AMAX_EPS as INT8_EPS, SWIGLU_LIMIT, ROUTE_SCALE

def _bf16_rint(value):
    rounded = value.contiguous().view(torch.int32) + 32768 & -65536
    return rounded.view(torch.float32).to(torch.bfloat16)

def _hc_pre(source, fn, scale, base):
    flat = source.reshape(T, HC_DIM).float()
    square_sum = torch.zeros(T, 1)
    for offset in range(0, HC_DIM, 512):
        part = flat[:, offset:offset + 512]
        square_sum += (part * part).sum(1, keepdim=True)
    inv_rms = torch.rsqrt(square_sum * (1.0 / HC_DIM) + HC_EPS)
    columns = []
    for mix in range(MIX):
        dot = torch.zeros(T, 1)
        for offset in range(0, HC_DIM, 256):
            dot += (flat[:, offset:offset + 256] * fn[mix:mix + 1, offset:offset + 256].float()).sum(1, keepdim=True)
        columns.append(dot * inv_rms)
    mixed = torch.cat(columns, 1)
    (scale, base) = (scale.float(), base.float())
    pre = torch.sigmoid(mixed[:, :LANES] * scale[0] + base[:LANES]) + HC_EPS
    post = 2 * torch.sigmoid(mixed[:, LANES:2 * LANES] * scale[1] + base[LANES:2 * LANES])
    comb = (mixed[:, 2 * LANES:] * scale[2] + base[2 * LANES:]).reshape(T, LANES, LANES)
    comb = torch.softmax(comb, -1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    output = torch.zeros(T, D)
    for lane in range(LANES):
        output += source[:, lane].float() * pre[:, lane:lane + 1]
    return (_bf16_rint(output), post, comb.reshape(T, LANES * LANES))

def _quantize_rows(value):
    rows = value.float().reshape(-1, value.shape[-1])
    amax = rows.abs().amax(-1, keepdim=True).clamp_min(INT8_EPS)
    quant_scale = INT8_MAX / amax
    quantized = torch.round(rows * quant_scale).to(torch.int32).to(torch.float16).to(torch.int8)
    return (quantized.reshape_as(value), (1.0 / quant_scale).reshape(*value.shape[:-1], 1))

def _gate(x_mixed, norm_w, gate_w, gate_bias, tid2eid, input_ids, layer_id, num_tokens):
    x = x_mixed.float()
    inv_rms = torch.rsqrt((x * x).sum(-1, keepdim=True) / D + NORM_EPS)
    x_norm = (x * inv_rms * norm_w.float().reshape(1, D)).to(torch.bfloat16)
    (x_i8, x_scale) = _quantize_rows(x_norm)
    logits = x_norm.float() @ gate_w.float().T
    scores = (logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))).sqrt()
    if layer_id < N_HASH_LAYERS:
        indices = tid2eid[input_ids.flatten().long()]
    else:
        indices = torch.argsort(-(scores + gate_bias.float()), dim=-1, stable=True)[:, :TOPK]
    selected = scores.gather(1, indices.long())
    weights = selected / selected.sum(-1, keepdim=True) * ROUTE_SCALE
    if num_tokens < T:
        x_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0
    return (x_i8, x_scale, indices.to(torch.int32), weights.float())

def _dequant(weight, scale):
    return weight.float() * scale.float().unsqueeze(-1)

def _shared(x_i8, x_scale, inputs, rank):
    x = x_i8.float() * x_scale.float()
    w1 = _dequant(inputs['shared_w1'][rank], inputs['shared_w1_scale'][rank])
    w3 = _dequant(inputs['shared_w3'][rank], inputs['shared_w3_scale'][rank])
    w2 = _dequant(inputs['shared_w2'][rank], inputs['shared_w2_scale'][rank])
    (gate, up) = (x @ w1.T, x @ w3.T)
    if SWIGLU_LIMIT > 0:
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    (hidden_i8, hidden_scale) = _quantize_rows(F.silu(gate) * up)
    return (hidden_i8.float() * hidden_scale @ w2.T).to(torch.bfloat16)

def _routed(recv_x, recv_scale, recv_weight, counts, inputs, rank):
    w1 = _dequant(inputs['routed_w1'][rank], inputs['routed_w1_scale'][rank])
    w3 = _dequant(inputs['routed_w3'][rank], inputs['routed_w3_scale'][rank])
    w2 = _dequant(inputs['routed_w2'][rank], inputs['routed_w2_scale'][rank])
    output = torch.zeros(LOCAL, recv_x.shape[1], D)
    for expert in range(LOCAL):
        rows = int(counts[expert, 0])
        if rows == 0:
            continue
        x = recv_x[expert, :rows].float() * recv_scale[expert, :rows, None]
        (gate, up) = (x @ w1[expert].T, x @ w3[expert].T)
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        (hidden_i8, hidden_scale) = _quantize_rows(F.silu(gate) * up)
        hidden = hidden_i8.float() * (hidden_scale * recv_weight[expert, :rows, None])
        output[expert, :rows] = hidden @ w2[expert].T
    return output.to(torch.bfloat16)

def _hc_post(ffn, residual, post, comb):
    matrix = comb.reshape(T, LANES, LANES).float()
    output = torch.zeros(T, LANES, D)
    for out_lane in range(LANES):
        row = ffn.float() * post[:, out_lane:out_lane + 1]
        for in_lane in range(LANES):
            row = row + residual[:, in_lane].float() * matrix[:, in_lane, out_lane:out_lane + 1]
        output[:, out_lane] = row
    return output.to(torch.bfloat16)

def _compute_integrated_moe(inputs, scalars):
    if set(scalars) != {'layer_id', 'num_tokens'}:
        raise ValueError(f'unexpected MoE scalars: {sorted(scalars)}')
    layer_id = int(scalars['layer_id'])
    num_tokens = max(0, min(T, int(scalars['num_tokens'])))
    ranks = int(inputs['x_hc'].shape[0])
    recv_max = ranks * T
    if ranks not in (2, 4, 8):
        raise ValueError(f'unsupported official EP topology: {ranks}')
    (mixed, post, comb, x_i8, x_scale, indices, weights, shared) = ([] for _ in range(8))
    for rank in range(ranks):
        (rank_mixed, rank_post, rank_comb) = _hc_pre(inputs['x_hc'][rank], inputs['hc_ffn_fn'][rank], inputs['hc_ffn_scale'][rank], inputs['hc_ffn_base'][rank])
        (rank_i8, rank_scale, rank_indices, rank_weights) = _gate(rank_mixed, inputs['norm_w'][rank], inputs['gate_w'][rank], inputs['gate_bias'][rank], inputs['tid2eid'][rank], inputs['input_ids'][rank], layer_id, num_tokens)
        mixed.append(rank_mixed)
        post.append(rank_post)
        comb.append(rank_comb)
        x_i8.append(rank_i8)
        x_scale.append(rank_scale)
        indices.append(rank_indices)
        weights.append(rank_weights)
        shared.append(_shared(rank_i8, rank_scale, inputs, rank))
    counts = torch.zeros(ranks, ranks, LOCAL, dtype=torch.int32)
    for src in range(ranks):
        for token in range(num_tokens):
            for slot in range(TOPK):
                eid = int(indices[src][token, slot])
                counts[src, eid // LOCAL, eid % LOCAL] += 1
    (routed_by_dst, offsets_by_dst) = ([], [])
    for dst in range(ranks):
        recv_x = torch.zeros(LOCAL, recv_max, D, dtype=torch.int8)
        recv_scale = torch.zeros(LOCAL, recv_max)
        recv_weight = torch.zeros(LOCAL, recv_max)
        offsets = torch.zeros(ranks, LOCAL, dtype=torch.int32)
        running = torch.zeros(LOCAL, dtype=torch.int32)
        for src in range(ranks):
            offsets[src] = running.clone()
            running += counts[src, dst]
        cursor = torch.zeros(ranks, LOCAL, dtype=torch.int32)
        for src in range(ranks):
            for token in range(num_tokens):
                for slot in range(TOPK):
                    eid = int(indices[src][token, slot])
                    if eid // LOCAL != dst:
                        continue
                    expert = eid % LOCAL
                    packed = int(offsets[src, expert] + cursor[src, expert])
                    cursor[src, expert] += 1
                    recv_x[expert, packed] = x_i8[src][token]
                    recv_scale[expert, packed] = x_scale[src][token, 0]
                    recv_weight[expert, packed] = weights[src][token, slot]
        routed_by_dst.append(_routed(recv_x, recv_scale, recv_weight, running[:, None], inputs, dst))
        offsets_by_dst.append(offsets)
    outputs = torch.zeros(ranks, T, LANES, D, dtype=torch.bfloat16)
    for src in range(ranks):
        route_buffer = torch.zeros(T * TOPK, D, dtype=torch.bfloat16)
        cursor = torch.zeros(ranks, LOCAL, dtype=torch.int32)
        for token in range(num_tokens):
            for slot in range(TOPK):
                eid = int(indices[src][token, slot])
                (dst, expert) = (eid // LOCAL, eid % LOCAL)
                packed = int(offsets_by_dst[dst][src, expert] + cursor[dst, expert])
                cursor[dst, expert] += 1
                route_buffer[token * TOPK + slot] = routed_by_dst[dst][expert, packed]
        accumulated = shared[src].float().clone()
        for slot in range(TOPK):
            for token in range(num_tokens):
                accumulated[token] += route_buffer[token * TOPK + slot].float()
        outputs[src] = _hc_post(accumulated.to(torch.bfloat16), inputs['x_hc'][src], post[src], comb[src])
    return {'x_next': outputs}

def compute_moe(inputs, scalars):
    global T
    tokens = int(inputs['x_hc'].shape[1])
    prior_tokens = T
    try:
        T = tokens
        return _compute_integrated_moe(inputs, scalars)
    finally:
        T = prior_tokens

def compute(inputs, scalars):
    return compute_moe(inputs, scalars)
