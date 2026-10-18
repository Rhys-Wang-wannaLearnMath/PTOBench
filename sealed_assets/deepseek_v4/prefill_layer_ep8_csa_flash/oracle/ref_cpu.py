"""Independent packed CPU Oracle; MoE golden bodies consolidated without arithmetic changes."""
from config import D as _cpu_hc_pre_v2_D, HC_DIM as _cpu_hc_pre_v2_HC_DIM, HC_MULT as _cpu_hc_pre_v2_HC_MULT, MIX_HC as _cpu_hc_pre_v2_MIX_HC
from config import CASE_CONFIG as _cpu_hc_pre_v2_M
_cpu_hc_pre_v2_HC_DIM_INV = 1.0 / _cpu_hc_pre_v2_HC_DIM
_cpu_hc_pre_v2_HC_SINKHORN_ITER = _cpu_hc_pre_v2_M.hc_sinkhorn_iters
_cpu_hc_pre_v2_HC_EPS = _cpu_hc_pre_v2_M.hc_eps
_cpu_hc_pre_v2_NORM_EPS = _cpu_hc_pre_v2_M.rms_norm_eps
_cpu_hc_pre_v2_RMS_K_CHUNK = 512
_cpu_hc_pre_v2_LINEAR_K_CHUNK = 256

def _cpu_hc_pre_v2_golden_hc_pre(tensors):
    """Torch reference, direct port of model.py Block.hc_pre + hc_split_sinkhorn."""
    import torch
    x = tensors['x'].float()
    hc_fn = tensors['hc_fn'].float()
    hc_scale = tensors['hc_scale'].float()
    hc_base = tensors['hc_base'].float()
    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, _cpu_hc_pre_v2_HC_DIM)
    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, _cpu_hc_pre_v2_HC_DIM, _cpu_hc_pre_v2_RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + _cpu_hc_pre_v2_RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * _cpu_hc_pre_v2_HC_DIM_INV + _cpu_hc_pre_v2_NORM_EPS)
    mix_cols = []
    for m in range(_cpu_hc_pre_v2_MIX_HC):
        mix_col = torch.zeros(t_dim, 1, dtype=torch.float32)
        for k0 in range(0, _cpu_hc_pre_v2_HC_DIM, _cpu_hc_pre_v2_LINEAR_K_CHUNK):
            x_chunk = x_flat_2d[:, k0:k0 + _cpu_hc_pre_v2_LINEAR_K_CHUNK]
            w_chunk = hc_fn[m:m + 1, k0:k0 + _cpu_hc_pre_v2_LINEAR_K_CHUNK]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1)
    pre = torch.sigmoid(mixes[..., :_cpu_hc_pre_v2_HC_MULT] * hc_scale[0] + hc_base[:_cpu_hc_pre_v2_HC_MULT]) + _cpu_hc_pre_v2_HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., _cpu_hc_pre_v2_HC_MULT:_cpu_hc_pre_v2_HC_MULT * 2] * hc_scale[1] + hc_base[_cpu_hc_pre_v2_HC_MULT:_cpu_hc_pre_v2_HC_MULT * 2])
    comb_t = (mixes[..., _cpu_hc_pre_v2_HC_MULT * 2:] * hc_scale[2] + hc_base[_cpu_hc_pre_v2_HC_MULT * 2:]).view(t_dim, _cpu_hc_pre_v2_HC_MULT, _cpu_hc_pre_v2_HC_MULT)
    comb_t = torch.softmax(comb_t, dim=-1) + _cpu_hc_pre_v2_HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + _cpu_hc_pre_v2_HC_EPS)
    for _ in range(_cpu_hc_pre_v2_HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + _cpu_hc_pre_v2_HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + _cpu_hc_pre_v2_HC_EPS)
    y = torch.zeros(t_dim, _cpu_hc_pre_v2_D, dtype=torch.float32)
    for h in range(_cpu_hc_pre_v2_HC_MULT):
        y += x[:, h, :] * pre[:, h:h + 1]

    def _to_device_bf16(value):
        rounded = value.contiguous().view(torch.int32) + 32768 & -65536
        return rounded.view(torch.float32).to(torch.bfloat16)
    tensors['x_mixed'][:] = _to_device_bf16(y).reshape(t_dim, _cpu_hc_pre_v2_D)
    tensors['post'][:] = post_t.reshape(t_dim, _cpu_hc_pre_v2_HC_MULT)
    tensors['comb'][:] = comb_t.reshape(t_dim, _cpu_hc_pre_v2_HC_MULT * _cpu_hc_pre_v2_HC_MULT)
from config import D as _cpu_hc_post_v2_D, HC_MULT as _cpu_hc_post_v2_HC_MULT

def _cpu_hc_post_v2_golden_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    import torch
    x = tensors['x'].float()
    residual = tensors['residual'].float()
    post = tensors['post'].float()
    comb = tensors['comb'].float().reshape(-1, _cpu_hc_post_v2_HC_MULT, _cpu_hc_post_v2_HC_MULT)
    T = x.shape[0]
    y_fp32 = torch.zeros(T, _cpu_hc_post_v2_HC_MULT, _cpu_hc_post_v2_D, dtype=torch.float32)
    for out_h in range(_cpu_hc_post_v2_HC_MULT):
        y_row = x * post[:, out_h:out_h + 1]
        for in_h in range(_cpu_hc_post_v2_HC_MULT):
            y_row = y_row + residual[:, in_h, :] * comb[:, in_h, out_h:out_h + 1]
        y_fp32[:, out_h, :] = y_row
    y = y_fp32.to(torch.bfloat16)
    tensors['y'][:] = y
from config import D as _cpu_gate_D, TOPK as _cpu_gate_TOPK
from config import CASE_CONFIG as _cpu_gate_M, MOE_TOKENS as _cpu_gate_MOE_TOKENS, INT8_SCALE_MAX as _cpu_gate_INT8_SCALE_MAX, INT8_AMAX_EPS as _cpu_gate_INT8_AMAX_EPS
_cpu_gate_T = _cpu_gate_MOE_TOKENS
_cpu_gate_NORM_EPS = _cpu_gate_M.rms_norm_eps
_cpu_gate_ROUTE_SCALE = _cpu_gate_M.routed_scaling_factor
_cpu_gate_N_HASH_LAYERS = _cpu_gate_M.num_hash_layers

def _cpu_gate__per_token_int8_quant(x_bf16):
    import torch
    x_f32 = x_bf16.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(_cpu_gate_INT8_AMAX_EPS)
    scale_q = _cpu_gate_INT8_SCALE_MAX / amax
    scaled = x_f32 * scale_q
    x_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = (1.0 / scale_q).reshape(-1)
    return (x_i8, scale_dq)

def _cpu_gate_golden_gate_core(tensors):
    import torch
    num_tokens = max(0, min(_cpu_gate_T, int(tensors.get('num_tokens', _cpu_gate_T))))
    x_f = tensors['x_mixed'].float().view(_cpu_gate_T, _cpu_gate_D)
    norm_w = tensors['norm_w'].float()
    sq_sum = (x_f * x_f).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / _cpu_gate_D) + _cpu_gate_NORM_EPS)
    x_normalized = x_f * inv_rms * norm_w.view(1, _cpu_gate_D)
    x_flat = x_normalized.to(torch.bfloat16)
    (x_norm_i8, x_norm_scale) = _cpu_gate__per_token_int8_quant(x_flat)
    gate_w = tensors['gate_w'].float()
    gate_bias = tensors['gate_bias'].float()
    logits = x_flat.float() @ gate_w.T
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.view(1, -1)
    layer_id = int(tensors['layer_id'])
    if layer_id < _cpu_gate_N_HASH_LAYERS:
        tid2eid = tensors['tid2eid']
        input_ids = tensors['input_ids']
        indices = tid2eid[input_ids.flatten().long()]
    else:
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :_cpu_gate_TOPK]
    topk_vals = torch.gather(scores, dim=-1, index=indices.long())
    denom = topk_vals.sum(dim=-1, keepdim=True)
    weights = topk_vals / denom * _cpu_gate_ROUTE_SCALE
    if num_tokens < _cpu_gate_T:
        x_norm_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0
    tensors['x_norm'][:] = x_flat
    tensors['x_norm_i8'][:] = x_norm_i8
    tensors['x_norm_scale'][:] = x_norm_scale.reshape(_cpu_gate_T, 1)
    tensors['indices'][:] = indices.to(torch.int32)
    tensors['weights'][:] = weights.to(torch.float32)
from config import CASE_CONFIG as _cpu_expert_shared_M, INT8_SCALE_MAX as _cpu_expert_shared_INT8_SCALE_MAX, INT8_AMAX_EPS as _cpu_expert_shared_INT8_AMAX_EPS
_cpu_expert_shared_SWIGLU_LIMIT = _cpu_expert_shared_M.swiglu_limit

def _cpu_expert_shared__int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(_cpu_expert_shared_INT8_AMAX_EPS)
    scale_quant = _cpu_expert_shared_INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return (out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1))

def _cpu_expert_shared_golden_expert_shared(tensors):
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
    if _cpu_expert_shared_SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=_cpu_expert_shared_SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-_cpu_expert_shared_SWIGLU_LIMIT, _cpu_expert_shared_SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up
    (sh_h_i8, sh_h_sd) = _cpu_expert_shared__int8_quant_per_row(sh_h)
    sh_h = sh_h_i8.float() * sh_h_sd
    sh = sh_h @ sw2.T
    tensors['sh'][:] = sh.to(torch.bfloat16)
from config import D as _cpu_expert_routed_D
from config import CASE_CONFIG as _cpu_expert_routed_M, INT8_SCALE_MAX as _cpu_expert_routed_INT8_SCALE_MAX, INT8_AMAX_EPS as _cpu_expert_routed_INT8_AMAX_EPS, EP_WORLD_SIZE as _cpu_expert_routed_EP_WORLD_SIZE, RECV_MAX as _cpu_expert_routed_RECV_MAX
_cpu_expert_routed_SWIGLU_LIMIT = _cpu_expert_routed_M.swiglu_limit
_cpu_expert_routed_N_LOCAL_EXPERTS = _cpu_expert_routed_M.n_routed_experts // _cpu_expert_routed_EP_WORLD_SIZE

def _cpu_expert_routed__int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(_cpu_expert_routed_INT8_AMAX_EPS)
    scale_quant = _cpu_expert_routed_INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return (out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1))

def _cpu_expert_routed_golden_expert_routed(tensors):
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
    recv_y = torch.zeros(_cpu_expert_routed_N_LOCAL_EXPERTS, _cpu_expert_routed_RECV_MAX, _cpu_expert_routed_D)
    for e in range(_cpu_expert_routed_N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub_i8 = recv_x_i8[e, :n_rows, :]
        x_sub_sd = recv_scale_dq[e, :n_rows].reshape(-1, 1)
        x_sub_q = x_sub_i8.float() * x_sub_sd
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)
        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if _cpu_expert_routed_SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=_cpu_expert_routed_SWIGLU_LIMIT)
            up = up.clamp(-_cpu_expert_routed_SWIGLU_LIMIT, _cpu_expert_routed_SWIGLU_LIMIT)
        h = F.silu(gate) * up
        (h_i8, h_sd) = _cpu_expert_routed__int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_per_row)
        recv_y[e, :n_rows, :] = h @ w2[e].T
    tensors['recv_y'][:] = recv_y.to(torch.bfloat16)
from config import D as _cpu_moe_D, HC_MULT as _cpu_moe_HC_MULT, N_LOCAL as _cpu_moe_N_LOCAL, N_RANKS as _cpu_moe_N_RANKS, N_ROUTES as _cpu_moe_N_ROUTES, TOPK as _cpu_moe_TOPK
from config import MOE_TOKENS as _cpu_moe_MOE_TOKENS, RECV_MAX as _cpu_moe_RECV_MAX
_cpu_moe_T = _cpu_moe_MOE_TOKENS

def _cpu_moe_golden_moe(tensors):
    """Per-rank torch reference. Replays the 4 stages on host. Each rank's
    output depends only on its own inputs because the dispatch+combine round-
    trip is r_route-keyed and shape-preserving (test_l3 pattern).

    The per-route result is invariant to the packing layout (each recv row's
    SwiGLU output depends only on that row's own input), so this src-major host
    packing matches the device's per-source-lane cumsum layout by construction."""
    import torch
    x_next_out = torch.zeros(_cpu_moe_N_RANKS, _cpu_moe_T, _cpu_moe_HC_MULT, _cpu_moe_D, dtype=torch.bfloat16)
    num_tokens = max(0, min(_cpu_moe_T, int(tensors.get('num_tokens', _cpu_moe_T))))
    for r in range(_cpu_moe_N_RANKS):
        x_mixed = torch.zeros(_cpu_moe_T, _cpu_moe_D, dtype=torch.bfloat16)
        post_t = torch.zeros(_cpu_moe_T, _cpu_moe_HC_MULT, dtype=torch.float32)
        comb_t = torch.zeros(_cpu_moe_T, _cpu_moe_HC_MULT * _cpu_moe_HC_MULT, dtype=torch.float32)
        _cpu_hc_pre_v2_golden_hc_pre({'x': tensors['x_hc'][r], 'hc_fn': tensors['hc_ffn_fn'][r], 'hc_scale': tensors['hc_ffn_scale'][r], 'hc_base': tensors['hc_ffn_base'][r], 'x_mixed': x_mixed, 'post': post_t, 'comb': comb_t})
        x_norm = torch.zeros(_cpu_moe_T, _cpu_moe_D, dtype=torch.bfloat16)
        x_norm_i8 = torch.zeros(_cpu_moe_T, _cpu_moe_D, dtype=torch.int8)
        x_norm_scale = torch.zeros(_cpu_moe_T, 1, dtype=torch.float32)
        indices = torch.zeros(_cpu_moe_T, _cpu_moe_TOPK, dtype=torch.int32)
        weights = torch.zeros(_cpu_moe_T, _cpu_moe_TOPK, dtype=torch.float32)
        _cpu_gate_golden_gate_core({'x_mixed': x_mixed, 'norm_w': tensors['norm_w'][r], 'gate_w': tensors['gate_w'][r], 'gate_bias': tensors['gate_bias'][r], 'layer_id': tensors['layer_id'], 'num_tokens': tensors['num_tokens'], 'tid2eid': tensors['tid2eid'][r], 'input_ids': tensors['input_ids'][r], 'x_norm': x_norm, 'x_norm_i8': x_norm_i8, 'x_norm_scale': x_norm_scale, 'indices': indices, 'weights': weights})
        sh = torch.zeros(_cpu_moe_T, _cpu_moe_D, dtype=torch.bfloat16)
        _cpu_expert_shared_golden_expert_shared({'x_local_i8': x_norm_i8, 'x_local_scale_dq': x_norm_scale, 'num_tokens': tensors['num_tokens'], 'shared_w1': tensors['shared_w1'][r], 'shared_w1_scale': tensors['shared_w1_scale'][r], 'shared_w3': tensors['shared_w3'][r], 'shared_w3_scale': tensors['shared_w3_scale'][r], 'shared_w2': tensors['shared_w2'][r], 'shared_w2_scale': tensors['shared_w2_scale'][r], 'sh': sh})
        recv_x = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, _cpu_moe_D, dtype=torch.int8)
        recv_scale = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, dtype=torch.float32)
        recv_w = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, dtype=torch.float32)
        recv_r_route = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, dtype=torch.int32)
        recv_count = torch.zeros(_cpu_moe_N_LOCAL, 1, dtype=torch.int32)
        send_counts = torch.zeros(_cpu_moe_N_RANKS, _cpu_moe_N_RANKS, _cpu_moe_N_LOCAL, dtype=torch.int32)
        all_indices = []
        all_x_i8 = []
        all_scale = []
        all_weights = []
        for src in range(_cpu_moe_N_RANKS):
            src_x_mixed = torch.zeros(_cpu_moe_T, _cpu_moe_D, dtype=torch.bfloat16)
            src_post = torch.zeros(_cpu_moe_T, _cpu_moe_HC_MULT, dtype=torch.float32)
            src_comb = torch.zeros(_cpu_moe_T, _cpu_moe_HC_MULT * _cpu_moe_HC_MULT, dtype=torch.float32)
            _cpu_hc_pre_v2_golden_hc_pre({'x': tensors['x_hc'][src], 'hc_fn': tensors['hc_ffn_fn'][src], 'hc_scale': tensors['hc_ffn_scale'][src], 'hc_base': tensors['hc_ffn_base'][src], 'x_mixed': src_x_mixed, 'post': src_post, 'comb': src_comb})
            src_x_norm = torch.zeros(_cpu_moe_T, _cpu_moe_D, dtype=torch.bfloat16)
            src_x_norm_i8 = torch.zeros(_cpu_moe_T, _cpu_moe_D, dtype=torch.int8)
            src_x_norm_scale = torch.zeros(_cpu_moe_T, 1, dtype=torch.float32)
            src_indices = torch.zeros(_cpu_moe_T, _cpu_moe_TOPK, dtype=torch.int32)
            src_weights = torch.zeros(_cpu_moe_T, _cpu_moe_TOPK, dtype=torch.float32)
            _cpu_gate_golden_gate_core({'x_mixed': src_x_mixed, 'norm_w': tensors['norm_w'][src], 'gate_w': tensors['gate_w'][src], 'gate_bias': tensors['gate_bias'][src], 'layer_id': tensors['layer_id'], 'num_tokens': tensors['num_tokens'], 'tid2eid': tensors['tid2eid'][src], 'input_ids': tensors['input_ids'][src], 'x_norm': src_x_norm, 'x_norm_i8': src_x_norm_i8, 'x_norm_scale': src_x_norm_scale, 'indices': src_indices, 'weights': src_weights})
            all_indices.append(src_indices)
            all_x_i8.append(src_x_norm_i8)
            all_scale.append(src_x_norm_scale)
            all_weights.append(src_weights)
            for t in range(num_tokens):
                for k in range(_cpu_moe_TOPK):
                    eid = int(src_indices[t, k].item())
                    dst = eid // _cpu_moe_N_LOCAL
                    loc_e = eid % _cpu_moe_N_LOCAL
                    send_counts[src, dst, loc_e] += 1
        slot_offsets = torch.zeros(_cpu_moe_N_RANKS, _cpu_moe_N_LOCAL, dtype=torch.int32)
        running = torch.zeros(_cpu_moe_N_LOCAL, dtype=torch.int32)
        for src in range(_cpu_moe_N_RANKS):
            slot_offsets[src] = running.clone()
            running = running + send_counts[src, r]
        for e in range(_cpu_moe_N_LOCAL):
            recv_count[e, 0] = int(running[e].item())
        for src in range(_cpu_moe_N_RANKS):
            cursor = torch.zeros(_cpu_moe_N_LOCAL, dtype=torch.int32)
            for t in range(num_tokens):
                for k in range(_cpu_moe_TOPK):
                    eid = int(all_indices[src][t, k].item())
                    if eid // _cpu_moe_N_LOCAL != r:
                        continue
                    loc_e = eid % _cpu_moe_N_LOCAL
                    slot = int(slot_offsets[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    recv_x[loc_e, slot, :] = all_x_i8[src][t, :]
                    recv_scale[loc_e, slot] = float(all_scale[src][t, 0].item())
                    recv_w[loc_e, slot] = float(all_weights[src][t, k].item())
                    recv_r_route[loc_e, slot] = t * _cpu_moe_TOPK + k
        recv_y = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, _cpu_moe_D, dtype=torch.bfloat16)
        _cpu_expert_routed_golden_expert_routed({'recv_x': recv_x, 'recv_scale_dq': recv_scale, 'recv_weights': recv_w, 'recv_expert_count': recv_count, 'routed_w1': tensors['routed_w1'][r], 'routed_w1_scale': tensors['routed_w1_scale'][r], 'routed_w3': tensors['routed_w3'][r], 'routed_w3_scale': tensors['routed_w3_scale'][r], 'routed_w2': tensors['routed_w2'][r], 'routed_w2_scale': tensors['routed_w2_scale'][r], 'recv_y': recv_y})
        my_routes = []
        for t in range(num_tokens):
            for k in range(_cpu_moe_TOPK):
                eid = int(all_indices[r][t, k].item())
                dst = eid // _cpu_moe_N_LOCAL
                loc_e = eid % _cpu_moe_N_LOCAL
                my_routes.append((t, k, dst, loc_e))
        dst_recv_y = {}
        dst_recv_count = {}
        for dst in range(_cpu_moe_N_RANKS):
            d_recv_x = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, _cpu_moe_D, dtype=torch.int8)
            d_recv_scale = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, dtype=torch.float32)
            d_recv_w = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, dtype=torch.float32)
            d_recv_r_route = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, dtype=torch.int32)
            d_recv_count = torch.zeros(_cpu_moe_N_LOCAL, 1, dtype=torch.int32)
            d_slot_offsets = torch.zeros(_cpu_moe_N_RANKS, _cpu_moe_N_LOCAL, dtype=torch.int32)
            d_running = torch.zeros(_cpu_moe_N_LOCAL, dtype=torch.int32)
            for src in range(_cpu_moe_N_RANKS):
                d_slot_offsets[src] = d_running.clone()
                d_running = d_running + send_counts[src, dst]
            for e in range(_cpu_moe_N_LOCAL):
                d_recv_count[e, 0] = int(d_running[e].item())
            for src in range(_cpu_moe_N_RANKS):
                cursor = torch.zeros(_cpu_moe_N_LOCAL, dtype=torch.int32)
                for t in range(num_tokens):
                    for k in range(_cpu_moe_TOPK):
                        eid = int(all_indices[src][t, k].item())
                        if eid // _cpu_moe_N_LOCAL != dst:
                            continue
                        loc_e = eid % _cpu_moe_N_LOCAL
                        slot = int(d_slot_offsets[src, loc_e].item() + cursor[loc_e].item())
                        cursor[loc_e] += 1
                        d_recv_x[loc_e, slot, :] = all_x_i8[src][t, :]
                        d_recv_scale[loc_e, slot] = float(all_scale[src][t, 0].item())
                        d_recv_w[loc_e, slot] = float(all_weights[src][t, k].item())
                        d_recv_r_route[loc_e, slot] = t * _cpu_moe_TOPK + k
            d_recv_y = torch.zeros(_cpu_moe_N_LOCAL, _cpu_moe_RECV_MAX, _cpu_moe_D, dtype=torch.bfloat16)
            _cpu_expert_routed_golden_expert_routed({'recv_x': d_recv_x, 'recv_scale_dq': d_recv_scale, 'recv_weights': d_recv_w, 'recv_expert_count': d_recv_count, 'routed_w1': tensors['routed_w1'][dst], 'routed_w1_scale': tensors['routed_w1_scale'][dst], 'routed_w3': tensors['routed_w3'][dst], 'routed_w3_scale': tensors['routed_w3_scale'][dst], 'routed_w2': tensors['routed_w2'][dst], 'routed_w2_scale': tensors['routed_w2_scale'][dst], 'recv_y': d_recv_y})
            dst_recv_y[dst] = d_recv_y
            dst_recv_count[dst] = d_recv_count
        routed_y_buf_r = torch.zeros(_cpu_moe_N_ROUTES, _cpu_moe_D, dtype=torch.bfloat16)
        for (t, k, dst, loc_e) in my_routes:
            src_off = 0
            for s in range(r):
                src_off += int(send_counts[s, dst, loc_e].item())
            cursor = 0
            for (tt, kk, dd, ll) in my_routes:
                if (tt, kk) == (t, k):
                    break
                if dd == dst and ll == loc_e:
                    cursor += 1
            slot = src_off + cursor
            r_route = t * _cpu_moe_TOPK + k
            routed_y_buf_r[r_route, :] = dst_recv_y[dst][loc_e, slot, :]
        acc = sh.float().clone()
        for k in range(_cpu_moe_TOPK):
            for t in range(num_tokens):
                acc[t, :] += routed_y_buf_r[t * _cpu_moe_TOPK + k, :].float()
        ffn_out = acc.to(torch.bfloat16)
        x_next_r = torch.zeros(_cpu_moe_T, _cpu_moe_HC_MULT, _cpu_moe_D, dtype=torch.bfloat16)
        _cpu_hc_post_v2_golden_hc_post({'x': ffn_out, 'residual': tensors['x_hc'][r], 'post': post_t, 'comb': comb_t, 'y': x_next_r})
        x_next_out[r] = x_next_r
    tensors['x_next'][:] = x_next_out
import torch
from config import CASE_CONFIG, TOK_TILE as T, HC_MULT, D, EP_WORLD_SIZE
from prefill_attention_csa import golden_prefill_attention_csa

def compute_moe(inputs, scalars):
    values = {**inputs, **scalars}
    _cpu_moe_golden_moe(values)
    return {'x_next': values['x_next']}
_TOKEN_META = {'position_ids', 'ori_slot_mapping', 'cmp_sparse_indices', 'cmp_sparse_lens', 'cmp_slot_mapping', 'state_slot_mapping', 'idx_slot_mapping', 'inner_state_slot_mapping'}
_COMMON_PACKED_TO_CHILD = {'kv_cache': 'kv_cache', 'ori_block_table': 'ori_block_table', 'cmp_kv': 'cmp_kv', 'cmp_block_table': 'cmp_block_table', 'idx_kv_cache': 'idx_kv_cache', 'idx_kv_scale': 'idx_kv_scale', 'idx_block_table': 'idx_block_table'}
_MODE_PACKED_TO_CHILD = {'swa': {}, 'hca': {'hca_cmp_kv_state': 'cmp_kv_state', 'hca_cmp_score_state': 'cmp_score_state', 'hca_compress_state_block_table': 'compress_state_block_table'}, 'csa': {'csa_cmp_kv_state': 'cmp_kv_state', 'csa_cmp_score_state': 'cmp_score_state', 'csa_compress_state_block_table': 'compress_state_block_table', 'csa_inner_kv_state': 'inner_kv_state', 'csa_inner_score_state': 'inner_score_state', 'csa_inner_compress_state_block_table': 'inner_compress_state_block_table'}}
_MODE_ALIASES = {'swa': {'ori_block_table': 'block_table'}, 'hca': {'hca_cmp_wkv': 'cmp_wkv', 'hca_cmp_wgate': 'cmp_wgate', 'hca_cmp_ape': 'cmp_ape', 'hca_cmp_norm_w': 'cmp_norm_w', 'hca_cmp_slot_mapping': 'cmp_slot_mapping', 'hca_state_slot_mapping': 'state_slot_mapping'}, 'csa': {'csa_cmp_wkv': 'cmp_wkv', 'csa_cmp_wgate': 'cmp_wgate', 'csa_cmp_ape': 'cmp_ape', 'csa_cmp_norm_w': 'cmp_norm_w', 'csa_hadamard_idx': 'hadamard_idx', 'csa_idx_wq_b': 'idx_wq_b', 'csa_idx_wq_b_scale': 'idx_wq_b_scale', 'csa_weights_proj': 'idx_weights_proj', 'csa_inner_wkv': 'inner_wkv', 'csa_inner_wgate': 'inner_wgate', 'csa_inner_ape': 'inner_ape', 'csa_inner_norm_w': 'inner_norm_w', 'csa_cmp_slot_mapping': 'cmp_slot_mapping', 'csa_idx_slot_mapping': 'idx_slot_mapping', 'csa_state_slot_mapping': 'state_slot_mapping', 'csa_inner_state_slot_mapping': 'inner_state_slot_mapping'}}
_OUTPUTS = ('hca_cmp_kv_state', 'hca_cmp_score_state', 'csa_cmp_kv_state', 'csa_cmp_score_state', 'csa_inner_kv_state', 'csa_inner_score_state', 'kv_cache', 'cmp_kv', 'idx_kv_cache', 'idx_kv_scale', 'x_next')

def _tile(source: torch.Tensor, rank: int, base: int) -> torch.Tensor:
    result = torch.zeros((T, *source.shape[2:]), dtype=source.dtype)
    available = min(T, source.shape[1] - base)
    if available > 0:
        result[:available] = source[rank, base:base + available]
    return result

def compute(inputs, scalars):
    if set(scalars) != {'layer_id'}:
        raise ValueError(f'unexpected packed prefill scalars: {sorted(scalars)}')
    layer_id = int(scalars['layer_id'])
    ratios = CASE_CONFIG.compress_ratios
    if layer_id < 0 or layer_id >= len(ratios):
        raise ValueError(f'unsupported layer_id: {layer_id}')
    kind = {0: 'swa', 4: 'csa', 128: 'hca'}[ratios[layer_id]]
    result_inputs = {name: value.clone() for (name, value) in inputs.items()}
    ranks = int(result_inputs['x_hc'].shape[0])
    batch = int(result_inputs['chunk_lens'].shape[1])
    if ranks != EP_WORLD_SIZE:
        raise ValueError(f'unsupported official EP topology: EP{ranks}')
    if kind != 'csa':
        raise ValueError('the fixed Case selects CSA')
    attention_golden = golden_prefill_attention_csa
    chunk_lens = result_inputs['chunk_lens'][0]
    chunk_offsets = result_inputs['chunk_offsets'][0]
    x_next = torch.zeros_like(result_inputs['x_hc'])
    for request in range(batch):
        request_views = {}
        packed_state = {**_COMMON_PACKED_TO_CHILD, **_MODE_PACKED_TO_CHILD[kind]}
        for (packed, child) in packed_state.items():
            tensor = result_inputs[packed]
            count = tensor.shape[1] // batch
            request_views[child] = tensor[:, request * count:(request + 1) * count]
        chunk_len = int(chunk_lens[request])
        chunk_base = int(chunk_offsets[request])
        for tile_id in range((chunk_len + T - 1) // T):
            valid = min(T, chunk_len - tile_id * T)
            base = chunk_base + tile_id * T
            attention_output = torch.zeros(ranks, T, HC_MULT, D, dtype=torch.bfloat16)
            for rank in range(ranks):
                tile_inputs = {name: value[rank] for (name, value) in result_inputs.items()}
                for (packed, child) in _MODE_ALIASES[kind].items():
                    tile_inputs[child] = _tile(result_inputs[packed], rank, base) if child in _TOKEN_META else result_inputs[packed][rank]
                for (child, value) in request_views.items():
                    tile_inputs[child] = value[rank]
                for name in _TOKEN_META:
                    if name in result_inputs:
                        tile_inputs[name] = _tile(result_inputs[name], rank, base)
                tile_inputs['x_hc'] = _tile(result_inputs['x_hc'], rank, base)
                tile_inputs['num_tokens'] = valid
                tile_inputs['x_out'] = attention_output[rank]
                attention_golden(tile_inputs)
            moe_inputs = dict(result_inputs)
            moe_inputs['x_hc'] = attention_output
            moe_inputs['input_ids'] = torch.cat([_tile(result_inputs['input_ids'], rank, base).unsqueeze(0) for rank in range(ranks)])
            tile_output = compute_moe(moe_inputs, {'layer_id': layer_id, 'num_tokens': valid})['x_next']
            x_next[:, base:base + valid] = tile_output[:, :valid]
    result_inputs['x_next'] = x_next
    return {name: result_inputs[name] for name in _OUTPUTS}
