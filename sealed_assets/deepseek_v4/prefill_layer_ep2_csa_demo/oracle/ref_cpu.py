"""Case-local External CPU Oracle.

This regular file owns the full CPU reference math for its Case.  It was
materialized from the previously reviewed benchmark helper during the
2026-07-30 golden_data de-duplication migration, with import forwarding
removed so the Case remains independently auditable.
"""

import torch
import torch.nn.functional as F

import prefill_attention_csa as _attention_csa

from config import (
    CASE_CONFIG, TOKEN_TILE, HC_MULT, D, MIX_HC, HC_DIM, MOE_INTER, TOPK,
    N_LOCAL, EPS, INT8_SCALE_MAX, INT8_AMAX_EPS,
)

T, LANES = TOKEN_TILE, HC_MULT
MIX, INTER = MIX_HC, MOE_INTER
LOCAL = N_LOCAL
NORM_EPS, HC_EPS = EPS, CASE_CONFIG.hc_eps
INT8_MAX, INT8_EPS = INT8_SCALE_MAX, INT8_AMAX_EPS
SWIGLU_LIMIT, ROUTE_SCALE = CASE_CONFIG.swiglu_limit, CASE_CONFIG.routed_scaling_factor


def _bf16_rint(value):
    rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
    return rounded.view(torch.float32).to(torch.bfloat16)


def _hc_pre(source, fn, scale, base):
    flat = source.reshape(T, HC_DIM).float()
    square_sum = torch.zeros(T, 1)
    for offset in range(0, HC_DIM, 512):
        part = flat[:, offset:offset + 512]
        square_sum += (part * part).sum(1, keepdim=True)
    inv_rms = torch.rsqrt(square_sum / HC_DIM + HC_EPS)
    columns = []
    for mix in range(MIX):
        dot = torch.zeros(T, 1)
        for offset in range(0, HC_DIM, 256):
            dot += (flat[:, offset:offset + 256] * fn[mix:mix + 1, offset:offset + 256].float()).sum(1, keepdim=True)
        columns.append(dot * inv_rms)
    mixed = torch.cat(columns, 1)
    scale, base = scale.float(), base.float()
    pre = torch.sigmoid(mixed[:, :4] * scale[0] + base[:4]) + HC_EPS
    post = 2 * torch.sigmoid(mixed[:, 4:8] * scale[1] + base[4:8])
    comb = (mixed[:, 8:] * scale[2] + base[8:]).reshape(T, LANES, LANES)
    comb = torch.softmax(comb, -1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    output = torch.zeros(T, D)
    for lane in range(LANES):
        output += source[:, lane].float() * pre[:, lane:lane + 1]
    return _bf16_rint(output), post, comb.reshape(T, LANES * LANES)


def _quantize_rows(value):
    rows = value.float().reshape(-1, value.shape[-1])
    amax = rows.abs().amax(-1, keepdim=True).clamp_min(INT8_EPS)
    quant_scale = INT8_MAX / amax
    quantized = torch.round(rows * quant_scale).to(torch.int32).to(torch.float16).to(torch.int8)
    return quantized.reshape_as(value), (1.0 / quant_scale).reshape(*value.shape[:-1], 1)


def _gate(x_mixed, norm_w, gate_w, gate_bias, tid2eid, input_ids, layer_id, num_tokens):
    x = x_mixed.float()
    inv_rms = torch.rsqrt((x * x).sum(-1, keepdim=True) / D + NORM_EPS)
    x_norm = (x * inv_rms * norm_w.float().reshape(1, D)).to(torch.bfloat16)
    x_i8, x_scale = _quantize_rows(x_norm)
    logits = x_norm.float() @ gate_w.float().T
    scores = (logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))).sqrt()
    if layer_id < CASE_CONFIG.num_hash_layers:
        indices = tid2eid[input_ids.flatten().long()]
    else:
        indices = torch.argsort(-(scores + gate_bias.float()), dim=-1, stable=True)[:, :TOPK]
    selected = scores.gather(1, indices.long())
    weights = selected / selected.sum(-1, keepdim=True) * ROUTE_SCALE
    if num_tokens < T:
        x_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0
    return x_i8, x_scale, indices.to(torch.int32), weights.float()


def _dequant(weight, scale):
    return weight.float() * scale.float().unsqueeze(-1)


def _shared(x_i8, x_scale, inputs, rank):
    x = x_i8.float() * x_scale.float()
    w1 = _dequant(inputs["shared_w1"][rank], inputs["shared_w1_scale"][rank])
    w3 = _dequant(inputs["shared_w3"][rank], inputs["shared_w3_scale"][rank])
    w2 = _dequant(inputs["shared_w2"][rank], inputs["shared_w2_scale"][rank])
    gate, up = x @ w1.T, x @ w3.T
    if SWIGLU_LIMIT > 0.0:
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    hidden_i8, hidden_scale = _quantize_rows(F.silu(gate) * up)
    return ((hidden_i8.float() * hidden_scale) @ w2.T).to(torch.bfloat16)


def _routed(recv_x, recv_scale, recv_weight, counts, inputs, rank):
    # Preserve the official host-reference allocation/layout boundary: the
    # complete local expert banks are dequantized before slicing an expert.
    # Dequantizing a slice separately can select a different CPU GEMM packing
    # path and changes a few BF16 tie cases despite equivalent real arithmetic.
    w1 = _dequant(inputs["routed_w1"][rank], inputs["routed_w1_scale"][rank])
    w3 = _dequant(inputs["routed_w3"][rank], inputs["routed_w3_scale"][rank])
    w2 = _dequant(inputs["routed_w2"][rank], inputs["routed_w2_scale"][rank])
    output = torch.zeros(LOCAL, recv_x.shape[1], D)
    for expert in range(LOCAL):
        rows = int(counts[expert, 0])
        if rows == 0:
            continue
        x = recv_x[expert, :rows].float() * recv_scale[expert, :rows, None]
        gate, up = x @ w1[expert].T, x @ w3[expert].T
        if SWIGLU_LIMIT > 0.0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        hidden_i8, hidden_scale = _quantize_rows(F.silu(gate) * up)
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
    if set(scalars) != {"layer_id", "num_tokens"}:
        raise ValueError(f"unexpected MoE scalars: {sorted(scalars)}")
    layer_id = int(scalars["layer_id"])
    num_tokens = max(0, min(T, int(scalars["num_tokens"])))
    ranks = int(inputs["x_hc"].shape[0])
    recv_max = ranks * T
    if ranks not in (2, 4, 8):
        raise ValueError(f"unsupported official EP topology: {ranks}")

    mixed, post, comb, x_i8, x_scale, indices, weights, shared = ([] for _ in range(8))
    for rank in range(ranks):
        rank_mixed, rank_post, rank_comb = _hc_pre(
            inputs["x_hc"][rank], inputs["hc_ffn_fn"][rank],
            inputs["hc_ffn_scale"][rank], inputs["hc_ffn_base"][rank],
        )
        rank_i8, rank_scale, rank_indices, rank_weights = _gate(
            rank_mixed, inputs["norm_w"][rank], inputs["gate_w"][rank],
            inputs["gate_bias"][rank], inputs["tid2eid"][rank],
            inputs["input_ids"][rank], layer_id, num_tokens,
        )
        mixed.append(rank_mixed); post.append(rank_post); comb.append(rank_comb)
        x_i8.append(rank_i8); x_scale.append(rank_scale)
        indices.append(rank_indices); weights.append(rank_weights)
        shared.append(_shared(rank_i8, rank_scale, inputs, rank))

    counts = torch.zeros(ranks, ranks, LOCAL, dtype=torch.int32)
    for src in range(ranks):
        for token in range(num_tokens):
            for slot in range(TOPK):
                eid = int(indices[src][token, slot])
                counts[src, eid // LOCAL, eid % LOCAL] += 1

    routed_by_dst, offsets_by_dst = [], []
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
                eid = int(indices[src][token, slot]); dst, expert = eid // LOCAL, eid % LOCAL
                packed = int(offsets_by_dst[dst][src, expert] + cursor[dst, expert])
                cursor[dst, expert] += 1
                route_buffer[token * TOPK + slot] = routed_by_dst[dst][expert, packed]
        accumulated = shared[src].float().clone()
        for slot in range(TOPK):
            for token in range(num_tokens):
                accumulated[token] += route_buffer[token * TOPK + slot].float()
        outputs[src] = _hc_post(accumulated.to(torch.bfloat16), inputs["x_hc"][src], post[src], comb[src])
    return {"x_next": outputs}


def compute_moe(inputs, scalars):
    global T
    tokens = int(inputs["x_hc"].shape[1])
    prior_tokens = T
    try:
        T = tokens
        return _compute_integrated_moe(inputs, scalars)
    finally:
        T = prior_tokens




# packed prefill-layer CPU semantics.
import torch

T = TOKEN_TILE
_TOKEN_META = {
    "position_ids", "ori_slot_mapping", "cmp_sparse_indices", "cmp_sparse_lens",
    "cmp_slot_mapping", "state_slot_mapping", "idx_slot_mapping",
    "inner_state_slot_mapping",
}
_COMMON_PACKED_TO_CHILD = {
    "kv_cache": "kv_cache",
    "ori_block_table": "ori_block_table",
    "cmp_kv": "cmp_kv",
    "cmp_block_table": "cmp_block_table",
    "idx_kv_cache": "idx_kv_cache",
    "idx_kv_scale": "idx_kv_scale",
    "idx_block_table": "idx_block_table",
}
_MODE_PACKED_TO_CHILD = {
    "swa": {},
    "hca": {
        "hca_cmp_kv_state": "cmp_kv_state",
        "hca_cmp_score_state": "cmp_score_state",
        "hca_compress_state_block_table": "compress_state_block_table",
    },
    "csa": {
        "csa_cmp_kv_state": "cmp_kv_state",
        "csa_cmp_score_state": "cmp_score_state",
        "csa_compress_state_block_table": "compress_state_block_table",
        "csa_inner_kv_state": "inner_kv_state",
        "csa_inner_score_state": "inner_score_state",
        "csa_inner_compress_state_block_table": "inner_compress_state_block_table",
    },
}
_MODE_ALIASES = {
    "swa": {"ori_block_table": "block_table"},
    "hca": {
        "hca_cmp_wkv": "cmp_wkv",
        "hca_cmp_wgate": "cmp_wgate",
        "hca_cmp_ape": "cmp_ape",
        "hca_cmp_norm_w": "cmp_norm_w",
        "hca_cmp_slot_mapping": "cmp_slot_mapping",
        "hca_state_slot_mapping": "state_slot_mapping",
    },
    "csa": {
        "csa_cmp_wkv": "cmp_wkv",
        "csa_cmp_wgate": "cmp_wgate",
        "csa_cmp_ape": "cmp_ape",
        "csa_cmp_norm_w": "cmp_norm_w",
        "csa_hadamard_idx": "hadamard_idx",
        "csa_idx_wq_b": "idx_wq_b",
        "csa_idx_wq_b_scale": "idx_wq_b_scale",
        "csa_weights_proj": "idx_weights_proj",
        "csa_inner_wkv": "inner_wkv",
        "csa_inner_wgate": "inner_wgate",
        "csa_inner_ape": "inner_ape",
        "csa_inner_norm_w": "inner_norm_w",
        "csa_cmp_slot_mapping": "cmp_slot_mapping",
        "csa_idx_slot_mapping": "idx_slot_mapping",
        "csa_state_slot_mapping": "state_slot_mapping",
        "csa_inner_state_slot_mapping": "inner_state_slot_mapping",
    },
}
_OUTPUTS = (
    "hca_cmp_kv_state", "hca_cmp_score_state",
    "csa_cmp_kv_state", "csa_cmp_score_state",
    "csa_inner_kv_state", "csa_inner_score_state",
    "kv_cache", "cmp_kv", "idx_kv_cache", "idx_kv_scale", "x_next",
)


def _load_attention_reference(kind: str, ranks: int):
    del ranks
    if kind != "csa":
        raise ValueError(f"fixed Case supports only CSA attention, got {kind!r}")
    return _attention_csa


def _tile(source: torch.Tensor, rank: int, base: int) -> torch.Tensor:
    result = torch.zeros((T, *source.shape[2:]), dtype=source.dtype)
    available = min(T, source.shape[1] - base)
    if available > 0:
        result[:available] = source[rank, base:base + available]
    return result


def compute(inputs, scalars):
    if set(scalars) != {"layer_id"}:
        raise ValueError(f"unexpected packed prefill scalars: {sorted(scalars)}")
    layer_id = int(scalars["layer_id"])
    ratios = CASE_CONFIG.compress_ratios
    if layer_id < 0 or layer_id >= len(ratios):
        raise ValueError(f"unsupported layer_id: {layer_id}")
    kind = {0: "swa", 4: "csa", 128: "hca"}[ratios[layer_id]]

    result_inputs = {name: value.clone() for name, value in inputs.items()}
    ranks = int(result_inputs["x_hc"].shape[0])
    batch = int(result_inputs["chunk_lens"].shape[1])
    if ranks not in (2, 4, 8):
        raise ValueError(f"unsupported official EP topology: EP{ranks}")
    attention = _load_attention_reference(kind, ranks)
    attention_golden = getattr(attention, f"golden_prefill_attention_{kind}")

    chunk_lens = result_inputs["chunk_lens"][0]
    chunk_offsets = result_inputs["chunk_offsets"][0]
    x_next = torch.zeros_like(result_inputs["x_hc"])

    for request in range(batch):
        request_views = {}
        packed_state = {**_COMMON_PACKED_TO_CHILD, **_MODE_PACKED_TO_CHILD[kind]}
        for packed, child in packed_state.items():
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
                tile_inputs = {name: value[rank] for name, value in result_inputs.items()}
                for packed, child in _MODE_ALIASES[kind].items():
                    tile_inputs[child] = (
                        _tile(result_inputs[packed], rank, base)
                        if child in _TOKEN_META else result_inputs[packed][rank]
                    )
                for child, value in request_views.items():
                    tile_inputs[child] = value[rank]
                for name in _TOKEN_META:
                    if name in result_inputs:
                        tile_inputs[name] = _tile(result_inputs[name], rank, base)
                tile_inputs["x_hc"] = _tile(result_inputs["x_hc"], rank, base)
                tile_inputs["num_tokens"] = valid
                tile_inputs["x_out"] = attention_output[rank]
                attention_golden(tile_inputs)

            moe_inputs = dict(result_inputs)
            moe_inputs["x_hc"] = attention_output
            moe_inputs["input_ids"] = torch.cat(
                [_tile(result_inputs["input_ids"], rank, base).unsqueeze(0) for rank in range(ranks)]
            )
            tile_output = compute_moe(
                moe_inputs,
                {"layer_id": layer_id, "num_tokens": valid},
            )["x_next"]
            x_next[:, base:base + valid] = tile_output[:, :valid]

    result_inputs["x_next"] = x_next
    return {name: result_inputs[name] for name in _OUTPUTS}
