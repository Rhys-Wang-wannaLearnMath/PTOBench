import torch

from config import CASE_CONFIG, TOK_TILE as T, HC_MULT, D, EP_WORLD_SIZE

from prefill_attention_swa import golden_prefill_attention_swa

from moe import golden_moe

def compute_moe(inputs, scalars):
    values = {**inputs, **scalars}
    golden_moe(values)
    return {"x_next": values["x_next"]}

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
    if ranks != EP_WORLD_SIZE:
        raise ValueError(f"unsupported official EP topology: EP{ranks}")
    if kind != "swa":
        raise ValueError("the fixed Case selects SWA")
    attention_golden = golden_prefill_attention_swa

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
