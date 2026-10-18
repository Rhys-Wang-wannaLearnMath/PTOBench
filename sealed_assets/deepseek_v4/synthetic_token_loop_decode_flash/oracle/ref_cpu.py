"""Independent analytic CPU Oracle for the fixed zero-weight Flash session.

Zero embedding, projections, and normalization gamma keep hidden states zero.
Positive RMS/HC epsilon, quantization floors, and positive softmax/Sinkhorn
denominators keep every intermediate finite. Both hash and scored MoE routes
mix zero expert outputs. Greedy sampling selects the first tied maximum, zero.
All projected cache/state writes are zero. The only nonzero output is the
indexer dequantization scale at the 128-token Prefill compression boundaries.
Decode at position 128 has no new ratio-4 or ratio-128 compression boundary.
This proof applies only to the sealed fixed-zero weights and initial history.
"""

import torch
from config import (
    BLOCK_SIZE, CASE_CONFIG, CSA_COMPRESS_RATIO, CSA_NUM_LAYERS, IDX_KV_BLOCK_NUM,
    INT8_AMAX_EPS, INT8_SCALE_MAX, N_RANKS, NUM_TOKENS, PREFILL_TOKENS,
    START_POS, T,
)


OUTPUT_NAMES = (
    "pre_hc_hidden_out", "hidden_out", "logits", "sampled_ids", "kv_cache",
    "cmp_kv", "hca_compress_state", "csa_compress_state",
    "csa_inner_compress_state", "idx_kv_cache", "idx_kv_scale",
)
STATE_NAMES = OUTPUT_NAMES[4:]
ZERO_WEIGHT_NAMES = (
    "embed_weight", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv", "attn_sink",
    "wo_a", "wo_b", "wo_b_scale", "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape",
    "hca_cmp_norm_w", "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape",
    "csa_cmp_norm_w", "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale",
    "csa_weights_proj", "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape",
    "csa_inner_norm_w", "freqs_cos", "freqs_sin", "hc_ffn_fn", "hc_ffn_scale",
    "hc_ffn_base", "norm_w", "gate_w", "gate_bias", "routed_w1",
    "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2",
    "routed_w2_scale", "shared_w1", "shared_w1_scale", "shared_w3",
    "shared_w3_scale", "shared_w2", "shared_w2_scale", "hc_head_fn",
    "hc_head_scale", "hc_head_base", "final_norm_w", "lm_head_weight",
)


def expected_scale(values):
    shape = (N_RANKS, CSA_NUM_LAYERS * IDX_KV_BLOCK_NUM, BLOCK_SIZE, 1, 1)
    result = torch.zeros(shape, dtype=torch.float32)
    table = values["idx_block_table"]
    if table.ndim == 3:
        table = table[:, 0, :]
    compressed = torch.arange(PREFILL_TOKENS // CSA_COMPRESS_RATIO, dtype=torch.int64)
    slots = table[:, compressed // BLOCK_SIZE].to(torch.int64) * BLOCK_SIZE + compressed % BLOCK_SIZE
    scale = torch.reciprocal(
        torch.tensor(INT8_SCALE_MAX, dtype=torch.float32)
        / torch.tensor(INT8_AMAX_EPS, dtype=torch.float32)
    )
    rows = result.view(N_RANKS, CSA_NUM_LAYERS, IDX_KV_BLOCK_NUM * BLOCK_SIZE)
    for rank in range(N_RANKS):
        rows[rank, :, slots[rank]] = scale
    return result


def compute(values):
    if int(values["num_tokens"]) != NUM_TOKENS:
        raise ValueError("analytic Oracle requires the fixed active token count")
    for name in ZERO_WEIGHT_NAMES:
        if torch.count_nonzero(values[name]).item():
            raise ValueError(f"analytic Oracle requires zero weights: {name}")
    for name in STATE_NAMES:
        initial = values[name]
        if name == "idx_kv_scale" and START_POS:
            if not torch.equal(initial, expected_scale(values)):
                raise ValueError("Decode history differs from independent Prefill math")
        elif torch.count_nonzero(initial).item():
            raise ValueError(f"analytic Oracle requires initial zero state: {name}")
    if START_POS and (START_POS + 1) % CSA_COMPRESS_RATIO == 0:
        raise ValueError("Decode Oracle binding unexpectedly crosses a compression boundary")
    if START_POS:
        if torch.count_nonzero(values["input_ids"]).item():
            raise ValueError("Decode input must be the independent greedy token zero")
        if int(values["position_ids"][0, 0]) != START_POS:
            raise ValueError("Decode position differs from fixed history")
    else:
        expected_ids = torch.arange(PREFILL_TOKENS).expand(N_RANKS, -1)
        if not torch.equal(values["input_ids"], expected_ids):
            raise ValueError("Prefill input differs from fixed session tokens")
    outputs = {name: torch.zeros_like(values[name]) for name in OUTPUT_NAMES}
    if outputs["hidden_out"].shape != (N_RANKS, T, CASE_CONFIG.hidden_size):
        raise ValueError("Oracle hidden ABI differs")
    outputs["idx_kv_scale"] = expected_scale(values)
    return outputs
