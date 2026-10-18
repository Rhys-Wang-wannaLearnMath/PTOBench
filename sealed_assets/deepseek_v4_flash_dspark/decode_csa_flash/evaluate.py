"""Case-local ABI, evaluator, and comparator implementation for decode CSA."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common
import workflow


def _read_policy(case_dir: Path, case: dict) -> dict:
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}:
        raise ValueError("invalid comparator top-level fields")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("unsupported comparator schema")
    names = case["correctness"]["outputs"]
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(names) or len(outputs) != len(names):
        raise ValueError("comparator output coverage disagrees with contract")
    return policy


def _number(value: object, field: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"invalid comparator {field}")
    value = float(value)
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"invalid comparator {field}")
    return value


def _ratio_reldiff(rule: dict):
    diff_thd = _number(rule["diff_thd"], "diff_thd", positive=True)
    pct_thd = _number(rule["pct_thd"], "pct_thd")
    if pct_thd > 1:
        raise ValueError("invalid comparator pct_thd")
    max_diff_hd = _number(rule["max_diff_hd"], "max_diff_hd", positive=True)

    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, "output shape mismatch"
        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all()):
            return False, "nonfinite output"
        diff_abs = (actual_f - expected_f).abs()
        floor = (1.0 / (1 << 14)) / diff_thd
        denominator = torch.maximum(
            torch.maximum(actual_f.abs(), expected_f.abs()),
            torch.full_like(actual_f, floor),
        ) + 1e-9
        relative = torch.where(diff_abs < diff_thd, diff_abs, diff_abs / denominator)
        bad = relative > diff_thd
        error_count = int(bad.sum().item())
        threshold = round(pct_thd * actual_f.numel())
        worst = float(relative[bad].max().item()) if error_count else 0.0
        passed = error_count <= threshold and worst < max_diff_hd
        if passed:
            return True, ""
        return False, (
            f"ratio_reldiff: errors={error_count}/{actual_f.numel()} threshold={threshold} "
            f"worst={worst:.6g} max_diff_hd={max_diff_hd}"
        )

    return compare


def _mapped_pool_ratio_allclose(rule: dict):
    required = {
        "method", "name", "mapping", "mapping_shape", "block_size", "rtol",
        "atol", "max_error_ratio", "unmapped_policy",
    }
    if set(rule) != required or rule["name"] != "mapped_pool_ratio_allclose":
        raise ValueError("invalid mapped_pool_ratio_allclose fields")
    if type(rule["mapping"]) is not str or not rule["mapping"]:
        raise ValueError("invalid comparator mapping")
    mapping_shape = tuple(rule["mapping_shape"])
    if not mapping_shape or any(type(dim) is not int or dim <= 0 for dim in mapping_shape):
        raise ValueError("invalid comparator mapping_shape")
    block_size = rule["block_size"]
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("invalid comparator block_size")
    policy_rtol = _number(rule["rtol"], "rtol")
    policy_atol = _number(rule["atol"], "atol")
    policy_ratio = _number(rule["max_error_ratio"], "max_error_ratio")
    if policy_ratio > 1 or rule["unmapped_policy"] != "exact":
        raise ValueError("invalid mapped pool policy")
    mapping_name = rule["mapping"]

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, "output shape mismatch"
        if actual.ndim < 2 or actual.shape[1] != block_size:
            return False, "expected block-major pool layout"
        inputs = kwargs.get("inputs", {})
        mapping = inputs.get(mapping_name)
        if mapping is None or mapping.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            return False, "mapping input missing or non-integer"
        if tuple(mapping.shape) != mapping_shape:
            return False, "mapping shape mismatch"
        mapping = mapping.detach().cpu().to(torch.int64).reshape(-1)
        rows = actual.shape[0] * block_size
        if bool((mapping < -1).any()) or bool((mapping >= rows).any()):
            return False, "mapping contains invalid physical row"
        valid = mapping >= 0
        valid_mapping = mapping[valid]
        if valid_mapping.numel() and torch.unique(valid_mapping).numel() != valid_mapping.numel():
            return False, "mapping contains duplicate physical row"
        actual_rows = actual.detach().cpu().reshape(rows, -1).to(torch.float32)
        expected_rows = expected.detach().cpu().reshape(rows, -1).to(torch.float32)
        written = torch.zeros(rows, dtype=torch.bool)
        written[valid_mapping] = True
        if bool(((actual_rows != expected_rows).any(dim=1) & ~written).any()):
            return False, "unmapped physical pool row changed"
        if not bool(written.any()):
            return True, ""
        mapped_actual = actual_rows[written]
        mapped_expected = expected_rows[written]
        if not bool(torch.isfinite(mapped_actual).all() and torch.isfinite(mapped_expected).all()):
            return False, "nonfinite mapped pool value"
        tolerance = policy_atol + policy_rtol * mapped_expected.abs()
        errors = int(((mapped_actual - mapped_expected).abs() > tolerance).sum().item())
        threshold = round(policy_ratio * mapped_actual.numel())
        if errors <= threshold:
            return True, ""
        return False, f"mapped_pool_ratio_allclose: errors={errors}/{mapped_actual.numel()} threshold={threshold}"

    return compare


def make_compare(case_dir: Path, case: dict):
    policy = _read_policy(case_dir, case)
    callbacks = {}
    for name, rule in policy["outputs"].items():
        if not isinstance(rule, dict) or rule.get("method") != "custom":
            raise ValueError("unsupported comparator method")
        if name == "x_out":
            if set(rule) != {"method", "name", "diff_thd", "pct_thd", "max_diff_hd", "tail_policy"}:
                raise ValueError("invalid ratio_reldiff fields")
            if rule["name"] != "ratio_reldiff" or rule["tail_policy"] != "reject_undeclared":
                raise ValueError("invalid ratio_reldiff policy")
            callbacks[name] = _ratio_reldiff(rule)
        elif name == "kv_cache":
            callbacks[name] = _mapped_pool_ratio_allclose(rule)
        else:
            raise ValueError("undeclared custom comparator")
    if set(callbacks) != set(case["correctness"]["outputs"]):
        raise ValueError("callback output coverage disagrees with contract")
    return callbacks


def _config(case_dir: Path, case: dict):
    identity = case["case_id"].replace(".", "_")
    config = common.load_isolated_module(case_dir / case["config"], f"config_{identity}")
    if config.CASE_ID != case["case_id"] or config.CASE_CONFIG is not config.FLASH:
        raise ValueError("public config identity disagrees with Case")
    return config


def build_specs(fixture: Path, config):
    model = config.CASE_CONFIG
    t = config.DECODE_TOKENS
    b = config.DECODE_BATCH
    d = model.hidden_size
    h = model.num_attention_heads
    hd = model.head_dim
    ql = model.q_lora_rank
    idxh = model.index_head_dim
    idx_heads = model.index_n_heads
    block_size = config.BLOCK_SIZE
    compress_ratio = config.COMPRESS_RATIO
    main_out_dim = config.MAIN_OUT_DIM
    main_state_dim = config.MAIN_STATE_DIM
    inner_out_dim = config.INNER_OUT_DIM
    inner_state_dim = config.INNER_STATE_DIM
    bindings = [
        ("x_hc", (t, config.HC_MULT, d), torch.float32),
        ("hc_attn_fn", (config.MIX_HC, config.HC_DIM), torch.float32),
        ("hc_attn_scale", (model.num_hash_layers,), torch.float32), ("hc_attn_base", (config.MIX_HC,), torch.float32),
        ("attn_norm_w", (d,), torch.bfloat16), ("wq_a", (d, ql), torch.bfloat16),
        ("wq_b", (ql, h * hd), torch.int8), ("wq_b_scale", (h * hd,), torch.float32),
        ("wkv", (d, hd), torch.bfloat16), ("gamma_cq", (ql,), torch.bfloat16),
        ("gamma_ckv", (hd,), torch.bfloat16),
        ("freqs_cos", (model.max_position_embeddings, model.qk_rope_head_dim), torch.bfloat16),
        ("freqs_sin", (model.max_position_embeddings, model.qk_rope_head_dim), torch.bfloat16),
        ("cmp_wkv", (main_out_dim, d), torch.bfloat16),
        ("cmp_wgate", (main_out_dim, d), torch.bfloat16),
        ("cmp_ape", (compress_ratio, main_out_dim), torch.float32),
        ("cmp_norm_w", (hd,), torch.bfloat16),
        ("compress_state", (config.MAIN_STATE_PHYSICAL_BLOCKS, config.MAIN_STATE_BLOCK_SIZE, main_state_dim), torch.float32),
        ("compress_state_block_table", (b, config.MAIN_STATE_MAX_BLOCKS), torch.int32),
        ("idx_wq_b", (ql, idx_heads * idxh), torch.int8),
        ("idx_wq_b_scale", (idx_heads * idxh,), torch.float32), ("weights_proj", (d, idx_heads), torch.bfloat16),
        ("hadamard_idx", (idxh, idxh), torch.bfloat16), ("inner_wkv", (inner_out_dim, d), torch.bfloat16),
        ("inner_wgate", (inner_out_dim, d), torch.bfloat16),
        ("inner_ape", (compress_ratio, inner_out_dim), torch.float32),
        ("inner_norm_w", (idxh,), torch.bfloat16),
        ("inner_compress_state", (config.INNER_STATE_PHYSICAL_BLOCKS, config.INNER_STATE_BLOCK_SIZE, inner_state_dim), torch.float32),
        ("inner_compress_state_block_table", (b, config.INNER_STATE_MAX_BLOCKS), torch.int32),
        ("kv_cache", (config.ORI_BLOCK_NUM, block_size, 1, hd), torch.bfloat16),
        ("cmp_kv", (config.CMP_BLOCK_NUM, block_size, 1, hd), torch.bfloat16),
        ("cmp_block_table", (b, config.CMP_MAX_BLOCKS), torch.int32),
        ("idx_kv_cache", (config.IDX_CACHE_BLOCK_NUM, block_size, 1, idxh), torch.int8),
        ("idx_kv_scale", (config.IDX_CACHE_BLOCK_NUM, block_size, 1, 1), torch.float32),
        ("idx_block_table", (b, config.IDX_CACHE_MAX_BLOCKS), torch.int32), ("ori_slot_mapping", (t,), torch.int64),
        ("window_swa_indices", (t, model.sliding_window), torch.int32), ("window_swa_lens", (t,), torch.int32),
        ("cmp_slot_mapping", (t,), torch.int64), ("idx_slot_mapping", (t,), torch.int64),
        ("state_slot_mapping", (t,), torch.int64), ("inner_state_slot_mapping", (t,), torch.int64),
        ("position_ids", (t,), torch.int32), ("kv_seq_lens", (b,), torch.int32),
        ("attn_sink", (h,), torch.float32), ("wo_a", (model.o_groups, model.o_lora_rank, h * hd // model.o_groups), torch.bfloat16),
        ("wo_b", (d, model.o_groups * model.o_lora_rank), torch.int8), ("wo_b_scale", (d,), torch.float32),
        ("x_out", (t, config.HC_MULT, d), torch.float32),
    ]
    specs = []
    for name, shape, dtype in bindings:
        init = None if name == "x_out" else torch.load(fixture / "in" / f"{name}.pt", map_location="cpu", weights_only=True)
        specs.append(TensorSpec(name, list(shape), dtype, init_value=init))
    return specs


def _make_program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    outputs = oracle.compute(values)
    expected = set(case_outputs := {"x_out", "kv_cache"})
    if set(outputs) != expected:
        raise ValueError("Oracle output coverage disagrees with contract")
    for name in expected:
        values[name].copy_(outputs[name])


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"ref_{identity}"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"oracle_{identity}"),
        make_program=_make_program,
        make_compare=lambda: make_compare(case_dir, case),
        run=run, oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"candidate_{identity}"),
        make_program=_make_program, make_compare=lambda: make_compare(case_dir, case), run=run,
    )
