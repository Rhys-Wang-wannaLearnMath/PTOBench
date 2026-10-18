"""Case-local ABI, evaluator, and comparator implementation for decode HCA."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common
import workflow


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate comparator key: {key}")
        result[key] = value
    return result


def _read_policy(case_dir: Path, case: dict) -> dict:
    policy = json.loads(
        (case_dir / case["comparator"]).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
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
    max_diff_hd = _number(rule["max_diff_hd"], "max_diff_hd", positive=True)
    if pct_thd > 1:
        raise ValueError("invalid comparator pct_thd")

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


def _mapped_pool_ratio_allclose(rule: dict, config):
    mapping_name = rule["mapping_name"]
    mapping_shape = rule["mapping_shape"]
    block_size = rule["block_size"]
    leading_rank_axis = rule["leading_rank_axis"]
    if not isinstance(mapping_name, str) or not mapping_name:
        raise ValueError("invalid mapping_name")
    if (not isinstance(mapping_shape, list) or not mapping_shape
            or any(type(dim) is not int or dim <= 0 for dim in mapping_shape)):
        raise ValueError("invalid mapping_shape")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("invalid block_size")
    if type(leading_rank_axis) is not bool:
        raise ValueError("invalid leading_rank_axis")
    if (mapping_name != "ori_slot_mapping"
            or mapping_shape != [config.DECODE_TOKENS]
            or block_size != config.BLOCK_SIZE
            or leading_rank_axis):
        raise ValueError("mapped-pool binding disagrees with fixed HCA ABI")
    policy_atol = _number(rule["atol"], "atol")
    policy_rtol = _number(rule["rtol"], "rtol")
    policy_ratio = _number(rule["max_error_ratio"], "max_error_ratio")
    if policy_ratio > 1:
        raise ValueError("invalid max_error_ratio")
    mapping_shape = tuple(mapping_shape)
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, "output shape mismatch"
        if actual.ndim < 2 or actual.shape[1] != block_size:
            return False, "expected unranked block-major pool layout"
        mapping = kwargs.get("inputs", {}).get(mapping_name)
        if not isinstance(mapping, torch.Tensor):
            return False, f"missing mapping input {mapping_name}"
        if mapping.dtype not in integer_dtypes or tuple(mapping.shape) != mapping_shape:
            return False, "mapping dtype or shape mismatch"
        row_count = actual.shape[0] * block_size
        actual_rows = actual.detach().cpu().reshape(row_count, -1)
        expected_rows = expected.detach().cpu().reshape(row_count, -1)
        mapping = mapping.detach().cpu().to(torch.int64).reshape(-1)
        if bool((mapping < -1).any() or (mapping >= row_count).any()):
            return False, "mapping outside physical row range or invalid negative sentinel"
        mapped = mapping[mapping >= 0]
        if torch.unique(mapped).numel() != mapped.numel():
            return False, "duplicate physical row in mapping"
        written = torch.zeros(row_count, dtype=torch.bool)
        written[mapped] = True
        # Equality-only preservation deliberately accepts matching infinities,
        # while NaNs fail equality. Finiteness applies to mapped rows below.
        if not bool((actual_rows[~written] == expected_rows[~written]).all()):
            return False, "unmapped pool row changed"
        if not bool(written.any()):
            return True, ""
        mapped_actual = actual_rows[written]
        mapped_expected = expected_rows[written]
        for rows in (mapped_actual, mapped_expected):
            if torch.is_floating_point(rows) and not bool(torch.isfinite(rows).all()):
                return False, "nonfinite mapped pool row"
        actual_f = mapped_actual.to(torch.float32)
        expected_f = mapped_expected.to(torch.float32)
        errors = int(((actual_f - expected_f).abs()
                      > policy_atol + policy_rtol * expected_f.abs()).sum().item())
        threshold = round(policy_ratio * actual_f.numel())
        if errors <= threshold:
            return True, ""
        return False, (f"mapped_pool_ratio_allclose: errors={errors}/{actual_f.numel()} "
                       f"threshold={threshold}")

    return compare


def make_compare(case_dir: Path, case: dict):
    policy = _read_policy(case_dir, case)
    config = _config(case_dir, case)
    callbacks = {}
    for name, rule in policy["outputs"].items():
        if not isinstance(rule, dict):
            raise ValueError("invalid comparator rule")
        if name == "x_out":
            required = {"method", "name", "diff_thd", "pct_thd", "max_diff_hd"}
            if set(rule) != required or rule.get("method") != "custom" or rule.get("name") != "ratio_reldiff":
                raise ValueError("invalid ratio_reldiff fields")
            callbacks[name] = _ratio_reldiff(rule)
        elif name == "kv_cache":
            required = {"method", "name", "mapping_name", "mapping_shape", "block_size",
                        "leading_rank_axis", "rtol", "atol", "max_error_ratio"}
            if (set(rule) != required or rule.get("method") != "custom"
                    or rule.get("name") != "mapped_pool_ratio_allclose"):
                raise ValueError("invalid mapped_pool_ratio_allclose fields")
            callbacks[name] = _mapped_pool_ratio_allclose(rule, config)
        else:
            raise ValueError("undeclared comparator output")
    if set(callbacks) != set(case["correctness"]["outputs"]):
        raise ValueError("callback output coverage disagrees with contract")
    return callbacks

def _config(case_dir: Path, case: dict):
    identity = case["case_id"].replace(".", "_")
    config = common.load_isolated_module(case_dir / case["config"], f"config_{identity}")
    if config.CASE_ID != case["case_id"] or config.CASE_CONFIG.name != case["case_id"].rsplit("_", 1)[1]:
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
    bindings = [
        ("x_hc", (t, config.HC_MULT, d), torch.float32),
        ("hc_attn_fn", (config.MIX_HC, config.HC_DIM), torch.float32),
        ("hc_attn_scale", (3,), torch.float32),
        ("hc_attn_base", (config.MIX_HC,), torch.float32),
        ("attn_norm_w", (d,), torch.bfloat16),
        ("wq_a", (d, ql), torch.bfloat16),
        ("wq_b", (ql, h * hd), torch.int8),
        ("wq_b_scale", (h * hd,), torch.float32),
        ("wkv", (d, hd), torch.bfloat16),
        ("gamma_cq", (ql,), torch.bfloat16),
        ("gamma_ckv", (hd,), torch.bfloat16),
        ("freqs_cos", (config.HCA_MAX_SEQ_LEN, model.qk_rope_head_dim), torch.bfloat16),
        ("freqs_sin", (config.HCA_MAX_SEQ_LEN, model.qk_rope_head_dim), torch.bfloat16),
        ("cmp_wkv", (config.MAIN_OUT_DIM, d), torch.bfloat16),
        ("cmp_wgate", (config.MAIN_OUT_DIM, d), torch.bfloat16),
        ("cmp_ape", (config.COMPRESS_RATIO, config.MAIN_OUT_DIM), torch.float32),
        ("cmp_norm_w", (hd,), torch.bfloat16),
        ("compress_state", (config.COMPRESS_STATE_BLOCK_NUM, config.COMPRESS_STATE_BLOCK_SIZE, config.COMPRESS_STATE_DIM), torch.float32),
        ("compress_state_block_table", (b, config.COMPRESS_STATE_MAX_BLOCKS), torch.int32),
        ("kv_cache", (config.DECODE_ORI_BLOCK_NUM, config.BLOCK_SIZE, 1, hd), torch.bfloat16),
        ("cmp_kv", (config.DECODE_CMP_BLOCK_NUM, config.BLOCK_SIZE, 1, hd), torch.bfloat16),
        ("cmp_block_table", (b, config.KV_CMP_MAX_BLOCKS), torch.int32),
        ("ori_slot_mapping", (t,), torch.int64),
        ("window_swa_indices", (t, model.sliding_window), torch.int32),
        ("window_swa_lens", (t,), torch.int32),
        ("cmp_slot_mapping", (t,), torch.int64),
        ("state_slot_mapping", (t,), torch.int64),
        ("position_ids", (t,), torch.int32),
        ("kv_seq_lens", (b,), torch.int32),
        ("attn_sink", (h,), torch.float32),
        ("wo_a", (model.o_groups, model.o_lora_rank, h * hd // model.o_groups), torch.bfloat16),
        ("wo_b", (d, model.o_groups * model.o_lora_rank), torch.int8),
        ("wo_b_scale", (d,), torch.float32),
    ]
    specs = []
    for name, shape, dtype in bindings:
        value = torch.load(fixture / "in" / f"{name}.pt", map_location="cpu", weights_only=True)
        specs.append(TensorSpec(name, list(shape), dtype, init_value=value))
    specs.append(TensorSpec("x_out", [t, config.HC_MULT, d], torch.float32))
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
