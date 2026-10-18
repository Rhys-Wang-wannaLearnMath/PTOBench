from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common
import workflow


_TENSOR_NAMES = [
    "x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "block_table", "ori_slot_mapping",
    "position_ids", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "x_out",
]


def _number(value, name, *, positive=False, upper=None):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"invalid comparator {name}")
    result = float(value)
    if result < 0 or (positive and result <= 0) or (upper is not None and result > upper):
        raise ValueError(f"invalid comparator {name}")
    return result


def _policy(case_dir: Path, case: dict) -> dict:
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or type(policy.get("schema_version")) is not int or policy["schema_version"] != 1:
        raise ValueError("invalid comparator top-level fields")
    outputs = policy.get("outputs")
    expected = set(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != expected:
        raise ValueError("comparator output coverage disagrees with contract")
    return policy


def _valid_ratio(rule):
    required = {"method", "name", "diff_thd", "pct_thd", "max_diff_hd", "valid_prefix", "zero_tail"}
    if set(rule) != required or rule.get("method") != "custom" or rule.get("name") != "valid_ratio_reldiff":
        raise ValueError("invalid valid_ratio_reldiff fields")
    if rule.get("valid_prefix") != "num_tokens" or rule.get("zero_tail") is not True:
        raise ValueError("invalid valid_ratio_reldiff scope")
    diff_thd = _number(rule["diff_thd"], "diff_thd", positive=True)
    pct_thd = _number(rule["pct_thd"], "pct_thd", upper=1)
    max_diff_hd = _number(rule["max_diff_hd"], "max_diff_hd", positive=True)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape or actual.ndim == 0:
            return False, "output shape mismatch"
        inputs = kwargs.get("inputs", {})
        scalar = inputs.get("num_tokens")
        if scalar is None:
            return False, "num_tokens input missing"
        if not isinstance(scalar, torch.Tensor) or scalar.numel() != 1 or scalar.dtype != torch.int32:
            return False, "num_tokens must be one int32 scalar"
        valid_rows = int(scalar.item())
        if valid_rows < 1 or valid_rows > actual.shape[0]:
            return False, "num_tokens out of bounds"
        if actual.shape[0] > valid_rows and bool(actual[valid_rows:].detach().cpu().to(torch.float32).ne(0).any()):
            return False, "nonzero inactive tail"
        a = actual[:valid_rows].detach().cpu().to(torch.float32)
        b = expected[:valid_rows].detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            return False, "nonfinite output"
        absdiff = (a - b).abs()
        floor = (1.0 / (1 << 14)) / diff_thd
        denom = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.full_like(a, floor)) + 1e-9
        rel = torch.where(absdiff < diff_thd, absdiff, absdiff / denom)
        bad = rel > diff_thd
        count = int(bad.sum().item())
        threshold = round(pct_thd * a.numel())
        worst = float(rel[bad].max().item()) if count else 0.0
        passed = count <= threshold and worst < max_diff_hd
        return passed, "" if passed else f"valid_ratio_reldiff: errors={count}/{a.numel()} threshold={threshold} worst={worst}"

    return compare


def _ratio(rule):
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if set(rule) != required or rule.get("method") != "ratio_allclose":
        raise ValueError("invalid ratio_allclose fields")
    policy_rtol = _number(rule["rtol"], "rtol")
    policy_atol = _number(rule["atol"], "atol")
    policy_ratio = _number(rule["max_error_ratio"], "max_error_ratio", upper=1)

    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, "output shape mismatch"
        actual_fp32 = actual.detach().cpu().to(torch.float32)
        expected_fp32 = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_fp32).all() and torch.isfinite(expected_fp32).all()):
            return False, "nonfinite output"
        errors = int((actual_fp32 - expected_fp32).abs().gt(
            policy_atol + policy_rtol * expected_fp32.abs()).sum().item())
        threshold = round(policy_ratio * actual_fp32.numel())
        return errors <= threshold, "" if errors <= threshold else f"ratio_allclose: errors={errors}/{actual_fp32.numel()} threshold={threshold}"

    return compare


def make_compare(case_dir: Path, case: dict):
    rules = _policy(case_dir, case)["outputs"]
    callbacks = {}
    for name, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError(f"invalid comparator rule for {name}")
        method = rule.get("method")
        if method == "custom" and name == "x_out":
            callbacks[name] = _valid_ratio(rule)
        elif method == "ratio_allclose" and name == "kv_cache":
            callbacks[name] = _ratio(rule)
        else:
            raise ValueError(f"unsupported comparator method {method!r}")
    return callbacks


def _config(case_dir: Path, case):
    identity = case["case_id"].replace(".", "_")
    config = common.load_isolated_module(case_dir / case["config"], f"config_{identity}")
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config identity disagrees with Case")
    return config


def build_specs(fixture: Path, config):
    specs = []
    for name in _TENSOR_NAMES:
        if name == "num_tokens":
            continue
        if name == "x_out":
            specs.append(TensorSpec(name, [config.PREFILL_TOKENS, config.HC_MULT, config.D], torch.float32))
            continue
        value = torch.load(fixture / "in" / f"{name}.pt", map_location="cpu", weights_only=True)
        specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    scalar = torch.load(fixture / "in" / "num_tokens.pt", map_location="cpu", weights_only=True)
    specs.append(ScalarSpec("num_tokens", torch.int32, int(scalar.item())))
    return specs


def _program(module, case):
    return getattr(module, case["entry"])


def _golden(oracle, values):
    outputs = oracle.compute(values)
    if set(outputs) != {"kv_cache", "x_out"}:
        raise ValueError("Oracle output coverage disagrees with contract")
    for name, value in outputs.items():
        values[name].copy_(value)


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"ref_{identity}"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"oracle_{identity}"),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run,
        oracle_golden=_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase, build_specs=lambda fixture: build_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"candidate_{identity}"),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run,
    )
