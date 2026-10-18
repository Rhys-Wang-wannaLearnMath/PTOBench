from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common
import workflow


_NAMES = [
    "x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "compress_state", "compress_state_block_table", "hadamard_idx", "idx_wq_b",
    "idx_wq_b_scale", "idx_weights_proj", "inner_wkv", "inner_wgate", "inner_ape",
    "inner_norm_w", "inner_compress_state", "inner_compress_state_block_table",
    "kv_cache", "ori_block_table", "ori_slot_mapping", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table", "position_ids",
    "cmp_slot_mapping", "idx_slot_mapping", "state_slot_mapping", "inner_state_slot_mapping",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale", "x_out", "num_tokens",
]


def _policy(case_dir: Path, case: dict) -> dict:
    value = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "outputs"} or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("invalid comparator top-level fields")
    outputs = value.get("outputs")
    names = set(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != names:
        raise ValueError("comparator output coverage disagrees with contract")
    return value


def _finite_number(value, name, *, positive=False, upper=None):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"invalid comparator {name}")
    value = float(value)
    if value < 0 or (positive and value <= 0) or (upper is not None and value > upper):
        raise ValueError(f"invalid comparator {name}")
    return value


def _valid_ratio(rule):
    required = {"method", "name", "valid_rows_from", "valid_axis", "diff_thd", "pct_thd", "max_diff_hd", "tail_policy"}
    if set(rule) != required or rule["method"] != "custom" or rule["name"] != "valid_ratio_reldiff":
        raise ValueError("invalid valid_ratio_reldiff fields")
    if rule["valid_rows_from"] != "num_tokens" or type(rule["valid_axis"]) is not int or rule["valid_axis"] != 0 or rule["tail_policy"] != "reject_undeclared":
        raise ValueError("invalid valid_ratio_reldiff scope")
    diff = _finite_number(rule["diff_thd"], "diff_thd", positive=True)
    pct = _finite_number(rule["pct_thd"], "pct_thd", upper=1)
    max_diff = _finite_number(rule["max_diff_hd"], "max_diff_hd", positive=True)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.ndim == 0:
            return False, "output shape mismatch"
        inputs = kwargs.get("inputs", {})
        scalar = inputs.get("num_tokens")
        if scalar is None:
            return False, "num_tokens input missing"
        valid_rows = int(scalar.item())
        if valid_rows < 0 or valid_rows > actual.shape[0]:
            return False, "num_tokens out of bounds"
        if actual.shape[0] > valid_rows and bool(actual[valid_rows:].detach().cpu().to(torch.float32).ne(0).any()):
            return False, "nonzero inactive tail"
        a = actual[:valid_rows].detach().cpu().to(torch.float32)
        b = expected[:valid_rows].detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            return False, "nonfinite output"
        absdiff = (a - b).abs()
        floor = (1.0 / (1 << 14)) / diff
        denom = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.full_like(a, floor)) + 1e-9
        rel = torch.where(absdiff < diff, absdiff, absdiff / denom)
        bad = rel > diff
        count = int(bad.sum().item())
        threshold = round(pct * a.numel())
        worst = float(rel[bad].max().item()) if count else 0.0
        ok = count <= threshold and worst < max_diff
        return ok, "" if ok else f"valid_ratio_reldiff: errors={count}/{a.numel()} threshold={threshold} worst={worst}"
    return compare


def _ratio(rule):
    if set(rule) != {"method", "rtol", "atol", "max_error_ratio"} or rule["method"] != "ratio_allclose":
        raise ValueError("invalid ratio_allclose fields")
    policy_rtol = _finite_number(rule["rtol"], "rtol")
    policy_atol = _finite_number(rule["atol"], "atol")
    policy_ratio = _finite_number(rule["max_error_ratio"], "max_error_ratio", upper=1)
    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False, "output shape mismatch"
        a = actual.detach().cpu().to(torch.float32)
        b = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            return False, "nonfinite output"
        errors = int((a - b).abs().gt(policy_atol + policy_rtol * b.abs()).sum().item())
        threshold = round(policy_ratio * a.numel())
        return errors <= threshold, "" if errors <= threshold else f"ratio_allclose: errors={errors}/{a.numel()} threshold={threshold}"
    return compare


def make_compare(case_dir: Path, case: dict):
    rules = _policy(case_dir, case)["outputs"]
    callbacks = {}
    for name, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError("invalid comparator rule")
        callbacks[name] = _valid_ratio(rule) if name == "x_out" else _ratio(rule)
    return callbacks


def _config(case_dir, case):
    identity = case["case_id"].replace(".", "_")
    config = common.load_isolated_module(case_dir / case["config"], f"config_{identity}")
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config identity disagrees with Case")
    return config


class _CorrectnessTensorSpec(TensorSpec):
    """Select public correctness outputs while retaining stamped ABI directions.

    is_input and direction remain authoritative for scratch cloning and all five
    per-launch InOut resets. Auxiliary writes are part of the ABI, not numerical
    correctness outputs in this selected official workload.
    """
    @property
    def is_output(self):
        return super().is_output and self.name in {"x_out", "kv_cache"}


def build_specs(fixture: Path, config):
    specs = []
    for name in _NAMES:
        if name == "num_tokens":
            value = torch.load(fixture / "in" / "num_tokens.pt", map_location="cpu", weights_only=True)
            specs.append(ScalarSpec(name, torch.int32, int(value.item())))
            continue
        if name == "x_out":
            specs.append(_CorrectnessTensorSpec(name, [config.PREFILL_TOKENS, config.HC_MULT, config.D], torch.float32))
            continue
        value = torch.load(fixture / "in" / f"{name}.pt", map_location="cpu", weights_only=True)
        specs.append(_CorrectnessTensorSpec(name, list(value.shape), value.dtype, init_value=value))
    return specs


def _program(module, case):
    return getattr(module, case["entry"])


def _golden(oracle, values):
    result = oracle.compute(values)
    if set(result) != {"x_out", "kv_cache"}:
        raise ValueError("Oracle output coverage disagrees with contract")
    for name, value in result.items():
        values[name].copy_(value)


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"ref_{identity}"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"oracle_{identity}"),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run,
        oracle_golden=_golden)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"candidate_{identity}"),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run)
