"""Case-local evaluator for the DeepSeek-V4 RMSNorm prefill FLASH Ref."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common

workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator/workflow.py", "workflow_rmsnorm_prefill_flash")


def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], "rmsnorm_prefill_flash_config")


def _fixture_values(fixture):
    return {p.stem: torch.load(p, map_location="cpu", weights_only=True)
            for p in (fixture / "in").glob("*.pt")}


def build_tensor_specs(fixture, config):
    values = _fixture_values(fixture)
    return [
        TensorSpec("x", [config.PREFILL_TOKENS, config.D], torch.bfloat16, init_value=values["x"]),
        TensorSpec("norm_w", [config.D], torch.bfloat16, init_value=values["norm_w"]),
        TensorSpec("x_normed", [config.PREFILL_TOKENS, config.D], torch.bfloat16),
    ]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return value


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if (not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}
            or isinstance(policy["schema_version"], bool)
            or not isinstance(policy["schema_version"], int)
            or policy["schema_version"] != 1):
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy["outputs"]
    names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator outputs must exactly match correctness outputs")
    settings = outputs["x_normed"]
    if not isinstance(settings, dict) or set(settings) != {"method", "rtol", "atol", "max_error_ratio"}:
        raise ValueError("x_normed ratio_allclose policy fields are incomplete or extra")
    if settings["method"] != "ratio_allclose":
        raise ValueError("x_normed must use ratio_allclose")
    policy_rtol = _finite_nonnegative(settings["rtol"], "x_normed.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], "x_normed.atol")
    max_error_ratio = _finite_nonnegative(settings["max_error_ratio"], "x_normed.max_error_ratio")
    if max_error_ratio > 1:
        raise ValueError("x_normed.max_error_ratio must be at most 1")

    def compare(actual, expected, **_kwargs):
        if actual.shape != expected.shape:
            return False, f"ratio_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
            return False, "ratio_allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > (policy_atol + policy_rtol * expected_f.abs())
        count = int(bad.sum().item())
        allowed = round(max_error_ratio * actual_f.numel())
        return count <= allowed, f"ratio_allclose: mismatches={count}/{actual_f.numel()}, allowed={max_error_ratio}, rtol={policy_rtol}, atol={policy_atol}"
    return {"x_normed": compare}


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
    return run(fn=fn, specs=specs, golden_fn=golden_fn,
               golden_data=str(golden_data) if golden_data else None,
               config=config, rtol=1e-3, atol=1e-3, compare_fn=compare_fn,
               save_data=False, capture_outputs=capture_outputs)


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, "rmsnorm_prefill_flash_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, "rmsnorm_prefill_flash_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle_golden)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, "rmsnorm_prefill_flash_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
