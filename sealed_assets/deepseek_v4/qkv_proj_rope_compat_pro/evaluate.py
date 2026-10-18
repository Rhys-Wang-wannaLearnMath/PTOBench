"""Case-local evaluator for one fixed QKV/RoPE compatibility preset."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common

workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator/workflow.py",
    "deepseek_v4_qkv_proj_rope_compat_workflow",
)


def _identity(case, role):
    return f"{case['case_id'].replace('.', '_')}_{role}"


def _specs(case_dir: Path, sealed: Path, fixture: Path, case: dict):
    builder = common.load_module(sealed / "fixture_builder.py", _identity(case, "fixture_builder"))
    legacy = builder.build(case_dir, sealed, case, 0)
    result = []
    for spec in legacy:
        path = fixture / "in" / f"{spec.name}.pt"
        value = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
        result.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=value))
    return result


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return result


def _compare(case_dir: Path, case: dict):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if (not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}
            or type(policy["schema_version"]) is not int or policy["schema_version"] != 1):
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy["outputs"]
    names = list(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator output coverage mismatch")

    callbacks = {}
    for name in names:
        settings = outputs[name]
        expected_fields = {"method", "rtol", "atol", "max_error_ratio"}
        if (not isinstance(settings, dict) or set(settings) != expected_fields
                or settings.get("method") != "ratio_allclose"):
            raise ValueError(f"{name} must declare ratio_allclose with exactly four fields")
        policy_rtol = _number(settings["rtol"], f"{name}.rtol")
        policy_atol = _number(settings["atol"], f"{name}.atol")
        policy_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio")
        if policy_ratio > 1:
            raise ValueError(f"{name}.max_error_ratio must be at most 1")

        def compare(actual, expected, policy_rtol=policy_rtol, policy_atol=policy_atol,
                    policy_ratio=policy_ratio, **_runner_fallbacks):
            if actual.shape != expected.shape:
                return False, "shape mismatch"
            actual_f = actual.cpu().to(torch.float32)
            expected_f = expected.cpu().to(torch.float32)
            if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
                return False, "non-finite value"
            allowed_error = policy_atol + policy_rtol * expected_f.abs()
            mismatch_count = int(((actual_f - expected_f).abs() > allowed_error).sum().item())
            allowed_count = round(policy_ratio * actual_f.numel())
            return mismatch_count <= allowed_count, (
                f"mismatches={mismatch_count}/{actual_f.numel()}, allowed={allowed_count}"
            )

        callbacks[name] = compare
    return callbacks


def _run(**kwargs):
    return run(save_data=False, **kwargs)


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, answer in oracle.compute(tensors, {}).items():
        values[name][:] = answer


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: _specs(case_dir, sealed, fixture, case),
        load_ref=lambda: common.load_ref(case_dir, sealed, _identity(case, "ref")),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, _identity(case, "oracle")),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: _specs(case_dir, sealed, fixture, case),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, _identity(case, "candidate"),
        ),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run,
    )
