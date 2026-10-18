"""Case-local evaluator for the DeepSeek-V4 legacy T=16 decode QKV/RoPE Case."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common


workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator/workflow.py",
    "legacy_decode_qkv_proj_rope_t16_workflow",
)


def _specs(case_dir: Path, sealed: Path, fixture: Path, case: dict):
    builder = common.load_module(
        sealed / "fixture_builder.py",
        "legacy_decode_qkv_proj_rope_t16_fixture_builder",
    )
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


def _integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"comparator {field} must be an integer")
    if value < 0 or value > 255:
        raise ValueError(f"comparator {field} must be in [0, 255]")
    return value


def _allclose_callback(policy_rtol, policy_atol):
    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, "shape mismatch"
        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        passed = bool(
            ((actual_f - expected_f).abs() <= policy_atol + policy_rtol * expected_f.abs())
            .all()
            .item()
        )
        return passed, "" if passed else "allclose tolerance exceeded"

    return compare


def _int8_lsb_callback(policy_max_abs_diff):
    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, "shape mismatch"
        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu()
        if not bool(torch.isfinite(actual_cpu).all()) or not bool(torch.isfinite(expected_cpu).all()):
            return False, "non-finite value"
        diff = (actual_cpu.to(torch.int16) - expected_cpu.to(torch.int16)).abs()
        maximum = int(diff.max().item()) if diff.numel() else 0
        passed = maximum <= policy_max_abs_diff
        return passed, "" if passed else f"max INT8 diff {maximum} > {policy_max_abs_diff}"

    return compare


def _callbacks(policy, case):
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or policy.get("schema_version") != 1
    ):
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy.get("outputs")
    names = list(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator output coverage mismatch")

    callbacks = {}
    for name in names:
        settings = outputs[name]
        if not isinstance(settings, dict):
            raise ValueError(f"comparator rule for {name} must be an object")
        method = settings.get("method")
        if method == "allclose":
            if set(settings) != {"method", "rtol", "atol"}:
                raise ValueError(f"{name} allclose must declare exactly method, rtol, and atol")
            policy_rtol = _number(settings["rtol"], f"{name}.rtol")
            policy_atol = _number(settings["atol"], f"{name}.atol")
            callbacks[name] = _allclose_callback(policy_rtol, policy_atol)
        elif method == "custom":
            if set(settings) != {"method", "name", "max_abs_diff"}:
                raise ValueError(
                    f"{name} int8_lsb must declare exactly method, name, and max_abs_diff"
                )
            if settings.get("name") != "int8_lsb":
                raise ValueError(f"unsupported custom comparator for {name}: {settings.get('name')}")
            maximum = _integer(settings["max_abs_diff"], f"{name}.max_abs_diff")
            callbacks[name] = _int8_lsb_callback(maximum)
        else:
            raise ValueError(f"unsupported comparator method for {name}: {method}")
    return callbacks


def _compare(case_dir: Path, case: dict):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    return _callbacks(policy, case)


def _run(**kwargs):
    return run(save_data=False, **kwargs)


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    tensors = {key: value for key, value in values.items() if isinstance(value, torch.Tensor)}
    for name, answer in oracle.compute(tensors, {}).items():
        values[name][:] = answer


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: _specs(case_dir, sealed, fixture, case),
        load_ref=lambda: common.load_ref(
            case_dir, sealed, "legacy_decode_qkv_proj_rope_t16_ref"
        ),
        load_oracle=lambda: common.load_oracle(
            case_dir, sealed, "legacy_decode_qkv_proj_rope_t16_oracle"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(
    case_dir,
    sealed,
    case,
    candidate_workspace,
    platform=None,
    phase="all",
):
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: _specs(case_dir, sealed, fixture, case),
        load_candidate=lambda: common.load_candidate(
            case_dir,
            case,
            candidate_workspace,
            "legacy_decode_qkv_proj_rope_t16_candidate",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
