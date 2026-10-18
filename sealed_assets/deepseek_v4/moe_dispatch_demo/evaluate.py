"""Case-local evaluator for DeepSeek-V4 quantizing packed dispatch."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common

_CASE_NAME = Path(__file__).resolve().parent.name
workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    f"workflow_deepseek_v4_{_CASE_NAME}",
)


def _config(case_dir, case):
    return common.load_module(
        case_dir / case["config"], f"{case['case_id'].replace('.', '_')}_config"
    )


def _values(fixture):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture, config):
    values = _values(fixture)
    return [
        TensorSpec("x_norm", list(config.X_NORM_SHAPE), torch.bfloat16, init_value=values["x_norm"]),
        TensorSpec("indices", list(config.INDICES_SHAPE), torch.int32, init_value=values["indices"]),
        TensorSpec("weights", list(config.WEIGHTS_SHAPE), torch.float32, init_value=values["weights"]),
        TensorSpec("recv_x", list(config.RECV_X_SHAPE), torch.int8),
        TensorSpec("recv_scale_dq", list(config.RECV_SCALE_DQ_SHAPE), torch.float32),
        TensorSpec("recv_weights", list(config.RECV_WEIGHTS_SHAPE), torch.float32),
        TensorSpec("recv_token", list(config.RECV_TOKEN_SHAPE), torch.int32),
        TensorSpec("recv_expert_count", list(config.RECV_EXPERT_COUNT_SHAPE), torch.int32),
    ]


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _standard_comparator(name, settings):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    method = settings.get("method")
    if method == "allclose":
        required = {"method", "rtol", "atol"}
        if set(settings) != required:
            raise ValueError(f"allclose comparator {name!r} requires exactly {sorted(required)}")
        policy_rtol = _number(settings["rtol"], f"{name}.rtol")
        policy_atol = _number(settings["atol"], f"{name}.atol")

        def compare(actual, expected, **_):
            if actual.shape != expected.shape:
                return False, f"allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
            passed = bool(torch.allclose(actual, expected, rtol=policy_rtol, atol=policy_atol))
            return passed, (f"torch.allclose(rtol={policy_rtol}, atol={policy_atol})" if passed else "allclose mismatch")

        return compare
    if method == "ratio_allclose":
        required = {"method", "rtol", "atol", "max_error_ratio"}
        if set(settings) != required:
            raise ValueError(f"ratio_allclose comparator {name!r} requires exactly {sorted(required)}")
        policy_rtol = _number(settings["rtol"], f"{name}.rtol")
        policy_atol = _number(settings["atol"], f"{name}.atol")
        max_error_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio")
        if max_error_ratio > 1:
            raise ValueError(f"comparator {name}.max_error_ratio must be at most 1")

        def compare(actual, expected, **_):
            if actual.shape != expected.shape:
                return False, f"ratio_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
            if actual.numel() == 0:
                return True, "ratio_allclose: empty"
            actual_f = actual.cpu().to(torch.float32)
            expected_f = expected.cpu().to(torch.float32)
            if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
                return False, "ratio_allclose: non-finite value"
            bad = (actual_f - expected_f).abs() > (policy_atol + policy_rtol * expected_f.abs())
            errors = int(bad.sum().item())
            allowed = round(max_error_ratio * actual_f.numel())
            passed = errors <= allowed
            return passed, ("ratio_allclose" if passed else f"ratio_allclose: mismatches={errors}/{actual_f.numel()}, allowed={allowed}")

        return compare
    raise ValueError(f"unsupported comparator method for {name!r}: {method!r}")


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or policy["schema_version"] != 1:
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    names = list(case["correctness"]["outputs"])
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator outputs must exactly match correctness outputs")
    return {name: _standard_comparator(name, outputs[name]) for name in names}


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
    return run(
        fn=fn, specs=specs, golden_fn=golden_fn,
        golden_data=str(golden_data) if golden_data else None,
        config=config, rtol=1e-3, atol=1e-3, compare_fn=compare_fn,
        save_data=False, capture_outputs=capture_outputs,
    )


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    inputs = {name: values[name] for name in ("x_norm", "indices", "weights")}
    for name, value in oracle.compute(inputs, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"{case['case_id'].replace('.', '_')}_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"{case['case_id'].replace('.', '_')}_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"{case['case_id'].replace('.', '_')}_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run,
    )
