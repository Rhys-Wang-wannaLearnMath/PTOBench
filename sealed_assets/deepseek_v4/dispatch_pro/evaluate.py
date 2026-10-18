"""Case-local evaluator for packed DeepSeek-V4 dispatch."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common

workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    f"workflow_deepseek_v4_{Path(__file__).resolve().parent.name}",
)


def _config(case_dir: Path, case: dict):
    return common.load_module(
        case_dir / case["config"], f"{case['case_id'].replace('.', '_')}_config"
    )


def _values(fixture: Path):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config):
    values = _values(fixture)
    return [
        TensorSpec("x_norm_i8", list(config.X_NORM_I8_SHAPE), torch.int8, init_value=values["x_norm_i8"]),
        TensorSpec("x_norm_scale", list(config.X_NORM_SCALE_SHAPE), torch.float32, init_value=values["x_norm_scale"]),
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
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return value


def _valid_rows(name, settings, expected_names):
    required = {"method", "name", "count_output", "recv_max", "is_3d", "rtol", "atol"}
    if set(settings) != required or settings.get("method") != "custom":
        raise ValueError(f"custom comparator {name!r} requires exactly {sorted(required)}")
    if settings["name"] != "valid_rows_allclose":
        raise ValueError(f"unsupported custom comparator for {name!r}")
    count_output = settings["count_output"]
    if count_output not in expected_names:
        raise ValueError(f"custom comparator {name!r} references undeclared output")
    recv_max = settings["recv_max"]
    if isinstance(recv_max, bool) or not isinstance(recv_max, int) or recv_max <= 0:
        raise ValueError(f"comparator {name}.recv_max must be a positive integer")
    if not isinstance(settings["is_3d"], bool):
        raise ValueError(f"comparator {name}.is_3d must be boolean")
    rtol = _number(settings["rtol"], f"{name}.rtol")
    atol = _number(settings["atol"], f"{name}.atol")

    def compare(actual, expected, *, expected_outputs, **_):
        counts = expected_outputs[count_output].cpu().reshape(-1)
        if actual.shape != expected.shape:
            return False, f"valid_rows_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        valid = torch.arange(recv_max).reshape(1, recv_max) < counts.reshape(-1, 1)
        if settings["is_3d"]:
            valid = valid.unsqueeze(-1).expand_as(actual)
        a = actual.cpu()[valid].to(torch.float32)
        e = expected.cpu()[valid].to(torch.float32)
        if bool(torch.allclose(a, e, rtol=rtol, atol=atol)):
            return True, f"valid_rows_allclose(rtol={rtol}, atol={atol})"
        bad = (a - e).abs() > (atol + rtol * e.abs())
        return False, f"valid_rows_allclose: mismatches={int(bad.sum())}/{a.numel()}"

    compare.__name__ = "valid_rows_allclose"
    return compare


def _allclose(name, settings):
    if set(settings) != {"method", "rtol", "atol"} or settings.get("method") != "allclose":
        raise ValueError(f"allclose comparator {name!r} requires exactly method, rtol, atol")
    rtol = _number(settings["rtol"], f"{name}.rtol")
    atol = _number(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, f"allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
        return passed, f"torch.allclose(rtol={rtol}, atol={atol})" if passed else f"allclose mismatch"

    return compare


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or policy["schema_version"] != 1:
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    expected_names = list(case["correctness"]["outputs"])
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        raise ValueError("comparator outputs must exactly match correctness outputs")
    callbacks = {}
    for name in expected_names:
        settings = outputs[name]
        if not isinstance(settings, dict):
            raise ValueError(f"comparator output {name!r} must be an object")
        if settings.get("method") == "custom":
            callbacks[name] = _valid_rows(name, settings, expected_names)
        elif settings.get("method") == "allclose":
            callbacks[name] = _allclose(name, settings)
        else:
            raise ValueError(f"unsupported comparator method for {name!r}: {settings.get('method')!r}")
    return callbacks


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
    inputs = {name: values[name] for name in ("x_norm_i8", "x_norm_scale", "indices", "weights")}
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
