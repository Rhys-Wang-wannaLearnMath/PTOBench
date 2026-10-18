"""Case-local evaluator for the DeepSeek-V4 decode gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common


_CASE_NAME = Path(__file__).resolve().parent.name
workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    f"workflow_deepseek_v4_{_CASE_NAME}",
)


def _identity(case, role):
    return f"{case['case_id'].replace('.', '_')}_{role}"


def _config(case_dir, case):
    return common.load_module(
        case_dir / case["config"], _identity(case, "public_config")
    )


def _fixture_values(fixture):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def _build_specs(fixture, config):
    values = _fixture_values(fixture)
    return [
        TensorSpec(
            "x_mixed", [config.T, config.D], torch.bfloat16,
            init_value=values["x_mixed"],
        ),
        TensorSpec(
            "norm_w", [config.D], torch.bfloat16,
            init_value=values["norm_w"],
        ),
        TensorSpec(
            "gate_w", [config.N_EXPERTS, config.D], torch.float32,
            init_value=values["gate_w"],
        ),
        TensorSpec(
            "gate_bias", [config.N_EXPERTS], torch.float32,
            init_value=values["gate_bias"],
        ),
        ScalarSpec("layer_id", torch.int32, values["layer_id"]),
        ScalarSpec("num_tokens", torch.int32, values["num_tokens"]),
        TensorSpec(
            "tid2eid", [config.VOCAB, config.TOPK], torch.int32,
            init_value=values["tid2eid"],
        ),
        TensorSpec(
            "input_ids", [config.T], torch.int64,
            init_value=values["input_ids"],
        ),
        TensorSpec("x_norm", [config.T, config.D], torch.bfloat16),
        TensorSpec("x_norm_i8", [config.T, config.D], torch.int8),
        TensorSpec("x_norm_scale", [config.T, 1], torch.float32),
        TensorSpec("indices", [config.T, config.TOPK], torch.int32),
        TensorSpec("weights", [config.T, config.TOPK], torch.float32),
    ]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"comparator {field} must be finite and nonnegative"
        )
    return number


def _allclose_comparator(name, settings):
    required = {"method", "rtol", "atol"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f"allclose comparator {name!r} requires exactly "
            f"{', '.join(sorted(required))}"
        )
    rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    atol = _finite_nonnegative(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, (
                f"allclose: shape={tuple(actual.shape)}/"
                f"{tuple(expected.shape)}"
            )
        passed = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
        if passed:
            return True, f"torch.allclose(rtol={rtol}, atol={atol})"
        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        mismatches = int((~close).sum().item())
        return False, (
            f"allclose: mismatches={mismatches}/{actual.numel()}, "
            f"rtol={rtol}, atol={atol}"
        )

    return compare


def _ratio_allclose_comparator(name, settings):
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f"ratio_allclose comparator {name!r} requires exactly "
            f"{', '.join(sorted(required))}"
        )
    rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    atol = _finite_nonnegative(settings["atol"], f"{name}.atol")
    max_error_ratio = _finite_nonnegative(
        settings["max_error_ratio"], f"{name}.max_error_ratio"
    )
    if max_error_ratio > 1:
        raise ValueError(
            f"comparator {name}.max_error_ratio must be at most 1"
        )

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, (
                f"ratio_allclose: shape={tuple(actual.shape)}/"
                f"{tuple(expected.shape)}"
            )
        if actual.numel() == 0:
            return True, ""
        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all()) or not bool(
            torch.isfinite(expected_f).all()
        ):
            return False, "ratio_allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > (
            atol + rtol * expected_f.abs()
        )
        error_count = int(bad.sum().item())
        threshold = round(max_error_ratio * actual_f.numel())
        passed = error_count <= threshold
        return passed, (
            "" if passed else
            f"ratio_allclose: mismatches={error_count}/"
            f"{actual_f.numel()}, allowed={threshold}, "
            f"rtol={rtol}, atol={atol}"
        )

    return compare


def _topk_pair_comparator(name, settings, output_names):
    required = {"method", "name", "values_output", "dim", "descending"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f"custom comparator {name!r} requires exactly "
            f"{', '.join(sorted(required))}"
        )
    if settings.get("method") != "custom" or settings.get("name") != (
        "topk_pair_compare"
    ):
        raise ValueError(f"unsupported custom comparator for {name!r}")
    values_output = settings["values_output"]
    if not isinstance(values_output, str) or values_output not in output_names:
        raise ValueError(
            f"custom comparator {name!r} references undeclared output"
        )
    dim = settings["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"comparator {name}.dim must be an integer")
    descending = settings["descending"]
    if not isinstance(descending, bool):
        raise ValueError(f"comparator {name}.descending must be boolean")

    def compare(actual, expected, *, actual_outputs, **_):
        if values_output not in actual_outputs:
            return False, (
                f"topk_pair_compare: paired output {values_output!r} missing"
            )
        actual_idx = actual.cpu()
        expected_idx = expected.cpu()
        actual_values = actual_outputs[values_output].cpu().to(torch.float32)
        if actual_idx.shape != expected_idx.shape:
            return False, (
                f"topk_pair_compare: shape={tuple(actual_idx.shape)}/"
                f"{tuple(expected_idx.shape)}"
            )
        if actual_idx.shape != actual_values.shape:
            return False, (
                "topk_pair_compare: index/value shape mismatch "
                f"{tuple(actual_idx.shape)}/{tuple(actual_values.shape)}"
            )
        dim_pos = dim if dim >= 0 else dim + actual_idx.dim()
        if not 0 <= dim_pos < actual_idx.dim():
            return False, (
                f"topk_pair_compare: dim={dim} out of range for "
                f"shape {tuple(actual_idx.shape)}"
            )
        actual_idx = actual_idx.movedim(dim_pos, -1).reshape(
            -1, actual_idx.shape[dim_pos]
        )
        expected_idx = expected_idx.movedim(dim_pos, -1).reshape_as(actual_idx)
        actual_values = actual_values.movedim(dim_pos, -1).reshape_as(
            actual_idx
        )
        mismatch = actual_idx != expected_idx
        if not bool(mismatch.any()):
            return True, ""
        if actual_idx.shape[-1] < 2:
            return True, ""
        adjacent = (
            actual_values[:, :-1] >= actual_values[:, 1:]
            if descending
            else actual_values[:, :-1] <= actual_values[:, 1:]
        )
        position_ok = torch.ones_like(mismatch)
        position_ok[:, 1:] &= adjacent
        position_ok[:, :-1] &= adjacent
        failures = mismatch & ~position_ok
        passed = not bool(failures.any())
        return passed, (
            "" if passed else
            "topk_pair_compare: actual paired values break declared order "
            f"at {int(failures.sum().item())} mismatched position(s)"
        )

    compare.__name__ = "topk_pair_compare"
    return compare


def _compare(case_dir, case):
    policy = json.loads(
        (case_dir / case["comparator"]).read_text(encoding="utf-8")
    )
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
    ):
        raise ValueError(
            "comparator.json must contain only schema_version=1 and outputs"
        )
    output_names = list(case["correctness"]["outputs"])
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(output_names):
        raise ValueError(
            "comparator outputs must exactly match case correctness outputs"
        )
    callbacks = {}
    for name in output_names:
        settings = outputs[name]
        if not isinstance(settings, dict):
            raise ValueError(f"comparator output {name!r} must be an object")
        method = settings.get("method")
        if method == "allclose":
            callbacks[name] = _allclose_comparator(name, settings)
        elif method == "ratio_allclose":
            callbacks[name] = _ratio_allclose_comparator(name, settings)
        elif method == "custom":
            callbacks[name] = _topk_pair_comparator(
                name, settings, output_names
            )
        else:
            raise ValueError(
                f"unsupported comparator method for {name!r}: {method!r}"
            )
    return callbacks


def _run(
    *, fn, specs, config, compare_fn, golden_data=None, golden_fn=None,
    capture_outputs=False,
):
    return run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=str(golden_data) if golden_data else None,
        config=config,
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    input_names = (
        "x_mixed", "norm_w", "gate_w", "gate_bias", "tid2eid", "input_ids"
    )
    scalars = {
        name: values[name] for name in ("layer_id", "num_tokens")
    }
    outputs = oracle.compute(
        {name: values[name] for name in input_names}, scalars
    )
    for name, value in outputs.items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: _build_specs(fixture, config),
        load_ref=lambda: common.load_ref(
            case_dir, sealed, _identity(case, "ref")
        ),
        load_oracle=lambda: common.load_oracle(
            case_dir, sealed, _identity(case, "oracle")
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(
    case_dir, sealed, case, candidate_workspace, platform=None, phase="all",
):
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: _build_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(
            case_dir,
            case,
            candidate_workspace,
            _identity(case, "candidate"),
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
