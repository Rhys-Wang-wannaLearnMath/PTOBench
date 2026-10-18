"""Case-local evaluator for DeepSeek-V4 HC pre-mix."""

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common


_CASE_NAME = Path(__file__).resolve().parent.name
workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator/workflow.py",
    f"workflow_deepseek_v4_{_CASE_NAME}",
)


def _identity(case, role):
    return f"{case['case_id'].replace('.', '_')}_{role}"


def _values(fixture):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def _config(case_dir, case):
    return common.load_module(
        case_dir / case["config"], _identity(case, "public_config")
    )


def _build_specs(fixture, config):
    values = _values(fixture)
    return [
        TensorSpec(
            "x", config.X_SHAPE, torch.bfloat16, init_value=values["x"]
        ),
        TensorSpec(
            "hc_fn",
            config.HC_FN_SHAPE,
            torch.float32,
            init_value=values["hc_fn"],
        ),
        TensorSpec(
            "hc_scale",
            config.HC_SCALE_SHAPE,
            torch.float32,
            init_value=values["hc_scale"],
        ),
        TensorSpec(
            "hc_base",
            config.HC_BASE_SHAPE,
            torch.float32,
            init_value=values["hc_base"],
        ),
        TensorSpec("x_mixed", config.X_MIXED_SHAPE, torch.bfloat16),
        TensorSpec("post", config.POST_SHAPE, torch.float32),
        TensorSpec("comb", config.COMB_SHAPE, torch.float32),
    ]


def _program(module, case):
    return getattr(module, case["entry"])


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _ratio_comparator(name, settings):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    if settings.get("method") != "ratio_allclose":
        raise ValueError(
            f"unsupported comparator method for {name!r}: "
            f"{settings.get('method')!r}"
        )
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if set(settings) != required:
        fields = ", ".join(sorted(required))
        raise ValueError(
            f"ratio_allclose comparator {name!r} requires exactly {fields}"
        )
    policy_rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], f"{name}.atol")
    max_error_ratio = _finite_nonnegative(
        settings["max_error_ratio"], f"{name}.max_error_ratio"
    )
    if max_error_ratio > 1:
        raise ValueError(f"comparator {name}.max_error_ratio must be at most 1")

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
            policy_atol + policy_rtol * expected_f.abs()
        )
        error_count = int(bad.sum().item())
        threshold = round(max_error_ratio * actual_f.numel())
        passed = error_count <= threshold
        return passed, (
            ""
            if passed
            else f"ratio_allclose: mismatches={error_count}/"
            f"{actual_f.numel()}, allowed={max_error_ratio}, "
            f"rtol={policy_rtol}, atol={policy_atol}"
        )

    return compare


def _compare(case_dir, case):
    path = case_dir / case["comparator"]
    policy = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
    ):
        raise ValueError(
            "comparator.json must contain only schema_version=1 and outputs"
        )
    outputs = policy["outputs"]
    expected_names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        actual_names = (
            sorted(outputs) if isinstance(outputs, dict) else type(outputs).__name__
        )
        raise ValueError(
            "comparator outputs must exactly match case correctness outputs: "
            f"{actual_names} != {sorted(expected_names)}"
        )
    return {
        name: _ratio_comparator(name, outputs[name]) for name in expected_names
    }


def _run(**kwargs):
    return run(rtol=1e-3, atol=1e-3, save_data=False, **kwargs)


def _oracle(oracle, values):
    tensors = {
        name: value for name, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: _build_specs(
            fixture, _config(case_dir, case)
        ),
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
    case_dir, sealed, case, candidate_workspace, platform=None, phase="all"
):
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: _build_specs(
            fixture, _config(case_dir, case)
        ),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, _identity(case, "candidate")
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
