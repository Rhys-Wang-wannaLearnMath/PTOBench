"""Case-local evaluator for fixed-shape DeepSeek-V4 prefill QKV/RoPE."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common


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
    expected = {
        "x", "norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv",
        "freqs_cos", "freqs_sin", "gamma_cq", "gamma_ckv", "start_pos",
    }
    if set(values) != expected:
        raise ValueError(
            "fixture inputs do not match the fixed ABI: "
            f"{sorted(values)} != {sorted(expected)}"
        )
    return [
        TensorSpec(
            "x", [config.B, config.S, config.D], torch.bfloat16,
            init_value=values["x"],
        ),
        TensorSpec(
            "norm_w", [config.D], torch.float32,
            init_value=values["norm_w"],
        ),
        TensorSpec(
            "wq_a", [config.D, config.Q_LORA], torch.bfloat16,
            init_value=values["wq_a"],
        ),
        TensorSpec(
            "wq_b", [config.Q_LORA, config.H * config.HEAD_DIM], torch.int8,
            init_value=values["wq_b"],
        ),
        TensorSpec(
            "wq_b_scale", [config.H * config.HEAD_DIM], torch.float32,
            init_value=values["wq_b_scale"],
        ),
        TensorSpec(
            "wkv", [config.D, config.HEAD_DIM], torch.bfloat16,
            init_value=values["wkv"],
        ),
        TensorSpec(
            "freqs_cos", [config.MAX_SEQ_LEN, config.ROPE_DIM],
            torch.bfloat16, init_value=values["freqs_cos"],
        ),
        TensorSpec(
            "freqs_sin", [config.MAX_SEQ_LEN, config.ROPE_DIM],
            torch.bfloat16, init_value=values["freqs_sin"],
        ),
        TensorSpec(
            "gamma_cq", [config.Q_LORA], torch.bfloat16,
            init_value=values["gamma_cq"],
        ),
        TensorSpec(
            "gamma_ckv", [config.HEAD_DIM], torch.bfloat16,
            init_value=values["gamma_ckv"],
        ),
        TensorSpec(
            "q", [config.T, config.H, config.HEAD_DIM], torch.bfloat16,
        ),
        TensorSpec(
            "kv", [config.T, config.HEAD_DIM], torch.bfloat16,
        ),
        TensorSpec("qr", [config.T, config.Q_LORA], torch.int8),
        TensorSpec("qr_scale", [config.T, 1], torch.float32),
        ScalarSpec("start_pos", torch.int32, values["start_pos"]),
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


def _ratio_allclose(name, settings):
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f"ratio_allclose comparator {name!r} requires exactly "
            f"{', '.join(sorted(required))}"
        )
    if settings.get("method") != "ratio_allclose":
        raise ValueError(
            f"unsupported comparator method for {name!r}: "
            f"{settings.get('method')!r}"
        )
    policy_rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], f"{name}.atol")
    policy_ratio = _finite_nonnegative(
        settings["max_error_ratio"], f"{name}.max_error_ratio"
    )
    if policy_ratio > 1:
        raise ValueError(
            f"comparator {name}.max_error_ratio must be at most 1"
        )

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, (
                f"ratio_allclose: shape={tuple(actual.shape)}/"
                f"{tuple(expected.shape)}"
            )
        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all()) or not bool(
            torch.isfinite(expected_f).all()
        ):
            return False, "ratio_allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > (
            policy_atol + policy_rtol * expected_f.abs()
        )
        mismatch_count = int(bad.sum().item())
        allowed_count = round(policy_ratio * actual_f.numel())
        passed = mismatch_count <= allowed_count
        return passed, (
            "" if passed else
            f"ratio_allclose: mismatches={mismatch_count}/"
            f"{actual_f.numel()}, allowed={allowed_count}, "
            f"rtol={policy_rtol}, atol={policy_atol}"
        )

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
        if method != "ratio_allclose":
            raise ValueError(
                f"unsupported comparator method for {name!r}: {method!r}"
            )
        callbacks[name] = _ratio_allclose(name, settings)
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
        rtol=0.005,
        atol=0.005,
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


def _program(module, case):
    return getattr(module, case["entry"])


_INPUT_NAMES = (
    "x", "norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv",
    "freqs_cos", "freqs_sin", "gamma_cq", "gamma_ckv",
)


def _oracle(oracle, values):
    inputs = {name: values[name] for name in _INPUT_NAMES}
    outputs = oracle.compute(inputs, {"start_pos": values["start_pos"]})
    for name, value in outputs.items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    workflow = common.load_module(
        common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
        _identity(case, "workflow"),
    )
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
    workflow = common.load_module(
        common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
        _identity(case, "workflow"),
    )
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

