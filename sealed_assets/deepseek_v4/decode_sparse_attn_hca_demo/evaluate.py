"""Case-local evaluator for DeepSeek-V4 sparse HCA decode attention."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common


INPUT_NAMES = (
    "q",
    "ori_kv",
    "window_swa_indices",
    "mtp_kv_overlay",
    "cmp_kv",
    "cmp_block_table",
    "cmp_sparse_indices",
    "attn_sink",
    "freqs_cos",
    "freqs_sin",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)


def _identity(case: dict, role: str) -> str:
    return f"{case['case_id'].replace('.', '_')}_{role}"


def _workflow(case: dict):
    return common.load_module(
        common.BENCHMARK_ROOT / "evaluator/workflow.py",
        _identity(case, "workflow"),
    )


def _config(case_dir: Path, case: dict):
    return common.load_module(
        case_dir / case["config"], _identity(case, "config")
    )


def _values(fixture: Path) -> dict[str, torch.Tensor]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config):
    values = _values(fixture)
    if set(values) != set(INPUT_NAMES):
        raise ValueError(
            f"fixture inputs mismatch: {sorted(values)} != {sorted(INPUT_NAMES)}"
        )
    specs = [
        TensorSpec(name, list(values[name].shape), values[name].dtype,
                   init_value=values[name])
        for name in INPUT_NAMES
    ]
    specs.append(TensorSpec("attn_out", [config.T, config.D], torch.bfloat16))
    return specs


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _ratio_allclose(name: str, settings: dict):
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
    rtol = _number(settings["rtol"], f"{name}.rtol")
    atol = _number(settings["atol"], f"{name}.atol")
    max_error_ratio = _number(
        settings["max_error_ratio"], f"{name}.max_error_ratio"
    )
    if max_error_ratio > 1:
        raise ValueError(f"comparator {name}.max_error_ratio must be at most 1")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, (
                f"ratio_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
            )
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
        allowed = round(max_error_ratio * actual_f.numel())
        passed = error_count <= allowed
        return passed, (
            f"ratio_allclose: mismatches={error_count}/{actual_f.numel()}, "
            f"allowed={allowed}, rtol={rtol}, atol={atol}"
        )

    return compare


def _compare(case_dir: Path, case: dict):
    policy = json.loads(
        (case_dir / case["comparator"]).read_text(encoding="utf-8")
    )
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or policy["schema_version"] != 1
    ):
        raise ValueError(
            "comparator.json must contain only schema_version=1 and outputs"
        )
    outputs = policy["outputs"]
    expected_names = list(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        actual_names = (
            sorted(outputs) if isinstance(outputs, dict) else type(outputs).__name__
        )
        raise ValueError(
            "comparator outputs must exactly match case correctness outputs: "
            f"{actual_names} != {sorted(expected_names)}"
        )
    return {
        name: _ratio_allclose(name, outputs[name]) for name in expected_names
    }


def _program(module, case):
    return getattr(module, case["entry"])


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None,
         capture_outputs=False):
    return run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=str(golden_data) if golden_data else None,
        config=config,
        rtol=config.rtol,
        atol=config.atol,
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


def _oracle(oracle, values):
    tensors = {
        name: value for name, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    outputs = oracle.compute(tensors, {})
    for name, value in outputs.items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    return _workflow(case).prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: build_tensor_specs(
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


def evaluate_candidate(case_dir, sealed, case, candidate_workspace,
                       platform=None, phase="all"):
    return _workflow(case).evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: build_tensor_specs(
            fixture, _config(case_dir, case)
        ),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, _identity(case, "candidate")
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
