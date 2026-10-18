"""Case-local evaluator for qwen3_32b.qwen3_32b_decode_4d."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common


workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    "workflow_qwen3_32b_decode_4d",
)


def _legacy_specs(case_dir: Path, sealed: Path, case: dict) -> list:
    builder = common.load_module(
        sealed / "fixture_builder.py", "qwen3_32b_decode_4d_fixture_specs"
    )
    return builder.build(case_dir, sealed, case, 0)


def build_tensor_specs(fixture: Path, case_dir: Path | None = None,
                       sealed: Path | None = None, case: dict | None = None):
    """Recreate the official blocked ABI while binding cached fixture inputs."""
    if case_dir is None or sealed is None or case is None:
        raise ValueError("fixture spec construction requires Case paths")
    values = {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }
    specs = []
    for legacy in _legacy_specs(case_dir, sealed, case):
        if hasattr(legacy, "shape"):
            init_value = values.get(legacy.name)
            specs.append(TensorSpec(legacy.name, legacy.shape, legacy.dtype,
                                    init_value=init_value))
        else:
            value = values.get(legacy.name)
            specs.append(ScalarSpec(legacy.name, legacy.dtype, value))
    return specs


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _allclose_comparator(name, settings):
    if not isinstance(settings, dict) or set(settings) != {"method", "rtol", "atol"}:
        raise ValueError(f"allclose comparator {name!r} requires exactly method, rtol, atol")
    if settings.get("method") != "allclose":
        raise ValueError(f"unsupported comparator method for {name!r}: {settings.get('method')!r}")
    rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    atol = _finite_nonnegative(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, f"allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
        if passed:
            return True, f"torch.allclose(rtol={rtol}, atol={atol})"
        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        mismatches = int((~close).sum().item())
        return False, (
            f"torch.allclose(rtol={rtol}, atol={atol}): "
            f"mismatches={mismatches}/{actual.numel()}"
        )

    return compare


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if (not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}
            or policy["schema_version"] != 1):
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy["outputs"]
    expected_names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        raise ValueError("comparator outputs must exactly match case correctness outputs")
    return {name: _allclose_comparator(name, outputs[name]) for name in expected_names}


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None,
         capture_outputs=False):
    return run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=str(golden_data) if golden_data else None,
        config=config,
        rtol=getattr(config, "rtol", 1e-5),
        atol=getattr(config, "atol", 1e-5),
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


def _program(module, case):
    return getattr(module, case["entry"])()


def _oracle(oracle, values):
    tensors = {name: value for name, value in values.items()
               if isinstance(value, torch.Tensor)}
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    slug = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, case_dir, sealed, case),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"{slug}_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"{slug}_oracle"),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None,
                       phase="all"):
    slug = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, case_dir, sealed, case),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, f"{slug}_candidate"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
