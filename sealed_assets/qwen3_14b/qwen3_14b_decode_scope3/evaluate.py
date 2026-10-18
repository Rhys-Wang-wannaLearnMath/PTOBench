"""Case-local evaluator for Qwen3-14B decode Scope 3."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common

workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    "workflow_qwen3_14b_decode_scope3",
)


def _config(case_dir: Path, case: dict):
    return common.load_module(case_dir / case["config"], "qwen3_14b_decode_scope3_config")


def _values(fixture: Path) -> dict[str, torch.Tensor]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config) -> list[TensorSpec]:
    values = _values(fixture)
    return [
        TensorSpec("attn_out", [config.BATCH, config.HIDDEN], torch.bfloat16,
                   init_value=values["attn_out"]),
        TensorSpec("hidden_states", [config.BATCH, config.HIDDEN], torch.bfloat16,
                   init_value=values["hidden_states"]),
        TensorSpec("wo", [config.HIDDEN, config.HIDDEN], torch.bfloat16,
                   init_value=values["wo"]),
        TensorSpec("post_rms_weight", [1, config.HIDDEN], torch.float32,
                   init_value=values["post_rms_weight"]),
        TensorSpec("w_gate", [config.HIDDEN, config.INTERMEDIATE], torch.bfloat16,
                   init_value=values["w_gate"]),
        TensorSpec("w_up", [config.HIDDEN, config.INTERMEDIATE], torch.bfloat16,
                   init_value=values["w_up"]),
        TensorSpec("w_down", [config.INTERMEDIATE, config.HIDDEN], torch.bfloat16,
                   init_value=values["w_down"]),
        TensorSpec("out", [config.BATCH, config.HIDDEN], torch.bfloat16),
    ]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _allclose_comparator(name: str, settings: object):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    if set(settings) != {"method", "rtol", "atol"}:
        raise ValueError(f"allclose comparator {name!r} requires exactly method, rtol, atol")
    if settings.get("method") != "allclose":
        raise ValueError(f"unsupported comparator method for {name!r}: {settings.get('method')!r}")
    rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    atol = _finite_nonnegative(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, f"allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        passed = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
        if passed:
            return True, f"torch.allclose(rtol={rtol}, atol={atol})"
        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        mismatches = int((~close).sum().item())
        return False, (
            f"torch.allclose(rtol={rtol}, atol={atol}): "
            f"mismatches={mismatches}/{actual.numel()}"
        )

    return compare


def _compare(case_dir: Path, case: dict):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if (not isinstance(policy, dict)
            or set(policy) != {"schema_version", "outputs"}
            or policy["schema_version"] != 1):
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy["outputs"]
    expected_names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        raise ValueError("comparator outputs must exactly match case correctness outputs")
    callbacks = {name: _allclose_comparator(name, outputs[name]) for name in expected_names}
    if set(callbacks) != set(expected_names):
        raise ValueError("comparator callback coverage mismatch")
    return callbacks


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None,
         capture_outputs=False):
    return run(fn=fn, specs=specs, golden_fn=golden_fn,
               golden_data=str(golden_data) if golden_data else None,
               config=config, compare_fn=compare_fn, save_data=False,
               capture_outputs=capture_outputs)


def _program(module, case):
    return getattr(module, case["entry"])()


def _oracle(oracle, values):
    inputs = {name: value for name, value in values.items()
              if isinstance(value, torch.Tensor) and name != "out"}
    for name, answer in oracle.compute(inputs, {}).items():
        values[name][:] = answer


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, "qwen3_14b_scope3_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, "qwen3_14b_scope3_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, "qwen3_14b_scope3_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run,
    )
