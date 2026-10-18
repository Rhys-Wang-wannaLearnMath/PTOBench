"""Case-local evaluator for the Qwen3-14B single-layer decode Case."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common

INPUT_NAMES = (
    "hidden_states", "input_rms_weight", "wq", "wk", "wv",
    "q_norm_weight", "k_norm_weight", "seq_lens", "block_table",
    "slot_mapping", "rope_cos", "rope_sin", "k_cache", "v_cache",
    "wo", "w_gate", "w_up", "w_down", "post_rms_weight",
)


def _config(case_dir: Path, case: dict):
    identity = case["case_id"].replace(".", "_")
    return common.load_module(case_dir / case["config"], f"{identity}_config")


def _values(fixture: Path) -> dict[str, torch.Tensor]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config) -> list[TensorSpec]:
    values = _values(fixture)
    specs = []
    for name in INPUT_NAMES:
        value = values[name]
        specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    specs.append(TensorSpec("out", [config.BATCH, config.HIDDEN], torch.bfloat16))
    return specs


def _number(value, field: str, *, ratio: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (ratio and value > 1):
        raise ValueError(f"comparator {field} has invalid range")
    return value


def _callback(name: str, settings: dict):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if settings.get("method") != "ratio_allclose" or set(settings) != required:
        raise ValueError(f"ratio_allclose requires exactly {sorted(required)} for {name!r}")
    policy_rtol = _number(settings["rtol"], f"{name}.rtol")
    policy_atol = _number(settings["atol"], f"{name}.atol")
    max_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio", ratio=True)

    def compare(actual, expected, **_runner_fallback):
        if actual.shape != expected.shape:
            return False, "ratio_allclose: shape mismatch"
        actual = actual.detach().cpu().float()
        expected = expected.detach().cpu().float()
        if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
            return False, "ratio_allclose: non-finite value"
        bad = (actual - expected).abs() > (policy_atol + policy_rtol * expected.abs())
        count = int(bad.sum().item())
        allowed = round(max_ratio * actual.numel())
        return count <= allowed, (
            f"ratio_allclose: mismatches={count}/{actual.numel()}, allowed={allowed}, "
            f"rtol={policy_rtol}, atol={policy_atol}, max_error_ratio={max_ratio}"
        )

    return compare


def _compare(case_dir: Path, case: dict):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if (not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}
            or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1):
        raise ValueError("comparator.json top-level schema mismatch")
    names = case["correctness"]["outputs"]
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator output coverage mismatch")
    return {name: _callback(name, outputs[name]) for name in names}


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values: dict) -> None:
    inputs = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, value in oracle.compute(inputs, {}).items():
        values[name][:] = value


def _run(**kwargs):
    return run(rtol=0.0, atol=0.0, save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case["case_id"].replace(".", "_")
    workflow = common.load_module(
        common.BENCHMARK_ROOT / "evaluator/workflow.py", f"workflow_{identity}"
    )
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"{identity}_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"{identity}_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=lambda **kwargs: _run(
            reset_input_names=tuple(case["state"]["initial_state"]), **kwargs
        ), oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    identity = case["case_id"].replace(".", "_")
    workflow = common.load_module(
        common.BENCHMARK_ROOT / "evaluator/workflow.py", f"workflow_{identity}_candidate"
    )
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"{identity}_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=lambda **kwargs: _run(
            reset_input_names=tuple(case["state"]["initial_state"]), **kwargs
        ),
    )
