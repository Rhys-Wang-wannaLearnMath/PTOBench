"""Case-local evaluator for qwen3_32b.qwen3_32b_decode."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common


workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    "workflow_qwen3_32b_decode",
)


def _values(fixture: Path):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path):
    values = _values(fixture)
    return [
        TensorSpec("hidden_states", [16, 8192], torch.bfloat16, init_value=values["hidden_states"]),
        TensorSpec("input_rms_weight", [1, 8192], torch.float32, init_value=values["input_rms_weight"]),
        TensorSpec("wq", [8192, 8192], torch.bfloat16, init_value=values["wq"]),
        TensorSpec("wk", [8192, 1024], torch.bfloat16, init_value=values["wk"]),
        TensorSpec("wv", [8192, 1024], torch.bfloat16, init_value=values["wv"]),
        TensorSpec("seq_lens", [16], torch.int32, init_value=values["seq_lens"]),
        TensorSpec("rope_cos", [4096, 128], torch.float32, init_value=values["rope_cos"]),
        TensorSpec("rope_sin", [4096, 128], torch.float32, init_value=values["rope_sin"]),
        TensorSpec("k_cache", [524288, 128], torch.bfloat16, init_value=values["k_cache"]),
        TensorSpec("v_cache", [524288, 128], torch.bfloat16, init_value=values["v_cache"]),
        TensorSpec("wo", [8192, 8192], torch.bfloat16, init_value=values["wo"]),
        TensorSpec("post_rms_weight", [1, 8192], torch.float32, init_value=values["post_rms_weight"]),
        TensorSpec("w_gate", [8192, 25600], torch.bfloat16, init_value=values["w_gate"]),
        TensorSpec("w_up", [8192, 25600], torch.bfloat16, init_value=values["w_up"]),
        TensorSpec("w_down", [25600, 8192], torch.bfloat16, init_value=values["w_down"]),
        TensorSpec("out", [16, 8192], torch.bfloat16),
    ]


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None,
         capture_outputs=False):
    return run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=str(golden_data) if golden_data else None,
        config=config,
        rtol=0.003,
        atol=0.003,
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


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
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or policy["schema_version"] != 1:
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy["outputs"]
    expected_names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        raise ValueError("comparator outputs must exactly match case correctness outputs")
    return {name: _allclose_comparator(name, outputs[name]) for name in expected_names}


def _program(module, case):
    return getattr(module, case["entry"])()


def _oracle(oracle, values):
    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    slug = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=build_tensor_specs,
        load_ref=lambda: common.load_ref(case_dir, sealed, f"{slug}_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"{slug}_oracle"),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    slug = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=build_tensor_specs,
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, f"{slug}_candidate"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
