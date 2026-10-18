"""Case-local evaluator for the legacy DeepSeek-V4 ratio-128 decoder."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common

workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    "workflow_deepseek_v4_deepseek_v4_decode_compressor_ratio128",
)


def _config(fixture: Path):
    sealed = fixture.parents[1]
    relative_case = sealed.relative_to(common.sealed_root())
    case_dir = common.BENCHMARK_ROOT / "cases" / relative_case
    return common.load_module(
        case_dir / "public" / "config.py",
        "deepseek_v4_deepseek_v4_decode_compressor_ratio128_config",
    )


def _values(fixture: Path):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path):
    config = _config(fixture)
    values = _values(fixture)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }
    return [
        TensorSpec("x", config.X_SHAPE, dtype[config.X_DTYPE], init_value=values["x"]),
        TensorSpec(
            "kv_state", config.STATE_SHAPE, dtype[config.STATE_DTYPE],
            init_value=values["kv_state"],
        ),
        TensorSpec(
            "score_state", config.STATE_SHAPE, dtype[config.STATE_DTYPE],
            init_value=values["score_state"],
        ),
        TensorSpec("wkv", config.W_SHAPE, dtype[config.W_DTYPE], init_value=values["wkv"]),
        TensorSpec("wgate", config.W_SHAPE, dtype[config.W_DTYPE], init_value=values["wgate"]),
        TensorSpec("ape", config.APE_SHAPE, dtype[config.APE_DTYPE], init_value=values["ape"]),
        TensorSpec(
            "norm_w", config.NORM_W_SHAPE, dtype[config.NORM_W_DTYPE],
            init_value=values["norm_w"],
        ),
        TensorSpec("cos", config.ROPE_SHAPE, dtype[config.ROPE_DTYPE], init_value=values["cos"]),
        TensorSpec("sin", config.ROPE_SHAPE, dtype[config.ROPE_DTYPE], init_value=values["sin"]),
        TensorSpec(
            "hadamard", config.HADAMARD_SHAPE, dtype[config.HADAMARD_DTYPE],
            init_value=values["hadamard"],
        ),
        ScalarSpec("start_pos", dtype[config.START_POS_DTYPE], values["start_pos"]),
        TensorSpec("out", config.OUT_SHAPE, dtype[config.OUT_DTYPE]),
    ]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _allclose(name, settings):
    if not isinstance(settings, dict) or set(settings) != {"method", "rtol", "atol"}:
        raise ValueError(
            f"allclose comparator {name!r} requires exactly method, rtol, atol",
        )
    if settings["method"] != "allclose":
        raise ValueError(
            f"unsupported comparator method for {name!r}: {settings['method']!r}",
        )
    policy_rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, f"allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        passed = torch.allclose(
            actual, expected, rtol=policy_rtol, atol=policy_atol,
        )
        if passed:
            return True, f"torch.allclose(rtol={policy_rtol}, atol={policy_atol})"
        close = torch.isclose(
            actual, expected, rtol=policy_rtol, atol=policy_atol,
        )
        mismatches = int((~close).sum().item())
        return False, f"allclose: mismatches={mismatches}/{actual.numel()}"

    return compare


def _compare(case_dir, case):
    policy = json.loads(
        (case_dir / case["comparator"]).read_text(encoding="utf-8"),
    )
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or policy["schema_version"] != 1
    ):
        raise ValueError(
            "comparator.json must contain only schema_version=1 and outputs",
        )
    expected = list(case["correctness"]["outputs"])
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError(
            "comparator outputs must exactly match case correctness outputs",
        )
    return {name: _allclose(name, outputs[name]) for name in expected}


def _oracle(oracle, values):
    tensors = {
        name: value for name, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    outputs = oracle.compute(tensors, {"start_pos": values["start_pos"]})
    for name, value in outputs.items():
        values[name][:] = value


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


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=build_tensor_specs,
        load_ref=lambda: common.load_ref(
            case_dir,
            sealed,
            "deepseek_v4_deepseek_v4_decode_compressor_ratio128_ref",
        ),
        load_oracle=lambda: common.load_oracle(
            case_dir,
            sealed,
            "deepseek_v4_deepseek_v4_decode_compressor_ratio128_oracle",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(
    case_dir, sealed, case, candidate_workspace, platform=None, phase="all",
):
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=build_tensor_specs,
        load_candidate=lambda: common.load_candidate(
            case_dir,
            case,
            candidate_workspace,
            "deepseek_v4_deepseek_v4_decode_compressor_ratio128_candidate",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
