"""Case evaluator for Qwen3-14B sampled-token embedding lookup."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from golden import TensorSpec, run

import common


workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    "workflow_qwen3_14b_token_embed",
)


def _values(fixture: Path):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def _load_config(case_dir: Path):
    path = case_dir / "public" / "config.py"
    name = "qwen3_14b_token_embed_eval_config"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Case config: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def build_tensor_specs(fixture: Path, case_dir: Path):
    values = _values(fixture)
    config = _load_config(case_dir)
    return [
        TensorSpec(
            "sampled_ids",
            [config.BATCH, config.SAMPLED_IDS_PAD],
            torch.int32,
            init_value=values["sampled_ids"],
        ),
        TensorSpec(
            "embed_weight",
            [config.VOCAB, config.HIDDEN],
            torch.bfloat16,
            init_value=values["embed_weight"],
        ),
        TensorSpec(
            "next_hidden", [config.BATCH, config.HIDDEN], torch.bfloat16
        ),
    ]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _allclose_comparator(name, settings):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    method = settings.get("method")
    if method != "allclose":
        raise ValueError(f"unsupported comparator method for {name!r}: {method!r}")
    required = {"method", "rtol", "atol"}
    if set(settings) != required:
        raise ValueError(
            f"allclose comparator {name!r} requires exactly method, rtol, atol"
        )
    rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    atol = _finite_nonnegative(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, (
                f"torch.allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
            )
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
    path = case_dir / case["comparator"]
    policy = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
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
        name: _allclose_comparator(name, outputs[name])
        for name in expected_names
    }


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
        rtol=0,
        atol=0,
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


def _program(module, case):
    return getattr(module, case["entry"])


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
        build_specs=lambda fixture: build_tensor_specs(fixture, case_dir),
        load_ref=lambda: common.load_ref(
            case_dir, sealed, "qwen3_14b_token_embed_ref"
        ),
        load_oracle=lambda: common.load_oracle(
            case_dir, sealed, "qwen3_14b_token_embed_oracle"
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
        build_specs=lambda fixture: build_tensor_specs(fixture, case_dir),
        load_candidate=lambda: common.load_candidate(
            case_dir,
            case,
            candidate_workspace,
            "qwen3_14b_token_embed_candidate",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
