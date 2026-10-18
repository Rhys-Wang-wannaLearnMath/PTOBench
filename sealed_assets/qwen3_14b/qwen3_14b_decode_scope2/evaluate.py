"""Case-local evaluator for Qwen3-14B decode Scope 2."""

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
    "workflow_qwen3_14b_decode_scope2",
)


def _config(case_dir: Path, case: dict):
    path = case_dir / case["config"]
    spec = importlib.util.spec_from_file_location(
        "qwen3_14b_decode_scope2_eval_config", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Case config: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def _values(fixture: Path):
    return {
        path.stem: torch.load(
            path, map_location="cpu", weights_only=True
        )
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config):
    values = _values(fixture)
    return [
        TensorSpec(
            "q_proj",
            [config.BATCH, config.HIDDEN],
            torch.float32,
            init_value=values["q_proj"],
        ),
        TensorSpec(
            "k_proj",
            [config.BATCH, config.KV_HIDDEN],
            torch.float32,
            init_value=values["k_proj"],
        ),
        TensorSpec(
            "v_proj",
            [config.BATCH, config.KV_HIDDEN],
            torch.float32,
            init_value=values["v_proj"],
        ),
        TensorSpec(
            "q_norm_weight",
            [1, config.HEAD_DIM],
            torch.float32,
            init_value=values["q_norm_weight"],
        ),
        TensorSpec(
            "k_norm_weight",
            [1, config.HEAD_DIM],
            torch.float32,
            init_value=values["k_norm_weight"],
        ),
        TensorSpec(
            "seq_lens",
            [config.BATCH],
            torch.int32,
            init_value=values["seq_lens"],
        ),
        TensorSpec(
            "block_table",
            [config.NUM_BLOCKS],
            torch.int32,
            init_value=values["block_table"],
        ),
        TensorSpec(
            "slot_mapping",
            [config.BATCH],
            torch.int32,
            init_value=values["slot_mapping"],
        ),
        TensorSpec(
            "rope_cos",
            [config.MAX_SEQ, config.HEAD_DIM],
            torch.float32,
            init_value=values["rope_cos"],
        ),
        TensorSpec(
            "rope_sin",
            [config.MAX_SEQ, config.HEAD_DIM],
            torch.float32,
            init_value=values["rope_sin"],
        ),
        TensorSpec(
            "k_cache",
            [config.CACHE_ROWS, config.HEAD_DIM],
            torch.bfloat16,
            init_value=values["k_cache"],
        ),
        TensorSpec(
            "v_cache",
            [config.CACHE_ROWS, config.HEAD_DIM],
            torch.bfloat16,
            init_value=values["v_cache"],
        ),
        TensorSpec(
            "attn_out",
            [config.BATCH, config.HIDDEN],
            torch.bfloat16,
        ),
    ]


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"comparator {field} must be finite and nonnegative"
        )
    return value


def _compare(case_dir, case):
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
    expected_names = case["correctness"]["outputs"]
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        raise ValueError(
            "comparator outputs must exactly match case correctness outputs"
        )
    callbacks = {}
    for name in expected_names:
        settings = outputs[name]
        if not isinstance(settings, dict) or set(settings) != {
            "method",
            "rtol",
            "atol",
        }:
            raise ValueError(
                f"comparator {name!r} requires exactly method, rtol, atol"
            )
        if settings["method"] != "allclose":
            raise ValueError(
                f"unsupported comparator method for {name!r}: "
                f"{settings['method']!r}"
            )
        rtol = _number(settings["rtol"], f"{name}.rtol")
        atol = _number(settings["atol"], f"{name}.atol")

        def compare(actual, expected, *, _rtol=rtol, _atol=atol, **_):
            if actual.shape != expected.shape:
                return (
                    False,
                    "allclose: "
                    f"shape={tuple(actual.shape)}/{tuple(expected.shape)}",
                )
            passed = bool(
                torch.all(
                    torch.abs(actual - expected)
                    <= _atol + _rtol * torch.abs(expected)
                )
            )
            if passed:
                return (
                    True,
                    f"elementwise allclose(rtol={_rtol}, atol={_atol})",
                )
            tolerance = _atol + _rtol * torch.abs(expected)
            mismatches = int(
                (torch.abs(actual - expected) > tolerance).sum().item()
            )
            return (
                False,
                "elementwise allclose"
                f"(rtol={_rtol}, atol={_atol}): "
                f"mismatches={mismatches}/{actual.numel()}",
            )

        callbacks[name] = compare
    return callbacks


def _run(
    *,
    fn,
    specs,
    config,
    compare_fn,
    golden_data=None,
    golden_fn=None,
    capture_outputs=False,
):
    return run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=str(golden_data) if golden_data else None,
        config=config,
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


def _program(module, case):
    return getattr(module, case["entry"])()


def _oracle(oracle, values):
    tensors = {
        name: value
        for name, value in values.items()
        if isinstance(value, torch.Tensor) and name != "attn_out"
    }
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(
            case_dir, sealed, "qwen3_14b_decode_scope2_ref"
        ),
        load_oracle=lambda: common.load_oracle(
            case_dir, sealed, "qwen3_14b_decode_scope2_oracle"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(
    case_dir,
    sealed,
    case,
    candidate_workspace,
    platform=None,
    phase="all",
):
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(
            case_dir,
            case,
            candidate_workspace,
            "qwen3_14b_decode_scope2_candidate",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
