"""Case evaluator for Qwen3-14B top-k selection (selection_k=1)."""
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
    "workflow_qwen3_14b_topk_select_k1",
)


def _values(fixture: Path):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def _load_config(case_dir: Path):
    path = case_dir / "public" / "config.py"
    name = "qwen3_14b_topk_select_k1_eval_config"
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
        TensorSpec("logits", [config.BATCH, config.VOCAB], torch.float32,
                   init_value=values["logits"]),
        TensorSpec("sampling_control", [2], torch.int32,
                   init_value=values["sampling_control"]),
        TensorSpec("topk_values", [config.BATCH, config.TOPK], torch.float32),
        TensorSpec("topk_indices", [config.BATCH, config.TOPK], torch.int32),
    ]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _allclose(name, settings):
    required = {"method", "rtol", "atol"}
    if not isinstance(settings, dict) or set(settings) != required or settings.get("method") != "allclose":
        raise ValueError(f"allclose comparator {name!r} requires exactly method, rtol, atol")
    policy_rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_kwargs):
        if actual.shape != expected.shape:
            return False, f"allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        passed = torch.allclose(actual, expected, rtol=policy_rtol, atol=policy_atol)
        if passed:
            return True, f"torch.allclose(rtol={policy_rtol}, atol={policy_atol})"
        close = torch.isclose(actual, expected, rtol=policy_rtol, atol=policy_atol)
        return False, f"allclose: mismatches={int((~close).sum().item())}/{actual.numel()}"

    return compare


def _topk_pair(name, settings, expected_names):
    required = {"method", "name", "values_output", "dim", "descending"}
    if not isinstance(settings, dict) or set(settings) != required or settings.get("method") != "custom":
        raise ValueError(f"custom comparator {name!r} requires exactly {sorted(required)}")
    if settings["name"] != "topk_pair_compare":
        raise ValueError(f"unsupported custom comparator for {name!r}")
    values_output = settings["values_output"]
    if not isinstance(values_output, str) or values_output not in expected_names or values_output == name:
        raise ValueError(f"custom comparator {name!r} requires a distinct declared paired output")
    dim = settings["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"custom comparator {name!r}.dim must be an integer")
    if not -2 <= dim < 2:
        raise ValueError(f"custom comparator {name!r}.dim is out of range for rank-2 outputs")
    descending = settings["descending"]
    if not isinstance(descending, bool):
        raise ValueError(f"custom comparator {name!r}.descending must be boolean")

    def compare(actual, expected, *, actual_outputs, **_kwargs):
        if values_output not in actual_outputs:
            return False, f"topk_pair_compare: missing paired output {values_output!r}"
        if actual.shape != expected.shape:
            return False, f"topk_pair_compare: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        paired = actual_outputs[values_output]
        if paired.shape != actual.shape:
            return False, "topk_pair_compare: paired shape mismatch"
        dim_pos = dim if dim >= 0 else dim + actual.dim()
        if not 0 <= dim_pos < actual.dim():
            return False, f"topk_pair_compare: dim={dim} out of range"
        if torch.equal(actual, expected):
            return True, "exact index equality"
        a_idx = actual.cpu().movedim(dim_pos, -1).reshape(-1, actual.shape[dim_pos])
        e_idx = expected.cpu().movedim(dim_pos, -1).reshape(-1, expected.shape[dim_pos])
        vals = paired.cpu().to(torch.float32).movedim(dim_pos, -1).reshape(-1, paired.shape[dim_pos])
        mismatch = a_idx != e_idx
        if vals.shape != a_idx.shape:
            return False, "topk_pair_compare: paired shape mismatch"
        if vals.shape[1] >= 2:
            ordered = vals[:, :-1] >= vals[:, 1:] if descending else vals[:, :-1] <= vals[:, 1:]
            position_ok = torch.ones_like(mismatch)
            position_ok[:, 1:] &= ordered
            position_ok[:, :-1] &= ordered
        else:
            position_ok = torch.ones_like(mismatch)
        failed = mismatch & ~position_ok
        if bool(failed.any()):
            return False, f"topk_pair_compare: order violations={int(failed.sum().item())}"
        return True, "tie-preserving index permutation"

    compare.__name__ = "topk_pair_compare"
    return compare


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if (not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}
            or type(policy["schema_version"]) is not int or policy["schema_version"] != 1):
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy["outputs"]
    expected_names = list(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        raise ValueError("comparator outputs must exactly match correctness outputs")
    callbacks = {}
    for name in expected_names:
        settings = outputs[name]
        method = settings.get("method") if isinstance(settings, dict) else None
        if method == "allclose":
            callbacks[name] = _allclose(name, settings)
        elif method == "custom":
            callbacks[name] = _topk_pair(name, settings, expected_names)
        else:
            raise ValueError(f"unsupported comparator method for {name!r}: {method!r}")
    return callbacks


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
    return run(fn=fn, specs=specs, golden_fn=golden_fn,
               golden_data=str(golden_data) if golden_data else None,
               config=config, rtol=0, atol=0, compare_fn=compare_fn,
               save_data=False, capture_outputs=capture_outputs)


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, case_dir),
        load_ref=lambda: common.load_ref(case_dir, sealed, "qwen3_14b_topk_select_k1_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, "qwen3_14b_topk_select_k1_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, case_dir),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace,
                                                      "qwen3_14b_topk_select_k1_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
