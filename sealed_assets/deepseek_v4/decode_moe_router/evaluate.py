"""Case-local evaluator for the frozen DeepSeek-V4 decode MoE router."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common

workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator/workflow.py",
    "workflow_deepseek_v4_decode_moe_router",
)


def _values(fixture: Path):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def _config(case_dir: Path, case: dict):
    return common.load_module(case_dir / case["config"], "decode_moe_router_config")


def build_tensor_specs(fixture: Path, config):
    values = _values(fixture)
    return [
        TensorSpec("x_hc", list(config.X_HC_SHAPE), torch.bfloat16, init_value=values["x_hc"]),
        TensorSpec("hc_ffn_fn", list(config.HC_FFN_FN_SHAPE), torch.float32, init_value=values["hc_ffn_fn"]),
        TensorSpec("hc_ffn_scale", list(config.HC_FFN_SCALE_SHAPE), torch.float32, init_value=values["hc_ffn_scale"]),
        TensorSpec("hc_ffn_base", list(config.HC_FFN_BASE_SHAPE), torch.float32, init_value=values["hc_ffn_base"]),
        TensorSpec("norm_w", list(config.NORM_W_SHAPE), torch.float32, init_value=values["norm_w"]),
        TensorSpec("gate_w", list(config.GATE_W_SHAPE), torch.float32, init_value=values["gate_w"]),
        TensorSpec("gate_bias", list(config.GATE_BIAS_SHAPE), torch.float32, init_value=values["gate_bias"]),
        TensorSpec("tid2eid", list(config.TID2EID_SHAPE), torch.int32, init_value=values["tid2eid"]),
        TensorSpec("input_ids", list(config.INPUT_IDS_SHAPE), torch.int64, init_value=values["input_ids"]),
        TensorSpec("x_norm", list(config.X_NORM_SHAPE), torch.bfloat16),
        TensorSpec("indices", list(config.INDICES_SHAPE), torch.int32),
        TensorSpec("weights", list(config.WEIGHTS_SHAPE), torch.float32),
        TensorSpec("post_ffn", list(config.POST_FFN_SHAPE), torch.float32),
        TensorSpec("comb_ffn", list(config.COMB_FFN_SHAPE), torch.float32),
    ]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _allclose(name, settings):
    if set(settings) != {"method", "rtol", "atol"} or settings.get("method") != "allclose":
        raise ValueError(f"allclose comparator {name!r} requires exactly method, rtol, atol")
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
        return False, f"allclose: mismatches={mismatches}/{actual.numel()}, rtol={rtol}, atol={atol}"

    return compare


def _topk_pair_compare(name, settings, expected_names):
    required = {"method", "name", "values_output", "dim", "descending"}
    if set(settings) != required or settings.get("method") != "custom":
        raise ValueError(f"custom comparator {name!r} requires exactly {sorted(required)}")
    if settings["name"] != "topk_pair_compare":
        raise ValueError(f"unsupported custom comparator for {name!r}")
    values_output = settings["values_output"]
    if values_output not in expected_names:
        raise ValueError(f"custom comparator {name!r} references undeclared output")
    dim = settings["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"custom comparator {name!r}.dim must be an integer")
    descending = settings["descending"]
    if not isinstance(descending, bool):
        raise ValueError(f"custom comparator {name!r}.descending must be boolean")

    def compare(actual, expected, *, actual_outputs, **_):
        if values_output not in actual_outputs:
            return False, f"topk_pair_compare: missing paired output {values_output!r}"
        if actual.shape != expected.shape:
            return False, f"topk_pair_compare: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        paired = actual_outputs[values_output]
        if paired.shape != actual.shape:
            return False, f"topk_pair_compare: idx/values shape mismatch {tuple(actual.shape)}/{tuple(paired.shape)}"
        if torch.equal(actual, expected):
            return True, "exact index equality"
        ndim = actual.dim()
        dim_pos = dim if dim >= 0 else dim + ndim
        if not 0 <= dim_pos < ndim:
            return False, f"topk_pair_compare: dim={dim} out of range"
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
        if not bool(failed.any()):
            return True, "tie-preserving index permutation"
        return False, f"topk_pair_compare: order violations={int(failed.sum().item())}"

    compare.__name__ = "topk_pair_compare"
    return compare


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or policy["schema_version"] != 1:
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy["outputs"]
    expected_names = list(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        raise ValueError("comparator outputs must exactly match correctness outputs")
    callbacks = {}
    for name in expected_names:
        settings = outputs[name]
        if not isinstance(settings, dict):
            raise ValueError(f"comparator output {name!r} must be an object")
        method = settings.get("method")
        if method == "allclose":
            callbacks[name] = _allclose(name, settings)
        elif method == "custom":
            callbacks[name] = _topk_pair_compare(name, settings, expected_names)
        else:
            raise ValueError(f"unsupported comparator method for {name!r}: {method!r}")
    return callbacks


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
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


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, _config(case_dir, case)),
        load_ref=lambda: common.load_ref(case_dir, sealed, "decode_moe_router_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, "decode_moe_router_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, _config(case_dir, case)),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, "decode_moe_router_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run,
    )
