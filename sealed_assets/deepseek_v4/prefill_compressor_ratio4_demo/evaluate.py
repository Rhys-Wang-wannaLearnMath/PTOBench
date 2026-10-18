"""Case-local evaluator for DeepSeek-V4 ratio-4 prefill compression."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common


def _config(case_dir: Path, case: dict):
    identity = case["case_id"].replace(".", "_")
    return common.load_module(case_dir / case["config"], f"{identity}_config")


def _values(fixture: Path):
    return {path.stem: torch.load(path, map_location="cpu", weights_only=True) for path in (fixture / "in").glob("*.pt")}


def build_tensor_specs(fixture: Path, config):
    values = _values(fixture)
    specs = []
    ordered_names = (
        "x", "kv_state", "score_state", "compress_state_block_table", "wkv", "wgate",
        "ape", "norm_w", "freqs_cos", "freqs_sin", "cmp_kv", "position_ids",
        "num_tokens", "cmp_slot_mapping", "state_slot_mapping",
    )
    for name in ordered_names:
        value = values[name]
        if name == "num_tokens":
            specs.append(ScalarSpec(name, torch.int32, value))
        else:
            specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    return specs


def _number(value, field, *, ratio=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (ratio and number > 1):
        raise ValueError(f"invalid comparator {field}")
    return number


def _compare(case_dir: Path, case: dict):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    names = case["correctness"]["outputs"]
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
        or not isinstance(policy["outputs"], dict)
        or set(policy["outputs"]) != set(names)
    ):
        raise ValueError("comparator schema/coverage mismatch")
    callbacks = {}
    for name in names:
        settings = policy["outputs"][name]
        if not isinstance(settings, dict) or settings.get("method") != "ratio_allclose":
            raise ValueError(f"unsupported comparator method for {name}")
        if set(settings) != {"method", "rtol", "atol", "max_error_ratio"}:
            raise ValueError(f"ratio_allclose fields for {name}")
        policy_rtol = _number(settings["rtol"], f"{name}.rtol")
        policy_atol = _number(settings["atol"], f"{name}.atol")
        max_error_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio", ratio=True)

        def compare(actual, expected, *, _rtol=policy_rtol, _atol=policy_atol,
                    _ratio=max_error_ratio, **kwargs):
            del kwargs
            if actual.shape != expected.shape:
                return False, f"shape mismatch: {tuple(actual.shape)}/{tuple(expected.shape)}"
            actual_f = actual.detach().cpu().to(torch.float32)
            expected_f = expected.detach().cpu().to(torch.float32)
            if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
                return False, "ratio_allclose: non-finite value"
            bad = (actual_f - expected_f).abs() > (_atol + _rtol * expected_f.abs())
            count = int(bad.sum().item())
            allowed = round(_ratio * actual_f.numel())
            return count <= allowed, f"mismatches={count}/{actual_f.numel()}, allowed={allowed}"

        callbacks[name] = compare
    return callbacks


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    outputs = oracle.compute({name: value for name, value in values.items() if isinstance(value, torch.Tensor)}, {})
    for name, value in outputs.items():
        values[name][:] = value


def _run(**kwargs):
    return run(rtol=0.0, atol=0.0, save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case["case_id"].replace(".", "_")
    workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", f"workflow_{identity}")
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"{identity}_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"{identity}_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    identity = case["case_id"].replace(".", "_")
    workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", f"workflow_{identity}")
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"{identity}_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run,
    )
