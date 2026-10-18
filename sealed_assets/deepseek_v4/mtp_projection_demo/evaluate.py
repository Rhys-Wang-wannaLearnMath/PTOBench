from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common

workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator/workflow.py", "workflow_deepseek_v4_mtp_projection_demo")


def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], "deepseek_v4_mtp_projection_demo_config")


def _fixture_values(fixture):
    return {path.stem: torch.load(path, map_location="cpu", weights_only=True)
            for path in (fixture / "in").glob("*.pt")}


def build_tensor_specs(fixture, config):
    values = _fixture_values(fixture)
    input_names = (
        "hidden_states", "prev_hidden_states", "enorm_w", "hnorm_w",
        "e_proj_w", "e_proj_w_scale", "e_proj_smooth", "h_proj_w",
        "h_proj_w_scale", "h_proj_smooth",
    )
    specs = [TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name])
             for name in input_names]
    specs.append(TensorSpec("hidden_states_out", [config.DECODE_BATCH, config.DECODE_SEQ, config.D], torch.bfloat16))
    return specs


def _number(value, field, *, ratio=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (ratio and result > 1):
        raise ValueError(f"invalid comparator {field}")
    return result


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    names = case["correctness"]["outputs"]
    if set(policy) != {"schema_version", "outputs"} or policy.get("schema_version") != 1:
        raise ValueError("comparator schema mismatch")
    outputs = policy.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator output coverage mismatch")
    callbacks = {}
    for name, settings in outputs.items():
        if not isinstance(settings, dict) or set(settings) != {"method", "rtol", "atol", "max_error_ratio"}:
            raise ValueError(f"invalid {name} comparator fields")
        if settings.get("method") != "ratio_allclose":
            raise ValueError(f"unsupported comparator method for {name}")
        policy_rtol = _number(settings["rtol"], f"{name}.rtol")
        policy_atol = _number(settings["atol"], f"{name}.atol")
        max_error_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio", ratio=True)

        def compare(actual, expected, **_kwargs):
            if actual.shape != expected.shape:
                return False, f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
            actual_f = actual.detach().cpu().float()
            expected_f = expected.detach().cpu().float()
            if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
                return False, "non-finite value"
            bad = (actual_f - expected_f).abs() > policy_atol + policy_rtol * expected_f.abs()
            count = int(bad.sum().item())
            allowed = round(max_error_ratio * actual_f.numel())
            return count <= allowed, f"mismatches={count}/{actual_f.numel()}, allowed={allowed}"

        callbacks[name] = compare
    return callbacks


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value


def _run(**kwargs):
    return run(rtol=0.0, atol=0.0, save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, identity + "_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + "_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle_golden)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + "_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
