import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common
import workflow


def make_compare(case_dir: Path, case: dict):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}:
        raise ValueError("invalid comparator top-level fields")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("unsupported comparator schema")
    rules = policy["outputs"]
    expected_names = set(case["correctness"]["outputs"])
    if not isinstance(rules, dict) or set(rules) != expected_names:
        raise ValueError("comparator output coverage disagrees with contract")
    callbacks = {}
    for output_name, rule in rules.items():
        if not isinstance(rule, dict) or rule.get("method") != "ratio_allclose":
            raise ValueError("unsupported comparator method")
        if set(rule) != {"method", "rtol", "atol", "max_error_ratio"}:
            raise ValueError("invalid ratio_allclose fields")
        for field in ("rtol", "atol"):
            value = rule[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid comparator {field}")
        ratio = rule["max_error_ratio"]
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("invalid comparator max_error_ratio")
        callbacks[output_name] = _ratio_callback(rule)
    return callbacks


def _ratio_callback(rule):
    policy_rtol = rule["rtol"]
    policy_atol = rule["atol"]
    policy_ratio = rule["max_error_ratio"]

    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, "output shape mismatch"
        actual_fp32 = actual.detach().cpu().float()
        expected_fp32 = expected.detach().cpu().float()
        if not bool(torch.isfinite(actual_fp32).all()) or not bool(torch.isfinite(expected_fp32).all()):
            return False, "non-finite output"
        tolerance = policy_atol + policy_rtol * expected_fp32.abs()
        mismatches = int((actual_fp32 - expected_fp32).abs().gt(tolerance).sum().item())
        allowed = round(policy_ratio * expected_fp32.numel())
        passed = mismatches <= allowed
        return passed, "" if passed else f"ratio_allclose mismatch: {mismatches} > {allowed}"

    return compare


def _config(case_dir, case):
    identity = case["case_id"].replace(".", "_")
    config = common.load_isolated_module(case_dir / case["config"], f"config_{identity}")
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config identity disagrees with Case")
    return config


def build_specs(fixture, config):
    shapes = {
        "main_hidden": (config.FIXTURE_TOKENS, config.MAIN_HIDDEN_DIM),
        "main_proj_w": (config.D, config.MAIN_HIDDEN_DIM),
        "main_norm_w": (config.D,),
    }
    specs = []
    for name, shape in shapes.items():
        value = torch.load(fixture / "in" / f"{name}.pt", map_location="cpu", weights_only=True)
        if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
            raise ValueError(f"Fixture ABI mismatch: {name}")
        specs.append(TensorSpec(name, list(shape), torch.bfloat16, init_value=value))
    specs.append(TensorSpec("main_x", [config.FIXTURE_TOKENS, config.D], torch.bfloat16))
    return specs


def _make_program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    outputs = oracle.compute(values)
    if set(outputs) != {"main_x"}:
        raise ValueError("Oracle output coverage disagrees with contract")
    values["main_x"].copy_(outputs["main_x"])


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"ref_{identity}"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"oracle_{identity}"),
        make_program=_make_program,
        make_compare=lambda: make_compare(case_dir, case),
        run=run, oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"candidate_{identity}"),
        make_program=_make_program,
        make_compare=lambda: make_compare(case_dir, case), run=run,
    )
