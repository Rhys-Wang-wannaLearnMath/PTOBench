"""DSpark Markov ABI and Case-local allclose policy execution."""

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
    if not isinstance(rules, dict) or set(rules) != set(case["correctness"]["outputs"]):
        raise ValueError("comparator output coverage disagrees with contract")
    callbacks = {}
    for output_name, rule in rules.items():
        if not isinstance(rule, dict) or rule.get("method") != "allclose":
            raise ValueError("unsupported comparator method")
        if set(rule) != {"method", "rtol", "atol"}:
            raise ValueError("invalid allclose fields")
        for field in ("rtol", "atol"):
            value = rule[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid comparator {field}")
        callbacks[output_name] = _allclose_callback(rule)
    return callbacks


def _allclose_callback(rule):
    policy_rtol = rule["rtol"]
    policy_atol = rule["atol"]

    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, "output shape mismatch"
        passed = torch.allclose(
            actual.detach().cpu(), expected.detach().cpu(),
            rtol=policy_rtol, atol=policy_atol,
        )
        return passed, "" if passed else (
            f"allclose mismatch: rtol={policy_rtol}, atol={policy_atol}"
        )

    return compare


def _config(case_dir, case):
    identity = case["case_id"].replace(".", "_")
    config = common.load_isolated_module(
        case_dir / case["config"], f"config_{identity}"
    )
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config identity disagrees with Case")
    return config


def build_specs(fixture, config):
    specs = []
    for name, shape in config.INPUT_SHAPES.items():
        value = torch.load(
            fixture / "in" / f"{name}.pt", map_location="cpu", weights_only=True
        )
        dtype = getattr(torch, config.INPUT_DTYPES[name])
        if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            raise ValueError(f"Fixture ABI mismatch: {name}")
        specs.append(TensorSpec(name, list(shape), dtype, init_value=value))
    for name, shape in config.OUTPUT_SHAPES.items():
        specs.append(TensorSpec(name, list(shape), getattr(torch, config.OUTPUT_DTYPES[name])))
    return specs


def _make_program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    outputs = oracle.compute(values)
    if set(outputs) != {"draft_token_ids"}:
        raise ValueError("Oracle output coverage disagrees with contract")
    values["draft_token_ids"].copy_(outputs["draft_token_ids"])


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
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, f"candidate_{identity}"
        ),
        make_program=_make_program,
        make_compare=lambda: make_compare(case_dir, case), run=run,
    )
