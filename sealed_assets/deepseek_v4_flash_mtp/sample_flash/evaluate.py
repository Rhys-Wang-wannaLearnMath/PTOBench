"""Case-local ABI and comparator boundary for deepseek_v4_flash_mtp.sample_flash."""

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common
import workflow


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate comparator key: {key}")
        result[key] = value
    return result


def _comparator_callbacks(policy, output_names):
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}:
        raise ValueError("Comparator requires exactly schema_version and outputs")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("Unsupported comparator schema")
    rules = policy["outputs"]
    if not isinstance(rules, dict) or set(rules) != set(output_names):
        raise ValueError("Comparator output coverage differs from the Case contract")
    callbacks = {}
    for name, rule in rules.items():
        if not isinstance(rule, dict) or rule.get("method") != "allclose":
            raise ValueError("This Case declares only allclose")
        if set(rule) != {"method", "rtol", "atol"}:
            raise ValueError("allclose requires exactly method, rtol and atol")
        for field in ("rtol", "atol"):
            value = rule[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {field}")
        callbacks[name] = _allclose_callback(rule["rtol"], rule["atol"])
    return callbacks


def _allclose_callback(policy_rtol, policy_atol):
    def compare(actual, expected, **kwargs):
        passed = torch.allclose(actual, expected, rtol=policy_rtol, atol=policy_atol)
        return passed, "" if passed else "Full output differs under public allclose policy"
    return compare


def make_compare(case_dir, case):
    policy = json.loads(
        (Path(case_dir) / case["comparator"]).read_text(),
        object_pairs_hook=_unique_object,
    )
    return _comparator_callbacks(policy, case["correctness"]["outputs"])


def _build_specs(case_dir, case, fixture):
    identity = case["case_id"].replace(".", "_")
    config = common.load_isolated_module(
        Path(case_dir) / case["config"], f"evaluator_{identity}_config",
    )
    specs = []
    for name, shape, dtype in (
        ("logits", config.LOGITS_SHAPE, config.LOGITS_DTYPE),
        ("temperatures", config.TEMPERATURES_SHAPE, config.TEMPERATURES_DTYPE),
        ("top_ks", config.TOP_KS_SHAPE, config.TOP_KS_DTYPE),
        ("seeds", config.SEEDS_SHAPE, config.SEEDS_DTYPE),
        ("positions", config.POSITIONS_SHAPE, config.POSITIONS_DTYPE),
        ("sampled_ids", config.SAMPLED_IDS_SHAPE, config.SAMPLED_IDS_DTYPE),
    ):
        if name in case["correctness"]["outputs"]:
            specs.append(TensorSpec(name, list(shape), getattr(torch, dtype)))
        else:
            value = torch.load(Path(fixture) / "in" / f"{name}.pt", weights_only=True)
            if tuple(value.shape) != tuple(shape) or value.dtype != getattr(torch, dtype):
                raise ValueError(f"Fixture ABI mismatch for {name}")
            specs.append(TensorSpec(name, list(shape), getattr(torch, dtype),
                                    init_value=lambda value=value: value.clone()))
    return specs


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    outputs = oracle.run_oracle(values)
    values["sampled_ids"].copy_(outputs["sampled_ids"])


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: _build_specs(case_dir, case, fixture),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"ref_{identity}"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"oracle_{identity}"),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case),
        run=run, oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: _build_specs(case_dir, case, fixture),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, Path(candidate_workspace), f"candidate_{identity}",
        ),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run,
    )
