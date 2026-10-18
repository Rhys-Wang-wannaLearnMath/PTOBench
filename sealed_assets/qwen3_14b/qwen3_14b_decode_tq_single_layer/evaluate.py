"""Sealed execution adapter for qwen3_14b.qwen3_14b_decode_tq_single_layer."""

import math
from functools import partial
from pathlib import Path

import torch
from golden import TensorSpec, run

import common
import workflow


def _identity(case, role):
    return f"{case['case_id'].replace('.', '_')}_{role}"


def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], _identity(case, "config"))


def _validate_policy(policy, outputs):
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}:
        raise ValueError("comparator requires exactly schema_version and outputs")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("unsupported comparator schema_version")
    rules = policy["outputs"]
    if not isinstance(rules, dict) or set(rules) != set(outputs):
        raise ValueError("comparator output coverage differs from correctness.outputs")
    for name, rule in rules.items():
        if not isinstance(rule, dict) or rule.get("method") != "ratio_allclose":
            raise ValueError(f"unsupported comparator method for {name}")
        if set(rule) != {"method", "rtol", "atol", "max_error_ratio"}:
            raise ValueError(f"invalid ratio_allclose fields for {name}")
        for field in ("rtol", "atol", "max_error_ratio"):
            value = rule[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field} for {name}")
        if rule["max_error_ratio"] > 1:
            raise ValueError(f"invalid max_error_ratio for {name}")
    return rules


def _ratio_callback(rule):
    policy_rtol = rule["rtol"]
    policy_atol = rule["atol"]
    policy_ratio = rule["max_error_ratio"]

    def ratio_allclose(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, "output shape mismatch"
        actual = actual.float()
        expected = expected.float()
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            return False, "nonfinite actual or expected"
        count = int((torch.abs(actual - expected) >
                     policy_atol + policy_rtol * torch.abs(expected)).sum().item())
        allowed = round(policy_ratio * actual.numel())
        return count <= allowed, f"{count}/{actual.numel()} mismatches; allowed={allowed}"

    return ratio_allclose


def make_compare(case_dir, case):
    policy = common.read_json(case_dir / case["comparator"])
    rules = _validate_policy(policy, case["correctness"]["outputs"])
    return {name: _ratio_callback(rule) for name, rule in rules.items()}


def build_specs(case_dir, case, fixture):
    config = _config(case_dir, case)
    outputs = set(case["correctness"]["outputs"])
    specs = []
    for name in config.ARGUMENT_ORDER:
        shape = config.TENSOR_SHAPES[name]
        dtype = getattr(torch, config.TENSOR_DTYPES[name])
        if name in outputs:
            init = 0
        else:
            path = fixture / "in" / f"{name}.pt"

            def init(path=path, shape=shape, dtype=dtype):
                value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
                    raise ValueError(f"Fixture ABI mismatch: {path}")
                return value.clone().contiguous()

        specs.append(TensorSpec(name, list(shape), dtype, init_value=init))
    return specs


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    outputs = oracle.compute({name: value for name, value in values.items() if name != "out"}, {})
    if set(outputs) != {"out"}:
        raise ValueError("Oracle output coverage mismatch")
    values["out"].copy_(outputs["out"])


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(case_dir, case, fixture),
        load_ref=lambda: common.load_ref(case_dir, sealed, _identity(case, "ref")),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, _identity(case, "oracle")),
        make_program=_program,
        make_compare=lambda: make_compare(case_dir, case),
        run=partial(run, reset_input_names=case["state"]["initial_state"]),
        oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        candidate_workspace=Path(candidate_workspace), phase=phase,
        build_specs=lambda fixture: build_specs(case_dir, case, fixture),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, Path(candidate_workspace), _identity(case, "candidate")),
        make_program=_program,
        make_compare=lambda: make_compare(case_dir, case),
        run=partial(run, reset_input_names=case["state"]["initial_state"]),
    )
