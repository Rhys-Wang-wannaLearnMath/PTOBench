"""Case-local ABI and comparator boundary for deepseek_v4.moe_expert_flash."""

import math

import torch
from golden import TensorSpec, run

import common
import workflow


def _config(case_dir, case):
    identity = case["case_id"].replace(".", "_")
    return common.load_module(case_dir / case["config"], f"config_{identity}_evaluation")


def _abi(config):
    experts = config.N_LOCAL_EXPERTS
    capacity = config.RECV_MAX
    width = config.D
    intermediate = config.MOE_INTER
    tokens = config.T
    return (
        ("recv_x", [experts, capacity, width], torch.int8),
        ("recv_scale_dq", [experts, capacity], torch.float32),
        ("recv_expert_count", [experts, 1], torch.int32),
        ("x_local", [tokens, width], torch.bfloat16),
        ("expert_w1", [experts, intermediate, width], torch.int8),
        ("expert_w1_scale", [experts, intermediate], torch.float32),
        ("expert_w3", [experts, intermediate, width], torch.int8),
        ("expert_w3_scale", [experts, intermediate], torch.float32),
        ("expert_w2", [experts, width, intermediate], torch.int8),
        ("expert_w2_scale", [experts, width], torch.float32),
        ("shared_w1", [intermediate, width], torch.int8),
        ("shared_w1_scale", [intermediate], torch.float32),
        ("shared_w3", [intermediate, width], torch.int8),
        ("shared_w3_scale", [intermediate], torch.float32),
        ("shared_w2", [width, intermediate], torch.int8),
        ("shared_w2_scale", [width], torch.float32),
        ("recv_y", [experts, capacity, width], torch.bfloat16),
        ("sh", [tokens, width], torch.bfloat16),
    )


def _build_specs(config, fixture):
    specs = []
    for name, shape, dtype in _abi(config):
        initial = None
        if name not in {"recv_y", "sh"}:
            def initial(input_name=name, input_shape=shape, input_dtype=dtype):
                value = torch.load(fixture / "in" / f"{input_name}.pt", weights_only=True)
                if list(value.shape) != input_shape or value.dtype != input_dtype:
                    raise ValueError(f"Fixture ABI mismatch for {input_name}")
                return value.clone()
        specs.append(TensorSpec(name, shape, dtype, init_value=initial))
    return specs


def _number(value, field, *, positive=False, fraction=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    if fraction and not 0 <= value <= 1:
        raise ValueError(f"{field} must lie in [0, 1]")
    return float(value)


def _ratio(actual, expected, policy_diff, policy_fraction, policy_cap):
    if actual.numel() == 0:
        return True, ""
    difference = (actual - expected).abs()
    denominator = torch.maximum(
        torch.maximum(actual.abs(), expected.abs()),
        torch.full_like(actual, (1.0 / (1 << 14)) / policy_diff),
    ) + 1e-9
    relative = torch.where(difference < policy_diff, difference, difference / denominator)
    bad = relative > policy_diff
    errors = int(bad.sum().item())
    budget = round(policy_fraction * actual.numel())
    worst = float(relative[bad].max().item()) if errors else 0.0
    passed = errors <= budget and worst < policy_cap
    detail = "" if passed else (
        f"ratio_reldiff: bad={errors}/{actual.numel()}, budget={budget}, "
        f"worst={worst}, cap={policy_cap}"
    )
    return passed, detail


def _callback(rule):
    policy_diff = _number(rule["diff_thd"], "diff_thd", positive=True)
    policy_fraction = _number(rule["pct_thd"], "pct_thd", fraction=True)
    policy_cap = (
        math.inf if rule["max_diff_hd"] is None
        else _number(rule["max_diff_hd"], "max_diff_hd", positive=True)
    )
    count_name = rule.get("count_input")

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, "Output shape mismatch"
        actual_float = actual.cpu().to(torch.float32)
        expected_float = expected.cpu().to(torch.float32)
        if not torch.isfinite(actual_float).all().item():
            return False, "Nonfinite actual output"
        if not torch.isfinite(expected_float).all().item():
            return False, "Nonfinite expected output"
        if count_name is not None:
            if actual.ndim != 3:
                return False, "recv_y must have rank 3"
            counts = kwargs.get("inputs", {}).get(count_name)
            if not isinstance(counts, torch.Tensor) or counts.dtype != torch.int32:
                return False, f"Missing or invalid INT32 input {count_name}"
            counts = counts.cpu().reshape(-1)
            experts, capacity, _ = actual.shape
            if counts.numel() != experts:
                return False, "Count input must have one value per expert"
            if ((counts < 0) | (counts > capacity)).any().item():
                return False, "Expert count exceeds receive capacity"
            active = torch.arange(capacity).reshape(1, -1) < counts.reshape(-1, 1)
            inactive = ~active
            if not torch.equal(actual.cpu()[inactive], expected.cpu()[inactive]):
                return False, "Inactive recv_y rows changed"
            actual_float = actual_float[active]
            expected_float = expected_float[active]
        return _ratio(
            actual_float, expected_float, policy_diff, policy_fraction, policy_cap,
        )

    compare.__name__ = rule["name"]
    return compare


def _make_compare(case_dir, case, config):
    policy = common.read_json(case_dir / case["comparator"])
    if type(policy) is not dict or set(policy) != {"schema_version", "outputs"}:
        raise ValueError("Comparator requires exactly schema_version and outputs")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("Unsupported comparator schema")
    outputs = case["correctness"]["outputs"]
    if (
        len(outputs) != 2 or set(outputs) != {"recv_y", "sh"}
        or type(policy["outputs"]) is not dict
        or set(policy["outputs"]) != set(outputs)
    ):
        raise ValueError("Comparator output coverage mismatch")
    declared_inputs = {
        name: dtype for name, _, dtype in _abi(config) if name not in outputs
    }
    callbacks = {}
    for output in outputs:
        rule = policy["outputs"][output]
        fields = {"method", "name", "diff_thd", "pct_thd", "max_diff_hd"}
        if output == "recv_y":
            fields.add("count_input")
        if type(rule) is not dict or set(rule) != fields or rule["method"] != "custom":
            raise ValueError(f"Invalid comparator method or fields for {output}")
        expected_name = "active_recv_ratio_reldiff" if output == "recv_y" else "ratio_reldiff"
        if rule["name"] != expected_name:
            raise ValueError(f"Unsupported custom comparator for {output}")
        if output == "recv_y" and (
            rule["count_input"] != "recv_expert_count"
            or declared_inputs.get(rule["count_input"]) != torch.int32
        ):
            raise ValueError("count_input must name the declared INT32 expert counts")
        callbacks[output] = _callback(rule)
    return callbacks


def _make_program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    inputs = {name: value for name, value in values.items() if name not in {"recv_y", "sh"}}
    outputs = oracle.compute(inputs, {})
    if set(outputs) != {"recv_y", "sh"}:
        raise ValueError("Oracle output coverage mismatch")
    for name, value in outputs.items():
        values[name].copy_(value)


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = sealed.name
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: _build_specs(config, fixture),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"ref_{identity}"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"oracle_{identity}"),
        make_program=_make_program,
        make_compare=lambda: _make_compare(case_dir, case, config),
        run=run, oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        candidate_workspace=candidate_workspace, phase=phase,
        build_specs=lambda fixture: _build_specs(config, fixture),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, f"candidate_{sealed.name}",
        ),
        make_program=_make_program,
        make_compare=lambda: _make_compare(case_dir, case, config),
        run=run,
    )
