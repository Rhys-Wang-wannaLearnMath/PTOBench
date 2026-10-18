import math

import torch
from golden import TensorSpec, run

import common
import workflow


def _callbacks(policy, output_names):
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}:
        raise ValueError("Comparator must contain exactly schema_version and outputs")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("Unsupported comparator schema")
    rules = policy["outputs"]
    if not isinstance(rules, dict) or set(rules) != set(output_names):
        raise ValueError("Comparator output coverage differs from correctness.outputs")
    callbacks = {}
    for name, rule in rules.items():
        if not isinstance(rule, dict) or rule.get("method") != "allclose":
            raise ValueError(f"Unsupported comparator for {name}")
        if set(rule) != {"method", "rtol", "atol"}:
            raise ValueError(f"Invalid allclose fields for {name}")
        for field in ("rtol", "atol"):
            value = rule[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {field} for {name}")
        callbacks[name] = _allclose(rule["rtol"], rule["atol"])
    return callbacks


def _allclose(policy_rtol, policy_atol):
    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False, "Output shape or dtype mismatch"
        passed = torch.allclose(actual, expected, rtol=policy_rtol, atol=policy_atol)
        return passed, "" if passed else f"allclose failed: rtol={policy_rtol}, atol={policy_atol}"
    return compare


def make_compare(case_dir, case):
    return _callbacks(common.read_json(case_dir / case["comparator"]), case["correctness"]["outputs"])


def build_specs(case_dir, case, fixture):
    config = common.load_module(
        case_dir / case["config"], case["case_id"].replace(".", "_") + "_evaluate_config"
    )
    tensor_dtype = getattr(torch, config.TENSOR_DTYPE)
    index_dtype = getattr(torch, config.INDEX_DTYPE)
    definitions = [
        ("local_kv_tail", [config.CP_SIZE, config.EPOCHS * 2 * config.TAIL_ROWS, config.HEAD_DIM], tensor_dtype),
        ("reverse_index", [config.NUM_SEGMENTS], index_dtype),
        ("owner_rank_table", [config.NUM_SEGMENTS], index_dtype),
        ("final_win_seg_src", [config.TAIL_ROWS], index_dtype),
        ("final_win_row_src", [config.TAIL_ROWS], index_dtype),
        ("logical_tails_out", [config.CP_SIZE, config.EPOCHS * config.CP_TAIL_WINDOW_ROWS, config.HEAD_DIM], tensor_dtype),
        ("decode_raw_window_out", [config.CP_SIZE, config.TAIL_ROWS, config.HEAD_DIM], tensor_dtype),
    ]
    outputs = set(case["correctness"]["outputs"])
    specs = []
    for name, shape, dtype in definitions:
        value = None if name in outputs else torch.load(fixture / "in" / f"{name}.pt", weights_only=True)
        if value is not None and (list(value.shape) != shape or value.dtype != dtype):
            raise ValueError(f"Fixture ABI mismatch for {name}")
        specs.append(TensorSpec(name, shape, dtype, init_value=value))
    return specs


def make_program(module, case):
    return getattr(module, case["entry"])


def oracle_golden(oracle, values):
    expected = oracle.compute(values)
    for name, tensor in expected.items():
        values[name].copy_(tensor)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(case_dir, case, fixture),
        load_ref=lambda: common.load_ref(case_dir, sealed, identity + "_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + "_oracle"),
        make_program=make_program, make_compare=lambda: make_compare(case_dir, case),
        run=run, oracle_golden=oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(case_dir, case, fixture),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + "_candidate"),
        make_program=make_program, make_compare=lambda: make_compare(case_dir, case),
        run=run,
    )
