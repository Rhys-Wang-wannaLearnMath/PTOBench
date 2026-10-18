"""Case-local decode indexer ABI and comparator execution."""

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common
import workflow


def _config(case_dir, case):
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], f"config_{identity}")
    if config.CASE_ID != case["case_id"]:
        raise ValueError("public config identity disagrees with the Case")
    return config


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate comparator key: {key}")
        result[key] = value
    return result


def _fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"comparator fields must be exactly {sorted(expected)}")


def _number(value, label, minimum=0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric policy")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is outside its allowed range")


def _ratio(actual, expected, policy):
    if actual.shape != expected.shape:
        return False, "shape mismatch"
    actual_f = actual.cpu().float()
    expected_f = expected.cpu().float()
    if not torch.isfinite(actual_f).all() or not torch.isfinite(expected_f).all():
        return False, "non-finite comparison value"
    difference = (actual_f - expected_f).abs()
    tolerance = policy["atol"] + policy["rtol"] * expected_f.abs()
    count = int((difference > tolerance).sum())
    allowed = round(policy["max_error_ratio"] * expected_f.numel())
    return count <= allowed, f"mismatches={count}/{expected_f.numel()}, allowed={allowed}"


def _ratio_callback(policy):
    def compare(actual, expected, **kwargs):
        return _ratio(actual, expected, policy)
    compare.__name__ = "ratio_allclose"
    return compare


def _score_callback(policy):
    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, "score shape mismatch"
        expected_f = expected.cpu().float()
        valid = expected_f != policy["tail_sentinel"]
        return _ratio(actual.cpu().float()[valid], expected_f[valid], policy)
    compare.__name__ = policy["name"]
    return compare


def _topk_callback(policy):
    def compare(actual, expected, *, actual_outputs, **kwargs):
        if actual.shape != expected.shape:
            return False, "index shape mismatch"
        score = actual_outputs.get(policy["paired_output"])
        if score is None or score.shape != actual.shape:
            return False, "paired score shape mismatch"
        if actual.ndim == 0 or actual.shape[-1] < policy["valid_prefix"]:
            return False, "index prefix exceeds output width"
        actual_idx = actual.cpu()[..., :policy["valid_prefix"]]
        expected_idx = expected.cpu()[..., :policy["valid_prefix"]]
        original = (actual_idx.long() - policy["offset"]).clamp(0, score.shape[-1] - 1)
        paired = torch.gather(score.cpu().float(), -1, original)
        if not torch.isfinite(paired).all():
            return False, "non-finite paired score"
        mismatch = actual_idx != expected_idx
        adjacent = paired[..., :-1] >= paired[..., 1:] if policy["descending"] else paired[..., :-1] <= paired[..., 1:]
        left_ok = torch.ones_like(mismatch)
        right_ok = torch.ones_like(mismatch)
        left_ok[..., 1:] = adjacent
        right_ok[..., :-1] = adjacent
        failures = mismatch & ~(left_ok & right_ok)
        return not bool(failures.any()), f"non-monotonic mismatched indices={int(failures.sum())}"
    compare.__name__ = policy["name"]
    return compare


def make_compare(case_dir, case):
    config = _config(case_dir, case)
    policy = json.loads((case_dir / case["comparator"]).read_text(), object_pairs_hook=_unique_object)
    _fields(policy, {"schema_version", "outputs"})
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("unsupported comparator schema")
    declared = {"score", "idx_kv_cache", "idx_kv_scale", "topk_idxs"}
    if set(case["correctness"]["outputs"]) != declared:
        raise ValueError("Case correctness output coverage disagrees with the indexer ABI")
    _fields(policy["outputs"], declared)
    callbacks = {}
    for name, rule in policy["outputs"].items():
        ratio_fields = {"method", "rtol", "atol", "max_error_ratio"}
        if name == "score":
            _fields(rule, ratio_fields | {"name", "tail_sentinel"})
            if rule["method"] != "custom" or rule["name"] != "score_valid_region_ratio_allclose":
                raise ValueError("unsupported score comparator")
            _number(rule["tail_sentinel"], "tail_sentinel", -float("inf"))
            if rule["tail_sentinel"] != config.FP32_NEG_INF:
                raise ValueError("score sentinel disagrees with the fixed contract")
            callback = _score_callback(rule)
        elif name == "topk_idxs":
            _fields(rule, {"method", "name", "paired_output", "valid_prefix", "offset", "descending", "tie_policy"})
            if rule["method"] != "custom" or rule["name"] != "topk_pair_compare":
                raise ValueError("unsupported index comparator")
            if rule["paired_output"] != "score" or rule["tie_policy"] != "actual_score_monotonic":
                raise ValueError("unsupported paired index policy")
            if type(rule["valid_prefix"]) is not int or rule["valid_prefix"] != config.IDX_TOPK:
                raise ValueError("index prefix disagrees with the fixed contract")
            if type(rule["offset"]) is not int or rule["offset"] != config.OFFSET:
                raise ValueError("index offset disagrees with the fixed contract")
            if type(rule["descending"]) is not bool or not rule["descending"]:
                raise ValueError("index order disagrees with the fixed contract")
            callbacks[name] = _topk_callback(rule)
            continue
        else:
            _fields(rule, ratio_fields)
            if rule["method"] != "ratio_allclose":
                raise ValueError("unsupported cache comparator")
            callback = _ratio_callback(rule)
        _number(rule["rtol"], "rtol")
        _number(rule["atol"], "atol")
        _number(rule["max_error_ratio"], "max_error_ratio", 0, 1)
        callbacks[name] = callback
    return callbacks


def build_specs(fixture, config):
    bindings = [
        ("x", [config.B, config.S, config.D], torch.bfloat16),
        ("qr", [config.T, config.Q_LORA], torch.int8),
        ("qr_scale", [config.T, 1], torch.float32),
        ("wq_b", [config.Q_LORA, config.IDX_N_HEADS * config.IDX_HEAD_DIM], torch.int8),
        ("wq_b_scale", [config.IDX_N_HEADS * config.IDX_HEAD_DIM], torch.float32),
        ("weights_proj", [config.D, config.IDX_N_HEADS], torch.bfloat16),
        ("cos", [config.B, config.HALF_ROPE], torch.float32),
        ("sin", [config.B, config.HALF_ROPE], torch.float32),
        ("hadamard", [config.IDX_HEAD_DIM, config.IDX_HEAD_DIM], torch.bfloat16),
        ("inner_kv", [config.B, config.S, config.INNER_HEAD_DIM], torch.float32),
        ("inner_compress_state", [config.INNER_STATE_BLOCK_NUM, config.INNER_STATE_BLOCK_SIZE, config.INNER_STATE_DIM], torch.float32),
        ("inner_compress_state_block_table", [config.B, config.INNER_STATE_MAX_BLOCKS], torch.int32),
        ("inner_wkv", [config.INNER_OUT_DIM, config.D], torch.bfloat16),
        ("inner_wgate", [config.INNER_OUT_DIM, config.D], torch.bfloat16),
        ("inner_ape", [config.COMPRESS_RATIO, config.INNER_OUT_DIM], torch.float32),
        ("inner_norm_w", [config.INNER_HEAD_DIM], torch.bfloat16),
        ("idx_kv_cache", [config.IDX_CACHE_BLOCK_NUM, config.BLOCK_SIZE, 1, config.IDX_HEAD_DIM], torch.int8),
        ("idx_kv_scale", [config.IDX_CACHE_BLOCK_NUM, config.BLOCK_SIZE, 1, 1], torch.float32),
        ("idx_block_table", [config.B, config.IDX_CACHE_MAX_BLOCKS], torch.int32),
        ("score", [config.B, config.S, config.SCORE_LEN], torch.float32),
        ("topk_idxs", [config.B, config.S, config.SCORE_LEN], torch.int32),
        ("position_ids", [config.B, config.S], torch.int32),
        ("idx_slot_mapping", [config.B, config.S], torch.int64),
        ("inner_state_slot_mapping", [config.B, config.S], torch.int64),
        ("kv_seq_lens", [config.B], torch.int32),
    ]
    specs = []
    for name, shape, dtype in bindings:
        value = None
        if name not in {"score", "topk_idxs"}:
            value = torch.load(fixture / "in" / f"{name}.pt", map_location="cpu", weights_only=True)
            if list(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"Fixture ABI mismatch for {name}")
        specs.append(TensorSpec(name, shape, dtype, init_value=value))
    offset = torch.load(fixture / "in" / "offset.pt", map_location="cpu", weights_only=True)
    if offset.ndim != 0 or int(offset) != config.OFFSET:
        raise ValueError("Fixture offset disagrees with the fixed contract")
    specs.append(ScalarSpec("offset", torch.int32, int(offset)))
    return specs


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle_golden(oracle, values):
    outputs = oracle.compute(values)
    if set(outputs) != {"score", "idx_kv_cache", "idx_kv_scale", "topk_idxs"}:
        raise ValueError("Oracle correctness output coverage mismatch")
    for name, value in outputs.items():
        values[name].copy_(value)


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"ref_{identity}"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"oracle_{identity}"),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case),
        run=run, oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        candidate_workspace=candidate_workspace, phase=phase,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"candidate_{identity}"),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run,
    )
