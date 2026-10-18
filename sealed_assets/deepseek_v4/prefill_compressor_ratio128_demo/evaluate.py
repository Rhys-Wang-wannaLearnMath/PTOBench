"""Case-local evaluator for DeepSeek-V4 ratio-128 prefill compression."""

from __future__ import annotations

import json
import math

import torch
from golden import ScalarSpec, TensorSpec, run

import common


def _identity(case, role):
    return f"{case['case_id'].replace('.', '_')}_{role}"


def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], _identity(case, "public_config"))


def _values(fixture):
    return {path.stem: torch.load(path, map_location="cpu", weights_only=True) for path in (fixture / "in").glob("*.pt")}


def _build_specs(fixture, config):
    values = _values(fixture)
    required = {"x", "kv_state", "score_state", "compress_state_block_table", "wkv", "wgate", "ape", "norm_w", "freqs_cos", "freqs_sin", "cmp_kv", "position_ids", "num_tokens", "cmp_slot_mapping", "state_slot_mapping"}
    if set(values) != required:
        raise ValueError(f"fixture inputs do not match fixed ABI: {sorted(values)} != {sorted(required)}")
    if int(values["num_tokens"]) != config.NUM_TOKENS:
        raise ValueError("fixture num_tokens does not match fixed public binding")
    return [
        TensorSpec("x", [config.T, config.D], torch.bfloat16, init_value=values["x"]),
        TensorSpec("kv_state", [config.HCA_STATE_BLOCK_NUM, config.HCA_STATE_BLOCK_SIZE, config.OUT_DIM], torch.float32, init_value=values["kv_state"]),
        TensorSpec("score_state", [config.HCA_STATE_BLOCK_NUM, config.HCA_STATE_BLOCK_SIZE, config.OUT_DIM], torch.float32, init_value=values["score_state"]),
        TensorSpec("compress_state_block_table", [config.HCA_STATE_MAX_BLOCKS], torch.int32, init_value=values["compress_state_block_table"]),
        TensorSpec("wkv", [config.OUT_DIM, config.D], torch.bfloat16, init_value=values["wkv"]),
        TensorSpec("wgate", [config.OUT_DIM, config.D], torch.bfloat16, init_value=values["wgate"]),
        TensorSpec("ape", [config.COMPRESS_RATIO, config.OUT_DIM], torch.float32, init_value=values["ape"]),
        TensorSpec("norm_w", [config.HEAD_DIM], torch.bfloat16, init_value=values["norm_w"]),
        TensorSpec("freqs_cos", [config.MAX_SEQ_LEN, config.ROPE_HEAD_DIM], torch.bfloat16, init_value=values["freqs_cos"]),
        TensorSpec("freqs_sin", [config.MAX_SEQ_LEN, config.ROPE_HEAD_DIM], torch.bfloat16, init_value=values["freqs_sin"]),
        TensorSpec("cmp_kv", [config.HCA_CMP_BLOCK_NUM, config.BLOCK_SIZE, 1, config.HEAD_DIM], torch.bfloat16, init_value=values["cmp_kv"]),
        TensorSpec("position_ids", [config.T], torch.int32, init_value=values["position_ids"]),
        ScalarSpec("num_tokens", torch.int32, int(values["num_tokens"])),
        TensorSpec("cmp_slot_mapping", [config.T], torch.int64, init_value=values["cmp_slot_mapping"]),
        TensorSpec("state_slot_mapping", [config.T], torch.int64, init_value=values["state_slot_mapping"]),
    ]


def _program(module, case):
    return getattr(module, case["entry"])


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return value


def _ratio(name, settings):
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if not isinstance(settings, dict) or set(settings) != required or settings.get("method") != "ratio_allclose":
        raise ValueError(f"ratio_allclose comparator {name!r} has invalid fields")
    policy_rtol = _number(settings["rtol"], f"{name}.rtol")
    policy_atol = _number(settings["atol"], f"{name}.atol")
    policy_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio")
    if policy_ratio > 1:
        raise ValueError(f"comparator {name}.max_error_ratio must be at most 1")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, f"ratio_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
            return False, "ratio_allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > policy_atol + policy_rtol * expected_f.abs()
        count = int(bad.sum().item())
        allowed = round(policy_ratio * actual_f.numel())
        return count <= allowed, "" if count <= allowed else f"ratio_allclose: mismatches={count}/{actual_f.numel()}, allowed={allowed}, rtol={policy_rtol}, atol={policy_atol}"

    return compare


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    expected = list(case["correctness"]["outputs"])
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError("comparator outputs must exactly match correctness outputs")
    return {name: _ratio(name, outputs[name]) for name in expected}


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
    return run(fn=fn, specs=specs, config=config, compare_fn=compare_fn, golden_fn=golden_fn, golden_data=str(golden_data) if golden_data else None, rtol=0.0, atol=0.0, save_data=False, capture_outputs=capture_outputs)


def _oracle(oracle, values):
    for name, value in oracle.compute({name: value for name, value in values.items() if isinstance(value, torch.Tensor)}, {"num_tokens": values["num_tokens"]}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", _identity(case, "workflow"))
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform, build_specs=lambda fixture: _build_specs(fixture, config), load_ref=lambda: common.load_ref(case_dir, sealed, _identity(case, "ref")), load_oracle=lambda: common.load_oracle(case_dir, sealed, _identity(case, "oracle")), make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _config(case_dir, case)
    workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", _identity(case, "workflow"))
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=lambda fixture: _build_specs(fixture, config), load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, _identity(case, "candidate")), make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
