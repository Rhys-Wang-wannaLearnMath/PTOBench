from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run
import common

workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator/workflow.py", "workflow_sparse_attn_demo")

NAMES = ["q", "ori_kv", "ori_block_table", "cmp_kv", "cmp_block_table", "cmp_sparse_indices", "attn_sink", "seqused_kv", "freqs_cos", "freqs_sin", "even_select_local", "odd_select_local", "wo_a", "wo_b", "wo_b_scale", "attn_out"]

def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], f"{case['case_id'].replace('.', '_')}_config")

def _values(fixture):
    return {p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in (fixture / "in").glob("*.pt")}

def build_tensor_specs(fixture: Path, config):
    values = _values(fixture)
    specs = []
    for name in NAMES:
        if name == "attn_out":
            specs.append(TensorSpec(name, [config.T, config.D], torch.bfloat16))
        else:
            value = values[name]
            specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    return specs

def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"invalid comparator {field}")
    return float(value)

def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or policy["schema_version"] != 1:
        raise ValueError("invalid comparator schema")
    outputs = policy["outputs"]
    expected = list(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError("comparator output coverage mismatch")
    result = {}
    for name in expected:
        settings = outputs[name]
        if not isinstance(settings, dict) or set(settings) != {"method", "rtol", "atol", "max_error_ratio"} or settings.get("method") != "ratio_allclose":
            raise ValueError(f"{name} must declare ratio_allclose exactly")
        policy_rtol = _number(settings["rtol"], f"{name}.rtol")
        policy_atol = _number(settings["atol"], f"{name}.atol")
        policy_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio")
        if policy_ratio > 1:
            raise ValueError("max_error_ratio out of range")
        def compare(actual, expected_value, _rtol=policy_rtol, _atol=policy_atol, _ratio=policy_ratio, **_kwargs):
            if actual.shape != expected_value.shape:
                return False, "shape mismatch"
            actual_f, expected_f = actual.cpu().float(), expected_value.cpu().float()
            if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
                return False, "non-finite value"
            bad = (actual_f - expected_f).abs() > (_atol + _rtol * expected_f.abs())
            count = int(bad.sum().item())
            allowed = round(_ratio * actual_f.numel())
            return count <= allowed, f"mismatches={count}/{actual_f.numel()}, allowed={allowed}"
        result[name] = compare
    return result

def _run(**kwargs):
    return run(rtol=1e-3, atol=1e-3, save_data=False, **kwargs)

def _program(module, case):
    return getattr(module, case["entry"])

def _oracle(oracle, values):
    tensors = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
    for name, value in oracle.compute(tensors, {}).items():
        values[name][:] = value

def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda f: build_tensor_specs(f, _config(case_dir, case)),
        load_ref=lambda: common.load_ref(case_dir, sealed, "sparse_attn_demo_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, "sparse_attn_demo_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)

def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase, build_specs=lambda f: build_tensor_specs(f, _config(case_dir, case)),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, "sparse_attn_demo_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
