from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from golden import TensorSpec, run
import common

workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator/workflow.py", "workflow_hca_flash")
NAMES = ["x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv", "freqs_cos", "freqs_sin", "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w", "compress_state", "compress_state_block_table", "kv_cache", "ori_block_table", "cmp_kv", "cmp_block_table", "attn_sink", "seqused_kv", "wo_a", "wo_b", "wo_b_scale", "x_out", "start_pos"]

def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], "hca_flash_config")

def _values(fixture):
    return {p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in (fixture / "in").glob("*.pt")}

def build_tensor_specs(fixture, config):
    values = _values(fixture)
    specs = []
    for name in NAMES:
        value = values.get(name)
        if name == "x_out":
            specs.append(TensorSpec(name, list(config.X_OUT_SHAPE), torch.bfloat16))
        else:
            specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    return specs

def _number(v, field):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0:
        raise ValueError(f"invalid comparator {field}")
    return float(v)

def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text())
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or policy["schema_version"] != 1:
        raise ValueError("invalid comparator schema")
    outputs = policy["outputs"]
    expected = list(case["correctness"]["outputs"])
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError("comparator output coverage mismatch")
    result = {}
    for name in expected:
        s = outputs[name]
        if not isinstance(s, dict) or set(s) != {"method", "rtol", "atol", "max_error_ratio"} or s.get("method") != "ratio_allclose":
            raise ValueError("x_out must declare ratio_allclose exactly")
        rtol, atol, ratio = _number(s["rtol"], "rtol"), _number(s["atol"], "atol"), _number(s["max_error_ratio"], "max_error_ratio")
        if ratio > 1:
            raise ValueError("max_error_ratio out of range")
        def compare(actual, expected_value, policy_rtol=rtol, policy_atol=atol, policy_ratio=ratio, **_):
            if actual.shape != expected_value.shape:
                return False, "shape mismatch"
            a, e = actual.cpu().float(), expected_value.cpu().float()
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(e).all()):
                return False, "non-finite value"
            bad = (a - e).abs() > (policy_atol + policy_rtol * e.abs())
            count = int(bad.sum().item())
            allowed = round(policy_ratio * a.numel())
            return count <= allowed, f"mismatches={count}/{a.numel()}, allowed={allowed}"
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
    def load_oracle_isolated():
        for name in ("hc_pre", "hc_post", "decode_qkv_proj_rope", "decode_compressor_ratio128", "decode_sparse_attn"):
            sys.modules.pop(name, None)
        return common.load_oracle(case_dir, sealed, "hca_flash_oracle")
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda f: build_tensor_specs(f, _config(case_dir, case)),
        load_ref=lambda: common.load_ref(case_dir, sealed, "hca_flash_ref"),
        load_oracle=load_oracle_isolated,
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)

def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase, build_specs=lambda f: build_tensor_specs(f, _config(case_dir, case)),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, "hca_flash_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
