OUTPUTS = ("x_next",)
"""Case-local evaluator for DeepSeek-V4 distributed EP8 MoE."""

import json, math

from pathlib import Path

import torch

from golden import ScalarSpec, TensorSpec, run

import common

workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", "workflow_deepseek_v4_prefill_moe_ep8_pro")

def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], case["case_id"].replace(".", "_") + "_config")

def _values(fixture):
    return {p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in (fixture / "in").glob("*.pt")}

def _build_specs(fixture, c):
    v = _values(fixture)
    tensors = [("x_hc", c.X_HC_SHAPE, torch.bfloat16), ("hc_ffn_fn", c.HC_FFN_FN_SHAPE, torch.float32), ("hc_ffn_scale", c.HC_FFN_SCALE_SHAPE, torch.float32), ("hc_ffn_base", c.HC_FFN_BASE_SHAPE, torch.float32), ("norm_w", c.NORM_W_SHAPE, torch.bfloat16), ("gate_w", c.GATE_W_SHAPE, torch.float32), ("gate_bias", c.GATE_BIAS_SHAPE, torch.float32), ("tid2eid", c.TID2EID_SHAPE, torch.int32), ("input_ids", c.INPUT_IDS_SHAPE, torch.int64), ("routed_w1", c.ROUTED_W1_SHAPE, torch.int8), ("routed_w1_scale", c.ROUTED_W1_SCALE_SHAPE, torch.float32), ("routed_w3", c.ROUTED_W3_SHAPE, torch.int8), ("routed_w3_scale", c.ROUTED_W3_SCALE_SHAPE, torch.float32), ("routed_w2", c.ROUTED_W2_SHAPE, torch.int8), ("routed_w2_scale", c.ROUTED_W2_SCALE_SHAPE, torch.float32), ("shared_w1", c.SHARED_W1_SHAPE, torch.int8), ("shared_w1_scale", c.SHARED_W1_SCALE_SHAPE, torch.float32), ("shared_w3", c.SHARED_W3_SHAPE, torch.int8), ("shared_w3_scale", c.SHARED_W3_SCALE_SHAPE, torch.float32), ("shared_w2", c.SHARED_W2_SHAPE, torch.int8), ("shared_w2_scale", c.SHARED_W2_SCALE_SHAPE, torch.float32)]
    specs = [TensorSpec(n, list(shape), dt, init_value=v[n]) for n, shape, dt in tensors]
    specs += [TensorSpec("x_next", list(c.X_NEXT_SHAPE), torch.bfloat16), ScalarSpec("layer_id", torch.int32, v["layer_id"]), ScalarSpec("num_tokens", torch.int32, v["num_tokens"])]
    for s in specs:
        if s.name in {n for n, _, _ in tensors if n not in {"x_hc", "input_ids"}}:
            s.resident = "stacked"
    return specs

def _program(module, case): return getattr(module, case["entry"])

def _oracle(oracle, values):
    scalars = {n: values[n] for n in ("layer_id", "num_tokens")}
    inputs = {n: values[n] for n in values if n not in {"x_next", "layer_id", "num_tokens"}}
    for n, val in oracle.compute(inputs, scalars).items(): values[n][:] = val

def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
    return run(fn=fn, specs=specs, config=config, golden_fn=golden_fn, golden_data=str(golden_data) if golden_data else None, compare_fn=compare_fn, rtol=1e-3, atol=1e-3, save_data=False, capture_outputs=capture_outputs)

def prepare_ref(case_dir, sealed, case, platform=None):
    c = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform, build_specs=lambda f: _build_specs(f, c), load_ref=lambda: common.load_ref(case_dir, sealed, identity + "_ref"), load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + "_oracle"), make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)

def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    c = _config(case_dir, case)
    identity = case["case_id"].replace(".", "_")
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=lambda f: _build_specs(f, c), load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + "_candidate"), make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)

def _number(value, name, *, positive=False, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('invalid ' + name)
    if value < 0 or (positive and value == 0) or (upper is not None and value > upper):
        raise ValueError('invalid ' + name)
    return float(value)

def _compare(case_dir, case):
    policy = common.read_json(case_dir / case['comparator'])
    if (not isinstance(policy, dict)
            or set(policy) != {'schema_version', 'outputs'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1
            or not isinstance(policy['outputs'], dict)
            or set(policy['outputs']) != set(case['correctness']['outputs'])
            or set(policy['outputs']) != set(OUTPUTS)):
        raise ValueError('invalid comparator schema or outputs')
    rule = policy['outputs']['x_next']
    fields = {'method', 'name', 'diff_thd', 'pct_thd', 'max_diff_hd', 'floor_numerator',
              'denominator_epsilon', 'nonfinite_policy'}
    if (not isinstance(rule, dict) or set(rule) != fields or rule['method'] != 'custom' or rule['name'] != 'ratio_reldiff'
            or rule['nonfinite_policy'] != 'reject_either'):
        raise ValueError('invalid MoE comparator rule')
    diff = _number(rule['diff_thd'], 'diff_thd', positive=True)
    pct = _number(rule['pct_thd'], 'pct_thd', upper=1)
    floor = _number(rule['floor_numerator'], 'floor_numerator', positive=True)
    eps = _number(rule['denominator_epsilon'], 'denominator_epsilon')
    cap = None if rule['max_diff_hd'] is None else _number(rule['max_diff_hd'], 'max_diff_hd', positive=True)

    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, 'shape mismatch'
        a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
        if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()):
            return False, 'nonfinite value'
        absolute = (a - e).abs()
        denominator = torch.maximum(torch.maximum(a.abs(), e.abs()), torch.full_like(a, floor / diff)) + eps
        relative = torch.where(absolute < diff, absolute, absolute / denominator)
        bad = relative > diff
        count = int(bad.sum())
        allowed = round(pct * a.numel())
        worst = float(relative[bad].max()) if count else 0.
        passed = count <= allowed and (cap is None or worst < cap)
        return passed, f'ratio_reldiff: mismatches={count}/{a.numel()}, allowed={allowed}, worst={worst}'
    return {'x_next': compare}
