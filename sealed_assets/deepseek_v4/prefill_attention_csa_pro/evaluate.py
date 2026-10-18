"""Case-local evaluator for DeepSeek V4 packed prefill CSA attention."""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from golden import ScalarSpec, TensorSpec, run
import common

def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], case["case_id"].replace(".", "_")+"_config")
def _values(fixture):
    return {p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in (fixture/"in").glob("*.pt")}
# Keep the TensorSpec order identical to ``prefill_attention_csa_test``.  The
# runner specializes @pl.jit entries positionally; fixture directory order is
# intentionally unspecified and must not determine the ABI.
_PARAM_ORDER = (
    "x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "cmp_wkv", "cmp_wgate", "cmp_ape",
    "cmp_norm_w", "cmp_kv_state", "cmp_score_state", "compress_state_block_table",
    "hadamard_idx", "idx_wq_b", "idx_wq_b_scale", "idx_weights_proj",
    "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w", "inner_kv_state",
    "inner_score_state", "inner_compress_state_block_table", "kv_cache",
    "ori_block_table", "ori_slot_mapping", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table", "position_ids",
    "cmp_slot_mapping", "idx_slot_mapping", "state_slot_mapping",
    "inner_state_slot_mapping", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "x_out", "num_tokens",
)

def build_tensor_specs(fixture, config):
    values = _values(fixture)
    expected = set(_PARAM_ORDER) - {"x_out"}
    if set(values) != expected:
        raise ValueError(f"fixture inputs do not match CSA ABI: {sorted(set(values) ^ expected)}")
    specs = []
    for name in _PARAM_ORDER:
        if name == "x_out":
            specs.append(TensorSpec(name, [config.T, config.HC_MULT, config.D], torch.bfloat16))
        elif name == "num_tokens":
            specs.append(ScalarSpec(name, torch.int32, values[name]))
        else:
            value = values[name]
            specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    return specs
def _num(v, field, ratio=False):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)) or float(v)<0 or (ratio and float(v)>1): raise ValueError(f"invalid comparator {field}")
    return float(v)
def _allclose(rtol, atol):
    def cb(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, "shape mismatch"
        a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
        if not torch.isfinite(a).all() or not torch.isfinite(e).all():
            return False, "nonfinite values"
        passed = bool(torch.allclose(a, e, rtol=rtol, atol=atol))
        return passed, "" if passed else f"allclose failed: rtol={rtol}, atol={atol}"
    return cb

def _ratio_allclose(policy_rtol, policy_atol, error_ratio):
    def cb(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, "shape mismatch"
        a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
        if not torch.isfinite(a).all() or not torch.isfinite(e).all():
            return False, "nonfinite values"
        bad = int(((a - e).abs() > policy_atol + policy_rtol * e.abs()).sum())
        budget = round(error_ratio * a.numel())
        passed = bad <= budget
        return passed, "" if passed else f"ratio_allclose: {bad} errors exceed budget {budget}"
    return cb


def _valid_reldiff(rows, axis, diff, pct, maxd):
    # Selected official active-prefix rule, using the current applicable
    # hybrid absolute/relative mechanics. Policy is captured lexically so
    # runner fallback rtol/atol cannot replace any JSON parameter.
    def cb(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, "shape mismatch"
        if axis >= actual.ndim or rows > actual.shape[axis]:
            return False, "valid prefix outside tensor"
        a = actual.narrow(axis, 0, rows).detach().cpu().float()
        e = expected.narrow(axis, 0, rows).detach().cpu().float()
        if a.numel() == 0:
            return True, ""
        if not torch.isfinite(a).all() or not torch.isfinite(e).all():
            return False, "nonfinite values"
        delta = (a - e).abs()
        floor = torch.full_like(a, (1.0 / (1 << 14)) / diff)
        denom = torch.maximum(torch.maximum(a.abs(), e.abs()), floor) + 1e-9
        relative = torch.where(delta < diff, delta, delta / denom)
        mask = relative > diff
        bad = int(mask.sum())
        worst = float(relative[mask].max()) if bad else 0.0
        budget = round(pct * a.numel())
        passed = bad <= budget and worst < maxd
        return passed, "" if passed else f"valid_ratio_reldiff: errors={bad}, budget={budget}, worst={worst}, cap={maxd}"
    return cb

def _compare(case_dir, case):
    p = json.loads((case_dir / case["comparator"]).read_text())
    names = case["correctness"]["outputs"]
    if (not isinstance(p, dict) or set(p) != {"schema_version", "outputs"}
            or type(p.get("schema_version")) is not int or p["schema_version"] != 1
            or not isinstance(p["outputs"], dict) or set(p["outputs"]) != set(names)):
        raise ValueError("comparator schema/coverage mismatch")
    out = {}
    for name, s in p["outputs"].items():
        if not isinstance(s, dict):
            raise ValueError("comparator rule must be object")
        method = s.get("method")
        if method == "ratio_allclose":
            if set(s) != {"method", "rtol", "atol", "max_error_ratio"}:
                raise ValueError("ratio_allclose fields")
            rtol, atol, ratio = (_num(s[k], f"{name}.{k}", k == "max_error_ratio")
                                 for k in ("rtol", "atol", "max_error_ratio"))
            out[name] = _ratio_allclose(rtol, atol, ratio)
        elif method == "allclose":
            if set(s) != {"method", "rtol", "atol"}:
                raise ValueError("allclose fields")
            out[name] = _allclose(_num(s["rtol"], f"{name}.rtol"),
                                 _num(s["atol"], f"{name}.atol"))
        elif method == "custom" and s.get("name") == "valid_ratio_reldiff":
            req = {"method", "name", "valid_rows", "valid_axis", "diff_thd",
                   "pct_thd", "max_diff_hd", "tail_policy"}
            if set(s) != req or s["tail_policy"] != "reject_undeclared":
                raise ValueError("custom fields")
            rows, axis = s["valid_rows"], s["valid_axis"]
            if type(rows) is not int or rows < 0 or type(axis) is not int or axis < 0:
                raise ValueError("invalid valid prefix")
            diff = _num(s["diff_thd"], "diff_thd")
            pct = _num(s["pct_thd"], "pct_thd", ratio=True)
            maxd = _num(s["max_diff_hd"], "max_diff_hd")
            if diff == 0 or maxd == 0:
                raise ValueError("diff_thd and max_diff_hd must be positive")
            out[name] = _valid_reldiff(rows, axis, diff, pct, maxd)
        else:
            raise ValueError(f"unsupported comparator method {method}")
    return out
def _program(module, case): return getattr(module, case["entry"])
def _oracle(oracle, values):
    inputs = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
    scalars = {k: v for k, v in values.items() if not isinstance(v, torch.Tensor)}
    outputs = oracle.compute(inputs, scalars)
    for k, v in outputs.items():
        values[k][:] = v
def _run(**kwargs): return run(rtol=0.0, atol=0.0, save_data=False, **kwargs)
def prepare_ref(case_dir,sealed,case,platform=None):
    identity=case["case_id"].replace(".","_"); wf=common.load_module(common.BENCHMARK_ROOT/"evaluator/workflow.py", "workflow_"+identity); cfg=_config(case_dir,case)
    return wf.prepare_ref(case_dir=case_dir,sealed=sealed,case=case,platform=platform,build_specs=lambda f:build_tensor_specs(f,cfg),load_ref=lambda:common.load_ref(case_dir,sealed,identity+"_ref"),load_oracle=lambda:common.load_oracle(case_dir,sealed,identity+"_oracle"),make_program=_program,make_compare=lambda:_compare(case_dir,case),run=_run,oracle_golden=_oracle)
def evaluate_candidate(case_dir,sealed,case,candidate_workspace,platform=None,phase="all"):
    identity=case["case_id"].replace(".","_"); wf=common.load_module(common.BENCHMARK_ROOT/"evaluator/workflow.py", "workflow_"+identity); cfg=_config(case_dir,case)
    return wf.evaluate_candidate(case_dir=case_dir,sealed=sealed,case=case,candidate_workspace=candidate_workspace,platform=platform,phase=phase,build_specs=lambda f:build_tensor_specs(f,cfg),load_candidate=lambda:common.load_candidate(case_dir,case,candidate_workspace,identity+"_candidate"),make_program=_program,make_compare=lambda:_compare(case_dir,case),run=_run)
