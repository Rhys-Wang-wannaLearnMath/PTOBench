"""Case-local evaluator for DeepSeek-V4 routed experts (FLASH)."""
import json, math
from pathlib import Path
import torch
from golden import TensorSpec, run
import common

workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator/workflow.py", "workflow_expert_routed_flash")

def _values(fixture):
    return {p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in (fixture / "in").glob("*.pt")}

def build_tensor_specs(fixture):
    v = _values(fixture)
    c = common.load_module(common.BENCHMARK_ROOT / "cases/deepseek_v4/expert_routed_flash/public/config.py", "expert_routed_flash_specs_config")
    dtypes = {"recv_x": torch.int8, "recv_scale_dq": torch.float32, "recv_weights": torch.float32, "recv_expert_count": torch.int32,
              "routed_w1": torch.int8, "routed_w1_scale": torch.float32, "routed_w3": torch.int8, "routed_w3_scale": torch.float32,
              "routed_w2": torch.int8, "routed_w2_scale": torch.float32}
    return [*(TensorSpec(n, list(v[n].shape), dtypes[n], init_value=v[n]) for n in dtypes), TensorSpec("recv_y", [c.N_LOCAL_EXPERTS, c.RECV_MAX, c.D], torch.bfloat16)]

def _program(module, case): return getattr(module, case["entry"])
def _num(x, field, allow_null=False):
    if allow_null and x is None: return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) < 0: raise ValueError(f"invalid comparator {field}")
    return float(x)
def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text()); names = case["correctness"]["outputs"]
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or policy["schema_version"] != 1: raise ValueError("invalid comparator top-level")
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(names): raise ValueError("comparator output coverage mismatch")
    callbacks = {}
    for name in names:
        s = outputs[name]; required = {"method", "name", "diff_threshold", "max_error_ratio", "max_relative_diff"}
        if not isinstance(s, dict) or set(s) != required or s.get("method") != "custom" or s.get("name") != "ratio_reldiff": raise ValueError("invalid ratio_reldiff declaration")
        diff, ratio, cap = _num(s["diff_threshold"], "diff_threshold"), _num(s["max_error_ratio"], "max_error_ratio"), _num(s["max_relative_diff"], "max_relative_diff", True)
        if diff == 0 or ratio > 1 or (cap is not None and cap < diff): raise ValueError("invalid ratio_reldiff range")
        def compare(actual, expected, diff=diff, ratio=ratio, cap=cap, **_):
            if actual.shape != expected.shape: return False, "ratio_reldiff: shape mismatch"
            a, e = actual.cpu().float(), expected.cpu().float()
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(e).all()): return False, "ratio_reldiff: non-finite value"
            delta = (a-e).abs(); floor = (1.0 / (1 << 14)) / diff
            denom = torch.maximum(torch.maximum(a.abs(), e.abs()), torch.full_like(a, floor)) + 1e-9
            rdiff = torch.where(delta < diff, delta, delta / denom); bad = rdiff > diff
            count, limit = int(bad.sum().item()), round(ratio*a.numel())
            worst = float(rdiff[bad].max().item()) if count else 0.0
            passed = count <= limit and (cap is None or worst < cap)
            return passed, "" if passed else f"ratio_reldiff: mismatches={count}/{a.numel()}, allowed={limit}"
        callbacks[name] = compare
    return callbacks
def _run(**kwargs): return run(save_data=False, **kwargs)
def _oracle(oracle, values):
    for n, x in oracle.compute({k:v for k,v in values.items() if isinstance(v, torch.Tensor)}, {}).items(): values[n][:] = x
def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform, build_specs=build_tensor_specs,
        load_ref=lambda: common.load_ref(case_dir, sealed, "expert_routed_flash_ref"), load_oracle=lambda: common.load_oracle(case_dir, sealed, "expert_routed_flash_oracle"), make_program=_program,
        make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)
def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=build_tensor_specs,
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, "expert_routed_flash_candidate"), make_program=_program,
        make_compare=lambda: _compare(case_dir, case), run=_run)
