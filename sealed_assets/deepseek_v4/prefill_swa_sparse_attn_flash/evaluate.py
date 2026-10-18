"""Case-local evaluator for deepseek_v4.prefill_swa_sparse_attn_flash."""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from golden import ScalarSpec, TensorSpec, run
import common

_WORKFLOW = common.load_module(common.BENCHMARK_ROOT / "evaluator/workflow.py", "deepseek_v4_prefill_swa_sparse_attn_flash_workflow")

def _values(fixture: Path):
    return {p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in (fixture / "in").glob("*.pt")}

def _build_specs(fixture: Path, case_dir: Path):
    v = _values(fixture)
    c = common.load_module(case_dir / "public/config.py", "deepseek_v4_prefill_swa_sparse_attn_flash_config")
    specs = [
      TensorSpec("q", [c.T,c.H,c.HEAD_DIM], torch.bfloat16, init_value=v["q"]),
      TensorSpec("kv", [c.T,c.HEAD_DIM], torch.bfloat16, init_value=v["kv"]),
      TensorSpec("kv_cache", [c.BLOCK_NUM,c.BLOCK_SIZE,1,c.HEAD_DIM], torch.bfloat16, init_value=v["kv_cache"]),
      TensorSpec("block_table", [c.B,c.MAX_BLOCKS], torch.int32, init_value=v["block_table"]),
      TensorSpec("attn_sink", [c.H], torch.float32, init_value=v["attn_sink"]),
      TensorSpec("freqs_cos_t", [c.T,c.ROPE_HEAD_DIM], torch.bfloat16, init_value=v["freqs_cos_t"]),
      TensorSpec("freqs_sin_t", [c.T,c.ROPE_HEAD_DIM], torch.bfloat16, init_value=v["freqs_sin_t"]),
      TensorSpec("even_select_local", list(c.SELECTOR_SHAPE), torch.bfloat16, init_value=v["even_select_local"]),
      TensorSpec("odd_select_local", list(c.SELECTOR_SHAPE), torch.bfloat16, init_value=v["odd_select_local"]),
      TensorSpec("wo_a", [c.O_GROUPS,c.O_LORA,c.O_GROUP_IN], torch.bfloat16, init_value=v["wo_a"]),
      TensorSpec("wo_b", [c.D,c.O_GROUPS*c.O_LORA], torch.int8, init_value=v["wo_b"]),
      TensorSpec("wo_b_scale", [c.D], torch.float32, init_value=v["wo_b_scale"]),
      TensorSpec("attn_out", [c.T,c.D], torch.bfloat16),
      ScalarSpec("start_pos", torch.int32, int(v["start_pos"])) ,
    ]
    return specs

def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return float(value)

def _callback(name, settings):
    if not isinstance(settings, dict): raise ValueError(f"comparator {name!r} must be an object")
    method = settings.get("method")
    if method == "allclose":
        if set(settings) != {"method","rtol","atol"}: raise ValueError(f"allclose comparator {name!r} has invalid fields")
        policy_rtol, policy_atol = _number(settings["rtol"], f"{name}.rtol"), _number(settings["atol"], f"{name}.atol")
        def compare(actual, expected, **kwargs):
            del kwargs
            if actual.shape != expected.shape: return False, f"allclose shape mismatch {tuple(actual.shape)}/{tuple(expected.shape)}"
            ok = torch.isclose(actual, expected, rtol=policy_rtol, atol=policy_atol)
            passed = bool(ok.all())
            return passed, "" if passed else f"allclose mismatches={int((~ok).sum())}/{actual.numel()}"
        return compare
    if method == "ratio_allclose":
        required = {"method","rtol","atol","max_error_ratio"}
        if set(settings) != required: raise ValueError(f"ratio_allclose comparator {name!r} has invalid fields")
        policy_rtol, policy_atol = _number(settings["rtol"], f"{name}.rtol"), _number(settings["atol"], f"{name}.atol")
        max_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio")
        if max_ratio > 1: raise ValueError(f"comparator {name}.max_error_ratio must be at most 1")
        def compare(actual, expected, **kwargs):
            del kwargs
            if actual.shape != expected.shape: return False, f"ratio_allclose shape mismatch {tuple(actual.shape)}/{tuple(expected.shape)}"
            a, e = actual.float(), expected.float()
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(e).all()): return False, "ratio_allclose non-finite value"
            bad = (a-e).abs() > policy_atol + policy_rtol*e.abs()
            count = int(bad.sum())
            passed = count <= round(max_ratio*a.numel())
            return passed, "" if passed else f"ratio_allclose mismatches={count}/{a.numel()}"
        return compare
    raise ValueError(f"unsupported comparator method for {name!r}: {method!r}")

def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version","outputs"} or type(policy["schema_version"]) is not int or policy["schema_version"] != 1: raise ValueError("invalid comparator top-level")
    names = case["correctness"]["outputs"]
    if not isinstance(policy["outputs"], dict) or set(policy["outputs"]) != set(names): raise ValueError("comparator output coverage mismatch")
    return {name: _callback(name, policy["outputs"][name]) for name in names}

def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
    return run(fn=fn, specs=specs, golden_fn=golden_fn, golden_data=str(golden_data) if golden_data else None, config=config, rtol=config.rtol, atol=config.atol, compare_fn=compare_fn, save_data=False, capture_outputs=capture_outputs)

def _program(module, case): return getattr(module, case["entry"])

def _oracle(oracle, values):
    tensors = {k:v for k,v in values.items() if isinstance(v, torch.Tensor)}
    scalars = {k:v for k,v in values.items() if not isinstance(v, torch.Tensor)}
    for name, value in oracle.compute(tensors, scalars).items(): values[name][:] = value

def prepare_ref(case_dir, sealed, case, platform=None):
    return _WORKFLOW.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform, build_specs=lambda f:_build_specs(f,case_dir), load_ref=lambda:common.load_ref(case_dir,sealed,"deepseek_v4_prefill_swa_sparse_attn_flash_ref"), load_oracle=lambda:common.load_oracle(case_dir,sealed,"deepseek_v4_prefill_swa_sparse_attn_flash_oracle"), make_program=_program, make_compare=lambda:_compare(case_dir,case), run=_run, oracle_golden=_oracle)

def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return _WORKFLOW.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=lambda f:_build_specs(f,case_dir), load_candidate=lambda:common.load_candidate(case_dir,case,candidate_workspace,"deepseek_v4_prefill_swa_sparse_attn_flash_candidate"), make_program=_program, make_compare=lambda:_compare(case_dir,case), run=_run)
