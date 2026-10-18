"""Case-local evaluator for DeepSeek-V4 HC prefill pre-mix."""
import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run
import common

_CASE_NAME = Path(__file__).resolve().parent.name
workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator/workflow.py", f"workflow_{_CASE_NAME}")

def _identity(case, role):
    return f"{case['case_id'].replace('.', '_')}_{role}"

def _values(fixture):
    return {p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in (fixture / "in").glob("*.pt")}

def _config(case_dir, case):
    return common.load_module(case_dir / case["config"], _identity(case, "public_config"))

def _build_specs(fixture, config):
    v = _values(fixture)
    return [
        TensorSpec("x", [config.B, config.S, config.HC_MULT, config.D], torch.bfloat16, init_value=v["x"]),
        TensorSpec("hc_fn", [(2 + config.HC_MULT) * config.HC_MULT, config.HC_DIM], torch.float32, init_value=v["hc_fn"]),
        TensorSpec("hc_scale", [3], torch.float32, init_value=v["hc_scale"]),
        TensorSpec("hc_base", [(2 + config.HC_MULT) * config.HC_MULT], torch.float32, init_value=v["hc_base"]),
        TensorSpec("x_mixed", [config.B, config.S, config.D], torch.bfloat16),
        TensorSpec("post", [config.B, config.S, config.HC_MULT], torch.float32),
        TensorSpec("comb", [config.B, config.S, config.HC_MULT, config.HC_MULT], torch.float32),
    ]

def _program(module, case):
    return getattr(module, case["entry"])

def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return float(value)

def _ratio(name, settings):
    if not isinstance(settings, dict) or settings.get("method") != "ratio_allclose":
        raise ValueError(f"unsupported comparator method for {name!r}")
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if set(settings) != required:
        raise ValueError(f"ratio_allclose comparator {name!r} has invalid fields")
    rtol, atol, ratio = (_number(settings[k], f"{name}.{k}") for k in ("rtol", "atol", "max_error_ratio"))
    if ratio > 1:
        raise ValueError(f"comparator {name}.max_error_ratio must be at most 1")
    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, f"ratio_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        if actual.numel() == 0:
            return True, ""
        a, e = actual.cpu().to(torch.float32), expected.cpu().to(torch.float32)
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(e).all()):
            return False, "ratio_allclose: non-finite value"
        bad = (a - e).abs() > (atol + rtol * e.abs())
        count, limit = int(bad.sum().item()), round(ratio * a.numel())
        return (count <= limit, "" if count <= limit else f"ratio_allclose: mismatches={count}/{a.numel()}, allowed={ratio}, rtol={rtol}, atol={atol}")
    return compare

def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or policy.get("schema_version") != 1:
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs, expected = policy["outputs"], case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError("comparator outputs must exactly match case correctness outputs")
    return {name: _ratio(name, outputs[name]) for name in expected}

def _run(**kwargs):
    return run(rtol=1e-3, atol=1e-3, save_data=False, **kwargs)

def _oracle(oracle, values):
    tensors = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, value in oracle.compute(tensors, {}).items():
        tensors[name][:] = value

def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: _build_specs(fixture, _config(case_dir, case)),
        load_ref=lambda: common.load_ref(case_dir, sealed, _identity(case, "ref")),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, _identity(case, "oracle")),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)

def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase, build_specs=lambda fixture: _build_specs(fixture, _config(case_dir, case)),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, _identity(case, "candidate")),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
