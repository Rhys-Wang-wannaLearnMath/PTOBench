"""Case-local evaluator for the DeepSeek-V4 DEMO shared expert."""
import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run
import common


workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator/workflow.py",
    "workflow_expert_shared_demo",
)


def _values(fixture: Path):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path):
    values = _values(fixture)
    config = common.load_module(
        common.BENCHMARK_ROOT / "cases/deepseek_v4/expert_shared_demo/public/config.py",
        "expert_shared_demo_specs_config",
    )
    input_names = (
        "x_local_i8", "x_local_scale_dq", "shared_w1", "shared_w1_scale",
        "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    )
    dtypes = {
        "x_local_i8": torch.int8, "x_local_scale_dq": torch.float32,
        "shared_w1": torch.int8, "shared_w1_scale": torch.float32,
        "shared_w3": torch.int8, "shared_w3_scale": torch.float32,
        "shared_w2": torch.int8, "shared_w2_scale": torch.float32,
    }
    specs = [TensorSpec(name, list(values[name].shape), dtypes[name], init_value=values[name])
             for name in input_names]
    specs.append(TensorSpec("sh", [config.T, config.D], torch.bfloat16))
    return specs


def _program(module, case):
    return getattr(module, case["entry"])


def _finite_number(value, field, *, allow_null=False):
    if allow_null and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid comparator {field}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"invalid comparator {field}")
    return value


def _compare(case_dir, case):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if (not isinstance(policy, dict)
            or set(policy) != {"schema_version", "outputs"}
            or policy["schema_version"] != 1):
        raise ValueError("invalid comparator top-level")
    outputs = policy["outputs"]
    names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator output coverage mismatch")
    callbacks = {}
    required = {"method", "name", "diff_thd", "pct_thd", "max_diff_hd",
                "valid_rows", "valid_axis", "zero_tail"}
    for output_name in names:
        spec = outputs[output_name]
        if not isinstance(spec, dict) or set(spec) != required:
            raise ValueError(f"invalid comparator fields for {output_name}")
        if spec["method"] != "custom" or spec["name"] != "ratio_reldiff":
            raise ValueError(f"unsupported comparator method for {output_name}")
        diff = _finite_number(spec["diff_thd"], f"{output_name}.diff_thd")
        pct = _finite_number(spec["pct_thd"], f"{output_name}.pct_thd")
        cap = _finite_number(spec["max_diff_hd"], f"{output_name}.max_diff_hd", allow_null=True)
        rows = spec["valid_rows"]
        if rows is not None and (isinstance(rows, bool) or not isinstance(rows, int) or rows < 0):
            raise ValueError(f"invalid comparator {output_name}.valid_rows")
        axis = spec["valid_axis"]
        if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
            raise ValueError(f"invalid comparator {output_name}.valid_axis")
        if not isinstance(spec["zero_tail"], bool):
            raise ValueError(f"invalid comparator {output_name}.zero_tail")
        if diff <= 0 or pct < 0 or pct > 1 or (cap is not None and cap < 0):
            raise ValueError(f"invalid comparator range for {output_name}")

        def compare(actual, expected, *, diff=diff, pct=pct, cap=cap,
                    rows=rows, axis=axis, zero_tail=spec["zero_tail"], **_):
            if actual.shape != expected.shape:
                return False, "ratio_reldiff: shape mismatch"
            actual = actual.cpu().float()
            expected = expected.cpu().float()
            if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
                return False, "ratio_reldiff: non-finite value"
            if rows is not None:
                if axis >= actual.ndim or rows > actual.shape[axis]:
                    return False, "ratio_reldiff: valid_rows out of range"
                tail = actual.narrow(axis, rows, actual.shape[axis] - rows)
                expected_tail = expected.narrow(axis, rows, expected.shape[axis] - rows)
                if zero_tail and (not bool(torch.all(tail == 0)) or not bool(torch.all(expected_tail == 0))):
                    return False, "ratio_reldiff: nonzero tail"
                actual = actual.narrow(axis, 0, rows)
                expected = expected.narrow(axis, 0, rows)
            delta = (actual - expected).abs()
            floor = (1.0 / (1 << 14)) / diff
            denominator = torch.maximum(torch.maximum(actual.abs(), expected.abs()),
                                        torch.full_like(actual, floor)) + 1e-9
            relative = torch.where(delta < diff, delta, delta / denominator)
            bad = relative > diff
            count = int(bad.sum().item())
            limit = round(pct * actual.numel())
            worst = float(relative[bad].max().item()) if count else 0.0
            passed = count <= limit and (cap is None or worst < cap)
            detail = "" if passed else f"ratio_reldiff: mismatches={count}/{actual.numel()}, allowed={limit}"
            return passed, detail

        callbacks[output_name] = compare
    return callbacks


def _run(**kwargs):
    return run(save_data=False, **kwargs)


def _oracle(oracle, values):
    outputs = oracle.compute({k: v for k, v in values.items() if isinstance(v, torch.Tensor)}, {})
    for name, value in outputs.items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=build_tensor_specs,
        load_ref=lambda: common.load_ref(case_dir, sealed, "expert_shared_demo_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, "expert_shared_demo_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=build_tensor_specs,
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, "expert_shared_demo_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run,
    )
