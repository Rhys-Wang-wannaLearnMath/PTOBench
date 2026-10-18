"""Case-local evaluator for the legacy DeepSeek-V4 SWA decode."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common


TENSOR_NAMES = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "even_select_t",
    "odd_select_t",
    "kv_cache",
    "block_table",
    "attn_sink",
    "seqused_kv",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)


def _config(case_dir: Path, case: dict):
    identity = case["case_id"].replace(".", "_")
    return common.load_module(case_dir / case["config"], f"{identity}_config")


def _values(fixture: Path) -> dict[str, torch.Tensor]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config) -> list[TensorSpec | ScalarSpec]:
    values = _values(fixture)
    expected = {*TENSOR_NAMES, "start_pos"}
    if set(values) != expected:
        raise ValueError(
            f"fixture inputs do not match the fixed ABI: {sorted(values)} != {sorted(expected)}"
        )
    specs: list[TensorSpec | ScalarSpec] = []
    for name in TENSOR_NAMES:
        value = values[name]
        specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    specs.append(
        TensorSpec(
            "x_out",
            [config.B, config.S, config.HC_MULT, config.D],
            torch.bfloat16,
        )
    )
    specs.append(ScalarSpec("start_pos", torch.int32, values["start_pos"]))
    return specs


def _number(value, field: str, *, ratio: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (ratio and number > 1):
        raise ValueError(f"comparator {field} has invalid range")
    return number


def _callback(name: str, settings: dict):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    if settings.get("method") != "ratio_allclose":
        raise ValueError(f"unsupported comparator method for {name!r}")
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if set(settings) != required:
        raise ValueError("ratio_allclose requires exactly its declared fields")
    rtol = _number(settings["rtol"], f"{name}.rtol")
    atol = _number(settings["atol"], f"{name}.atol")
    max_error_ratio = _number(
        settings["max_error_ratio"], f"{name}.max_error_ratio", ratio=True
    )

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, "ratio_allclose: shape mismatch"
        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all()) or not bool(
            torch.isfinite(expected_f).all()
        ):
            return False, "ratio_allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > (atol + rtol * expected_f.abs())
        mismatch_count = int(bad.sum().item())
        allowed_count = round(max_error_ratio * actual_f.numel())
        passed = mismatch_count <= allowed_count
        return passed, (
            f"ratio_allclose: mismatches={mismatch_count}/{actual_f.numel()}, "
            f"allowed={allowed_count}, rtol={rtol}, atol={atol}"
        )

    return compare


def _compare(case_dir: Path, case: dict) -> dict:
    policy = json.loads(
        (case_dir / case["comparator"]).read_text(encoding="utf-8")
    )
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
    ):
        raise ValueError("comparator.json top-level schema mismatch")
    outputs = policy["outputs"]
    names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator output coverage mismatch")
    return {name: _callback(name, outputs[name]) for name in names}


def _program(module, case: dict):
    return getattr(module, case["entry"])


def _oracle(oracle, values: dict) -> None:
    tensors = {
        name: value for name, value in values.items() if isinstance(value, torch.Tensor)
    }
    scalars = {"start_pos": values["start_pos"]}
    for name, value in oracle.compute(tensors, scalars).items():
        values[name][:] = value


def _run(**kwargs):
    return run(rtol=0.0, atol=0.0, save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    workflow = common.load_module(
        common.BENCHMARK_ROOT / "evaluator/workflow.py",
        "workflow_deepseek_v4_deepseek_v4_decode_swa",
    )
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(
            case_dir, sealed, "deepseek_v4_deepseek_v4_decode_swa_ref"
        ),
        load_oracle=lambda: common.load_oracle(
            case_dir, sealed, "deepseek_v4_deepseek_v4_decode_swa_oracle"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(
    case_dir,
    sealed,
    case,
    candidate_workspace,
    platform=None,
    phase="all",
):
    workflow = common.load_module(
        common.BENCHMARK_ROOT / "evaluator/workflow.py",
        "workflow_deepseek_v4_deepseek_v4_decode_swa",
    )
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(
            case_dir,
            case,
            candidate_workspace,
            "deepseek_v4_deepseek_v4_decode_swa_candidate",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
