"""Case evaluator for the DeepSeek V4 CP4 zigzag exchange."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import TensorSpec, run

import common


workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    "workflow_deepseek_v4_flash_mtp_cp4_pro",
)


def _values(fixture: Path) -> dict[str, torch.Tensor]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def _specs(case_dir: Path, fixture: Path, case: dict) -> list[TensorSpec]:
    config = common.load_module(case_dir / case["config"], "cp4_pro_public_config")
    values = _values(fixture)
    layout = [
        ("local_kv_tail", (config.CP_SIZE, 2 * config.TAIL_ROWS, config.HEAD_DIM), torch.bfloat16),
        ("reverse_index", (config.NUM_SEGMENTS,), torch.int32),
        ("owner_rank_table", (config.NUM_SEGMENTS,), torch.int32),
        ("final_win_seg_src", (config.TAIL_ROWS,), torch.int32),
        ("final_win_row_src", (config.TAIL_ROWS,), torch.int32),
        ("logical_tails_out", (config.CP_SIZE, config.EPOCHS * config.CP_TAIL_WINDOW_ROWS, config.HEAD_DIM), torch.bfloat16),
        ("decode_raw_window_out", (config.CP_SIZE, config.TAIL_ROWS, config.HEAD_DIM), torch.bfloat16),
    ]
    outputs = set(case["correctness"]["outputs"])
    return [TensorSpec(name, shape, dtype, init_value=values[name]) if name not in outputs
            else TensorSpec(name, shape, dtype) for name, shape, dtype in layout]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _allclose(name: str, settings: object):
    if not isinstance(settings, dict) or set(settings) != {"method", "rtol", "atol"}:
        raise ValueError(f"allclose comparator {name!r} requires exactly method, rtol, atol")
    if settings.get("method") != "allclose":
        raise ValueError(f"unsupported comparator method for {name!r}: {settings.get('method')!r}")
    policy_rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_kwargs):
        if actual.shape != expected.shape:
            return False, f"allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        passed = bool(torch.allclose(actual, expected, rtol=policy_rtol, atol=policy_atol))
        if passed:
            return True, f"torch.allclose(rtol={policy_rtol}, atol={policy_atol})"
        close = torch.isclose(actual, expected, rtol=policy_rtol, atol=policy_atol)
        mismatches = int((~close).sum().item())
        return False, f"torch.allclose(rtol={policy_rtol}, atol={policy_atol}): mismatches={mismatches}/{actual.numel()}"

    return compare


def _compare(case_dir: Path, case: dict):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    if (not isinstance(policy, dict)
            or set(policy) != {"schema_version", "outputs"}
            or isinstance(policy["schema_version"], bool)
            or not isinstance(policy["schema_version"], int)
            or policy["schema_version"] != 1):
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy["outputs"]
    expected = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError("comparator outputs must exactly match case correctness outputs")
    callbacks = {name: _allclose(name, outputs[name]) for name in expected}
    if set(callbacks) != set(expected):
        raise ValueError("comparator callback coverage mismatch")
    return callbacks


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
    return run(fn=fn, specs=specs, golden_fn=golden_fn,
               golden_data=str(golden_data) if golden_data else None,
               config=config, compare_fn=compare_fn, save_data=False,
               capture_outputs=capture_outputs)


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    inputs = {name: value for name, value in values.items() if isinstance(value, torch.Tensor)}
    for name, answer in oracle.compute(inputs, {}).items():
        values[name][:] = answer


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: _specs(case_dir, fixture, case),
        load_ref=lambda: common.load_ref(case_dir, sealed, "cp4_pro_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, "cp4_pro_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: _specs(case_dir, fixture, case),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, "cp4_pro_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run,
    )
