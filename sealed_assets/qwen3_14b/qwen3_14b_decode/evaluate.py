"""Case-local evaluator for Qwen3-14B single-layer decode."""

from __future__ import annotations

import json
import math

from pathlib import Path

import torch
from golden import TensorSpec, run

import common


workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    "workflow_qwen3_14b_decode",
)


def _config(case_dir: Path, case: dict):
    return common.load_module(
        case_dir / case["config"],
        "qwen3_14b_decode_config",
    )


def _values(fixture: Path) -> dict[str, torch.Tensor]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config) -> list[TensorSpec]:
    values = _values(fixture)
    return [
        TensorSpec("hidden_states", [config.BATCH, config.HIDDEN], torch.bfloat16,
                   init_value=values["hidden_states"]),
        TensorSpec("input_rms_weight", [1, config.HIDDEN], torch.float32,
                   init_value=values["input_rms_weight"]),
        TensorSpec("wq", [config.HIDDEN, config.HIDDEN], torch.bfloat16,
                   init_value=values["wq"]),
        TensorSpec("wk", [config.HIDDEN, config.KV_HIDDEN], torch.bfloat16,
                   init_value=values["wk"]),
        TensorSpec("wv", [config.HIDDEN, config.KV_HIDDEN], torch.bfloat16,
                   init_value=values["wv"]),
        TensorSpec("q_norm_weight", [1, config.HEAD_DIM], torch.float32,
                   init_value=values["q_norm_weight"]),
        TensorSpec("k_norm_weight", [1, config.HEAD_DIM], torch.float32,
                   init_value=values["k_norm_weight"]),
        TensorSpec("seq_lens", [config.BATCH], torch.int32,
                   init_value=values["seq_lens"]),
        TensorSpec("block_table", [config.NUM_BLOCKS], torch.int32,
                   init_value=values["block_table"]),
        TensorSpec("slot_mapping", [config.BATCH], torch.int32,
                   init_value=values["slot_mapping"]),
        TensorSpec("rope_cos", [config.MAX_SEQ, config.HEAD_DIM], torch.float32,
                   init_value=values["rope_cos"]),
        TensorSpec("rope_sin", [config.MAX_SEQ, config.HEAD_DIM], torch.float32,
                   init_value=values["rope_sin"]),
        TensorSpec("k_cache", [config.CACHE_ROWS, config.HEAD_DIM], torch.bfloat16,
                   init_value=values["k_cache"]),
        TensorSpec("v_cache", [config.CACHE_ROWS, config.HEAD_DIM], torch.bfloat16,
                   init_value=values["v_cache"]),
        TensorSpec("wo", [config.HIDDEN, config.HIDDEN], torch.bfloat16,
                   init_value=values["wo"]),
        TensorSpec("post_rms_weight", [1, config.HIDDEN], torch.float32,
                   init_value=values["post_rms_weight"]),
        TensorSpec("w_gate", [config.HIDDEN, config.INTERMEDIATE], torch.bfloat16,
                   init_value=values["w_gate"]),
        TensorSpec("w_up", [config.HIDDEN, config.INTERMEDIATE], torch.bfloat16,
                   init_value=values["w_up"]),
        TensorSpec("w_down", [config.INTERMEDIATE, config.HIDDEN], torch.bfloat16,
                   init_value=values["w_down"]),
        TensorSpec("out", [config.BATCH, config.HIDDEN], torch.bfloat16),
    ]


def _run(
    *, fn, specs, config, compare_fn, golden_data=None, golden_fn=None,
    capture_outputs=False,
):
    return run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=str(golden_data) if golden_data else None,
        config=config,
        rtol=0.003,
        atol=0.003,
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


def _program(module, case):
    return getattr(module, case["entry"])()


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _standard_comparator(name, settings):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    method = settings.get("method")

    if method == "exact":
        if set(settings) != {"method"}:
            raise ValueError(f"exact comparator {name!r} accepts only method")

        def compare(actual, expected, **_):
            passed = torch.equal(actual, expected)
            if passed:
                return True, "torch.equal"
            if actual.shape != expected.shape:
                return False, (
                    f"torch.equal: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
                )
            mismatches = int((actual != expected).sum().item())
            return False, f"torch.equal: mismatches={mismatches}/{actual.numel()}"

        return compare

    if method not in {"allclose", "ratio_allclose"}:
        raise ValueError(f"unsupported comparator method for {name!r}: {method!r}")

    required = {"method", "rtol", "atol"}
    if method == "ratio_allclose":
        required.add("max_error_ratio")
    if set(settings) != required:
        fields = ", ".join(sorted(required))
        raise ValueError(f"{method} comparator {name!r} requires exactly {fields}")

    rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    atol = _finite_nonnegative(settings["atol"], f"{name}.atol")
    if method == "allclose":
        def compare(actual, expected, **_):
            passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
            if passed:
                return True, f"torch.allclose(rtol={rtol}, atol={atol})"
            close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
            mismatches = int((~close).sum().item())
            return False, (
                f"torch.allclose(rtol={rtol}, atol={atol}): "
                f"mismatches={mismatches}/{actual.numel()}"
            )

        return compare

    max_error_ratio = _finite_nonnegative(
        settings["max_error_ratio"], f"{name}.max_error_ratio"
    )
    if max_error_ratio > 1:
        raise ValueError(f"comparator {name}.max_error_ratio must be at most 1")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, (
                f"ratio_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
            )
        if actual.numel() == 0:
            return True, ""
        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all()) or not bool(
            torch.isfinite(expected_f).all()
        ):
            return False, "ratio_allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > (atol + rtol * expected_f.abs())
        error_count = int(bad.sum().item())
        threshold = round(max_error_ratio * actual_f.numel())
        passed = error_count <= threshold
        return passed, (
            ""
            if passed
            else f"ratio_allclose: mismatches={error_count}/{actual_f.numel()}, "
            f"allowed={max_error_ratio}, rtol={rtol}, atol={atol}"
        )

    return compare


def _compare(case_dir, case):
    path = case_dir / case["comparator"]
    policy = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or policy["schema_version"] != 1
    ):
        raise ValueError(
            "comparator.json must contain only schema_version=1 and outputs"
        )
    outputs = policy["outputs"]
    expected_names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names):
        actual_names = (
            sorted(outputs) if isinstance(outputs, dict) else type(outputs).__name__
        )
        raise ValueError(
            "comparator outputs must exactly match case correctness outputs: "
            f"{actual_names} != {sorted(expected_names)}"
        )
    return {
        name: _standard_comparator(name, outputs[name])
        for name in expected_names
    }


def _oracle(oracle, values):
    inputs = {
        name: value
        for name, value in values.items()
        if isinstance(value, torch.Tensor) and name != "out"
    }
    for name, value in oracle.compute(inputs, {}).items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(
            case_dir, sealed, "qwen3_14b_decode_ref",
        ),
        load_oracle=lambda: common.load_oracle(
            case_dir, sealed, "qwen3_14b_decode_oracle",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(
    case_dir, sealed, case, candidate_workspace, platform=None, phase="all",
):
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
            "qwen3_14b_decode_candidate",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
