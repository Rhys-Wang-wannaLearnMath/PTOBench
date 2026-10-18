"""Case-local evaluator for deepseek.v4.prefill.indexer.compressor.pro."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common

workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator/workflow.py",
    "workflow_deepseek_v4_prefill_indexer_compressor_pro",
)


def _oracle_golden(oracle, values):
    scalars = {"num_tokens": values["num_tokens"]}
    tensors = {
        name: value
        for name, value in values.items()
        if isinstance(value, torch.Tensor) and name != "num_tokens"
    }
    for name, value in oracle.compute(tensors, scalars).items():
        values[name][:] = value


def _config(case_dir: Path, case):
    return common.load_module(
        case_dir / case["config"],
        "deepseek_v4_prefill_indexer_compressor_pro_config",
    )


def _fixture_values(fixture: Path):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config):
    values = _fixture_values(fixture)
    return [
        TensorSpec("x", [config.T, config.D], torch.bfloat16, init_value=values["x"]),
        TensorSpec("kv", [config.MAX_CMP_WRITES, config.HEAD_DIM], torch.int8),
        TensorSpec(
            "kv_state",
            [config.INNER_STATE_BLOCK_NUM, config.INNER_STATE_BLOCK_SIZE, config.OUT_DIM],
            torch.float32,
            init_value=values["kv_state"],
        ),
        TensorSpec(
            "score_state",
            [config.INNER_STATE_BLOCK_NUM, config.INNER_STATE_BLOCK_SIZE, config.OUT_DIM],
            torch.float32,
            init_value=values["score_state"],
        ),
        TensorSpec(
            "inner_compress_state_block_table",
            [config.INNER_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=values["inner_compress_state_block_table"],
        ),
        TensorSpec("wkv", [config.OUT_DIM, config.D], torch.bfloat16, init_value=values["wkv"]),
        TensorSpec("wgate", [config.OUT_DIM, config.D], torch.bfloat16, init_value=values["wgate"]),
        TensorSpec(
            "ape",
            [config.COMPRESS_RATIO, config.OUT_DIM],
            torch.float32,
            init_value=values["ape"],
        ),
        TensorSpec("norm_w", [config.HEAD_DIM], torch.bfloat16, init_value=values["norm_w"]),
        TensorSpec(
            "freqs_cos",
            [config.MAX_SEQ_LEN, config.ROPE_HEAD_DIM],
            torch.bfloat16,
            init_value=values["freqs_cos"],
        ),
        TensorSpec(
            "freqs_sin",
            [config.MAX_SEQ_LEN, config.ROPE_HEAD_DIM],
            torch.bfloat16,
            init_value=values["freqs_sin"],
        ),
        TensorSpec(
            "hadamard",
            [config.HEAD_DIM, config.HEAD_DIM],
            torch.bfloat16,
            init_value=values["hadamard"],
        ),
        TensorSpec(
            "idx_kv_cache",
            [config.PREFILL_IDX_BLOCK_NUM, config.BLOCK_SIZE, 1, config.HEAD_DIM],
            torch.int8,
            init_value=values["idx_kv_cache"],
        ),
        TensorSpec(
            "idx_kv_scale",
            [config.PREFILL_IDX_BLOCK_NUM, config.BLOCK_SIZE, 1, 1],
            torch.float32,
            init_value=values["idx_kv_scale"],
        ),
        TensorSpec(
            "idx_block_table",
            [config.IDX_CACHE_MAX_BLOCKS],
            torch.int32,
            init_value=values["idx_block_table"],
        ),
        TensorSpec(
            "position_ids",
            [config.T],
            torch.int32,
            init_value=values["position_ids"],
        ),
        ScalarSpec("num_tokens", torch.int32, values["num_tokens"]),
        TensorSpec(
            "idx_slot_mapping",
            [config.T],
            torch.int64,
            init_value=values["idx_slot_mapping"],
        ),
        TensorSpec(
            "inner_state_slot_mapping",
            [config.T],
            torch.int64,
            init_value=values["inner_state_slot_mapping"],
        ),
    ]


def _finite_nonnegative(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    return number


def _ratio_allclose(name, settings):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    if settings.get("method") != "ratio_allclose":
        raise ValueError(
            f"unsupported comparator method for {name!r}: {settings.get('method')!r}"
        )
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if set(settings) != required:
        fields = ", ".join(sorted(required))
        raise ValueError(
            f"ratio_allclose comparator {name!r} requires exactly {fields}"
        )
    policy_rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], f"{name}.atol")
    policy_max_error_ratio = _finite_nonnegative(
        settings["max_error_ratio"], f"{name}.max_error_ratio"
    )
    if policy_max_error_ratio > 1:
        raise ValueError(f"comparator {name}.max_error_ratio must be at most 1")

    def compare(actual, expected, **_runner_fallbacks):
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
        mismatches = (actual_f - expected_f).abs() > (
            policy_atol + policy_rtol * expected_f.abs()
        )
        error_count = int(mismatches.sum().item())
        allowed_count = round(policy_max_error_ratio * actual_f.numel())
        passed = error_count <= allowed_count
        return passed, (
            ""
            if passed
            else f"ratio_allclose: mismatches={error_count}/{actual_f.numel()}, "
            f"allowed={allowed_count}, max_error_ratio={policy_max_error_ratio}, "
            f"rtol={policy_rtol}, atol={policy_atol}"
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
        name: _ratio_allclose(name, outputs[name])
        for name in expected_names
    }


def _run(
    *,
    fn,
    specs,
    config,
    compare_fn,
    golden_data=None,
    golden_fn=None,
    capture_outputs=False,
):
    return run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=str(golden_data) if golden_data else None,
        config=config,
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
        save_data=False,
        capture_outputs=capture_outputs,
    )


def _program(module, case):
    return getattr(module, case["entry"])


def prepare_ref(case_dir, sealed, case, platform=None):
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: build_tensor_specs(
            fixture, _config(case_dir, case)
        ),
        load_ref=lambda: common.load_ref(
            case_dir, sealed, "deepseek_v4_prefill_indexer_compressor_pro_ref"
        ),
        load_oracle=lambda: common.load_oracle(
            case_dir, sealed, "deepseek_v4_prefill_indexer_compressor_pro_oracle"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
        oracle_golden=_oracle_golden,
    )


def evaluate_candidate(
    case_dir,
    sealed,
    case,
    candidate_workspace,
    platform=None,
    phase="all",
):
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: build_tensor_specs(
            fixture, _config(case_dir, case)
        ),
        load_candidate=lambda: common.load_candidate(
            case_dir,
            case,
            candidate_workspace,
            "deepseek_v4_prefill_indexer_compressor_pro_candidate",
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=_run,
    )
