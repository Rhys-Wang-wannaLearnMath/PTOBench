"""Case-local evaluator for the DeepSeek-V4 decode indexer."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common

_CASE_ID = Path(__file__).resolve().parent.name
_IDENTITY = "deepseek_v4_decode_indexer_" + _CASE_ID.rsplit("_", 1)[-1]
workflow = common.load_module(
    common.BENCHMARK_ROOT / "evaluator" / "workflow.py",
    "workflow_" + _IDENTITY,
)


def _config(case_dir, case):
    return common.load_module(
        case_dir / case["config"],
        _IDENTITY + "_config",
    )


def _values(fixture):
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture, config):
    values = _values(fixture)
    return [
        TensorSpec("x", [config.B, config.S, config.D], torch.bfloat16,
                   init_value=values["x"]),
        TensorSpec("qr", [config.T, config.Q_LORA], torch.int8,
                   init_value=values["qr"]),
        TensorSpec("qr_scale", [config.T, 1], torch.float32,
                   init_value=values["qr_scale"]),
        TensorSpec("wq_b", [config.Q_LORA, config.IDX_N_HEADS * config.IDX_HEAD_DIM],
                   torch.int8, init_value=values["wq_b"]),
        TensorSpec("wq_b_scale", [config.IDX_N_HEADS * config.IDX_HEAD_DIM],
                   torch.float32, init_value=values["wq_b_scale"]),
        TensorSpec("weights_proj", [config.D, config.IDX_N_HEADS], torch.bfloat16,
                   init_value=values["weights_proj"]),
        TensorSpec("cos", [config.B, config.ROPE_HEAD_DIM // 2], torch.float32,
                   init_value=values["cos"]),
        TensorSpec("sin", [config.B, config.ROPE_HEAD_DIM // 2], torch.float32,
                   init_value=values["sin"]),
        TensorSpec("hadamard", [config.IDX_HEAD_DIM, config.IDX_HEAD_DIM],
                   torch.bfloat16, init_value=values["hadamard"]),
        TensorSpec("inner_kv", [config.B, config.S, config.INNER_HEAD_DIM],
                   torch.float32, init_value=values["inner_kv"]),
        TensorSpec("inner_compress_state",
                   [config.INNER_STATE_BLOCK_NUM, config.INNER_STATE_BLOCK_SIZE,
                    config.INNER_STATE_DIM], torch.float32,
                   init_value=values["inner_compress_state"]),
        TensorSpec("inner_compress_state_block_table",
                   [config.B, config.INNER_STATE_MAX_BLOCKS], torch.int32,
                   init_value=values["inner_compress_state_block_table"]),
        TensorSpec("inner_wkv", [config.D, config.INNER_OUT_DIM], torch.bfloat16,
                   init_value=values["inner_wkv"]),
        TensorSpec("inner_wgate", [config.D, config.INNER_OUT_DIM], torch.bfloat16,
                   init_value=values["inner_wgate"]),
        TensorSpec("inner_ape", [config.COMPRESS_RATIO, config.INNER_OUT_DIM],
                   torch.float32, init_value=values["inner_ape"]),
        TensorSpec("inner_norm_w", [config.INNER_HEAD_DIM], torch.float32,
                   init_value=values["inner_norm_w"]),
        TensorSpec("idx_kv_cache",
                   [config.IDX_CACHE_BLOCK_NUM, config.BLOCK_SIZE, 1,
                    config.IDX_HEAD_DIM], torch.bfloat16,
                   init_value=values["idx_kv_cache"]),
        TensorSpec("idx_block_table", [config.B, config.IDX_CACHE_MAX_BLOCKS],
                   torch.int32, init_value=values["idx_block_table"]),
        TensorSpec("score", [config.B, config.S, config.SCORE_LEN],
                   torch.float32),
        TensorSpec("topk_idxs", [config.B, config.S, config.SCORE_LEN],
                   torch.int32),
        TensorSpec("start_pos", [config.B], torch.int32,
                   init_value=values["start_pos"]),
        ScalarSpec("offset", torch.int32, values["offset"]),
    ]


def _finite_nonnegative(value, field, *, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"comparator {field} must be finite and nonnegative")
    if upper is not None and number > upper:
        raise ValueError(f"comparator {field} must be at most {upper}")
    return number


def _ratio_allclose(name, settings):
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f"ratio_allclose comparator {name!r} requires exactly {sorted(required)}"
        )
    if settings.get("method") != "ratio_allclose":
        raise ValueError(f"unsupported comparator method for {name!r}")
    policy_rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], f"{name}.atol")
    max_error_ratio = _finite_nonnegative(
        settings["max_error_ratio"], f"{name}.max_error_ratio", upper=1
    )

    def compare(actual, expected, **_kwargs):
        if actual.shape != expected.shape:
            return False, (
                f"ratio_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
            )
        if actual.numel() == 0:
            return True, "ratio_allclose: empty"
        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all()) or not bool(
            torch.isfinite(expected_f).all()
        ):
            return False, "ratio_allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > (
            policy_atol + policy_rtol * expected_f.abs()
        )
        error_count = int(bad.sum().item())
        allowed = round(max_error_ratio * actual_f.numel())
        return error_count <= allowed, (
            ""
            if error_count <= allowed
            else (
                f"ratio_allclose: mismatches={error_count}/{actual_f.numel()}, "
                f"allowed={allowed}, rtol={policy_rtol}, atol={policy_atol}"
            )
        )

    compare.__name__ = "ratio_allclose"
    return compare


def _topk_pair(name, settings, expected_names, config):
    required = {"method", "name", "values_output", "dim", "descending",
                "topk", "index_offset"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f"custom comparator {name!r} requires exactly {sorted(required)}"
        )
    if settings.get("method") != "custom" or settings.get("name") != "topk_pair_compare":
        raise ValueError(f"unsupported custom comparator for {name!r}")
    values_output = settings["values_output"]
    if not isinstance(values_output, str) or values_output not in expected_names:
        raise ValueError(f"custom comparator {name!r} references undeclared output")
    dim = settings["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"comparator {name!r}.dim must be an integer")
    shape = (config.B, config.S, config.SCORE_LEN)
    if not -len(shape) <= dim < len(shape):
        raise ValueError(f"comparator {name!r}.dim is out of range")
    descending = settings["descending"]
    if not isinstance(descending, bool):
        raise ValueError(f"comparator {name!r}.descending must be boolean")
    topk = settings["topk"]
    offset = settings["index_offset"]
    if type(topk) is not int or not 0 < topk <= shape[dim] or topk != config.IDX_TOPK:
        raise ValueError(f"comparator {name!r}.topk must match the fixed contract")
    if type(offset) is not int or offset < 0 or offset != config.OFFSET:
        raise ValueError(f"comparator {name!r}.index_offset must match the fixed contract")

    def compare(actual, expected, *, actual_outputs, **_kwargs):
        if values_output not in actual_outputs:
            return False, f"topk_pair_compare: missing paired output {values_output!r}"
        if actual.shape != expected.shape:
            return False, (
                f"topk_pair_compare: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
            )
        paired = actual_outputs[values_output]
        if paired.shape != actual.shape:
            return False, "topk_pair_compare: paired shape mismatch"
        dim_pos = dim if dim >= 0 else dim + actual.dim()
        if not 0 <= dim_pos < actual.dim():
            return False, f"topk_pair_compare: dim={dim} out of range"
        a_idx = actual.cpu().movedim(dim_pos, -1)
        e_idx = expected.cpu().movedim(dim_pos, -1)
        paired_score = paired.detach().cpu().to(torch.float32).movedim(dim_pos, -1)
        if a_idx.shape != e_idx.shape or a_idx.shape != paired_score.shape:
            return False, "topk_pair_compare: paired shape mismatch"
        a_idx = a_idx[..., :topk]
        e_idx = e_idx[..., :topk]
        transformed = (a_idx.to(torch.int64) - offset).clamp(
            min=0, max=paired_score.shape[-1] - 1
        )
        paired_score = torch.gather(paired_score, dim=-1,
                                    index=transformed)
        a_idx = a_idx.reshape(-1, a_idx.shape[-1])
        e_idx = e_idx.reshape(-1, e_idx.shape[-1])
        paired_score = paired_score.reshape(-1, paired_score.shape[-1])
        mismatch = a_idx != e_idx
        if not bool(mismatch.any()):
            return True, "exact index equality"
        if a_idx.shape[-1] >= 2:
            adjacent = (
                paired_score[:, :-1] >= paired_score[:, 1:]
                if descending else paired_score[:, :-1] <= paired_score[:, 1:]
            )
            position_ok = torch.ones_like(mismatch)
            position_ok[:, 1:] &= adjacent
            position_ok[:, :-1] &= adjacent
        else:
            position_ok = torch.ones_like(mismatch)
        failed = mismatch & ~position_ok
        if not bool(failed.any()):
            return True, "tie-preserving index permutation"
        return False, (
            f"topk_pair_compare: order violations={int(failed.sum().item())}"
        )

    compare.__name__ = "topk_pair_compare"
    return compare


def _score_valid_region(name, settings):
    required = {"method", "name", "rtol", "atol", "max_error_ratio",
                "invalid_value", "hard_multiplier", "hard_atol"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f"score comparator {name!r} requires exactly {sorted(required)}")
    if settings["method"] != "custom" or settings["name"] != "score_valid_region_compare":
        raise ValueError(f"unsupported custom comparator for {name!r}")
    policy_rtol = _finite_nonnegative(settings["rtol"], f"{name}.rtol")
    policy_atol = _finite_nonnegative(settings["atol"], f"{name}.atol")
    ratio = _finite_nonnegative(settings["max_error_ratio"], f"{name}.max_error_ratio", upper=1)
    hard_multiplier = _finite_nonnegative(settings["hard_multiplier"], f"{name}.hard_multiplier")
    hard_atol = _finite_nonnegative(settings["hard_atol"], f"{name}.hard_atol")
    invalid_value = settings["invalid_value"]
    if (isinstance(invalid_value, bool) or not isinstance(invalid_value, (int, float))
            or not math.isfinite(invalid_value)
            or abs(invalid_value) > torch.finfo(torch.float32).max):
        raise ValueError(f"comparator {name!r}.invalid_value must be finite in FP32")

    def compare(actual, expected, **_kwargs):
        if actual.shape != expected.shape:
            return False, "score_valid_region_compare: shape mismatch"
        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
            return False, "score_valid_region_compare: non-finite full tensor"
        valid = expected_f != invalid_value
        a, e = actual_f[valid], expected_f[valid]
        error = (a - e).abs()
        bad_count = int((error > policy_atol + policy_rtol * e.abs()).sum().item())
        allowed = round(ratio * e.numel())
        if bad_count > allowed:
            return False, f"score_valid_region_compare: mismatches={bad_count}/{e.numel()}, allowed={allowed}"
        if e.numel() == 0:
            return True, "score_valid_region_compare: empty visible region"
        hard_limit = torch.maximum(
            hard_multiplier * (policy_atol + policy_rtol * torch.maximum(a.abs(), e.abs())),
            torch.full_like(e, hard_atol),
        )
        if not bool((error <= hard_limit).all()):
            return False, "score_valid_region_compare: hard error ceiling exceeded"
        return True, "score_valid_region_compare: visible region passed"

    compare.__name__ = "score_valid_region_compare"
    return compare


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate comparator field: {key!r}")
        result[key] = value
    return result


def _compare(case_dir, case, config):
    policy = json.loads(
        (case_dir / case["comparator"]).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema_version", "outputs"}
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
    ):
        raise ValueError(
            "comparator.json must contain only schema_version=1 and outputs"
        )
    output_names = list(case["correctness"]["outputs"])
    outputs = policy["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(output_names):
        raise ValueError(
            "comparator outputs must exactly match correctness outputs"
        )
    callbacks = {}
    for name in output_names:
        settings = outputs[name]
        if not isinstance(settings, dict):
            raise ValueError(f"comparator output {name!r} must be an object")
        method = settings.get("method")
        if name == "idx_kv_cache" and method == "ratio_allclose":
            callbacks[name] = _ratio_allclose(name, settings)
        elif name == "score" and method == "custom":
            callbacks[name] = _score_valid_region(name, settings)
        elif name == "topk_idxs" and method == "custom":
            callbacks[name] = _topk_pair(name, settings, output_names, config)
        else:
            raise ValueError(
                f"unsupported comparator method for {name!r}: {method!r}"
            )
    return callbacks


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None,
         capture_outputs=False):
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
        reset_input_names=("inner_kv", "inner_compress_state"),
    )


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    tensors = {
        name: value for name, value in values.items()
        if isinstance(value, torch.Tensor) and name != "offset"
    }
    outputs = oracle.compute(tensors, {"offset": int(values["offset"])})
    for name, value in outputs.items():
        values[name][:] = value


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, _IDENTITY + "_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, _IDENTITY + "_oracle"),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case, config),
        run=_run,
        oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None,
                       phase="all"):
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
            case_dir, case, candidate_workspace, _IDENTITY + "_candidate"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case, config),
        run=_run,
    )
