"""Case-local evaluator for the packed DeepSeek-V4 prefill indexer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common


NAMES = (
    "x",
    "qr",
    "qr_scale",
    "wq_b",
    "wq_b_scale",
    "weights_proj",
    "cos",
    "sin",
    "freqs_cos",
    "freqs_sin",
    "hadamard",
    "inner_kv_state",
    "inner_score_state",
    "inner_compress_state_block_table",
    "inner_wkv",
    "inner_wgate",
    "inner_ape",
    "inner_norm_w",
    "idx_kv_cache",
    "idx_kv_scale",
    "idx_block_table",
    "score",
    "topk_idxs",
    "position_ids",
    "num_tokens",
    "idx_slot_mapping",
    "inner_state_slot_mapping",
)
PURE_OUTPUTS = {"score", "topk_idxs"}
INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def _identity(case: dict) -> str:
    return case["case_id"].replace(".", "_")


def _config(case_dir: Path, case: dict):
    return common.load_module(
        case_dir / case["config"], f"{_identity(case)}_config"
    )


def _fixture_values(fixture: Path) -> dict[str, torch.Tensor]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config):
    values = _fixture_values(fixture)
    expected = set(NAMES) - PURE_OUTPUTS
    if set(values) != expected:
        raise ValueError(
            f"fixture input coverage mismatch: {sorted(values)} != {sorted(expected)}"
        )
    specs = []
    for name in NAMES:
        if name == "score":
            specs.append(
                TensorSpec(
                    name,
                    [config.T, config.INDEXER_SCORE_CAP],
                    torch.float32,
                )
            )
        elif name == "topk_idxs":
            specs.append(
                TensorSpec(
                    name,
                    [config.T, config.INDEXER_SCORE_CAP],
                    torch.int32,
                )
            )
        elif name == "num_tokens":
            value = values[name]
            if value.ndim != 0 or value.dtype != torch.int32:
                raise ValueError("num_tokens fixture must be a scalar int32 tensor")
            specs.append(ScalarSpec(name, torch.int32, value))
        else:
            value = values[name]
            specs.append(
                TensorSpec(name, list(value.shape), value.dtype, init_value=value)
            )
    return specs


def _number(value, field: str, *, ratio: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (ratio and number > 1):
        raise ValueError(f"comparator {field} has invalid range")
    return number


def _integer(value, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"comparator {field} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"comparator {field} has invalid range")
    return value


def _ratio_result(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    policy_rtol: float,
    policy_atol: float,
    max_error_ratio: float,
):
    if actual.shape != expected.shape:
        return False, (
            f"ratio_allclose: shape={tuple(actual.shape)}/{tuple(expected.shape)}"
        )
    if actual.numel() == 0:
        return True, ""
    actual_f = actual.detach().cpu().to(torch.float32)
    expected_f = expected.detach().cpu().to(torch.float32)
    if not bool(torch.isfinite(actual_f).all()) or not bool(
        torch.isfinite(expected_f).all()
    ):
        return False, "ratio_allclose: non-finite value"
    bad = (actual_f - expected_f).abs() > (
        policy_atol + policy_rtol * expected_f.abs()
    )
    count = int(bad.sum().item())
    allowed = round(max_error_ratio * actual_f.numel())
    return count <= allowed, (
        f"ratio_allclose: mismatches={count}/{actual_f.numel()}, allowed={allowed}, "
        f"rtol={policy_rtol}, atol={policy_atol}"
    )


def _ratio_callback(name: str, settings: dict):
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f"ratio_allclose comparator {name!r} requires exactly {sorted(required)}"
        )
    if settings.get("method") != "ratio_allclose":
        raise ValueError(f"unsupported comparator method for {name!r}")
    policy_rtol = _number(settings["rtol"], f"{name}.rtol")
    policy_atol = _number(settings["atol"], f"{name}.atol")
    max_error_ratio = _number(
        settings["max_error_ratio"], f"{name}.max_error_ratio", ratio=True
    )

    def compare(actual, expected, **_):
        return _ratio_result(
            actual,
            expected,
            policy_rtol=policy_rtol,
            policy_atol=policy_atol,
            max_error_ratio=max_error_ratio,
        )

    compare.__name__ = "ratio_allclose"
    return compare


def _mapped_rows_callback(name: str, settings: dict):
    required = {
        "method",
        "name",
        "mapping_input",
        "flatten_leading_dims",
        "active",
        "inactive",
    }
    if set(settings) != required:
        raise ValueError(
            f"mapped comparator {name!r} requires exactly {sorted(required)}"
        )
    if settings["name"] != "mapped_active_rows_ratio_allclose":
        raise ValueError(f"unsupported custom comparator for {name!r}")
    mapping_input = settings["mapping_input"]
    if not isinstance(mapping_input, str) or not mapping_input:
        raise ValueError(f"comparator {name}.mapping_input must be a name")
    flatten_leading_dims = _integer(
        settings["flatten_leading_dims"],
        f"{name}.flatten_leading_dims",
        positive=True,
    )
    active = settings["active"]
    active_required = {"method", "rtol", "atol", "max_error_ratio"}
    if not isinstance(active, dict) or set(active) != active_required:
        raise ValueError(f"comparator {name}.active has invalid fields")
    if active["method"] != "ratio_allclose":
        raise ValueError(f"comparator {name}.active must use ratio_allclose")
    policy_rtol = _number(active["rtol"], f"{name}.active.rtol")
    policy_atol = _number(active["atol"], f"{name}.active.atol")
    max_error_ratio = _number(
        active["max_error_ratio"],
        f"{name}.active.max_error_ratio",
        ratio=True,
    )
    inactive = settings["inactive"]
    if not isinstance(inactive, dict) or inactive != {"method": "exact"}:
        raise ValueError(f"comparator {name}.inactive must be exact")

    def compare(actual, expected, *, inputs, **_):
        if actual.shape != expected.shape:
            return False, "mapped rows: shape mismatch"
        if actual.ndim <= flatten_leading_dims:
            return False, "mapped rows: tensor rank is too small"
        mapping = inputs.get(mapping_input)
        if mapping is None or mapping.dtype not in INTEGER_DTYPES:
            return False, f"mapped rows: missing or non-integer {mapping_input}"
        mapping = mapping.detach().cpu().to(torch.int64).flatten()
        if bool((mapping < -1).any()):
            return False, "mapped rows: only -1 is a valid negative sentinel"
        active_rows = mapping[mapping >= 0]
        row_count = math.prod(actual.shape[:flatten_leading_dims])
        if active_rows.numel() != torch.unique(active_rows).numel():
            return False, "mapped rows: active row mappings must be unique"
        if active_rows.numel() and int(active_rows.max().item()) >= row_count:
            return False, "mapped rows: active row mapping is out of range"
        actual_rows = actual.detach().cpu().reshape(row_count, -1)
        expected_rows = expected.detach().cpu().reshape(row_count, -1)
        active_mask = torch.zeros(row_count, dtype=torch.bool)
        active_mask[active_rows] = True
        passed, detail = _ratio_result(
            actual_rows[active_mask],
            expected_rows[active_mask],
            policy_rtol=policy_rtol,
            policy_atol=policy_atol,
            max_error_ratio=max_error_ratio,
        )
        if not passed:
            return False, f"mapped active rows: {detail}"
        if not torch.equal(actual_rows[~active_mask], expected_rows[~active_mask]):
            return False, "mapped inactive rows: torch.equal failed"
        return True, ""

    compare.__name__ = "mapped_active_rows_ratio_allclose"
    return compare


def _indexed_topk_callback(name: str, settings: dict):
    required = {
        "method",
        "name",
        "values_output",
        "compared_columns",
        "index_offset",
        "invalid_below_offset_to_neg_inf",
        "dim",
        "descending",
    }
    if set(settings) != required:
        raise ValueError(
            f"indexed top-k comparator {name!r} requires exactly {sorted(required)}"
        )
    if settings["name"] != "indexed_topk_pair":
        raise ValueError(f"unsupported custom comparator for {name!r}")
    values_output = settings["values_output"]
    if not isinstance(values_output, str) or not values_output:
        raise ValueError(f"comparator {name}.values_output must be a name")
    compared_columns = _integer(
        settings["compared_columns"], f"{name}.compared_columns", positive=True
    )
    index_offset = _integer(settings["index_offset"], f"{name}.index_offset")
    invalid_to_neg_inf = settings["invalid_below_offset_to_neg_inf"]
    descending = settings["descending"]
    if not isinstance(invalid_to_neg_inf, bool) or not isinstance(descending, bool):
        raise ValueError(f"comparator {name} boolean fields must be bool")
    dim = settings["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"comparator {name}.dim must be an integer")

    def compare(actual, expected, *, actual_outputs, **_):
        values = actual_outputs.get(values_output)
        if values is None or actual.shape != expected.shape:
            return False, "indexed top-k: missing score output or shape mismatch"
        dim_pos = dim if dim >= 0 else dim + actual.ndim
        if not 0 <= dim_pos < actual.ndim:
            return False, "indexed top-k: dim is out of range"
        actual_m = actual.detach().cpu().movedim(dim_pos, -1)
        expected_m = expected.detach().cpu().movedim(dim_pos, -1)
        values_m = values.detach().cpu().to(torch.float32).movedim(dim_pos, -1)
        if (
            actual_m.shape[:-1] != values_m.shape[:-1]
            or actual_m.shape[-1] < compared_columns
            or values_m.shape[-1] == 0
        ):
            return False, "indexed top-k: incompatible index/value shapes"
        actual_idx = actual_m[..., :compared_columns]
        expected_idx = expected_m[..., :compared_columns]
        mismatch = actual_idx != expected_idx
        if not bool(mismatch.any()):
            return True, ""
        gather_idx = (actual_idx.to(torch.int64) - index_offset).clamp(
            min=0, max=values_m.shape[-1] - 1
        )
        paired = torch.gather(values_m, -1, gather_idx)
        if invalid_to_neg_inf:
            paired = torch.where(
                actual_idx < index_offset,
                torch.full_like(paired, -torch.inf),
                paired,
            )
        if compared_columns < 2:
            return True, ""
        left = paired[..., :-1]
        right = paired[..., 1:]
        ordered = left >= right if descending else left <= right
        pair_ok = ordered
        position_ok = torch.ones_like(mismatch)
        position_ok[..., 1:] &= pair_ok
        position_ok[..., :-1] &= pair_ok
        failures = mismatch & ~position_ok
        if bool(failures.any()):
            return False, (
                f"indexed top-k: {int(failures.sum().item())} mismatches break "
                f"the declared {'descending' if descending else 'ascending'} order"
            )
        return True, ""

    compare.__name__ = "indexed_topk_pair"
    return compare


def _callback(name: str, settings: dict):
    if not isinstance(settings, dict):
        raise ValueError(f"comparator output {name!r} must be an object")
    method = settings.get("method")
    if method == "ratio_allclose":
        return _ratio_callback(name, settings)
    if method != "custom":
        raise ValueError(f"unsupported comparator method for {name!r}: {method!r}")
    custom_name = settings.get("name")
    if custom_name == "mapped_active_rows_ratio_allclose":
        return _mapped_rows_callback(name, settings)
    if custom_name == "indexed_topk_pair":
        return _indexed_topk_callback(name, settings)
    raise ValueError(f"unsupported custom comparator for {name!r}: {custom_name!r}")


def _compare(case_dir: Path, case: dict):
    policy = json.loads(
        (case_dir / case["comparator"]).read_text(encoding="utf-8")
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
    outputs = policy["outputs"]
    names = case["correctness"]["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != set(names):
        raise ValueError("comparator output coverage mismatch")
    return {name: _callback(name, outputs[name]) for name in names}


def _oracle_golden(oracle, values):
    scalars = {"num_tokens": values["num_tokens"]}
    tensors = {
        name: value for name, value in values.items() if isinstance(value, torch.Tensor)
    }
    for name, value in oracle.compute(tensors, scalars).items():
        values[name][:] = value


def _program(module, case: dict):
    return getattr(module, case["entry"])


def _run(**kwargs):
    return run(rtol=1e-3, atol=1e-3, save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = _identity(case)
    config = _config(case_dir, case)
    workflow = common.load_module(
        common.BENCHMARK_ROOT / "evaluator/workflow.py", f"workflow_{identity}"
    )
    return workflow.prepare_ref(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"{identity}_ref"),
        load_oracle=lambda: common.load_oracle(
            case_dir, sealed, f"{identity}_oracle"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=lambda **kwargs: _run(
            reset_input_names=case["state"]["initial_state"], **kwargs
        ),
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
    identity = _identity(case)
    config = _config(case_dir, case)
    workflow = common.load_module(
        common.BENCHMARK_ROOT / "evaluator/workflow.py", f"workflow_{identity}"
    )
    return workflow.evaluate_candidate(
        case_dir=case_dir,
        sealed=sealed,
        case=case,
        candidate_workspace=candidate_workspace,
        platform=platform,
        phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, f"{identity}_candidate"
        ),
        make_program=_program,
        make_compare=lambda: _compare(case_dir, case),
        run=lambda **kwargs: _run(
            reset_input_names=case["state"]["initial_state"], **kwargs
        ),
    )
