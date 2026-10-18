"""Case-local evaluator for DeepSeek-V4 FLASH packed prefill HCA EP4."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import contextlib
import contextvars
import functools
import hashlib
import inspect
import json as _psr_json
from golden import ScalarSpec, TensorSpec, run

import common


TENSOR_ORDER = (
    "x_hc", "seq_lens", "chunk_lens", "chunk_offsets", "chunk_tile_offsets",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w", "wq_a",
    "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv", "freqs_cos",
    "freqs_sin", "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape",
    "hca_cmp_norm_w", "hca_cmp_kv_state", "hca_cmp_score_state",
    "hca_compress_state_block_table", "csa_cmp_wkv", "csa_cmp_wgate",
    "csa_cmp_ape", "csa_cmp_norm_w", "csa_cmp_kv_state",
    "csa_cmp_score_state", "csa_compress_state_block_table", "csa_hadamard_idx",
    "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj", "csa_inner_wkv",
    "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_kv_state", "csa_inner_score_state",
    "csa_inner_compress_state_block_table", "kv_cache", "ori_block_table",
    "ori_slot_mapping", "cmp_kv", "cmp_block_table", "cmp_sparse_indices",
    "cmp_sparse_lens", "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "position_ids", "hca_cmp_slot_mapping", "hca_state_slot_mapping",
    "csa_cmp_slot_mapping", "csa_idx_slot_mapping", "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w", "gate_w",
    "gate_bias", "tid2eid", "input_ids", "routed_w1", "routed_w1_scale",
    "routed_w3", "routed_w3_scale", "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale", "x_next",
)
PURE_OUTPUTS = {"x_next"}
RESIDENT = {
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w", "wq_a",
    "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv", "freqs_cos",
    "freqs_sin", "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape",
    "hca_cmp_norm_w", "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape",
    "csa_cmp_norm_w", "csa_hadamard_idx", "csa_idx_wq_b",
    "csa_idx_wq_b_scale", "csa_weights_proj", "csa_inner_wkv",
    "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w", "attn_sink",
    "wo_a", "wo_b", "wo_b_scale", "hc_ffn_fn", "hc_ffn_scale",
    "hc_ffn_base", "norm_w", "gate_w", "gate_bias", "tid2eid", "routed_w1", "routed_w1_scale", "routed_w3",
    "routed_w3_scale", "routed_w2", "routed_w2_scale", "shared_w1",
    "shared_w1_scale", "shared_w3", "shared_w3_scale", "shared_w2",
    "shared_w2_scale",
}


def _identity(case: dict) -> str:
    return case["case_id"].replace(".", "_")


def _config(case_dir: Path, case: dict):
    return common.load_module(case_dir / case["config"], f"{_identity(case)}_config")


def _fixture_values(fixture: Path) -> dict[str, torch.Tensor]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=True)
        for path in (fixture / "in").glob("*.pt")
    }


def build_tensor_specs(fixture: Path, config):
    values = _fixture_values(fixture)
    expected = set(TENSOR_ORDER) - PURE_OUTPUTS | {"layer_id"}
    if set(values) != expected:
        raise ValueError(f"fixture input coverage mismatch: {sorted(values)} != {sorted(expected)}")
    specs = []
    for name in TENSOR_ORDER:
        if name == "x_next":
            specs.append(TensorSpec(name, [config.EP_WORLD_SIZE, config.PADDED_TOKENS,
                                           config.HC_MULT, config.D], torch.bfloat16))
            continue
        value = values[name]
        spec = TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        if name in RESIDENT:
            spec.resident = "stacked"
        specs.append(spec)
    layer_id = values["layer_id"]
    if layer_id.ndim != 0 or layer_id.dtype != torch.int32:
        raise ValueError("layer_id fixture must be a scalar int32 tensor")
    specs.append(ScalarSpec("layer_id", torch.int32, layer_id))
    return specs


def _number(value, field: str, *, ratio: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparator {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (ratio and number > 1):
        raise ValueError(f"comparator {field} has invalid range")
    return number


def _allclose_callback(name: str, settings: dict):
    required = {"method", "rtol", "atol"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f"allclose comparator {name!r} requires exactly {sorted(required)}")
    policy_rtol = _number(settings["rtol"], f"{name}.rtol")
    policy_atol = _number(settings["atol"], f"{name}.atol")

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, "allclose: shape mismatch"
        actual_f = actual.detach().cpu().float()
        expected_f = expected.detach().cpu().float()
        if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
            return False, "allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > (policy_atol + policy_rtol * expected_f.abs())
        count = int(bad.sum().item())
        return count == 0, f"allclose: mismatches={count}/{actual_f.numel()}"

    compare.__name__ = "allclose"
    return compare


def _ratio_callback(name: str, settings: dict):
    required = {"method", "rtol", "atol", "max_error_ratio"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f"ratio_allclose comparator {name!r} requires exactly {sorted(required)}")
    policy_rtol = _number(settings["rtol"], f"{name}.rtol")
    policy_atol = _number(settings["atol"], f"{name}.atol")
    max_error_ratio = _number(settings["max_error_ratio"], f"{name}.max_error_ratio", ratio=True)

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, "ratio_allclose: shape mismatch"
        actual_f = actual.detach().cpu().float()
        expected_f = expected.detach().cpu().float()
        if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
            return False, "ratio_allclose: non-finite value"
        bad = (actual_f - expected_f).abs() > (policy_atol + policy_rtol * expected_f.abs())
        count = int(bad.sum().item())
        allowed = round(max_error_ratio * actual_f.numel())
        return count <= allowed, f"ratio_allclose: mismatches={count}/{actual_f.numel()}, allowed={allowed}"

    compare.__name__ = "ratio_allclose"
    return compare


def _valid_ratio_callback(name: str, settings: dict, config):
    required = {"method", "name", "diff_thd", "pct_thd", "chunk_lens_input", "chunk_offsets_input"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f"custom comparator {name!r} requires exactly {sorted(required)}")
    if settings["name"] != "valid_ratio_reldiff":
        raise ValueError(f"unsupported custom comparator for {name!r}")
    diff_thd = _number(settings["diff_thd"], f"{name}.diff_thd")
    if diff_thd == 0:
        raise ValueError("custom comparator diff_thd must be positive")
    pct_thd = _number(settings["pct_thd"], f"{name}.pct_thd", ratio=True)
    chunk_lens_input = settings["chunk_lens_input"]
    chunk_offsets_input = settings["chunk_offsets_input"]
    if chunk_lens_input != "chunk_lens" or chunk_offsets_input != "chunk_offsets":
        raise ValueError("custom comparator input bindings do not match the fixed ABI")

    output_shape = (config.EP_WORLD_SIZE, config.PADDED_TOKENS, config.HC_MULT, config.D)
    metadata_shape = (config.EP_WORLD_SIZE, config.USER_BATCH)

    def compare(actual, expected, *, inputs=None, **_):
        if actual.shape != expected.shape or tuple(actual.shape) != output_shape:
            return False, "valid_ratio_reldiff: shape mismatch"
        if not isinstance(inputs, dict):
            return False, "valid_ratio_reldiff: missing chunk metadata"
        chunk_lens = inputs.get(chunk_lens_input)
        chunk_offsets = inputs.get(chunk_offsets_input)
        if (not isinstance(chunk_lens, torch.Tensor) or not isinstance(chunk_offsets, torch.Tensor)
                or tuple(chunk_lens.shape) != metadata_shape or tuple(chunk_offsets.shape) != metadata_shape
                or chunk_lens.dtype != torch.int32 or chunk_offsets.dtype != torch.int32):
            return False, "valid_ratio_reldiff: invalid chunk metadata"
        if (not torch.equal(chunk_lens, chunk_lens[0].expand_as(chunk_lens))
                or not torch.equal(chunk_offsets, chunk_offsets[0].expand_as(chunk_offsets))):
            return False, "valid_ratio_reldiff: rank metadata mismatch"
        mask = torch.zeros(actual.shape[1], dtype=torch.bool)
        for length, offset in zip(chunk_lens[0].tolist(), chunk_offsets[0].tolist()):
            if length <= 0 or offset < 0 or offset + length > actual.shape[1]:
                return False, "valid_ratio_reldiff: chunk interval out of range"
            if bool(mask[offset:offset + length].any()):
                return False, "valid_ratio_reldiff: overlapping chunk intervals"
            mask[offset:offset + length] = True
        actual_f = actual.detach().cpu().float()[:, mask]
        expected_f = expected.detach().cpu().float()[:, mask]
        if not bool(torch.isfinite(actual_f).all()) or not bool(torch.isfinite(expected_f).all()):
            return False, "valid_ratio_reldiff: non-finite selected value"
        absolute = (actual_f - expected_f).abs()
        floor = (1.0 / (1 << 14)) / diff_thd
        denominator = torch.maximum(torch.maximum(actual_f.abs(), expected_f.abs()),
                                    torch.full_like(actual_f, floor)) + 1e-9
        relative = torch.where(absolute < diff_thd, absolute, absolute / denominator)
        count = int((relative > diff_thd).sum().item())
        allowed = round(pct_thd * actual_f.numel())
        return count <= allowed, f"valid_ratio_reldiff: mismatches={count}/{actual_f.numel()}, allowed={allowed}"

    compare.__name__ = "valid_ratio_reldiff"
    return compare


def _callbacks(policy: dict, output_names: list[str], config):
    if (not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}
            or type(policy.get("schema_version")) is not int or policy["schema_version"] != 1):
        raise ValueError("comparator.json must contain only schema_version=1 and outputs")
    outputs = policy.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(output_names):
        raise ValueError("comparator outputs must exactly match case outputs")
    callbacks = {}
    for name in output_names:
        settings = outputs[name]
        method = settings.get("method") if isinstance(settings, dict) else None
        if method == "allclose":
            callbacks[name] = _allclose_callback(name, settings)
        elif method == "ratio_allclose":
            callbacks[name] = _ratio_callback(name, settings)
        elif method == "custom" and name == "x_next":
            callbacks[name] = _valid_ratio_callback(name, settings, config)
        else:
            raise ValueError(f"unsupported comparator method for {name!r}: {method!r}")
    return callbacks


def _compare(case_dir: Path, case: dict):
    policy = json.loads((case_dir / case["comparator"]).read_text(encoding="utf-8"))
    return _callbacks(policy, list(case["correctness"]["outputs"]), _config(case_dir, case))


def _program(module, case):
    return getattr(module, case["entry"])


def _oracle(oracle, values):
    inputs = {name: values[name] for name in TENSOR_ORDER if name != "x_next"}
    outputs = oracle.compute(inputs, {"layer_id": values["layer_id"]})
    for name, value in outputs.items():
        values[name][:] = value


def _run(*, fn, specs, config, compare_fn, golden_data=None, golden_fn=None, capture_outputs=False):
    return run(fn=fn, specs=specs, config=config, golden_fn=golden_fn,
               golden_data=str(golden_data) if golden_data else None,
               compare_fn=compare_fn, rtol=0.0, atol=0.0, save_data=False,
               capture_outputs=capture_outputs)



# Case-local runtime profiling state isolation.  Native L3 swimlane profiling
# intentionally executes graph and timing passes; this adapter restores the
# declared single-step mutable state before each pass without changing the
# Ref/Candidate algorithm or the public result format.
@contextlib.contextmanager
def _profile_state_restore(case, specs, config, mode):
    if (mode != "runtime" or case.get("state", {}).get("behavior") != "mutating"
            or not getattr(config, "enable_chip_swimlane", 0)
            or str(getattr(config, "platform", "")).endswith("sim")):
        yield
        return
    names = case.get("state", {}).get("initial_state", [])
    if (case.get("state", {}).get("restore_before_each_launch") is not True
            or not names or len(names) != len(set(names))):
        raise ValueError("distributed runtime profiling requires a complete mutable-state contract")
    initial = {}
    for spec in specs:
        if spec.name not in names:
            continue
        value = getattr(spec, "init_value", None)
        if callable(value):
            value = value()
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or tuple(value.shape) != tuple(spec.shape) or value.dtype != spec.dtype):
            raise ValueError(f"state {spec.name!r} needs an exact CPU Fixture tensor")
        initial[spec.name] = value.detach().contiguous().clone()
    if set(initial) != set(names):
        raise ValueError("mutable-state contract is absent from the evaluator ABI")
    hashes = {name: hashlib.sha256(
        initial[name].detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest() for name in names}
    from pypto.runtime import distributed_runner as _psr_runtime
    from pypto.runtime.device_tensor import DeviceTensor as _PSRDeviceTensor
    from pypto.runtime.device_tensor import StackedDeviceTensor as _PSRStackedDeviceTensor
    _psr_worker_type = _psr_runtime.DistributedWorker
    _psr_original_execute = _psr_runtime._execute_distributed
    _psr_original_submit = _psr_worker_type._submit_compiled
    _psr_original_two_pass = _psr_runtime._run_l3_swimlane_two_pass
    _psr_active = contextvars.ContextVar("ptobench_profile_state_restore", default=None)
    _psr_passes = 0

    def _psr_base_name(name):
        base, marker, version = str(name).rpartition("__ssa_v")
        return base if marker and version.isdigit() else str(name)

    def _psr_copy_device(worker, handle, value, worker_id):
        if (tuple(handle.shape) != tuple(value.shape) or handle.dtype != value.dtype
                or worker is None):
            raise ValueError("resident profiling state has no matching Worker ABI")
        worker.copy_to(handle.data_ptr, value.data_ptr(), handle.nbytes, worker_id=worker_id)
        check = torch.empty_like(value)
        worker.copy_from(check.data_ptr(), handle.data_ptr, handle.nbytes, worker_id=worker_id)
        if check.detach().contiguous().view(torch.uint8).numpy().tobytes() != value.detach().contiguous().view(torch.uint8).numpy().tobytes():
            raise RuntimeError("device profiling state restoration verification failed")

    def _psr_make_restore(worker, compiled, args):
        infos = compiled._get_metadata()[0]
        if len(infos) != len(args):
            raise ValueError("distributed profiling argument count mismatch")
        bindings = {}
        for info, value in zip(infos, args, strict=True):
            key = _psr_base_name(info.name)
            if key in bindings:
                raise ValueError(f"distributed profiling ABI has duplicate state name {key!r}")
            bindings[key] = value
        if not set(names) <= bindings.keys():
            raise ValueError("distributed profiling state is absent from compiled ABI")
        def restore():
            for name in names:
                arg, value = bindings[name], initial[name]
                if isinstance(arg, torch.Tensor):
                    if (tuple(arg.shape) != tuple(value.shape) or arg.dtype != value.dtype):
                        raise ValueError(f"host profiling state {name!r} ABI mismatch")
                    with torch.no_grad():
                        arg.copy_(value)
                    actual = arg.detach().contiguous().view(torch.uint8).numpy().tobytes()
                    expected = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
                    if actual != expected:
                        raise RuntimeError(f"host profiling state {name!r} restoration verification failed")
                elif isinstance(arg, _PSRStackedDeviceTensor):
                    if tuple(arg.full_shape) != tuple(value.shape) or arg.dtype != value.dtype:
                        raise ValueError(f"stacked profiling state {name!r} ABI mismatch")
                    for rank, (shard, worker_id) in enumerate(zip(arg.shards, arg.worker_ids, strict=True)):
                        _psr_copy_device(worker, shard, value[rank].contiguous(), worker_id)
                elif isinstance(arg, _PSRDeviceTensor):
                    owners = [wid for (wid, ptr), buf in worker._device_buffers.items()
                              if ptr == arg.data_ptr and buf is arg.buffer]
                    if len(owners) != 1:
                        raise ValueError(f"cannot resolve resident state owner for {name!r}")
                    _psr_copy_device(worker, arg, value, owners[0])
                else:
                    raise TypeError(f"unsupported distributed state storage for {name!r}")
        return restore

    def _psr_two_pass(dc, cfg, dfx_base, run_pass):
        restore = _psr_active.get()
        if restore is None:
            return _psr_original_two_pass(dc, cfg, dfx_base, run_pass)
        local = 0
        def pass_with_restore(pass_cfg):
            nonlocal local, _psr_passes
            if local >= 2:
                raise RuntimeError("unexpected extra native profiling pass")
            restore()
            local += 1
            _psr_passes += 1
            print(_psr_json.dumps({"ptobench_profile_state_restore": {
                "case_id": case["case_id"], "pass_index": local,
                "initial_state_sha256": hashes, "verified": True,
            }}, sort_keys=True), flush=True)
            return run_pass(pass_cfg)
        result = _psr_original_two_pass(dc, cfg, dfx_base, pass_with_restore)
        if local != 2:
            raise RuntimeError("expected exactly native graph and timing profiling passes")
        return result

    @functools.wraps(_psr_original_execute)
    def _psr_execute(compiled, coerced_args, config=None):
        if (config is None or not getattr(config, "enable_chip_swimlane", 0)
                or compiled.platform.endswith("sim")):
            return _psr_original_execute(compiled, coerced_args, config)
        token = _psr_active.set(_psr_make_restore(None, compiled, coerced_args))
        try:
            return _psr_original_execute(compiled, coerced_args, config)
        finally:
            _psr_active.reset(token)

    @functools.wraps(_psr_original_submit)
    def _psr_submit(self, compiled, *args, config=None):
        if (compiled is not self._compiled or config is None
                or not getattr(config, "enable_chip_swimlane", 0)
                or compiled.platform.endswith("sim")):
            return _psr_original_submit(self, compiled, *args, config=config)
        token = _psr_active.set(_psr_make_restore(self, compiled, args))
        try:
            return _psr_original_submit(self, compiled, *args, config=config)
        finally:
            _psr_active.reset(token)

    _psr_runtime._run_l3_swimlane_two_pass = _psr_two_pass
    _psr_runtime._execute_distributed = _psr_execute
    _psr_worker_type._submit_compiled = _psr_submit
    try:
        yield
        if _psr_passes != 2:
            raise RuntimeError("target profiling did not execute exactly two native passes")
    finally:
        _psr_runtime._run_l3_swimlane_two_pass = _psr_original_two_pass
        _psr_runtime._execute_distributed = _psr_original_execute
        _psr_worker_type._submit_compiled = _psr_original_submit

def prepare_ref(case_dir, sealed, case, platform=None):
    identity = _identity(case)
    workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", f"workflow_{identity}")
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f"{identity}_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f"{identity}_oracle"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    identity = _identity(case)
    workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", f"workflow_{identity}")
    config = _config(case_dir, case)
    def _run_profile_aware(**kwargs):
        with _profile_state_restore(case, kwargs["specs"], kwargs["config"], common.profiling_mode()):
            return _run(**kwargs)

    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f"{identity}_candidate"),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run_profile_aware,
    )
