from __future__ import annotations

import math
from pathlib import Path

import torch
import contextlib
import contextvars
import functools
import hashlib
import inspect
import json as _psr_json
from golden import TensorSpec, run
import common

ORDER = ("x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv", "freqs_cos", "freqs_sin", "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "segment_starts_t", "predecessor_segments", "query_position_ids", "query_token_to_request", "overlay_position_ids", "overlay_token_to_request", "overlay_active_lengths", "swa_indices", "reverse_index", "owner_rank_table", "final_win_seg_src", "final_win_row_src", "final_slot_mapping", "x_out")
OUTPUTS = ("x_out", "kv_cache")

def build_tensor_specs(fixture: Path, config=None):
    specs = []
    for name in ORDER:
        path = fixture / "in" / f"{name}.pt"
        if path.exists():
            value = torch.load(path, map_location="cpu", weights_only=True)
        elif name == "x_out":
            value = torch.zeros_like(torch.load(fixture / "in" / "x_hc.pt", map_location="cpu", weights_only=True))
        else:
            raise ValueError(f"missing fixture input {name}")
        specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value if name == "kv_cache" or name not in OUTPUTS else None))
    return specs

def _number(value, name, *, upper=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"invalid {name}")
    value = float(value)
    if value < 0 or (positive and value == 0) or (upper is not None and value > upper):
        raise ValueError(f"invalid {name}")
    return value

def _callbacks(case_dir, case):
    policy = common.read_json(case_dir / case["comparator"])
    declared = case.get("correctness", {}).get("outputs")
    if (not isinstance(declared, list) or len(declared) != len(OUTPUTS)
            or any(not isinstance(name, str) for name in declared)
            or set(declared) != set(OUTPUTS)):
        raise ValueError("invalid Case correctness output coverage")
    if (not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}
            or type(policy["schema_version"]) is not int or policy["schema_version"] != 1
            or not isinstance(policy["outputs"], dict)
            or set(policy["outputs"]) != set(declared)):
        raise ValueError("invalid comparator schema or output coverage")
    callbacks = {}
    for name in OUTPUTS:
        rule = policy["outputs"][name]
        if not isinstance(rule, dict):
            raise ValueError(f"invalid comparator rule for {name}")
        if name == "x_out":
            required = {"method", "name", "diff_thd", "pct_thd", "max_diff_hd", "floor_power", "denominator_epsilon"}
            if set(rule) != required or rule.get("method") != "custom" or rule.get("name") != "ratio_reldiff": raise ValueError("invalid x_out comparator")
            diff = _number(rule["diff_thd"], "diff_thd", positive=True); pct = _number(rule["pct_thd"], "pct_thd", upper=1); cap = _number(rule["max_diff_hd"], "max_diff_hd", positive=True); floor_power = rule["floor_power"]; eps = _number(rule["denominator_epsilon"], "denominator_epsilon")
            if isinstance(floor_power, bool) or not isinstance(floor_power, int) or floor_power <= 0: raise ValueError("invalid floor_power")
            def compare(actual, expected, *, _diff=diff, _pct=pct, _cap=cap, _floor=floor_power, _eps=eps, **kwargs):
                del kwargs
                if actual.shape != expected.shape: return False, "shape mismatch"
                a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
                if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()): return False, "nonfinite value"
                absolute = (a - e).abs(); denominator = torch.maximum(torch.maximum(a.abs(), e.abs()), torch.full_like(a, (1 / (2 ** _floor)) / _diff)) + _eps
                relative = torch.where(absolute < _diff, absolute, absolute / denominator); bad = relative > _diff; count = int(bad.sum()); allowed = round(_pct * a.numel()); worst = float(relative[bad].max()) if count else 0.0
                return count <= allowed and worst < _cap, f"ratio_reldiff: mismatches={count}/{a.numel()}, allowed={allowed}, worst={worst}"
            callbacks[name] = compare
        else:
            required = {"method", "rtol", "atol", "max_error_ratio"}
            if set(rule) != required or rule.get("method") != "ratio_allclose": raise ValueError("invalid kv_cache comparator")
            rtol = _number(rule["rtol"], "rtol"); atol = _number(rule["atol"], "atol"); budget = _number(rule["max_error_ratio"], "max_error_ratio", upper=1)
            def compare(actual, expected, *, _rtol=rtol, _atol=atol, _budget=budget, **kwargs):
                del kwargs
                if actual.shape != expected.shape: return False, "shape mismatch"
                a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
                if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()): return False, "nonfinite value"
                count = int(((a - e).abs() > _atol + _rtol * e.abs()).sum()); allowed = round(_budget * a.numel())
                return count <= allowed, f"ratio_allclose: mismatches={count}/{a.numel()}, allowed={allowed}"
            callbacks[name] = compare
    return callbacks

def _program(module, case): return getattr(module, case["entry"])
def _oracle(oracle, values):
    for name, value in oracle.compute(values, {}).items(): values[name][:] = value
def _run(**kwargs): return run(rtol=0.0, atol=0.0, save_data=False, **kwargs)

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
    identity = case["case_id"].replace(".", "_"); workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", "workflow_" + identity)
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform, build_specs=build_tensor_specs, load_ref=lambda: common.load_ref(case_dir, sealed, identity + "_ref"), load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + "_oracle"), make_program=_program, make_compare=lambda: _callbacks(case_dir, case), run=_run, oracle_golden=_oracle)
def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    identity = case["case_id"].replace(".", "_"); workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", "workflow_" + identity)
    def _run_profile_aware(**kwargs):
        with _profile_state_restore(case, kwargs["specs"], kwargs["config"], common.profiling_mode()):
            return _run(**kwargs)

    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=build_tensor_specs, load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + "_candidate"), make_program=_program, make_compare=lambda: _callbacks(case_dir, case), run=_run_profile_aware)


def _compare(case_dir, case):
    return _callbacks(case_dir, case)
