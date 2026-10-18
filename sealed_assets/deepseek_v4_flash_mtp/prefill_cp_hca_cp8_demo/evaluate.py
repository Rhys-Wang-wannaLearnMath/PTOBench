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

ORDER = ("x_hc","hc_attn_fn","hc_attn_scale","hc_attn_base","attn_norm_w","wq_a","wq_b","wq_b_scale","wkv","gamma_cq","gamma_ckv","freqs_cos","freqs_sin","cmp_wkv","cmp_wgate","cmp_ape","cmp_norm_w","compress_state","compress_state_block_table","kv_cache","cmp_kv","cmp_block_table","segment_starts_t","segment_active_lengths","owner_segments_t","predecessor_segments","query_positions","query_requests","overlay_positions","overlay_requests","overlay_active_lengths","swa_indices","cmp_indices","segment_tail_positions","snapshot_positions","snapshot_valid","final_segment_t","reverse_index","owner_rank_table","owner_part_table","final_win_seg_src","final_win_row_src","final_slot_mapping","attn_sink","wo_a","wo_b","wo_b_scale","x_out")
OUTPUTS = {"compress_state", "kv_cache", "cmp_kv", "x_out"}
STATE = {"compress_state", "kv_cache", "cmp_kv"}

def _load_config(case_dir, case):
    identity = case["case_id"].replace(".", "_")
    return common.load_module(case_dir / case["config"], identity + "_evaluator_config")


def _tensor_contract(c):
    """Selected official host ABI, bound only to the owning public config."""
    fp, bf, i8, i32 = torch.float32, torch.bfloat16, torch.int8, torch.int32
    rank_tiles = (c.CP_SIZE, c.LOCAL_PARTS, c.MAX_SEGMENT_TILES)
    rows = (*rank_tiles, c.TAIL_ROWS)
    parts = (c.CP_SIZE, c.LOCAL_PARTS)
    contract = {
        "x_hc": ((*rows, c.HC_MULT, c.D), fp),
        "hc_attn_fn": ((c.MIX_HC, c.HC_DIM), fp),
        "hc_attn_scale": ((3,), fp),
        "hc_attn_base": ((c.MIX_HC,), fp),
        "attn_norm_w": ((c.D,), bf),
        "wq_a": ((c.D, c.Q_LORA), bf),
        "wq_b": ((c.Q_LORA, c.H * c.HEAD_DIM), i8),
        "wq_b_scale": ((c.H * c.HEAD_DIM,), fp),
        "wkv": ((c.D, c.HEAD_DIM), bf),
        "gamma_cq": ((c.Q_LORA,), bf),
        "gamma_ckv": ((c.HEAD_DIM,), bf),
        "freqs_cos": ((c.MAX_SEQ_LEN, c.ROPE_HEAD_DIM), bf),
        "freqs_sin": ((c.MAX_SEQ_LEN, c.ROPE_HEAD_DIM), bf),
        "cmp_wkv": ((c.HEAD_DIM, c.D), bf),
        "cmp_wgate": ((c.HEAD_DIM, c.D), bf),
        "cmp_ape": ((c.COMPRESS_RATIO, c.HEAD_DIM), fp),
        "cmp_norm_w": ((c.HEAD_DIM,), bf),
        "compress_state": ((c.CP_SIZE, c.HCA_STATE_PHYSICAL_BLOCKS, c.HCA_STATE_BLOCK_SIZE, c.COMPRESS_STATE_DIM), fp),
        "compress_state_block_table": ((c.CP_SIZE, c.HCA_STATE_MAX_BLOCKS), i32),
        "kv_cache": ((c.CP_SIZE, c.ORI_MAX_BLOCKS, c.BLOCK_SIZE, 1, c.HEAD_DIM), bf),
        "cmp_kv": ((c.CP_SIZE, c.PREFILL_CMP_BLOCK_NUM, c.BLOCK_SIZE, 1, c.HEAD_DIM), bf),
        "cmp_block_table": ((c.CP_SIZE, c.PREFILL_CMP_MAX_BLOCKS), i32),
        "segment_starts_t": ((c.NUM_SEGMENTS,), i32),
        "segment_active_lengths": (parts, i32),
        "owner_segments_t": (parts, i32),
        "predecessor_segments": (parts, i32),
        "query_positions": (rows, i32),
        "query_requests": (rows, i32),
        "overlay_positions": ((*rank_tiles, c.OVERLAY_ROWS), i32),
        "overlay_requests": ((*rank_tiles, c.OVERLAY_ROWS), i32),
        "overlay_active_lengths": ((*rank_tiles, c.OVERLAY_SOURCES), i32),
        "swa_indices": ((*rows, c.WIN), i32),
        "cmp_indices": ((*rows, c.IDX_TOPK), i32),
        "segment_tail_positions": ((c.NUM_SEGMENTS, c.TAIL_ROWS), i32),
        "snapshot_positions": ((*parts, c.TAIL_ROWS), i32),
        "snapshot_valid": (parts, i32),
        "final_segment_t": ((1,), i32),
        "reverse_index": ((c.NUM_SEGMENTS,), i32),
        "owner_rank_table": ((c.NUM_SEGMENTS,), i32),
        "owner_part_table": ((c.NUM_SEGMENTS,), i32),
        "final_win_seg_src": ((c.TAIL_ROWS,), i32),
        "final_win_row_src": ((c.TAIL_ROWS,), i32),
        "final_slot_mapping": ((c.TAIL_ROWS,), i32),
        "attn_sink": ((c.H,), fp),
        "wo_a": ((c.O_GROUPS, c.O_LORA, c.O_GROUP_IN), bf),
        "wo_b": ((c.D, c.O_GROUPS * c.O_LORA), i8),
        "wo_b_scale": ((c.D,), fp),
        "x_out": ((*rows, c.HC_MULT, c.D), fp),
    }
    if tuple(contract) != ORDER:
        raise ValueError("fixed ABI order mismatch")
    return contract


def build_tensor_specs(fixture: Path, config):
    contract = _tensor_contract(config)
    expected_inputs = set(ORDER) - {"x_out"}
    if {p.stem for p in (fixture / "in").glob("*.pt")} != expected_inputs:
        raise ValueError("Fixture input coverage differs from the fixed public ABI")
    specs = []
    for name in ORDER:
        shape, dtype = contract[name]
        value = None
        if name != "x_out":
            value = torch.load(fixture / "in" / f"{name}.pt", map_location="cpu", weights_only=True)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"Fixture {name} shape/dtype differs from the fixed public ABI: {shape}, {dtype}")
        specs.append(TensorSpec(name, list(shape), dtype, init_value=value))
    return specs

def _number(value, field, *, ratio=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)): raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0) or (ratio and value > 1): raise ValueError(f"{field} has invalid range")
    return value

def _callbacks(policy, output_names):
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"} or type(policy.get("schema_version")) is not int or policy.get("schema_version") != 1: raise ValueError("invalid comparator schema")
    if not isinstance(policy["outputs"], dict) or set(policy["outputs"]) != set(output_names): raise ValueError("comparator output coverage mismatch")
    callbacks = {}
    for name in output_names:
        rule = policy["outputs"][name]
        if not isinstance(rule, dict): raise ValueError(f"invalid comparator for {name}")
        method = rule.get("method")
        if method in ("allclose", "ratio_allclose"):
            required = {"method", "rtol", "atol"} | ({"max_error_ratio"} if method == "ratio_allclose" else set())
            if set(rule) != required: raise ValueError(f"invalid fields for {name}")
            policy_rtol, policy_atol = _number(rule["rtol"], f"{name}.rtol"), _number(rule["atol"], f"{name}.atol")
            budget = _number(rule.get("max_error_ratio", 0), f"{name}.max_error_ratio", ratio=True)
            def compare(actual, expected, *, _rtol=policy_rtol, _atol=policy_atol, _budget=budget, _method=method, **kwargs):
                del kwargs
                if actual.shape != expected.shape or actual.dtype != expected.dtype: return False, "shape/dtype mismatch"
                if _method == "allclose":
                    a, e = actual.detach().cpu(), expected.detach().cpu()
                    count = int((~torch.isclose(a, e, rtol=_rtol, atol=_atol)).sum())
                    return bool(torch.allclose(a, e, rtol=_rtol, atol=_atol)), f"allclose: mismatches={count}/{a.numel()}"
                a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
                if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()): return False, "nonfinite value"
                count = int(((a-e).abs() > _atol + _rtol*e.abs()).sum()); allowed = round(_budget*a.numel())
                return count <= allowed, f"{_method}: mismatches={count}/{a.numel()}, allowed={allowed}"
            callbacks[name] = compare
        elif method == "custom":
            required = {"method", "implementation", "diff_thd", "pct_thd", "max_diff_hd", "denominator_floor_numerator", "denominator_epsilon"}
            if set(rule) != required or rule["implementation"] != "ratio_reldiff": raise ValueError(f"invalid custom comparator for {name}")
            diff = _number(rule["diff_thd"], f"{name}.diff_thd", positive=True); pct = _number(rule["pct_thd"], f"{name}.pct_thd", ratio=True); cap = _number(rule["max_diff_hd"], f"{name}.max_diff_hd", positive=True); floor_num = _number(rule["denominator_floor_numerator"], f"{name}.denominator_floor_numerator", positive=True); epsilon = _number(rule["denominator_epsilon"], f"{name}.denominator_epsilon")
            def compare(actual, expected, *, _diff=diff, _pct=pct, _cap=cap, _floor=floor_num, _eps=epsilon, **kwargs):
                del kwargs
                if actual.shape != expected.shape or actual.dtype != expected.dtype: return False, "shape/dtype mismatch"
                a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
                if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()): return False, "nonfinite value"
                absolute = (a-e).abs(); denominator = torch.maximum(torch.maximum(a.abs(), e.abs()), torch.full_like(a, _floor/_diff)) + _eps; relative = torch.where(absolute < _diff, absolute, absolute/denominator); bad = relative > _diff; count = int(bad.sum()); worst = float(relative[bad].max()) if count else 0.0; allowed = round(_pct*a.numel())
                return count <= allowed and worst < _cap, f"ratio_reldiff: mismatches={count}/{a.numel()}, allowed={allowed}, worst={worst}"
            callbacks[name] = compare
        else: raise ValueError(f"unsupported comparator method for {name}")
    return callbacks

def _compare(case_dir, case): return _callbacks(common.read_json(case_dir / case["comparator"]), case["correctness"]["outputs"])
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
    config = _load_config(case_dir, case)
    identity = case["case_id"].replace(".", "_"); workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", "workflow_" + identity)
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform, build_specs=lambda fixture: build_tensor_specs(fixture, config), load_ref=lambda: common.load_ref(case_dir, sealed, identity+"_ref"), load_oracle=lambda: common.load_oracle(case_dir, sealed, identity+"_oracle"), make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)

def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    config = _load_config(case_dir, case)
    identity = case["case_id"].replace(".", "_"); workflow = common.load_module(common.BENCHMARK_ROOT / "evaluator" / "workflow.py", "workflow_" + identity)
    def _run_profile_aware(**kwargs):
        with _profile_state_restore(case, kwargs["specs"], kwargs["config"], common.profiling_mode()):
            return _run(**kwargs)

    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=lambda fixture: build_tensor_specs(fixture, config), load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity+"_candidate"), make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run_profile_aware)
