"""Case-local ABI, restoration, and exact/allclose execution boundary."""

import inspect
import json
import math
import os
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import common
import torch
import contextlib
import contextvars
import functools
import hashlib
import json as _psr_json
import workflow
from golden import ScalarSpec, TensorSpec, run as official_run


def comparators(policy, output_names):
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "outputs"}:
        raise ValueError("comparator top-level fields disagree")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("unsupported comparator schema")
    rules = policy["outputs"]
    if not isinstance(rules, dict) or set(rules) != set(output_names):
        raise ValueError("comparator output coverage disagrees")

    def exact(actual, expected, **kwargs):
        passed = torch.equal(actual, expected)
        return passed, "" if passed else "exact tensor comparison failed"

    def allclose(policy_rtol, policy_atol):
        def compare(actual, expected, **kwargs):
            if actual.shape != expected.shape:
                return False, "allclose shape mismatch"
            passed = torch.allclose(actual, expected, rtol=policy_rtol, atol=policy_atol)
            return passed, "" if passed else "declared allclose comparison failed"
        return compare

    callbacks = {}
    for name, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError(f"invalid rule for {name}")
        method = rule.get("method")
        if method == "exact":
            if set(rule) != {"method"}:
                raise ValueError(f"invalid exact fields for {name}")
            callbacks[name] = exact
        elif method == "allclose":
            if set(rule) != {"method", "rtol", "atol"}:
                raise ValueError(f"invalid allclose fields for {name}")
            for field in ("rtol", "atol"):
                value = rule[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {field} for {name}")
            callbacks[name] = allclose(rule["rtol"], rule["atol"])
        else:
            raise ValueError(f"unsupported comparator method for {name}")
    return callbacks


def make_compare(case_dir, case):
    with (case_dir / case["comparator"]).open(encoding="utf-8") as stream:
        policy = json.load(stream)
    return comparators(policy, case["correctness"]["outputs"])


def build_specs(case_dir, sealed, case, fixture):
    identity = case["case_id"].replace(".", "_")
    builder = common.load_module(sealed / "fixture_builder.py", identity + "_spec_builder")
    manifest = common.read_json(fixture / "manifest.json")
    policy = common.read_json(sealed / "fixture_policy.json")
    seed = policy["fixture_seeds"][fixture.name]
    if manifest["fixture_id"] != fixture.name or manifest["seed"] != seed:
        raise ValueError("fixture identity or seed disagrees")
    specs = builder.build(case_dir, sealed, case, seed)
    pure_outputs = set(case["correctness"]["outputs"]) - set(case["state"]["initial_state"])

    def load_input(path, shape, dtype):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if list(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"fixture ABI mismatch: {path.name}")
        return value.contiguous()

    for spec in specs:
        path = fixture / "in" / f"{spec.name}.pt"
        if isinstance(spec, ScalarSpec):
            value = torch.load(path, map_location="cpu", weights_only=True)
            if value.numel() != 1 or value.dtype != spec.dtype or int(value) != int(spec.value):
                raise ValueError(f"fixture scalar disagrees: {spec.name}")
            spec.value = value
        elif spec.name not in pure_outputs:
            spec.init_value = lambda path=path, shape=spec.shape, dtype=spec.dtype: load_input(path, shape, dtype)
    return specs


class ResetWorker:
    def __init__(self, worker, names, snapshots, evidence):
        self.worker = worker
        self.names = names
        self.snapshots = snapshots
        self.evidence = evidence

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def __call__(self, *args, **kwargs):
        for name, value in zip(self.names, args):
            if name not in self.snapshots:
                continue
            initial = self.snapshots[name]
            if isinstance(value, torch.Tensor):
                value.copy_(initial)
                self.evidence["host_restores"] += 1
            else:
                if len(value.shards) != initial.shape[0]:
                    raise ValueError(f"resident shard count disagrees: {name}")
                for rank, (shard, worker_id) in enumerate(zip(value.shards, value.worker_ids)):
                    row = initial[rank]
                    self.worker.copy_to(
                        shard.data_ptr, row.data_ptr(), row.numel() * row.element_size(),
                        worker_id=worker_id,
                    )
                    self.evidence["shard_restores"] += 1
        self.evidence["launches"] += 1
        return self.worker(*args, **kwargs)


class ResetEntry:
    def __init__(self, entry, names, snapshots, evidence):
        self.entry = entry
        self.names = names
        self.snapshots = snapshots
        self.evidence = evidence

    def __getattr__(self, name):
        return getattr(self.entry, name)

    def compile(self, *args, **kwargs):
        compiled = copy(self.entry.compile(*args, **kwargs))
        original_prepare = compiled.prepare

        @contextmanager
        def prepare(*prepare_args, **prepare_kwargs):
            with original_prepare(*prepare_args, **prepare_kwargs) as worker:
                yield ResetWorker(worker, self.names, self.snapshots, self.evidence)

        compiled.prepare = prepare
        return compiled


@contextmanager
def _case_reset_scope():
    key = "PYPTO_BENCH_RESET_INPUTS"
    previous = os.environ.get(key)
    os.environ[key] = "none"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def run_restored(case, **kwargs):
    entry = kwargs.pop("fn")
    specs = kwargs["specs"]
    mutable = set(case["state"]["initial_state"])
    snapshots = {
        spec.name: spec.create_tensor().clone().contiguous()
        for spec in specs if isinstance(spec, TensorSpec) and spec.name in mutable
    }
    if set(snapshots) != mutable:
        raise ValueError("incomplete initial mutable state")
    names = list(inspect.signature(entry._func).parameters)
    if names != [spec.name for spec in specs]:
        raise ValueError("spec order disagrees with entry ABI")
    evidence = {"launches": 0, "host_restores": 0, "shard_restores": 0}
    kwargs["fn"] = ResetEntry(entry, names, snapshots, evidence)
    with _case_reset_scope():
        result = official_run(**kwargs, save_data=False)
    print("CASE_RESTORATION", case["case_id"], evidence, flush=True)
    return result


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
    identity = case["case_id"].replace(".", "_")
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(case_dir, sealed, case, fixture),
        load_ref=lambda: common.load_ref(case_dir, sealed, identity + "_ref"),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + "_oracle"),
        make_program=lambda module, metadata: getattr(module, metadata["entry"]),
        make_compare=lambda: make_compare(case_dir, case),
        run=lambda **kwargs: run_restored(case, **kwargs),
        oracle_golden=lambda oracle, values: values.update(oracle.compute(values)),
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase="all"):
    identity = case["case_id"].replace(".", "_")
    def _run_profile_aware(**kwargs):
        with _profile_state_restore(case, kwargs["specs"], kwargs["config"], common.profiling_mode()):
            return run_restored(case, **kwargs)

    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=Path(candidate_workspace), platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(case_dir, sealed, case, fixture),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, Path(candidate_workspace), identity + "_candidate",
        ),
        make_program=lambda module, metadata: getattr(module, metadata["entry"]),
        make_compare=lambda: make_compare(case_dir, case),
        run=_run_profile_aware,
    )
