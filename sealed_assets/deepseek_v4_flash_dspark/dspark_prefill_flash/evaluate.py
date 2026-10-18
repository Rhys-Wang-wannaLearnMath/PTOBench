"""Case-local ABI and comparator boundary for the fixed FLASH prefill Case."""
from pathlib import Path
import json
import math

import torch
from golden import TensorSpec, run

import common
import workflow


def make_compare(case_dir, case):
    policy = json.loads((Path(case_dir) / case['comparator']).read_text())
    if not isinstance(policy, dict) or set(policy) != {'schema_version', 'outputs'}:
        raise ValueError('Comparator must contain exactly schema_version and outputs')
    if type(policy['schema_version']) is not int or policy['schema_version'] != 1:
        raise ValueError('Unsupported comparator schema_version')
    rules = policy['outputs']
    names = case['correctness']['outputs']
    if not isinstance(rules, dict) or set(rules) != set(names):
        raise ValueError('Comparator output coverage does not match correctness.outputs')

    def number(rule, field, upper=None):
        value = rule[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid comparator {field}')
        if upper is not None and value > upper:
            raise ValueError(f'Invalid comparator {field}')
        return value

    callbacks = {}
    for name in names:
        rule = rules[name]
        if not isinstance(rule, dict):
            raise ValueError('Comparator rule must be an object')
        method = rule.get('method')
        fields = {'allclose': {'method', 'rtol', 'atol'},
                  'ratio_allclose': {'method', 'rtol', 'atol', 'max_error_ratio'}}
        if not isinstance(method, str) or method not in fields:
            raise ValueError('Unsupported comparator method')
        if set(rule) != fields[method]:
            raise ValueError('Comparator method fields do not match selected method')
        policy_rtol = number(rule, 'rtol')
        policy_atol = number(rule, 'atol')
        if method == 'allclose':
            def callback(actual, expected, *, policy_rtol=policy_rtol,
                         policy_atol=policy_atol, **kwargs):
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    return False, 'Output shape or dtype mismatch'
                passed = torch.allclose(actual, expected, rtol=policy_rtol, atol=policy_atol)
                return bool(passed), f'allclose rtol={policy_rtol} atol={policy_atol}'
        else:
            policy_ratio = number(rule, 'max_error_ratio', 1)

            def callback(actual, expected, *, policy_rtol=policy_rtol,
                         policy_atol=policy_atol, policy_ratio=policy_ratio, **kwargs):
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    return False, 'Output shape or dtype mismatch'
                actual, expected = actual.float(), expected.float()
                if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                    return False, 'Non-finite output'
                bad = int(((actual - expected).abs() >
                           policy_atol + policy_rtol * expected.abs()).sum().item())
                allowed = round(policy_ratio * actual.numel())
                return bad <= allowed, f'ratio_allclose mismatches={bad} allowed={allowed}'
        callback.__name__ = method
        callbacks[name] = callback
    return callbacks


def build_specs(case_dir, sealed, case, fixture):
    """Use official ABI metadata and bind every input to its persisted Fixture."""
    identity = case['case_id'].replace('.', '_')
    builder = common.load_module(Path(sealed) / 'fixture_builder.py', identity + '__spec_builder')
    seed = common.read_json(Path(sealed) / 'fixture_policy.json')['fixture_seeds'][fixture.name]
    metadata = builder.build(case_dir, sealed, case, seed)
    outputs = set(case['correctness']['outputs'])
    state = set(case['state']['initial_state'])
    specs = []
    for spec in metadata:
        if spec.name in outputs - state:
            initializer = 0
        else:
            path = fixture / 'in' / (spec.name + '.pt')

            def initializer(path=path, shape=tuple(spec.shape), dtype=spec.dtype):
                value = torch.load(path, map_location='cpu', weights_only=True)
                if list(value.shape) != list(shape) or value.dtype != dtype:
                    raise ValueError(f'Fixture shape/dtype mismatch: {path.name}')
                return value
        specs.append(TensorSpec(spec.name, list(spec.shape), spec.dtype,
                                init_value=initializer, resident=spec.resident))
    return specs


def make_program(module, case):
    return getattr(module, case['entry'])


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case['case_id'].replace('.', '_')
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(case_dir, sealed, case, fixture),
        load_ref=lambda: common.load_ref(case_dir, sealed, identity + '__ref'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + '__oracle'),
        make_program=make_program,
        make_compare=lambda: make_compare(case_dir, case), run=run,
        oracle_golden=lambda oracle, values: oracle.golden_dspark_drafter(values),
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    def run_profile_aware(**kwargs):
        with _distributed_profile_state(case, kwargs["specs"], kwargs["config"], common.profiling_mode()) as outcome:
            result = run(**kwargs)
            if outcome is not None:
                outcome["golden_failed"] = not getattr(result, "passed", True)
            return result

    identity = case['case_id'].replace('.', '_')
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(case_dir, sealed, case, fixture),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace,
                                                     identity + '__candidate'),
        make_program=make_program,
        make_compare=lambda: make_compare(case_dir, case), run=run_profile_aware,
    )


"""Sealed evaluator support for single-step native L3 profiling passes.

This source is embedded into each evaluator; production never imports mig_data.
"""
import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import re

import torch


def _profile_state_hash(value):
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


@contextlib.contextmanager
def _distributed_profile_state(case, specs, config, mode):
    if (mode != "runtime"
            or case.get("state", {}).get("behavior") != "mutating"
            or case.get("correctness", {}).get("mode") != "single_step"
            or not getattr(config, "enable_chip_swimlane", 0)
            or str(getattr(config, "platform", "")).endswith("sim")):
        yield
        return
    dc = getattr(config, "distributed_config", None)
    if dc is None or len(dc.device_ids) < 2:
        yield
        return
    if case["state"].get("restore_before_each_launch") is not True:
        raise ValueError("Distributed profiling requires initial-state restoration")
    names = case["state"].get("initial_state", [])
    if not names or any(not isinstance(n, str) for n in names) or len(names) != len(set(names)):
        raise ValueError("Distributed profiling requires complete unique state names")
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError("Duplicate profiling TensorSpec names")
    initial = {}
    for name in names:
        spec = by_name.get(name)
        value = getattr(spec, "init_value", None)
        # Existing Case builders use deterministic Fixture-backed factories.
        value = value() if callable(value) else value
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or tuple(value.shape) != tuple(spec.shape) or value.dtype != spec.dtype):
            raise ValueError(f"State {name!r} needs an exact CPU Fixture tensor")
        initial[name] = value.detach().contiguous().clone()
    baseline_hashes = {name: _profile_state_hash(value) for name, value in initial.items()}

    from pypto.runtime import distributed_runner as runtime
    from pypto.runtime.device_tensor import DeviceTensor, StackedDeviceTensor

    worker_type = runtime.DistributedWorker
    original_submit = worker_type._submit_compiled
    original_execute = runtime._execute_distributed
    original_two_pass = runtime._run_l3_swimlane_two_pass
    for function, expected in [
        (original_two_pass, ["dc", "config", "dfx_base", "run_pass"]),
        (original_execute, ["compiled", "coerced_args", "config"]),
        (original_submit, ["self", "compiled", "args", "config"]),
    ]:
        if list(inspect.signature(function).parameters) != expected:
            raise RuntimeError("Unsupported native distributed profiling interface")
    active_reset = contextvars.ContextVar("ptobench_case_profile_state", default=None)
    pass_count = 0

    def eligible(compiled, run_config):
        return (run_config is config and bool(run_config.enable_chip_swimlane)
                and not compiled.platform.endswith("sim"))

    def make_reset(worker, compiled, args, path):
        infos = compiled._get_metadata()[0]
        param_names = [re.sub(r"__ssa_v\d+$", "", info.name) for info in infos]
        if len(infos) != len(args) or len(set(param_names)) != len(infos):
            raise ValueError("Distributed profiling argument binding mismatch")
        bindings = dict(zip(param_names, args, strict=True))
        if not set(names) <= bindings.keys():
            raise ValueError("Declared state absent from compiled ABI")
        physical_ids = list(compiled._distributed_config.device_ids)
        if physical_ids != list(dc.device_ids):
            raise ValueError("Distributed profiling device binding drift")
        addresses = {}

        def device_copy(name, shard, value, worker_id, rank):
            if worker is None or tuple(shard.shape) != tuple(value.shape) or shard.dtype != value.dtype:
                raise ValueError("Resident profiling state shape/dtype mismatch")
            key = (name, rank)
            address = (int(worker_id), int(shard.data_ptr))
            if addresses.setdefault(key, address) != address:
                raise RuntimeError("Resident profiling state address changed")
            worker.copy_to(shard.data_ptr, value.data_ptr(), shard.nbytes, worker_id=worker_id)
            actual = torch.empty_like(value)
            worker.copy_from(actual.data_ptr(), shard.data_ptr, shard.nbytes, worker_id=worker_id)
            expected_hash = _profile_state_hash(value)
            actual_hash = _profile_state_hash(actual)
            if actual_hash != expected_hash:
                raise RuntimeError(f"Device profiling state {name!r} was not restored")
            return dict(rank=rank, worker_id=int(worker_id), physical_device=physical_ids[worker_id],
                        storage="device", address=address[1], address_unchanged=True,
                        expected_sha256=expected_hash, actual_sha256=actual_hash, verified=True)

        def restore():
            evidence = {}
            for name in names:
                arg, value = bindings[name], initial[name]
                if isinstance(arg, torch.Tensor):
                    if (arg.device.type != "cpu" or not arg.is_shared() or not arg.is_contiguous()
                            or tuple(arg.shape) != tuple(value.shape) or arg.dtype != value.dtype):
                        raise ValueError(f"State {name!r} needs matching shared CPU storage")
                    ptr = arg.data_ptr()
                    if addresses.setdefault((name, "host"), ptr) != ptr:
                        raise RuntimeError("Shared profiling state address changed")
                    with torch.no_grad():
                        arg.copy_(value)
                    actual_hash = _profile_state_hash(arg)
                    if actual_hash != baseline_hashes[name]:
                        raise RuntimeError(f"Host profiling state {name!r} was not restored")
                    if value.ndim == 0 or value.shape[0] != len(physical_ids):
                        raise ValueError("Distributed host state lacks the declared rank axis")
                    evidence[name] = [dict(rank=rank, worker_id=rank, physical_device=device,
                        storage="shared_cpu", address=arg[rank].data_ptr(), address_unchanged=True,
                        expected_sha256=_profile_state_hash(value[rank]),
                        actual_sha256=_profile_state_hash(arg[rank]), verified=True)
                        for rank, device in enumerate(physical_ids)]
                elif isinstance(arg, StackedDeviceTensor):
                    if (tuple(arg.full_shape) != tuple(value.shape) or arg.dtype != value.dtype
                            or len(arg.shards) != len(physical_ids)
                            or sorted(arg.worker_ids) != list(range(len(physical_ids)))):
                        raise ValueError(f"Stacked profiling state {name!r} rank/ABI mismatch")
                    # Leading-axis order follows the handle's shard/worker_ids mapping.
                    evidence[name] = [device_copy(name, shard, value[rank].contiguous(), wid, rank)
                        for rank, (shard, wid) in enumerate(zip(arg.shards, arg.worker_ids, strict=True))]
                elif isinstance(arg, DeviceTensor):
                    owners = [wid for (wid, ptr), buffer in worker._device_buffers.items()
                              if ptr == arg.data_ptr and buffer is arg.buffer]
                    if len(owners) != 1 or not 0 <= owners[0] < len(physical_ids):
                        raise ValueError(f"Cannot resolve resident state owner for {name!r}")
                    evidence[name] = [device_copy(name, arg, value, owners[0], owners[0])]
                else:
                    raise TypeError(f"Unsupported distributed state storage for {name!r}")
            return dict(path=path, states=evidence)
        return restore

    @functools.wraps(original_submit)
    def submit(self, compiled, *args, config=None):
        if not eligible(compiled, config):
            return original_submit(self, compiled, *args, config=config)
        token = active_reset.set(make_reset(self, compiled, args, "prepared"))
        try:
            return original_submit(self, compiled, *args, config=config)
        finally:
            active_reset.reset(token)

    @functools.wraps(original_execute)
    def execute(compiled, coerced_args, config=None):
        if not eligible(compiled, config):
            return original_execute(compiled, coerced_args, config)
        # Native code shares these same objects before invoking two_pass; bind
        # them now, but record/check their final shared addresses at the pass boundary.
        token = active_reset.set(make_reset(None, compiled, coerced_args, "one_shot"))
        try:
            return original_execute(compiled, coerced_args, config)
        finally:
            active_reset.reset(token)

    @functools.wraps(original_two_pass)
    def two_pass(dc, config, dfx_base, run_pass):
        reset = active_reset.get()
        if reset is None:
            return original_two_pass(dc, config, dfx_base, run_pass)
        local_pass_count = 0

        def restored_pass(call_config):
            nonlocal local_pass_count, pass_count
            if local_pass_count >= 2:
                raise RuntimeError("Unexpected extra native profiling pass")
            evidence = reset()
            local_pass_count += 1
            pass_count += 1
            print(json.dumps({"ptobench_profile_state_restore": dict(
                case_id=case["case_id"], pass_index=local_pass_count,
                initial_state_sha256=baseline_hashes, verified=True, **evidence)},
                sort_keys=True), flush=True)
            # Native call blocks until every rank completes. Do not restore
            # after the timing pass: the unchanged comparator needs its real output.
            return run_pass(call_config)

        result = original_two_pass(dc, config, dfx_base, restored_pass)
        if local_pass_count != 2:
            raise RuntimeError("Expected native graph and timing profiling passes")
        return result

    worker_type._submit_compiled = submit
    runtime._execute_distributed = execute
    runtime._run_l3_swimlane_two_pass = two_pass
    try:
        outcome = {"golden_failed": False}
        yield outcome
        if not outcome["golden_failed"] and pass_count != 2:
            raise RuntimeError("Target profiling must execute exactly one native two-pass protocol")
    finally:
        runtime._run_l3_swimlane_two_pass = original_two_pass
        runtime._execute_distributed = original_execute
        worker_type._submit_compiled = original_submit
