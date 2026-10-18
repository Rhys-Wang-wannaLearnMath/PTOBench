"""Case-local ABI, CPU certification bridge and public comparator execution."""
import json
import math
from pathlib import Path
import torch
from golden import TensorSpec, ScalarSpec, run
import common
import workflow

ABI = [('x_hc', 'float32', None), ('hc_attn_fn', 'float32', 'stacked'), ('hc_attn_scale', 'float32', 'stacked'), ('hc_attn_base', 'float32', 'stacked'), ('attn_norm_w', 'bfloat16', 'stacked'), ('wq_a', 'bfloat16', 'stacked'), ('wq_b', 'int8', 'stacked'), ('wq_b_scale', 'float32', 'stacked'), ('wkv', 'bfloat16', 'stacked'), ('gamma_cq', 'bfloat16', 'stacked'), ('gamma_ckv', 'bfloat16', 'stacked'), ('freqs_cos', 'bfloat16', 'stacked'), ('freqs_sin', 'bfloat16', 'stacked'), ('cmp_freqs_cos', 'float32', 'stacked'), ('cmp_freqs_sin', 'float32', 'stacked'), ('cmp_wkv', 'bfloat16', 'stacked'), ('cmp_wgate', 'bfloat16', 'stacked'), ('cmp_ape', 'float32', 'stacked'), ('cmp_norm_w', 'bfloat16', 'stacked'), ('compress_state', 'float32', 'stacked'), ('compress_state_block_table', 'int32', None), ('kv_cache', 'bfloat16', 'stacked'), ('cmp_kv', 'bfloat16', 'stacked'), ('cmp_block_table', 'int32', None), ('ori_slot_mapping', 'int64', None), ('window_swa_indices', 'int32', None), ('window_swa_lens', 'int32', None), ('cmp_slot_mapping', 'int64', None), ('state_slot_mapping', 'int64', None), ('position_ids', 'int32', None), ('kv_seq_lens', 'int32', None), ('attn_sink', 'float32', 'stacked'), ('wo_a', 'bfloat16', 'stacked'), ('wo_b', 'int8', 'stacked'), ('wo_b_scale', 'float32', 'stacked'), ('hc_ffn_fn', 'float32', 'stacked'), ('hc_ffn_scale', 'float32', 'stacked'), ('hc_ffn_base', 'float32', 'stacked'), ('norm_w', 'bfloat16', 'stacked'), ('gate_w', 'float32', 'stacked'), ('gate_bias', 'float32', 'stacked'), ('tid2eid', 'int32', 'stacked'), ('input_ids', 'int64', None), ('routed_w1', 'int8', 'stacked'), ('routed_w1_scale', 'float32', 'stacked'), ('routed_w3', 'int8', 'stacked'), ('routed_w3_scale', 'float32', 'stacked'), ('routed_w2', 'int8', 'stacked'), ('routed_w2_scale', 'float32', 'stacked'), ('shared_w1', 'int8', 'stacked'), ('shared_w1_scale', 'float32', 'stacked'), ('shared_w3', 'int8', 'stacked'), ('shared_w3_scale', 'float32', 'stacked'), ('shared_w2', 'int8', 'stacked'), ('shared_w2_scale', 'float32', 'stacked'), ('x_attn_active', 'float32', None), ('x_moe_next', 'float32', None), ('x_next', 'float32', None), ('layer_id', 'int32', None), ('local_t', 'int32', None)]
MODE = 'hca'
IDENTITY = 'deepseek_v4_flash_dspark_decode_layer_hca_demo'


def _number(value, name, *, positive=False, ratio=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name}: expected finite number')
    if (value <= 0 if positive else value < 0) or (ratio and value > 1):
        raise ValueError(f'{name}: invalid range')


def read_policy(case_dir, case):
    policy = json.loads((case_dir / case['comparator']).read_text())
    if set(policy) != {'schema_version', 'outputs'} or type(policy['schema_version']) is not int or policy['schema_version'] != 1:
        raise ValueError('Invalid comparator schema')
    if not isinstance(policy['outputs'], dict) or set(policy['outputs']) != set(case['correctness']['outputs']):
        raise ValueError('Comparator output coverage mismatch')
    for name, rule in policy['outputs'].items():
        if not isinstance(rule, dict) or rule.get('method') != 'custom':
            raise ValueError(f'{name}: unsupported method')
        method = rule.get('comparator')
        if method == 'mapped_pool_ratio_allclose':
            fields = {'method','comparator','mapping_name','mapping_shape','block_size','leading_rank_axis','atol','rtol','max_error_ratio'}
            if set(rule) != fields: raise ValueError(f'{name}: invalid fields')
            if type(rule['mapping_name']) is not str or rule['mapping_name'] not in {s[0] for s in ABI}: raise ValueError('Invalid mapping name')
            shape = rule['mapping_shape']
            if type(shape) is not list or not shape or any(type(d) is not int or d <= 0 for d in shape): raise ValueError('Invalid mapping shape')
            if type(rule['block_size']) is not int or rule['block_size'] <= 0: raise ValueError('Invalid block size')
            if type(rule['leading_rank_axis']) is not bool or (rule['leading_rank_axis'] and len(shape) < 2): raise ValueError('Invalid rank axis')
            for k in ('atol','rtol'): _number(rule[k],k)
            _number(rule['max_error_ratio'],'max_error_ratio',ratio=True)
        elif method == 'ratio_reldiff':
            fields = {'method','comparator','diff_thd','pct_thd','max_diff_hd','valid_rows','valid_axis','zero_tail','small_value_floor_numerator','denominator_epsilon'}
            if set(rule) != fields: raise ValueError(f'{name}: invalid fields')
            for k in ('diff_thd','small_value_floor_numerator','denominator_epsilon'): _number(rule[k],k,positive=True)
            _number(rule['pct_thd'],'pct_thd',ratio=True)
            if rule['max_diff_hd'] is not None: _number(rule['max_diff_hd'],'max_diff_hd',positive=True)
            if rule['valid_rows'] is not None and (type(rule['valid_rows']) is not int or rule['valid_rows'] < 0): raise ValueError('Invalid valid_rows')
            if type(rule['valid_axis']) is not int or rule['valid_axis'] < 0: raise ValueError('Invalid valid_axis')
            if type(rule['zero_tail']) is not bool: raise ValueError('Invalid zero_tail')
        else: raise ValueError(f'{name}: unapproved comparator')
    return policy


def _relative(actual, expected, rule, **kwargs):
    del kwargs
    if actual.shape != expected.shape: return False, 'shape mismatch'
    rows, axis = rule['valid_rows'], rule['valid_axis']
    if rows is not None:
        if axis >= actual.ndim or rows > actual.shape[axis]: return False, 'invalid prefix'
        if rule['zero_tail'] and torch.count_nonzero(actual.narrow(axis, rows, actual.shape[axis]-rows)).item(): return False, 'nonzero tail'
        actual = actual.narrow(axis, 0, rows); expected = expected.narrow(axis, 0, rows)
    if not actual.numel(): return True, ''
    a,e = actual.cpu().float(),expected.cpu().float()
    if not torch.isfinite(a).all().item() or not torch.isfinite(e).all().item(): return False,'nonfinite'
    diff=(a-e).abs(); threshold=rule['diff_thd']
    denominator=torch.maximum(a.abs(),e.abs()).clamp_min(rule['small_value_floor_numerator']/threshold)+rule['denominator_epsilon']
    relative=torch.where(diff < threshold,diff,diff/denominator)
    bad=relative > threshold;count=int(bad.sum().item());worst=float(relative[bad].max().item()) if count else 0.
    cap=rule['max_diff_hd']
    ok=count <= round(rule['pct_thd']*a.numel()) and (cap is None or worst < cap)
    return ok, '' if ok else f'relative mismatch count={count}/{a.numel()} worst={worst}'


def _mapped(actual, expected, rule, **kwargs):
    if actual.shape != expected.shape: return False,'shape mismatch'
    ranked=rule['leading_rank_axis'];axis=2 if ranked else 1;bs=rule['block_size']
    if actual.ndim <= axis or actual.shape[axis] != bs: return False,'block layout mismatch'
    mapping=kwargs.get('inputs',{}).get(rule['mapping_name'])
    if mapping is None or mapping.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8): return False,'missing/integer mapping required'
    if list(mapping.shape) != rule['mapping_shape']: return False,'mapping shape mismatch'
    ranks=rule['mapping_shape'][0] if ranked else 1
    if ranked and actual.shape[0]!=ranks: return False,'rank count mismatch'
    count=actual.shape[1 if ranked else 0]*bs
    a=actual.cpu().reshape(ranks,count,-1);e=expected.cpu().reshape(ranks,count,-1);m=mapping.cpu().long().reshape(ranks,-1)
    if ((m < -1) | (m >= count)).any().item(): return False,'mapping outside physical rows'
    written=torch.zeros((ranks,count),dtype=torch.bool)
    for rank in range(ranks):
        valid=m[rank][m[rank]>=0]
        if valid.unique().numel()!=valid.numel(): return False,'duplicate physical mapping'
        written[rank,valid]=True
    if ((~written) & ~(a==e).all(dim=-1)).any().item(): return False,'unmapped row changed'
    if not written.any().item(): return True,''
    a=a[written].float();e=e[written].float()
    if not torch.isfinite(a).all().item() or not torch.isfinite(e).all().item(): return False,'nonfinite mapped rows'
    bad=(a-e).abs() > rule['atol']+rule['rtol']*e.abs()
    errors=int(bad.sum().item());ok=errors <= round(rule['max_error_ratio']*a.numel())
    return ok,'' if ok else f'mapped mismatch count={errors}/{a.numel()}'


def make_compare(case_dir, case):
    policy=read_policy(case_dir,case)
    callbacks={}
    for name,rule in policy['outputs'].items():
        def compare(actual,expected,*,policy_rule=rule,**kwargs):
            if policy_rule['comparator']=='ratio_reldiff':return _relative(actual,expected,policy_rule,**kwargs)
            return _mapped(actual,expected,policy_rule,**kwargs)
        callbacks[name]=compare
    return callbacks


def build_specs(fixture, case_dir, case):
    cfg=common.load_module(case_dir/case['config'],'config_'+IDENTITY)
    specs=[]
    for name,dtype,resident in ABI:
        dtype=getattr(torch,dtype)
        if name in case['scalars']:
            value=int(torch.load(fixture/'in'/f'{name}.pt',weights_only=True))
            expected=cfg.LAYER_ID if name=='layer_id' else cfg.LOCAL_T
            if value!=expected:raise ValueError(f'{name}: fixture scalar drift')
            specs.append(ScalarSpec(name,dtype,value));continue
        if name in case['correctness']['outputs'] and name not in case['state']['initial_state']:
            tokens=cfg.MOE_TOKENS if name=='x_moe_next' else cfg.LOCAL_T
            specs.append(TensorSpec(name,[cfg.N_RANKS,tokens,cfg.HC_MULT,cfg.D],dtype,resident=resident));continue
        value=torch.load(fixture/'in'/f'{name}.pt',weights_only=True)
        if value.dtype!=dtype:raise ValueError(f'{name}: fixture dtype drift')
        specs.append(TensorSpec(name,list(value.shape),dtype,init_value=lambda value=value:value.clone(),resident=resident))
    return specs


def _program(module, case): return getattr(module,case['entry'])

def _oracle(oracle, values): getattr(oracle,'golden_decode_layer_'+MODE)(values)


def prepare_ref(case_dir,sealed,case,platform=None):
    return workflow.prepare_ref(case_dir=case_dir,sealed=sealed,case=case,platform=platform,
        build_specs=lambda fixture:build_specs(fixture,case_dir,case),
        load_ref=lambda:common.load_ref(case_dir,sealed,'ref_'+IDENTITY),
        load_oracle=lambda:common.load_oracle(case_dir,sealed,'oracle_'+IDENTITY),
        make_program=_program,make_compare=lambda:make_compare(case_dir,case),run=run,oracle_golden=_oracle)


def evaluate_candidate(case_dir,sealed,case,candidate_workspace,platform=None,phase='all'):
    def run_profile_aware(**kwargs):
        with _distributed_profile_state(case, kwargs["specs"], kwargs["config"], common.profiling_mode()) as outcome:
            result = run(**kwargs)
            if outcome is not None:
                outcome["golden_failed"] = not getattr(result, "passed", True)
            return result

    return workflow.evaluate_candidate(case_dir=case_dir,sealed=sealed,case=case,candidate_workspace=candidate_workspace,platform=platform,phase=phase,
        build_specs=lambda fixture:build_specs(fixture,case_dir,case),
        load_candidate=lambda:common.load_candidate(case_dir,case,candidate_workspace,'candidate_'+IDENTITY),
        make_program=_program,make_compare=lambda:make_compare(case_dir,case),run=run_profile_aware)


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
