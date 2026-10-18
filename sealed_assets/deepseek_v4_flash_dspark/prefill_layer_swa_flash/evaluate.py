"""Case-local fixed ABI and comparator; lifecycle belongs to common workflow."""
from pathlib import Path
import json
import math

import torch
from golden import TensorSpec, ScalarSpec, run
import common
import workflow


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate comparator key: ' + key)
        result[key] = value
    return result


def _number(rule, key, *, positive=False, fraction=False):
    value = rule[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Invalid finite numeric field: ' + key)
    if value < 0 or (positive and value == 0) or (fraction and value > 1):
        raise ValueError('Comparator value out of range: ' + key)


def _validate_comparator(policy, case):
    if not isinstance(policy, dict) or set(policy) != {'schema_version', 'outputs'}:
        raise ValueError('Invalid comparator top-level fields')
    if type(policy['schema_version']) is not int or policy['schema_version'] != 1:
        raise ValueError('Invalid comparator schema_version')
    outputs = policy['outputs']
    expected_names = case['correctness']['outputs']
    if not isinstance(outputs, dict) or set(outputs) != set(expected_names) or len(expected_names) != len(set(expected_names)):
        raise ValueError('Comparator output coverage mismatch')
    for name, rule in outputs.items():
        if not isinstance(rule, dict):
            raise ValueError('Comparator output rule is not an object: ' + name)
        method = rule.get('method')
        if method == 'exact':
            fields = {'method'}
        elif method == 'ratio_allclose':
            fields = {'method', 'rtol', 'atol', 'max_error_ratio'}
        elif method == 'custom' and rule.get('name') == 'ratio_reldiff':
            fields = {'method', 'name', 'diff_thd', 'pct_thd', 'max_diff_hd',
                      'relative_floor_numerator', 'denominator_epsilon'}
        else:
            raise ValueError('Unsupported comparator method/name: ' + str(method))
        if set(rule) != fields:
            raise ValueError('Comparator selected-method fields mismatch: ' + name)
        if method == 'ratio_allclose':
            _number(rule, 'rtol')
            _number(rule, 'atol')
            _number(rule, 'max_error_ratio', fraction=True)
        elif method == 'custom':
            _number(rule, 'diff_thd', positive=True)
            _number(rule, 'pct_thd', fraction=True)
            _number(rule, 'relative_floor_numerator', positive=True)
            _number(rule, 'denominator_epsilon', positive=True)
            if rule['max_diff_hd'] is not None:
                _number(rule, 'max_diff_hd', positive=True)
    return outputs


def _callback(rule):
    # Lexical policy state cannot be overridden by runner rtol/atol keywords.
    policy = dict(rule)
    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False, 'Output ABI shape/dtype mismatch'
        if policy['method'] == 'exact':
            ok = torch.equal(actual, expected)
            return ok, '' if ok else 'Exact output mismatch'
        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)
        if not torch.isfinite(actual_f).all().item() or not torch.isfinite(expected_f).all().item():
            return False, 'Non-finite output'
        difference = (actual_f - expected_f).abs()
        if policy['method'] == 'ratio_allclose':
            bad = difference > policy['atol'] + policy['rtol'] * expected_f.abs()
            count = int(bad.sum().item())
            allowed = round(policy['max_error_ratio'] * actual_f.numel())
            ok = count <= allowed
            return ok, '' if ok else f'ratio_allclose: {count} bad elements, allowed {allowed}'
        if actual_f.numel() == 0:
            return True, ''
        floor = policy['relative_floor_numerator'] / policy['diff_thd']
        denominator = torch.maximum(actual_f.abs(), expected_f.abs()).clamp_min(floor)
        denominator = denominator + policy['denominator_epsilon']
        relative = torch.where(difference < policy['diff_thd'], difference, difference / denominator)
        bad = relative > policy['diff_thd']
        count = int(bad.sum().item())
        allowed = round(policy['pct_thd'] * actual_f.numel())
        worst = float(relative[bad].max().item()) if count else 0.0
        cap = policy['max_diff_hd']
        ok = count <= allowed and (cap is None or worst < cap)
        return ok, '' if ok else f'ratio_reldiff: {count} bad elements, allowed {allowed}, worst {worst}'
    compare.__name__ = policy.get('name', policy['method'])
    return compare


def make_compare(case_dir, case):
    policy = json.loads((Path(case_dir) / case['comparator']).read_text(), object_pairs_hook=_unique_object)
    return {name: _callback(rule) for name, rule in _validate_comparator(policy, case).items()}


def _config(case_dir, case):
    identity = case['case_id'].replace('.', '_')
    config = common.load_isolated_module(Path(case_dir) / case['config'], identity + '_evaluator_config')
    if config.CASE_ID != case['case_id']:
        raise ValueError('Public configuration Case identity mismatch')
    return config


def build_specs(case_dir, case, fixture):
    config = _config(case_dir, case)
    fixture = Path(fixture)
    specs = []
    for name, shape, dtype, resident in _tensor_abi(config):
        path = fixture / 'in' / (name + '.pt')
        required_input = name not in case['correctness']['outputs'] or name in case['state']['initial_state']
        if required_input:
            value = torch.load(path, map_location='cpu', weights_only=True)
            if list(value.shape) != shape or value.dtype != dtype:
                raise ValueError('Fixture tensor ABI mismatch: ' + name)
        else:
            if path.exists():
                raise ValueError('Unexpected pure-output Fixture input: ' + name)
            value = None
        specs.append(TensorSpec(name, shape, dtype, init_value=value, resident=resident))
    layer_id = torch.load(fixture / 'in/layer_id.pt', map_location='cpu', weights_only=True)
    if layer_id.shape != torch.Size([]) or layer_id.dtype != torch.int32 or layer_id.item() != config.LAYER_ID:
        raise ValueError('Fixture scalar differs from fixed layer_id')
    specs.append(ScalarSpec('layer_id', torch.int32, layer_id))
    return specs


def _make_program(module, case):
    return getattr(module, case['entry'])


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case['case_id'].replace('.', '_')
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(case_dir, case, fixture),
        load_ref=lambda: common.load_ref(case_dir, sealed, identity + '_ref'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + '_oracle'),
        make_program=_make_program, make_compare=lambda: make_compare(case_dir, case),
        run=run, oracle_golden=lambda oracle, values: oracle.golden_prefill_layer(values),
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
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(case_dir, case, fixture),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + '_candidate'),
        make_program=_make_program, make_compare=lambda: make_compare(case_dir, case), run=run_profile_aware,
    )


def _tensor_abi(config):
    return [
        ('x_hc', [config.N_RANKS, config.GROUP_TOKENS, config.HC_MULT, config.D], torch.float32, None),
        ('hc_attn_fn', [config.N_RANKS, config.MIX_HC, config.HC_DIM], torch.float32, 'stacked'),
        ('hc_attn_scale', [config.N_RANKS, 3], torch.float32, 'stacked'),
        ('hc_attn_base', [config.N_RANKS, config.MIX_HC], torch.float32, 'stacked'),
        ('attn_norm_w', [config.N_RANKS, config.D], torch.bfloat16, 'stacked'),
        ('wq_a', [config.N_RANKS, config.D, config.Q_LORA], torch.bfloat16, 'stacked'),
        ('wq_b', [config.N_RANKS, config.Q_LORA, config.H * config.HEAD_DIM], torch.int8, 'stacked'),
        ('wq_b_scale', [config.N_RANKS, config.H * config.HEAD_DIM], torch.float32, 'stacked'),
        ('wkv', [config.N_RANKS, config.D, config.HEAD_DIM], torch.bfloat16, 'stacked'),
        ('gamma_cq', [config.N_RANKS, config.Q_LORA], torch.bfloat16, 'stacked'),
        ('gamma_ckv', [config.N_RANKS, config.HEAD_DIM], torch.bfloat16, 'stacked'),
        ('freqs_cos', [config.N_RANKS, 2, config.MAX_SEQ_LEN, config.ROPE_HEAD_DIM], torch.bfloat16, 'stacked'),
        ('freqs_sin', [config.N_RANKS, 2, config.MAX_SEQ_LEN, config.ROPE_HEAD_DIM], torch.bfloat16, 'stacked'),
        ('hca_cmp_wkv', [config.N_RANKS, config.HCA_MAIN_OUT_DIM, config.D], torch.bfloat16, 'stacked'),
        ('hca_cmp_wgate', [config.N_RANKS, config.HCA_MAIN_OUT_DIM, config.D], torch.bfloat16, 'stacked'),
        ('hca_cmp_ape', [config.N_RANKS, 128, config.HCA_MAIN_OUT_DIM], torch.float32, 'stacked'),
        ('hca_cmp_norm_w', [config.N_RANKS, config.HEAD_DIM], torch.bfloat16, 'stacked'),
        ('hca_compress_state', [config.N_RANKS, config.HCA_STATE_BLOCK_NUM, config.HCA_STATE_BLOCK_SIZE, config.HCA_COMPRESS_STATE_DIM], torch.float32, None),
        ('hca_compress_state_block_table', [config.N_RANKS, config.HCA_STATE_MAX_BLOCKS], torch.int32, None),
        ('csa_cmp_wkv', [config.N_RANKS, config.CSA_MAIN_OUT_DIM, config.D], torch.bfloat16, 'stacked'),
        ('csa_cmp_wgate', [config.N_RANKS, config.CSA_MAIN_OUT_DIM, config.D], torch.bfloat16, 'stacked'),
        ('csa_cmp_ape', [config.N_RANKS, 4, config.CSA_MAIN_OUT_DIM], torch.float32, 'stacked'),
        ('csa_cmp_norm_w', [config.N_RANKS, config.HEAD_DIM], torch.bfloat16, 'stacked'),
        ('csa_compress_state', [config.N_RANKS, config.CSA_STATE_BLOCK_NUM, config.CSA_STATE_BLOCK_SIZE, config.CSA_COMPRESS_STATE_DIM], torch.float32, None),
        ('csa_compress_state_block_table', [config.N_RANKS, config.CSA_STATE_MAX_BLOCKS], torch.int32, None),
        ('csa_hadamard_idx', [config.N_RANKS, config.IDX_HEAD_DIM, config.IDX_HEAD_DIM], torch.bfloat16, 'stacked'),
        ('csa_idx_wq_b', [config.N_RANKS, config.Q_LORA, config.IDX_N_HEADS * config.IDX_HEAD_DIM], torch.int8, 'stacked'),
        ('csa_idx_wq_b_scale', [config.N_RANKS, config.IDX_N_HEADS * config.IDX_HEAD_DIM], torch.float32, 'stacked'),
        ('csa_weights_proj', [config.N_RANKS, config.D, config.IDX_N_HEADS], torch.bfloat16, 'stacked'),
        ('csa_inner_wkv', [config.N_RANKS, config.INNER_OUT_DIM, config.D], torch.bfloat16, 'stacked'),
        ('csa_inner_wgate', [config.N_RANKS, config.INNER_OUT_DIM, config.D], torch.bfloat16, 'stacked'),
        ('csa_inner_ape', [config.N_RANKS, 4, config.INNER_OUT_DIM], torch.float32, 'stacked'),
        ('csa_inner_norm_w', [config.N_RANKS, config.IDX_HEAD_DIM], torch.bfloat16, 'stacked'),
        ('csa_inner_compress_state', [config.N_RANKS, config.INNER_STATE_BLOCK_NUM, config.INNER_STATE_BLOCK_SIZE, config.INNER_COMPRESS_STATE_DIM], torch.float32, None),
        ('csa_inner_compress_state_block_table', [config.N_RANKS, config.INNER_STATE_MAX_BLOCKS], torch.int32, None),
        ('kv_cache', [config.N_RANKS, config.CSA_ORI_BLOCK_NUM, config.BLOCK_SIZE, 1, config.HEAD_DIM], torch.bfloat16, None),
        ('ori_block_table', [config.N_RANKS, config.ORI_BLOCK_TABLE_COLS], torch.int32, None),
        ('ori_slot_mapping_full', [config.N_RANKS, config.GROUP_TOKENS], torch.int64, None),
        ('hca_cmp_kv', [config.N_RANKS, config.HCA_CMP_BLOCK_NUM, config.BLOCK_SIZE, 1, config.HEAD_DIM], torch.bfloat16, None),
        ('csa_cmp_kv', [config.N_RANKS, config.CSA_CMP_BLOCK_NUM, config.BLOCK_SIZE, 1, config.HEAD_DIM], torch.bfloat16, None),
        ('hca_cmp_block_table', [config.N_RANKS, config.SPARSE_CMP_MAX_BLOCKS], torch.int32, None),
        ('csa_cmp_block_table', [config.N_RANKS, config.SPARSE_CMP_MAX_BLOCKS], torch.int32, None),
        ('idx_kv_cache', [config.N_RANKS, config.IDX_CACHE_BLOCK_NUM, config.BLOCK_SIZE, 1, config.IDX_HEAD_DIM], torch.int8, None),
        ('idx_kv_scale', [config.N_RANKS, config.IDX_CACHE_BLOCK_NUM, config.BLOCK_SIZE, 1, 1], torch.float32, None),
        ('idx_block_table', [config.N_RANKS, config.IDX_CACHE_MAX_BLOCKS], torch.int32, None),
        ('position_ids_local', [config.N_RANKS, config.T], torch.int32, None),
        ('position_ids_full', [config.N_RANKS, config.GROUP_TOKENS], torch.int32, None),
        ('hca_cmp_slot_mapping_full', [config.N_RANKS, config.GROUP_TOKENS], torch.int64, None),
        ('hca_state_slot_mapping_full', [config.N_RANKS, config.GROUP_TOKENS], torch.int64, None),
        ('csa_cmp_slot_mapping_full', [config.N_RANKS, config.GROUP_TOKENS], torch.int64, None),
        ('csa_idx_slot_mapping_full', [config.N_RANKS, config.GROUP_TOKENS], torch.int64, None),
        ('csa_state_slot_mapping_full', [config.N_RANKS, config.GROUP_TOKENS], torch.int64, None),
        ('csa_inner_state_slot_mapping_full', [config.N_RANKS, config.GROUP_TOKENS], torch.int64, None),
        ('attn_sink', [config.N_RANKS, config.H], torch.float32, 'stacked'),
        ('wo_a', [config.N_RANKS, config.O_PROJ_LOCAL_GROUPS, config.O_LORA, config.O_GROUP_IN], torch.bfloat16, 'stacked'),
        ('wo_b', [config.N_RANKS, config.D, config.O_PROJ_LOCAL_COLS], torch.int8, 'stacked'),
        ('wo_b_scale', [config.N_RANKS, config.D], torch.float32, 'stacked'),
        ('hc_ffn_fn', [config.N_RANKS, config.MIX_HC, config.HC_DIM], torch.float32, 'stacked'),
        ('hc_ffn_scale', [config.N_RANKS, 3], torch.float32, 'stacked'),
        ('hc_ffn_base', [config.N_RANKS, config.MIX_HC], torch.float32, 'stacked'),
        ('norm_w', [config.N_RANKS, config.D], torch.bfloat16, 'stacked'),
        ('gate_w', [config.N_RANKS, config.N_EXPERTS_GLOBAL, config.D], torch.float32, 'stacked'),
        ('gate_bias', [config.N_RANKS, config.N_EXPERTS_GLOBAL], torch.float32, 'stacked'),
        ('tid2eid', [config.N_RANKS, config.VOCAB, config.TOPK], torch.int32, 'stacked'),
        ('input_ids', [config.N_RANKS, config.T], torch.int64, None),
        ('routed_w1', [config.N_RANKS, config.N_LOCAL, config.MOE_INTER, config.D], torch.int8, 'stacked'),
        ('routed_w1_scale', [config.N_RANKS, config.N_LOCAL, config.MOE_INTER], torch.float32, 'stacked'),
        ('routed_w3', [config.N_RANKS, config.N_LOCAL, config.MOE_INTER, config.D], torch.int8, 'stacked'),
        ('routed_w3_scale', [config.N_RANKS, config.N_LOCAL, config.MOE_INTER], torch.float32, 'stacked'),
        ('routed_w2', [config.N_RANKS, config.N_LOCAL, config.D, config.MOE_INTER], torch.int8, 'stacked'),
        ('routed_w2_scale', [config.N_RANKS, config.N_LOCAL, config.D], torch.float32, 'stacked'),
        ('shared_w1', [config.N_RANKS, config.MOE_INTER, config.D], torch.int8, 'stacked'),
        ('shared_w1_scale', [config.N_RANKS, config.MOE_INTER], torch.float32, 'stacked'),
        ('shared_w3', [config.N_RANKS, config.MOE_INTER, config.D], torch.int8, 'stacked'),
        ('shared_w3_scale', [config.N_RANKS, config.MOE_INTER], torch.float32, 'stacked'),
        ('shared_w2', [config.N_RANKS, config.D, config.MOE_INTER], torch.int8, 'stacked'),
        ('shared_w2_scale', [config.N_RANKS, config.D], torch.float32, 'stacked'),
        ('attn_stage', [config.N_RANKS, config.GROUP_TOKENS, config.HC_MULT, config.D], torch.float32, None),
        ('x_mixed', [config.N_RANKS, config.GROUP_TOKENS, config.D], torch.bfloat16, None),
        ('post_ffn', [config.N_RANKS, config.GROUP_TOKENS, config.HC_MULT], torch.float32, None),
        ('comb_ffn', [config.N_RANKS, config.GROUP_TOKENS, config.HC_MULT * config.HC_MULT], torch.float32, None),
        ('ffn_out', [config.N_RANKS, config.T, config.D], torch.bfloat16, None),
        ('x_next', [config.N_RANKS, config.GROUP_TOKENS, config.HC_MULT, config.D], torch.float32, None),
    ]


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
