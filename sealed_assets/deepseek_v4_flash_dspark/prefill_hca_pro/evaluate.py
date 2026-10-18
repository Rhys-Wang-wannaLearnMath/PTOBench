"""Case-local packed HCA ABI and comparator execution; lifecycle is shared."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common
import workflow


_NAMES = ('x_hc', 'hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'attn_norm_w', 'wq_a', 'wq_b', 'wq_b_scale', 'wkv', 'gamma_cq', 'gamma_ckv', 'freqs_cos', 'freqs_sin', 'cmp_wkv', 'cmp_wgate', 'cmp_ape', 'cmp_norm_w', 'compress_state', 'compress_state_block_table', 'kv_cache', 'ori_slot_mapping', 'ori_block_table', 'cmp_kv', 'cmp_block_table', 'position_ids', 'cmp_slot_mapping', 'state_slot_mapping', 'attn_sink', 'wo_a', 'wo_b', 'wo_b_scale', 'x_out', 'num_tokens')


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate comparator key: {key}')
        value[key] = item
    return value


def _number(value, name, *, positive=False, upper=None):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f'invalid comparator {name}')
    value = float(value)
    if value < 0 or (positive and value <= 0) or (upper is not None and value > upper):
        raise ValueError(f'invalid comparator {name}')
    return value


def _policy(case_dir, case):
    value = json.loads((Path(case_dir) / case['comparator']).read_text(), object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {'schema_version', 'outputs'} or type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('invalid comparator top-level fields')
    outputs = value['outputs']
    if not isinstance(outputs, dict) or set(outputs) != set(case['correctness']['outputs']) or set(outputs) != {'x_out', 'kv_cache'}:
        raise ValueError('comparator output coverage disagrees with contract')
    return value


def _valid_ratio(rule, shape):
    required = {'method', 'name', 'diff_thd', 'pct_thd', 'max_diff_hd', 'valid_rows', 'valid_axis', 'zero_tail', 'small_value_floor_numerator', 'denominator_epsilon', 'nonfinite'}
    if not isinstance(rule, dict) or set(rule) != required or rule['method'] != 'custom' or rule['name'] != 'valid_ratio_reldiff':
        raise ValueError('invalid valid_ratio_reldiff fields')
    if rule['valid_rows'] != 'num_tokens' or type(rule['valid_axis']) is not int or rule['valid_axis'] != 0 or rule['zero_tail'] is not True or rule['nonfinite'] != 'reject_both':
        raise ValueError('invalid valid_ratio_reldiff scope')
    policy_diff = _number(rule['diff_thd'], 'diff_thd', positive=True)
    policy_pct = _number(rule['pct_thd'], 'pct_thd', upper=1)
    policy_cap = _number(rule['max_diff_hd'], 'max_diff_hd', positive=True)
    policy_floor = _number(rule['small_value_floor_numerator'], 'small_value_floor_numerator', positive=True)
    policy_epsilon = _number(rule['denominator_epsilon'], 'denominator_epsilon', positive=True)

    def compare(actual, expected, **kwargs):
        if tuple(actual.shape) != shape or tuple(expected.shape) != shape or actual.dtype != torch.float32 or expected.dtype != torch.float32:
            return False, 'x_out ABI mismatch'
        scalar = (kwargs.get('inputs') or {}).get(rule['valid_rows'])
        if not isinstance(scalar, torch.Tensor) or scalar.dtype != torch.int32 or scalar.ndim != 0:
            return False, 'num_tokens must be an INT32 scalar'
        valid_rows = int(scalar.item())
        if not 1 <= valid_rows <= shape[0]:
            return False, 'num_tokens out of bounds'
        a = actual.detach().cpu()
        b = expected.detach().cpu()
        if bool(a[valid_rows:].ne(0).any()):
            return False, 'nonzero inactive tail'
        a = a[:valid_rows]
        b = b[:valid_rows]
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            return False, 'nonfinite output'
        delta = (a - b).abs()
        denominator = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.full_like(a, policy_floor / policy_diff)) + policy_epsilon
        relative = torch.where(delta < policy_diff, delta, delta / denominator)
        bad = relative > policy_diff
        count = int(bad.sum().item())
        limit = round(policy_pct * a.numel())
        worst = float(relative[bad].max().item()) if count else 0.0
        ok = count <= limit and worst < policy_cap
        return ok, '' if ok else f'valid_ratio_reldiff: errors={count}/{a.numel()} threshold={limit} worst={worst}'

    return compare


def _ratio(rule):
    if not isinstance(rule, dict) or set(rule) != {'method', 'rtol', 'atol', 'max_error_ratio'} or rule['method'] != 'ratio_allclose':
        raise ValueError('invalid ratio_allclose fields')
    policy_rtol = _number(rule['rtol'], 'rtol')
    policy_atol = _number(rule['atol'], 'atol')
    policy_ratio = _number(rule['max_error_ratio'], 'max_error_ratio', upper=1)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, 'output shape mismatch'
        a = actual.detach().cpu().to(torch.float32)
        b = expected.detach().cpu().to(torch.float32)
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            return False, 'nonfinite output'
        count = int((a - b).abs().gt(policy_atol + policy_rtol * b.abs()).sum().item())
        limit = round(policy_ratio * a.numel())
        return count <= limit, '' if count <= limit else f'ratio_allclose: errors={count}/{a.numel()} threshold={limit}'

    return compare


def _config(case_dir, case):
    identity = case['case_id'].replace('.', '_')
    config = common.load_isolated_module(Path(case_dir) / case['config'], f'config_{identity}')
    if config.CASE_ID != case['case_id']:
        raise ValueError('public config identity disagrees with Case')
    return config


def make_compare(case_dir, case):
    rules = _policy(case_dir, case)['outputs']
    config = _config(case_dir, case)
    return {'x_out': _valid_ratio(rules['x_out'], (config.T, config.HC_MULT, config.D)),
            'kv_cache': _ratio(rules['kv_cache'])}


def _shapes(config):
    c = config
    return {'x_hc': [c.T, c.HC_MULT, c.D], 'hc_attn_fn': [c.MIX_HC, c.HC_DIM], 'hc_attn_scale': [3], 'hc_attn_base': [c.MIX_HC],
        'attn_norm_w': [c.D], 'wq_a': [c.D, c.Q_LORA], 'wq_b': [c.Q_LORA, c.H*c.HEAD_DIM], 'wq_b_scale': [c.H*c.HEAD_DIM],
        'wkv': [c.D, c.HEAD_DIM], 'gamma_cq': [c.Q_LORA], 'gamma_ckv': [c.HEAD_DIM], 'freqs_cos': [c.MAX_SEQ_LEN,c.ROPE_DIM], 'freqs_sin': [c.MAX_SEQ_LEN,c.ROPE_DIM],
        'cmp_wkv': [c.OUT_DIM,c.D], 'cmp_wgate': [c.OUT_DIM,c.D], 'cmp_ape': [c.COMPRESS_RATIO,c.OUT_DIM], 'cmp_norm_w': [c.HEAD_DIM],
        'compress_state': [c.HCA_STATE_BLOCK_NUM,c.HCA_STATE_BLOCK_SIZE,c.COMPRESS_STATE_DIM], 'compress_state_block_table': [c.HCA_STATE_MAX_BLOCKS],
        'kv_cache': [c.HCA_ORI_BLOCK_NUM,c.BLOCK_SIZE,1,c.HEAD_DIM], 'ori_slot_mapping': [c.T], 'ori_block_table': [c.SPARSE_ORI_MAX_BLOCKS],
        'cmp_kv': [c.HCA_CMP_BLOCK_NUM,c.BLOCK_SIZE,1,c.HEAD_DIM], 'cmp_block_table': [c.SPARSE_CMP_MAX_BLOCKS], 'position_ids': [c.T],
        'cmp_slot_mapping': [c.T], 'state_slot_mapping': [c.T], 'attn_sink': [c.H], 'wo_a': [c.O_GROUPS,c.O_LORA,c.O_GROUP_IN],
        'wo_b': [c.D,c.O_GROUPS*c.O_LORA], 'wo_b_scale': [c.D], 'x_out': [c.T,c.HC_MULT,c.D]}


class _CorrectnessSpec(TensorSpec):
    """Preserve compiled directions and input/reset membership, project validation."""
    @property
    def is_output(self):
        return super().is_output and self.name in {'x_out', 'kv_cache'}


def build_specs(fixture, config):
    fixture = Path(fixture)
    shapes = _shapes(config)
    specs = []
    for name in _NAMES:
        dtype = getattr(torch, config.TENSOR_DTYPES[name])
        if name == 'x_out':
            specs.append(_CorrectnessSpec(name, shapes[name], dtype))
            continue
        value = torch.load(fixture / 'in' / f'{name}.pt', map_location='cpu', weights_only=True)
        if name == 'num_tokens':
            if value.ndim != 0 or value.dtype != dtype or value.item() != config.NUM_TOKENS:
                raise ValueError('Fixture num_tokens ABI mismatch')
            specs.append(ScalarSpec(name, dtype, value))
        else:
            if list(value.shape) != shapes[name] or value.dtype != dtype:
                raise ValueError(f'Fixture {name} ABI mismatch')
            specs.append(_CorrectnessSpec(name, shapes[name], dtype, init_value=value))
    return specs


def _program(module, case):
    return getattr(module, case['entry'])


def _golden(oracle, values):
    result = oracle.compute(values)
    if set(result) != {'x_out', 'kv_cache'}:
        raise ValueError('Oracle output coverage disagrees with contract')
    for name, value in result.items():
        values[name].copy_(value)


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = case['case_id'].replace('.', '_')
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f'ref_{identity}'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f'oracle_{identity}'),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run,
        oracle_golden=_golden)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    config = _config(case_dir, case)
    identity = case['case_id'].replace('.', '_')
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, f'candidate_{identity}'),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run)
