import json
import math
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run

import common
import workflow


def _config(case_dir, case):
    identity = case['case_id'].replace('.', '_')
    config = common.load_isolated_module(case_dir / case['config'], f'config_{identity}')
    if config.CASE_ID != case['case_id']:
        raise ValueError('public config Case identity mismatch')
    return config


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate comparator key: {key}')
        result[key] = value
    return result


def _fields(value, expected):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f'invalid comparator fields; expected {sorted(expected)}')


def _number(value, name, *, positive=False, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'invalid comparator {name}')
    if value < 0 or (positive and value == 0) or (maximum is not None and value > maximum):
        raise ValueError(f'invalid comparator {name}')


def _ratio_reldiff_callback(rule):
    threshold = rule['diff_thd']
    allowed_ratio = rule['pct_thd']
    hard_cap = rule['max_diff_hd']
    floor = rule['floor_numerator'] / threshold
    epsilon = rule['denominator_epsilon']

    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, 'output shape mismatch'
        actual_fp32 = actual.detach().cpu().float()
        expected_fp32 = expected.detach().cpu().float()
        if not torch.isfinite(actual_fp32).all() or not torch.isfinite(expected_fp32).all():
            return False, 'nonfinite output rejected on either side'
        if actual_fp32.numel() == 0:
            return True, ''
        absolute = (actual_fp32 - expected_fp32).abs()
        denominator = torch.maximum(
            torch.maximum(actual_fp32.abs(), expected_fp32.abs()),
            torch.full_like(actual_fp32, floor),
        ) + epsilon
        relative = torch.where(absolute < threshold, absolute, absolute / denominator)
        bad = relative > threshold
        bad_count = int(bad.sum())
        allowed_count = round(allowed_ratio * actual_fp32.numel())
        worst = float(relative[bad].max()) if bad_count else 0.0
        passed = bad_count <= allowed_count and (hard_cap is None or worst < hard_cap)
        return passed, '' if passed else (
            f'ratio_reldiff: bad={bad_count}/{actual_fp32.numel()}, '
            f'allowed={allowed_count}, worst={worst}, cap={hard_cap}'
        )

    compare.__name__ = rule['name']
    return compare


def make_compare(case_dir: Path, case: dict):
    policy = json.loads(
        (case_dir / case['comparator']).read_text(encoding='utf-8'),
        object_pairs_hook=_unique_object,
    )
    _fields(policy, {'schema_version', 'outputs'})
    if type(policy['schema_version']) is not int or policy['schema_version'] != 1:
        raise ValueError('unsupported comparator schema')
    if case['correctness']['outputs'] != ['x_next']:
        raise ValueError('Case correctness output coverage disagrees with MoE ABI')
    _fields(policy['outputs'], {'x_next'})
    rule = policy['outputs']['x_next']
    _fields(rule, {'method', 'name', 'diff_thd', 'pct_thd', 'max_diff_hd',
                   'floor_numerator', 'denominator_epsilon', 'nonfinite_policy'})
    if rule['method'] != 'custom' or rule['name'] != 'ratio_reldiff':
        raise ValueError('unsupported MoE comparator')
    if rule['nonfinite_policy'] != 'reject_either':
        raise ValueError('unsupported nonfinite policy')
    _number(rule['diff_thd'], 'diff_thd', positive=True)
    _number(rule['pct_thd'], 'pct_thd', maximum=1)
    _number(rule['floor_numerator'], 'floor_numerator', positive=True)
    _number(rule['denominator_epsilon'], 'denominator_epsilon', positive=True)
    if rule['max_diff_hd'] is not None:
        _number(rule['max_diff_hd'], 'max_diff_hd', positive=True)
    return {'x_next': _ratio_reldiff_callback(rule)}


def build_specs(fixture, config):
    specs = []
    for name, shape in config.TENSOR_SHAPES.items():
        dtype = getattr(torch, config.TENSOR_DTYPES[name])
        if name == 'x_next':
            specs.append(TensorSpec(name, list(shape), dtype))
            continue
        value = torch.load(fixture / 'in' / f'{name}.pt', map_location='cpu', weights_only=True)
        if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            raise ValueError(f'Fixture ABI mismatch: {name}')
        resident = 'stacked' if name not in {'x_hc', 'input_ids'} else None
        specs.append(TensorSpec(name, list(shape), dtype, init_value=value, resident=resident))
    for name, expected in [('layer_id', config.LAYER_ID), ('num_tokens', config.NUM_TOKENS)]:
        value = torch.load(fixture / 'in' / f'{name}.pt', map_location='cpu', weights_only=True)
        if value.ndim != 0 or int(value) != expected:
            raise ValueError(f'Fixture scalar mismatch: {name}')
        specs.append(ScalarSpec(name, getattr(torch, config.SCALAR_DTYPES[name]), int(value)))
    return specs


def _program(module, case):
    return getattr(module, case['entry'])


def _oracle_golden(oracle, values):
    outputs = oracle.compute(values)
    if set(outputs) != {'x_next'}:
        raise ValueError('Oracle correctness output coverage mismatch')
    values['x_next'].copy_(outputs['x_next'])


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    identity = case['case_id'].replace('.', '_')
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, f'ref_{identity}'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, f'oracle_{identity}'),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case),
        run=run, oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    config = _config(case_dir, case)
    identity = case['case_id'].replace('.', '_')
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        candidate_workspace=candidate_workspace, phase=phase,
        build_specs=lambda fixture: build_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, f'candidate_{identity}'
        ),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run,
    )
