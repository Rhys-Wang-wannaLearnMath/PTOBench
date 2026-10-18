"""Case-local evaluation for a maintainer-repaired fixed model preset."""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch
from golden import ScalarSpec, TensorSpec, run
import common


def build_tensor_specs(fixture, config=None):
    abi = common.read_json(fixture.parents[1] / 'fixture_abi.json')
    specs = []
    for item in abi:
        name = item['name']
        dtype = getattr(torch, item['dtype'])
        path = fixture / 'in' / (name + '.pt')
        value = torch.load(path, weights_only=True, map_location='cpu') if path.exists() else None
        if item['scalar']:
            specs.append(ScalarSpec(name, dtype, value))
        else:
            spec = TensorSpec(name, item['shape'], dtype, init_value=value)
            if item.get('resident'):
                spec.resident = item['resident']
            specs.append(spec)
    return specs


def _number(value, name, *, positive=False, ratio=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be numeric')
    if not math.isfinite(value) or value < 0 or (positive and value == 0) or (ratio and value > 1):
        raise ValueError(f'{name} has invalid range')
    return float(value)


def _callback(settings, name):
    if not isinstance(settings, dict):
        raise ValueError('comparator rule must be an object')
    method = settings.get('method')
    if method in ('allclose', 'ratio_allclose'):
        keys = {'method', 'rtol', 'atol'} | ({'max_error_ratio'} if method == 'ratio_allclose' else set())
        if set(settings) != keys:
            raise ValueError('unexpected comparator fields')
        tolerance = _number(settings['rtol'], 'rtol')
        absolute = _number(settings['atol'], 'atol')
        budget = _number(settings.get('max_error_ratio', 0), 'max_error_ratio', ratio=True)

        def compare(actual, expected, **kwargs):
            if actual.shape != expected.shape:
                return False, 'shape mismatch'
            a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
            if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()):
                return False, 'nonfinite value'
            count = int(((a-e).abs() > absolute + tolerance * e.abs()).sum())
            allowed = round(budget * a.numel())
            return count <= allowed, f'{name}: mismatches={count}/{a.numel()}, allowed={allowed}'
        return compare
    if method != 'custom':
        raise ValueError(f'unsupported comparator {method}')
    kind = settings.get('name')
    if kind == 'token_partition_ratio_reldiff':
        return _partition_callback(settings, name)
    if kind == 'ratio_reldiff':
        keys = {'method', 'name', 'diff_threshold', 'max_error_ratio', 'max_relative_diff'}
        if set(settings) != keys or settings['max_relative_diff'] is not None:
            raise ValueError('invalid ratio_reldiff fields')
        diff = _number(settings['diff_threshold'], 'diff_threshold', positive=True)
        budget = _number(settings['max_error_ratio'], 'max_error_ratio', ratio=True)
        cap = float('inf')
    elif kind == 'valid_ratio_reldiff':
        keys = {'method', 'name', 'valid_rows', 'valid_axis', 'diff_thd', 'pct_thd', 'max_diff_hd', 'tail_policy'}
        if set(settings) != keys or settings['tail_policy'] != 'reject_undeclared':
            raise ValueError('invalid valid_ratio_reldiff fields')
        if type(settings['valid_rows']) is not int or settings['valid_rows'] <= 0 or settings['valid_axis'] != 0:
            raise ValueError('invalid prefix binding')
        diff = _number(settings['diff_thd'], 'diff_thd', positive=True)
        budget = _number(settings['pct_thd'], 'pct_thd', ratio=True)
        cap = _number(settings['max_diff_hd'], 'max_diff_hd', positive=True)
    else:
        raise ValueError(f'unsupported custom comparator {kind}')

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, 'shape mismatch'
        if kind == 'valid_ratio_reldiff':
            rows = settings['valid_rows']
            if actual.shape[0] != rows:
                return False, 'undeclared tail or missing prefix'
        a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
        if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()):
            return False, 'nonfinite value'
        absolute = (a-e).abs()
        floor = (1 / (1 << 14)) / diff
        denominator = torch.maximum(torch.maximum(a.abs(), e.abs()), torch.full_like(a, floor)) + 1e-9
        relative = torch.where(absolute < diff, absolute, absolute / denominator)
        bad = relative > diff
        count = int(bad.sum())
        worst = float(relative[bad].max()) if count else 0.0
        allowed = round(budget * a.numel())
        return count <= allowed and worst < cap, f'{name}: mismatches={count}/{a.numel()}, allowed={allowed}, worst={worst}'
    return compare


def _partition_callback(settings, name):
    keys = {'method', 'name', 'valid_rows', 'token_axis', 'base_rule'}
    if set(settings) != keys or type(settings['valid_rows']) is not int or settings['valid_rows'] < 0 or type(settings['token_axis']) is not int or settings['token_axis'] != 1:
        raise ValueError('invalid token partition')
    if not isinstance(settings['base_rule'], dict) or settings['base_rule'].get('name') != 'ratio_reldiff':
        raise ValueError('invalid partition base rule')
    base = _callback(settings['base_rule'], name)
    def compare(actual, expected, **kwargs):
        rows = settings['valid_rows']
        if actual.shape != expected.shape or actual.ndim < 2 or rows > actual.shape[1]:
            return False, 'partition shape mismatch'
        for region in (slice(0, rows), slice(rows, None)):
            ok, detail = base(actual[:, region], expected[:, region], **kwargs)
            if not ok:
                return ok, detail
        return True, ''
    return compare


def _compare(case_dir, case):
    policy = common.read_json(case_dir / case['comparator'])
    if (not isinstance(policy, dict) or set(policy) != {'schema_version', 'outputs'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1):
        raise ValueError('invalid comparator schema')
    if set(policy['outputs']) != set(case['correctness']['outputs']):
        raise ValueError('comparator output coverage mismatch')
    return {name: _callback(settings, name) for name, settings in policy['outputs'].items()}


def _program(module, case):
    return getattr(module, case['entry'])


def _oracle(oracle, values):
    # Each CPU Oracle owns its independent golden implementation and output allocation.
    for name, value in oracle.compute(values, {}).items():
        values[name][:] = value


def _run(**kwargs):
    return run(rtol=0.0, atol=0.0, save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case['case_id'].replace('.', '_')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', 'workflow_' + identity)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=build_tensor_specs,
        load_ref=lambda: common.load_ref(case_dir, sealed, identity + '_ref'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + '_oracle'),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=_run, oracle_golden=_oracle)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    identity = case['case_id'].replace('.', '_')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', 'workflow_' + identity)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase, build_specs=build_tensor_specs,
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + '_candidate'),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
