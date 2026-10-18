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
    if kind == 'mapped_pool_compare':
        return _mapped_callback(settings, name)
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
        if type(settings['valid_rows']) is not int or settings['valid_rows'] <= 0 or type(settings['valid_axis']) is not int or settings['valid_axis'] != 0:
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


def _mapped_callback(settings, name):
    required = {'method', 'name', 'mapping_inputs', 'ranked', 'valid_rows', 'unmapped_policy', 'base_rule'}
    if set(settings) != required or settings['unmapped_policy'] != 'exact':
        raise ValueError('invalid mapped pool policy')
    bindings = settings['mapping_inputs']
    if not isinstance(bindings, list) or not bindings or any(not isinstance(n, str) or not n for n in bindings) or len(set(bindings)) != len(bindings):
        raise ValueError('invalid mapping bindings')
    ranked = settings['ranked']
    rows = settings['valid_rows']
    if type(ranked) is not bool or (rows is not None and (type(rows) is not int or rows < 0)):
        raise ValueError('invalid mapped pool dimensions')
    base = settings['base_rule']
    if not isinstance(base, dict) or base.get('method') not in ('allclose', 'ratio_allclose'):
        raise ValueError('invalid mapped pool base rule')
    base_compare = _callback(base, name)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape or actual.ndim < (3 if ranked else 2):
            return False, 'mapped pool shape mismatch'
        a, e = actual.detach().cpu(), expected.detach().cpu()
        if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()):
            return False, 'nonfinite pool'
        ranks = a.shape[0] if ranked else 1
        a, e = a.reshape(ranks, -1, a.shape[-1]), e.reshape(ranks, -1, e.shape[-1])
        written = torch.zeros(a.shape[:2], dtype=torch.bool)
        inputs = kwargs.get('inputs', {})
        for binding in bindings:
            mapping = inputs.get(binding)
            if not isinstance(mapping, torch.Tensor) or mapping.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                return False, 'missing or invalid mapping ' + binding
            mapping = mapping.cpu().to(torch.int64)
            if mapping.ndim != (2 if ranked else 1) or (ranked and mapping.shape[0] != ranks):
                return False, 'mapping shape mismatch'
            mapping = mapping.reshape(ranks, -1)
            if bool((mapping < -1).any()):
                return False, 'invalid negative mapping'
            if rows is not None:
                if rows > mapping.shape[1] or bool((mapping[:, rows:] != -1).any()):
                    return False, 'invalid mapping tail'
                mapping = mapping[:, :rows]
            for rank in range(ranks):
                chosen = mapping[rank][mapping[rank] >= 0]
                if bool((chosen >= a.shape[1]).any()) or chosen.numel() != torch.unique(chosen).numel():
                    return False, 'out of range or duplicate mapping'
                if bool(written[rank, chosen].any()):
                    return False, 'overlapping mappings'
                written[rank, chosen] = True
        if not torch.equal(a[~written], e[~written]):
            return False, 'unmapped pool row changed'
        return base_compare(a[written], e[written], **kwargs)
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
    result = run(rtol=0.0, atol=0.0, save_data=False, **kwargs)
    return result


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
