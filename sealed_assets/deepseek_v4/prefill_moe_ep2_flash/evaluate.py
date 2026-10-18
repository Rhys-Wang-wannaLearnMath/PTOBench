import json
import math

import torch
from golden import ScalarSpec, TensorSpec, run

import common
import workflow


def _config(case_dir, case):
    return common.load_module(
        case_dir / case['config'], 'config_deepseek_v4_prefill_moe_ep2_flash',
    )


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'duplicate comparator key: {name}')
        result[name] = value
    return result


def _number(rule, name, *, positive=False, upper=None):
    value = rule[name]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if value <= 0 if positive else value < 0:
        raise ValueError(f'{name} outside valid range')
    if upper is not None and value > upper:
        raise ValueError(f'{name} outside valid range')


def _validate_comparator(policy, case):
    if type(policy) is not dict or set(policy) != {'schema_version', 'outputs'}:
        raise ValueError('invalid comparator top-level fields')
    if type(policy['schema_version']) is not int or policy['schema_version'] != 1:
        raise ValueError('invalid comparator schema version')
    outputs = policy['outputs']
    if type(outputs) is not dict or set(outputs) != set(case['correctness']['outputs']):
        raise ValueError('comparator output coverage mismatch')
    fields = {'method', 'name', 'diff_threshold', 'max_error_ratio',
              'denominator_floor_numerator', 'denominator_epsilon',
              'reject_nonfinite', 'max_relative_diff'}
    for rule in outputs.values():
        if type(rule) is not dict or set(rule) != fields:
            raise ValueError('invalid selected-method fields')
        if rule['method'] != 'custom' or rule['name'] != 'ratio_reldiff':
            raise ValueError('unsupported comparator method/name')
        _number(rule, 'diff_threshold', positive=True)
        _number(rule, 'max_error_ratio', upper=1)
        _number(rule, 'denominator_floor_numerator', positive=True)
        _number(rule, 'denominator_epsilon')
        if type(rule['reject_nonfinite']) is not bool:
            raise ValueError('reject_nonfinite must be boolean')
        if rule['max_relative_diff'] is not None:
            _number(rule, 'max_relative_diff', positive=True)
    return outputs


def _ratio_reldiff(rule):
    threshold = rule['diff_threshold']
    ratio = rule['max_error_ratio']
    floor = rule['denominator_floor_numerator'] / threshold
    epsilon = rule['denominator_epsilon']
    reject_nonfinite = rule['reject_nonfinite']
    cap = rule['max_relative_diff']

    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, 'output shape mismatch'
        actual_values = actual.detach().cpu().float().reshape(-1)
        expected_values = expected.detach().cpu().float().reshape(-1)
        if reject_nonfinite and (
            not torch.isfinite(actual_values).all() or not torch.isfinite(expected_values).all()
        ):
            return False, 'nonfinite output'
        difference = (actual_values - expected_values).abs()
        denominator = torch.maximum(actual_values.abs(), expected_values.abs()).clamp_min(floor) + epsilon
        relative = torch.where(difference < threshold, difference, difference / denominator)
        bad = relative > threshold
        bad_count = int(bad.sum().item())
        budget = round(ratio * actual_values.numel())
        passed = bad_count <= budget
        if cap is not None and bad_count:
            passed = passed and float(relative[bad].max().item()) < cap
        return passed, f'ratio_reldiff: {bad_count}/{actual_values.numel()} errors, budget={budget}'

    return compare


def _make_compare(case_dir, case):
    policy = json.loads(
        (case_dir / case['comparator']).read_text(encoding='utf-8'),
        object_pairs_hook=_unique_object,
    )
    return {name: _ratio_reldiff(rule) for name, rule in _validate_comparator(policy, case).items()}


def _build_specs(fixture, config, case):
    specs = []
    output_names = set(case['correctness']['outputs'])
    for name, shape in config.TENSOR_SHAPES.items():
        dtype = getattr(torch, config.TENSOR_DTYPES[name])
        initializer = None
        if name not in output_names:
            path = fixture / 'in' / f'{name}.pt'
            initializer = lambda input_path=path: torch.load(input_path, map_location='cpu', weights_only=True)
        resident = config.RESIDENT_LAYOUT if name in config.RESIDENT_INPUTS else None
        specs.append(TensorSpec(name, list(shape), dtype, init_value=initializer, resident=resident))
    for name in case['scalars']:
        value = torch.load(fixture / 'in' / f'{name}.pt', map_location='cpu', weights_only=True)
        specs.append(ScalarSpec(name, getattr(torch, config.SCALAR_DTYPES[name]), int(value.item())))
    return specs


def _make_program(module, case):
    return getattr(module, case['entry'])


def _oracle_golden(oracle, values, config, case):
    scalar_values = {name: int(values[name]) for name in case['scalars']}
    inputs = {name: values[name] for name in config.TENSOR_SHAPES
              if name not in case['correctness']['outputs']}
    outputs = oracle.compute(inputs, scalar_values)
    if set(outputs) != set(case['correctness']['outputs']):
        raise ValueError('Oracle output coverage mismatch')
    for name, output in outputs.items():
        values[name].copy_(output)


def prepare_ref(case_dir, sealed, case, platform=None):
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: _build_specs(fixture, config, case),
        load_ref=lambda: common.load_ref(case_dir, sealed, 'ref_deepseek_v4_prefill_moe_ep2_flash'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, 'oracle_deepseek_v4_prefill_moe_ep2_flash'),
        make_program=_make_program, make_compare=lambda: _make_compare(case_dir, case),
        run=run, oracle_golden=lambda oracle, values: _oracle_golden(oracle, values, config, case),
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: _build_specs(fixture, config, case),
        load_candidate=lambda: common.load_candidate(
            case_dir, case, candidate_workspace, 'candidate_deepseek_v4_prefill_moe_ep2_flash',
        ),
        make_program=_make_program, make_compare=lambda: _make_compare(case_dir, case), run=run,
    )
