"""Fixed QKV-v2 ABI and Case-local public comparator execution."""
from pathlib import Path
import math

import torch
from golden import TensorSpec, run
import common

ORDER = ('x', 'wq_a', 'wq_b', 'wq_b_scale', 'wkv', 'rope_cos', 'rope_sin',
         'gamma_cq', 'gamma_ckv', 'q', 'kv', 'qr', 'qr_scale')
OUTPUTS = {'q', 'kv', 'qr', 'qr_scale'}
INPUTS = set(ORDER) - OUTPUTS


def build_tensor_specs(fixture: Path, config):
    values = {p.stem: torch.load(p, map_location='cpu', weights_only=True)
              for p in (fixture / 'in').glob('*.pt')}
    if set(values) != INPUTS:
        raise ValueError('fixed Fixture coverage mismatch')
    t, d, ql, hd, h, r = (config.T, config.D, config.Q_LORA, config.HEAD_DIM,
                          config.H, config.ROPE_DIM)
    shapes = {
        'x': (t, d), 'wq_a': (d, ql), 'wq_b': (ql, h * hd),
        'wq_b_scale': (h * hd,), 'wkv': (d, hd), 'rope_cos': (t, r),
        'rope_sin': (t, r), 'gamma_cq': (ql,), 'gamma_ckv': (hd,),
        'q': (t, h, hd), 'kv': (t, hd), 'qr': (t, ql), 'qr_scale': (t, 1),
    }
    dtypes = {n: getattr(torch, dtype) for n, dtype in
              {**config.INPUT_DTYPES, **config.OUTPUT_DTYPES}.items()}
    specs = []
    for name in ORDER:
        value = values.get(name)
        if value is not None and (tuple(value.shape) != shapes[name]
                                  or value.dtype != dtypes[name]):
            raise ValueError('Fixture shape/dtype mismatch: ' + name)
        specs.append(TensorSpec(name, list(shapes[name]), dtypes[name], init_value=value))
    return specs


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('invalid ' + name)
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError('invalid ' + name) from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError('invalid ' + name)
    return number


def _ratio_callback(policy_rtol, policy_atol, policy_ratio):
    # Runner fallback rtol/atol keywords cannot replace these lexical JSON values.
    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, 'shape mismatch'
        actual_fp32, expected_fp32 = actual.float(), expected.float()
        if not (torch.isfinite(actual_fp32).all() and torch.isfinite(expected_fp32).all()):
            return False, 'nonfinite actual or expected'
        tolerance = policy_atol + policy_rtol * expected_fp32.abs()
        mismatches = int(((actual_fp32 - expected_fp32).abs() > tolerance).sum())
        allowed = round(policy_ratio * actual_fp32.numel())
        return mismatches <= allowed, f'mismatches={mismatches}, allowed={allowed}'
    return compare


def _callbacks(policy, names):
    if (not isinstance(policy, dict) or set(policy) != {'schema_version', 'outputs'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1):
        raise ValueError('invalid comparator schema')
    if (len(names) != len(set(names)) or not isinstance(policy['outputs'], dict)
            or set(policy['outputs']) != set(names)):
        raise ValueError('output coverage mismatch')
    callbacks = {}
    for name in names:
        rule = policy['outputs'][name]
        if (not isinstance(rule, dict) or set(rule) != {'method', 'rtol', 'atol', 'max_error_ratio'}
                or rule['method'] != 'ratio_allclose'):
            raise ValueError('invalid ratio rule')
        rt, at, ratio = (_number(rule[key], name + '.' + key)
                         for key in ('rtol', 'atol', 'max_error_ratio'))
        if ratio > 1:
            raise ValueError('invalid error ratio')
        callbacks[name] = _ratio_callback(rt, at, ratio)
    return callbacks


def _compare(case_dir, case):
    return _callbacks(common.read_json(case_dir / case['comparator']), case['correctness']['outputs'])


def _program(module, case):
    return getattr(module, case['entry'])


def _oracle(oracle, values):
    outputs = oracle.compute({name: values[name] for name in INPUTS}, {})
    if set(outputs) != OUTPUTS:
        raise ValueError('Oracle output coverage mismatch')
    for name, value in outputs.items():
        values[name][:] = value


def _build_specs(case_dir, case):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_evaluation_config')
    return lambda fixture: build_tensor_specs(fixture, config)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case['case_id'].replace('.', '_')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', identity + '_workflow')
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=_build_specs(case_dir, case),
        load_ref=lambda: common.load_ref(case_dir, sealed, identity + '_ref'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + '_oracle'),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=run, oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    identity = case['case_id'].replace('.', '_')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', identity + '_workflow')
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase, build_specs=_build_specs(case_dir, case),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + '_candidate'),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=run,
    )
