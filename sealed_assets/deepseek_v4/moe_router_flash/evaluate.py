"""Case-local standalone MoE router evaluator and fixed official comparison."""
from pathlib import Path
import math
import torch
from golden import ScalarSpec, TensorSpec, run
import common

ORDER = ('x_mixed', 'norm_w', 'gate_w', 'gate_bias', 'layer_id', 'tid2eid', 'input_ids', 'x_norm', 'indices', 'weights')
OUTPUTS = {'x_norm', 'indices', 'weights'}
INPUTS = set(ORDER) - OUTPUTS


def build_tensor_specs(fixture: Path, config):
    values = {p.stem: torch.load(p, map_location='cpu', weights_only=True)
              for p in (fixture / 'in').glob('*.pt')}
    if set(values) != INPUTS:
        raise ValueError('fixed Fixture input coverage mismatch')
    abi = {
        'x_mixed': ([config.B, config.S, config.D], torch.bfloat16),
        'norm_w': ([config.D], torch.float32),
        'gate_w': ([config.N_EXPERTS, config.D], torch.float32),
        'gate_bias': ([config.N_EXPERTS], torch.float32),
        'layer_id': ([], torch.int32),
        'tid2eid': ([config.VOCAB, config.TOPK], torch.int32),
        'input_ids': ([config.B, config.S], torch.int64),
        'x_norm': ([config.T, config.D], torch.bfloat16),
        'indices': ([config.T, config.TOPK], torch.int32),
        'weights': ([config.T, config.TOPK], torch.float32),
    }
    specs = []
    for name in ORDER:
        shape, dtype = abi[name]
        if name in INPUTS:
            value = values[name]
            if list(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError('fixed Fixture ABI mismatch: ' + name)
        if name == 'layer_id':
            if int(values[name]) != config.LAYER_ID:
                raise ValueError('fixed layer_id mismatch')
            specs.append(ScalarSpec(name, dtype, values[name]))
        elif name in OUTPUTS:
            specs.append(TensorSpec(name, shape, dtype))
        else:
            specs.append(TensorSpec(name, shape, dtype, init_value=values[name]))
    return specs


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError('invalid ' + field)
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError('invalid ' + field) from exc
    if not math.isfinite(number):
        raise ValueError('invalid ' + field)
    return number


def _callbacks(policy, names):
    if not isinstance(policy, dict) or set(policy) != {'schema_version', 'outputs'} or type(policy['schema_version']) is not int or policy['schema_version'] != 1:
        raise ValueError('invalid comparator schema')
    if not isinstance(policy['outputs'], dict) or set(policy['outputs']) != set(names):
        raise ValueError('comparator output coverage mismatch')
    callbacks = {}
    for name in names:
        rule = policy['outputs'][name]
        if not isinstance(rule, dict) or set(rule) != {'method', 'rtol', 'atol'} or rule['method'] != 'allclose':
            raise ValueError('invalid allclose rule for ' + name)
        rtol, atol = _number(rule['rtol'], name + '.rtol'), _number(rule['atol'], name + '.atol')
        def compare(actual, expected, *, _rtol=rtol, _atol=atol, **kwargs):
            del kwargs
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                return False, 'shape or dtype mismatch'
            passed = torch.allclose(actual.detach().cpu(), expected.detach().cpu(), rtol=_rtol, atol=_atol)
            return bool(passed), f'allclose: rtol={_rtol}, atol={_atol}'
        callbacks[name] = compare
    return callbacks


def _compare(case_dir, case):
    names = case['correctness']['outputs']
    if len(names) != len(OUTPUTS) or set(names) != OUTPUTS:
        raise ValueError('Case correctness output coverage mismatch')
    return _callbacks(common.read_json(case_dir / case['comparator']), case['correctness']['outputs'])

def _program(module, case):
    return getattr(module, case['entry'])

def _oracle(oracle, values):
    for name, value in oracle.compute(values, {}).items():
        values[name][:] = value

def _run(**kwargs):
    return run(rtol=0., atol=0., save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_config')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', 'workflow_' + identity)
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform, build_specs=lambda fixture: build_tensor_specs(fixture, config), load_ref=lambda: common.load_ref(case_dir, sealed, identity + '_ref'), load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + '_oracle'), make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_config')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', 'workflow_' + identity)
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=lambda fixture: build_tensor_specs(fixture, config), load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + '_candidate'), make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
