"""Case-owned preserved whole-output MoE comparison and public ABI."""
import math
from pathlib import Path
import torch
from golden import TensorSpec, ScalarSpec, run
import common

ORDER = ('x_hc', 'hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base', 'norm_w', 'gate_w', 'gate_bias',
         'tid2eid', 'input_ids', 'routed_w1', 'routed_w1_scale', 'routed_w3', 'routed_w3_scale',
         'routed_w2', 'routed_w2_scale', 'shared_w1', 'shared_w1_scale', 'shared_w3',
         'shared_w3_scale', 'shared_w2', 'shared_w2_scale', 'x_next', 'layer_id')
OUTPUTS = ('x_next',)


def build_tensor_specs(fixture: Path):
    # Resolve the owning public contract, independently of caller CWD/module cache.
    identity = fixture.parents[1].name
    case_dir = common.BENCHMARK_ROOT / 'cases' / 'deepseek_v4' / identity
    config = common.load_module(case_dir / 'public/config.py', identity + '_spec_config')
    specs = []
    for name in ORDER:
        dtype = getattr(torch, getattr(config, name.upper() + '_DTYPE'))
        if name == 'x_next':
            specs.append(TensorSpec(name, list(config.X_NEXT_SHAPE), dtype))
        else:
            initial = torch.load(fixture / 'in' / (name + '.pt'), weights_only=True, map_location='cpu')
            if initial.dtype != dtype:
                raise ValueError('fixture dtype mismatch: ' + name)
            if name == 'layer_id':
                if initial.ndim != 0 or initial.item() != config.LAYER_ID:
                    raise ValueError('fixture scalar mismatch: ' + name)
                specs.append(ScalarSpec(name, dtype, int(initial.item())))
            else:
                shape = getattr(config, name.upper() + '_SHAPE')
                if tuple(initial.shape) != tuple(shape):
                    raise ValueError('fixture shape mismatch: ' + name)
                specs.append(TensorSpec(name, list(shape), dtype, init_value=initial))
    return specs


def _number(value, name, *, positive=False, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('invalid ' + name)
    if value < 0 or (positive and value == 0) or (upper is not None and value > upper):
        raise ValueError('invalid ' + name)
    return float(value)


def _compare(case_dir, case):
    policy = common.read_json(case_dir / case['comparator'])
    declared = case['correctness']['outputs']
    if (not isinstance(policy, dict) or set(policy) != {'schema_version', 'outputs'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1
            or declared != list(OUTPUTS) or not isinstance(policy['outputs'], dict)
            or set(policy['outputs']) != set(declared)):
        raise ValueError('invalid comparator schema or outputs')
    rule = policy['outputs']['x_next']
    fields = {'method', 'name', 'diff_thd', 'pct_thd', 'max_diff_hd', 'floor_numerator',
              'denominator_epsilon', 'nonfinite_policy'}
    if (not isinstance(rule, dict) or set(rule) != fields or rule['method'] != 'custom'
            or rule['name'] != 'ratio_reldiff' or rule['nonfinite_policy'] != 'reject_either'):
        raise ValueError('invalid MoE comparator rule')
    diff = _number(rule['diff_thd'], 'diff_thd', positive=True)
    pct = _number(rule['pct_thd'], 'pct_thd', upper=1)
    floor = _number(rule['floor_numerator'], 'floor_numerator', positive=True)
    eps = _number(rule['denominator_epsilon'], 'denominator_epsilon')
    cap = None if rule['max_diff_hd'] is None else _number(rule['max_diff_hd'], 'max_diff_hd', positive=True)

    def compare(actual, expected, **kwargs):
        del kwargs
        if actual.shape != expected.shape:
            return False, 'shape mismatch'
        a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
        if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()):
            return False, 'nonfinite value'
        absolute = (a - e).abs()
        denominator = torch.maximum(torch.maximum(a.abs(), e.abs()), torch.full_like(a, floor / diff)) + eps
        relative = torch.where(absolute < diff, absolute, absolute / denominator)
        bad = relative > diff
        count = int(bad.sum())
        allowed = round(pct * a.numel())
        worst = float(relative[bad].max()) if count else 0.
        passed = count <= allowed and (cap is None or worst < cap)
        return passed, f'ratio_reldiff: mismatches={count}/{a.numel()}, allowed={allowed}, worst={worst}'
    return {'x_next': compare}


def _program(module, case):
    return getattr(module, case['entry'])


def _oracle(oracle, values):
    inputs = {n: v for n, v in values.items() if n not in ('x_next', 'layer_id')}
    layer_id = int(values['layer_id'])
    for name, output in oracle.compute(inputs, {'layer_id': layer_id}).items():
        values[name][:] = output


def _run(**kwargs):
    return run(rtol=0., atol=0., save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case['case_id'].replace('.', '_')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', identity + '_workflow')
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=build_tensor_specs, load_ref=lambda: common.load_ref(case_dir, sealed, identity + '_ref'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + '_oracle'), make_program=_program,
        make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    identity = case['case_id'].replace('.', '_')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', identity + '_workflow')
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=build_tensor_specs,
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + '_candidate'),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
