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


def build_tensor_specs(fixture: Path, config):
    ranks, tokens, hidden = config.N_RANKS, config.T, config.D
    inter, local = config.MOE_INTER, config.N_LOCAL
    tensor_contract = {
        'x_hc': ([ranks, tokens, config.HC_MULT, hidden], torch.bfloat16),
        'hc_ffn_fn': ([ranks, config.MIX_HC, config.HC_DIM], torch.float32),
        'hc_ffn_scale': ([ranks, 3], torch.float32),
        'hc_ffn_base': ([ranks, config.MIX_HC], torch.float32),
        'norm_w': ([ranks, hidden], torch.bfloat16),
        'gate_w': ([ranks, config.N_EXPERTS_GLOBAL, hidden], torch.float32),
        'gate_bias': ([ranks, config.N_EXPERTS_GLOBAL], torch.float32),
        'tid2eid': ([ranks, config.VOCAB, config.TOPK], torch.int32),
        'input_ids': ([ranks, tokens], torch.int64),
        'routed_w1': ([ranks, local, inter, hidden], torch.int8),
        'routed_w1_scale': ([ranks, local, inter], torch.float32),
        'routed_w3': ([ranks, local, inter, hidden], torch.int8),
        'routed_w3_scale': ([ranks, local, inter], torch.float32),
        'routed_w2': ([ranks, local, hidden, inter], torch.int8),
        'routed_w2_scale': ([ranks, local, hidden], torch.float32),
        'shared_w1': ([ranks, inter, hidden], torch.int8),
        'shared_w1_scale': ([ranks, inter], torch.float32),
        'shared_w3': ([ranks, inter, hidden], torch.int8),
        'shared_w3_scale': ([ranks, inter], torch.float32),
        'shared_w2': ([ranks, hidden, inter], torch.int8),
        'shared_w2_scale': ([ranks, hidden], torch.float32),
        'x_next': ([ranks, tokens, config.HC_MULT, hidden], torch.bfloat16),
    }
    specs = []
    for name in ORDER:
        if name == 'layer_id':
            value = torch.load(fixture / 'in/layer_id.pt', weights_only=True, map_location='cpu')
            if value.ndim != 0 or value.dtype != torch.int32 or value.item() != config.LAYER_ID:
                raise ValueError('Fixture layer_id does not match public config')
            specs.append(ScalarSpec(name, torch.int32, config.LAYER_ID))
        else:
            shape, dtype = tensor_contract[name]
            if name == 'x_next':
                specs.append(TensorSpec(name, shape, dtype))
            else:
                initial = torch.load(fixture / 'in' / (name + '.pt'), weights_only=True, map_location='cpu')
                if list(initial.shape) != shape or initial.dtype != dtype or not initial.is_contiguous():
                    raise ValueError('Fixture tensor does not match public config: ' + name)
                specs.append(TensorSpec(name, shape, dtype, init_value=initial))
    return specs


def _number(value, name, *, positive=False, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('invalid ' + name)
    if value < 0 or (positive and value == 0) or (upper is not None and value > upper):
        raise ValueError('invalid ' + name)
    return float(value)


def _compare(case_dir, case):
    policy = common.read_json(case_dir / case['comparator'])
    if (case['correctness']['outputs'] != list(OUTPUTS)
            or not isinstance(policy, dict)
            or set(policy) != {'schema_version', 'outputs'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1
            or not isinstance(policy['outputs'], dict)
            or set(policy['outputs']) != set(OUTPUTS)):
        raise ValueError('invalid comparator schema or outputs')
    rule = policy['outputs']['x_next']
    fields = {'method', 'name', 'diff_thd', 'pct_thd', 'max_diff_hd', 'floor_numerator',
              'denominator_epsilon', 'nonfinite_policy'}
    if (not isinstance(rule, dict) or set(rule) != fields or rule['method'] != 'custom' or rule['name'] != 'ratio_reldiff'
            or rule['nonfinite_policy'] != 'reject_either'):
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
    config = common.load_module(case_dir / case['config'], identity + '_evaluator_config')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', identity + '_workflow')
    return workflow.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config), load_ref=lambda: common.load_ref(case_dir, sealed, identity + '_ref'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + '_oracle'), make_program=_program,
        make_compare=lambda: _compare(case_dir, case), run=_run, oracle_golden=_oracle)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_evaluator_config')
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', identity + '_workflow')
    return workflow.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case,
        candidate_workspace=candidate_workspace, platform=platform, phase=phase, build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, identity + '_candidate'),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
