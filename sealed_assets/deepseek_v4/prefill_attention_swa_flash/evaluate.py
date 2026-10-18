"""Case-local Fixture ABI and official single-card SWA comparison policy."""
import json
import math

import torch
from golden import ScalarSpec, TensorSpec, run

import common

PARAM_ORDER = (
    'x_hc', 'hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'attn_norm_w',
    'wq_a', 'wq_b', 'wq_b_scale', 'wkv', 'gamma_cq', 'gamma_ckv',
    'freqs_cos', 'freqs_sin', 'kv_cache', 'block_table', 'ori_slot_mapping',
    'cmp_sparse_indices', 'cmp_sparse_lens', 'position_ids', 'attn_sink',
    'wo_a', 'wo_b', 'wo_b_scale', 'x_out', 'num_tokens',
)


def build_tensor_specs(fixture, config):
    values = {p.stem: torch.load(p, map_location='cpu', weights_only=True)
              for p in (fixture / 'in').glob('*.pt')}
    if set(values) != set(PARAM_ORDER) - {'x_out'}:
        raise ValueError('Fixture input coverage does not match SWA ABI')
    specs = []
    for name in PARAM_ORDER:
        if name == 'x_out':
            specs.append(TensorSpec(name, [config.T, config.HC_MULT, config.D], torch.bfloat16))
        elif name == 'num_tokens':
            value = values[name]
            if value.ndim != 0 or value.dtype != torch.int32 or value.item() != config.NUM_TOKENS:
                raise ValueError('num_tokens must match the fixed INT32 scalar contract')
            specs.append(ScalarSpec(name, torch.int32, value))
        else:
            value = values[name]
            specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    return specs


def _number(value, field, ratio=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0 or (ratio and value > 1)):
        raise ValueError('Invalid comparator ' + field)
    return float(value)


def _ratio_allclose(settings):
    if set(settings) != {'method', 'rtol', 'atol', 'max_error_ratio'}:
        raise ValueError('Invalid ratio_allclose fields')
    rtol = _number(settings['rtol'], 'rtol')
    atol = _number(settings['atol'], 'atol')
    ratio = _number(settings['max_error_ratio'], 'max_error_ratio', ratio=True)

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, 'shape mismatch'
        a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(e).all()):
            return False, 'nonfinite values'
        count = int(((a - e).abs() > atol + rtol * e.abs()).sum().item())
        allowed = round(ratio * a.numel())
        return count <= allowed, f'ratio_allclose: mismatches={count}/{a.numel()}, allowed={allowed}'

    compare.__name__ = 'ratio_allclose'
    return compare


def _valid_ratio_reldiff(settings):
    required = {'method', 'name', 'valid_rows', 'valid_axis', 'diff_thd',
                'pct_thd', 'max_diff_hd', 'zero_tail'}
    if set(settings) != required or settings['name'] != 'valid_ratio_reldiff':
        raise ValueError('Invalid valid_ratio_reldiff fields')
    rows, axis = settings['valid_rows'], settings['valid_axis']
    if type(rows) is not int or rows < 0 or type(axis) is not int or axis < 0:
        raise ValueError('Invalid valid prefix')
    if type(settings['zero_tail']) is not bool:
        raise ValueError('zero_tail must be a boolean')
    zero_tail = settings['zero_tail']
    diff = _number(settings['diff_thd'], 'diff_thd')
    pct = _number(settings['pct_thd'], 'pct_thd', ratio=True)
    cap = _number(settings['max_diff_hd'], 'max_diff_hd')
    if diff == 0 or cap == 0:
        raise ValueError('diff_thd and max_diff_hd must be positive')

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, 'shape mismatch'
        if axis >= actual.ndim or rows > actual.shape[axis]:
            return False, 'valid prefix outside tensor'
        if zero_tail and bool(actual.narrow(axis, rows, actual.shape[axis] - rows).count_nonzero()):
            return False, 'inactive tail is nonzero'
        a = actual.narrow(axis, 0, rows).detach().cpu().float()
        e = expected.narrow(axis, 0, rows).detach().cpu().float()
        if a.numel() == 0:
            return True, ''
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(e).all()):
            return False, 'nonfinite values'
        absolute = (a - e).abs()
        floor = torch.full_like(a, (1. / (1 << 14)) / diff)
        denominator = torch.maximum(torch.maximum(a.abs(), e.abs()), floor) + 1e-9
        relative = torch.where(absolute < diff, absolute, absolute / denominator)
        bad = relative > diff
        count = int(bad.sum().item())
        allowed = round(pct * a.numel())
        worst = float(relative[bad].max().item()) if count else 0.
        return count <= allowed and worst < cap, (
            f'valid_ratio_reldiff: mismatches={count}/{a.numel()}, allowed={allowed}, worst={worst}, cap={cap}')

    compare.__name__ = 'valid_ratio_reldiff'
    return compare


def _callbacks(policy, output_names, output_shapes=None):
    if (not isinstance(policy, dict) or set(policy) != {'schema_version', 'outputs'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1
            or not isinstance(policy['outputs'], dict) or set(policy['outputs']) != set(output_names)):
        raise ValueError('Comparator schema/output coverage mismatch')
    callbacks = {}
    for name in output_names:
        settings = policy['outputs'][name]
        if not isinstance(settings, dict):
            raise ValueError('Comparator must be an object')
        if settings.get('method') == 'ratio_allclose':
            callbacks[name] = _ratio_allclose(settings)
        elif settings.get('method') == 'custom':
            callbacks[name] = _valid_ratio_reldiff(settings)
        else:
            raise ValueError('Unsupported comparator method')
        if output_shapes is not None:
            shape = tuple(output_shapes[name])
            if settings['method'] == 'custom':
                axis, rows = settings['valid_axis'], settings['valid_rows']
                if axis >= len(shape) or rows > shape[axis]:
                    raise ValueError('Comparator valid prefix outside fixed output shape')
            callbacks[name] = _fixed_shape(callbacks[name], shape)
    return callbacks


def _fixed_shape(callback, shape):
    def compare(actual, expected, **kwargs):
        if tuple(actual.shape) != shape or tuple(expected.shape) != shape:
            return False, 'fixed output shape mismatch'
        return callback(actual, expected, **kwargs)

    compare.__name__ = callback.__name__
    return compare


def _compare(case_dir, case):
    identity = case['case_id'].replace('.', '_')
    config = common.load_module(case_dir / case['config'], identity + '_comparator_config')
    shapes = {
        'x_out': (config.T, config.HC_MULT, config.D),
        'kv_cache': (config.PREFILL_ORI_BLOCK_NUM, config.BLOCK_SIZE, 1, config.HEAD_DIM),
    }
    return _callbacks(json.loads((case_dir / case['comparator']).read_text()),
                      case['correctness']['outputs'], shapes)


def _program(module, case):
    return getattr(module, case['entry'])


def _oracle(oracle, values):
    inputs = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
    scalars = {k: v for k, v in values.items() if not isinstance(v, torch.Tensor)}
    for name, answer in oracle.compute(inputs, scalars).items():
        values[name].copy_(answer)


def _run(**kwargs):
    return run(rtol=0., atol=0., save_data=False, **kwargs)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case['case_id'].replace('.', '_')
    wf = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', identity + '_workflow')
    config = common.load_module(case_dir / case['config'], identity + '_config')
    return wf.prepare_ref(case_dir=case_dir, sealed=sealed, case=case, platform=platform,
                          build_specs=lambda f: build_tensor_specs(f, config),
                          load_ref=lambda: common.load_ref(case_dir, sealed, identity + '_ref'),
                          load_oracle=lambda: common.load_oracle(case_dir, sealed, identity + '_oracle'),
                          make_program=_program, make_compare=lambda: _compare(case_dir, case),
                          run=_run, oracle_golden=_oracle)


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    identity = case['case_id'].replace('.', '_')
    wf = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', identity + '_workflow')
    config = common.load_module(case_dir / case['config'], identity + '_config')
    return wf.evaluate_candidate(case_dir=case_dir, sealed=sealed, case=case,
                                candidate_workspace=candidate_workspace, platform=platform, phase=phase,
                                build_specs=lambda f: build_tensor_specs(f, config),
                                load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace,
                                                                             identity + '_candidate'),
                                make_program=_program, make_compare=lambda: _compare(case_dir, case), run=_run)
