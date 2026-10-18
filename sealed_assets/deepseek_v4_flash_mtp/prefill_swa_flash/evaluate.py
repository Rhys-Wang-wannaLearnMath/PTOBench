"""Case-local ABI and comparator boundary for deepseek_v4_flash_mtp.prefill_swa_flash."""
import json
import math
from pathlib import Path
import torch
from golden import TensorSpec, ScalarSpec, run
import common
import workflow


def _policy(case_dir, case):
    value = json.loads((Path(case_dir) / case['comparator']).read_text())
    if not isinstance(value, dict) or set(value) != {'schema_version', 'outputs'} or type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Invalid comparator top-level schema')
    outputs = value['outputs']
    if not isinstance(outputs, dict) or set(outputs) != set(case['correctness']['outputs']) or set(outputs) != {'kv_cache', 'x_out'}:
        raise ValueError('Comparator output coverage mismatch')
    fixed = common.load_module(Path(case_dir) / case['config'], 'policy_config_' + case['case_id'].replace('.', '_'))
    for name, rule in outputs.items():
        if not isinstance(rule, dict):
            raise ValueError('Comparator rule must be an object')
        if name == 'kv_cache':
            fields = {'method', 'rtol', 'atol', 'max_error_ratio'}
            if set(rule) != fields or rule['method'] != 'ratio_allclose':
                raise ValueError('Unsupported cache method or fields')
            for field in ('rtol', 'atol', 'max_error_ratio'):
                number = rule[field]
                if type(number) not in (int, float) or not math.isfinite(number) or number < 0 or (field == 'max_error_ratio' and number > 1):
                    raise ValueError('Invalid cache policy ' + field)
        else:
            fields = {'method', 'name', 'num_tokens', 'diff_thd', 'pct_thd', 'max_diff_hd'}
            if set(rule) != fields or rule['method'] != 'custom' or rule['name'] != 'valid_ratio_reldiff':
                raise ValueError('Unsupported custom method or fields')
            if type(rule['num_tokens']) is not int or (not 1 <= rule['num_tokens'] <= fixed.T or rule['num_tokens'] != fixed.NUM_TOKENS):
                raise ValueError('Invalid active prefix')
            for field in ('diff_thd', 'pct_thd', 'max_diff_hd'):
                number = rule[field]
                if type(number) not in (int, float) or not math.isfinite(number) or number < 0 or (field != 'pct_thd' and number == 0) or (field == 'pct_thd' and number > 1):
                    raise ValueError('Invalid custom policy ' + field)
    return outputs


def make_compare(case_dir, case):
    policies = _policy(case_dir, case)
    callbacks = {}
    for name, rule in policies.items():
        if name == 'kv_cache':
            def build_cache(policy):
                policy_rtol, policy_atol, ratio = policy['rtol'], policy['atol'], policy['max_error_ratio']
                def compare(actual, expected, **kwargs):
                    if actual.shape != expected.shape:
                        return False, 'cache shape mismatch'
                    a, e = actual.float(), expected.float()
                    if not torch.isfinite(a).all() or not torch.isfinite(e).all():
                        return False, 'cache contains nonfinite values'
                    bad = int(((a - e).abs() > policy_atol + policy_rtol * e.abs()).sum().item())
                    limit = round(ratio * a.numel())
                    return bad <= limit, f'cache bad={bad}, allowed={limit}'
                compare.__name__ = 'ratio_allclose'
                return compare
            callbacks[name] = build_cache(rule)
        else:
            def build_output(policy):
                prefix, diff, ratio, cap = (policy[k] for k in ('num_tokens', 'diff_thd', 'pct_thd', 'max_diff_hd'))
                def compare(actual, expected, **kwargs):
                    if actual.shape != expected.shape or actual.ndim < 1 or prefix > actual.shape[0]:
                        return False, 'x_out shape/prefix mismatch'
                    tail = int(actual[prefix:].count_nonzero().item())
                    if tail:
                        return False, f'inactive x_out tail contains {tail} nonzero values'
                    a, e = actual[:prefix].float(), expected[:prefix].float()
                    if not torch.isfinite(a).all() or not torch.isfinite(e).all():
                        return False, 'active x_out contains nonfinite values'
                    absolute = (a - e).abs()
                    denominator = torch.maximum(torch.maximum(a.abs(), e.abs()), torch.full_like(a, (1 / 2**14) / diff)) + 1e-9
                    relative = torch.where(absolute < diff, absolute, absolute / denominator)
                    bad_mask = relative > diff
                    bad = int(bad_mask.sum().item())
                    limit = round(ratio * a.numel())
                    maximum = float(relative[bad_mask].max().item()) if bad else 0.0
                    passed = bad <= limit and (bad == 0 or maximum < cap)
                    return passed, f'x_out bad={bad}, allowed={limit}, max={maximum}, strict_cap={cap}'
                compare.__name__ = 'valid_ratio_reldiff'
                return compare
            callbacks[name] = build_output(rule)
    return callbacks


def build_specs(case_dir, case, fixture):
    fixed = common.load_module(Path(case_dir) / case['config'], 'spec_config_' + case['case_id'].replace('.', '_'))
    c = fixed
    # Ordered evaluated ABI. Inputs are the canonical official Fixture tensors.
    contracts = [
        ('x_hc', [c.T, c.HC_MULT, c.D], torch.float32),
        ('hc_attn_fn', [c.MIX_HC, c.HC_DIM], torch.float32),
        ('hc_attn_scale', [3], torch.float32),
        ('hc_attn_base', [c.MIX_HC], torch.float32),
        ('attn_norm_w', [c.D], torch.bfloat16),
        ('wq_a', [c.D, c.Q_LORA], torch.bfloat16),
        ('wq_b', [c.Q_LORA, c.H * c.HEAD_DIM], torch.int8),
        ('wq_b_scale', [c.H * c.HEAD_DIM], torch.float32),
        ('wkv', [c.D, c.HEAD_DIM], torch.bfloat16),
        ('gamma_cq', [c.Q_LORA], torch.bfloat16),
        ('gamma_ckv', [c.HEAD_DIM], torch.bfloat16),
        ('freqs_cos', [c.MAX_SEQ_LEN, c.ROPE_HEAD_DIM], torch.bfloat16),
        ('freqs_sin', [c.MAX_SEQ_LEN, c.ROPE_HEAD_DIM], torch.bfloat16),
        ('kv_cache', [c.BLOCK_NUM, c.BLOCK_SIZE, 1, c.HEAD_DIM], torch.bfloat16),
        ('block_table', [c.BLOCK_NUM], torch.int32),
        ('ori_slot_mapping', [c.T], torch.int64),
        ('position_ids', [c.T], torch.int32),
        ('attn_sink', [c.H], torch.float32),
        ('wo_a', [c.O_GROUPS, c.O_LORA, c.O_GROUP_IN], torch.bfloat16),
        ('wo_b', [c.D, c.O_GROUPS * c.O_LORA], torch.int8),
        ('wo_b_scale', [c.D], torch.float32),
        ('x_out', [c.T, c.HC_MULT, c.D], torch.float32),
    ]
    specs = []
    for name, shape, dtype in contracts:
        initial = None if name == 'x_out' else torch.load(Path(fixture) / 'in' / (name + '.pt'), weights_only=True)
        if initial is not None and (list(initial.shape) != shape or initial.dtype != dtype):
            raise ValueError('Fixture tensor ABI mismatch: ' + name)
        specs.append(TensorSpec(name, shape, dtype, init_value=initial))
    scalar = torch.load(Path(fixture) / 'in/num_tokens.pt', weights_only=True)
    if scalar.dtype != torch.int32 or scalar.ndim != 0 or int(scalar) != c.NUM_TOKENS:
        raise ValueError('Fixture scalar ABI mismatch')
    specs.append(ScalarSpec('num_tokens', torch.int32, int(scalar)))
    return specs


def _program(module, case):
    return getattr(module, case['entry'])


def _oracle_golden(oracle, tensors):
    oracle.run(tensors)


def prepare_ref(case_dir, sealed, case, platform=None):
    identity = case['case_id'].replace('.', '_')
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_specs(case_dir, case, fixture),
        load_ref=lambda: common.load_ref(case_dir, sealed, 'ref_' + identity),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, 'oracle_' + identity),
        make_program=_program,
        make_compare=lambda: make_compare(case_dir, case),
        run=run, oracle_golden=_oracle_golden,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    identity = case['case_id'].replace('.', '_')
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, candidate_workspace=candidate_workspace,
        platform=platform, phase=phase,
        build_specs=lambda fixture: build_specs(case_dir, case, fixture),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, 'candidate_' + identity),
        make_program=_program, make_compare=lambda: make_compare(case_dir, case), run=run,
    )
