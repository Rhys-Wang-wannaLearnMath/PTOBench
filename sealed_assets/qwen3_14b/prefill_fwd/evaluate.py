"""Case-local evaluator for the fixed two-layer Qwen3-14B packed prefill."""
import json
import math

import torch
from golden import TensorSpec, run

import common


def _config(case_dir, case):
    return common.load_module(case_dir / case['config'], 'qwen3_14b_prefill_fwd_fixed_config')


def build_tensor_specs(fixture, config):
    values = {p.stem: torch.load(p, weights_only=True, map_location='cpu')
              for p in (fixture / 'in').glob('*.pt')}
    layers, batch = config.CASE_NUM_LAYERS, config.CASE_BATCH
    if 'chunk_lens' not in values:
        raise ValueError('missing packed sequence metadata')
    tokens = int(values['chunk_lens'].sum())
    layout = [
        ('hidden_states', (tokens, config.HIDDEN), torch.bfloat16),
        ('seq_lens', (batch,), torch.int32),
        ('chunk_lens', (batch,), torch.int32),
        ('chunk_offsets', (batch,), torch.int32),
        ('input_rms_weight', (layers, config.HIDDEN), torch.float32),
        ('wq', (layers * config.HIDDEN, config.HIDDEN), torch.bfloat16),
        ('wk', (layers * config.HIDDEN, config.KV_HIDDEN), torch.bfloat16),
        ('wv', (layers * config.HIDDEN, config.KV_HIDDEN), torch.bfloat16),
        ('q_norm_weight', (layers, config.HEAD_DIM), torch.float32),
        ('k_norm_weight', (layers, config.HEAD_DIM), torch.float32),
        ('rope_cos', (config.MAX_SEQ, config.HEAD_DIM), torch.float32),
        ('rope_sin', (config.MAX_SEQ, config.HEAD_DIM), torch.float32),
        ('block_table', (config.CASE_NUM_BLOCKS,), torch.int32),
        ('slot_mapping', (tokens,), torch.int32),
        ('k_cache', (config.CASE_CACHE_ROWS, config.HEAD_DIM), torch.bfloat16),
        ('v_cache', (config.CASE_CACHE_ROWS, config.HEAD_DIM), torch.bfloat16),
        ('wo', (layers * config.HIDDEN, config.HIDDEN), torch.bfloat16),
        ('post_rms_weight', (layers, config.HIDDEN), torch.float32),
        ('w_gate', (layers * config.HIDDEN, config.INTERMEDIATE), torch.bfloat16),
        ('w_up', (layers * config.HIDDEN, config.INTERMEDIATE), torch.bfloat16),
        ('w_down', (layers * config.INTERMEDIATE, config.HIDDEN), torch.bfloat16),
        ('final_norm_weight', (1, config.HIDDEN), torch.float32),
        ('lm_head_weight', (config.VOCAB, config.HIDDEN), torch.bfloat16),
        ('out', (batch, config.VOCAB), torch.float32),
    ]
    if set(values) != {name for name, _, _ in layout if name != 'out'}:
        raise ValueError('Fixture input coverage mismatch')
    specs = []
    for name, shape, dtype in layout:
        if name == 'out':
            specs.append(TensorSpec(name, shape, dtype))
        else:
            value = values[name]
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f'Fixture input shape/dtype mismatch: {name}')
            specs.append(TensorSpec(name, shape, dtype, init_value=value))
    return specs


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('comparator tolerance must be numeric')
    if not math.isfinite(value) or value < 0:
        raise ValueError('comparator tolerance must be finite and nonnegative')
    return float(value)


def _compare(case_dir, case):
    policy = json.loads((case_dir / case['comparator']).read_text())
    if (not isinstance(policy, dict) or set(policy) != {'schema_version', 'outputs'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1
            or case['correctness']['outputs'] != ['out']
            or not isinstance(policy['outputs'], dict) or set(policy['outputs']) != {'out'}):
        raise ValueError('comparator schema/output coverage mismatch')
    settings = policy['outputs']['out']
    if (not isinstance(settings, dict) or set(settings) != {'method', 'rtol', 'atol'}
            or settings['method'] != 'allclose'):
        raise ValueError('unsupported comparator method/fields')
    policy_rtol, policy_atol = _number(settings['rtol']), _number(settings['atol'])

    def compare(actual, expected, **_):
        if actual.shape != expected.shape:
            return False, 'allclose: shape mismatch'
        passed = bool(torch.allclose(actual.cpu(), expected.cpu(),
                                     rtol=policy_rtol, atol=policy_atol, equal_nan=False))
        return passed, f'allclose(rtol={policy_rtol}, atol={policy_atol})'

    compare.__name__ = 'allclose'
    return {'out': compare}


def _oracle(oracle, values):
    inputs = {name: value for name, value in values.items() if name != 'out'}
    values['out'].copy_(oracle.compute(inputs, {})['out'])


def _program(module, case):
    return getattr(module, case['entry'])


def prepare_ref(case_dir, sealed, case, platform=None):
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', 'qwen3_14b_prefill_fwd_workflow')
    config = _config(case_dir, case)
    return workflow.prepare_ref(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_ref=lambda: common.load_ref(case_dir, sealed, 'qwen3_14b_prefill_fwd_ref'),
        load_oracle=lambda: common.load_oracle(case_dir, sealed, 'qwen3_14b_prefill_fwd_oracle'),
        make_program=_program, make_compare=lambda: _compare(case_dir, case),
        run=run, oracle_golden=_oracle,
    )


def evaluate_candidate(case_dir, sealed, case, candidate_workspace, platform=None, phase='all'):
    workflow = common.load_module(common.BENCHMARK_ROOT / 'evaluator/workflow.py', 'qwen3_14b_prefill_fwd_workflow')
    config = _config(case_dir, case)
    return workflow.evaluate_candidate(
        case_dir=case_dir, sealed=sealed, case=case, platform=platform,
        candidate_workspace=candidate_workspace, phase=phase,
        build_specs=lambda fixture: build_tensor_specs(fixture, config),
        load_candidate=lambda: common.load_candidate(case_dir, case, candidate_workspace, 'qwen3_14b_prefill_fwd_candidate'),
        make_program=_program, make_compare=lambda: _compare(case_dir, case), run=run,
    )
