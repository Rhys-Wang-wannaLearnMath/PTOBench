"""Conservative public tuning contract, separate from runtime capability.

An installed parameter is not automatically an Agent permission. Pure metadata
and configuration inspection here never compile or launch an NPU program.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

DEFAULTS = {'memory_planner': 'PYPTO', 'analyze_auto_scopes_for_deps': False}
MANAGED_DEFAULTS = {'strategy': 'Default', 'ring_task_window': None, 'ring_heap': None,
                    'ring_dep_pool': None, 'aicpu_thread_num': None}
REASONS = {
    'strategy': 'Read-only Default; the installed runtime has no alternative registered pipeline.',
    'pass_options': 'Arbitrary pass modifications are not a supported public performance control.',
    'ring_task_window': 'Inherited; public exposure requires per-Case transport/safety review and default-configuration acceptance.',
    'ring_heap': 'Evaluator-managed allocation; unrestricted sizes can exhaust device memory.',
    'ring_dep_pool': 'Evaluator-managed dependency capacity; undersizing can break scheduling.',
    'aicpu_thread_num': 'Evaluator-managed threads; distributed dispatch reads DistributedConfig, not this override.',
}
DIAGNOSTICS = ('dump_passes', 'dump_ptoas_passes', 'compile_profiling', 'enable_pmu',
               'enable_chip_swimlane', 'enable_dep_gen', 'enable_scope_stats', 'enable_dump_args',
               'diagnostic_phase', 'disabled_diagnostics')

POLICY_PATH = Path(__file__).with_name('tuning_policies.json')


@lru_cache(maxsize=4)
def _policies(path, modified, size):
    del modified, size
    return json.loads(Path(path).read_text())['cases']


def policy(case=None):
    if case is not None and POLICY_PATH.is_file():
        stat = POLICY_PATH.stat()
        cases = _policies(str(POLICY_PATH), stat.st_mtime_ns, stat.st_size)
        if case['case_id'] in cases:
            return cases[case['case_id']]
        # CPU fixtures / external deployments retain the established interface.
        # A canonical Case cannot acquire new controls without its inventory.
    return {'case_id': case.get('case_id') if case else None,
            'defaults': dict(DEFAULTS), 'options': {
                'memory_planner': {'editable': True, 'default': 'PYPTO', 'stage': 'compile',
                                   'values': ['PYPTO', 'DSA_RP', 'PTOAS']},
                'analyze_auto_scopes_for_deps': {'editable': True, 'default': False,
                                                'stage': 'compile', 'values': [False, True]}}}


def defaults(case=None):
    return dict(policy(case)['defaults'])


def validate(value, case=None):
    if not isinstance(value, dict):
        raise ValueError('tuning.json must contain an object')
    selected = policy(case)
    options = selected['options']
    for name in value:
        if name in REASONS and name not in options:
            raise ValueError(f'{name} is not Agent-configurable: {REASONS[name]}')
        if name in DIAGNOSTICS:
            raise ValueError(f'{name} is controlled by profile, not tuning.json')
    unknown = set(value) - set(options)
    if unknown:
        raise ValueError(f"unknown tuning option(s): {', '.join(sorted(unknown))}")
    result = {**selected['defaults'], **value}
    if type(result['memory_planner']) is not str or result['memory_planner'] not in ('PYPTO', 'DSA_RP', 'PTOAS'):
        raise ValueError('memory_planner must be one of: DSA_RP, PYPTO, PTOAS')
    if type(result['analyze_auto_scopes_for_deps']) is not bool:
        raise ValueError('analyze_auto_scopes_for_deps must be true or false')
    for name, actual in result.items():
        if not any(type(actual) is type(allowed) and actual == allowed for allowed in options[name]['values']):
            raise ValueError(f"{name} is not supported by this Case policy; allowed: {options[name]['values']}")
    return result


def read(workspace, case=None):
    import os
    import file_tree
    workspace = Path(workspace)
    path = workspace / 'candidate' / 'tuning.json'
    # Read legacy externally supplied workspaces while all newly created
    # PTOBench workspaces use candidate/tuning.json.
    if not path.exists() and (workspace / 'tuning.json').exists():
        path = workspace / 'tuning.json'
    try:
        with file_tree.open_file(path) as descriptor:
            if os.fstat(descriptor).st_size > 65536:
                raise ValueError('tuning.json exceeds 64 KiB')
            with os.fdopen(os.dup(descriptor), 'r', encoding='utf-8') as stream:
                def unique(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError(f'duplicate tuning option: {key}')
                        result[key] = value
                    return result
                value = json.load(stream, object_pairs_hook=unique)
    except FileNotFoundError:
        value = {}
    except (OSError, UnicodeError, json.JSONDecodeError, file_tree.UnsafeTreeError) as error:
        raise ValueError(f'invalid tuning.json: {error}') from error
    return validate(value, case)


def capabilities(case, profile):
    inherited = {**profile.get('runtime', {}), **case.get('runtime', {})}
    options = {name: dict(spec) for name, spec in policy(case)['options'].items()}
    for name, default in MANAGED_DEFAULTS.items():
        if name in options:
            continue
        options[name] = {'editable': False, 'default': default, 'reason': REASONS[name],
                         'stage': 'compile' if name == 'strategy' else 'dispatch'}
        if name != 'strategy':
            options[name]['inherited_value'] = inherited.get(name)
            options[name]['inheritance'] = 'Case runtime > environment runtime > compiled/runtime default'
    return {'schema_version': 2, 'case_id': case['case_id'], 'options': options,
            'baseline': defaults(case), 'hardware_tested_in_this_change': False,
            'diagnostics': {'editable_in_tuning': False, 'interface': 'profile runtime/pmu/compiler',
                            'inherited_case_settings': {k: v for k, v in inherited.items() if k in DIAGNOSTICS}},
            'note': 'Editable options are optimization permissions, not a hardware PASS or speedup claim for every value. Baseline acceptance verifies the Case default configuration; final Candidate acceptance uses its frozen settings and requires successful evaluate. Unlisted scheduling fields preserve Case/environment/native defaults.'}


def agent_capabilities(case, profile, baseline, current_identity=None):
    """Add Agent guidance and measured default status to the static permissions.

    Keep capabilities() independent of measurement results: its stable contents
    are already part of the published baseline identity.
    """
    result = capabilities(case, profile)
    guidance = {
        'memory_planner': (
            'Choose local tile-buffer allocation and lifetime reuse.',
            'Applies to compiled local buffers; a different planner must pass evaluate.'),
        'analyze_auto_scopes_for_deps': (
            'Analyze dependencies in automatically generated orchestration scopes.',
            'Applies to reachable AUTO scopes; manual regions are skipped.'),
        'ring_task_window': (
            'Set the number of in-flight task slots per runtime ring.',
            'Task-producing a2a3 orchestration only. Allowed sizes preserve or double the default; heaps and dependency pools are unchanged.'),
        'aicpu_thread_num': (
            'Set the AICPU thread count used by orchestration and scheduling.',
            'Reviewed single-card paths only. Distributed thread settings remain managed.'),
    }
    for name, option in result['options'].items():
        if option['editable']:
            option['type'] = {str: 'string', bool: 'boolean', int: 'integer'}[type(option['default'])]
            option.setdefault('unit', 'not applicable')
            option['description'], option['restrictions'] = guidance[name]
    configuration = defaults(case)
    fixtures = baseline.get('fixtures', {})
    rounds = baseline.get('iterations')
    verified = bool(
        baseline.get('status') == 'ready'
        and baseline.get('compiler_tuning') == configuration
        and (current_identity is None or baseline.get('tuning_identity') in (None, current_identity))
        and type(rounds) is int and rounds >= 3
        and fixtures and set(fixtures) == set(case.get('fixture_ids', []))
        and all(len(run.get('samples_effective_us', [])) == rounds
                and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                        for v in run['samples_effective_us']) for run in fixtures.values()))
    result['hardware_tested_in_this_change'] = verified
    result['default_configuration_validation'] = {
        'status': 'passed' if verified else 'unverified',
        'scope': 'Registered starting implementation with the complete baseline default configuration.',
        'configuration': configuration,
        'correctness': 'passed' if verified else 'unverified',
        'clean_performance_baseline': 'passed' if verified else 'unverified',
        'other_values_verified': False,
        'note': 'Other allowed values and modified implementations require ptobench evaluate .; permissions do not certify all combinations or guarantee speedup.',
    }
    return result


def check_projection(runtime, requested, case=None):
    """Reject an unsupported/misrouted compiler setting before Candidate import."""
    value = validate(requested, case)
    if not callable(getattr(runtime, 'compile_kwargs', None)):
        raise ValueError('installed RunConfig lacks the required compile_kwargs adapter')
    projected = runtime.compile_kwargs()
    for name in DEFAULTS:
        expected = value[name]
        actual = projected.get(name)
        actual = getattr(actual, 'name', actual)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'{name} is not supported by the installed compiler adapter')
    for name in ('ring_task_window', 'aicpu_thread_num'):
        if name not in value or value[name] is None:
            continue
        if type(getattr(runtime, name, None)) is not type(value[name]) or getattr(runtime, name) != value[name]:
            raise ValueError(f'{name} is not supported by the installed dispatch adapter')
        if name == 'aicpu_thread_num' and getattr(runtime, 'distributed_config', None) is not None:
            raise ValueError('aicpu_thread_num override is not supported for distributed dispatch')
        if getattr(runtime, 'platform', None) != 'a2a3':
            raise ValueError(f'{name} policy is only reviewed for the a2a3 onboard target')
    return record(runtime, value)


def record(runtime, requested):
    effective = {name: getattr(getattr(runtime, name, None), 'name', getattr(runtime, name, None))
                 for name in {**DEFAULTS, **MANAGED_DEFAULTS}}
    return {'requested': requested, 'resolved_run_config': effective,
            'unresolved_runtime_defaults': [n for n in MANAGED_DEFAULTS if n != 'strategy' and effective[n] is None],
            'diagnostics': {n: getattr(runtime, n, None) for n in DIAGNOSTICS
                            if isinstance(getattr(runtime, n, None), (bool, int, str))}}
