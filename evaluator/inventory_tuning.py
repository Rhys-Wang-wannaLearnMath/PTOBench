#!/usr/bin/env python3
"""Read-only source/API inventory; generate reviewed per-Case tuning policies.

Never import a Candidate, compile a program or dispatch to an accelerator.
Reachability is conservative Python AST name resolution, not a speedup claim.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import inspect
from dataclasses import fields
from pathlib import Path

import common


def dotted(node):
    return ast.unparse(node)


def source_features(case_dir, case):
    seed = case_dir / case['workspace']
    trees, functions, aliases = {}, {}, {}
    for path in sorted(seed.rglob('*.py')):
        module = '.'.join(path.relative_to(seed).with_suffix('').parts)
        tree = ast.parse(path.read_text(), filename=str(path))
        trees[module] = tree
        aliases[module] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for item in node.names:
                    aliases[module][item.asname or item.name] = (node.module, item.name)
            elif isinstance(node, ast.Import):
                for item in node.names:
                    aliases[module][item.asname or item.name] = (item.name, None)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                functions.setdefault((module, node.name), []).append(node)
    entry_module = '.'.join(Path(case['module']).with_suffix('').parts)
    pending = [(entry_module, case['entry'])]
    visited, nodes = set(), []
    while pending:
        key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        if key not in functions:
            imported = aliases.get(key[0], {}).get(key[1])
            if imported:
                pending.append(imported)
        for definition in functions.get(key, []):
            nodes.append((key[0], definition))
            for node in ast.walk(definition):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    local = (key[0], node.id)
                    if local in functions:
                        pending.append(local)
                    imported = aliases[key[0]].get(node.id)
                    if imported in functions:
                        pending.append(imported)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        base = node.func.value.id
                        if base in ('self', 'cls'):
                            pending.append((key[0], node.func.attr))
                        imported = aliases[key[0]].get(base)
                        if imported and imported[1] is None:
                            pending.append((imported[0], node.func.attr))
    if not nodes:
        raise ValueError(f"Cannot resolve Case entry: {case['case_id']}")
    # Nested definitions may appear in several reachable roots; count once.
    unique = {}
    for module, root in nodes:
        for node in ast.walk(root):
            unique[(module, id(node))] = node
    calls, evidence = Counter(), defaultdict(list)
    auto_scope, manual_scope, host = False, False, False
    for (module, _), node in unique.items():
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            if name.startswith(('pl.', 'pld.')):
                calls[name] += 1
                if len(evidence[name]) < 4:
                    evidence[name].append(f'{module.replace(".", "/")}.py:{node.lineno}')
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                name = dotted(decorator.func if isinstance(decorator, ast.Call) else decorator)
                if name.startswith('pl.jit') or name == 'pl.function':
                    text = dotted(decorator)
                    host |= name == 'pl.jit.host' or 'Host' in text
                    incore = name == 'pl.jit.incore' or 'InCore' in text
                    disabled = isinstance(decorator, ast.Call) and any(
                        k.arg == 'auto_scope' and isinstance(k.value, ast.Constant) and k.value.value is False
                        for k in decorator.keywords)
                    if not incore and not disabled:
                        auto_scope = True
    manual_scope = bool(calls['pl.manual_scope'])
    scheduling = any(calls[k] for k in ('pl.parallel', 'pl.spmd', 'pl.scope', 'pl.manual_scope', 'pl.at'))
    # More than one call site, or a task-producing iteration, can exercise the
    # scheduler. This establishes applicability, not a latency improvement.
    multiple_tasks = scheduling and (bool(calls['pl.parallel'] or calls['pl.spmd']) or
        sum(calls[k] for k in ('pl.at', 'pl.scope', 'pl.manual_scope')) > 1 or len(nodes) > 2)
    from workspace import source_manifest
    return {'entry_resolved': True, 'reachable_definitions': sorted(f'{m}.{n}' for m, n in visited if (m, n) in functions),
            'source_sha256': source_manifest(seed)['content_sha256'],
            'auto_scope_possible': auto_scope, 'manual_scope_present': manual_scope,
            'host_orchestration': host, 'multiple_tasks_possible': multiple_tasks,
            'matmul_present': any('matmul' in k for k in calls),
            'pipeline_present': bool(calls['pl.pipeline']),
            'dsl_calls': dict(sorted(calls.items())), 'evidence': dict(evidence),
            'analysis_limit': 'Conservative reachable source closure; lowering can eliminate sites. No hardware or speedup claim.'}


def make_policy(case, profile, features):
    inherited = {**profile.get('runtime', {}), **case.get('runtime', {})}
    multi = bool(inherited.get('device_ids')) or features['host_orchestration']
    board = profile['platform'] == 'a2a3'
    opts = {
        'memory_planner': {'editable': True, 'default': 'PYPTO', 'values': ['PYPTO', 'DSA_RP', 'PTOAS'],
                           'stage': 'compile', 'reason': 'Controls local tile-buffer lifetime reuse and allocation.'},
        'analyze_auto_scopes_for_deps': {
            'editable': features['auto_scope_possible'], 'default': False,
            'values': [False, True] if features['auto_scope_possible'] else [False], 'stage': 'compile',
            'reason': 'Reachable orchestration can materialize AUTO scopes.' if features['auto_scope_possible'] else
                      'No reachable automatic orchestration scope; manual scopes are skipped by this pass.'},
    }
    if board and features['multiple_tasks_possible']:
        window = inherited.get('ring_task_window')
        # No shrinking and no unbounded allocation. Unusual existing windows
        # remain managed until their independent resource budget is reviewed.
        if window is None or window == 16384:
            opts['ring_task_window'] = {'editable': True, 'default': window or 16384, 'stage': 'dispatch',
                'values': [16384, 32768], 'unit': 'in-flight task slots per ring',
                'resolved_default': 16384,
                'reason': 'Task-producing orchestration uses native task rings; permit default or at most 2x capacity, never shrink.'}
        threads = inherited.get('aicpu_thread_num')
        if not multi and (threads is None or threads in (2, 3, 4)):
            opts['aicpu_thread_num'] = {'editable': True, 'default': threads or 4, 'stage': 'dispatch',
                'values': [2, 3, 4], 'unit': 'AICPU threads', 'resolved_default': threads or 4,
                'reason': 'Single-card scheduler accepts 2..4 threads (at least one orchestrator and one scheduler).'}
    return {'case_id': case['case_id'], 'options': opts,
            'defaults': {name: spec['default'] for name, spec in opts.items()},
            'inherited_runtime': inherited, 'distributed_path': multi,
            'source_sha256': features['source_sha256'],
            'case_sha256': common.sha256_file(common.case_paths(case['case_id'])[0] / 'case.json'),
            'environment_sha256': common.sha256_file(common.BENCHMARK_ROOT/'environments'/(case['environment']+'.json')),
            'validation_scope': 'Static applicability and bounded policy; default correctness/performance evidence is separate.'}


def inventory():
    import pypto.ir
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig
    from simpler_setup.environment import PROJECT_ROOT
    import pypto.runtime.runner as runner
    import pypto.runtime.distributed_runner as distributed_runner
    import pypto.jit.decorator as decorator
    runtime_types = PROJECT_ROOT/'src/a2a3/runtime/tensormap_and_ringbuffer/runtime/runtime_types.h'
    platform_config = PROJECT_ROOT/'src/a2a3/platform/include/common/platform_config.h'
    runtime_maker = PROJECT_ROOT/'src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp'
    native_dispatch = PROJECT_ROOT/'src/common/platform/onboard/host/device_runner_base.cpp'
    if '#define CHIP_TASK_WINDOW_SIZE 16384 ' not in runtime_types.read_text():
        raise ValueError('Review native task window bound for this installed runtime')
    if 'PLATFORM_MAX_AICPU_THREADS = 4;' not in platform_config.read_text():
        raise ValueError('Review AICPU bound for this installed runtime')
    entries, policies = [], {}
    for path in common.formal_case_configs():
        case = common.read_json(path)
        features = source_features(path.parent, case)
        policies[case['case_id']] = make_policy(case, common._environment_profile(case), features)
        entries.append({'case_id': case['case_id'], **features})
    parameters = list(inspect.signature(pypto.ir.compile).parameters)
    decisions = {
        'memory_planner': 'Open: direct compile adapter and local memory-planning effect.',
        'analyze_auto_scopes_for_deps': 'Open where reachable AUTO scopes are possible; false otherwise.',
        'enable_pypto_l0c_double_buffer': 'Withheld: experimental PYPTO-only flag; known chained Mat-scratch operand-buffer overflow (#1908), not carried by RunConfig and JIT cache keys read outer context.',
        'runtime': 'Withheld: changes runtime ABI, graph construction and worker binding; not an ordinary safe optimization control.',
        'strategy': 'Fixed Default: installed version registers no alternative strategy.',
        'ring_task_window': 'Open for task-producing a2a3 orchestration: 16384/32768 only, default 16384; no smaller windows or unbounded sizes.',
        'aicpu_thread_num': 'Open for single-card task-producing a2a3 paths: 2/3/4, default 4 unless already overridden; withheld on distributed paths whose thread source differs.',
        'ring_heap': 'Managed: affects output allocations; shrinking risks failures and increasing can exhaust HBM.',
        'ring_dep_pool': 'Managed: dependency capacity; undersizing risks blocked scheduling; no demonstrated need to expose.',
        'pass_options': 'Withheld: no typed installed RunConfig adapter; arbitrary pass removal/reordering can invalidate required IR invariants.',
        'auto_scope': 'Source-level DSL control: already editable in Candidate decorators/scopes; no independent global RunConfig override. Must retain correct dependencies.',
        'pipeline_stages_and_tile_shapes': 'Source-level DSL performance controls: Agent can edit Candidate pipeline/tile structure; not global compiler configuration.',
    }
    for name in parameters:
        decisions.setdefault(name, 'Managed contract, target, artifact or diagnostic input; not an Agent performance control.')
    import pypto.ir.pass_manager as pass_manager
    import pypto.pypto_core as core
    import golden.runner as golden_runner
    compile_module = inspect.getmodule(pypto.ir.compile)
    bindings = {str(Path(m.__file__)): common.sha256_file(Path(m.__file__))
                for m in (runner, distributed_runner, decorator, compile_module, pass_manager, core, golden_runner)}
    bindings.update({str(p): common.sha256_file(p) for p in (runtime_types, platform_config, runtime_maker, native_dispatch)})
    counts = Counter(name for p in policies.values() for name, spec in p['options'].items() if spec['editable'])
    policy = {'schema_version': 2, 'scope': 'Per-Case reviewed optimization permissions; hardware PASS is required only for defaults.',
              'installed_adapter_sha256': bindings, 'cases': policies}
    report = {'schema_version': 2, 'cases': len(entries), 'editable_option_counts': dict(counts),
              'compile_parameters': parameters, 'installed_strategies': [x.name for x in OptimizationStrategy],
              'run_config_fields': [f.name for f in fields(RunConfig)],
              'pass_context_controls': ['memory_planner', 'enable_pypto_l0c_double_buffer', 'runtime',
                                        'verification_level', 'diagnostic_phase', 'disabled_diagnostics', 'instruments'],
              'source_level_controls': ['auto_scope', 'pipeline stages', 'tile shapes', 'scope placement'],
              'option_decisions': decisions, 'installed_adapter_sha256': bindings, 'entries': entries,
              'hardware_experiments_started': 0}
    return policy, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=common.BENCHMARK_ROOT/'evaluator/tuning_policies.json')
    parser.add_argument('--report', type=Path, default=common.BENCHMARK_ROOT/'reports/tuning/case_option_inventory.json')
    args = parser.parse_args()
    policy, report = inventory()
    common.write_json(args.policy, policy)
    common.write_json(args.report, report)
    print({'cases': report['cases'], 'editable_option_counts': report['editable_option_counts']})


if __name__ == '__main__':
    main()
