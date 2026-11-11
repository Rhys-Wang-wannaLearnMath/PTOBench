#!/usr/bin/env python3
"""Audit every canonical Case's config plumbing without compiling/using an NPU."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import hashlib
import itertools
import json
import os
from pathlib import Path
from unittest.mock import patch

import common
import profile_modes
import tuning


def audit():
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import runner, distributed_runner
    import golden.runner as golden
    import pypto.ir
    entries = []
    totals = Counter()
    diagnostics = []
    # Fail loudly if a future edit accidentally turns this into a device test.
    with ExitStack() as stack:
        compile_guard = stack.enter_context(patch.object(pypto.ir, 'compile', side_effect=AssertionError('compilation forbidden')))
        run_guard = stack.enter_context(patch.object(golden, 'run', side_effect=AssertionError('NPU execution forbidden')))
        for path in common.formal_case_configs():
            case = common.read_json(path)
            profile = common._environment_profile(case)
            case_dir = path.parent
            relative = case_dir.relative_to(common.BENCHMARK_ROOT / 'cases')
            evaluator = common.sealed_root() / relative / 'evaluate.py'
            tree = ast.parse(evaluator.read_text())
            shared_workflow = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                  and n.func.attr == 'evaluate_candidate' and
                                  ((isinstance(n.func.value, ast.Name) and n.func.value.id in ('workflow','wf','_WORKFLOW')) or
                                   (isinstance(n.func.value, ast.Call) and isinstance(n.func.value.func, ast.Name) and
                                    n.func.value.func.id == '_workflow'))
                                  for n in ast.walk(tree))
            failures = []
            if not shared_workflow:
                failures.append('Case adapter does not use the audited shared workflow')
            cap = common.tuning_capabilities(case)
            inherited = {**profile.get('runtime', {}), **case.get('runtime', {})}
            diagnostic_fields = {k:v for k,v in inherited.items() if k in tuning.DIAGNOSTICS and v}
            if diagnostic_fields:
                diagnostics.append({'case_id':case['case_id'], 'settings':diagnostic_fields})
            checks = 0
            defaults = common.tuning_defaults(case)
            requests = [{**defaults, 'memory_planner': planner, 'analyze_auto_scopes_for_deps': analyze}
                        for planner, analyze in itertools.product(('PYPTO','DSA_RP','PTOAS'), (False,True))
                        if analyze in cap['options']['analyze_auto_scopes_for_deps']['values']]
            for name in ('ring_task_window', 'aicpu_thread_num'):
                if cap['options'][name]['editable']:
                    requests.extend({**defaults, name: value} for value in cap['options'][name]['values']
                                    if value != defaults[name])
            for mode, request in itertools.product(('clean', *profile_modes.ENABLED_PROFILE_MODES), requests):
                try:
                    with patch.dict(os.environ, {'PYPTO_BENCH_PROFILE':mode}):
                        _, runtime = common.environment(case, None, tuning=request)
                    tuning.check_projection(runtime, request, case)
                    assert runtime.strategy.name == 'Default'
                    for field in ('ring_task_window','ring_heap','ring_dep_pool','aicpu_thread_num'):
                        assert getattr(runtime, field) == request.get(field, inherited.get(field)), field
                    for field, value in profile_modes.settings(mode).items():
                        assert getattr(runtime, field) == value, field
                    checks += 1
                except (AssertionError, TypeError, ValueError, AttributeError) as error:
                    failures.append(f'{mode}/{request}: {error}')
            if not failures:
                totals['configuration_pass'] += 1
            totals['configurations_checked'] += checks
            totals['cases'] += 1
            totals['shared_workflow'] += int(shared_workflow)
            entries.append({'case_id':case['case_id'], 'status':'PASS' if not failures else 'FAIL',
                            'configurations_checked':checks, 'failures':failures,
                            'case_sha256':common.sha256_file(path),
                            'evaluator_sha256':common.sha256_file(evaluator),
                            'managed_defaults':{k:v.get('inherited_value',v['default']) for k,v in cap['options'].items() if not v['editable']}})
        assert compile_guard.call_count == run_guard.call_count == 0
    return {
        'schema_version':1,
        'scope':'Configuration/interface audit only; no Candidate import, fixture generation, actual compilation or NPU dispatch.',
        'summary':dict(totals),
        'installed_strategies':[s.name for s in OptimizationStrategy],
        'adapter_sha256':{str(Path(module.__file__)):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for module in (runner,distributed_runner)},
        'existing_clean_diagnostic_settings':diagnostics,
        'baseline_policy':'Per-Case defaults in evaluator/tuning_policies.json; PYPTO + false preserved.',
        'public_policy':tuning.capabilities({'case_id':'all_cases'}, {}),
        'hardware_experiments_started':0,
        'entries':entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    common.write_json(args.output, report)
    print(json.dumps({'summary':report['summary'], 'hardware_experiments_started':0,
                      'existing_clean_diagnostic_cases':len(report['existing_clean_diagnostic_settings'])}))
    return int(any(e['status'] != 'PASS' for e in report['entries']))


if __name__ == '__main__':
    raise SystemExit(main())
