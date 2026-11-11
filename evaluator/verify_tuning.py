#!/usr/bin/env python3
"""Serial clean verification of each Case's default tuning configuration.

Reuse canonical baseline certification and matching successful history. Never
prepare Ref/fixtures, edit Case code, or enable managed scheduling parameters.
Exhaustive combinations are an explicit optional diagnostic, not acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import common
import session
import session_actions
import workspace

PLANNERS = ('PYPTO', 'DSA_RP', 'PTOAS')


def identity(case_dir, case):
    return {'source_sha256':workspace.source_manifest(case_dir / case['workspace'])['content_sha256'],
            'case_sha256':common.sha256_file(case_dir / 'case.json'),
            'config_sha256':common.sha256_file(case_dir / case['config']),
            'comparator_sha256':common.comparator_sha256(case_dir, case),
            'environment_sha256':common.sha256_file(common.BENCHMARK_ROOT / 'environments' / (case['environment']+'.json')),
            'tuning_policy_sha256':hashlib.sha256(json.dumps(common.tuning_capabilities(case),sort_keys=True).encode()).hexdigest()}


def valid_result(result, request, case=None):
    runs = result.get('runs', [])
    if not isinstance(runs, list) or any(not isinstance(r, dict) for r in runs):
        return False
    if case is not None:
        declared = common.declared_fixture_ids(case)
        if len(runs) != len(declared) or {r.get('fixture_id') for r in runs} != set(declared):
            return False
    elif len(runs) != 1:
        return False
    return (result.get('profiling_mode') == 'clean' and result.get('verdict') == 'PASS' and
            result.get('effective_tuning') == request and
            all(r.get('passed') is True and len(r.get('samples_effective_us', [])) >= 3 and
                all(type(v) in (float,int) and math.isfinite(v) and v > 0 for v in r['samples_effective_us'])
                for r in runs))


def requests(scope, case=None):
    defaults = common.tuning_defaults(case) if case else dict(common.TUNING_DEFAULTS)
    if scope == 'defaults':
        return [defaults]
    if scope == 'combinations':
        return [{**defaults, 'memory_planner': planner, 'analyze_auto_scopes_for_deps': analyze}
                for planner, analyze in itertools.product(PLANNERS, (False, True))]
    raise ValueError(f'Unknown verification scope: {scope}')


def baseline_measurement_complete(baseline, case):
    """Check usable benchmark data without requiring optional tuning metadata."""
    fixtures = set(common.declared_fixture_ids(case))
    records = baseline.get('fixtures', {})
    return (baseline.get('status') == 'ready'
            and baseline.get('compiler_tuning') == common.tuning_defaults(case)
            and set(records) == fixtures
            and all(isinstance(run, dict)
                    and isinstance(run.get('samples_effective_us'), list)
                    and len(run['samples_effective_us']) >= 3
                    and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                            for value in run['samples_effective_us'])
                    for run in records.values()))


def initialize(scope='defaults'):
    rows = []
    # Small cases first, so runtime/software failures surface before large jobs.
    inventory = []
    for p in common.formal_case_configs():
        case = common.read_json(p)
        fixtures = common.sealed_root()/p.parent.relative_to(common.BENCHMARK_ROOT/'cases')/'fixtures'
        size = sum(f.stat().st_size for fixture in common.declared_fixture_ids(case)
                   for f in (fixtures/fixture/'in').glob('*.pt'))
        inventory.append((size,p,case))
    for _,p,case in sorted(inventory, key=lambda row:(row[0],row[2]['case_id'])):
        case_dir,sealed = p.parent,common.sealed_root()/p.parent.relative_to(common.BENCHMARK_ROOT/'cases')
        binding = identity(case_dir,case)
        baseline = common.read_json(sealed/'ref_performance.json')
        cert = common.read_json(sealed/'certification.json')
        baseline_ok = (cert.get('status') == 'ready'
                       and cert.get('comparator_sha256') == binding['comparator_sha256']
                       and baseline_measurement_complete(baseline, case))
        for request in requests(scope, case):
            reuse = baseline_ok and request == baseline.get('compiler_tuning')
            rows.append({'case_id':case['case_id'], 'tuning':request, 'identity':binding,
                         'status':'REUSED_BASELINE' if reuse else 'PENDING',
                         'evidence':str((sealed/'ref_performance.json').relative_to(common.BENCHMARK_ROOT)) if reuse else None})
    import tuning
    return {'schema_version':2, 'verification_scope':scope,
            'scope':'Exact canonical Seed plus compiler settings; no certification of modified Candidates or untested values.',
            'installed_adapter_sha256': common.read_json(tuning.POLICY_PATH)['installed_adapter_sha256'] if tuning.POLICY_PATH.is_file() else {},
            'queue_status':'prepared', 'items':rows}


def publish_default_baseline(row, case_dir, sealed, case, result):
    """Promote only a canonical Seed identical to Ref, under the execution lock."""
    def python_tree(root, main):
        return {('ENTRY.py' if p.name == main else str(p.relative_to(root))): common.sha256_file(p)
                for p in root.rglob('*.py')}
    if python_tree(case_dir/case['workspace'], case['module']) != python_tree(sealed/'ref', 'ref_npu.py'):
        raise ValueError('Canonical Seed differs from Ref; cannot publish a Ref performance baseline')
    if row['tuning'] != common.tuning_defaults(case) or not valid_result(result, row['tuning'], case):
        raise ValueError('Only a complete passing default clean run can publish the baseline')
    old = common.read_json(sealed/'ref_performance.json')
    runs = result['runs']
    warmup, rounds = common.benchmark_sizes(case)
    if (result.get('phase') != 'all' or any(
            run['measurement_config'].get('actual_rounds') != rounds or len(run['samples_effective_us']) != rounds or
            run['measurement_config'].get('actual_warmup') != warmup for run in runs)):
        raise ValueError('Baseline must use full formal warmups/rounds and correctness')
    updated = {**old, 'status': 'ready', 'compiler_tuning': dict(row['tuning']),
               'warmup': warmup, 'iterations': rounds, 'tuning_identity': row['identity'],
               'tuning_evidence': row['evidence'],
               'resolved_tuning': runs[0]['measurement_config'].get('tuning') if len(runs) == 1 else None,
               'resolved_tuning_by_fixture': {r['fixture_id']:r['measurement_config'].get('tuning') for r in runs},
               'fixtures': {run['fixture_id']: {
                   'raw_samples_effective_us': run['raw_samples_effective_us'],
                   'samples_effective_us': run['samples_effective_us'],
                   'median_effective_us': run['median_effective_us'],
                   'mean_effective_us': run['mean_effective_us']} for run in runs}}
    common.write_json(common.BENCHMARK_ROOT/row['evidence']/'previous_ref_performance.json', old)
    common.write_json(sealed/'ref_performance.json', updated)
    row['baseline_published'] = True


def save(path, report):
    report['summary'] = dict(Counter(row['status'] for row in report['items']))
    report['fully_verified_cases'] = sum(all(row['status'] in ('PASS','REUSED_BASELINE','REUSED_HISTORY') for row in report['items'] if row['case_id']==case)
                                         for case in {r['case_id'] for r in report['items']})
    common.write_json(path,report)


def no_device_processes(devices):
    result = subprocess.run(['npu-smi','info'], capture_output=True,text=True,timeout=30)
    if result.returncode:
        return False
    # Only a positive empty-process statement authorizes use; utilization is
    # known to be unreliable on this host. Do not use it to infer occupancy.
    return all(f'No running processes found in NPU {device}' in result.stdout for device in devices)


@contextlib.contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError('hardware tuning verification timed out')
    previous = signal.signal(signal.SIGALRM,expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)


def run_one(row, root, timeout):
    case_dir,sealed,case = common.case_paths(row['case_id'])
    if identity(case_dir,case) != row['identity']:
        row.update(status='STALE',error='Canonical source/configuration changed; re-inventory required')
        return
    profile = common._environment_profile(case)
    runtime = {**profile.get('runtime',{}),**case.get('runtime',{})}
    devices = runtime.get('device_ids',[case.get('runtime',{}).get('device_id',profile.get('device',0))])
    with session.execution_lock():
        if not no_device_processes(devices):
            row['status'] = 'WAITING_DEVICE'
            return
        planner = row['tuning']['memory_planner']
        job = root/'jobs'/row['case_id']/(planner+'_deps'+str(int(row['tuning']['analyze_auto_scopes_for_deps'])))
        if job.exists():
            row.update(status='INCOMPLETE',error='Existing attempt is preserved; inspect before retrying')
            return
        job.mkdir(parents=True)
        workspace.copy_candidate_tree(case_dir/case['workspace'], job/'workspace/candidate')
        common.write_json(job/'workspace/candidate/tuning.json',row['tuning'])
        argv = [sys.executable,str(common.BENCHMARK_ROOT/'evaluator/evaluate.py'),'--case',case['case_id'],
                '--candidate',str(job/'workspace'),'--profile-mode','clean','--phase','all','--result',str(job/'result.json')]
        environment = os.environ.copy()
        for key in ('PTOBENCH_SESSION_ROOT','PTOBENCH_WORKER','PYPTO_BENCH_ENV','PYPTO_BENCH_PROFILE'):
            environment.pop(key,None)
        row['evidence'] = str(job.relative_to(common.BENCHMARK_ROOT))
        try:
            with deadline(timeout):
                code = session_actions.run_maintenance_process(argv,environment,job)
            result = common.read_json(job/'result.json') if (job/'result.json').exists() else {}
            row['status'] = 'PASS' if code == 0 and valid_result(result,row['tuning'],case) else 'FAIL'
            row['exit_code'] = code
            if row['status'] == 'PASS':
                aggregate = common.fixture_aggregate(result['runs'])
                row['median_us'] = aggregate['latency_us']
                row['metric_name'] = aggregate['metric_name']
                row['fixture_metrics'] = {r['fixture_id']:{
                    'median_us':r['median_effective_us'], 'samples_us':r['samples_effective_us']}
                    for r in result['runs']}
                row['samples_us'] = result['runs'][0]['samples_effective_us'] if len(result['runs']) == 1 else None
                if row['tuning'] == common.tuning_defaults(case):
                    publish_default_baseline(row,case_dir,sealed,case,result)
        except TimeoutError as error:
            row.update(status='TIMEOUT',error=str(error))
        except Exception as error:
            row.update(status='ERROR',error=f'{type(error).__name__}: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('defaults','combinations'),default='defaults')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--timeout',type=int,default=1800)
    args = parser.parse_args()
    name = 'case_defaults_verification' if args.scope == 'defaults' else 'hardware_verification'
    if args.report is None:
        args.report = common.BENCHMARK_ROOT/'reports/tuning'/(name+'.json')
    report = common.read_json(args.report) if args.report.exists() else initialize(args.scope)
    if report.get('verification_scope', 'combinations') != args.scope:
        raise SystemExit('Report scope differs from requested scope; use a separate report')
    if not args.prepare and report.get('installed_adapter_sha256'):
        if any(not Path(path).is_file() or common.sha256_file(Path(path)) != expected
               for path, expected in report['installed_adapter_sha256'].items()):
            report['queue_status'] = 'environment_changed'
            save(args.report,report)
            raise SystemExit('Installed runtime changed; re-inventory required')
    if args.prepare:
        save(args.report,report)
        print(json.dumps(report['summary']))
        return 0
    if any(row['status'] == 'RUNNING' for row in report['items']):
        raise SystemExit('An interrupted/running attempt exists; inspect it before resuming')
    root = common.BENCHMARK_ROOT/'runs'/('tuning_case_defaults' if args.scope == 'defaults' else 'tuning_verification')
    root.mkdir(parents=True,exist_ok=True)
    report['queue_status'] = 'running'
    completed = 0
    for row in report['items']:
        if (root/'STOP').exists():
            report['queue_status'] = 'paused_by_stop_file'
            save(args.report,report)
            return 0
        if row['status'] not in ('PENDING','WAITING_DEVICE'):
            continue
        while True:
            row['status'] = 'RUNNING'
            save(args.report,report)
            run_one(row,root,args.timeout)
            save(args.report,report)
            if row['status'] != 'WAITING_DEVICE':
                break
            time.sleep(30)
        completed += 1
        print(json.dumps({k:row[k] for k in ('case_id','tuning','status')}),flush=True)
        if row['status'] in ('TIMEOUT','ERROR','INCOMPLETE','STALE'):
            report['queue_status'] = 'stopped_for_inspection'
            save(args.report,report)
            return 1
        if args.limit and completed >= args.limit:
            break
    report['queue_status'] = 'complete' if not any(r['status'] in ('PENDING','WAITING_DEVICE') for r in report['items']) else 'paused_after_limit'
    save(args.report,report)
    print(json.dumps(report['summary']),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
