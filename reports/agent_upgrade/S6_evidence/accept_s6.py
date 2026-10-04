"""Reproducible S6 representative-only acceptance; no S7 inventory execution."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

BENCH = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
sys.path[:0] = [str(BENCH / 'evaluator'), str(BENCH)]
import common
import session
import session_actions as actions
import snapshots
import workspace

CASES = ('deepseek_v4.rmsnorm_decode_flash', 'deepseek_v4.compressor_ratio4_flash',
         'deepseek_v4.decode_attention_hca_flash', 'deepseek_v4.prefill_layer_ep2_hca_flash')
FORMULA = {'pairs': 5, 'ordering': 'direct then managed, repeated five times',
           'D': 'median(direct median_effective_us)', 'M': 'median(managed median_effective_us)',
           'MAD_D': 'median(abs(direct median - D))', 'allowed': 'max(0.10, 3 * MAD_D / D)',
           'pass': 'abs(M - D) / D <= allowed'}


def save(name, value):
    common.write_json(EVIDENCE / name, value)


def artifacts(raw):
    root = Path(raw['artifact_dir'])
    files = [p for p in root.rglob('*') if p.is_file() and p.stat().st_size > 0]
    mode = raw['profiling_mode']
    return {'root': str(root), 'nonempty_files': len(files),
            'diagnostic_files': [{'path': str(p), 'bytes': p.stat().st_size} for p in files
                                 if p.suffix.lower() in {'.json', '.csv', '.html', '.svg', '.log', '.mlir', '.txt'}],
            'build_inside_artifacts': bool(raw.get('build_dir')) and Path(raw['build_dir']).is_relative_to(root),
            'mode': mode}


def direct(case_id, label, mode):
    case_dir, sealed, case = common.resolve_case(case_id)
    directory = BENCH / 'runs/s6_controls' / case_id / label
    directory.mkdir(parents=True, exist_ok=False)
    with session.execution_lock():
        actions.recover_global_action()
        workspace.init_workspace(case_dir, sealed, case, directory / 'workspace')
        manifest = workspace.source_manifest(directory / 'workspace')
        common.write_json(directory / 'source_manifest.json', manifest)
        argv = [sys.executable, str(BENCH / 'evaluator/evaluate.py'), '--case', case_id,
                '--candidate', str(directory / 'workspace'), '--phase', 'all' if mode == 'clean' else 'correctness',
                '--profile-mode', mode, '--result', str(directory / 'result.json')]
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        env.pop('PYPTO_BENCH_ENV', None); env.pop('PYPTO_BENCH_PROFILE', None)
        code = actions.run_maintenance_process(argv, env, directory)
        raw = common.read_json(directory / 'result.json') if (directory / 'result.json').is_file() else None
        intact = manifest == workspace.source_manifest(directory / 'workspace')
    return inspect_raw(case_id, mode, raw, code, str(directory / 'result.json'), manifest, intact)


def inspect_raw(case_id, mode, raw, code, result_path, manifest, intact=True):
    good = bool(raw and code == 0 and raw.get('case_id') == case_id and raw.get('verdict') == 'PASS'
                and raw.get('profiling_mode') == mode and raw.get('official_score') is (mode == 'clean')
                and raw.get('diagnostic_only') is (mode != 'clean')
                and raw.get('phase') == ('all' if mode == 'clean' else 'correctness')
                and len(raw.get('runs', [])) == 1 and raw['runs'][0].get('passed') and intact)
    detail = artifacts(raw) if raw else None
    samples = raw['runs'][0].get('samples_effective_us') if raw else None
    if mode == 'clean':
        good = good and bool(samples and len(samples) == 10 and all(actions._positive(x) for x in samples))
    else:
        good = good and bool(detail and detail['nonempty_files'] and detail['diagnostic_files'])
    return {'status': 'PASS' if good else 'FAIL', 'exit_code': code, 'result_path': result_path,
            'profiling_mode': mode, 'source_hash': manifest['content_sha256'], 'source_intact': intact,
            'samples_effective_us': samples, 'median_effective_us': raw['runs'][0].get('median_effective_us') if raw else None,
            'artifacts': detail}


def managed(handle, label, public=None):
    root = handle.session_root
    argv = [sys.executable, str(BENCH / 'ptobench'), *(['evaluate'] if public is None else ['profile', public])]
    env = {**os.environ, 'PTOBENCH_SESSION_ROOT': str(root), 'PYTHONDONTWRITEBYTECODE': '1'}
    log = EVIDENCE / ('managed_' + root.parent.name + '_' + label)
    common.write_json(log.with_suffix('.command.json'), {'argv': argv, 'cwd': str(handle.initial_workspace)})
    with log.with_suffix('.stdout.log').open('wb') as out, log.with_suffix('.stderr.log').open('wb') as err:
        process = subprocess.run(argv, env=env, cwd=handle.initial_workspace, stdout=out, stderr=err, shell=False)
    event = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()][-1]
    directory = root / 'staging' / f'action_{event["seq"]:06d}'
    raw = common.read_json(directory / 'authority_result.json') if (directory / 'authority_result.json').exists() else event.get('result')
    manifest = common.read_json(directory / 'source_manifest.json')
    result = inspect_raw(root.parent.name, 'clean' if public is None else session.PUBLIC_MODES[public],
                         raw, process.returncode, str(directory / 'authority_result.json'), manifest, event['integrity_passed'])
    result.update(event_seq=event['seq'], snapshot_id=event['snapshot_id'], session_root=str(root), error_code=event['error_code'])
    return result


def equivalence(pairs):
    if len(pairs) != 5 or any(row['direct']['status'] != 'PASS' or row['managed']['status'] != 'PASS'
                             or row['direct']['source_hash'] != row['managed']['source_hash'] for row in pairs):
        return {'status': 'FAIL', 'reason': 'five passing matching-source pairs required'}
    direct_values = [row['direct']['median_effective_us'] for row in pairs]
    managed_values = [row['managed']['median_effective_us'] for row in pairs]
    d, m = statistics.median(direct_values), statistics.median(managed_values)
    mad = statistics.median(abs(value - d) for value in direct_values)
    allowed = max(0.10, 3 * mad / d)
    relative = abs(m - d) / d
    return {'status': 'PASS' if relative <= allowed else 'FAIL', 'direct_medians_us': direct_values,
            'managed_medians_us': managed_values, 'D': d, 'M': m, 'MAD_D': mad,
            'allowed': allowed, 'relative_difference': relative}


def hardware():
    if not (EVIDENCE / 'canonical_before.json').is_file():
        raise RuntimeError('Complete canonical byte baseline before any hardware action')
    save('equivalence_formula.json', FORMULA)  # fixed before the first sample
    matrix = {}
    for case_id in CASES:
        print('START', case_id, flush=True)
        try:
            _, _, case, environment = session.preflight_case(case_id)
            pair_count = 5 if case_id in CASES[:2] else 1 if case_id == CASES[2] else 0
            profiles = ['runtime', 'pmu', 'compiler'] if pair_count else ['compiler']
            handle = session.create_session(case_id, {'id': 's6_hardware', 'backend': 'maintenance'},
                {'runs_root': BENCH / 'runs', 'snapshots_root': BENCH / 'snapshots'},
                {'max_hardware_actions': max(pair_count, 1) + len(profiles), 'finish_actions': 1})
            row = {'session_root': str(handle.session_root), 'environment': environment,
                   'seed_manifest': workspace.source_manifest(handle.initial_workspace), 'pairs': [], 'profiles': {}}
            matrix[case_id] = row; save('hardware_matrix.json', matrix)
            if pair_count:
                for index in range(1, pair_count + 1):
                    d = direct(case_id, f'pair_{index:02d}', 'clean')
                    m = managed(handle, f'pair_{index:02d}')
                    row['pairs'].append({'index': index, 'direct': d, 'managed': m})
                    print('PAIR', case_id, index, d['status'], m['status'], d['median_effective_us'], m['median_effective_us'], flush=True)
                    save('hardware_matrix.json', matrix)
                if pair_count == 5:
                    row['equivalence'] = equivalence(row['pairs'])
            else:
                row['clean'] = managed(handle, 'clean')
            for public in profiles:
                control = direct(case_id, 'profile_' + public, session.PUBLIC_MODES[public]) if pair_count else None
                m = managed(handle, 'profile_' + public, public)
                row['profiles'][public] = {'direct': control, 'managed': m}
                print('PROFILE', case_id, public, (control or {}).get('status'), m['status'], flush=True)
                save('hardware_matrix.json', matrix)
            row['final'] = snapshots.finalize_session(handle.session_root, 's6_acceptance_complete').to_dict()
            record = session.read_session(handle.session_root)
            row['usage'] = record['usage']; row['active_action'] = record['active_action']
            if row['final']['finish_event_seq']:
                raw = common.read_json(handle.session_root / 'final/authority_result.json')
                row['final_artifacts'] = artifacts(raw)
            row['status'] = 'PASS' if row['final']['status'] == 'completed' and record['active_action'] is None else 'FAIL'
            save('hardware_matrix.json', matrix)
            print('FINAL', case_id, row['final']['status'], row['final']['final_verified_latency_us'], flush=True)
        except Exception as error:
            matrix.setdefault(case_id, {}).update(status='BLOCKED' if getattr(error, 'error_code', None) in {'invalid_request', 'infrastructure_error'} else 'FAIL',
                                                  error_type=type(error).__name__, reason=str(error))
            save('hardware_matrix.json', matrix)
            print('ERROR', case_id, type(error).__name__, str(error), flush=True)
    return matrix


def optimizer(backend):
    from launcher import runner
    import yaml
    path = EVIDENCE / f'real_{backend}.yaml'
    path.write_text(yaml.safe_dump({'case': CASES[0], 'optimizer': {'id': 's6_real_' + backend, 'backend': backend, 'profile': 'default'},
        'output': {'runs_root': str(BENCH / 'runs'), 'snapshots_root': str(BENCH / 'snapshots')},
        'budget': {'max_hardware_actions': 2, 'finish_actions': 1}}))
    def smoke_prompt(experiment, record, case):
        return ('Integration acceptance task. Read .benchmark_context/README.md, run exactly one '
                'ptobench evaluate in your current registered workspace, wait for its result, then exit. '
                'Do not optimize or edit source. PATH and Session environment are already set. '
                'Do not call internal scripts or inspect sealed assets. Bench will independently finalize on exit. '
                'Exploration budget is 2; finish budget is 1. No special completion text is required.')
    # Only task text is injected; the real public runner, adapter, profile,
    # external Optimizer CLI, ptobench and Evaluator execute without mocks.
    runner.make_prompt = smoke_prompt
    value, code = runner.run(path)
    root = Path(value['session_root'])
    record = session.read_session(root)
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    identity = common.read_json(root / 'optimizer_process.json')
    final = common.read_json(root / 'final_result.json') if (root / 'final_result.json').exists() else None
    good = code == 0 and final and final['status'] == 'completed' and record['usage'] == {'hardware_actions': 1, 'finish_actions': 1}
    good = good and [e['action'] for e in events] == ['evaluate', 'finish'] and record['active_action'] is None and not session._group_alive(identity)
    result = {'backend': backend, 'profile': 'default', 'case': CASES[0], 'status': 'PASS' if good else 'FAIL',
              'exit_code': code, 'launcher_result': value, 'usage': record['usage'], 'events': [e['action'] for e in events],
              'optimizer_group_alive': session._group_alive(identity), 'final': final,
              'backend_provenance': common.read_json(root / 'backend.json')}
    save(f'real_{backend}_result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'hardware':
        hardware()
    elif sys.argv[1] == 'optimizer':
        optimizer(sys.argv[2])
    else:
        raise SystemExit('expected hardware or optimizer BACKEND')
