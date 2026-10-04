"""Read-only inspection of actual S6 results and native diagnostic schemas."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2]
sys.path[:0] = [str(BENCH / 'evaluator'), str(BENCH)]
import common
import session
import workspace


def json_shape(path):
    value = json.loads(path.read_text())
    assert isinstance(value, dict) and value, path
    return {key: type(child).__name__ for key, child in sorted(value.items())}


def diagnostics(item, mode):
    root = Path(item['artifacts']['root'])
    schema = {}
    evidence = []
    if mode == 'runtime':
        for pattern in ('chip_swimlane_records.json', 'deps.json', 'merged_swimlane_*.json'):
            files = list(root.rglob(pattern))
            assert files, (root, pattern)
            shapes = [json_shape(path) for path in files]
            schema[pattern] = shapes
            evidence.extend(map(str, files))
        scope = list(root.rglob('scope_stats.jsonl'))
        assert scope, root
        for path in scope:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            assert rows and all(isinstance(row, dict) for row in rows), path
            schema['scope_stats.jsonl'] = sorted(set(key for row in rows for key in row))
        evidence.extend(map(str, scope))
    elif mode == 'pmu':
        files = list(root.rglob('pmu.csv'))
        assert files, root
        headers = []
        for path in files:
            with path.open() as reader:
                rows = list(csv.reader(reader))
            assert len(rows) > 1 and 'pmu_total_cycles' in rows[0], path
            assert all(len(row) == len(rows[0]) for row in rows[1:]), path
            headers.append(rows[0])
        schema['pmu.csv'] = headers
        evidence.extend(map(str, files))
    elif mode == 'compiler':
        files = list(root.rglob('pipeline_profile.json'))
        assert files, root
        schema['pipeline_profile.json'] = [json_shape(path) for path in files]
        dumps = sorted(path.name for path in root.rglob('passes_dump/*.py') if path.stat().st_size)
        assert dumps, root
        schema['passes_dump'] = dumps
        evidence.extend(map(str, files))
    return {'schema': schema, 'evidence': evidence}


def check_hardware():
    matrix = common.read_json(HERE / 'hardware_matrix.json')
    results = {}
    for case_id, row in matrix.items():
        result = {'profiles': {}, 'checks': {}, 'errors': []}
        results[case_id] = result
        if row.get('status') == 'BLOCKED':
            result.update(status='BLOCKED', reason=row.get('reason'), required_runs=row.get('required_runs'))
            continue
        if 'final' not in row:
            result.update(status='NOT_RUN', reason='hardware acceptance is still in progress')
            continue
        try:
            root = Path(row['session_root'])
            # Raw immutable records suffice here; completed historical Sessions
            # must not restore old public code into the current implementation.
            record = common.read_json(root / 'session.json')
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
            final = common.read_json(root / 'final_result.json')
            raw_final = common.read_json(root / 'final/authority_result.json')
            seed_hash = row['seed_manifest']['content_sha256']
            result['checks']['all_actions_pass'] = all(e['error_code'] is None and e['integrity_passed'] for e in events)
            result['checks']['source_hash_constant'] = all(e['content_sha256'] == seed_hash for e in events)
            result['checks']['one_finish'] = record['usage']['finish_actions'] == 1 and sum(e['action'] == 'finish' for e in events) == 1
            result['checks']['no_active'] = record['active_action'] is None
            result['checks']['fresh_final'] = raw_final['candidate'] == str(root / 'final/workspace') and raw_final['profiling_mode'] == 'clean' and raw_final['phase'] == 'all'
            result['checks']['final_corresponds_to_clean'] = final['status'] == 'completed' and final['final_verified_latency_us'] == raw_final['runs'][0]['median_effective_us'] and final['final_verified_speedup'] == raw_final['runs'][0]['speedup']
            result['checks']['ten_positive_final_samples'] = len(raw_final['runs'][0]['samples_effective_us']) == 10 and all(x > 0 for x in raw_final['runs'][0]['samples_effective_us'])
            result['checks']['final_source_matches_snapshot'] = workspace.source_manifest(root / 'final/workspace')['content_sha256'] == final['selected_content_sha256'] == seed_hash
            live = []
            control_root = BENCH / 'runs/s6_controls' / case_id
            for process in Path('/proc').iterdir():
                if process.name.isdecimal():
                    try:
                        cwd = Path(os.readlink(process / 'cwd'))
                        if cwd.is_relative_to(root) or cwd.is_relative_to(control_root):
                            live.append(int(process.name))
                    except (OSError, ValueError):
                        pass
            result['live_processes_in_workspaces'] = live
            result['checks']['no_live_process_in_owned_workspaces'] = not live
            identities = [common.read_json(path) for path in control_root.rglob('process_identity.json')]
            result['checks']['direct_process_groups_dead'] = not any(session._group_alive(identity) for identity in identities)
            result['checks']['direct_and_managed_clean'] = all(pair['direct']['status'] == pair['managed']['status'] == 'PASS' and pair['direct']['source_hash'] == pair['managed']['source_hash'] for pair in row['pairs'])
            result['checks']['expected_devices'] = record['environment'].get('runtime', {}).get('device_ids') == ([0, 1] if case_id.endswith('prefill_layer_ep2_hca_flash') else None)
            result['checks']['warmup_rounds_unchanged'] = record['environment']['benchmark'] == {'warmup': 5, 'rounds': 10}
            result['source_files'] = [entry['path'] for entry in row['seed_manifest']['files']]
            for public, profile in row['profiles'].items():
                managed = diagnostics(profile['managed'], public)
                control = diagnostics(profile['direct'], public) if profile['direct'] else None
                equal = control is None or control['schema'] == managed['schema']
                result['profiles'][public] = {'status': 'PASS' if equal and profile['managed']['status'] == 'PASS' else 'FAIL',
                                             'schema_equal_to_direct': equal if control else None,
                                             'direct': control, 'managed': managed}
            result['status'] = 'PASS' if all(result['checks'].values()) and all(p['status'] == 'PASS' for p in result['profiles'].values()) else 'FAIL'
        except Exception as error:
            result['status'] = 'FAIL'
            result['errors'].append(type(error).__name__ + ': ' + str(error))
    common.write_json(HERE / 'hardware_checks.json', results)
    print(json.dumps(results, indent=2))


def check_hashes():
    before = common.read_json(HERE / 'canonical_before.json')
    after = common.read_json(HERE / 'canonical_after.json')
    changed = {name: {'before': before['entries'].get(name), 'after': after['entries'].get(name)}
               for name in before['entries'].keys() | after['entries'].keys()
               if before['entries'].get(name) != after['entries'].get(name)}
    unstable = [name for name, info in after['entries'].items() if info.get('stable_during_read') is False]
    result = {'status': 'PASS' if not changed and not unstable else 'FAIL',
              'algorithm': 'SHA-256 over all file bytes', 'before_bytes': before['bytes'], 'after_bytes': after['bytes'],
              'before_entries': len(before['entries']), 'after_entries': len(after['entries']),
              'differences': changed, 'unstable_reads': unstable}
    common.write_json(HERE / 'canonical_comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'hardware': check_hardware()
    elif sys.argv[1] == 'hashes': check_hashes()
    else: raise SystemExit('expected hardware or hashes')
