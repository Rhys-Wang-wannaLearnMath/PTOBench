"""Rebuild selected Case coverage from existing PTOBench metadata, without hardware."""
from __future__ import annotations
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

BENCH = Path(__file__).resolve().parents[5]
REPORT = BENCH / 'modes/basic_wiki-tool/reports/stage5'
sys.path[:0] = [str(BENCH), str(BENCH / 'evaluator')]
from workspace import source_manifest, copy_candidate_tree, _ensure_tuning
profile = importlib.import_module('modes.basic_wiki-tool.pto_analyzer.profile')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected():
    cases, sources = {}, []
    for split in ('formal_60', 'heldout_15'):
        path = BENCH / f'case_selection/{split}/cases.csv'
        rows = list(csv.DictReader(path.open()))
        sources.append({'path': str(path.relative_to(BENCH)), 'sha256': digest(path), 'rows': len(rows)})
        for row in rows:
            case_id = row['case_id']
            family, name = case_id.split('.', 1)
            case_dir = BENCH / 'cases' / family / name
            case = json.loads((case_dir / 'case.json').read_text())
            env = json.loads((BENCH / 'environments' / f"{case['environment']}.json").read_text())
            runtime = {**env.get('runtime', {}), **case.get('runtime', {})}
            devices = runtime.get('device_ids') or [env.get('device', 0)]
            seed = case_dir / case['workspace']
            with tempfile.TemporaryDirectory(prefix='stage5-seed-') as temporary:
                staged = Path(temporary)
                copy_candidate_tree(seed, staged / 'candidate')
                _ensure_tuning(staged, case)
                canonical_hash = source_manifest(staged)['content_sha256']
            cases[case_id] = {
                'split': split, 'device_count': int(row['device_count']),
                'configured_devices': devices, 'canonical_starting_candidate': str(seed.relative_to(BENCH)),
                'canonical_candidate_sha256': canonical_hash,
                'canonical_hash_method': 'Public Workspace copy_candidate_tree + _ensure_tuning + source_manifest in a temporary CPU directory, including normal generated tuning.json.',
                'records': [], 'selected_profile': None,
            }
    return cases, sources


def indexed_artifacts(root, event, expected_count):
    """Check control identities and physical rank coverage, without parsing large diagnostics."""
    ref = profile.ProfileRef.from_event(root, event['seq'])
    archive = profile.ProfileArchive(ref)
    required = ('chip_swimlane_records.json', 'deps.json', 'pmu.csv', 'pipeline_profile.json', 'perf_hints.log')
    by_name = {name: archive.artifact_paths(name) for name in required}
    errors = []
    for name, paths in by_name.items():
        if not paths:
            errors.append(f'not_indexed:{name}')
        for rel in paths:
            path = archive.artifact_root / rel
            if not path.is_file() or path.is_symlink():
                errors.append(f'file_unavailable:{rel}')
            elif name != 'perf_hints.log' and path.stat().st_size == 0:
                errors.append(f'empty:{rel}')
    # Every runtime dispatch directory needs matching PMU/deps; multi-card outputs need all ranks.
    rank_counts = {}
    for rel in by_name['chip_swimlane_records.json']:
        parts = Path(rel).parts
        rank = next((x for x in parts if x.startswith('rank') and x[4:].isdigit()), 'unranked')
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        for name in ('deps.json', 'pmu.csv'):
            if (Path(rel).parent / name).as_posix() not in archive._entries:
                errors.append(f'dispatch_missing:{(Path(rel).parent / name).as_posix()}')
    if expected_count > 1:
        expected = {f'rank{i}' for i in range(expected_count)}
        if set(rank_counts) != expected:
            errors.append(f'rank_coverage:{sorted(rank_counts)} expected {sorted(expected)}')
    return {
        'archive_authorization': 'valid',
        'artifact_index': str(archive.artifact_root / 'index.json'),
        'artifact_index_sha256': digest(archive.artifact_root / 'index.json'),
        'indexed_diagnostic_counts': {k: len(v) for k, v in by_name.items()},
        'runtime_dispatches_by_rank': rank_counts,
        'errors': errors,
        'scope': 'Session/Event/index/protection identity plus indexed diagnostic presence/nonempty and rank pairing; large diagnostic payloads not reparsed or rehashed.',
    }


def scan():
    cases, sources = selected()
    scan_errors, event_files, index_files, queues = [], [], [], []
    # Use find for metadata names only; no profile data is opened during discovery.
    scan_roots = ['runs', 'manual_runs', 'reports']
    external = Path('/data2/ziruiwang/PTOBench/stage5_profile_fill/runs')
    if external.is_dir():
        scan_roots.append(str(external))
    paths = subprocess.run(['find', *scan_roots, '-type', 'f', '(',
                            '-name', 'events.jsonl', '-o', '-name', 'index.json', '-o', '-name', 'queue.json', ')'],
                           cwd=BENCH, capture_output=True, text=True, check=True).stdout.splitlines()
    for rel in paths:
        p = BENCH / rel
        if p.name == 'events.jsonl' and p.parent.name != 'agent':
            event_files.append(p)
        elif p.name == 'index.json':
            index_files.append(p)
        elif p.name == 'queue.json':
            queues.append(p)
    for events in event_files:
        root = events.parent
        try:
            session = json.loads((root / 'session.json').read_text())
        except (OSError, ValueError):
            continue
        case_id = session.get('case_id')
        if case_id not in cases:
            continue
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                scan_errors.append({'path': str(events), 'error': 'invalid_event_json'})
                continue
            if event.get('action') != 'profile' or event.get('mode') != 'combined':
                continue
            result = event.get('result') or {}
            identities = [run.get('device_identity', {}) for run in result.get('runs', [])]
            row = {
                'case_id': case_id, 'session_root': str(root), 'session_id': session.get('session_id'),
                'event_seq': event.get('seq'), 'event_id': f"{session.get('session_id')}:{event.get('seq')}",
                'candidate_sha256': event.get('content_sha256'), 'mode': event.get('mode'),
                'state': event.get('state'), 'compile_status': event.get('compile_status'),
                'runtime_status': event.get('runtime_status'), 'error_code': event.get('error_code'),
                'integrity_passed': event.get('integrity_passed'), 'event_time': event.get('time'),
                'device_identities': identities,
                'canonical_starting_candidate_match': event.get('content_sha256') == cases[case_id]['canonical_candidate_sha256'],
                'events_jsonl': str(events), 'origin': 'stage5_fill' if 'stage5_profile_fill' in root.parts else 'existing',
                'usable': False,
            }
            if event.get('state') == 'completed' and event.get('runtime_status') == 'pass' and event.get('compile_status') == 'pass' and event.get('integrity_passed') is True:
                try:
                    row['artifact_validation'] = indexed_artifacts(root, event, cases[case_id]['device_count'])
                    row['usable'] = not row['artifact_validation']['errors']
                except (OSError, ValueError, TypeError, KeyError) as error:
                    row['artifact_validation'] = {'errors': [f'{type(error).__name__}: {error}']}
            cases[case_id]['records'].append(row)
    # Indexes are checked for orphaned raw combined outputs; report rather than fabricate Events.
    orphaned = []
    combined_indexes = 0
    for path in index_files:
        try:
            index = json.loads(path.read_text())
            entries = index.get('files', [])
        except (OSError, ValueError, AttributeError):
            continue
        if not any('/combined/' in x.get('path', '') and x.get('path', '').endswith('chip_swimlane_records.json') for x in entries if isinstance(x, dict)):
            continue
        combined_indexes += 1
        root = next((p for p in path.parents if (p / 'session.json').is_file()), None)
        if root is None or not (root / 'events.jsonl').is_file():
            orphaned.append({'artifact_index': str(path), 'session_root': str(root) if root else None})
    collection_log = Path('/data2/ziruiwang/PTOBench/stage5_profile_fill/collection.log')
    if collection_log.is_file():
        for line_number, line in enumerate(collection_log.read_text().splitlines(), 1):
            try:
                notice = json.loads(line)
            except ValueError:
                continue
            case_id = notice.get('failed')
            if case_id in cases:
                cases[case_id].setdefault('collection_setup_failures', []).append({
                    'error': notice.get('error'), 'session_root': notice.get('session_root'),
                    'original_log': str(collection_log), 'line': line_number,
                })
    covered, failed, pending = [], [], []
    for case_id, item in cases.items():
        valid = sorted((r for r in item['records'] if r['usable']), key=lambda r: (r['canonical_starting_candidate_match'], r.get('event_time') or '', r['event_id']), reverse=True)
        if valid:
            chosen = valid[0]
            item['selected_profile'] = chosen['event_id']
            item['selected_session_root'] = chosen['session_root']
            item['selection_reason'] = 'Prefer a canonical starting Candidate match, then the latest authorized completed combined event with complete indexed diagnostics and device ranks.'
            item['status'] = 'covered'
            covered.append(case_id)
            for r in item['records']:
                r['retained'] = r is chosen
                r['disposition'] = 'selected_for_coverage' if r is chosen else ('duplicate_not_selected_by_stated_preference' if r['usable'] else 'unusable_original_evidence_retained')
        else:
            status = 'failed_collection' if item.get('collection_setup_failures') or any(r['origin'] == 'stage5_fill' and r['state'] in ('completed', 'failed', 'interrupted') for r in item['records']) else 'pending_collection'
            item['status'] = status
            (failed if status == 'failed_collection' else pending).append(case_id)
    return {
        'schema_version': 2, 'audited_at': datetime.now(timezone.utc).isoformat(),
        'decision': 'Latest user instruction authorizes missing canonical Candidate combined Profile collection through ptobench profile ., without waiting for selected-Mode protection; public code remains unchanged.',
        'selection_sources': sources, 'selection_count': len(cases),
        'discovery': {'roots': scan_roots, 'non_agent_event_files': len(event_files),
                      'artifact_index_files': len(index_files), 'combined_artifact_indexes': combined_indexes,
                      'queue_files': [{'path': str(p), 'sha256': digest(p)} for p in queues],
                      'orphaned_combined_indexes': orphaned, 'errors': scan_errors},
        'summary': {'covered': len(covered), 'failed_collection': len(failed), 'pending_collection': len(pending),
                    'selected_existing': sum(cases[c]['records'][next(i for i,r in enumerate(cases[c]['records']) if r.get('retained'))]['origin'] == 'existing' for c in covered),
                    'covered_case_ids': covered, 'failed_case_ids': failed, 'pending_case_ids': pending},
        'accounting': 'Coverage references only; Session/Event/artifact records remain the sole hardware action ledger.',
        'cases': cases,
    }


if __name__ == '__main__':
    value = scan()
    target = REPORT / 'evidence/selected_coverage_audit.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, target)
    print(json.dumps(value['summary'], indent=2))
