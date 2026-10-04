"""Read-only S3 acceptance audit; writes only this stage's audit summary."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

repo = Path(__file__).resolve().parents[5]
bench = repo / 'benchmarks/pypto_lib_bench-v0.2'
evidence = Path(__file__).parent
before = json.loads((evidence / 'before.json').read_text())
preservation = json.loads((evidence / 'preservation.json').read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}

def literal(path, name):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))

allowed = {str((bench / name).relative_to(repo)) for name in (
    'bench', 'evaluator/bench.py', 'evaluator/README.md', 'evaluator/SESSION.md',
    'evaluator/common.py', 'evaluator/evaluate.py', 'evaluator/profile_modes.py',
    'evaluator/session.py', 'evaluator/workspace.py',
    'evaluator/tests/session_test_support.py', 'evaluator/tests/test_session.py',
)}
assert preservation['index_unchanged'] and not preservation['tracked_changed_paths']
assert set(preservation['previous_public_and_history_changed_paths']) <= allowed
history = {path: value for path, value in before['preserved'].items() if '/reports/agent_upgrade/' in path}
assert all(digest(repo / path) == value for path, value in history.items())
for name in ('integrity.py', 'file_tree.py', 'workflow.py'):
    path = bench / 'evaluator' / name
    assert digest(path) == before['preserved'][str(path.relative_to(repo))], name
old_workspace = functions(evidence / 'workspace.py.before')
new_workspace = functions(bench / 'evaluator/workspace.py')
copy_functions = ('_source_entries', '_source_manifest', 'source_manifest', '_copy_source_file',
                  '_copy_source_tree', 'copy_candidate_tree')
assert all(old_workspace[name] == new_workspace[name] for name in copy_functions)
old_common = functions(evidence / 'common.py.before')
new_common = functions(bench / 'evaluator/common.py')
assert all(new_common[name] == body for name, body in old_common.items())
assert set(new_common) - set(old_common) == {'await_action_start'}
old_eval = ast.parse((evidence / 'evaluate.py.before').read_text())
new_eval = ast.parse((bench / 'evaluator/evaluate.py').read_text())
main = next(node for node in new_eval.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
gate = main.body.pop(0)
assert ast.dump(gate) == ast.dump(ast.parse('common.await_action_start()').body[0])
assert ast.dump(old_eval) == ast.dump(new_eval)
for name in ('PROFILE_MODES', 'ENABLED_PROFILE_MODES'):
    assert literal(evidence / 'profile_modes.py.before', name) == literal(bench / 'evaluator/profile_modes.py', name)
assert not os.path.lexists(bench / 'bench') and not os.path.lexists(bench / 'evaluator/bench.py')
assert os.access(bench / 'ptobench', os.X_OK)
assert set(functions(bench / 'evaluator/snapshots.py')) == {'create_snapshot'}
for name in ('workspace.py', 'prepare_ref.py', 'evaluate.py', 'generate_fixtures.py'):
    assert (bench / 'evaluator' / name).is_file()

log = (evidence / 'unittest_final.log').read_text()
match = re.search(r'Ran (\d+) tests in ([\d.]+)s', log)
assert match and int(match[1]) == 100 and '\nOK\n' in log and '\nFAILED' not in log
test_counts = {}
for path in sorted((bench / 'evaluator/tests').glob('test_*.py')):
    tree = ast.parse(path.read_text())
    test_counts[path.name] = sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                               for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body)
assert sum(test_counts.values()) == 100

fake = evidence / 'fake_actions'
events = [json.loads(line) for line in (fake / 'events.jsonl').read_text().splitlines()]
calls = [json.loads(line) for line in (fake / 'evaluator_calls.jsonl').read_text().splitlines()]
assert len(events) == 7 and len(calls) == 5
assert [event['seq'] for event in events] == list(range(1, 8))
assert sum(event['hardware_action'] for event in events) == 5
assert events[4]['action'] == 'fork' and not events[4]['hardware_action']
assert events[5]['state'] == 'rejected' and events[5]['error_code'] == 'invalid_request'
assert events[6]['error_code'] == 'action_failed' and events[6]['snapshot_id'] is None
for event in events[:4]:
    snapshot = fake / 'snapshots' / event['snapshot_id']
    metadata = json.loads((snapshot / 'metadata.json').read_text())
    assert metadata['event_seq'] == event['seq'] and metadata['content_sha256'] == event['content_sha256']
    assert metadata['observation'] == event['observation']
    assert metadata['observation']['latency_us'] == (10.0 if event['action'] == 'evaluate' else None)
    assert not (snapshot / 'source/.benchmark_context').exists()
    assert not (snapshot / 'source/.benchmark_outputs').exists()
    files = [{'path': str(p.relative_to(snapshot / 'source')), 'sha256': digest(p)}
             for p in sorted((snapshot / 'source').rglob('*')) if p.is_file()]
    encoded = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(encoded).hexdigest() == event['content_sha256']
    for path in [snapshot, *snapshot.rglob('*')]:
        assert path.stat().st_mode & 0o777 == (0o555 if path.is_dir() else 0o444)
remaining = []
for path in Path('/proc').iterdir():
    if path.name.isdecimal():
        try:
            command = (path / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(re.fullmatch(rb'/tmp/s2_[^/]+/benchmarks/pypto_lib_bench-v0\.2/evaluator/evaluate\.py', arg) for arg in command):
            remaining.append(int(path.name))
assert not remaining, remaining

source_names = ('ptobench', 'README.md', 'evaluator/AGENT_CLI.md', 'evaluator/README.md',
                'evaluator/SESSION.md', 'evaluator/agent_cli.py', 'evaluator/session_actions.py',
                'evaluator/snapshots.py', 'evaluator/session.py', 'evaluator/common.py',
                'evaluator/evaluate.py', 'evaluator/profile_modes.py', 'evaluator/workspace.py',
                'evaluator/tests/action_test_support.py', 'evaluator/tests/test_agent_cli.py',
                'evaluator/tests/test_session_actions.py', 'evaluator/tests/test_snapshots.py',
                'evaluator/tests/session_test_support.py', 'evaluator/tests/test_session.py')
sources = {name: digest(bench / name) for name in source_names}
artifact_hashes = {str(path.relative_to(evidence)): digest(path) for path in sorted(evidence.rglob('*'))
                   if path.is_file() and path.name not in {'validation_summary.json', 'validation_final.log'}}
report = bench / 'reports/agent_upgrade/S3_cli_events_and_snapshots.md'
summary = {
    'stage': 'S3', 'status': 'PASS', 'audit_time_utc': datetime.now(timezone.utc).isoformat(),
    'full_suite': {'exit_code': 0, 'tests': int(match[1]), 'seconds': float(match[2]), 'skipped': 0},
    'test_counts': test_counts, 's3_new_tests': 26, 'existing_regression_tests': 74,
    'old_entries_absent': True, 'legacy_session_restoration_blocked': True,
    'maintenance_modules_retained': True, 's1_integrity_copy_protocol_unchanged': True,
    'measurement_runconfig_bootstrap_unchanged': True,
    'common_only_added_start_gate': True, 'evaluate_only_added_start_gate_call': True,
    'historical_files_hash_checked': len(history), 'historical_files_unchanged': True,
    'unexpected_public_history_changes': [], 'tracked_unchanged': True, 'index_unchanged': True,
    'canonical_delta_observed': len(preservation['canonical_changed_paths']),
    'canonical_delta_groups': dict(Counter('/'.join(p.split('/')[:3]) for p in preservation['canonical_changed_paths'])),
    'canonical_delta_provenance': 'Concurrent external Case work observed; this task writes only public layer/docs/S3 evidence. No global unchanged claim.',
    'tensor_verification': preservation['tensor_verification'],
    'archived_fake_evidence': {'cli_operations': 7, 'evaluator_calls': 5, 'snapshots': 4, 'usage': 5, 'profile_latencies': [None]*3},
    'remaining_fake_evaluators': remaining, 'source_sha256': sources, 'evidence_sha256': artifact_hashes,
    'report_sha256': digest(report), 'real_hardware': 'NOT_RUN', 'real_optimizer': 'NOT_RUN',
    's4_ready': True, 's4_implemented': False,
}
(evidence / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ('stage', 'status', 'full_suite', 'historical_files_hash_checked',
                  'canonical_delta_observed', 'old_entries_absent', 'remaining_fake_evaluators', 's4_ready')}, indent=2))
