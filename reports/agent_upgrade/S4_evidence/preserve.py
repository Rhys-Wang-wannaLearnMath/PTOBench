"""Read-only worktree preservation evidence for S4; tensors use metadata only."""
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[5]
bench = repo / 'benchmarks/pypto_lib_bench-v0.2'
evidence = Path(__file__).parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory():
    canonical = {}
    for root in ('cases', 'sealed_assets', 'environments'):
        for path in sorted((bench / root).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
                continue
            st = path.stat()
            value = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'mode': st.st_mode}
            if path.suffix != '.pt':
                value['sha256'] = digest(path)
            canonical[str(path.relative_to(bench))] = value
    tracked = {}
    for name in subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().split('\0'):
        if name and (repo / name).is_file():
            tracked[name] = digest(repo / name)
    preserved = {}
    for root in (bench / 'evaluator', bench / 'reports/agent_upgrade/S0_evidence', bench / 'reports/agent_upgrade/S1_evidence', bench / 'reports/agent_upgrade/S2_evidence', bench / 'reports/agent_upgrade/S3_evidence'):
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'):
                preserved[str(path.relative_to(repo))] = digest(path) if path.is_file() else None
    for path in (bench / 'bench', bench / 'reports/agent_upgrade/S0_baseline.md', bench / 'reports/agent_upgrade/S1_file_integrity.md', bench / 'reports/agent_upgrade/S2_session_and_budget.md', bench / 'reports/agent_upgrade/S3_cli_events_and_snapshots.md', bench / 'README.md', bench / 'ptobench'):
        preserved[str(path.relative_to(repo))] = digest(path) if path.is_file() else None
    return {'canonical': canonical, 'tracked': tracked, 'preserved': preserved,
            'index': subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=repo).decode()}

if sys.argv[1] == 'before':
    (evidence / 'before.json').write_text(json.dumps(inventory(), indent=2) + '\n')
    (evidence / 'git_status_before.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=repo))
    print('S4 preservation baseline captured.')
else:
    before = json.loads((evidence / 'before.json').read_text())
    after = inventory()
    result = {key + '_changed_paths': sorted(path for path in before[key].keys() | after[key].keys()
              if before[key].get(path) != after[key].get(path)) for key in ('canonical', 'tracked')}
    result['previous_public_and_history_changed_paths'] = sorted(path for path, value in before['preserved'].items() if after['preserved'].get(path) != value)
    result['index_unchanged'] = before['index'] == after['index']
    result['canonical_file_count'] = len(before['canonical'])
    result['tensor_metadata_count'] = sum('.pt' == Path(path).suffix for path in before['canonical'])
    result['non_tensor_content_hash_count'] = result['canonical_file_count'] - result['tensor_metadata_count']
    result['tensor_verification'] = 'size/mtime_ns/mode only; no full byte hashes'
    (evidence / 'preservation.json').write_text(json.dumps(result, indent=2) + '\n')
    (evidence / 'git_status_after.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=repo))
    print(json.dumps(result, indent=2))
