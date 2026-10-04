"""Real S6 NPU followup for the six requested backend/model combinations."""
import hashlib
import json
from pathlib import Path
import sys

BENCH = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(BENCH / 'evaluator'), str(BENCH)]
import common
import session
import workspace
from launcher import runner
import yaml

MODELS = ('glm-5.3-flash', 'deepseek-flash', 'gpt-6.1-sol')
CASE = 'deepseek_v4.rmsnorm_decode_flash'


def scope_hash():
    """Read every byte of this followup's canonical Case, assets and environment."""
    case_dir, sealed, case = common.resolve_case(CASE)
    roots = [case_dir, sealed, BENCH / 'environments' / (case['environment'] + '.json')]
    roots += sorted(sealed.parent.glob('*.py'))
    entries = {}
    for root in roots:
        files = sorted(root.rglob('*')) if root.is_dir() else [root]
        for path in files:
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            size = 0
            with path.open('rb') as reader:
                for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
                    size += len(chunk)
            entries[str(path.relative_to(BENCH))] = {'sha256': digest.hexdigest(), 'bytes': size}
    return entries


def prompt(experiment, record, case):
    location = str(Path(record['session_root']) / record['workspaces']['worker_000']['path'])
    return ('Integration acceptance task. Your registered workspace is ' + location + '. '
            'Read .benchmark_context/README.md and use your shell tool to run exactly one '
            'ptobench evaluate in this workspace. Wait for its result, then exit. '
            'Do not optimize or edit any source, and do not change to the repository root. '
            'PATH and Session environment are already set. Do not call internal scripts or inspect sealed assets. '
            'The benchmark will independently finalize on exit. Exploration budget is 2; finish budget is 1.')


def run(backend, model):
    label = backend + '_' + model
    profile = 'default' if model == 'glm-5.3-flash' else model
    experiment = HERE / (label + '.yaml')
    experiment.write_text(yaml.safe_dump({'case': CASE, 'optimizer':
        {'id': 's6_provider_' + label, 'backend': backend, 'profile': profile},
        'output': {'runs_root': str(BENCH / 'runs'), 'snapshots_root': str(BENCH / 'snapshots')},
        'budget': {'max_hardware_actions': 2, 'finish_actions': 1}}))
    runner.make_prompt = prompt
    value, code = runner.run(experiment)
    root = Path(value['session_root'])
    # Completed records are inspected directly; old Sessions must not restore public code.
    record = common.read_json(root / 'session.json')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    identity = common.read_json(root / 'optimizer_process.json')
    final = common.read_json(root / 'final_result.json')
    raw = common.read_json(root / 'final/authority_result.json') if (root / 'final/authority_result.json').exists() else {}
    seed_hash = workspace.source_manifest(root / 'workspaces/worker_000')['content_sha256']
    checks = {'launcher_completed': code == 0 and final['status'] == 'completed',
        'optimizer_exit_zero': value['optimizer_exit_code'] == 0,
        'one_exploration_one_finish': record['usage'] == {'hardware_actions': 1, 'finish_actions': 1},
        'evaluate_then_finish': [e['action'] for e in events] == ['evaluate', 'finish'],
        'all_events_passed': bool(events) and all(e['error_code'] is None and e['integrity_passed'] for e in events),
        'source_unchanged': bool(events) and all(e['content_sha256'] == seed_hash for e in events),
        'no_active_action': record['active_action'] is None,
        'optimizer_group_stopped': not session._group_alive(identity),
        'fresh_clean_final': raw.get('candidate') == str(root / 'final/workspace') and raw.get('profiling_mode') == 'clean',
        'ten_final_samples': len((raw.get('runs') or [{}])[0].get('samples_effective_us') or []) == 10}
    result = {'backend': backend, 'model': model, 'profile': profile, 'case': CASE,
        'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'exit_code': code, 'launcher_result': value, 'usage': record['usage'],
        'events': [e['action'] for e in events], 'final': final}
    common.write_json(HERE / (label + '_acceptance.json'), result)
    print(json.dumps({'backend': backend, 'model': model, 'status': result['status'],
        'session_root': str(root), 'latency_us': final['final_verified_latency_us']}), flush=True)
    return result


if __name__ == '__main__':
    before = scope_hash()
    common.write_json(HERE / 'canonical_before_followup.json', before)
    results = []
    for backend in ('pi', 'opencode'):
        for model in MODELS:
            print('START', backend, model, flush=True)
            results.append(run(backend, model))
            common.write_json(HERE / 'acceptance_matrix.json', results)
    after = scope_hash()
    common.write_json(HERE / 'canonical_after_followup.json', after)
    common.write_json(HERE / 'canonical_followup_check.json', {
        'status': 'PASS' if before == after else 'FAIL', 'case': CASE,
        'files': len(before), 'bytes': sum(x['bytes'] for x in before.values()),
        'changed_paths': sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p)),
        'global_s6_hash_gate': 'Historical FAIL retained; this is a scoped followup check.'})
