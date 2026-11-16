"""After the live collector exits, validate inputs and run exactly the six planned trials.

This only orchestrates the normal CLI. PTOBench Session/Event records are authoritative.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BENCH = Path(__file__).resolve().parents[5]
MODE = BENCH / 'modes/basic_wiki-tool'
REPORT = MODE / 'reports/stage5'
OUTPUT = Path('/data2/ziruiwang/PTOBench/stage5_trials')
RECOVERY_SESSIONS = {
    1: ('20261115T013039Z_41d365f68093', 'completed'),
    2: ('20261115T021536Z_aecc178a0ae4', 'failed_before_agent_start'),
}
sys.path[:0] = [str(BENCH), str(BENCH / 'evaluator')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def validate_inputs():
    from modes._shared import provider, recording
    from modes._shared.config import load_profile
    from workspace import source_manifest
    manifest = json.loads((REPORT / 'experiment_manifest.json').read_text())
    inputs = json.loads((REPORT / 'inputs.json').read_text())
    differences = [name for name, expected in inputs['source_sha256'].items()
                   if not (BENCH / name).is_file() or sha(BENCH / name) != expected]
    if differences:
        raise RuntimeError('Frozen public/Basic/Analyzer/Wiki inputs changed: ' + ', '.join(differences))
    for name, expected in json.loads((MODE / 'analyzer_manifest.json').read_text())['files'].items():
        if sha(MODE / 'pto_analyzer' / name) != expected['copy_sha256']:
            raise RuntimeError(f'Analyzer runtime identity changed: {name}')
    for row in manifest['configs']:
        if sha(BENCH / row['configuration']) != row['configuration_sha256']:
            raise RuntimeError(f'Config identity changed: {row["configuration"]}')
    for case_id, row in manifest['case_inputs'].items():
        case_path = BENCH / 'cases' / case_id.replace('.', '/') / 'case.json'
        if sha(case_path) != row['case_sha256'] or sha(BENCH / row['environment']) != row['environment_sha256']:
            raise RuntimeError(f'Case/environment identity changed: {case_id}')
        if source_manifest(BENCH / row['seed']) != row['initial_seed_manifest']:
            raise RuntimeError(f'Canonical seed identity changed: {case_id}')
    for mode_id, expected in manifest['paired_effective_profiles'].items():
        configured = recording.prepare_profile(provider.configure(load_profile('pi', 'default'), BENCH / 'modes' / mode_id))
        actual = {key: configured.get(key) for key in ('model', 'reasoning', 'context_window', 'max_output_tokens', 'argv')}
        if any(actual[key] != expected[key] for key in actual):
            raise RuntimeError(f'Effective model/reasoning/limits/argv changed for {mode_id}')
        # mode_provider contains the same nonsecret Provider/model configuration frozen in the manifest.
        mode_provider = configured.get('mode_provider') or {}
        actual_provider = {key: mode_provider.get(key) for key in expected['provider']}
        if actual_provider != expected['provider']:
            raise RuntimeError(f'Effective Provider configuration changed for {mode_id}')
    pi = shutil.which('pi')
    if pi is None or sha(Path(pi).resolve()) != manifest['pi_cli_sha256']:
        raise RuntimeError('Pi CLI identity changed')
    return manifest


def validate_trial_recovery(manifest):
    """Reuse the known Basic success; allow one zero-action retry for the socket failure."""
    states = {}
    for row in manifest['configs']:
        parent = Path(row['expected_runs_root']) / row['optimizer_id'] / row['case_id']
        sessions = sorted(parent.glob('*/session.json')) if parent.exists() else []
        order = row['order']
        if order in (1, 2):
            expected_id, expected_state = RECOVERY_SESSIONS[order]
            if len(sessions) != 1 or sessions[0].parent.name != expected_id:
                raise RuntimeError(f'Trial {order} has unexpected prior Sessions; inspect before continuing')
            root = sessions[0].parent
            record = json.loads(sessions[0].read_text())
            if order == 1:
                valid = (record.get('status') == 'completed' and record.get('mode', {}).get('id') == 'basic'
                         and record.get('profile_strategy') == 'combined'
                         and record.get('usage', {}).get('hardware_actions') == 17)
            else:
                result_path = root / 'launcher_result.json'
                result = json.loads(result_path.read_text()) if result_path.is_file() else {}
                agent = json.loads((root / 'agent/session.json').read_text()) if (root / 'agent/session.json').is_file() else {}
                valid = (record.get('status') == 'failed' and record.get('mode', {}).get('id') == 'basic_wiki-tool'
                         and record.get('profile_strategy') == 'combined'
                         and record.get('usage', {}).get('hardware_actions') == 0
                         and record.get('next_event_seq') == 1
                         and not (root / 'events.jsonl').read_text().strip()
                         and result.get('optimizer_exit_code') == 5
                         and result.get('error_code') == 'recording_incomplete'
                         and agent.get('status') == 'missing'
                         and not list((root / 'agent/native').glob('*')))
            if not valid:
                raise RuntimeError(f'Trial {order} prior Session does not match the recorded {expected_state} recovery evidence')
            states[order] = expected_state
        elif sessions or parent.exists():
            raise RuntimeError(f'Trial {order} already has output; refusing an accidental duplicate')
        else:
            states[order] = 'not_started'
    return states


def main():
    pid, start = int(sys.argv[1]), sys.argv[2]
    print(json.dumps({'waiting_for_collector_pid': pid, 'start_ticks': start}), flush=True)
    while process_start(pid) == start:
        time.sleep(30)
    print('Collector process is terminal; rebuilding coverage from original metadata.', flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('audit_profiles.py'))], cwd=BENCH, check=True)
    coverage = json.loads((REPORT / 'evidence/selected_coverage_audit.json').read_text())
    if coverage['summary']['pending_collection']:
        raise RuntimeError('Collector left unattempted Cases; inspect its original progress log before launching trials')
    manifest = validate_inputs()
    states = validate_trial_recovery(manifest)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'inputs_revalidated': True, 'recovery_states': states, 'coverage': coverage['summary'], 'ordered_configs': [r['configuration'] for r in manifest['configs']]}), flush=True)
    env = dict(os.environ)
    env.update({
        'PATH': os.pathsep.join((str(BENCH.parents[1] / '.venv-official/bin'), str(BENCH), env.get('PATH', ''))),
        'PTOBENCH_ROOT': str(BENCH), 'PTOBENCH_REPO_ROOT': str(BENCH.parents[1]),
        'PTOBENCH_CANN_ROOT': env.get('PTOBENCH_CANN_ROOT', '/usr/local/Ascend/cann-9.0.0/cann'),
        'PTOBENCH_PTOAS_ROOT': env.get('PTOBENCH_PTOAS_ROOT', '/data1/home/ziruiwang/ptoas-v0.60'),
        'PTOBENCH_MINIFORGE_ROOT': env.get('PTOBENCH_MINIFORGE_ROOT', '/data1/home/ziruiwang/miniforge3'),
    })
    for row in manifest['configs']:
        if states[row['order']] == 'completed':
            print(json.dumps({'reusing_completed_trial': row['order'], 'session': RECOVERY_SESSIONS[row['order']][0]}), flush=True)
            continue
        retry = states[row['order']] == 'failed_before_agent_start'
        log_name = f"{row['order']:02d}_{row['optimizer_id']}{'_retry1' if retry else ''}.log"
        log = OUTPUT / log_name
        print(json.dumps({'starting_trial': row['order'], 'case': row['case_id'], 'mode': row['mode'], 'config': row['configuration'], 'log': str(log)}), flush=True)
        with log.open('w') as stream:
            result = subprocess.run([str(BENCH / 'pto-launcher'), 'run', row['configuration']], cwd=BENCH, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        print(json.dumps({'finished_trial': row['order'], 'return_code': result.returncode, 'log': str(log)}), flush=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(f'Trial {row["order"]} returned {result.returncode}; inspect original Session/log before an affected retry')
    print('All six CLI invocations are terminal; analyze their original Sessions and write the Stage 5 report.', flush=True)


if __name__ == '__main__':
    main()
