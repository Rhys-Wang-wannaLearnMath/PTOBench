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
        if (Path(row['expected_runs_root']) / row['optimizer_id']).exists():
            raise RuntimeError(f'Trial output already exists; inspect original Session instead of restarting: {row["optimizer_id"]}')
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'inputs_revalidated': True, 'coverage': coverage['summary'], 'ordered_configs': [r['configuration'] for r in manifest['configs']]}), flush=True)
    env = dict(os.environ)
    env.update({
        'PATH': os.pathsep.join((str(BENCH.parents[1] / '.venv-official/bin'), str(BENCH), env.get('PATH', ''))),
        'PTOBENCH_ROOT': str(BENCH), 'PTOBENCH_REPO_ROOT': str(BENCH.parents[1]),
        'PTOBENCH_CANN_ROOT': env.get('PTOBENCH_CANN_ROOT', '/usr/local/Ascend/cann-9.0.0/cann'),
        'PTOBENCH_PTOAS_ROOT': env.get('PTOBENCH_PTOAS_ROOT', '/data1/home/ziruiwang/ptoas-v0.60'),
        'PTOBENCH_MINIFORGE_ROOT': env.get('PTOBENCH_MINIFORGE_ROOT', '/data1/home/ziruiwang/miniforge3'),
    })
    for row in manifest['configs']:
        log = OUTPUT / f"{row['order']:02d}_{row['optimizer_id']}.log"
        print(json.dumps({'starting_trial': row['order'], 'case': row['case_id'], 'mode': row['mode'], 'config': row['configuration'], 'log': str(log)}), flush=True)
        with log.open('w') as stream:
            result = subprocess.run([str(BENCH / 'pto-launcher'), 'run', row['configuration']], cwd=BENCH, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        print(json.dumps({'finished_trial': row['order'], 'return_code': result.returncode, 'log': str(log)}), flush=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(f'Trial {row["order"]} returned {result.returncode}; inspect original Session/log before an affected retry')
    print('All six CLI invocations are terminal; analyze their original Sessions and write the Stage 5 report.', flush=True)


if __name__ == '__main__':
    main()
