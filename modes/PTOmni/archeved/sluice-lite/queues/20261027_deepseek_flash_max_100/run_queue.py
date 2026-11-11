#!/usr/bin/env python3
"""Run the prepared SLUICE Lite experiments serially in one tmux session."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

QUEUE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path('/data1/home/ziruiwang/pypto-lib')
PTOBENCH_ROOT = REPO_ROOT / 'benchmarks/PTOBench'
PYTHON = REPO_ROOT / '.venv-official/bin/python'
MANIFEST_PATH = QUEUE_DIR / 'manifest.json'
STATE_PATH = QUEUE_DIR / 'state.json'
QUEUE_LOG = QUEUE_DIR / 'queue.log'


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save_state(state):
    temporary = STATE_PATH.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE_PATH)


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    if STATE_PATH.exists():
        existing = json.loads(STATE_PATH.read_text())
        if existing.get('status') != 'prepared':
            raise SystemExit(f"Queue already started (status={existing.get('status')}); inspect state.json")
        state = existing
    else:
        state = {**manifest, 'created_at': now()}
        save_state(state)
    if not PYTHON.is_file():
        raise SystemExit(f'PTOBench Python environment is missing: {PYTHON}')

    env = os.environ.copy()
    env['PTOBENCH_ROOT'] = str(PTOBENCH_ROOT)
    env['PTOBENCH_REPO_ROOT'] = str(REPO_ROOT)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PATH'] = str(PTOBENCH_ROOT) + os.pathsep + env.get('PATH', '')
    failures = []
    state['status'] = 'running'
    state['started_at'] = now()
    save_state(state)

    with QUEUE_LOG.open('a', buffering=1) as queue_log:
        def emit(text):
            sys.stdout.write(text)
            sys.stdout.flush()
            queue_log.write(text)
            queue_log.flush()

        emit(f"QUEUE START {state['queue']} at {state['started_at']}\n")
        emit(f"Mode: sluice-lite | Pi profile: deepseek-flash-max | reasoning: max\n")
        emit('Execution: serial | each case: 100 exploration actions + 1 final verification action\n')
        for item in state['cases']:
            case_id = item['case']
            config = QUEUE_DIR / item['config']
            case_slug = case_id.replace('.', '_')
            log_path = QUEUE_DIR / 'logs' / f'{case_slug}.log'
            item['status'] = 'running'
            item['started_at'] = now()
            item['log'] = str(log_path)
            save_state(state)
            emit(f"\n===== START {case_id} {item['started_at']} =====\n")
            with log_path.open('w', buffering=1) as case_log:
                process = subprocess.Popen(
                    [str(PYTHON), str(PTOBENCH_ROOT / 'pto-launcher'), 'run', str(config)],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    case_log.write(line)
                    emit(line)
                return_code = process.wait()
            item['return_code'] = return_code
            item['finished_at'] = now()
            item['status'] = 'succeeded' if return_code == 0 else 'failed'
            if return_code != 0:
                failures.append(case_id)
            save_state(state)
            emit(f"===== END {case_id} exit={return_code} at {item['finished_at']} =====\n")

        state['status'] = 'complete' if not failures else 'complete_with_failures'
        state['finished_at'] = now()
        state['failed_cases'] = failures
        save_state(state)
        emit(f"\nQUEUE {state['status'].upper()} at {state['finished_at']}\n")
        if failures:
            emit('Failed cases: ' + ', '.join(failures) + '\n')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
