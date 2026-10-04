"""Real native-CLI read-tool probes, with private credentials redacted."""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = Path(__file__).resolve().parent
BRIDGE = ROOT / 'benchmarks/mig_workflow/agent_provider/launch.py'
MODELS = ('glm-5.3-flash', 'deepseek-flash', 'gpt-6.1-sol')


def probe(pair):
    backend, model = pair
    label = backend + '_' + model
    expected = 'READY_' + secrets.token_hex(8)
    prompt = 'Use the read tool to read model-test.txt in the current directory. Reply with exactly its content.'
    argv = [str(BRIDGE), backend]
    if backend == 'pi':
        argv += ['--print', '--mode', 'json', '--no-session', '--no-extensions',
                 '--no-skills', '--no-context-files', '--tools', 'read',
                 '--provider', 'profile-gateway', '--model', model, prompt]
    else:
        argv += ['run', '--format', 'json', '--model', 'profile-gateway/' + model, prompt]
    start = time.monotonic()
    row = {'backend': backend, 'model': model, 'argv': argv}
    with tempfile.TemporaryDirectory(prefix='pypto-provider-') as directory:
        Path(directory, 'model-test.txt').write_text(expected + '\n')
        child = subprocess.Popen(argv, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 start_new_session=True, stdin=subprocess.DEVNULL)
        try:
            raw, _ = child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                raw, _ = child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                raw, _ = child.communicate()
            row['timeout'] = True
    key = json.loads((ROOT / 'benchmarks/mig_workflow/profiles_codex/glm-5.3-flash-max/auth.json').read_text())['OPENAI_API_KEY']
    output = raw.decode(errors='replace').replace(key, '[REDACTED]')
    output = re.sub(r'(?i)(Bearer\s+\S+|sk-[A-Za-z0-9_-]+)', '[REDACTED]', output)
    (EVIDENCE / (label + '.log')).write_text(output)
    events = []
    for line in output.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    if backend == 'pi':
        tools = [e.get('toolName') for e in events if e.get('type') == 'tool_execution_start']
        texts = [c.get('text', '') for e in events if e.get('type') == 'message_end'
                 and e.get('message', {}).get('role') == 'assistant'
                 for c in e['message'].get('content', []) if c.get('type') == 'text']
        models = sorted({e['message']['model'] for e in events
                         if e.get('message', {}).get('role') == 'assistant' and e['message'].get('model')})
    else:
        tools = [e.get('part', {}).get('tool') for e in events if e.get('type') == 'tool_use'
                 and e.get('part', {}).get('state', {}).get('status') == 'completed']
        texts = [e.get('part', {}).get('text', '') for e in events if e.get('type') == 'text']
        models = []
    row.update(exit_code=child.returncode, tool_names=tools, returned_models=models,
               token_received=expected in '\n'.join(texts),
               elapsed_seconds=round(time.monotonic() - start, 3),
               log=str(EVIDENCE / (label + '.log')))
    row['status'] = 'PASS' if child.returncode == 0 and row['token_received'] and 'read' in tools else 'FAIL'
    print(json.dumps(row), flush=True)
    return row


if __name__ == '__main__':
    pairs = [(backend, model) for backend in ('pi', 'opencode') for model in MODELS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(probe, pairs))
    (EVIDENCE / 'agent_probes.json').write_text(json.dumps(rows, indent=2) + '\n')
