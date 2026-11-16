import importlib
import json
import os
from pathlib import Path
import signal
import pwd
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from conftest import MODE, ROOT, hashes


def runner_module():
    assert (MODE / 'runner.py').is_file(), 'Task 3 Mode not implemented'
    return importlib.import_module('modes.basic_wiki-tool.runner')


@pytest.fixture
def mode_request():
    with tempfile.TemporaryDirectory(prefix='s5_') as temporary:
        root = Path(temporary) / 'session'
        workspace = root / 'workspaces/worker_000'
        (workspace / 'candidate').mkdir(parents=True)
        (workspace / '.benchmark_context').mkdir()
        (workspace / 'candidate/candidate.py').write_text('pass\n')
        (workspace / '.benchmark_context/README.md').write_text('original context')
        (root / 'session.json').write_text(json.dumps({'session_id': 'fake', 'case_id': 'fixture.case', 'session_root': str(root)}))
        (root / 'events.jsonl').write_text('')
        yield {'experiment': {'mode': 'basic_wiki-tool', 'case': 'fixture.case',
                             'optimizer': {'backend': 'pi', 'profile': 'default'}},
               'context': {'session_root': str(root), 'workspace': str(workspace),
                           'case': {'case_id': 'fixture.case', 'module': 'candidate.py', 'entry': 'entry'},
                           'budget': {'max_hardware_actions': 20, 'finish_actions': 1}},
               'started': '2026-11-15T00:00:00Z', 'close_reason': 'agent_exited'}


def materialize(root, response):
    for name, value in response['records'].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))


def test_mode_wires_tools_and_preserves_basic_contract(mode_request, monkeypatch):
    runner = runner_module()
    import agent_launch
    import yaml
    request = mode_request
    root = Path(request['context']['session_root'])
    workspace = Path(request['context']['workspace'])
    monkeypatch.setattr(runner, 'get_adapter', lambda backend: SimpleNamespace(preflight=lambda p: SimpleNamespace(version='fake-version', to_dict=lambda: {'backend': 'pi'})))
    with pytest.raises(runner.ConfigError, match='Pi'):
        runner.preflight({'experiment': {'optimizer': {'backend': 'codex', 'profile': 'default'}}})
    request['preflight'] = runner.preflight(request)
    monkeypatch.setattr(runner.provider, 'prepare', lambda p, *a, **k: p)
    monkeypatch.setattr(runner.provenance, 'freeze_agent', lambda *a: {'fixture': True})
    monkeypatch.setattr(runner.provider, 'credentials', lambda p: {})
    monkeypatch.setattr(runner, 'render', lambda p, w, r, prompt: (['fake-agent', prompt], {}))
    before = hashes(workspace)
    prepared = runner.prepare(request)
    assert yaml.safe_load((MODE / 'mode.yaml').read_text())['profile_strategy'] == 'combined'
    assert yaml.safe_load((MODE / 'mode.yaml').read_text())['schema_version'] == 2
    assert (MODE / 'instructions.md').read_bytes() == (ROOT / 'modes/basic/instructions.md').read_bytes()
    prompt = prepared['records']['prompt.txt']
    from modes.basic.prompt import make_prompt
    assert prompt.startswith(make_prompt(request['experiment'], request['context'], request['context']['case'], (MODE / 'instructions.md').read_text()))
    assert 'ptobench profile .' in prompt and 'profile' in prompt and 'INDEX.md' in prompt
    assert 'method/resources' in prepared['archives'].values()
    assert all(not p.startswith('workspaces/') for p in prepared['archives'].values())
    assert 'method/analyzer_calls' not in prepared['archives'].values()
    assert {p: hashes(workspace)[p] for p in before} == before
    materialize(root, prepared)
    captured = []
    def command(argv, **kwargs):
        assert kwargs['backend'] == 'pi'
        assert kwargs['pi_tool_extension'] == root / 'agent/pi-tools/pi_analyzer.ts'
        assert kwargs['pi_tool_socket'] == root / 'agent/analyzer.sock'
        assert kwargs['pi_tool_socket'].is_socket()
        assert len(json.loads((root / 'agent/pi-tools/pto-analyzer-specs.json').read_text())) == 9
        captured.append(kwargs)
        return argv, kwargs['environment']
    monkeypatch.setattr(runner.isolation, 'agent_command', command)
    monkeypatch.setattr(runner.isolation, 'require_agent_isolation', lambda p: None)
    monkeypatch.setattr(agent_launch, 'run_process', lambda *a, **k: 0)
    assert runner.run(request) == 0
    assert not (root / 'agent/analyzer.sock').exists()
    def failed(*a, **k):
        raise RuntimeError('fake agent startup failed')
    monkeypatch.setattr(agent_launch, 'run_process', failed)
    with pytest.raises(RuntimeError, match='startup failed'):
        runner.run(request)
    assert not (root / 'agent/analyzer.sock').exists() and len(captured) == 2
    # Simulate a hard-killed Broker: collect must finalize the already-durable request.
    pending = root / 'method/analyzer_calls/pending'
    pending.mkdir()
    (pending / 'request.json').write_text(json.dumps({'tool_call_id': 'lost', 'name': 'tile_access_check', 'arguments': {}, 'started_at': 'test'}))
    monkeypatch.setattr(runner.recording, 'collect', lambda *a, **k: {'status': 'recorded'})
    collected = runner.collect(request)
    assert collected['recording']['status'] == 'recorded'
    assert json.loads((pending / 'result.json').read_text())['status'] == 'interrupted'
    assert 'method/analyzer_calls' in collected['archives'].values()
    assert not set(prepared['archives']) & set(collected['archives'])
    assert collected['records']['method/wiki_verification.json']['unchanged']


def test_sigterm_closes_socket(mode_request, tmp_path):
    runner_module()
    request = mode_request
    root = Path(request['context']['session_root'])
    request['preflight'] = {'profile': {'backend': 'pi'}}
    (root / 'prompt.txt').write_text('fake')
    request_path = root / 'request.json'
    request_path.write_text(json.dumps(request))
    script = tmp_path / 'signal_driver.py'
    script.write_text(f'''
import importlib, json, pathlib, signal
runner = importlib.import_module('modes.basic_wiki-tool.runner')
import agent_launch
runner.isolation.require_agent_isolation = lambda p: None
runner.provider.credentials = lambda p: {{}}
runner.render = lambda *a: (['fake-agent'], {{}})
runner.isolation.agent_command = lambda argv, **kw: (argv, kw['environment'])
def wait(*a, **kw):
    pathlib.Path({str(root / 'ready')!r}).write_text('ready')
    signal.pause()
agent_launch.run_process = wait
runner.run(json.loads(pathlib.Path({str(request_path)!r}).read_text()))
''')
    process = subprocess.Popen([sys.executable, str(script)], cwd=ROOT, env={**os.environ, 'PYTHONPATH': f'{ROOT}:{ROOT / "evaluator"}:{MODE}'}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (root / 'ready').exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.01)
        assert (root / 'ready').exists(), process.communicate(timeout=2)
        assert (root / 'agent/analyzer.sock').is_socket()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 128 + signal.SIGTERM, stdout + stderr
        assert not (root / 'agent/analyzer.sock').exists()
    finally:
        if process.poll() is None:
            process.kill(); process.communicate()


def test_wiki_visible_readonly_without_context_changes():
    runner = runner_module()
    import common
    import integrity
    import isolation
    import agent_launch
    import session
    import sandbox_supervisor
    from session_test_support import SessionTestBase
    helper = SessionTestBase('runTest')
    helper.setUp()
    endpoints = []
    transport_directory = None
    try:
        handle = helper.create(optimizer='pi', profile_strategy='combined')
        root, workspace = handle.session_root, handle.initial_workspace
        before_candidate = hashes(workspace / 'candidate')
        before_context = hashes(workspace / '.benchmark_context')
        before_manifest = (root / 'protection/manifest.json').read_bytes()
        root_mode = workspace.stat().st_mode
        wiki = runner.resources.freeze_wiki(ROOT / 'pto_wiki/pypto_wiki', root, workspace)
        assert wiki['snapshot_path'] == str(root / 'method/resources/pypto_wiki')
        assert wiki['visible_path'] == str(workspace / '.mode_resources/pypto_wiki')
        assert all(entry['source_path'].startswith('pto_wiki/pypto_wiki/')
                   for entry in wiki['files'])
        assert workspace.stat().st_mode == root_mode
        assert (root / 'protection/manifest.json').read_bytes() == before_manifest
        assert integrity.check_only(root / 'protection').passed
        fork_id = session.fork_workspace(root, workspace)
        fork = root / 'workspaces' / fork_id
        with patch.object(common, 'BENCHMARK_ROOT', ROOT):
            transport_directory, transport = sandbox_supervisor._copy_cli_transport(root)
        for name in ('rpc', 'api'):
            endpoint = socket.socket(socket.AF_UNIX)
            endpoint.bind(str(root / (name + '.sock')))
            endpoints.append(endpoint)
        environment = {'PATH': os.environ['PATH'], 'PTOBENCH_RPC_SOCKET': str(root / 'rpc.sock'),
                       'PTOBENCH_API_PROXY_SOCKET': str(root / 'api.sock'), 'PTOBENCH_CLIENT_PATH': str(transport)}
        visible = Path(wiki['visible_path'])
        code = f'''
import os,pathlib,pwd
assert os.getuid() == {pwd.getpwnam('ptobench-agent').pw_uid}
p = pathlib.Path({str(visible / 'INDEX.md')!r})
assert p.read_text()
for operation in (lambda: p.write_text('bad'), lambda: p.unlink(), lambda: p.rename(p.with_name('replaced')), lambda: p.parent.rename(p.parent.with_name('replaced'))):
 try: operation()
 except PermissionError: pass
 else: raise AssertionError('Wiki was mutable')
print('WIKI_READONLY_FROM_FORK_OK')
'''
        # Use the real existing Agent view; restore production roots only for the launch.
        with patch.object(common, 'REPO_ROOT', ROOT.parents[1]), patch.object(common, 'BENCHMARK_ROOT', ROOT):
            argv, env = isolation.agent_command([sys.executable, '-c', code], cwd=fork, environment=environment,
                                                workspace=fork, session_root=root)
            result = subprocess.run(argv, cwd=fork, env=env, capture_output=True, text=True, timeout=30)
            agent_launch.cleanup_request(env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'WIKI_READONLY_FROM_FORK_OK' in result.stdout
        assert hashes(workspace / 'candidate') == before_candidate
        assert hashes(workspace / '.benchmark_context') == before_context
        assert integrity.check_only(root / 'protection').passed
        assert runner.resources.verify_wiki(wiki)['unchanged']
    finally:
        for endpoint in endpoints:
            endpoint.close()
        if transport_directory:
            import shutil
            shutil.rmtree(transport_directory, ignore_errors=True)
        helper.doCleanups()


def test_wiki_rejects_links_and_special_nodes(tmp_path):
    runner = runner_module()
    for kind in ('symlink', 'fifo'):
        source = tmp_path / kind
        source.mkdir()
        (source / 'INDEX.md').write_text('index')
        if kind == 'symlink':
            (source / 'bad').symlink_to(source / 'INDEX.md')
        else:
            os.mkfifo(source / 'bad')
        with pytest.raises(ValueError, match='link|special'):
            runner.resources.freeze_wiki(source, tmp_path / ('run_' + kind), tmp_path / ('workspace_' + kind))
