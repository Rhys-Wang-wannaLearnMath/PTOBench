import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

import pytest
from conftest import MODE, hashes


def server_type():
    assert (MODE / 'tools.py').is_file(), 'Task 2 broker not implemented'
    import importlib
    return importlib.import_module('modes.basic_wiki-tool.tools').ToolServer


def extension(tmp_path):
    path = tmp_path / 'bridge'
    path.mkdir()
    shutil.copyfile(MODE / 'bridge/pi_analyzer.ts', path / 'pi_analyzer.ts')
    specs = json.loads((MODE / 'analyzer_manifest.json').read_text())['specifications']
    (path / 'pto-analyzer-specs.json').write_text(json.dumps(specs))
    return path / 'pi_analyzer.ts'


def node_script(tmp_path, bridge, socket_path, body):
    script = tmp_path / 'driver.mjs'
    script.write_text('''
import extension from %s;
const tools = [], handlers = {};
const pi = { registerTool: t => tools.push(t), on: (n, cb) => handlers[n] = cb,
  getAllTools: () => tools, getActiveTools: () => tools.map(t => t.name) };
process.env.PTOBENCH_PI_TOOL_SOCKET = %s;
extension(pi);
''' % (json.dumps(bridge.as_uri()), json.dumps(str(socket_path))) + body)
    return subprocess.Popen(['node', '--experimental-strip-types', str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def exchange(path, request):
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(path))
        client.sendall(json.dumps(request).encode() + b'\n')
        return json.loads(client.makefile('rb').readline())


def records(log):
    return [json.loads(p.read_text()) for p in sorted(log.glob('*/result.json'))]


def test_registered_tools_roundtrip_and_audit(profile_root, tmp_path):
    Server = server_type()
    from analyzer_adapter import AnalyzerAdapter
    initial = hashes(profile_root)
    log = profile_root / 'method/analyzer_calls'
    path = tmp_path / 'a.sock'
    server = Server(AnalyzerAdapter(profile_root), path, log)
    bridge = extension(tmp_path)
    driver = node_script(tmp_path, bridge, path, '''
await handlers.session_start();
const output = [];
for (const [i, tool] of tools.entries()) {
  const args = tool.parameters.properties.limit ? { limit: 1 } : {};
  output.push(await tool.execute(`pi-${i}`, args, new AbortController().signal));
}
const tile = tools.find(t => t.name === 'tile_access_check');
const first = JSON.parse(output[3].content[0].text);
output.push(await tile.execute('page', { profile: first.profile.event_id, cursor: first.details.next_cursor, limit: 1 }));
output.push(await tile.execute('bad', { limit: 0 }));
console.log(JSON.stringify({names: tools.map(t => t.name), output}));
''')
    try:
        stdout, stderr = driver.communicate(timeout=20)
        assert driver.returncode == 0, stderr
        result = json.loads(stdout)
        specs = json.loads((MODE / 'analyzer_manifest.json').read_text())['specifications']
        assert result['names'] == [s['name'] for s in specs]
        registration = json.loads((log.parent / 'analyzer_registration.json').read_text())
        assert registration['names'] == registration['active_names'] == result['names']
        assert registration['complete'] is True
        rows = records(log)
        assert len(rows) == 11
        for i, reply in enumerate(result['output']):
            call_id = f'pi-{i}' if i < 9 else ('page' if i == 9 else 'bad')
            row = next(r for r in rows if r['tool_call_id'] == call_id)
            directory = next(p.parent for p in log.glob('*/result.json') if json.loads(p.read_text())['tool_call_id'] == call_id)
            request = json.loads((directory / 'request.json').read_text())
            assert directory.name != call_id
            assert request['tool_call_id'] == call_id
            assert row['name'] == request['name']
            assert row['arguments'] == request['arguments']
            assert row['started_at'] and row['ended_at']
            if i < 10:
                assert not reply['isError']
                assert row['status'] == 'success'
                assert json.loads(reply['content'][0]['text']) == row['output'] == reply['details']
                assert row['profile'] == row['output']['profile']
                assert row['profile']['event_id'] == 'fixture-session:1'
            else:
                assert reply['isError'] and row['status'] == 'error'
                assert row['error']['type'] == 'ValueError'
                assert row['profile']['event_id'] == 'fixture-session:1'
        assert {p: hashlib.sha256((profile_root / p).read_bytes()).hexdigest() for p in initial} == initial
    finally:
        server.close()
    assert not path.exists()


def test_cancel_disconnect_and_close_are_recorded(tmp_path):
    Server = server_type()
    bridge = extension(tmp_path)
    started, release = threading.Event(), threading.Event()
    class Slow:
        def __init__(self):
            self.called = []
        def specifications(self):
            return json.loads((MODE / 'analyzer_manifest.json').read_text())['specifications']
        def call(self, name, args):
            self.called.append(args)
            started.set()
            assert release.wait(10)
            return {'profile': {'event_id': 'slow:1'}, 'payload': args}
    adapter = Slow()
    path, log = tmp_path / 'a.sock', tmp_path / 'method/analyzer_calls'
    server = Server(adapter, path, log)
    driver = node_script(tmp_path, bridge, path, '''
const early = new AbortController(); early.abort();
const before = await tools[0].execute('early', {}, early.signal);
const controller = new AbortController();
const promise = tools[0].execute('running', { marker: 'running' }, controller.signal);
setTimeout(() => controller.abort(), 100);
console.log(JSON.stringify([before, await promise]));
''')
    try:
        assert started.wait(5)
        # Observe the actual cancellation before letting synchronous analysis return.
        deadline = time.monotonic() + 5
        while not any(json.loads(p.read_text()).get('tool_call_id') == 'running' for p in log.glob('*/cancel.json')) and time.monotonic() < deadline:
            time.sleep(.01)
        assert any(json.loads(p.read_text()).get('tool_call_id') == 'running' for p in log.glob('*/cancel.json'))
        release.set()
        stdout, stderr = driver.communicate(timeout=10)
        assert driver.returncode == 0, stderr
        assert all(r['isError'] for r in json.loads(stdout))
        assert adapter.called == [{'marker': 'running'}]
        assert {r['status'] for r in records(log)} == {'cancelled'}
        duplicate = exchange(path, dict(action='call', tool_call_id='running', name='compute_mapping_check', arguments={}))
        assert duplicate['isError'] and 'DuplicateCallError' in duplicate['content'][0]['text']
        assert len(adapter.called) == 1
        assert len(records(log)) == 3
        # Disconnect after execution starts: result must survive the lost client.
        started.clear(); release.clear()
        client = socket.socket(socket.AF_UNIX)
        client.connect(str(path))
        client.sendall(json.dumps(dict(action='call', tool_call_id='disconnected', name='compute_mapping_check', arguments={})).encode() + b'\n')
        assert started.wait(5)
        client.close()
        release.set()
        deadline = time.monotonic() + 5
        while len(records(log)) < 4 and time.monotonic() < deadline:
            time.sleep(.01)
        assert next(r for r in records(log) if r['tool_call_id'] == 'disconnected')['status'] == 'interrupted'
        # Close while a request is in flight; no socket may remain accepting.
        started.clear(); release.clear()
        client = socket.socket(socket.AF_UNIX); client.connect(str(path))
        client.sendall(json.dumps(dict(action='call', tool_call_id='closing', name='compute_mapping_check', arguments={})).encode() + b'\n')
        assert started.wait(5)
        closing = threading.Thread(target=server.close); closing.start()
        release.set(); closing.join(5); client.close()
        assert not closing.is_alive()
        assert not path.exists()
        assert next(r for r in records(log) if r['tool_call_id'] == 'closing')['status'] == 'interrupted'
        with pytest.raises(OSError):
            exchange(path, dict(action='cancel', tool_call_id='closing'))
    finally:
        release.set()
        server.close()
        if driver.poll() is None:
            driver.kill(); driver.communicate()


def test_nonfinite_output_has_terminal_typed_error_and_profile(tmp_path):
    Server = server_type()
    class Nonfinite:
        def call(self, name, args):
            return {'profile': {'event_id': 'nonfinite:1'}, 'value': float('inf')}
    log, path = tmp_path / 'method/analyzer_calls', tmp_path / 'a.sock'
    server = Server(Nonfinite(), path, log)
    try:
        response = exchange(path, dict(action='call', tool_call_id='nonfinite', name='buffer_pressure_check', arguments={}))
        assert response['isError'] and 'ValueError' in response['content'][0]['text']
        assert len(records(log)) == 1, 'serialization failure must immediately write a terminal record'
        terminal = records(log)[0]
        assert terminal['status'] == 'error'
        assert terminal['error']['type'] == 'ValueError'
        assert terminal['profile'] == {'event_id': 'nonfinite:1'}
        import importlib
        importlib.import_module('modes.basic_wiki-tool.tools').interrupt_unfinished(log)
        assert records(log) == [terminal]
    finally:
        server.close()
    assert not path.exists()


def test_long_socket_path_binds_and_cleans_up_through_short_alias(tmp_path):
    Server = server_type()
    session = tmp_path / ('session-component-' * 3)
    path = session / 'agent/analyzer.sock'
    log = session / 'method/analyzer_calls'
    path.parent.mkdir(parents=True)
    assert len(str(path).encode()) > 107

    class EmptyAdapter:
        def specifications(self):
            return []

    server = Server(EmptyAdapter(), path, log)
    alias_root = server.endpoint.parent.parent
    try:
        assert len(str(server.endpoint).encode()) <= 107
        assert server.endpoint.resolve(strict=True) == path.resolve(strict=True)
        assert path.is_socket()
        assert path.resolve(strict=True).is_relative_to(session.resolve(strict=True))
        assert path.lstat().st_uid == os.getuid()
        result = exchange(server.endpoint, dict(action='registered', names=[], active_names=[]))
        assert not result['isError']
        assert json.loads((session / 'method/analyzer_registration.json').read_text())['complete']
    finally:
        server.close()
    assert not path.exists()
    assert not alias_root.exists()
