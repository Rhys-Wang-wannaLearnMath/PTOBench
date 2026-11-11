"""Actual kernel boundary tests; no mocked path checks or model calls."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import signal
import time
import unittest
from unittest.mock import patch

import isolation
import sandbox_supervisor

TEST_PYTHON = str(Path(sys.prefix) / 'bin/python')


class FilesystemBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'PTOBENCH_SANDBOX': '1'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        # Missing isolation must fail loudly rather than turning tests into a
        # false PASS; this suite is intended for namespace-capable Linux hosts.
        isolation.check_available()

    def test_read_write_symlink_proc_and_capability_boundary(self):
        with tempfile.TemporaryDirectory(prefix='isolation-test-') as temporary:
            root = Path(temporary)
            secret = root / 'host-wiki.txt'
            secret.write_text('UNAUTHORIZED_CANARY')
            public = root / 'public'
            public.mkdir()
            (public / 'contract').write_text('contract')
            work = root / 'work'
            work.mkdir()
            (work / 'escape').symlink_to(secret)
            view = isolation.View().system().add(public, '/public').add(work, '/work', write=True)
            program = r'''
import os,pathlib,subprocess
assert pathlib.Path('/public/contract').read_text()=='contract'
pathlib.Path('/work/candidate.py').write_text('ok')
pathlib.Path('/tmp/scratch').write_text('ok')
for p in [SECRET,'/work/escape','/proc/'+HOSTPID+'/root'+SECRET]:
 try: pathlib.Path(p).read_bytes()
 except OSError: pass
 else: raise AssertionError('host read succeeded: '+p)
for p in ['/public/contract',SECRET]:
 try: pathlib.Path(p).write_text('bad')
 except OSError: pass
 else: raise AssertionError('unauthorized write: '+p)
try: pathlib.Path('/public/contract').chmod(0o600)
except OSError: pass
else: raise AssertionError('readonly contract chmod succeeded')
assert subprocess.run(['/usr/bin/mount','--bind','/tmp','/public'],capture_output=True).returncode != 0
assert '0000000000000000' in pathlib.Path('/proc/self/status').read_text().split('CapEff:')[1].splitlines()[0]
'''
            program = 'SECRET='+repr(str(secret))+'\nHOSTPID='+repr(str(os.getpid()))+'\n'+program
            result = subprocess.run(view.argv([TEST_PYTHON, '-c', program], cwd='/work',
                environment=isolation.clean_environment()), capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(secret.read_text(), 'UNAUTHORIZED_CANARY')
            self.assertEqual((public / 'contract').read_text(), 'contract')

    def test_installed_upstream_cache_exposes_only_headers(self):
        for checkout in Path(sys.prefix).glob('lib/python*/site-packages/simpler_setup/_assets/build/pto-isa'):
            program = ('from pathlib import Path; p=Path('+repr(str(checkout))+'); '
                       'assert (p/"include/pto/pto-inst.hpp").is_file(); '
                       'assert not (p/"docs").exists(); assert not (p/".git").exists(); '
                       'assert not (p/"tests").exists()')
            view = isolation.View().system()
            result = subprocess.run(view.argv([TEST_PYTHON, '-c', program], cwd='/tmp',
                environment=isolation.clean_environment()), capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_copy_ignores_unselected_mode_directories(self):
        import common
        with tempfile.TemporaryDirectory(prefix='runtime-copy-') as temporary:
            benchmark = Path(temporary) / 'PTOBench'
            for name in ('evaluator', 'launcher', 'modes'):
                directory = benchmark / name
                directory.mkdir(parents=True)
                (directory / '__init__.py').write_text('')
            (benchmark / 'evaluator/pi_file_guard.mjs').write_text('export default () => {}\n')
            selected = {'basic', '_shared', 'cannbot'}
            ignored = {'ours', 'profile_tool', 'ptomni', 'PTOmni', 'future_mode'}
            for name in selected | ignored:
                directory = benchmark / 'modes' / name
                directory.mkdir()
                (directory / 'runner.py').write_text(f'MODE = {name!r}\n')

            with patch.object(common, 'BENCHMARK_ROOT', benchmark):
                runtime = isolation.copy_runtime(Path(temporary) / 'runtime')

            copied = {path.name for path in (runtime / 'modes').iterdir() if path.is_dir()}
            self.assertEqual(copied, selected)
            self.assertTrue((runtime / 'evaluator/pi_file_guard.mjs').is_file())
            for name in ignored:
                self.assertFalse((runtime / 'modes' / name).exists())

    def test_pi_file_tools_cannot_access_private_config_or_sessions(self):
        with tempfile.TemporaryDirectory(prefix='pi-file-guard-') as temporary:
            root = Path(temporary)
            private = root / 'private'
            sessions = root / 'sessions'
            workspace = root / 'workspace'
            (private / 'extensions').mkdir(parents=True)
            sessions.mkdir()
            (workspace / 'candidate').mkdir(parents=True)
            (private / 'settings.json').write_text('private config')
            (private / 'extensions/security.mjs').write_text('private extension')
            (sessions / 'session.jsonl').write_text('private session')
            (workspace / 'candidate/candidate.py').write_text('candidate')
            (workspace / 'private-link').symlink_to(private, target_is_directory=True)
            os.link(private / 'settings.json', workspace / 'candidate/private-hardlink.json')

            script = r'''import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import pathModule from 'node:path';
const extension = await import(pathToFileURL(process.env.PI_FILE_GUARD));
let guard;
extension.default({ on: (name, handler) => {
    assert.equal(name, 'tool_call');
    guard = handler;
} });
const context = { cwd: process.env.PI_GUARD_CWD };
const privateFile = pathModule.join(process.env.PI_CODING_AGENT_DIR, 'settings.json');
const sessionFile = pathModule.join(process.env.PTOBENCH_AGENT_SESSION_DIR, 'session.jsonl');
for (const toolName of ['read', 'write', 'edit', 'grep', 'find', 'ls']) {
    const blocked = await guard({ toolName, input: { path: privateFile } }, context);
    assert.equal(blocked?.block, true, toolName);
    assert.match(blocked.reason, /private Pi runtime/);
}
for (const path of [
    process.env.PI_CODING_AGENT_DIR,
    sessionFile,
    pathModule.join(context.cwd, 'private-link/extensions/security.mjs'),
    pathModule.relative(context.cwd, privateFile),
    'candidate/private-hardlink.json',
]) {
    const blocked = await guard({ toolName: 'read', input: { path } }, context);
    assert.equal(blocked?.block, true, path);
}
assert.equal(await guard({ toolName: 'read', input: { path: 'candidate/candidate.py' } }, context), undefined);
assert.equal((await guard({ toolName: 'grep', input: { path: '.' } }, context))?.block, true);
'''
            result = subprocess.run(
                [shutil.which('node'), '--input-type=module', '-e', script],
                cwd=workspace,
                env={**os.environ, 'PI_FILE_GUARD': str(Path(isolation.__file__).with_name('pi_file_guard.mjs')),
                     'PI_CODING_AGENT_DIR': str(private), 'PTOBENCH_AGENT_SESSION_DIR': str(sessions),
                     'PI_GUARD_CWD': str(workspace)},
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_pi_shell_wrapper_hides_private_files_but_keeps_candidate_and_cli(self):
        import common
        with tempfile.TemporaryDirectory(prefix='pi-shell-boundary-') as temporary:
            root = Path(temporary)
            workspace = root / 'workspaces/worker_000'
            candidates = root / 'candidates/worker_000'
            public = root / 'public/worker_000'
            private = root / 'agent/pi'
            sessions = root / 'agent/native'
            for directory in (workspace, candidates, public, private, sessions):
                directory.mkdir(parents=True, exist_ok=True)
            (workspace / 'candidate').symlink_to('/ptobench/candidates/worker_000')
            (workspace / '.benchmark_context').symlink_to('/ptobench/public/worker_000')
            (workspace / '.benchmark_outputs').symlink_to('/ptobench/public/worker_000/.benchmark_outputs')
            (candidates / 'candidate.py').write_text('VALUE = 1\n')
            (candidates / 'tuning.json').write_text('{}\n')
            (public / '.benchmark_outputs').mkdir()
            (public / 'contract.txt').write_text('public contract')
            (private / 'settings.json').write_text('{"theme": "dark"}\n')
            (sessions / 'session.jsonl').write_text('private session')
            isolation.install_pi_private_tools(private)

            client = root / 'ptobench-client'
            shutil.copyfile(Path(isolation.__file__).with_name('sandbox_client.py'), client)
            client.chmod(0o755)
            rpc = root / 'broker.sock'
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(rpc))
            server.listen(1)
            self.addCleanup(server.close)
            request = {}

            def respond():
                connection, _ = server.accept()
                with connection:
                    raw = connection.makefile('rb').readline()
                    request.update(json.loads(raw))
                    connection.sendall(b'{"output":"broker ok\\n"}\n{"exit_code":0}\n')

            response_thread = threading.Thread(target=respond, daemon=True)
            response_thread.start()
            view = isolation.View().system().native_agent()
            view.add(workspace, '/workspace')
            view.add(candidates.parent, '/ptobench/candidates', write=True)
            view.add(public.parent, '/ptobench/public')
            view.add(private, '/ptobench/pi-private')
            view.add(sessions, '/ptobench/agent-sessions', write=True)
            view.add(rpc, '/ptobench/cli.sock')
            view.add(client, common.BENCHMARK_ROOT / 'ptobench')
            environment = isolation.clean_environment()
            environment.update(PATH=str(common.BENCHMARK_ROOT) + os.pathsep + environment['PATH'],
                PI_CODING_AGENT_DIR='/ptobench/pi-private',
                PTOBENCH_AGENT_SESSION_DIR='/ptobench/agent-sessions',
                PTOBENCH_RPC_SOCKET='/ptobench/cli.sock')
            script = r'''python3 -c 'import json,os,pathlib,subprocess
assert not pathlib.Path("/ptobench/pi-private/settings.json").exists()
assert not pathlib.Path("/ptobench/agent-sessions/session.jsonl").exists()
assert pathlib.Path(".benchmark_context/contract.txt").read_text() == "public contract"
pathlib.Path("candidate/new_helper.py").write_text("ANSWER = 42" + chr(10))
pathlib.Path("candidate/tuning.json").write_text(json.dumps({"ok": True}) + chr(10))
try: pathlib.Path(".benchmark_context/changed").write_text("bad")
except OSError: pass
else: raise AssertionError("context was writable")
result=subprocess.run(["ptobench","check"],capture_output=True,text=True)
assert result.returncode == 0 and result.stdout == "broker ok" + chr(10), (result.stdout,result.stderr)
assert "PI_CODING_AGENT_DIR" not in os.environ
assert "PTOBENCH_AGENT_SESSION_DIR" not in os.environ'
'''
            result = subprocess.run(
                view.argv(['/ptobench/pi-private/ptobench-shell', '-c', script],
                          cwd='/workspace', environment=environment),
                cwd='/', capture_output=True, text=True, timeout=30)
            response_thread.join(5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(request['argv'], ['check'])
            self.assertEqual((candidates / 'new_helper.py').read_text(), 'ANSWER = 42\n')
            self.assertEqual((candidates / 'tuning.json').read_text(), '{"ok": true}\n')

    def test_network_remains_available_and_host_environment_is_not_inherited(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        self.addCleanup(listener.close)
        def respond():
            with listener.accept()[0] as connection:
                connection.sendall(b'network allowed')
        thread = threading.Thread(target=respond, daemon=True); thread.start()
        program = ('import socket,os; assert "UNAUTHORIZED_ENV" not in os.environ; '
                   f's=socket.create_connection(("127.0.0.1",{listener.getsockname()[1]}),3); '
                   'assert s.recv(100)==b"network allowed"')
        with patch.dict(os.environ, {'UNAUTHORIZED_ENV': 'private'}):
            result = subprocess.run(isolation.View().system().argv([TEST_PYTHON, '-c', program],
                cwd='/tmp', environment=isolation.clean_environment()), capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        thread.join(5)

    def test_agent_network_is_limited_to_the_model_api_proxy(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        port = listener.getsockname()[1]
        self.addCleanup(listener.close)

        def respond():
            connection, _ = listener.accept()
            with connection:
                self.assertIn(b'GET / HTTP/1.1', connection.recv(4096))
                connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nmodel reply')

        thread = threading.Thread(target=respond, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory(prefix='agent-network-') as temporary:
            root = Path(temporary)
            network = root / 'network'
            network.mkdir()
            proxy_socket = network / 'proxy.sock'
            proxy = subprocess.Popen([sys.executable, str(Path(isolation.__file__).parent / 'network_proxy.py'),
                'serve', '--socket', str(proxy_socket), '--host', '127.0.0.1', '--port', str(port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.addCleanup(lambda: proxy.terminate() if proxy.poll() is None else None)
            deadline = time.monotonic() + 5
            while not proxy_socket.exists() and time.monotonic() < deadline:
                if proxy.poll() is not None:
                    self.fail(proxy.stderr.read().decode())
                time.sleep(.02)
            self.assertTrue(proxy_socket.exists())
            program = f'''import socket
try:
    socket.create_connection(("127.0.0.1", {port}), .5)
except OSError:
    pass
else:
    raise AssertionError("Agent reached host network without proxy")
s=socket.create_connection(("127.0.0.1", 18080), 3)
s.sendall(b"CONNECT 127.0.0.1:{port} HTTP/1.1\\r\\nHost: 127.0.0.1:{port}\\r\\n\\r\\n")
assert b"200 Connection Established" in s.recv(4096)
s.sendall(b"GET / HTTP/1.1\\r\\nHost: 127.0.0.1:{port}\\r\\n\\r\\n")
assert b"model reply" in s.recv(4096)
s.close()
denied=socket.create_connection(("127.0.0.1", 18080), 3)
denied.sendall(b"CONNECT 127.0.0.1:{port+1} HTTP/1.1\\r\\nHost: 127.0.0.1:{port+1}\\r\\n\\r\\n")
assert b"403 Forbidden" in denied.recv(4096)
denied.close()
'''
            environment = isolation.clean_environment(agent=True)
            environment.update(HTTP_PROXY='http://127.0.0.1:18080',
                HTTPS_PROXY='http://127.0.0.1:18080', NO_PROXY='', no_proxy='')
            view = isolation.View(isolate_network=True).system().add(network, '/ptobench/network')
            self.assertEqual(view.manifest()['network'], 'model_api_only')
            proxy_program = Path(isolation.__file__).parent / 'network_proxy.py'
            view.add(proxy_program, '/ptobench/network_proxy.py')
            agent = [TEST_PYTHON, '/ptobench/network_proxy.py',
                'agent', '--socket', '/ptobench/network/proxy.sock', '--', TEST_PYTHON, '-c', program]
            result = subprocess.run(view.argv(agent, cwd='/tmp', environment=environment),
                cwd='/tmp', capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            proxy.terminate()
            proxy.wait(timeout=5)
            thread.join(5)

    def test_pi_agent_can_write_private_config_while_feedback_stays_readonly(self):
        import common

        upstream_headers = []
        response_body = b'data: {"type":"response.completed"}\n\n'

        class Provider(BaseHTTPRequestHandler):
            def do_POST(self):
                upstream_headers.append(self.headers.get('Authorization'))
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, *_args):
                pass

        provider = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
        provider_thread.start()
        self.addCleanup(provider.shutdown)
        self.addCleanup(provider.server_close)
        self.addCleanup(provider_thread.join, 5)

        with tempfile.TemporaryDirectory(prefix='agent-gateway-') as temporary:
            root = Path(temporary) / 'run'
            workspace = root / 'workspaces/initial'
            public = Path(temporary) / 'public'
            candidates = root / 'candidates'
            untrusted_candidates = Path(temporary) / 'untrusted-candidates'
            initial_candidate = candidates / 'initial'
            workspace.mkdir(parents=True)
            initial_candidate.mkdir(parents=True)
            (untrusted_candidates / 'initial').mkdir(parents=True)
            (untrusted_candidates / 'initial/candidate.py').write_text('candidate = False\n')
            (public / 'initial/.benchmark_outputs/combined').mkdir(parents=True)
            (public / 'initial/README.md').write_text('public contract')
            (public / 'initial/.benchmark_outputs/combined/latest_result.json').write_text('{"run": 1}\n')
            (workspace / 'candidate').symlink_to('/ptobench/candidates/initial')
            (workspace / '.benchmark_context').symlink_to('/ptobench/public/initial')
            (workspace / '.benchmark_outputs').symlink_to('/ptobench/public/initial/.benchmark_outputs')
            (initial_candidate / 'candidate.py').write_text('candidate = True\n')
            (initial_candidate / 'tuning.json').write_text('{"memory_planner":"PYPTO"}\n')
            initial_candidate.chmod(0o2770)
            (initial_candidate / 'candidate.py').chmod(0o660)
            (initial_candidate / 'tuning.json').chmod(0o660)
            (root / 'agent/pi').mkdir(parents=True)
            (root / 'agent/native').mkdir(parents=True)
            client = Path(temporary) / 'sandbox-client'
            client.write_text('untrusted-side CLI transport\n')
            client.chmod(0o755)
            rpc = Path(temporary) / 'rpc.sock'
            rpc_server = socket.socket(socket.AF_UNIX)
            rpc_server.bind(str(rpc))
            rpc_server.listen(1)
            self.addCleanup(rpc_server.close)
            gateway_dir = Path(temporary) / 'gateway'
            gateway_dir.mkdir()
            gateway_socket = gateway_dir / 'api.sock'
            proxy_script = Path(isolation.__file__).with_name('network_proxy.py')
            gateway = subprocess.Popen([
                sys.executable, str(proxy_script), 'serve-api', '--socket', str(gateway_socket),
                '--base-url', f'http://127.0.0.1:{provider.server_port}/v1',
                '--api', 'openai-responses',
            ], env={**os.environ, 'PTOBENCH_UPSTREAM_API_KEY': 'TEST_UPSTREAM_SECRET'},
               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.addCleanup(lambda: gateway.terminate() if gateway.poll() is None else None)
            deadline = time.monotonic() + 5
            while not gateway_socket.exists() and time.monotonic() < deadline:
                if gateway.poll() is not None:
                    self.fail(gateway.stderr.read().decode())
                time.sleep(.02)
            self.assertTrue(gateway_socket.exists())

            code = f'''import http.client, os, pathlib, socket, subprocess
assert pathlib.Path("candidate/candidate.py").read_text() == "candidate = True\\n"
assert pathlib.Path(".benchmark_context/README.md").read_text() == "public contract"
private_lock = pathlib.Path("/ptobench/pi-private/settings.json.lock")
private_lock.write_text("Pi config lock")
private_lock.unlink()
pathlib.Path("candidate/candidate.py").write_text("candidate = False\\n")
pathlib.Path("candidate/tuning.json").write_text("{{}}\\n")
for protected in ("candidate", ".benchmark_context", ".benchmark_outputs"):
    try: pathlib.Path(protected).rename(protected + "_replaced")
    except OSError: pass
    else: raise AssertionError("Agent replaced protected workspace entry: " + protected)
result = pathlib.Path(".benchmark_outputs/combined/latest_result.json")
assert result.read_text() == "{{\\\"run\\\": 1}}\\n"
try:
    pathlib.Path(".benchmark_outputs").rename(".benchmark_outputs_replaced")
except OSError:
    pass
else:
    raise AssertionError("Agent replaced the read-only feedback root")
for action in (
    lambda: result.write_text("changed"),
    lambda: result.unlink(),
    lambda: result.rename(result.with_name("renamed.json")),
    lambda: result.chmod(0o600),
    lambda: (result.parent / "new.json").write_text("new"),
    lambda: pathlib.Path(".benchmark_outputs/combined").rmdir(),
):
    try: action()
    except OSError: pass
    else: raise AssertionError("Agent modified read-only result feedback")
context = pathlib.Path(".benchmark_context/README.md")
try: context.write_text("changed")
except OSError: pass
else: raise AssertionError("Agent modified read-only task context")
assert "untrusted-side CLI transport" in pathlib.Path({str(common.BENCHMARK_ROOT / "ptobench")!r}).read_text()
for hidden in ({str(common.BENCHMARK_ROOT / "evaluator/agent_cli.py")!r},
               {str(common.BENCHMARK_ROOT / "launcher/methods.py")!r},
               {str(common.BENCHMARK_ROOT / "modes/cannbot/tools.py")!r},
               {str(common.BENCHMARK_ROOT / "modes/basic/.env")!r},
               {str(common.BENCHMARK_ROOT / "modes/cannbot/.env")!r},
               {str(common.BENCHMARK_ROOT / "modes/ptomni/.env")!r},
               {str(common.BENCHMARK_ROOT / "modes/basic/knowledge/raw_wiki/README.md")!r},
               {str(common.BENCHMARK_ROOT / "modes/cannbot/knowledge/raw_wiki/README.md")!r}):
    assert not pathlib.Path(hidden).exists(), hidden
assert os.environ["PYPTO_PROVIDER_API_KEY"] == "ptobench-local-model-gateway"
assert os.environ["PTOBENCH_RPC_SOCKET"] == "/ptobench/cli.sock"
assert "PTOBENCH_API_PROXY_SOCKET" not in os.environ
for process_environment in pathlib.Path("/proc").glob("[0-9]*/environ"):
    try:
        environment = process_environment.read_bytes()
        assert b"TEST_UPSTREAM_SECRET" not in environment
        assert b"must-be-replaced" not in environment
    except OSError:
        pass
connection = http.client.HTTPConnection("127.0.0.1", 18081, timeout=3)
connection.request("POST", "/v1/responses", "{{}}", {{"Authorization":"Bearer fake"}})
reply = connection.getresponse()
assert reply.status == 200 and b"response.completed" in reply.read()
connection.request("POST", "/v1/files", "{{}}")
assert connection.getresponse().status == 403
try:
    socket.create_connection(("127.0.0.1", {provider.server_port}), .5)
except OSError:
    pass
else:
    raise AssertionError("Agent reached the provider without the gateway")
try:
    socket.create_connection(("1.1.1.1", 443), .5)
except OSError:
    pass
else:
    raise AssertionError("Agent reached the public network")
assert not pathlib.Path("/ptobench/runtime/network_proxy.py").exists()
'''
            environment = {**os.environ,
                'PTOBENCH_RPC_SOCKET': str(rpc),
                'PTOBENCH_PUBLIC_DIR': str(public),
                'PTOBENCH_CLIENT_PATH': str(client),
                'PTOBENCH_API_PROXY_SOCKET': str(gateway_socket),
                'PTOBENCH_AGENT_CANDIDATES_ROOT': str(untrusted_candidates),
                'PYPTO_PROVIDER_API_KEY': 'must-be-replaced',
            }
            argv, launch_environment = isolation.agent_command(
                [TEST_PYTHON, '-c', code], cwd=workspace, environment=environment,
                workspace=workspace, session_root=root)
            result = subprocess.run(argv, cwd='/', env=launch_environment, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((initial_candidate / 'candidate.py').read_text(), 'candidate = False\n')
            self.assertEqual((initial_candidate / 'tuning.json').read_text(), '{}\n')
            self.assertEqual((public / 'initial/.benchmark_outputs/combined/latest_result.json').read_text(), '{"run": 1}\n')
            self.assertEqual((public / 'initial/README.md').read_text(), 'public contract')
            self.assertEqual(upstream_headers, ['Bearer TEST_UPSTREAM_SECRET'])
            for path in root.rglob('*'):
                if path.is_file():
                    self.assertNotIn(b'TEST_UPSTREAM_SECRET', path.read_bytes(), str(path))
            manifest = json.loads((root / 'filesystem_isolation.json').read_text())
            self.assertEqual(manifest['network'], 'model_api_only')
            self.assertFalse(any('/evaluator' in item['destination'] or '/launcher' in item['destination']
                                 or '/modes' in item['destination'] for item in manifest['mounts']))
            self.assertFalse(list((root / 'agent/.runtime').glob('*/network_proxy.py')))

    def test_agent_launcher_fails_closed_when_unavailable(self):
        import agent_launch

        with tempfile.TemporaryDirectory(prefix='agent-launch-fail-closed-') as temporary:
            root = Path(temporary) / 'run'
            workspace = root / 'workspaces/initial'
            candidates = root / 'candidates/initial'
            public = Path(temporary) / 'public/initial'
            client = Path(temporary) / 'sandbox-client'
            rpc = Path(temporary) / 'rpc.sock'
            gateway = Path(temporary) / 'gateway/api.sock'
            workspace.mkdir(parents=True)
            candidates.mkdir(parents=True)
            public.mkdir(parents=True)
            client.write_text('test client')
            client.chmod(0o755)
            rpc_server = socket.socket(socket.AF_UNIX)
            rpc_server.bind(str(rpc))
            rpc_server.listen(1)
            self.addCleanup(rpc_server.close)
            gateway.parent.mkdir()
            gateway_server = socket.socket(socket.AF_UNIX)
            gateway_server.bind(str(gateway))
            gateway_server.listen(1)
            self.addCleanup(gateway_server.close)
            environment = {
                'PATH': os.defpath,
                'PTOBENCH_PUBLIC_DIR': str(public.parent),
                'PTOBENCH_CLIENT_PATH': str(client),
                'PTOBENCH_API_PROXY_SOCKET': str(gateway),
                'PTOBENCH_RPC_SOCKET': str(rpc),
                'PTOBENCH_AGENT_LAUNCH_PATH': '/opt/ptobench-agent-launch',
            }
            with patch.object(agent_launch, '_host_credentials', return_value={
                    'uid': os.getuid(), 'gid': os.getgid(), 'groups': [10], 'cap_eff': 0,
                    'no_new_privs': False}), \
                 patch.object(agent_launch, '_setpriv_clears_groups', return_value=False), \
                 patch.object(agent_launch, '_trusted_launcher', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'supplementary groups'):
                    isolation.agent_command(
                        ['/bin/true'], cwd=workspace, environment=environment,
                        workspace=workspace, session_root=root)

    def test_agent_startup_refuses_candidate_without_group_write_access(self):
        with tempfile.TemporaryDirectory(prefix='agent-candidate-preflight-') as temporary:
            root = Path(temporary) / 'run'
            workspace = root / 'workspaces/initial'
            candidate = root / 'candidates/initial'
            for directory in (candidate, workspace, workspace / '.benchmark_context',
                              workspace / '.benchmark_outputs'):
                directory.mkdir(parents=True, exist_ok=True)
            (workspace / 'candidate').symlink_to('/ptobench/candidates/initial')
            candidate.chmod(0o750)
            public = Path(temporary) / 'public/initial'
            public.mkdir(parents=True)
            client = Path(temporary) / 'sandbox-client'
            client.write_text('test client')
            client.chmod(0o755)
            rpc = Path(temporary) / 'rpc.sock'
            rpc_socket = socket.socket(socket.AF_UNIX)
            rpc_socket.bind(str(rpc))
            rpc_socket.listen(1)
            self.addCleanup(rpc_socket.close)
            gateway = Path(temporary) / 'gateway/api.sock'
            gateway.parent.mkdir()
            gateway_socket = socket.socket(socket.AF_UNIX)
            gateway_socket.bind(str(gateway))
            gateway_socket.listen(1)
            self.addCleanup(gateway_socket.close)
            environment = {
                'PATH': os.defpath,
                'PTOBENCH_PUBLIC_DIR': str(public.parent),
                'PTOBENCH_CLIENT_PATH': str(client),
                'PTOBENCH_API_PROXY_SOCKET': str(gateway),
                'PTOBENCH_RPC_SOCKET': str(rpc),
            }

            command, launch_environment = isolation.agent_command(
                ['/bin/true'], cwd=workspace, environment=environment,
                workspace=workspace, session_root=root)
            result = subprocess.run(command, cwd=workspace, env=launch_environment,
                                    capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 126, result.stderr)
            self.assertIn('Candidate permission check failed', result.stderr)

    def test_broker_refreshes_feedback_in_place_under_the_readonly_public_mount(self):
        with tempfile.TemporaryDirectory(prefix='broker-feedback-') as temporary:
            root = Path(temporary) / 'run'
            original = root / 'workspaces/worker_000'
            (original / 'candidate').mkdir(parents=True)
            (original / '.benchmark_context').mkdir(parents=True)
            (original / '.benchmark_context/README.md').write_text('contract')
            (original / '.benchmark_outputs/combined').mkdir(parents=True)
            (original / '.benchmark_outputs/combined/latest_result.json').write_text(
                json.dumps({'path': str(root / 'staging/action_000001/workspace/.benchmark_outputs/combined/trace.json')})
            )
            (original / '.benchmark_outputs/combined/old.json').write_text('{}\n')
            (original / 'candidate/candidate.py').write_text('def entry(): pass\n')
            (original / 'candidate/tuning.json').write_text('{}\n')
            private = Path(temporary) / 'private'
            private.mkdir()
            broker = sandbox_supervisor.Broker.__new__(sandbox_supervisor.Broker)
            broker.root, broker.private = root, private
            broker.view, broker.public = private / 'view', private / 'public'
            broker.exports = broker.view / 'workspaces'
            broker.candidates = broker.view / 'candidates'
            broker.exports.mkdir(parents=True)
            broker.candidates.mkdir()
            broker.public.mkdir()

            exported = broker.export('worker_000')
            self.assertEqual(os.readlink(exported / '.benchmark_outputs'),
                             '/ptobench/public/worker_000/.benchmark_outputs')
            self.assertEqual(os.readlink(exported / '.benchmark_context'),
                             '/ptobench/public/worker_000')
            self.assertEqual(os.readlink(exported / 'candidate'),
                             '/ptobench/candidates/worker_000')
            broker.sync_source('worker_000')
            self.assertEqual((original / 'candidate/candidate.py').read_text(), 'def entry(): pass\n')
            self.assertEqual((original / 'candidate/tuning.json').read_text(), '{}\n')
            exported_candidate = broker.candidates / 'worker_000'
            (exported_candidate / 'candidate.py').write_text('def entry(): return 2\n')
            broker.sync_source('worker_000')
            self.assertEqual((original / 'candidate/candidate.py').read_text(), 'def entry(): return 2\n')

            unexpected_link = exported_candidate / 'escape.py'
            unexpected_link.symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                broker.sync_source('worker_000')
            unexpected_link.unlink()

            context_link = exported / '.benchmark_context'
            context_link.unlink()
            context_link.symlink_to('/ptobench/public/other-worker')
            with self.assertRaisesRegex(ValueError, 'protected workspace link changed: .benchmark_context'):
                broker.sync_source('worker_000')
            context_link.unlink()
            context_link.symlink_to('/ptobench/public/worker_000')

            outputs_link = exported / '.benchmark_outputs'
            outputs_link.unlink()
            outputs_link.symlink_to('/ptobench/public/other-worker/.benchmark_outputs')
            with self.assertRaisesRegex(ValueError, 'protected workspace link changed: .benchmark_outputs'):
                broker.sync_source('worker_000')
            outputs_link.unlink()
            outputs_link.symlink_to('/ptobench/public/worker_000/.benchmark_outputs')

            public_outputs = broker.public / 'worker_000/.benchmark_outputs'
            original_inode = public_outputs.stat().st_ino

            broker.feedback('worker_000')
            first = json.loads((public_outputs / 'combined/latest_result.json').read_text())
            self.assertIn(str(root / 'workspaces/worker_000/.benchmark_outputs/combined/trace.json'),
                          first['path'])
            self.assertEqual(public_outputs.stat().st_ino, original_inode)

            (original / '.benchmark_outputs/combined/old.json').unlink()
            (original / '.benchmark_outputs/combined/new.json').write_text('{}\n')
            broker.feedback('worker_000')
            self.assertFalse((public_outputs / 'combined/old.json').exists())
            self.assertTrue((public_outputs / 'combined/new.json').is_file())
            self.assertEqual(public_outputs.stat().st_ino, original_inode)
            self.assertEqual(os.readlink(exported / '.benchmark_outputs'),
                             '/ptobench/public/worker_000/.benchmark_outputs')

            candidate_link = exported / 'candidate'
            candidate_link.unlink()
            candidate_link.symlink_to('/ptobench/candidates/worker_999')
            with self.assertRaisesRegex(ValueError, 'protected workspace link changed: candidate'):
                broker.sync_source('worker_000')

    def test_missing_sandbox_does_not_fallback(self):
        with patch.object(isolation, 'executable', side_effect=RuntimeError('unavailable')):
            with self.assertRaises(RuntimeError):
                isolation.View().system().argv(['/usr/bin/true'], cwd='/tmp', environment={})

    def test_collection_skips_identical_readonly_files(self):
        with tempfile.TemporaryDirectory(prefix='isolation-copy-') as temporary:
            root = Path(temporary)
            source = root / 'view' / 'method/work'
            target = root / 'session' / 'method/work'
            (source / 'programs').mkdir(parents=True)
            (target / 'programs').mkdir(parents=True)
            source_file = source / 'programs/tuning.json'
            target_file = target / 'programs/tuning.json'
            source_file.write_text('{"tuning": 1}\n')
            target_file.write_text('{"tuning": 1}\n')
            target_file.chmod(0o444)
            (target / 'programs').chmod(0o555)

            with patch.object(sandbox_supervisor.shutil, 'copy2', side_effect=AssertionError('identical file copied')):
                sandbox_supervisor.copy_tree(source, target)
            self.assertEqual(target_file.read_text(), '{"tuning": 1}\n')

    def test_worker_validation_hides_golden_and_uses_host_comparator(self):
        import torch
        import candidate_execution
        import common
        with tempfile.TemporaryDirectory(prefix='isolation-wire-') as temporary:
            root = Path(temporary); fixture = root / 'fixture'; (fixture / 'out').mkdir(parents=True)
            torch.save(torch.tensor([1., 2.]), fixture / 'out/y.pt')
            records = [{'name':'y','kind':'tensor','shape':[2],'dtype':'torch.float32','resident':None}]
            validation = candidate_execution.Validation(root, {'fixture_000':(fixture,records)}, ['y'],
                {'y': lambda actual,expected,**kw: (torch.equal(actual,expected),'exact mismatch')})
            self.addCleanup(validation.close)
            runtime = isolation.copy_runtime(root / 'runtime')
            self.assertTrue((runtime / 'modes/_shared/profile_dispatch_pairing.py').is_file())
            artifact_modes = [path for path in (common.BENCHMARK_ROOT / 'modes').iterdir()
                              if (path / 'ARTIFACT.md').is_file()]
            for artifact_mode in artifact_modes:
                relative = artifact_mode.relative_to(common.BENCHMARK_ROOT)
                self.assertFalse((runtime / relative).exists())
            view = isolation.View().system().add(common.REPO_ROOT / 'golden')
            isolation.runtime_mounts(view, runtime)
            view.add(validation.socket_path, '/validation.sock').add(validation.wire, '/wire', write=True)
            env = isolation.clean_environment()
            env.update(PYTHONPATH=str(common.BENCHMARK_ROOT)+':'+str(common.BENCHMARK_ROOT / 'evaluator')+':'+str(common.REPO_ROOT),
                       PTOBENCH_WORKER_WIRE='/wire',PTOBENCH_VALIDATION_SOCKET='/validation.sock',
                       PTOBENCH_SKIP_CORRECTNESS='0')
            program = ('import pathlib,torch; from golden import TensorSpec; import golden.runner as r; '
                'from candidate_execution import bridge; '
                'assert not pathlib.Path('+repr(str(fixture / 'out/y.pt'))+').exists(); '
                's=TensorSpec("y",[2],torch.float32); s.direction="out"; '
                '\nwith bridge("fixture_000"): r._validate([s],{"y":torch.tensor([1.,2.])},{},0,0,{})')
            result = subprocess.run(view.argv([TEST_PYTHON,'-c',program],cwd='/tmp',environment=env),
                                    capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(validation.validated,{'fixture_000'})
            self.assertFalse(validation.failed)
            bad = program.replace('torch.tensor([1.,2.])','torch.tensor([9.,9.])')
            result = subprocess.run(view.argv([TEST_PYTHON,'-c',bad],cwd='/tmp',environment=env),
                                    capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertTrue(validation.failed)


class IsolatedLauncherTests(unittest.TestCase):
    def test_basic_pi_cli_budget_finish_and_native_history(self):
        from test_launcher import LauncherTests
        case = LauncherTests(next(name for name in dir(LauncherTests) if name.startswith('test_')))
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.launcher_config(backend='pi', budget=1)
        case.tool.write_text('#!'+case.python+'\n'+r'''
import sys,json,os,subprocess,uuid
from pathlib import Path
if '--version' in sys.argv: print('fake pi 1.0');raise SystemExit()
if '--help' in sys.argv: print('--print --json --model --provider --thinking --mode --session-dir');raise SystemExit()
identity=str(uuid.uuid4())
native=Path(sys.argv[sys.argv.index('--session-dir')+1])/(identity+'.jsonl');native.parent.mkdir(parents=True,exist_ok=True)
header={'type':'session','id':identity,'cwd':os.getcwd()};native.write_text(json.dumps(header)+'\n');print(json.dumps(header),flush=True)
assert not Path('/data1/home/ziruiwang/pypto-lib/pto_wiki').exists()
assert not Path('/data1/home/ziruiwang/pypto-ci-head-672871bd-wt').exists()
try: Path('.benchmark_context/config.py').write_text('bad')
except OSError: pass
else: raise AssertionError('contract was writable')
r=subprocess.run(['ptobench','evaluate','.'],capture_output=True,text=True);assert r.returncode==0,(r.returncode,r.stdout,r.stderr)
''')
        case.env['PTOBENCH_SANDBOX']='1'
        process, result = case.launch()
        root = Path(result['session_root'])
        self.assertEqual(process.returncode,0,(process.stderr.decode(),(root/'stderr.log').read_text()))
        final=json.loads((root/'final_result.json').read_text())
        self.assertEqual(final['hardware_actions_used'],1)
        self.assertEqual(final['finish_actions_used'],1)
        self.assertEqual(final['final_verified_correctness'],'passed')
        self.assertEqual(result['agent_recording']['status'],'recorded')
        self.assertTrue((root/'filesystem_isolation.json').exists())

    def test_cannbot_pi_and_codex_children_profiles_and_fork(self):
        from test_cannbot import CANNBotTests, FAKE_NATIVE
        for backend in ('pi', 'codex'):
            with self.subTest(backend=backend):
                case = CANNBotTests('test_codex_mcp_fresh_sessions_same_budget_and_original_resources')
                case.setUp()
                try:
                    case.cannbot_config(backend=backend, budget=5, fork_child=True)
                    # Control belongs to the trusted test harness; embed only
                    # public fake-Agent instructions, never mount its host dir.
                    native = FAKE_NATIVE.replace(
                        "control=json.loads(Path(os.environ['S5_CONTROL']).read_text())",
                        'control='+repr({'node':case.node, 'fork_child':True}))
                    native = native.replace("root=Path(config['run_root'])", """root=Path(config['run_root'])
assert not Path('/data1/home/ziruiwang/pypto-lib/pto_wiki').exists()
capabilities=json.loads(Path('.benchmark_context/tuning_capabilities.json').read_text())
assert capabilities['baseline']==json.loads(Path('candidate/tuning.json').read_text())
assert capabilities['default_configuration_validation']['other_values_verified'] is False
assert 'complete defaults vary by Case' in Path('.benchmark_context/README.md').read_text()
""")
                    case.tool.write_text('#!'+case.python+'\n'+native)
                    case.env['PTOBENCH_SANDBOX']='1'
                    process, value = case.launch()
                    root = Path(value['session_root'])
                    self.assertEqual(process.returncode, 0,
                        (value, process.stderr.decode(), (root/'stderr.log').read_text()))
                    final = json.loads((root/'final_result.json').read_text())
                    self.assertEqual(final['hardware_actions_used'], 2)
                    self.assertEqual(final['finish_actions_used'], 1)
                    self.assertEqual(final['final_verified_correctness'], 'passed')
                    self.assertEqual(value['agent_recording']['status'], 'recorded')
                    self.assertEqual(len(case.dispatches(root)), 3)
                    for dispatch in case.dispatches(root):
                        prompt = (root/'method/dispatches'/dispatch['id']/'prompt.txt').read_text()
                        self.assertIn('.benchmark_context/tuning_capabilities.json', prompt)
                        self.assertIn('complete defaults vary by Case', prompt)
                        self.assertIn('PYPTO + false', prompt)
                        self.assertIn('candidate/tuning.json', prompt)
                        self.assertNotIn('workspace-root tuning.json', prompt)
                        self.assertIn('ptobench evaluate .', prompt)
                    import session
                    self.assertEqual(len(session.read_session(root)['workspaces']), 2)
                finally:
                    case.doCleanups()

    def test_interrupt_stops_namespace_descendants_and_finalizes(self):
        from test_launcher import LauncherTests
        case = LauncherTests(next(name for name in dir(LauncherTests) if name.startswith('test_')))
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.launcher_config(backend='pi', budget=3)
        case.tool.write_text('#!'+case.python+'\n'+r'''
import sys,json,os,subprocess,uuid,time
from pathlib import Path
if '--version' in sys.argv: print('fake pi 1.0');raise SystemExit()
if '--help' in sys.argv: print('--print --json --model --provider --thinking --mode --session-dir');raise SystemExit()
identity=str(uuid.uuid4());native=Path(sys.argv[sys.argv.index('--session-dir')+1])/(identity+'.jsonl')
native.parent.mkdir(parents=True,exist_ok=True)
header={'type':'session','id':identity,'cwd':os.getcwd()};native.write_text(json.dumps(header)+'\n');print(json.dumps(header),flush=True)
assert subprocess.run(['ptobench','evaluate','.']).returncode==0
subprocess.Popen([sys.executable,'-c',"import os,time,signal,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);p=pathlib.Path(os.environ['PTOBENCH_AGENT_SESSION_DIR'])/'heartbeat';\nwhile True: p.write_text(str(time.monotonic()));time.sleep(.05)"],start_new_session=True)
print('ISOLATED_READY',flush=True)
while True:time.sleep(.05)
''')
        case.env['PTOBENCH_SANDBOX']='1'
        process = case.launch(asynchronous=True)
        deadline = time.monotonic()+30
        root = None
        while time.monotonic() < deadline:
            paths = list(case.runs.rglob('stdout.log')) if case.runs.exists() else []
            if paths and 'ISOLATED_READY' in paths[0].read_text():
                root = paths[0].parent
                break
            if process.poll() is not None:
                break
            time.sleep(.05)
        self.assertIsNotNone(root)
        process.send_signal(signal.SIGTERM)
        value, out, err = case.result(process)
        final = json.loads((root/'final_result.json').read_text())
        self.assertEqual(final['close_reason'], 'user_interrupt', (value,err.decode()))
        self.assertEqual(final['finish_actions_used'], 1)
        self.assertEqual(final['final_verified_correctness'], 'passed')
        private = root/'agent/native/heartbeat'
        previous = private.read_text()
        time.sleep(.2)
        self.assertEqual(private.read_text(), previous)
