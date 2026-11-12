"""Actual kernel boundary tests; no mocked path checks or model calls."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import signal
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import isolation
import sandbox_supervisor
import agent_launch
from session_test_support import SessionTestBase

TEST_PYTHON = str(Path(sys.prefix) / 'bin/python')


class TrustedBubblewrapResolutionTests(unittest.TestCase):
    def test_broker_runtime_stubs_are_traversable_but_not_listable_by_agent(self):
        runtime = Path(isolation._EMPTY_RUNTIME.name)
        self.assertEqual(stat.S_IMODE(runtime.stat().st_mode), 0o711)
        for name in ('passwd', 'group'):
            self.assertTrue((runtime / name).is_file())
            self.assertTrue((runtime / name).stat().st_mode & stat.S_IROTH)

    def test_bubblewrap_uses_verified_system_binary_not_overrides_or_home(self):
        with tempfile.TemporaryDirectory(prefix='untrusted-bwrap-') as temporary:
            root = Path(temporary)
            attacker_path = root / 'attacker-bin'
            home_binary = root / 'home/.local/share/npm/bin/bwrap'
            attacker_path.mkdir()
            home_binary.parent.mkdir(parents=True)
            for path in (attacker_path / 'bwrap', home_binary):
                path.write_text('#!/bin/sh\nexit 99\n')
                path.chmod(0o755)
            with patch.dict(os.environ, {'PATH': str(attacker_path),
                                         'HOME': str(root / 'home'),
                                         'PTOBENCH_BWRAP': str(home_binary)}):
                selected = Path(isolation.executable()).resolve(strict=True)

            trusted_system_roots = [Path(directory).resolve() for directory in os.defpath.split(os.pathsep)]
            self.assertTrue(any(selected.is_relative_to(directory) for directory in trusted_system_roots))
            self.assertNotEqual(selected, home_binary.resolve())
            self.assertNotEqual(selected, (attacker_path / 'bwrap').resolve())
            self.assertEqual(agent_launch._trusted_binary(selected), str(selected))


class AgentMountManifestTests(unittest.TestCase):
    def test_agent_view_uses_one_real_workspace_bind_and_single_file_cli_shim(self):
        import common

        with tempfile.TemporaryDirectory(prefix='agent-mount-manifest-') as temporary:
            root = Path(temporary) / 'session'
            workspaces = root / 'workspaces'
            workspace = workspaces / 'worker_000'
            for name in ('worker_000', 'worker_001'):
                worker = workspaces / name
                (worker / 'candidate').mkdir(parents=True)
                (worker / '.benchmark_context').mkdir()
                (worker / '.benchmark_outputs').mkdir()
                (worker / 'candidate/candidate.py').write_text('pass\n')
            workspace.mkdir(exist_ok=True)
            workspace_context = workspace / '.benchmark_context'
            (workspace_context / 'workspace.json').write_text('{}\n')
            (root / 'mode.json').write_text('{}\n')
            (root / 'method_request.json').write_text('{}\n')
            (root / 'profile.json').write_text('{}\n')
            (root / 'prompt.txt').write_text('private prompt\n')

            def bind_socket(path):
                path.parent.mkdir(parents=True, exist_ok=True)
                endpoint = socket.socket(socket.AF_UNIX)
                endpoint.bind(str(path))
                self.addCleanup(endpoint.close)
                return path

            rpc = bind_socket(root / 'sockets/rpc.sock')
            gateway = bind_socket(root / 'sockets/api.sock')
            tools = root / 'agent/pi-tools'
            tools.mkdir(parents=True)
            extension = tools / 'cannbot-extension.ts'
            extension.write_text('export default () => {};\n')
            native = root / 'agent/native'
            native.mkdir(parents=True)
            transport_directory, transport = sandbox_supervisor._copy_cli_transport(root)

            environment = {
                'PATH': os.environ['PATH'],
                'PTOBENCH_RPC_SOCKET': str(rpc),
                'PTOBENCH_API_PROXY_SOCKET': str(gateway),
                'PTOBENCH_CLIENT_PATH': str(transport),
            }
            with patch.object(agent_launch, 'prepare',
                              side_effect=lambda command, cwd, request_directory:
                              (command, {'PATH': os.defpath})):
                command, launch_environment = isolation.agent_command(
                    ['pi', '--extension', str(extension), '--session-dir', str(native), '{prompt}'],
                    cwd=workspace, environment=environment, workspace=workspace,
                    session_root=root, cann_bot_extension=extension)

            manifest = json.loads((root / 'filesystem_isolation.json').read_text())
            mounts = manifest['mounts']
            workspace_mounts = [item for item in mounts
                                 if item['source'] == str(workspaces.resolve())]
            self.assertEqual(len(workspace_mounts), 1)
            self.assertEqual(workspace_mounts[0]['access'], 'rw')
            self.assertFalse(any('/candidates' in item['destination']
                                 or '/public' in item['destination'] for item in mounts))
            self.assertFalse(manifest['symlinks'])

            cli = [item for item in mounts if item['destination'] == '/ptobench/ptobench']
            self.assertEqual(len(cli), 1)
            self.assertEqual(cli[0]['access'], 'ro')
            self.assertEqual(Path(cli[0]['source']).name, 'sandbox_client.py')
            self.assertNotEqual(cli[0]['source'], str(common.BENCHMARK_ROOT / 'ptobench'))
            self.assertNotEqual(cli[0]['source'], str(common.BENCHMARK_ROOT / 'evaluator/agent_cli.py'))
            self.assertEqual(Path(cli[0]['source']).read_bytes(),
                             (common.BENCHMARK_ROOT / 'evaluator/sandbox_client.py').read_bytes())

            config_mount = [item for item in mounts
                            if item['destination'] == '/ptobench/pi-config']
            self.assertEqual(config_mount, [{'source': 'tmpfs',
                                             'destination': '/ptobench/pi-config',
                                             'access': 'tmpfs'}])
            for name in ('settings.json', 'auth.json'):
                self.assertTrue(any(item['destination'] == f'/ptobench/pi-config/{name}'
                                    and item['access'] == 'ro' for item in mounts))
            self.assertTrue(any(item['destination'] == '/ptobench/pi-tools'
                                and item['access'] == 'ro' for item in mounts))
            self.assertTrue(any(item['source'] == str(native.resolve())
                                and item['access'] == 'rw' for item in mounts))
            self.assertIn('/ptobench/pi-tools/cannbot-extension.ts', command)
            sandbox_environment = {
                command[index + 1]: command[index + 2]
                for index, value in enumerate(command[:-2]) if value == '--setenv'
            }
            self.assertEqual(sandbox_environment['PI_CODING_AGENT_DIR'], '/ptobench/pi-config')
            self.assertEqual(sandbox_environment['PTOBENCH_PI_TOOLS_DIR'], '/ptobench/pi-tools')
            self.assertEqual(sandbox_environment['PTOBENCH_AGENT_SESSION_DIR'],
                             '/ptobench/agent-sessions')
            pi_tools = next(Path(item['source']) for item in mounts
                            if item['destination'] == '/ptobench/pi-tools')
            shell_config = json.loads((pi_tools / 'shell.json').read_text())
            self.assertEqual(shell_config['workspace_root'], str(workspaces.resolve()))
            self.assertEqual(sandbox_environment['PATH'].split(os.pathsep)[0], '/ptobench')
            self.assertNotIn(str(common.BENCHMARK_ROOT), sandbox_environment['PATH'])

            forbidden = (str(common.BENCHMARK_ROOT), str(common.BENCHMARK_ROOT / 'ptobench'),
                         str(common.BENCHMARK_ROOT / 'evaluator/agent_cli.py'),
                         str(common.BENCHMARK_ROOT / 'evaluator/sandbox_supervisor.py'),
                         str(common.BENCHMARK_ROOT / 'launcher/methods.py'),
                         str(root / 'mode.json'), str(root / 'method_request.json'),
                         str(root / 'profile.json'), str(root / 'prompt.txt'))
            for path in forbidden:
                self.assertFalse(any(item['source'] == path or item['destination'] == path
                                     for item in mounts), path)
            self.assertFalse(any('/candidates' in item['source'] or '/public' in item['source']
                                 for item in mounts))

    def test_bubblewrap_does_not_inherit_host_variables_without_clearenv(self):
        with patch.dict(os.environ, {'UNAUTHORIZED_ENV': 'hidden'}):
            launcher_environment = isolation.sandbox_launcher_environment(os.environ)
            result = subprocess.run(
                isolation.View().system().argv(
                    ['/usr/bin/python3', '-c',
                     'import os; assert "UNAUTHORIZED_ENV" not in os.environ'],
                    cwd='/tmp', environment={},
                ),
                env=launcher_environment, capture_output=True, text=True, timeout=20,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


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
            environment = isolation.clean_environment()
            result = subprocess.run(view.argv([TEST_PYTHON, '-c', program], cwd='/work',
                environment=environment), env=isolation.sandbox_launcher_environment(environment),
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(secret.read_text(), 'UNAUTHORIZED_CANARY')
            self.assertEqual((public / 'contract').read_text(), 'contract')

    def test_tmpfs_is_private_and_outside_candidate_manifest_and_freeze(self):
        import workspace as workspace_tree

        with tempfile.TemporaryDirectory(prefix='isolated-tmpfs-') as temporary:
            root = Path(temporary)
            candidate = root / 'workspaces/worker_000/candidate'
            candidate.mkdir(parents=True)
            (candidate / 'solution.py').write_text('VALUE = 42\n')
            before = workspace_tree.source_manifest(candidate)
            token = os.urandom(8).hex()
            host_tmp = Path('/tmp') / f'ptobench-host-{token}'
            self.assertFalse(host_tmp.exists())
            host_tmp.write_text('host tmp remains unchanged\n')
            self.addCleanup(host_tmp.unlink, missing_ok=True)

            probe = f'''from pathlib import Path
path = Path("/tmp/ptobench-host-{token}")
assert not path.exists()
path.write_text("isolated tmpfs content")
assert path.read_text() == "isolated tmpfs content"
'''
            view = isolation.View().system().add(candidate, '/candidate', write=True)
            environment = isolation.clean_environment()
            result = subprocess.run(view.argv([TEST_PYTHON, '-c', probe], cwd='/candidate',
                                               environment=environment),
                                    env=isolation.sandbox_launcher_environment(environment),
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(host_tmp.read_text(), 'host tmp remains unchanged\n')
            self.assertEqual(workspace_tree.source_manifest(candidate), before)

            frozen = root / 'frozen/candidate'
            workspace_tree.copy_candidate_tree(candidate, frozen)
            self.assertEqual(workspace_tree.source_manifest(frozen), before)

    def test_locked_runtime_and_pi_mounts_work_as_dedicated_agent_without_devices(self):
        import grp
        import hashlib
        import pwd
        import shutil
        import environment_lock

        pi = shutil.which('pi')
        node = shutil.which('node')
        self.assertIsNotNone(pi)
        self.assertIsNotNone(node)
        agent_uid = pwd.getpwnam('ptobench-agent').pw_uid
        with tempfile.TemporaryDirectory(prefix='agent-runtime-manifest-') as temporary:
            base = Path(temporary)
            share_gid = grp.getgrnam('ptobench-share').gr_gid

            def shared(path, mode):
                os.chown(path, os.getuid(), share_gid)
                path.chmod(mode)

            shared(base, 0o2710)
            root = base / 'session'
            root.mkdir()
            shared(root, 0o2710)
            workspaces = root / 'workspaces'
            workspaces.mkdir()
            shared(workspaces, 0o2750)
            workspace = workspaces / 'worker_000'
            workspace.mkdir()
            shared(workspace, 0o2750)
            for name, mode in (('candidate', 0o2770),
                               ('.benchmark_context', 0o2750),
                               ('.benchmark_outputs', 0o2750)):
                directory = workspace / name
                directory.mkdir()
                shared(directory, mode)
            candidate_source = workspace / 'candidate/solution.py'
            candidate_source.write_text('VALUE = 42\n')
            os.chown(candidate_source, os.getuid(), share_gid)
            candidate_source.chmod(0o660)

            endpoints = root / 'broker-endpoints'
            endpoints.mkdir()
            shared(endpoints, 0o2710)
            rpc, gateway = endpoints / 'rpc.sock', endpoints / 'api.sock'
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(rpc))
            server.listen(1)
            self.addCleanup(server.close)
            api = socket.socket(socket.AF_UNIX)
            api.bind(str(gateway))
            api.listen(1)
            self.addCleanup(api.close)
            transport_directory, transport = sandbox_supervisor._copy_cli_transport(root)

            environment = {'PATH': os.environ['PATH'],
                           'PTOBENCH_RPC_SOCKET': str(rpc),
                           'PTOBENCH_API_PROXY_SOCKET': str(gateway),
                           'PTOBENCH_CLIENT_PATH': str(transport),
                           'PYPTO_PROVIDER_API_KEY': 'must-be-replaced',
                           'PTOBENCH_UPSTREAM_API_KEY': 'TEST_UPSTREAM_SECRET'}
            pi_command, pi_environment = isolation.agent_command(
                [str(Path(pi).resolve()), '--version'], cwd=workspace, environment=environment,
                workspace=workspace, session_root=root, backend='pi')
            pi_manifest = json.loads((root / 'filesystem_isolation.json').read_text())
            mounts = pi_manifest['mounts']
            self.assertTrue(any(item['source'] == str(Path(node).resolve())
                                and item['destination'] == '/ptobench/runtime/node'
                                and item['access'] == 'ro'
                                for item in mounts))
            self.assertTrue(any(item['source'] == str(Path(pi).resolve().parent)
                                and item['destination'] == '/ptobench/runtime/pi'
                                and item['access'] == 'ro'
                                for item in mounts))
            pi_package = Path(pi).resolve().parents[2]
            self.assertTrue(any(item['source'] == str((pi_package / 'node_modules/jiti').resolve())
                                and item['destination'] == '/ptobench/runtime/node_modules/jiti'
                                and item['access'] == 'ro'
                                for item in mounts))
            self.assertTrue(any(item['destination'] == '/ptobench/ptobench'
                                and item['access'] == 'ro' for item in mounts))
            self.assertEqual(agent_launch.run_process(
                pi_command, cwd=workspace, environment=pi_environment), 0)

            received = {}

            def respond_to_cli():
                connection, _ = server.accept()
                with connection:
                    received.update(json.loads(connection.makefile('rb').readline()))
                    connection.sendall(b'{"output":"agent runtime transport ok\\n"}\n{"exit_code":0}\n')

            response_thread = threading.Thread(target=respond_to_cli, daemon=True)
            response_thread.start()
            locked_python_sha256 = hashlib.sha256(
                environment_lock.locked_python_executable().read_bytes()).hexdigest()
            probe = f'''import os, pathlib, subprocess, sys
import hashlib
assert os.getuid() == {agent_uid}
assert set(os.getgroups()) == {{{share_gid}}}
status = pathlib.Path('/proc/self/status').read_text()
assert 'CapEff:\\t0000000000000000' in status
assert os.environ['PYPTO_PROVIDER_API_KEY'] == 'ptobench-local-model-gateway'
assert 'PTOBENCH_UPSTREAM_API_KEY' not in os.environ
for process_environment in pathlib.Path('/proc').glob('[0-9]*/environ'):
 try:
  environment = process_environment.read_bytes()
  assert b'TEST_UPSTREAM_SECRET' not in environment
  assert b'must-be-replaced' not in environment
 except OSError:
  pass
assert hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest() == {locked_python_sha256!r}
import pypto
module = pathlib.Path(pypto.__file__).resolve()
assert module.is_file() and module.read_bytes()
assert not os.access(module, os.W_OK)
transport = pathlib.Path('/ptobench/ptobench')
assert transport.is_file() and os.access(transport, os.R_OK | os.X_OK)
result = subprocess.run([str(transport), '--help'], capture_output=True, text=True)
assert result.returncode == 0 and result.stdout == 'agent runtime transport ok\\n', (result.stdout, result.stderr)
for path in ('/dev/davinci0', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc'):
 try:
  descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
 except OSError:
  continue
 else:
  os.close(descriptor)
  raise AssertionError('Agent opened an NPU device: ' + path)
'''
            runtime_command, runtime_environment = isolation.agent_command(
                [TEST_PYTHON, '-c', probe], cwd=workspace, environment=environment,
                workspace=workspace, session_root=root)
            self.assertEqual(agent_launch.run_process(
                runtime_command, cwd=workspace, environment=runtime_environment), 0)
            response_thread.join(5)
            self.assertFalse(response_thread.is_alive())
            self.assertEqual(received['argv'], ['--help'])
            self.assertEqual(received['cwd'], str(workspace))
            for path in root.rglob('*'):
                if path.is_file():
                    self.assertNotIn(b'TEST_UPSTREAM_SECRET', path.read_bytes(), str(path))
                    self.assertNotIn(b'must-be-replaced', path.read_bytes(), str(path))

    def test_real_pi_starts_with_readonly_config_and_broker_readable_native_session(self):
        import grp
        import shutil
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from modes._shared import provider, recording

        pi = shutil.which('pi')
        self.assertIsNotNone(pi)
        upstream_headers = []
        upstream_requests = []
        calls = [
            ('read', {'path': '/ptobench/pi-config/settings.json'}),
            ('write', {'path': '/ptobench/agent-sessions/PI_PRIVATE_WRITE_CANARY',
                       'content': 'must not be written'}),
            ('edit', {'path': '/ptobench/pi-tools/cannbot-extension.ts',
                      'edits': [{'oldText': 'export default', 'newText': 'export const'}]}),
            ('grep', {'path': '/ptobench/pi-config', 'pattern': 'apiKey'}),
            ('find', {'path': '/ptobench/pi-tools', 'pattern': '**/*'}),
            ('ls', {'path': '/ptobench/agent-sessions'}),
        ]
        tool_calls = [{'index': index, 'id': f'call-private-{index}', 'type': 'function',
                       'function': {'name': name, 'arguments': json.dumps(arguments)}}
                      for index, (name, arguments) in enumerate(calls)]
        tool_response = (
            b'data: ' + json.dumps({'id': 'chatcmpl-tools', 'object': 'chat.completion.chunk',
                'created': 1, 'model': 'test-model', 'choices': [{'index': 0,
                'delta': {'role': 'assistant', 'tool_calls': tool_calls},
                'finish_reason': None}]}).encode() + b'\n\n'
            + b'data: ' + json.dumps({'id': 'chatcmpl-tools', 'object': 'chat.completion.chunk',
                'created': 1, 'model': 'test-model', 'choices': [{'index': 0,
                'delta': {}, 'finish_reason': 'tool_calls'}]}).encode() + b'\n\n'
            + b'data: [DONE]\n\n')
        response = (
            b'data: {"id":"chatcmpl-startup","object":"chat.completion.chunk","created":1,'
            b'"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant",'
            b'"content":"PI_STARTUP_CANARY"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-startup","object":"chat.completion.chunk","created":1,'
            b'"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n')

        class Provider(BaseHTTPRequestHandler):
            def do_POST(self):
                upstream_headers.append(self.headers.get('Authorization'))
                upstream_requests.append(json.loads(
                    self.rfile.read(int(self.headers.get('Content-Length', '0')))))
                body = tool_response if len(upstream_requests) == 1 else response
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        self.addCleanup(upstream.shutdown)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream_thread.join, 5)

        with tempfile.TemporaryDirectory(prefix='real-pi-config-') as temporary:
            base = Path(temporary)
            base.chmod(0o711)
            share_gid = grp.getgrnam('ptobench-share').gr_gid

            def shared(path, mode):
                os.chown(path, os.getuid(), share_gid)
                path.chmod(mode)

            root = base / 'session'
            root.mkdir()
            shared(root, 0o2710)
            workspaces = root / 'workspaces'
            workspaces.mkdir()
            shared(workspaces, 0o2750)
            workspace = workspaces / 'worker_000'
            workspace.mkdir()
            shared(workspace, 0o2750)
            for name in ('candidate', '.benchmark_context', '.benchmark_outputs'):
                (workspace / name).mkdir()
                shared(workspace / name, 0o2770 if name == 'candidate' else 0o2750)

            endpoints = root / 'broker-endpoints'
            endpoints.mkdir()
            shared(endpoints, 0o2710)
            rpc = endpoints / 'rpc.sock'
            rpc_server = socket.socket(socket.AF_UNIX)
            rpc_server.bind(str(rpc))
            rpc_server.listen(1)
            self.addCleanup(rpc_server.close)

            api_socket = endpoints / 'api.sock'
            gateway_environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                                   'PTOBENCH_UPSTREAM_API_KEY': 'TEST_UPSTREAM_SECRET'}
            gateway = subprocess.Popen([
                sys.executable, str(Path(isolation.__file__).with_name('network_proxy.py')),
                'serve-api', '--socket', str(api_socket),
                '--base-url', f'http://127.0.0.1:{upstream.server_port}/v1',
                '--api', 'openai-completions',
            ], env=gateway_environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.addCleanup(lambda: gateway.terminate() if gateway.poll() is None else None)
            deadline = time.monotonic() + 5
            while not api_socket.exists() and time.monotonic() < deadline:
                if gateway.poll() is not None:
                    self.fail(gateway.stderr.read().decode())
                time.sleep(.02)
            self.assertTrue(api_socket.exists())

            profile = {
                'backend': 'pi', 'reasoning': 'off', 'env': {},
                'argv': [str(Path(pi).resolve()), '--print', '--mode', 'json',
                         '--tools', 'read,write,edit,grep,find,ls,skill,todowrite,todoread',
                         '--provider', 'test-provider', '--model', 'test-model',
                         '--thinking', 'off', '{prompt}'],
                'mode_provider': {
                    'provider': 'test-provider', 'base_url': f'http://127.0.0.1:{upstream.server_port}/v1',
                    'api': 'openai-completions',
                    'model': {'id': 'test-model', 'contextWindow': 8192, 'maxTokens': 1024},
                },
            }
            prepared = provider.prepare(profile, root, gateway=True)
            extension_source = Path(isolation.__file__).parents[1] / 'modes/cannbot/bridge/pi_task.ts'
            extension = root / 'agent/pi-tools/cannbot-extension.ts'
            extension.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(extension_source, extension)
            os.chown(extension, -1, share_gid)
            extension.chmod(0o440)
            prepared['argv'][prepared['argv'].index('{prompt}'):prepared['argv'].index('{prompt}')] = [
                '--extension', str(extension)]
            transport_directory, transport = sandbox_supervisor._copy_cli_transport(root)
            self.addCleanup(shutil.rmtree, transport_directory, True)
            environment = {
                **prepared['env'], 'PATH': os.environ['PATH'],
                'PTOBENCH_RPC_SOCKET': str(rpc),
                'PTOBENCH_API_PROXY_SOCKET': str(api_socket),
                'PTOBENCH_CLIENT_PATH': str(transport),
            }
            config = root / 'agent/pi-config'
            argv, launch_environment = isolation.agent_command(
                [*prepared['argv'], 'Say hello with the exact text PI_STARTUP_CANARY.'],
                cwd=workspace, environment=environment, workspace=workspace, session_root=root,
                cann_bot_extension=extension, backend='pi')
            settings_before = (config / 'settings.json').read_bytes()
            models_before = (config / 'models.json').read_bytes()
            process = subprocess.Popen(argv, cwd=workspace, env=launch_environment,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=45)
            finally:
                agent_launch.cleanup_request_after_process(launch_environment, process)
            self.assertEqual(process.returncode, 0, stdout + stderr)
            native = list((root / 'agent/native').glob('*.jsonl'))
            self.assertEqual(len(native), 1)
            self.assertIn(b'PI_STARTUP_CANARY', native[0].read_bytes())
            native_metadata = native[0].stat()
            agent_uid, _ = agent_launch._agent_identity()
            self.assertEqual(native_metadata.st_uid, agent_uid)
            self.assertEqual(native_metadata.st_gid, share_gid)
            self.assertTrue(native_metadata.st_mode & stat.S_IRGRP)
            native_inode = native_metadata.st_ino
            self.assertGreater(len(native[0].read_bytes().splitlines()), 1)
            archive = recording.collect(root, prepared, workspace, started=True,
                                       close_reason='completed')
            self.assertEqual(archive['status'], 'recorded')
            self.assertEqual(archive['files'][0]['path'], str(native[0].relative_to(root)))
            archived_metadata = native[0].stat()
            self.assertEqual(archived_metadata.st_uid, os.getuid())
            self.assertEqual(stat.S_IMODE(archived_metadata.st_mode), 0o444)
            self.assertNotEqual(archived_metadata.st_ino, native_inode)
            self.assertTrue(upstream_headers)
            self.assertTrue(all(value == 'Bearer TEST_UPSTREAM_SECRET'
                                for value in upstream_headers))
            self.assertGreaterEqual(len(upstream_requests), 2)
            registered_tools = json.dumps(upstream_requests[0].get('tools', []))
            self.assertIn('skill', registered_tools)
            self.assertIn('todowrite', registered_tools)
            self.assertNotRegex(registered_tools, r'(?i)web[\s_-]?(search|fetch)')
            self.assertIn('private Pi runtime files', json.dumps(upstream_requests[1]))
            self.assertNotIn('TEST_UPSTREAM_SECRET', json.dumps(upstream_requests))
            for path in root.rglob('*'):
                if path.is_file():
                    self.assertNotIn(b'TEST_UPSTREAM_SECRET', path.read_bytes(), str(path))
            self.assertFalse((root / 'agent/native/PI_PRIVATE_WRITE_CANARY').exists())
            self.assertEqual(extension.read_bytes(), extension_source.read_bytes())
            self.assertEqual(stat.S_IMODE((config / 'settings.json').stat().st_mode), 0o440)
            self.assertEqual(stat.S_IMODE((config / 'models.json').stat().st_mode), 0o440)
            self.assertEqual((config / 'settings.json').read_bytes(), settings_before)
            self.assertEqual((config / 'models.json').read_bytes(), models_before)
            self.assertEqual(json.loads((config / 'auth.json').read_text()), {})
            self.assertFalse((config / 'settings.json.lock').exists())

    def test_installed_upstream_cache_exposes_only_headers(self):
        for checkout in Path(sys.prefix).glob('lib/python*/site-packages/simpler_setup/_assets/build/pto-isa'):
            program = ('from pathlib import Path; p=Path('+repr(str(checkout))+'); '
                       'assert (p/"include/pto/pto-inst.hpp").is_file(); '
                       'assert not (p/"docs").exists(); assert not (p/".git").exists(); '
                       'assert not (p/"tests").exists()')
            view = isolation.View().system()
            environment = isolation.clean_environment()
            result = subprocess.run(view.argv([TEST_PYTHON, '-c', program], cwd='/tmp',
                environment=environment), env=isolation.sandbox_launcher_environment(environment),
                capture_output=True, text=True, timeout=20)
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

    def test_pi_shell_wrapper_rebinds_real_workspace_and_masks_private_roots(self):
        import pi_shell_wrapper

        with tempfile.TemporaryDirectory(prefix='pi-shell-boundary-') as temporary:
            workspaces = Path(temporary) / 'session/workspaces'
            workspace = workspaces / 'worker_000'
            workspace.mkdir(parents=True)
            argv = pi_shell_wrapper.sandbox_argv(
                ['-c', 'true'], cwd=workspace,
                environment={'PTOBENCH_RPC_SOCKET': '/ptobench/cli.sock',
                             'PTOBENCH_AGENT_DISPATCH_ID': 'dispatch-id',
                             'PTOBENCH_AGENT_PARENT_TOOL_CALL_ID': 'tool-call-id'},
                config={'bwrap': isolation.executable(), 'workspace_root': str(workspaces)})

            workspace_index = argv.index('--bind')
            self.assertEqual(argv[workspace_index + 1:workspace_index + 3],
                             [str(workspaces), str(workspaces)])
            hidden = {argv[index + 1] for index, option in enumerate(argv[:-1])
                      if option == '--tmpfs'}
            self.assertTrue({'/ptobench/pi-config', '/ptobench/pi-tools',
                             '/ptobench/agent-sessions'} <= hidden)
            self.assertIn('--ro-bind', argv)
            self.assertIn('/', argv)
            self.assertNotIn('/ptobench/candidates', argv)
            self.assertNotIn('/ptobench/public', argv)
            self.assertIn('PTOBENCH_RPC_SOCKET', argv)
            self.assertIn('dispatch-id', argv)
            self.assertIn('tool-call-id', argv)
            self.assertEqual(argv[-3:], ['/bin/bash', '-c', 'true'])

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
            environment = isolation.clean_environment()
            result = subprocess.run(isolation.View().system().argv([TEST_PYTHON, '-c', program],
                cwd='/tmp', environment=environment),
                env=isolation.sandbox_launcher_environment(environment),
                capture_output=True, text=True, timeout=20)
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
                cwd='/tmp', env=isolation.sandbox_launcher_environment(environment),
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            proxy.terminate()
            proxy.wait(timeout=5)
            thread.join(5)

    def test_agent_can_modify_candidate_but_not_private_config_feedback_or_secret(self):
        import common
        from modes._shared import provider

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

        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        self.addCleanup(upstream.shutdown)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream_thread.join, 5)

        with tempfile.TemporaryDirectory(prefix='agent-gateway-') as temporary:
            base = Path(temporary)
            base.chmod(0o711)
            share_gid = agent_launch.grp.getgrnam(agent_launch.SHARE_GROUP).gr_gid

            def shared(path, mode):
                os.chown(path, os.getuid(), share_gid)
                path.chmod(mode)

            root = base / 'run'
            root.mkdir()
            shared(root, 0o2710)
            workspaces = root / 'workspaces'
            workspaces.mkdir()
            shared(workspaces, 0o2750)
            workspace = workspaces / 'initial'
            workspace.mkdir()
            shared(workspace, 0o2750)
            candidate = workspace / 'candidate'
            candidate.mkdir()
            shared(candidate, 0o2770)
            (candidate / 'candidate.py').write_text('candidate = True\n')
            (candidate / 'tuning.json').write_text('{"memory_planner":"PYPTO"}\n')
            for path in candidate.iterdir():
                shared(path, 0o660)
            context = workspace / '.benchmark_context'
            context.mkdir()
            shared(context, 0o2750)
            (context / 'README.md').write_text('public contract\n')
            shared(context / 'README.md', 0o440)
            outputs = workspace / '.benchmark_outputs/combined'
            outputs.mkdir(parents=True)
            shared(workspace / '.benchmark_outputs', 0o2750)
            shared(outputs, 0o2750)
            result_file = outputs / 'latest_result.json'
            result_file.write_text('{"run": 1}\n')
            shared(result_file, 0o440)

            endpoints = root / 'broker-endpoints'
            endpoints.mkdir()
            shared(endpoints, 0o2710)
            rpc = endpoints / 'rpc.sock'
            rpc_server = socket.socket(socket.AF_UNIX)
            rpc_server.bind(str(rpc))
            rpc_server.listen(1)
            self.addCleanup(rpc_server.close)
            cli_request = {}

            def respond_to_cli():
                connection, _ = rpc_server.accept()
                with connection:
                    stream = connection.makefile('rb')
                    cli_request.update(json.loads(stream.readline()))
                    connection.sendall(b'{"output":"broker response\\n"}\n{"exit_code":0}\n')

            cli_thread = threading.Thread(target=respond_to_cli, daemon=True)
            cli_thread.start()

            api_socket = endpoints / 'api.sock'
            gateway_environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                                   'PTOBENCH_UPSTREAM_API_KEY': 'TEST_UPSTREAM_SECRET'}
            gateway = subprocess.Popen([
                sys.executable, str(Path(isolation.__file__).with_name('network_proxy.py')),
                'serve-api', '--socket', str(api_socket),
                '--base-url', f'http://127.0.0.1:{upstream.server_port}/v1',
                '--api', 'openai-responses',
            ], env=gateway_environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.addCleanup(lambda: gateway.terminate() if gateway.poll() is None else None)
            deadline = time.monotonic() + 5
            while not api_socket.exists() and time.monotonic() < deadline:
                if gateway.poll() is not None:
                    self.fail(gateway.stderr.read().decode())
                time.sleep(.02)
            self.assertTrue(api_socket.exists())

            profile = {'backend': 'pi', 'reasoning': 'off', 'env': {},
                       'argv': ['pi', '--print', '{prompt}'],
                       'mode_provider': {'provider': 'test-provider',
                           'base_url': f'http://127.0.0.1:{upstream.server_port}/v1',
                           'api': 'openai-responses',
                           'model': {'id': 'test-model', 'contextWindow': 8192, 'maxTokens': 1024}}}
            prepared = provider.prepare(profile, root, gateway=True)
            transport_directory, transport = sandbox_supervisor._copy_cli_transport(root)
            self.addCleanup(shutil.rmtree, transport_directory, True)
            environment = {**prepared['env'], 'PATH': os.environ['PATH'],
                           'PTOBENCH_RPC_SOCKET': str(rpc),
                           'PTOBENCH_API_PROXY_SOCKET': str(api_socket),
                           'PTOBENCH_CLIENT_PATH': str(transport),
                           'PYPTO_PROVIDER_API_KEY': 'must-be-replaced',
                           'PTOBENCH_UPSTREAM_API_KEY': 'TEST_UPSTREAM_SECRET'}
            probe_code = f'''import http.client, os, pathlib, shutil, socket, subprocess
candidate = pathlib.Path("candidate/candidate.py")
assert candidate.read_text() == "candidate = True\\n", "candidate seed is missing"
pathlib.Path(".benchmark_context/README.md").read_text()
result = pathlib.Path(".benchmark_outputs/combined/latest_result.json")
assert result.read_text() == "{{\\\"run\\\": 1}}\\n", "feedback seed is missing"
for protected in ("candidate", ".benchmark_context", ".benchmark_outputs"):
    try: pathlib.Path(protected).rename(protected + "_replaced")
    except OSError: pass
    else: raise AssertionError("Agent replaced protected Workspace entry: " + protected)
candidate.write_text("candidate = False\\n")
pathlib.Path("candidate/tuning.json").write_text("{{}}\\n")
for path in ("/ptobench/pi-config/settings.json", "/ptobench/pi-config/models.json",
             "/ptobench/pi-config/auth.json"):
    try: pathlib.Path(path).write_text("tampered")
    except OSError: pass
    else: raise AssertionError("Broker Pi config was writable: " + path)
lock = pathlib.Path("/ptobench/pi-config/settings.json.lock")
lock.mkdir()
(lock / "probe").write_text("temporary lock")
shutil.rmtree(lock)
for action in (lambda: result.write_text("changed"), lambda: result.unlink(),
               lambda: result.chmod(0o600),
               lambda: (result.parent / "new.json").write_text("new"),
               lambda: pathlib.Path(".benchmark_context/README.md").write_text("changed")):
    try: action()
    except OSError: pass
    else: raise AssertionError("Agent modified read-only Workspace feedback")
for hidden in ({str(common.BENCHMARK_ROOT / "ptobench")!r},
               {str(common.BENCHMARK_ROOT / "evaluator/agent_cli.py")!r},
               {str(common.BENCHMARK_ROOT / "launcher/methods.py")!r}):
    assert not pathlib.Path(hidden).exists(), hidden
assert os.environ["PYPTO_PROVIDER_API_KEY"] == "ptobench-local-model-gateway", "local placeholder was not set"
assert os.environ["PTOBENCH_RPC_SOCKET"] == "/ptobench/cli.sock", "RPC socket mapping is wrong"
assert os.environ["PTOBENCH_API_PROXY_SOCKET"] == "/ptobench/network/api.sock", "host API socket path leaked"
assert "PTOBENCH_AGENT_LAUNCH_REQUEST" not in os.environ, "launcher request leaked"
assert "TEST_UPSTREAM_SECRET" not in os.environ, "upstream key leaked"
assert "must-be-replaced" not in os.environ, "caller key was not replaced"
cli = subprocess.run(["ptobench", "--help"], capture_output=True, text=True)
assert cli.returncode == 0 and cli.stdout == "broker response\\n", (cli.stdout, cli.stderr)
connection = http.client.HTTPConnection("127.0.0.1", 18081, timeout=3)
connection.request("POST", "/v1/responses", "{{}}", {{"Authorization":"Bearer fake"}})
reply = connection.getresponse()
assert reply.status == 200 and b"response.completed" in reply.read(), reply.status
connection.request("POST", "/v1/files", "{{}}")
assert connection.getresponse().status == 403, "Gateway allowed files API"
try: socket.create_connection(("127.0.0.1", {upstream.server_port}), .5)
except OSError: pass
else: raise AssertionError("Agent reached the provider without the Gateway")
try: socket.create_connection(("1.1.1.1", 443), .5)
except OSError: pass
else: raise AssertionError("Agent reached the public network")
assert not pathlib.Path("/ptobench/runtime/bridge/network_proxy.py").exists(), "bridge source remains visible"
'''
            argv, launch_environment = isolation.agent_command(
                [TEST_PYTHON, '-c', probe_code], cwd=workspace, environment=environment,
                workspace=workspace, session_root=root)
            settings = (root / 'agent/pi-config/settings.json').read_bytes()
            models = (root / 'agent/pi-config/models.json').read_bytes()
            auth = (root / 'agent/pi-config/auth.json').read_bytes()
            process = subprocess.Popen(argv, cwd=workspace, env=launch_environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=45)
            finally:
                agent_launch.cleanup_request_after_process(launch_environment, process)
            cli_thread.join(5)
            self.assertEqual(process.returncode, 0, stdout + stderr)
            self.assertEqual(cli_request['argv'], ['--help'])
            self.assertEqual(cli_request['cwd'], str(workspace))
            self.assertEqual((candidate / 'candidate.py').read_text(), 'candidate = False\n')
            self.assertEqual((candidate / 'tuning.json').read_text(), '{}\n')
            self.assertEqual((context / 'README.md').read_text(), 'public contract\n')
            self.assertEqual(result_file.read_text(), '{"run": 1}\n')
            self.assertEqual((root / 'agent/pi-config/settings.json').read_bytes(), settings)
            self.assertEqual((root / 'agent/pi-config/models.json').read_bytes(), models)
            self.assertEqual((root / 'agent/pi-config/auth.json').read_bytes(), auth)
            self.assertTrue(upstream_headers)
            self.assertTrue(all(value == 'Bearer TEST_UPSTREAM_SECRET' for value in upstream_headers))
            for path in root.rglob('*'):
                if path.is_file():
                    self.assertNotIn(b'TEST_UPSTREAM_SECRET', path.read_bytes(), str(path))
            manifest = json.loads((root / 'filesystem_isolation.json').read_text())
            self.assertEqual(manifest['network'], 'model_api_only')
            self.assertFalse(any('/evaluator' in item['destination'] or '/launcher' in item['destination']
                                 or '/modes' in item['destination'] for item in manifest['mounts']))
            self.assertFalse(list((root / 'agent/.runtime').glob('*/network_proxy.py')))

    def test_production_agent_shell_rebinds_real_workspaces_and_hides_pi_private_roots(self):
        import grp

        with tempfile.TemporaryDirectory(prefix='agent-shell-manifest-') as temporary:
            base = Path(temporary)
            base.chmod(0o711)
            root = base / 'session'
            root.mkdir()
            share_gid = grp.getgrnam('ptobench-share').gr_gid
            os.chown(root, os.getuid(), share_gid)
            root.chmod(0o2710)
            workspaces = root / 'workspaces'
            workspaces.mkdir()
            os.chown(workspaces, os.getuid(), share_gid)
            workspaces.chmod(0o2750)
            for name in ('worker_000', 'worker_001'):
                worker = workspaces / name
                worker.mkdir()
                os.chown(worker, os.getuid(), share_gid)
                worker.chmod(0o2750)
                candidate = worker / 'candidate'
                candidate.mkdir()
                os.chown(candidate, os.getuid(), share_gid)
                candidate.chmod(0o2770)
                source = candidate / 'candidate.py'
                source.write_text('VALUE = 1\n')
                os.chown(source, os.getuid(), share_gid)
                source.chmod(0o660)
                for relative in ('.benchmark_context', '.benchmark_outputs'):
                    protected = worker / relative
                    protected.mkdir()
                    os.chown(protected, os.getuid(), share_gid)
                    protected.chmod(0o2750)
                    document = protected / 'README.md'
                    document.write_text('read only\n')
                    os.chown(document, os.getuid(), share_gid)
                    document.chmod(0o640)

            workspace = workspaces / 'worker_000'
            other_session = base / 'other-session'
            other_session.mkdir()
            (other_session / 'private.txt').write_text('OTHER_SESSION_CANARY')
            endpoint_root = root / 'broker-endpoints'
            endpoint_root.mkdir()
            os.chown(endpoint_root, os.getuid(), share_gid)
            endpoint_root.chmod(0o2710)
            rpc, api = endpoint_root / 'rpc.sock', endpoint_root / 'api.sock'
            transport_directory, transport = sandbox_supervisor._copy_cli_transport(root)
            server = socket.socket(socket.AF_UNIX)
            server.bind(str(rpc))
            server.listen(1)
            self.addCleanup(server.close)
            gateway = socket.socket(socket.AF_UNIX)
            gateway.bind(str(api))
            gateway.listen(1)
            self.addCleanup(gateway.close)
            received = {}

            def respond():
                connection, _ = server.accept()
                with connection:
                    received.update(json.loads(connection.makefile('rb').readline()))
                    connection.sendall(b'{"output":"broker transport ok\\n"}\n{"exit_code":0}\n')

            response_thread = threading.Thread(target=respond, daemon=True)
            response_thread.start()
            inner = f'''import os,pathlib,subprocess
candidate=pathlib.Path("candidate/candidate.py")
assert candidate.read_text()=="VALUE = 1\\n"
candidate.write_text("VALUE = 2\\n")
assert pathlib.Path(".benchmark_context/README.md").read_text()=="read only\\n"
assert pathlib.Path(".benchmark_outputs/README.md").read_text()=="read only\\n"
for root,child in (("/ptobench/pi-config","settings.json"),
                   ("/ptobench/pi-tools","ptobench-shell"),
                   ("/ptobench/agent-sessions","session.jsonl")):
 private=pathlib.Path(root)
 assert private.is_dir() and list(private.iterdir())==[], root
 assert not (private/child).exists(), str(private/child)
assert not pathlib.Path("/ptobench/pi-tools/cannbot-extension.ts").exists()
alias=pathlib.Path("candidate/private-config-link")
alias.symlink_to("/ptobench/pi-config", target_is_directory=True)
assert not (alias/"settings.json").exists()
for root in ("/proc/self/root", "/proc/1/root"):
 for relative in ("ptobench/pi-config/settings.json", "ptobench/pi-tools/cannbot-extension.ts",
                  "ptobench/agent-sessions/session.jsonl"):
  assert not (pathlib.Path(root)/relative).exists(), root+"/"+relative
try: os.link("/ptobench/pi-config/settings.json", "candidate/private-hardlink")
except OSError: pass
else: raise AssertionError("Shell created a private config hard link")
assert not pathlib.Path({str(other_session)!r}).exists()
result=subprocess.run(["ptobench","--help"],capture_output=True,text=True)
assert result.returncode==0 and result.stdout=="broker transport ok\\n", (result.stdout,result.stderr)
'''
            (workspace / 'candidate/sandbox_probe.py').write_text(inner)
            shell_command = 'python3 candidate/sandbox_probe.py'
            outer = ("import subprocess\n" + f"inner = {shell_command!r}\n"
                     + "result = subprocess.run(['/ptobench/pi-tools/ptobench-shell', '-c', inner], "
                       "capture_output=True, text=True)\n"
                     + "assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)\n")
            argv, launch_environment = isolation.agent_command(
                [TEST_PYTHON, '-c', outer], cwd=workspace,
                environment={'PATH': os.environ['PATH'], 'PTOBENCH_RPC_SOCKET': str(rpc),
                             'PTOBENCH_API_PROXY_SOCKET': str(api),
                             'PTOBENCH_CLIENT_PATH': str(transport)},
                workspace=workspace, session_root=root)
            result = agent_launch.run_process(argv, cwd=workspace, environment=launch_environment)
            response_thread.join(5)
            self.assertEqual(result, 0)
            self.assertEqual(received['argv'], ['--help'])
            self.assertEqual(received['cwd'], str(workspace))
            self.assertEqual((workspace / 'candidate/candidate.py').read_text(), 'VALUE = 2\n')

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
                                    env=isolation.sandbox_launcher_environment(env),
                                    capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(validation.validated,{'fixture_000'})
            self.assertFalse(validation.failed)
            bad = program.replace('torch.tensor([1.,2.])','torch.tensor([9.,9.])')
            result = subprocess.run(view.argv([TEST_PYTHON,'-c',bad],cwd='/tmp',environment=env),
                                    env=isolation.sandbox_launcher_environment(env),
                                    capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertTrue(validation.failed)


class TmpfsSnapshotBoundaryTests(SessionTestBase):
    def setUp(self):
        super().setUp()
        self.handle = self.create()

    def test_tmpfs_files_stay_out_of_candidate_freeze_and_snapshot(self):
        import session
        import snapshots
        import workspace as workspace_tree

        workspace = self.handle.initial_workspace
        original_manifest = workspace_tree.source_manifest(workspace)
        token = os.urandom(8).hex()
        host_tmp = Path('/tmp') / f'ptobench-snapshot-host-{token}'
        host_tmp.write_text('host tmp remains unchanged\n')
        self.addCleanup(host_tmp.unlink, missing_ok=True)
        sandbox_file_content = f'ISOLATED_TMP_CANARY_{token}'
        probe = f'''from pathlib import Path
path = Path("/tmp/ptobench-snapshot-host-{token}")
assert not path.exists()
path.write_text({sandbox_file_content!r})
assert path.read_text() == {sandbox_file_content!r}
'''
        view = isolation.View().system().add(workspace, '/workspace', write=True)
        environment = isolation.clean_environment()
        result = subprocess.run(
            view.argv([TEST_PYTHON, '-c', probe], cwd='/workspace', environment=environment),
            env=isolation.sandbox_launcher_environment(environment), capture_output=True,
            text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(host_tmp.read_text(), 'host tmp remains unchanged\n')
        self.assertEqual(workspace_tree.source_manifest(workspace), original_manifest)

        frozen = self.handle.session_root / 'staging/tmpfs-snapshot-probe/workspace'
        frozen.mkdir(parents=True)
        frozen_manifest = workspace_tree.copy_candidate_tree(workspace, frozen / 'candidate')
        self.assertEqual(frozen_manifest, original_manifest)
        event = {'seq': 1, 'workspace': str(workspace),
                 'content_sha256': frozen_manifest['content_sha256'], 'exit_code': 0,
                 'correctness': 'PASS', 'integrity_passed': True,
                 'observation': {'correctness': 'PASS'}}
        with session.execution_lock():
            snapshot_id = snapshots.create_snapshot(self.handle.session_root, frozen, event)

        snapshot_source = self.handle.snapshots_root / snapshot_id / 'source'
        self.assertEqual(workspace_tree.source_manifest(snapshot_source), original_manifest)
        self.assertFalse(any(path.name == host_tmp.name for path in snapshot_source.rglob('*')))
        snapshot_bytes = b''.join(path.read_bytes() for path in snapshot_source.rglob('*') if path.is_file())
        self.assertNotIn(sandbox_file_content.encode(), snapshot_bytes)


class IsolatedLauncherTests(unittest.TestCase):
    def test_generic_participant_entrypoint_refuses_unsandboxed_codex(self):
        from launcher import methods
        with tempfile.TemporaryDirectory(prefix='participant-unsupported-backend-') as temporary:
            root = Path(temporary)
            request_path = root / 'request.json'
            request_path.write_text(json.dumps({'preflight': {'profile': {'backend': 'codex'}}}))
            process = SimpleNamespace(pid=os.getpid())
            with patch.dict(os.environ, {'PTOBENCH_SANDBOX': '0'}), \
                 patch.object(methods.subprocess, 'Popen', return_value=process) as spawn:
                with self.assertRaisesRegex(RuntimeError, 'only Pi Agent is supported'):
                    methods.start({'path': 'unused.py'}, str(request_path), str(root), {})
                spawn.assert_not_called()

    def test_generic_participant_entrypoint_refuses_missing_bubblewrap(self):
        from launcher import methods
        with tempfile.TemporaryDirectory(prefix='participant-missing-bwrap-') as temporary:
            root = Path(temporary)
            request_path = root / 'request.json'
            request_path.write_text(json.dumps({'preflight': {'profile': {
                'backend': 'pi', 'mode_provider': {'base_url': 'https://example.invalid'}}}}))
            process = SimpleNamespace(pid=os.getpid())
            with patch.dict(os.environ, {'PTOBENCH_SANDBOX': '1'}), \
                 patch.object(isolation, 'executable', side_effect=RuntimeError(
                     'filesystem isolation requires bubblewrap')), \
                 patch.object(methods.subprocess, 'Popen', return_value=process) as spawn:
                with self.assertRaisesRegex(RuntimeError, 'requires bubblewrap'):
                    methods.start({'path': 'unused.py'}, str(request_path), str(root), {})
                spawn.assert_not_called()

    def test_basic_runner_refuses_unsandboxed_codex(self):
        from modes.basic import runner
        with tempfile.TemporaryDirectory(prefix='basic-unsupported-backend-') as temporary:
            root = Path(temporary) / 'session'
            workspace = root / 'workspaces/worker_000'
            workspace.mkdir(parents=True)
            (root / 'prompt.txt').write_text('prompt')
            request = {'context': {'session_root': str(root), 'workspace': str(workspace)},
                       'preflight': {'profile': {'backend': 'codex', 'argv': ['codex', '{prompt}'],
                                                 'env': {}}}}
            with patch.dict(os.environ, {'PTOBENCH_SANDBOX': '0'}), \
                 patch.object(runner, 'render', return_value=(['codex'], {})), \
                 patch.object(runner.os, 'execvpe') as execute:
                with self.assertRaisesRegex(RuntimeError, 'only Pi Agent is supported'):
                    runner.run(request)
                execute.assert_not_called()

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

    def test_cannbot_pi_children_profiles_and_fork(self):
        from test_cannbot import CANNBotTests, FAKE_NATIVE
        case = CANNBotTests('test_codex_mcp_fresh_sessions_same_budget_and_original_resources')
        case.setUp()
        try:
            case.cannbot_config(backend='pi', budget=5, fork_child=True)
            # Control belongs to the trusted test harness; embed only public
            # fake-Agent instructions, never mount its host dir.
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
