"""Trusted per-run supervisor and narrow CLI broker. Never mounted into an Agent."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import tempfile
import time

import common
sys.path.insert(0, str(common.BENCHMARK_ROOT))
import isolation
import session



def copy_tree(source, target):
    if not source.exists():
        return
    # Caller has stopped all untrusted writers before collecting native history.
    for base, dirs, files in os.walk(source, followlinks=False):
        for name in [*dirs, *files]:
            if (Path(base) / name).is_symlink():
                raise RuntimeError('link in exported Agent files')
    def copy_if_changed(src, dst):
        source_path, target_path = Path(src), Path(dst)
        if target_path.is_symlink():
            raise RuntimeError('link in exported Agent files')
        try:
            if (target_path.is_file()
                    and source_path.stat().st_size == target_path.stat().st_size):
                with source_path.open('rb') as left, target_path.open('rb') as right:
                    while True:
                        left_block, right_block = left.read(1024 * 1024), right.read(1024 * 1024)
                        if left_block != right_block:
                            break
                        if not left_block:
                            return str(target_path)
        except OSError:
            pass
        return shutil.copy2(source_path, target_path)

    shutil.copytree(source, target, dirs_exist_ok=True, copy_function=copy_if_changed)


class Broker:
    def __init__(self, root, private, *, cli_environment=None):
        self.root, self.private = Path(root), Path(private)
        session.recover_pending_forks(root)
        self.record = session.read_session(root)
        self.ipc = tempfile.TemporaryDirectory(prefix='ptobench-cli-')
        self.sockpath = Path(self.ipc.name) / 'rpc'
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(str(self.sockpath))
        isolation._agent_shared_socket(self.sockpath)
        self.server.listen(8)
        self.server.settimeout(.2)
        self.stop = threading.Event()
        self.child = None
        self.cli_environment = cli_environment

    def rewrite(self, text, name):
        prefix = re.escape(str(self.root))
        return re.sub(prefix + r'/(?:staging/action_\d+/workspace|final/workspace)/\.benchmark_outputs',
                      str(self.root / 'workspaces' / name / '.benchmark_outputs'), text)

    def serve(self):
        # CLI actions run in their own subprocess main thread so existing signal
        # handlers, reservation and process-group cleanup retain their semantics.
        while not self.stop.is_set():
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            with connection:
                try:
                    with connection.makefile('rb') as stream:
                        raw = stream.readline(65537)
                    if len(raw) > 65536:
                        raise ValueError('oversized request')
                    request = json.loads(raw)
                    allowed = {'argv', 'cwd', 'PTOBENCH_AGENT_DISPATCH_ID',
                               'PTOBENCH_AGENT_PARENT_TOOL_CALL_ID'}
                    if not {'argv', 'cwd'} <= set(request) or set(request) - allowed:
                        raise ValueError('invalid request fields')
                    argv = request['argv']
                    if (not isinstance(argv, list) or len(argv) > 8
                            or not all(isinstance(x, str) and len(x) < 4096 for x in argv)):
                        raise ValueError('invalid CLI arguments')
                    if not argv or argv[0] not in ('evaluate', 'profile', 'fork', '--help', '-h'):
                        raise ValueError('unsupported CLI operation')
                    name, canonical, _ = session.resolve_workspace(self.root, request['cwd'])
                    # The ordinary CLI still validates any explicit target. Sync
                    # that target too, rather than assuming it equals cwd.
                    target_name = name
                    position = 2 if argv[0] == 'profile' else 1
                    if len(argv) > position and not argv[position].startswith('-'):
                        explicit = Path(argv[position])
                        explicit = explicit if explicit.is_absolute() else canonical / explicit
                        target_name, _, _ = session.resolve_workspace(self.root, str(explicit))
                    env = {**os.environ, 'PTOBENCH_SESSION_ROOT': str(self.root),
                           'PTOBENCH_SANDBOX': '1'}
                    env.pop('PTOBENCH_RPC_SOCKET', None)
                    child_environment = dict(self.cli_environment or env)
                    for name in ('PTOBENCH_AGENT_DISPATCH_ID',
                                 'PTOBENCH_AGENT_PARENT_TOOL_CALL_ID'):
                        value = request.get(name)
                        if value is not None and (not isinstance(value, str)
                                                  or not value or len(value) > 256
                                                  or any(ord(char) < 32 for char in value)):
                            raise ValueError('invalid Agent dispatch identity')
                        child_environment.pop(name, None)
                        if value is not None:
                            child_environment[name] = value
                    self.child = subprocess.Popen([sys.executable, str(common.BENCHMARK_ROOT / 'ptobench'), *argv],
                        cwd=canonical, env=child_environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
                    for line in iter(self.child.stdout.readline, b''):
                        text = self.rewrite(line.decode(errors='replace'), target_name)
                        try:
                            connection.sendall((json.dumps({'output': text}) + '\n').encode())
                        except BrokenPipeError:
                            # Continue draining; the durable action still needs
                            # to complete or be stopped by the Launcher.
                            pass
                    code = self.child.wait()
                    self.child = None
                    connection.sendall((json.dumps({'exit_code': code}) + '\n').encode())
                except Exception as error:
                    try:
                        connection.sendall((json.dumps({'output': json.dumps({
                            'status': 'REJECTED', 'error_code': 'invalid_request',
                            'message': str(error)}) + '\n', 'exit_code': 2}) + '\n').encode())
                    except OSError:
                        pass



def wait_for_socket(process, path, name):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(f'{name} did not start')
        try:
            with socket.socket(socket.AF_UNIX) as probe:
                probe.connect(str(path))
            return
        except OSError:
            time.sleep(.02)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError(f'{name} did not become ready')


def _copy_cli_transport(session_root):
    """Copy the untrusted-side transport shim into a Broker-owned temp path."""
    import agent_launch
    import grp

    request_directory = agent_launch.ensure_request_directories(Path(session_root))
    share_gid = grp.getgrnam(agent_launch.SHARE_GROUP).gr_gid
    directory = Path(tempfile.mkdtemp(prefix='ptobench-client-', dir=request_directory))
    try:
        isolation._agent_shared_directory(directory, 0o2710, share_gid)
        client = directory / 'sandbox_client.py'
        shutil.copyfile(common.BENCHMARK_ROOT / 'evaluator/sandbox_client.py', client)
        os.chown(client, os.getuid(), share_gid)
        client.chmod(0o550)
        return directory, client
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def run_private_pi(method_path, request_path, request):
    """Keep the Broker on the host; isolate only the interactive Pi Agent."""
    from modes._shared import provider

    profile = request['preflight']['profile']
    api_key = provider.api_key(profile)
    root = Path(request['context']['session_root'])
    private = common.BENCHMARK_ROOT / '.ptobench/isolation' / root.name
    private_created = False
    broker = None
    transport_directory = None
    proxy_directory = None
    gateway = None
    thread = None
    process = None
    try:
        private.mkdir(parents=True, exist_ok=False, mode=0o700)
        private_created = True
        broker = Broker(root, private)
        transport_directory, client = _copy_cli_transport(root)

        proxy_directory = tempfile.TemporaryDirectory(prefix='ptobench-model-gateway-')
        api_socket = Path(proxy_directory.name) / 'api.sock'
        gateway_environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                               'LANG': 'C.UTF-8',
                               'PTOBENCH_UPSTREAM_API_KEY': api_key}
        gateway = subprocess.Popen([
            sys.executable, str(common.BENCHMARK_ROOT / 'evaluator/network_proxy.py'),
            'serve-api', '--socket', str(api_socket),
            '--base-url', profile['mode_provider']['base_url'],
            '--api', profile['mode_provider']['api'],
        ], cwd=private, env=gateway_environment, stdin=subprocess.DEVNULL,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        del api_key

        bwrap = isolation.executable()
        cli_environment = isolation.clean_environment(os.environ)
        for name in ('PTOBENCH_RPC_SOCKET', 'PTOBENCH_CLIENT_PATH',
                     'PTOBENCH_API_PROXY_SOCKET',
                     'PTOBENCH_PROVIDER_PREPARED', 'PTOBENCH_AGENT_BRIDGE_DIR'):
            cli_environment.pop(name, None)
        cli_environment.update(PTOBENCH_SESSION_ROOT=str(root), PTOBENCH_SANDBOX='1',
                                PTOBENCH_BWRAP=bwrap)
        broker.cli_environment = cli_environment

        runner_environment = isolation.clean_environment(os.environ)
        runner_environment.update(
            PTOBENCH_SESSION_ROOT=str(root),
            PTOBENCH_SANDBOX='1',
            PTOBENCH_BWRAP=bwrap,
            PTOBENCH_PROVIDER_PREPARED='1',
            PTOBENCH_RPC_SOCKET=str(broker.sockpath),
            PTOBENCH_CLIENT_PATH=str(client),
            PTOBENCH_API_PROXY_SOCKET=str(api_socket),
            PYPTO_PROVIDER_API_KEY=isolation.LOCAL_MODEL_KEY,
        )

        thread = threading.Thread(target=broker.serve, daemon=True)
        wait_for_socket(gateway, api_socket, 'model API gateway')
        thread.start()
        process = subprocess.Popen([sys.executable, method_path, 'run', str(request_path)],
            cwd=request['context']['workspace'], env=runner_environment,
            stdin=subprocess.DEVNULL)
        return process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if broker is not None:
            broker.stop.set()
        if broker is not None and broker.child is not None and broker.child.poll() is None:
            broker.child.terminate()
            try:
                broker.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                broker.child.kill()
                broker.child.wait()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        if broker is not None:
            broker.server.close()
            broker.ipc.cleanup()
        if transport_directory is not None:
            shutil.rmtree(transport_directory, ignore_errors=True)
        if gateway is not None and gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait()
        if proxy_directory is not None:
            proxy_directory.cleanup()
        shutil.rmtree(root / 'agent/.runtime', ignore_errors=True)
        if private_created:
            shutil.rmtree(private, ignore_errors=True)


def run(method_path, request_path):
    request = json.loads(Path(request_path).read_text())
    profile = request.get('preflight', {}).get('profile', {})
    isolation.require_agent_isolation(profile)
    return run_private_pi(method_path, request_path, request)


if __name__ == '__main__':
    raise SystemExit(run(*sys.argv[1:]))
