"""Trusted per-run supervisor and narrow CLI broker. Never mounted into an Agent."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import tempfile
import time
from urllib.parse import urlsplit

import common
sys.path.insert(0, str(common.BENCHMARK_ROOT))
import file_tree
import isolation
import session
import workspace


def model_api_endpoint(profile):
    """Resolve the one external host the selected Agent needs for model calls."""
    backend = profile['backend']
    base_url = (profile.get('mode_provider') or {}).get('base_url')
    if backend == 'codex':
        from modes._shared import recording
        settings = recording.host_settings(profile)
        selected = recording.option(profile['argv'], '--profile') or recording.option(profile['argv'], '-p')
        if selected:
            settings = {**settings, **settings.get('profiles', {}).get(selected, {})}
        provider_name = settings.get('model_provider', 'openai')
        for index, argument in enumerate(profile['argv'][:-1]):
            if argument in ('-c', '--config') and profile['argv'][index + 1].startswith('model_provider='):
                provider_name = profile['argv'][index + 1].split('=', 1)[1].strip().strip('"\'')
        base_url = settings.get('model_providers', {}).get(provider_name, {}).get('base_url')
        if not base_url and provider_name == 'openai':
            base_url = 'https://api.openai.com/v1'
    elif backend == 'pi' and not base_url:
        from modes._shared import recording
        provider_name = recording.option(profile['argv'], '--provider')
        selected_model = profile.get('model', '')
        if not provider_name and '/' in selected_model:
            provider_name = selected_model.split('/', 1)[0]
        config_dir = recording.config_directory(profile)
        try:
            config = json.loads((config_dir / 'models.json').read_text())
            providers = config.get('providers', {})
            if not provider_name:
                provider_name = json.loads((config_dir / 'settings.json').read_text()).get('defaultProvider')
            base_url = providers.get(provider_name, {}).get('baseUrl')
        except (OSError, ValueError):
            pass
        if (not base_url and '{repo_root}/benchmarks/mig_workflow/agent_provider/launch.py'
                in profile['argv']):
            config = json.loads((common.REPO_ROOT / 'benchmarks/mig_workflow/agent_provider/pi/models.json').read_text())
            provider_name = provider_name or 'provider-default'
            base_url = config['providers'][provider_name]['baseUrl']
    elif backend == 'opencode' and not base_url:
        config_path = profile.get('env', {}).get('OPENCODE_CONFIG')
        if not config_path:
            config_path = common.REPO_ROOT / 'benchmarks/mig_workflow/agent_provider/opencode.json'
        try:
            config = json.loads(Path(config_path).read_text())
            base_url = next(iter(config['provider'].values()))['options']['baseURL']
        except (OSError, ValueError, KeyError, StopIteration):
            pass
    elif backend == 'claudecode' and not base_url:
        base_url = profile.get('env', {}).get('ANTHROPIC_BASE_URL') or 'https://api.anthropic.com'

    try:
        parsed = urlsplit(base_url or '')
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username or parsed.password):
            raise ValueError()
        return parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)
    except ValueError as error:
        raise RuntimeError('cannot resolve the configured Agent model API endpoint') from error


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
        self.view = private / 'view'
        self.public = private / 'public'
        self.public.mkdir()
        self.exports = self.view / 'workspaces'
        self.exports.mkdir(parents=True)
        self.candidates = self.view / 'candidates'
        self.candidates.mkdir()
        self.record = session.read_session(root)
        self.ipc = tempfile.TemporaryDirectory(prefix='ptobench-cli-')
        self.sockpath = Path(self.ipc.name) / 'rpc'
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(str(self.sockpath))
        self.server.listen(8)
        self.server.settimeout(.2)
        self.stop = threading.Event()
        self.child = None
        self.cli_environment = cli_environment
        for name in self.record['workspaces']:
            self.export(name)

    def export(self, name):
        original = self.root / 'workspaces' / name
        target = self.exports / name
        if target.is_symlink():
            raise ValueError(f'workspace export is not a real directory: {name}')
        new_workspace = not target.exists()
        if new_workspace:
            workspace.copy_candidate_tree(original, self.candidates / name)
            target.mkdir()
            public_workspace = self.public / name
            copy_tree(original / '.benchmark_context', public_workspace)
            (public_workspace / '.benchmark_outputs').mkdir()
        elif (not target.is_dir()
              or (self.candidates / name).is_symlink()
              or not (self.candidates / name).is_dir()):
            raise ValueError(f'workspace export is not a real directory: {name}')
        for relative, expected in (
                ('candidate', f'/ptobench/candidates/{name}'),
                ('.benchmark_context', f'/ptobench/public/{name}'),
                ('.benchmark_outputs', f'/ptobench/public/{name}/.benchmark_outputs')):
            link = target / relative
            if link.is_symlink() and os.readlink(link) == expected:
                continue
            if link.exists() or link.is_symlink() or not new_workspace:
                raise ValueError(f'protected workspace link changed: {relative}')
            link.symlink_to(expected)
        return target

    def sync_source(self, name):
        source = self.exports / name
        target = self.root / 'workspaces' / name
        temporary = self.private / 'incoming'
        if temporary.exists():
            shutil.rmtree(temporary)
        # The exported contract is a link to a readonly public mount. It is not
        # part of the submitted source. All included entries must still be plain
        # files/directories, and the ordinary no-follow copier opens each file.
        protected = {
            'candidate': f'/ptobench/candidates/{name}',
            '.benchmark_context': f'/ptobench/public/{name}',
            '.benchmark_outputs': f'/ptobench/public/{name}/.benchmark_outputs',
        }
        for relative, expected in protected.items():
            link = source / relative
            if not link.is_symlink() or os.readlink(link) != expected:
                raise ValueError(f'protected workspace link changed: {relative}')
        candidate_source = self.candidates / name
        entries = workspace._candidate_payload_entries(candidate_source)
        if any(e['kind'] not in ('file', 'directory') for e in entries.values()):
            raise ValueError('non-regular Candidate source')
        temporary.mkdir()
        workspace._copy_source_tree(candidate_source, temporary, entries)
        after = workspace._candidate_payload_entries(candidate_source)
        if entries != after:
            raise ValueError('Candidate source changed while submitting')
        with file_tree.open_directory(target) as fd:
            for name in os.listdir(fd):
                if name not in workspace.CANDIDATE_EXCLUDES:
                    file_tree.remove_at(fd, name)
        candidate_target = target / 'candidate'
        with file_tree.open_directory(target, create=True):
            pass
        if candidate_target.exists():
            with file_tree.open_directory(candidate_target) as fd:
                for name in os.listdir(fd):
                    file_tree.remove_at(fd, name)
        else:
            candidate_target.mkdir()
        for path in temporary.iterdir():
            path.rename(candidate_target / path.name)
        temporary.rmdir()

    def feedback(self, name):
        original = self.root / 'workspaces' / name / '.benchmark_outputs'
        target = self.public / name / '.benchmark_outputs'
        # Keep the read-only mount target stable while replacing its contents.
        with file_tree.open_directory(target) as fd:
            for child in os.listdir(fd):
                file_tree.remove_at(fd, child)
        copy_tree(original, target)
        if target.exists():
            for p in target.rglob('*.json'):
                text = p.read_text()
                p.write_text(self.rewrite(text, name))
        self.export(name)

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
                    self.sync_source(target_name)
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
                    current = session.read_session(self.root)
                    for worker in current['workspaces']:
                        self.export(worker)
                    self.feedback(target_name)
                    connection.sendall((json.dumps({'exit_code': code}) + '\n').encode())
                except Exception as error:
                    try:
                        connection.sendall((json.dumps({'output': json.dumps({
                            'status': 'REJECTED', 'error_code': 'invalid_request',
                            'message': str(error)}) + '\n', 'exit_code': 2}) + '\n').encode())
                    except OSError:
                        pass


def run_legacy(method_path, request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    root = Path(request['context']['session_root'])
    private = common.BENCHMARK_ROOT / '.ptobench/isolation' / root.name
    private.mkdir(parents=True, exist_ok=False, mode=0o700)
    broker = Broker(root, private)
    runtime = isolation.copy_runtime(private / 'runtime')
    view = isolation.View(isolate_network=True).system().native_agent()
    isolation.runtime_mounts(view, runtime)
    # Only this run's public invocation records are visible, independently of
    # canonical protected records and the host method configuration.
    for name in ('method_request.json', 'prompt.txt', 'profile.json', 'mode.json'):
        source = root / name
        if source.exists():
            target = broker.view / name
            shutil.copyfile(source, target)
            view.add(target, source)
    for name in ('agent', 'method'):
        copy_tree(root / name, broker.view / name)
        (broker.view / name).mkdir(exist_ok=True)
    view.add(broker.view / 'method', root / 'method')
    for relative in ('agent', 'method/work', 'method/dispatches', 'method/tools', 'method/agent'):
        target = broker.view / relative
        target.mkdir(parents=True, exist_ok=True)
        view.add(target, root / relative, write=True)
    view.add(broker.exports, root / 'workspaces')
    view.add(broker.candidates, '/ptobench/candidates', write=True)
    view.add(broker.public, '/ptobench/public')
    view.add(broker.sockpath, '/ptobench/cli.sock')
    proxy_directory = tempfile.TemporaryDirectory(prefix='ptb-')
    proxy_socket = Path(proxy_directory.name) / 'proxy.sock'
    host, port = model_api_endpoint(request['preflight']['profile'])
    view.network_allowlist = {'host': host, 'port': port}
    view.add(proxy_socket.parent, '/ptobench/network')
    client = private / 'ptobench-client'
    shutil.copyfile(common.BENCHMARK_ROOT / 'evaluator/sandbox_client.py', client)
    client.chmod(0o755)
    view.add(client, common.BENCHMARK_ROOT / 'ptobench')
    env = isolation.clean_environment(agent=True)
    env.update(PTOBENCH_RPC_SOCKET='/ptobench/cli.sock', PTOBENCH_BWRAP=isolation.executable(),
               PI_CODING_AGENT_DIR=str(root / 'agent/pi'),
               CODEX_HOME=str(root / 'agent/codex'), PTOBENCH_PROVIDER_PREPARED='1',
               HTTP_PROXY='http://127.0.0.1:18080', HTTPS_PROXY='http://127.0.0.1:18080',
               http_proxy='http://127.0.0.1:18080', https_proxy='http://127.0.0.1:18080',
               NO_PROXY='localhost,127.0.0.1,::1', no_proxy='localhost,127.0.0.1,::1',
               NODE_OPTIONS='--use-env-proxy')
    from modes._shared import provider, recording
    profile = request['preflight']['profile']
    for argument in profile['argv']:
        p = Path(argument.replace('{repo_root}', str(common.REPO_ROOT)))
        if p.is_absolute() and p.is_file():
            view.add(p)
    env.update(provider.credentials(profile))
    if profile['backend'] == 'codex':
        old = recording.config_directory(profile)
        destination = broker.view / 'agent/codex'
        destination.mkdir(exist_ok=True)
        for filename in ('auth.json', 'config.toml'):
            if (old / filename).is_file():
                shutil.copyfile(old / filename, destination / filename)
    (root / 'filesystem_isolation.json').write_text(json.dumps(view.manifest(), indent=2) + '\n')
    proxy_environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'LANG': 'C.UTF-8'}
    proxy = subprocess.Popen([sys.executable, str(common.BENCHMARK_ROOT / 'evaluator/network_proxy.py'),
        'serve', '--socket', str(proxy_socket), '--host', host, '--port', str(port)],
        cwd=private, env=proxy_environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if proxy.poll() is not None:
            raise RuntimeError('model API proxy did not start')
        try:
            probe = socket.socket(socket.AF_UNIX)
            probe.connect(str(proxy_socket))
            probe.close()
            break
        except OSError:
            time.sleep(.02)
    else:
        proxy.terminate()
        proxy.wait(timeout=5)
        proxy_directory.cleanup()
        raise RuntimeError('model API proxy did not become ready')
    thread = threading.Thread(target=broker.serve, daemon=True)
    thread.start()
    bridge = common.BENCHMARK_ROOT / 'evaluator/network_proxy.py'
    argv = view.argv([sys.executable, str(bridge), 'agent', '--socket', '/ptobench/network/proxy.sock', '--',
                      sys.executable, method_path, 'run', request_path],
                     cwd=request['context']['workspace'], environment=env)
    process = None
    def stop(signum, frame):
        if process and process.poll() is None:
            process.terminate()
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
        code = process.wait()
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait()
        broker.stop.set()
        if broker.child and broker.child.poll() is None:
            broker.child.terminate()
        thread.join(timeout=5)
        broker.server.close()
        broker.ipc.cleanup()
        proxy.terminate()
        try:
            proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill()
            proxy.wait()
        proxy_directory.cleanup()
        for relative in ('agent/native', 'agent/codex/sessions', 'method/work', 'method/dispatches', 'method/tools', 'method/agent'):
            copy_tree(broker.view / relative, root / relative)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code


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


def run_private_pi(method_path, request_path, request):
    """Keep the Broker on the host; isolate only the interactive Pi Agent."""
    from modes._shared import provider

    profile = request['preflight']['profile']
    api_key = provider.api_key(profile)
    root = Path(request['context']['session_root'])
    private = common.BENCHMARK_ROOT / '.ptobench/isolation' / root.name
    private.mkdir(parents=True, exist_ok=False, mode=0o700)
    broker = Broker(root, private)
    client = private / 'ptobench-client'
    shutil.copyfile(common.BENCHMARK_ROOT / 'evaluator/sandbox_client.py', client)
    client.chmod(0o755)

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
    for name in ('PTOBENCH_RPC_SOCKET', 'PTOBENCH_PUBLIC_DIR', 'PTOBENCH_CLIENT_PATH',
                 'PTOBENCH_API_PROXY_SOCKET', 'PTOBENCH_AGENT_WORKSPACE_ROOT',
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
        PTOBENCH_PUBLIC_DIR=str(broker.public),
        PTOBENCH_CLIENT_PATH=str(client),
        PTOBENCH_API_PROXY_SOCKET=str(api_socket),
        PTOBENCH_AGENT_WORKSPACE_ROOT=str(broker.exports),
        PYPTO_PROVIDER_API_KEY=isolation.LOCAL_MODEL_KEY,
    )

    thread = threading.Thread(target=broker.serve, daemon=True)
    process = None
    try:
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
        broker.stop.set()
        if broker.child is not None and broker.child.poll() is None:
            broker.child.terminate()
            try:
                broker.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                broker.child.kill()
                broker.child.wait()
        if thread.is_alive():
            thread.join(timeout=5)
        broker.server.close()
        broker.ipc.cleanup()
        if gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait()
        proxy_directory.cleanup()
        shutil.rmtree(root / 'agent/.runtime', ignore_errors=True)
        shutil.rmtree(private, ignore_errors=True)


def run(method_path, request_path):
    request = json.loads(Path(request_path).read_text())
    profile = request.get('preflight', {}).get('profile', {})
    if isolation.enabled() and profile.get('backend') == 'pi':
        if not profile.get('mode_provider'):
            raise RuntimeError('isolated Pi requires the mode-local model API gateway configuration')
        return run_private_pi(method_path, request_path, request)
    return run_legacy(method_path, request_path)


if __name__ == '__main__':
    raise SystemExit(run(*sys.argv[1:]))
