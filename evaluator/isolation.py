"""OS-enforced, default-deny filesystem views shared by every mode and Case.

PTOBENCH_SANDBOX=0 is a trusted non-Pi harness escape hatch, never forwarded from
an Agent request. Pi Agent launch paths always require isolation.
"""
from __future__ import annotations

import hashlib
import ctypes
import grp
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

import common
import file_tree


POLICY_VERSION = 2
RUNTIME_MODE_DIRS = common.MODE_SOURCE_DIRS
LOCAL_MODEL_KEY = 'ptobench-local-model-gateway'
_EMPTY_RUNTIME = tempfile.TemporaryDirectory(prefix='ptobench-empty-')
Path(_EMPTY_RUNTIME.name).chmod(0o711)
(Path(_EMPTY_RUNTIME.name) / 'include').mkdir()
(Path(_EMPTY_RUNTIME.name) / 'pkg_inc').mkdir()
(Path(_EMPTY_RUNTIME.name) / 'passwd').write_text(
    f'ptobench:x:{os.getuid()}:{os.getgid()}:PTOBench:/home/ptobench:/bin/bash\n')
(Path(_EMPTY_RUNTIME.name) / 'group').write_text(f'ptobench:x:{os.getgid()}:\n')


def enabled():
    return os.environ.get('PTOBENCH_SANDBOX', '1') != '0'


def require_agent_isolation(profile):
    if not isinstance(profile, dict) or profile.get('backend') != 'pi':
        raise RuntimeError('only Pi Agent is supported by the isolated launcher')
    if not enabled():
        raise RuntimeError('Pi Agent requires filesystem isolation')
    if not profile.get('mode_provider'):
        raise RuntimeError('isolated Pi requires the mode-local model API gateway configuration')


def executable():
    import agent_launch
    value = agent_launch._system_binary('bwrap')
    if value:
        return value
    raise RuntimeError('filesystem isolation requires bubblewrap; refusing unsandboxed execution')


def clean_environment(environment=None, *, agent=False):
    original = os.environ if environment is None else environment
    allowed = {'PATH', 'LANG', 'LC_ALL', 'TZ', 'LD_LIBRARY_PATH', 'PYTHONPATH',
               'PTOAS_ROOT', 'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_AICPU_PATH',
               'ASCEND_TOOLKIT_HOME', 'ASCEND_RUNTIME_PATH', 'CANN_ROOT',
               'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'no_proxy'}
    prefixes = ('PTOBENCH_', 'PYPTO_', 'ASCEND_', 'HCCL_', 'TASK_QUEUE_')
    result = {k: v for k, v in original.items() if k in allowed or k.startswith(prefixes)}
    # Never let an inherited interpreter startup hook run outside the policy.
    for key in ('PTOBENCH_BWRAP', 'PTOBENCH_SANDBOX', 'PTOBENCH_WORKER',
                'PTOBENCH_ACTION_START_FD', 'PYPTO_PROVIDER_API_KEY'):
        result.pop(key, None)
    if agent:
        for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'no_proxy'):
            result.pop(key, None)
    result.update(HOME='/home/ptobench', TMPDIR='/tmp', PYTHONDONTWRITEBYTECODE='1',
                  PYTHONNOUSERSITE='1', PTOBENCH_SANDBOX='1')
    return result


def agent_environment(environment):
    """Pass only runtime settings the interactive Agent process needs."""
    names = {'PATH', 'LANG', 'LC_ALL', 'TZ', 'TERM', 'LD_LIBRARY_PATH',
             'PI_CODING_AGENT_DIR', 'CODEX_HOME', 'PTOBENCH_RPC_SOCKET',
             'PTOBENCH_CANNBOT_SOCKET', 'PTOBENCH_CANNBOT_ALLOW_TASK',
             'PTOBENCH_AGENT_DISPATCH_ID', 'PTOBENCH_AGENT_PARENT_TOOL_CALL_ID',
             'PI_OFFLINE', 'NO_PROXY', 'no_proxy'}
    result = {name: environment[name] for name in names if name in environment}
    result.update(
        HOME='/home/ptobench', TMPDIR='/tmp', PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1', PTOBENCH_SANDBOX='1',
        PYPTO_PROVIDER_API_KEY=LOCAL_MODEL_KEY,
    )
    return result


def sandbox_launcher_environment(environment):
    """Keep host credentials out of bubblewrap's PID namespace init process."""
    return {key: environment[key] for key in ('PATH',) if key in environment}


def _agent_shared_directory(path, mode=0o2770, group_id=None):
    path = Path(path)
    metadata = path.stat()
    expected_gid = os.getgid() if group_id is None else group_id
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f'Agent shared directory must be owned by the Broker UID/GID: {path}')
    if metadata.st_gid != expected_gid:
        os.chown(path, -1, expected_gid)
    path.chmod(mode)
    return path


def _agent_shared_socket(path):
    path = Path(path)
    metadata = path.lstat()
    if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise RuntimeError(f'Agent Broker endpoint must be a Broker-owned Unix socket: {path}')
    import grp
    share_gid = grp.getgrnam('ptobench-share').gr_gid
    if metadata.st_gid != share_gid:
        os.chown(path, -1, share_gid)
    path.chmod(0o660)
    _agent_shared_directory(path.parent, 0o710, group_id=share_gid)
    return path


def _sandbox_mount_path(view, host_path):
    host_path = Path(host_path).resolve()
    mounts = sorted(view.mounts, key=lambda item: len(Path(item[0]).parts), reverse=True)
    for source, destination, access in mounts:
        source_path = Path(source).resolve()
        try:
            relative = host_path.relative_to(source_path)
        except ValueError:
            continue
        if access != 'rw':
            raise RuntimeError('Candidate must be inside a writable Agent mount')
        return str(Path(destination) / relative)
    raise RuntimeError('Candidate is not covered by a writable Agent mount')


def _candidate_location(view, workspace, candidate_pool):
    workspace = Path(workspace)
    entry = workspace / 'candidate'
    if entry.is_symlink():
        target = os.readlink(entry)
        virtual_pool = Path('/ptobench/candidates')
        target_path = Path(target)
        if target_path.is_absolute() and target_path.is_relative_to(virtual_pool):
            host_path = Path(candidate_pool) / target_path.relative_to(virtual_pool)
            guest_path = str(target_path)
        else:
            host_path = entry.resolve()
            guest_path = _sandbox_mount_path(view, host_path)
    elif entry.is_dir():
        host_path = entry
        guest_path = _sandbox_mount_path(view, host_path)
    elif (workspace / 'candidate.py').is_file():
        host_path = workspace
        guest_path = _sandbox_mount_path(view, host_path)
    else:
        raise RuntimeError(f'Agent Candidate directory is unavailable: {entry}')
    if not host_path.is_dir():
        raise RuntimeError(f'Agent Candidate directory is unavailable: {host_path}')
    resolved = host_path.resolve()
    if not resolved.is_relative_to(Path(candidate_pool).resolve()) and not entry.is_dir():
        raise RuntimeError('Agent Candidate symlink leaves the mounted Candidate pool')
    return resolved, guest_path


def _exported_workspace(workspace, workspace_root, exported_workspaces):
    """Map a Session workspace path to the Broker's exported workspace tree."""
    try:
        relative = Path(workspace).resolve().relative_to(Path(workspace_root).resolve())
    except ValueError as error:
        raise RuntimeError('Agent workspace is outside registered Session workspaces') from error
    if not relative.parts:
        raise RuntimeError('Agent workspace must be a registered Session workspace')
    return Path(exported_workspaces) / relative


def install_pi_private_tools(directory):
    """Install trusted Pi guards in its private configuration directory."""
    import grp
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o2770)
    _agent_shared_directory(directory, group_id=grp.getgrnam('ptobench-share').gr_gid)
    for source_name, target_name in (
            ('pi_file_guard.mjs', 'ptobench-file-guard.mjs'),
            ('pi_shell_wrapper.py', 'ptobench-shell')):
        target = directory / target_name
        temporary = directory / f'.{target_name}.{uuid.uuid4().hex}'
        try:
            shutil.copyfile(Path(__file__).with_name(source_name), temporary)
            temporary.chmod(0o550 if source_name == 'pi_shell_wrapper.py' else 0o440)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    shell_config = directory / 'shell.json'
    file_tree.atomic_bytes(shell_config,
        (json.dumps({'bwrap': executable()}) + '\n').encode(), mode=0o440)
    settings_path = directory / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    settings['shellPath'] = '/ptobench/pi-private/ptobench-shell'
    file_tree.atomic_bytes(settings_path,
        (json.dumps(settings, indent=2) + '\n').encode(), mode=0o660)


def agent_command(command, *, cwd, environment, workspace, session_root,
                  cann_bot_extension=None, cann_bot_socket=None):
    """Build the verified host launcher and Bubblewrap command for an Agent."""
    if not enabled():
        raise RuntimeError('Pi Agent requires filesystem isolation')
    view = View(isolate_network=True).system().native_agent()
    workspace_root = Path(session_root) / 'workspaces'
    exported_workspaces = Path(environment.get('PTOBENCH_AGENT_WORKSPACE_ROOT', workspace_root))
    candidate_pool = exported_workspaces.parent / 'candidates'
    if candidate_pool.is_symlink() or not candidate_pool.is_dir():
        raise RuntimeError('Agent Candidate pool is unavailable')
    # Forks appear under the mounted Candidate pool while the Agent is running.
    # Workspace roots remain read-only, so their Broker-owned links cannot move.
    view.add(exported_workspaces, workspace_root)
    view.add(candidate_pool, '/ptobench/candidates', write=True)
    agent_workspace = _exported_workspace(workspace, workspace_root, exported_workspaces)
    _candidate_location(view, agent_workspace, candidate_pool)
    private_pi = Path(session_root) / 'agent/pi'
    import agent_launch
    agent_launch.ensure_request_directories(Path(session_root))
    install_pi_private_tools(private_pi)
    view.add(private_pi, '/ptobench/pi-private', write=True)
    public = environment.get('PTOBENCH_PUBLIC_DIR')
    client = environment.get('PTOBENCH_CLIENT_PATH')
    api_socket = environment.get('PTOBENCH_API_PROXY_SOCKET')
    if not public or not client or not api_socket:
        raise RuntimeError('Agent Broker or model API gateway is unavailable')
    view.add(public, '/ptobench/public')
    view.add(client, common.BENCHMARK_ROOT / 'ptobench')
    _agent_shared_socket(api_socket)
    view.add(Path(api_socket).parent, '/ptobench/network')
    socket_path = environment.get('PTOBENCH_RPC_SOCKET')
    if not socket_path:
        raise RuntimeError('PTOBench Agent Broker is unavailable')
    _agent_shared_socket(socket_path)
    view.add(socket_path, '/ptobench/cli.sock')

    # The bridge runs inside the Agent's private network namespace so its
    # loopback listener can reach only the mounted, path-limited API socket.
    # It deletes its temporary source before starting the Agent process.
    bridge_dir = Path(session_root) / 'agent/.runtime' / uuid.uuid4().hex
    bridge_dir.mkdir(parents=True, mode=0o2770)
    import grp
    _agent_shared_directory(bridge_dir, group_id=grp.getgrnam('ptobench-share').gr_gid)
    bridge = bridge_dir / 'network_proxy.py'
    shutil.copyfile(Path(__file__).with_name('network_proxy.py'), bridge)
    bridge.chmod(0o440)
    view.add(bridge_dir, '/ptobench/runtime', write=True)
    command = list(map(str, command))
    executable_path = Path(command[0])
    if executable_path.is_absolute() and executable_path.is_file():
        # system() and native_agent() already mount standard runtime roots.
        # Re-mounting a file inside one of those roots creates an invalid nested
        # target in bubblewrap for symlinked virtualenv paths.
        if not any(executable_path.is_relative_to(Path(destination))
                   for _, destination, _ in view.mounts):
            view.add(executable_path, '/ptobench/agent-tool')
            command[0] = '/ptobench/agent-tool'
    session_directory_mounted = False
    for index, argument in enumerate(command[:-1]):
        if argument == '--append-system-prompt':
            prompt_path = Path(command[index + 1])
            if prompt_path.is_absolute() and prompt_path.is_file():
                view.add(prompt_path)
        if argument == '--session-dir':
            session_directory = Path(command[index + 1])
            if not session_directory.is_absolute():
                session_directory = Path(cwd) / session_directory
            session_directory.mkdir(parents=True, exist_ok=True, mode=0o2770)
            _agent_shared_directory(session_directory, group_id=grp.getgrnam('ptobench-share').gr_gid)
            view.add(session_directory, '/ptobench/agent-sessions', write=True)
            command[index + 1] = '/ptobench/agent-sessions'
            session_directory_mounted = True
        elif argument.startswith('--session-dir='):
            session_directory = Path(argument.split('=', 1)[1])
            if not session_directory.is_absolute():
                session_directory = Path(cwd) / session_directory
            session_directory.mkdir(parents=True, exist_ok=True, mode=0o2770)
            _agent_shared_directory(session_directory, group_id=grp.getgrnam('ptobench-share').gr_gid)
            view.add(session_directory, '/ptobench/agent-sessions', write=True)
            command[index] = '--session-dir=/ptobench/agent-sessions'
            session_directory_mounted = True
    if not session_directory_mounted:
        native_sessions = Path(session_root) / 'agent/native'
        native_sessions.mkdir(parents=True, exist_ok=True, mode=0o2770)
        _agent_shared_directory(native_sessions, group_id=grp.getgrnam('ptobench-share').gr_gid)
        view.add(native_sessions, '/ptobench/agent-sessions', write=True)

    is_pi = (Path(command[0]).name == 'pi'
             or any('pi-coding-agent/dist/bundle/cli.js' in value for value in command[:3]))
    if is_pi:
        for index, value in enumerate(command):
            path_value = Path(value)
            if path_value.is_absolute():
                try:
                    relative = path_value.relative_to(private_pi)
                except ValueError:
                    continue
                command[index] = str(Path('/ptobench/pi-private') / relative)
        flags = ['--no-extensions', '--no-approve', '--extension',
                 '/ptobench/pi-private/ptobench-file-guard.mjs']
        insertion = command.index('--') if '--' in command else max(1, len(command) - 1)
        command[insertion:insertion] = flags
    if cann_bot_extension:
        view.add(cann_bot_extension, '/ptobench/cannbot-extension.ts')
        command = ['/ptobench/cannbot-extension.ts' if value == cann_bot_extension else value
                   for value in command]
    command = [sys.executable, '/ptobench/runtime/network_proxy.py', 'agent',
               '--api-socket', '/ptobench/network/api.sock', '--api-port', '18081',
               '--delete-runtime-files', '--', *command]
    if cann_bot_socket:
        _agent_shared_socket(cann_bot_socket)
        view.add(cann_bot_socket, '/ptobench/cannbot.sock')

    cleaned = agent_environment(environment)
    cleaned['PTOBENCH_RPC_SOCKET'] = '/ptobench/cli.sock'
    cleaned['PI_CODING_AGENT_DIR'] = '/ptobench/pi-private'
    cleaned['PTOBENCH_AGENT_SESSION_DIR'] = '/ptobench/agent-sessions'
    cleaned['PATH'] = str(common.BENCHMARK_ROOT) + os.pathsep + cleaned.get('PATH', '')
    if cann_bot_socket:
        cleaned['PTOBENCH_CANNBOT_SOCKET'] = '/ptobench/cannbot.sock'
    (Path(session_root) / 'filesystem_isolation.json').write_text(
        json.dumps(view.manifest(), indent=2) + '\n')
    bubblewrap = view.argv(command, cwd=cwd, environment=cleaned)
    return agent_launch.prepare(
        bubblewrap, cwd, Path(session_root) / 'agent/launcher')


def device_mapping(runtime):
    """Resolve host logical IDs to physical device nodes, without a context."""
    if getattr(runtime, 'platform', None) not in ('a2a3', 'a5'):
        return None
    distributed = getattr(runtime, 'distributed_config', None)
    configured = list(distributed.device_ids) if distributed else [runtime.device_id]
    library = ctypes.CDLL('libascendcl.so')
    physical = []
    for device in configured:
        value = ctypes.c_int32()
        code = library.aclrtGetPhyDevIdByLogicDevId(device, ctypes.byref(value))
        if code:
            raise RuntimeError(f'cannot resolve NPU {device} to a physical device (ACL {code})')
        physical.append(value.value)
    return {'configured': configured, 'physical': physical}


class View:
    def __init__(self, *, isolate_network=False):
        self.mounts = []
        self.devices = []
        self.links = []
        self.isolate_network = isolate_network
        self.network_allowlist = None

    def add(self, source, destination=None, *, write=False, device=False):
        source = Path(source).absolute()
        if not source.exists():
            raise RuntimeError(f'missing isolation dependency: {source}')
        destination = str(Path(destination or source).absolute())
        source = str(source.resolve())
        item = (source, destination, 'device' if device else 'rw' if write else 'ro')
        if item not in self.mounts:
            self.mounts.append(item)
        return self

    def system(self):
        # Explicit runtime roots, not /, /usr, /usr/local, /etc or host HOME.
        for path in ('/usr/bin', '/usr/lib', '/usr/lib64', '/usr/include', '/usr/share/cmake', '/bin', '/lib', '/lib64'):
            if Path(path).exists():
                self.add(path, path)
        for path in ('/etc/ld.so.cache', '/etc/resolv.conf', '/etc/hosts',
                     '/etc/nsswitch.conf', '/etc/gai.conf', '/etc/ssl/certs', '/etc/pki',
                     '/etc/ascend_install.info'):
            if Path(path).exists():
                self.add(path, path)
        for name in ('passwd', 'group'):
            self.add(Path(_EMPTY_RUNTIME.name) / name, '/etc/' + name)
        for path in {Path(sys.prefix), Path(sys.base_prefix)}:
            if not str(path).startswith('/usr'):
                self.add(path)
        # Wheels may cache an entire upstream checkout beside their runtime
        # libraries. Only its compiler headers are an authorized dependency.
        for checkout in Path(sys.prefix).glob('lib/python*/site-packages/simpler_setup/_assets/build/pto-isa'):
            self.add(_EMPTY_RUNTIME.name, checkout)
            for headers in ('include', 'pkg_inc'):
                if (checkout / headers).is_dir():
                    self.add(checkout / headers)
        for examples in Path(sys.prefix).glob('lib/python*/site-packages/simpler_setup/goldens'):
            self.add(_EMPTY_RUNTIME.name, examples)
        return self

    def native_agent(self):
        npm = Path.home() / '.local/npm'
        for relative in ('bin', 'lib/node_modules'):
            if (npm / relative).exists():
                self.add(npm / relative)
        node = shutil.which('node')
        if node:
            # Include both symlink spelling and its actual executable directory.
            self.add(Path(node).parent)
            self.add(Path(node).resolve().parent)
        return self

    def toolchain(self, profile, runtime, *, devices=None):
        roots = common.profile_values()
        # API/compiler dependencies are allowed; checkout docs/models/examples
        # are never dependencies and are not mounted.
        self.add(common.REPO_ROOT / 'golden')
        for key, children in (('ptoas_root', ('bin', 'lib', 'lib64', 'pyvenv.cfg')),
                              ('cann_root', ('bin', 'lib64', 'include', 'compiler',
                                             'runtime', 'opp', 'tools', 'pkg_inc', 'aarch64-linux', 'share/info')),
                              ('miniforge_root', ('lib',))):
            root = roots.get(key)
            if root:
                spelling = Path(root)
                actual = spelling.resolve()
                if spelling != actual:
                    self.links.append((str(actual), str(spelling)))
                for child in children:
                    p = actual / child
                    if p.exists():
                        if p.is_symlink() and p.is_dir():
                            self.add(p.resolve())
                            self.links.append((str(p.resolve()), str(p)))
                        else:
                            self.add(p)
        driver = Path('/usr/local/Ascend/driver')
        for child in ('lib64', 'include', 'version.info'):
            if (driver / child).exists():
                self.add(driver / child)
        if getattr(runtime, 'platform', profile.get('platform')) in ('a2a3', 'a5'):
            distributed = getattr(runtime, 'distributed_config', None)
            ids = devices if devices is not None else (list(distributed.device_ids) if distributed else [profile.get('device', 0)])
            for name in [*(f'davinci{i}' for i in ids), 'davinci_manager', 'devmm_svm', 'hisi_hdc']:
                path = Path('/dev') / name
                if path.exists():
                    self.add(path, device=True)
            if Path('/sys').exists():
                self.add('/sys')
        return self

    def argv(self, command, *, cwd, environment):
        argv = [executable(), '--die-with-parent', '--unshare-user', '--unshare-pid',
                '--unshare-ipc', '--unshare-uts', '--cap-drop', 'ALL',
                '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
                '--dir', '/home/ptobench', '--dir', '/ascend/log', '--dir', '/ptobench']
        if self.isolate_network:
            argv.insert(1, '--unshare-net')
        # Bind ancestors before their explicit children so restrictive overlays win.
        for source, destination, mode in sorted(self.mounts, key=lambda m: len(Path(m[1]).parts)):
            flag = '--dev-bind' if mode == 'device' else '--bind' if mode == 'rw' else '--ro-bind'
            argv.extend([flag, source, destination])
        for source, destination in self.links:
            argv.extend(['--symlink', source, destination])
        for key, value in environment.items():
            argv.extend(['--setenv', key, str(value)])
        return [*argv, '--chdir', str(cwd), '--', *map(str, command)]

    def manifest(self):
        binary = Path(executable())
        return {'policy_version': POLICY_VERSION,
                'network': ('model_api_only' if self.isolate_network else 'allowed'),
                'network_allowlist': self.network_allowlist,
                'bubblewrap_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'mounts': [{'source': s, 'destination': d, 'access': m} for s, d, m in self.mounts],
                'symlinks': [{'target': s, 'path': d} for s, d in self.links]}


def copy_runtime(destination):
    """Explicit Python tool code only, excluding tests, docs and credentials."""
    destination = Path(destination)
    directories = ['evaluator', 'launcher', 'modes',
                   *(f'modes/{name}' for name in RUNTIME_MODE_DIRS)]
    for directory in directories:
        for source in (common.BENCHMARK_ROOT / directory).glob('*.py'):
            if source.name == 'sandbox_supervisor.py':
                continue
            target = destination / directory / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for relative in ('modes/cannbot/bridge/pi_task.ts', 'evaluator/tuning_policies.json',
                     'evaluator/pi_file_guard.mjs'):
        source = common.BENCHMARK_ROOT / relative
        if source.exists():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return destination


def runtime_mounts(view, runtime):
    # Each top-level tool directory is an explicit minimal copy, not the repo.
    for name in ('evaluator', 'launcher', 'modes'):
        if (runtime / name).exists():
            view.add(runtime / name, common.BENCHMARK_ROOT / name)



def check_available():
    view = View().system()
    environment = clean_environment()
    subprocess.run(view.argv(['/usr/bin/true'], cwd='/tmp', environment=environment),
                   check=True, timeout=15, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   env=sandbox_launcher_environment(environment))
    return view.manifest()
