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
    executable()


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
             'PTOBENCH_PI_TOOLS_DIR',
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
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f'Agent shared path must be a real directory: {path}')
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


def install_pi_private_tools(config_directory, tools_directory, workspace_root):
    """Install Broker-owned Pi configuration and its read-only trusted tools."""
    import grp
    share_gid = grp.getgrnam('ptobench-share').gr_gid
    config_directory = Path(config_directory)
    tools_directory = Path(tools_directory)
    config_directory.mkdir(parents=True, exist_ok=True, mode=0o750)
    tools_directory.mkdir(parents=True, exist_ok=True, mode=0o750)
    _agent_shared_directory(config_directory, mode=0o750, group_id=share_gid)
    _agent_shared_directory(tools_directory, mode=0o750, group_id=share_gid)
    for source_name, target_name in (
            ('pi_file_guard.mjs', 'ptobench-file-guard.mjs'),
            ('pi_shell_wrapper.py', 'ptobench-shell')):
        target = tools_directory / target_name
        temporary = tools_directory / f'.{target_name}.{uuid.uuid4().hex}'
        try:
            shutil.copyfile(Path(__file__).with_name(source_name), temporary)
            os.chown(temporary, -1, share_gid)
            temporary.chmod(0o550 if source_name == 'pi_shell_wrapper.py' else 0o440)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    shell_config = tools_directory / 'shell.json'
    file_tree.atomic_bytes(shell_config,
        (json.dumps({'bwrap': executable(),
                     'workspace_root': str(Path(workspace_root).resolve())}) + '\n').encode(), mode=0o440)
    os.chown(shell_config, -1, share_gid)
    auth_path = config_directory / 'auth.json'
    file_tree.atomic_bytes(auth_path, b'{}\n', mode=0o440)
    os.chown(auth_path, -1, share_gid)
    settings_path = config_directory / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    settings['shellPath'] = '/ptobench/pi-tools/ptobench-shell'
    file_tree.atomic_bytes(settings_path,
        (json.dumps(settings, indent=2) + '\n').encode(), mode=0o440)
    os.chown(settings_path, -1, share_gid)


def agent_command(command, *, cwd, environment, workspace, session_root,
                  cann_bot_extension=None, cann_bot_socket=None, backend=None):
    """Build the verified host launcher and Bubblewrap command for an Agent."""
    if not enabled():
        raise RuntimeError('Pi Agent requires filesystem isolation')
    command = list(map(str, command))
    view = View(isolate_network=True).system()
    is_pi = (backend == 'pi' or Path(command[0]).name == 'pi'
             or any('pi-coding-agent/dist/bundle/cli.js' in value for value in command[:3]))
    view.native_agent(command=command, environment=environment, is_pi=is_pi)
    workspace_root = Path(session_root) / 'workspaces'
    workspace_root_metadata = workspace_root.lstat()
    if not stat.S_ISDIR(workspace_root_metadata.st_mode):
        raise RuntimeError('Agent Session workspaces must be a real directory')
    workspace = Path(workspace).absolute()
    if workspace.is_symlink() or not workspace.is_dir():
        raise RuntimeError('Agent Workspace must be a real registered directory')
    try:
        relative_workspace = workspace.resolve().relative_to(workspace_root.resolve())
    except ValueError as error:
        raise RuntimeError('Agent Workspace is outside the current Session') from error
    if len(relative_workspace.parts) != 1:
        raise RuntimeError('Agent Workspace must be a direct child of current Session workspaces')
    view.add(workspace_root, workspace_root, write=True)

    import agent_launch
    agent_launch.ensure_request_directories(Path(session_root))
    share_gid = grp.getgrnam('ptobench-share').gr_gid
    pi_config = Path(session_root) / 'agent/pi-config'
    pi_tools = Path(session_root) / 'agent/pi-tools'
    native_sessions = Path(session_root) / 'agent/native'
    for path, mode in ((pi_config, 0o750), (pi_tools, 0o750), (native_sessions, 0o2770)):
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        _agent_shared_directory(path, mode=mode, group_id=share_gid)
    install_pi_private_tools(pi_config, pi_tools, workspace_root)
    view.add_tmpfs('/ptobench/pi-config')
    for name in ('settings.json', 'models.json', 'auth.json', 'unlimited-output.ts'):
        config_file = pi_config / name
        try:
            metadata = config_file.lstat()
        except FileNotFoundError:
            continue
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_gid != share_gid or metadata.st_mode & 0o022):
            raise RuntimeError(f'Pi config file must be Broker-owned and read-only: {config_file}')
        view.add(config_file, f'/ptobench/pi-config/{name}')
    view.add(pi_tools, '/ptobench/pi-tools')
    view.add(native_sessions, '/ptobench/agent-sessions', write=True)
    api_socket = environment.get('PTOBENCH_API_PROXY_SOCKET')
    if not api_socket:
        raise RuntimeError('Agent Broker or model API gateway is unavailable')
    _agent_shared_socket(api_socket)
    view.add(api_socket, '/ptobench/network/api.sock')
    socket_path = environment.get('PTOBENCH_RPC_SOCKET')
    if not socket_path:
        raise RuntimeError('PTOBench Agent Broker is unavailable')
    _agent_shared_socket(socket_path)
    view.add(socket_path, '/ptobench/cli.sock')

    client_value = environment.get('PTOBENCH_CLIENT_PATH')
    if not client_value:
        raise RuntimeError('Broker CLI transport shim is unavailable')
    client = Path(client_value)
    launcher_root = Path(session_root) / 'agent/launcher'
    try:
        client.relative_to(launcher_root)
        client_parent = client.parent.lstat()
        metadata = client.lstat()
        client.resolve(strict=True).relative_to(launcher_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise RuntimeError('Broker CLI transport shim must be in the current Session launcher directory') from error
    if (not stat.S_ISDIR(client_parent.st_mode) or client_parent.st_uid != os.getuid()
            or client_parent.st_gid != share_gid or stat.S_IMODE(client_parent.st_mode) != 0o2710
            or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_gid != share_gid or stat.S_IMODE(metadata.st_mode) != 0o550
            or metadata.st_nlink != 1 or not os.access(client, os.X_OK)):
        raise RuntimeError('Broker CLI transport shim ownership or mode is invalid')
    if client.read_bytes() != Path(__file__).with_name('sandbox_client.py').read_bytes():
        raise RuntimeError('Broker CLI transport shim content is invalid')
    view.add(client, '/ptobench/ptobench')

    # The bridge runs inside the Agent's private network namespace so its
    # loopback listener can reach only the mounted, path-limited API socket.
    # It deletes its temporary source before starting the Agent process.
    bridge_dir = Path(session_root) / 'agent/.runtime' / uuid.uuid4().hex
    bridge_dir.mkdir(parents=True, mode=0o2770)
    _agent_shared_directory(bridge_dir, group_id=grp.getgrnam('ptobench-share').gr_gid)
    bridge = bridge_dir / 'network_proxy.py'
    shutil.copyfile(Path(__file__).with_name('network_proxy.py'), bridge)
    bridge.chmod(0o440)
    view.add(bridge_dir, '/ptobench/runtime/bridge', write=True)
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
                try:
                    relative_prompt = prompt_path.resolve().relative_to(pi_tools.resolve())
                except ValueError as error:
                    raise RuntimeError('Pi prompt files must be provided through the public prompt argument') from error
                command[index + 1] = str(Path('/ptobench/pi-tools') / relative_prompt)
        if argument == '--session-dir':
            session_directory = Path(command[index + 1])
            if not session_directory.is_absolute():
                session_directory = Path(cwd) / session_directory
            if session_directory.resolve() != native_sessions.resolve():
                raise RuntimeError('Pi session files must use the current Session native directory')
            command[index + 1] = '/ptobench/agent-sessions'
            session_directory_mounted = True
        elif argument.startswith('--session-dir='):
            session_directory = Path(argument.split('=', 1)[1])
            if not session_directory.is_absolute():
                session_directory = Path(cwd) / session_directory
            if session_directory.resolve() != native_sessions.resolve():
                raise RuntimeError('Pi session files must use the current Session native directory')
            view.add(session_directory, '/ptobench/agent-sessions', write=True)
            command[index] = '--session-dir=/ptobench/agent-sessions'
            session_directory_mounted = True
    if not session_directory_mounted:
        prompt_index = command.index('{prompt}') if '{prompt}' in command else len(command)
        command[prompt_index:prompt_index] = ['--session-dir', '/ptobench/agent-sessions']

    if is_pi:
        for index, value in enumerate(command):
            path_value = Path(value)
            if path_value.is_absolute():
                try:
                    relative = path_value.relative_to(pi_config)
                except ValueError:
                    continue
                command[index] = str(Path('/ptobench/pi-config') / relative)
        flags = ['--no-extensions', '--no-approve', '--extension',
                 '/ptobench/pi-tools/ptobench-file-guard.mjs']
        insertion = command.index('{prompt}') if '{prompt}' in command else len(command)
        command[insertion:insertion] = flags
    if cann_bot_extension:
        try:
            relative_extension = Path(cann_bot_extension).resolve().relative_to(pi_tools.resolve())
        except ValueError as error:
            raise RuntimeError('CANNBot extension must be stored in the Broker-owned Pi tools directory') from error
        extension_path = str(Path('/ptobench/pi-tools') / relative_extension)
        command = [extension_path if value == str(cann_bot_extension) else value for value in command]
    command = [sys.executable, '/ptobench/runtime/bridge/network_proxy.py', 'agent',
               '--api-socket', '/ptobench/network/api.sock', '--api-port', '18081',
               '--delete-runtime-files', '--', *command]
    if cann_bot_socket:
        _agent_shared_socket(cann_bot_socket)
        view.add(cann_bot_socket, '/ptobench/cannbot.sock')

    cleaned = agent_environment(environment)
    cleaned['PTOBENCH_RPC_SOCKET'] = '/ptobench/cli.sock'
    cleaned['PTOBENCH_API_PROXY_SOCKET'] = '/ptobench/network/api.sock'
    cleaned['PI_CODING_AGENT_DIR'] = '/ptobench/pi-config'
    cleaned['PTOBENCH_PI_TOOLS_DIR'] = '/ptobench/pi-tools'
    cleaned['PTOBENCH_AGENT_SESSION_DIR'] = '/ptobench/agent-sessions'
    cleaned['PATH'] = os.pathsep.join(('/ptobench', '/ptobench/runtime', '/usr/bin', '/bin'))
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

    def add_tmpfs(self, destination):
        item = ('tmpfs', str(Path(destination)), 'tmpfs')
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

    def native_agent(self, *, command=None, environment=None, is_pi=False):
        if not is_pi:
            return self
        if command is None:
            raise RuntimeError('Pi runtime command is unavailable')
        environment = os.environ if environment is None else environment
        search_path = environment.get('PATH', os.defpath)
        node = shutil.which('node', path=search_path)
        if not node:
            raise RuntimeError('Pi runtime requires Node from the frozen host manifest')
        node_path = Path(node).resolve(strict=True)
        if not node_path.is_file() or not os.access(node_path, os.X_OK):
            raise RuntimeError('Pi runtime Node executable is unavailable')
        self.add(node_path, '/ptobench/runtime/node')

        resolved = shutil.which(command[0], path=search_path) if not Path(command[0]).is_absolute() else command[0]
        bundle = Path(resolved).resolve(strict=True) if resolved else None
        if bundle is None or not bundle.is_file() or bundle.suffix != '.js':
            raise RuntimeError('Pi CLI bundle is unavailable from the frozen host manifest')
        self.add(bundle.parent, '/ptobench/runtime/pi')
        jiti = bundle.parents[2] / 'node_modules/jiti'
        if not jiti.is_dir():
            raise RuntimeError('Pi TypeScript extension runtime is unavailable from the frozen host manifest')
        self.add(jiti.resolve(strict=True), '/ptobench/runtime/node_modules/jiti')
        command[0] = str(Path('/ptobench/runtime/pi') / bundle.name)
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
            if mode == 'tmpfs':
                argv.extend(['--tmpfs', destination])
            else:
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
