"""Prepare the fixed, unprivileged Bubblewrap Agent launch."""
from __future__ import annotations

import grp
import json
import os
from pathlib import Path
import pwd
import re
import secrets
import stat
import subprocess

import environment_lock


AGENT_USER = 'ptobench-agent'
SHARE_GROUP = 'ptobench-share'
DISPATCHER = Path(__file__).resolve().with_name('agent_dispatch.py')
REQUEST_ENV = 'PTOBENCH_AGENT_LAUNCH_REQUEST'
MAX_REQUEST_BYTES = 1024 * 1024
_REQUEST_NAME = re.compile(r'[0-9a-f]{32}\.json\Z')


def _host_credentials():
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, separator, value = line.partition(':')
        if separator and key in {'Uid', 'Gid', 'Groups', 'CapEff', 'NoNewPrivs'}:
            status[key] = value.strip()
    uid, euid, _ = os.getresuid()
    gid, egid, _ = os.getresgid()
    if uid != euid or gid != egid:
        raise RuntimeError('Agent launch requires matching Broker real and effective UID/GID')
    groups = os.getgroups()
    status_groups = [int(value) for value in status['Groups'].split()]
    if groups != status_groups:
        raise RuntimeError('Broker supplementary groups differ from /proc/self/status')
    return {'uid': uid, 'gid': gid, 'groups': groups,
            'cap_eff': int(status['CapEff'], 16), 'no_new_privs': status['NoNewPrivs'] == '1'}


def _acl_output_has_write(output: str) -> bool:
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        entry, _, annotation = line.partition('#')
        permission = annotation.partition('effective:')[2].strip() if 'effective:' in annotation else entry.rsplit(':', 1)[-1]
        fields = entry.split(':')
        if fields[:2] == ['user', '']:
            continue  # The root-owned file/directory owner is trusted.
        if len(permission) == 3 and permission[1] == 'w':
            if fields[:2] == ['user', '0']:
                continue
            return True
    return False


def _acl_effective_permissions(output, metadata, uid, groups):
    entries = {}
    mask = 'rwx'
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        fields = line.partition('#')[0].strip().split(':')
        if fields[0] == 'default':
            continue
        if len(fields) != 3 or fields[0] not in {'user', 'group', 'mask', 'other'}:
            raise RuntimeError('cannot parse effective Agent dispatcher ACL')
        tag, qualifier, permission = fields
        annotation = line.partition('#')[2]
        effective = annotation.partition('effective:')[2].strip() if 'effective:' in annotation else None
        if len(permission) != 3 or any(value not in 'rwx-' for value in permission):
            raise RuntimeError('cannot parse effective Agent dispatcher ACL')
        if tag == 'mask' and not qualifier:
            mask = permission
        else:
            entries[(tag, qualifier)] = (permission, effective)

    def allowed(entry, *, apply_mask=False):
        if entry is None:
            return set()
        permission, effective = entry
        if effective is not None:
            permission = effective
        elif apply_mask:
            permission = ''.join(value if value in mask else '-' for value in permission)
        return {value for value in 'rwx' if value in permission}

    if metadata.st_uid == uid:
        permissions = allowed(entries.get(('user', '')))
    elif ('user', str(uid)) in entries:
        permissions = allowed(entries[('user', str(uid))], apply_mask=True)
    else:
        matching_groups = []
        if metadata.st_gid in groups:
            matching_groups.append(entries.get(('group', '')))
        matching_groups.extend(entry for (tag, qualifier), entry in entries.items()
                               if tag == 'group' and qualifier.isdigit()
                               and int(qualifier) in groups)
        if matching_groups:
            permissions = set().union(*(allowed(entry, apply_mask=True)
                                        for entry in matching_groups))
        else:
            permissions = allowed(entries.get(('other', '')))
    return ''.join(value if value in permissions else '-' for value in 'rwx')


def _getfacl_binary():
    for directory in os.defpath.split(os.pathsep):
        candidate = Path(directory) / 'getfacl'
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0 or metadata.st_mode & 0o022:
            continue
        if all(parent.stat().st_uid == 0 and not parent.stat().st_mode & 0o022
               for parent in resolved.parents):
            return str(resolved)
    return None


def _acl_has_write(path: Path) -> bool:
    getfacl = _getfacl_binary()
    if getfacl is None:
        return True
    try:
        result = subprocess.run([getfacl, '-e', '-n', '-cp', str(path)], check=True,
                                capture_output=True, text=True, timeout=5,
                                env={'PATH': os.defpath, 'LC_ALL': 'C'})
    except (OSError, subprocess.SubprocessError):
        return True
    return _acl_output_has_write(result.stdout)


def _agent_directory_permissions(path: Path, agent_uid: int, share_gid: int) -> str:
    getfacl = _getfacl_binary()
    if getfacl is None:
        raise RuntimeError('trusted getfacl is unavailable for Agent dispatcher path validation')
    try:
        result = subprocess.run([getfacl, '-e', '-n', '-cp', str(path)], check=True,
                                capture_output=True, text=True, timeout=5,
                                env={'PATH': os.defpath, 'LC_ALL': 'C'})
        metadata = path.stat()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f'cannot validate Agent dispatcher parent ACL: {path}') from error
    return _acl_effective_permissions(result.stdout, metadata, agent_uid, {share_gid})


def _validate_agent_dispatcher_path(agent_uid: int, share_gid: int) -> None:
    for directory in reversed(DISPATCHER.parents):
        permissions = _agent_directory_permissions(directory, agent_uid, share_gid)
        metadata = directory.stat()
        if metadata.st_uid == 0:
            valid = 'x' in permissions and 'w' not in permissions
        else:
            valid = permissions == '--x'
        if not valid:
            raise RuntimeError(f'Agent must have traversal-only access to the fixed dispatcher parent: {directory}')
    permissions = _agent_directory_permissions(DISPATCHER, agent_uid, share_gid)
    if 'r' not in permissions or 'w' in permissions:
        raise RuntimeError('Agent cannot safely read the fixed dispatcher path')


def _trusted_binary(path):
    if not path:
        return None
    try:
        candidate = Path(path).resolve(strict=True)
        metadata = candidate.stat()
    except OSError:
        return None
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0
            or metadata.st_mode & 0o022 or not os.access(candidate, os.X_OK)):
        return None
    for entry in (candidate, *candidate.parents):
        try:
            entry_stat = entry.stat()
        except OSError:
            return None
        if entry_stat.st_uid != 0 or entry_stat.st_mode & 0o022 or _acl_has_write(entry):
            return None
    return str(candidate)


def _system_binary(name):
    for directory in os.defpath.split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            trusted = _trusted_binary(candidate)
            if trusted is not None:
                return trusted
    return None


def _agent_identity():
    try:
        agent = pwd.getpwnam(AGENT_USER)
        shared = grp.getgrnam(SHARE_GROUP)
        groups = sorted(set(os.getgrouplist(agent.pw_name, agent.pw_gid)))
    except (KeyError, OSError) as error:
        raise RuntimeError('dedicated Agent account or shared group is unavailable') from error
    if (agent.pw_uid <= 0 or agent.pw_gid != shared.gr_gid
            or groups != [shared.gr_gid] or agent.pw_uid == os.getuid()):
        raise RuntimeError('ptobench-agent must be distinct from Broker and belong only to ptobench-share')
    if Path(agent.pw_shell).name not in {'nologin', 'false'}:
        raise RuntimeError('ptobench-agent must have a non-interactive login shell')
    if not Path(agent.pw_shell).is_file() or not os.access(agent.pw_shell, os.X_OK):
        raise RuntimeError('ptobench-agent non-interactive shell is unavailable')
    return agent.pw_uid, shared.gr_gid


def _prepare_dispatcher(broker_uid: int, share_gid: int, agent_uid: int) -> None:
    try:
        metadata = DISPATCHER.lstat()
    except OSError as error:
        raise RuntimeError(f'fixed Agent dispatcher is unavailable: {DISPATCHER}') from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_uid != broker_uid:
        raise RuntimeError('fixed Agent dispatcher must be a Broker-owned regular file')
    try:
        os.chown(DISPATCHER, broker_uid, share_gid)
        os.chmod(DISPATCHER, 0o440)
    except OSError as error:
        raise RuntimeError('cannot prepare the fixed Agent dispatcher for shared-group reading') from error
    metadata = DISPATCHER.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != broker_uid
            or metadata.st_gid != share_gid or stat.S_IMODE(metadata.st_mode) != 0o440):
        raise RuntimeError('fixed Agent dispatcher ownership or mode is invalid')
    _validate_agent_dispatcher_path(agent_uid, share_gid)


def _request_directories(request_directory: Path, broker_uid: int, share_gid: int) -> Path:
    request_directory = Path(request_directory)
    if request_directory.name != 'launcher' or request_directory.parent.name != 'agent':
        raise RuntimeError('Agent launch requests must use session_root/agent/launcher')
    session_root = request_directory.parent.parent
    root = session_root.lstat()
    if not stat.S_ISDIR(root.st_mode) or root.st_uid != broker_uid:
        raise RuntimeError('Agent request session root must be a Broker-owned directory')
    for directory in (request_directory.parent, request_directory):
        try:
            directory.mkdir(mode=0o2710)
        except FileExistsError:
            pass
        except OSError as error:
            raise RuntimeError(f'cannot create Agent request directory: {directory}') from error
        try:
            metadata = directory.lstat()
        except OSError as error:
            raise RuntimeError(f'cannot inspect Agent request directory: {directory}') from error
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != broker_uid:
            raise RuntimeError(f'Agent request directory must be Broker-owned: {directory}')
        if metadata.st_gid != os.getgid() and share_gid not in os.getgroups():
            raise RuntimeError('Broker cannot establish ptobench-share request directory ownership')
        try:
            os.chown(directory, broker_uid, share_gid)
            os.chmod(directory, 0o2710)
            metadata = directory.lstat()
        except OSError as error:
            raise RuntimeError(f'cannot prepare Agent request directory: {directory}') from error
        if (metadata.st_gid != share_gid or stat.S_IMODE(metadata.st_mode) != 0o2710):
            raise RuntimeError(f'Agent request directory permissions are invalid: {directory}')
    return request_directory


def ensure_request_directories(session_root: Path) -> Path:
    """Create the Plan 01 Agent/launcher directory chain for a new Session."""
    credentials = _host_credentials()
    _agent_uid, share_gid = _agent_identity()
    if share_gid not in credentials['groups']:
        raise RuntimeError('Broker must belong to ptobench-share before creating Agent directories')
    return _request_directories(Path(session_root) / 'agent' / 'launcher',
                                credentials['uid'], share_gid)


def _write_request(command, cwd, request_directory, broker_uid, share_gid):
    directory = _request_directories(Path(request_directory), broker_uid, share_gid)
    request = {'schema_version': 1, 'argv': list(map(str, command)),
               'cwd': str(Path(cwd).absolute())}
    payload = (json.dumps(request, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(payload) > MAX_REQUEST_BYTES:
        raise RuntimeError('Agent launcher request is too large')
    path = directory / f'{secrets.token_hex(16)}.json'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchown(descriptor, broker_uid, share_gid)
        os.fchmod(descriptor, 0o640)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short write to Agent launch request')
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != broker_uid
            or metadata.st_gid != share_gid or stat.S_IMODE(metadata.st_mode) != 0o640
            or metadata.st_nlink != 1):
        path.unlink(missing_ok=True)
        raise RuntimeError('Agent launch request ownership or mode is invalid')
    return path


def cleanup_request(environment):
    request_value = environment.get(REQUEST_ENV) if environment else None
    if not request_value:
        return
    path = Path(request_value)
    if (not path.is_absolute() or path.parent.name != 'launcher'
            or path.parent.parent.name != 'agent'
            or not re.fullmatch(r'[0-9a-f]{32}\.json', path.name)):
        raise RuntimeError('refusing to clean an invalid Agent request path')
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    shared_gid = grp.getgrnam(SHARE_GROUP).gr_gid
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_gid != shared_gid or stat.S_IMODE(metadata.st_mode) != 0o640
            or metadata.st_nlink != 1):
        raise RuntimeError('refusing to clean a replaced Agent request file')
    path.unlink()


def cleanup_stale_requests(session_root):
    """Remove Broker-owned requests after recovery has stopped Session children."""
    root = Path(session_root)
    agent_directory = root / 'agent'
    request_directory = agent_directory / 'launcher'
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_uid != os.getuid():
        raise RuntimeError('Agent request session root must be a Broker-owned directory')
    try:
        agent_metadata = agent_directory.lstat()
    except FileNotFoundError:
        return 0
    try:
        request_metadata = request_directory.lstat()
    except FileNotFoundError:
        return 0
    shared_gid = grp.getgrnam(SHARE_GROUP).gr_gid
    for path, metadata in ((agent_directory, agent_metadata),
                           (request_directory, request_metadata)):
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_gid != shared_gid or stat.S_IMODE(metadata.st_mode) != 0o2710):
            raise RuntimeError(f'Agent request directory permissions are invalid: {path}')
    removed = 0
    for path in request_directory.iterdir():
        if not _REQUEST_NAME.fullmatch(path.name):
            continue
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_gid != shared_gid or stat.S_IMODE(metadata.st_mode) != 0o640
                or metadata.st_nlink != 1):
            raise RuntimeError(f'refusing to clean a replaced Agent request file: {path}')
        path.unlink()
        removed += 1
    return removed


def run_process(command, *, cwd, environment):
    """Wait for the Agent process and remove its Broker-owned request."""
    process = None
    try:
        process = subprocess.Popen(command, cwd=cwd, env=environment, shell=False)
        return process.wait()
    finally:
        cleanup_request_after_process(environment, process)


def cleanup_request_after_process(environment, process):
    """Remove an Agent request only once its child has exited."""
    if process is not None and process.poll() is None:
        return
    cleanup_request(environment)


def prepare(command: list[str], cwd: Path, request_directory: Path):
    credentials = _host_credentials()
    agent_uid, share_gid = _agent_identity()
    if share_gid not in credentials['groups']:
        raise RuntimeError('Broker must belong to ptobench-share before launching the Agent')
    if credentials['cap_eff'] != 0:
        raise RuntimeError('Broker must have zero effective capabilities before launching the Agent')
    if credentials['no_new_privs']:
        raise RuntimeError('Broker NoNewPrivs=1 prevents the fixed sudo Agent launch')
    if agent_uid == credentials['uid'] or credentials['uid'] == 0:
        raise RuntimeError('Broker and Agent must be distinct non-root identities')
    python = environment_lock.locked_python_executable()
    if not isinstance(python, Path) or not python.is_absolute():
        raise RuntimeError('locked Python executable must resolve to an absolute Path')
    sudo = _system_binary('sudo')
    if sudo is None:
        raise RuntimeError('no trusted system sudo executable is available')
    if not DISPATCHER.is_absolute():
        raise RuntimeError('fixed Agent dispatcher path must be absolute')
    _prepare_dispatcher(credentials['uid'], share_gid, agent_uid)
    request_path = _write_request(command, cwd, request_directory,
                                  credentials['uid'], share_gid)
    argv = [sudo, '-n', '-u', AGENT_USER, '-g', SHARE_GROUP, '--',
            str(python), '-I', '-S', str(DISPATCHER)]
    return argv, {'PATH': os.defpath, REQUEST_ENV: str(request_path)}
