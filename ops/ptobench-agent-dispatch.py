#!/usr/bin/env python3
"""Unprivileged, one-shot Bubblewrap map setup and Agent exec dispatcher."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import pwd
import select
import shutil
import stat
import subprocess
import sys


MAX_REQUEST_BYTES = 1024 * 1024
PR_SET_NO_NEW_PRIVS = 38
UID_MAP = 'uid_map'
GID_MAP = 'gid_map'
_BOOTSTRAP = r'''
import ctypes, os, pathlib, sys
broker_uid, broker_gid, agent_uid, host_agent_uid, host_agent_gid = map(int, sys.argv[1:6])
if (os.getuid() != broker_uid or os.getgid() != broker_gid or os.getgroups()
        or os.geteuid() != broker_uid or os.getegid() != broker_gid):
    raise SystemExit('Agent identity bootstrap rejected unexpected Broker namespace credentials')
def read_map(path):
    return [tuple(map(int, line.split())) for line in pathlib.Path(path).read_text().splitlines()]
if read_map('/proc/self/uid_map') != sorted([(broker_uid, broker_uid, 1), (agent_uid, host_agent_uid, 1)]):
    raise SystemExit('Agent UID map does not match the detected Broker and subordinate IDs')
if read_map('/proc/self/gid_map') != [(0, host_agent_gid, 1), (broker_gid, broker_gid, 1)]:
    raise SystemExit('Agent GID map does not match the detected shared and subordinate IDs')
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(8, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'unable to preserve bootstrap capabilities during UID switch')
os.setresuid(agent_uid, agent_uid, agent_uid)
class Header(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
class Data(ctypes.Structure):
    _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32),
                ('inheritable', ctypes.c_uint32)]
header = Header(0x20080522, 0)
bootstrap_caps = (1 << 7) | (1 << 8)
data = (Data * 2)(Data(bootstrap_caps, bootstrap_caps, 0), Data(0, 0, 0))
if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
    raise OSError(ctypes.get_errno(), 'unable to retain only capability-drop privileges')
last_cap = int(pathlib.Path('/proc/sys/kernel/cap_last_cap').read_text())
for cap in range(last_cap + 1):
    if libc.prctl(24, cap, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'unable to drop capability bounding set')
data = (Data * 2)(Data(0, 0, 0), Data(0, 0, 0))
if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
    raise OSError(ctypes.get_errno(), 'unable to clear Agent capabilities')
if libc.prctl(38, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'unable to set Agent NoNewPrivs')
status = pathlib.Path('/proc/self/status').read_text()
fields = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
          for line in status.splitlines() if ':' in line}
cap_fields = ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')
if (os.getuid() != agent_uid or os.getgid() != broker_gid or os.getgroups()
        or any(int(fields[key], 16) for key in cap_fields) or fields['NoNewPrivs'] != '1'):
    raise SystemExit('Agent identity bootstrap failed final credential checks')
candidate = pathlib.Path(os.environ.pop('PTOBENCH_AGENT_STARTUP_CANDIDATE'))
read_canary = pathlib.Path(os.environ.pop('PTOBENCH_AGENT_STARTUP_READ_CANARY'))
dac_canary = pathlib.Path(os.environ.pop('PTOBENCH_AGENT_STARTUP_DAC_CANARY'))
marker = os.environ.pop('PTOBENCH_AGENT_STARTUP_MARKER')
os.environ.pop('PTOBENCH_AGENT_STARTUP_GROUPS', None)
try:
    if not candidate.is_dir() or not os.access(candidate, os.R_OK | os.W_OK | os.X_OK):
        raise SystemExit('Candidate permission check failed: Agent needs read/write access')
    if read_canary.read_text() != marker:
        raise SystemExit('Candidate permission check failed: Agent cannot read the Broker group canary')
    probe = candidate / ('.ptobench-agent-startup-write-' + os.urandom(12).hex())
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
    os.fchmod(descriptor, 0o660)
    os.write(descriptor, marker.encode())
    os.close(descriptor)
    if probe.read_text() != marker or probe.stat().st_gid != os.getgid():
        raise SystemExit('Candidate permission check failed: Agent-created files must use the Broker group')
    try:
        dac_canary.read_bytes()
    except PermissionError:
        pass
    except OSError as error:
        raise SystemExit('Candidate DAC canary could not be validated') from error
    else:
        raise SystemExit('Candidate permission check failed: Agent can read an unauthorized-group canary')
    for protected_name in ('.benchmark_context', '.benchmark_outputs'):
        protected = pathlib.Path(protected_name)
        if not protected.is_dir() or not os.access(protected, os.R_OK | os.X_OK):
            raise SystemExit('Sandbox permission check failed: protected workspace files are not readable')
        protected_probe = protected / ('.ptobench-agent-startup-write-' + os.urandom(12).hex())
        try:
            descriptor = os.open(protected_probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except OSError:
            pass
        else:
            os.close(descriptor)
            protected_probe.unlink(missing_ok=True)
            raise SystemExit('Sandbox permission check failed: Agent can write protected workspace data')
finally:
    read_canary.unlink(missing_ok=True)
    dac_canary.unlink(missing_ok=True)
command = sys.argv[7:]
if not command:
    raise SystemExit('Agent identity bootstrap has no command')
os.execvpe(command[0], command, os.environ)
'''


def _proc_credentials():
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, separator, value = line.partition(':')
        if separator:
            fields[key] = value.strip()
    return fields


def check_identity(expected_uid: int, expected_gid: int) -> None:
    fields = _proc_credentials()
    if os.getuid() != expected_uid or os.geteuid() != expected_uid:
        raise RuntimeError('Agent dispatcher UID does not match the requesting PTOBench user')
    if os.getgid() != expected_gid or os.getegid() != expected_gid:
        raise RuntimeError('Agent dispatcher GID does not match the requesting PTOBench group')
    if os.getgroups():
        raise RuntimeError('Agent dispatcher still has supplementary groups')
    if any(int(fields.get(name, '-1'), 16) != 0
           for name in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb')):
        raise RuntimeError('Agent dispatcher still has active capabilities')


def read_request(path: str):
    request_path = Path(path)
    if not request_path.is_absolute() or request_path.parent.is_symlink():
        raise RuntimeError('Agent launch request must be an absolute path in a private directory')
    directory = request_path.parent.lstat()
    if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.getuid()
            or stat.S_IMODE(directory.st_mode) != 0o700):
        raise RuntimeError('Agent launch request directory must be caller-owned mode 0700')
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(request_path, flags)
    except OSError as error:
        raise RuntimeError('Agent launch request must be a regular file') from error
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError('Agent launch request must be a regular file')
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RuntimeError('Agent launch request must be caller-owned mode 0600')
        if metadata.st_nlink != 1:
            raise RuntimeError('Agent launch request must have a single link')
        if metadata.st_size > MAX_REQUEST_BYTES:
            raise RuntimeError('Agent launch request exceeds the maximum size')
        chunks = []
        size = 0
        while True:
            chunk = os.read(fd, min(65536, MAX_REQUEST_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise RuntimeError('Agent launch request exceeds the maximum size')
    finally:
        os.close(fd)
    os.unlink(request_path)
    try:
        request = json.loads(b''.join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError('Agent launch request is malformed') from error
    required = {'schema_version', 'command', 'cwd', 'uid', 'gid', 'groups', 'environment'}
    if not isinstance(request, dict) or set(request) != required or request['schema_version'] != 1:
        raise RuntimeError('Agent launch request has an invalid schema')
    if (not isinstance(request['command'], list) or not request['command']
            or any(not isinstance(value, str) or '\x00' in value for value in request['command'])):
        raise RuntimeError('Agent launch request command is invalid')
    if not isinstance(request['cwd'], str) or not Path(request['cwd']).is_absolute():
        raise RuntimeError('Agent launch request working directory is invalid')
    if (type(request['uid']) is not int or type(request['gid']) is not int
            or not isinstance(request['groups'], list)
            or any(type(value) is not int or value < 0 for value in request['groups'])):
        raise RuntimeError('Agent launch request identity is invalid')
    if request['environment'] != {'PATH': os.defpath}:
        raise RuntimeError('Agent launch request environment is not minimal')
    if request['uid'] != os.getuid() or request['gid'] != os.getgid():
        raise RuntimeError('Agent launch request does not belong to the current PTOBench identity')
    return request['command'], request['environment'], request['cwd']


def _subid_ranges(path: str, uid: int):
    account = pwd.getpwuid(uid).pw_name
    ranges = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as error:
        raise RuntimeError(f'cannot read {path}; configure subordinate IDs for the PTOBench account') from error
    for number, line in enumerate(lines, 1):
        if not line or line.lstrip().startswith('#'):
            continue
        fields = line.split(':')
        if len(fields) != 3:
            raise RuntimeError(f'invalid subordinate-ID entry at {path}:{number}')
        owner, start_text, count_text = fields
        if owner not in (account, str(uid)):
            continue
        try:
            start, count = int(start_text), int(count_text)
        except ValueError as error:
            raise RuntimeError(f'invalid subordinate-ID entry at {path}:{number}') from error
        if start < 0 or count < 1 or start + count > 2**32:
            raise RuntimeError(f'invalid subordinate-ID range at {path}:{number}')
        ranges.append((start, count))
    if not ranges:
        raise RuntimeError(f'no subordinate-ID range for the current PTOBench user in {path}')
    return ranges


def _verified_map_helper(name: str):
    path = shutil.which(name)
    if not path or not Path(path).is_absolute():
        raise RuntimeError(f'{name} is required for the configured subordinate user namespace')
    metadata = Path(path).stat()
    if metadata.st_uid != 0 or not metadata.st_mode & stat.S_ISUID or metadata.st_mode & 0o022:
        raise RuntimeError(f'{name} must be a root-owned, non-writable setuid mapping helper')
    return path


def _mapping_plan(uid: int, gid: int):
    if uid <= 0 or gid <= 0 or uid >= 2**32 - 1 or gid >= 2**32 - 1:
        raise RuntimeError('Agent isolation requires a non-root Broker UID/GID with room for namespace IDs')
    uid_ranges = _subid_ranges('/etc/subuid', uid)
    gid_ranges = _subid_ranges('/etc/subgid', uid)

    def choose(ranges, identity, kind):
        for start, count in ranges:
            for offset in range(count):
                candidate = start + offset
                if candidate != identity:
                    return candidate
        raise RuntimeError(f'detected subordinate {kind} ranges overlap the Broker identity')

    agent_uid = choose(uid_ranges, uid, 'UID')
    agent_gid = choose(gid_ranges, gid, 'GID')
    if uid + 1 >= 2**32 - 1:
        raise RuntimeError('Broker UID leaves no safe namespace ID for the Agent')
    uid_map = sorted([(uid, uid, 1), (uid + 1, agent_uid, 1)])
    gid_map = [(0, agent_gid, 1), (gid, gid, 1)]
    return {'agent_uid': agent_uid, 'agent_gid': agent_gid,
            'agent_ns_uid': uid + 1, 'agent_ns_gid': gid,
            'broker_ns_uid': uid, 'broker_ns_gid': gid,
            'uid_map': uid_map, 'gid_map': gid_map}


def _mount_destinations(argv):
    two_paths = {'--bind', '--ro-bind', '--dev-bind', '--bind-try', '--ro-bind-try',
                 '--dev-bind-try', '--bind-fd', '--ro-bind-fd'}
    destinations = set()
    index = 1
    while index < len(argv):
        value = argv[index]
        if value == '--':
            break
        if value in two_paths:
            if index + 2 >= len(argv):
                raise RuntimeError(f'incomplete Bubblewrap mount option: {value}')
            destinations.add(str(Path(argv[index + 2])))
            index += 3
        elif value in {'--proc', '--dev', '--tmpfs', '--dir'}:
            if index + 1 >= len(argv):
                raise RuntimeError(f'incomplete Bubblewrap mount option: {value}')
            destinations.add(str(Path(argv[index + 1])))
            index += 2
        else:
            index += 1
    return destinations


def _mapped_bwrap_command(command, mapping, python_path, block_fd, info_fd):
    argv = list(command)
    try:
        separator = argv.index('--')
    except ValueError as error:
        raise RuntimeError('Agent command is not a complete Bubblewrap argv') from error
    options = argv[1:separator]
    if '--unshare-user' not in options or '--userns' in options or '--userns2' in options:
        raise RuntimeError('Agent Bubblewrap command does not use the required user namespace')
    if any(option in options for option in ('--userns-block-fd', '--info-fd', '--uid', '--gid')):
        raise RuntimeError('Agent Bubblewrap command contains conflicting identity options')
    if '--cap-drop' not in options or options[options.index('--cap-drop') + 1] != 'ALL':
        raise RuntimeError('Agent Bubblewrap command must drop all sandbox capabilities')
    original_command = argv[separator + 1:]
    if not original_command:
        raise RuntimeError('Agent Bubblewrap command is empty')
    destinations = _mount_destinations(argv)
    parent_dirs = set()
    for destination in destinations:
        parent = Path(destination).parent
        while str(parent) != '/':
            value = str(parent)
            if value not in destinations:
                parent_dirs.add(value)
            parent = parent.parent
    for value in sorted(parent_dirs, key=lambda item: (len(Path(item).parts), item)):
        argv[separator:separator] = ['--chmod', '0755', value]
        separator += 3
    argv[separator:separator] = [
        '--uid', str(mapping['broker_ns_uid']), '--gid', str(mapping['broker_ns_gid']),
        '--cap-add', 'cap_setuid', '--cap-add', 'cap_setpcap',
        '--userns-block-fd', str(block_fd), '--info-fd', str(info_fd)]
    separator += 12
    bootstrap = [python_path, '-S', '-c', _BOOTSTRAP,
                 str(mapping['broker_ns_uid']), str(mapping['broker_ns_gid']),
                 str(mapping['agent_ns_uid']), str(mapping['agent_uid']), str(mapping['agent_gid']),
                 '--', *original_command]
    return [*argv[:separator], *bootstrap]


def _read_info(fd, process):
    ready, _, _ = select.select([fd], [], [], 15)
    if not ready:
        process.kill()
        raise RuntimeError('Bubblewrap did not create its user namespace within 15 seconds')
    data = bytearray()
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        data.extend(chunk)
        if b'\n}\n' in data:
            break
        if len(data) > 65536:
            process.kill()
            raise RuntimeError('Bubblewrap namespace information exceeded the size limit')
    try:
        info = json.loads(data)
        child = info['child-pid']
        if type(child) is not int or child < 1:
            raise ValueError('invalid child pid')
        return child
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        stderr = process.stderr.read(4096).decode(errors='replace') if process.stderr else ''
        raise RuntimeError(f'Bubblewrap failed before namespace setup: {stderr.strip()}') from error


def _write_maps(pid: int, mapping):
    uid_helper = _verified_map_helper('newuidmap')
    gid_helper = _verified_map_helper('newgidmap')
    uid_args = [value for row in mapping['uid_map'] for value in map(str, row)]
    gid_args = [value for row in mapping['gid_map'] for value in map(str, row)]
    commands = [[uid_helper, str(pid), *uid_args], [gid_helper, str(pid), *gid_args]]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10,
                                env={'PATH': os.defpath})
        if result.returncode:
            raise RuntimeError(f'{Path(command[0]).name} could not establish the detected namespace map: '
                               f'{result.stderr.strip()}')
    actual_uid = _read_map(Path(f'/proc/{pid}/{UID_MAP}'))
    actual_gid = _read_map(Path(f'/proc/{pid}/{GID_MAP}'))
    if actual_uid != mapping['uid_map'] or actual_gid != mapping['gid_map']:
        raise RuntimeError('Bubblewrap user namespace maps differ from the detected subordinate ranges')
    return mapping


def _read_map(path: Path):
    try:
        return [tuple(map(int, line.split())) for line in path.read_text().splitlines()]
    except (OSError, ValueError) as error:
        raise RuntimeError(f'cannot verify actual namespace map at {path}') from error


def _set_no_new_privs():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot enable NoNewPrivs before Agent execution')


def dispatch(request_path: str):
    if os.geteuid() == 0 or os.getegid() == 0:
        raise RuntimeError('Agent dispatcher must run after permanent privilege drop')
    expected_uid, expected_gid = os.getuid(), os.getgid()
    if os.environ.get('SUDO_UID') is not None:
        if os.environ.get('SUDO_UID') != str(expected_uid) or os.environ.get('SUDO_GID') != str(expected_gid):
            raise RuntimeError('sudo caller identity differs from the dropped dispatcher identity')
    check_identity(expected_uid, expected_gid)
    if _proc_credentials().get('NoNewPrivs') != '0':
        raise RuntimeError('Agent dispatcher must map subordinate IDs before enabling NoNewPrivs')
    command, environment, cwd = read_request(request_path)
    python_path = sys.executable
    argv = list(command)
    mapping = _mapping_plan(expected_uid, expected_gid)
    block_read, block_write = os.pipe()
    info_read, info_write = os.pipe()
    try:
        mapped = _mapped_bwrap_command(argv, mapping, python_path, block_read, info_write)
        process = subprocess.Popen(mapped, cwd=cwd, env=environment,
                                   pass_fds=(block_read, info_write), close_fds=True)
        os.close(block_read)
        block_read = -1
        os.close(info_write)
        info_write = -1
        child = _read_info(info_read, process)
        _write_maps(child, mapping)
        _set_no_new_privs()
        check_identity(expected_uid, expected_gid)
        if _proc_credentials().get('NoNewPrivs') != '1':
            raise RuntimeError('Agent dispatcher failed to enable NoNewPrivs before releasing Bubblewrap')
        os.write(block_write, b'1')
        os.close(block_write)
        block_write = -1
        return process.wait()
    except Exception:
        if 'process' in locals() and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        for descriptor in (block_read, block_write, info_read, info_write):
            if descriptor >= 0:
                os.close(descriptor)


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 0:
        print('ptobench-agent-dispatch: this entry accepts no arguments', file=sys.stderr)
        return 2
    request_path = os.environ.get('PTOBENCH_AGENT_LAUNCH_REQUEST')
    if not request_path:
        print('ptobench-agent-dispatch: request path is missing', file=sys.stderr)
        return 2
    try:
        return dispatch(request_path)
    except Exception as error:
        print(f'ptobench-agent-dispatch: {error}', file=sys.stderr)
        return 126


if __name__ == '__main__':
    raise SystemExit(main())
