"""Read one Broker-owned launch request and exec its Bubblewrap command."""
from __future__ import annotations

import grp
import json
import os
from pathlib import Path
import pwd
import re
import stat
import sys


AGENT_USER = 'ptobench-agent'
SHARE_GROUP = 'ptobench-share'
REQUEST_ENV = 'PTOBENCH_AGENT_LAUNCH_REQUEST'
MAX_REQUEST_BYTES = 1024 * 1024
_REQUEST_NAME = re.compile(r'[0-9a-f]{32}\.json\Z')


def _sudo_identity():
    try:
        broker_uid = int(os.environ['SUDO_UID'])
        broker_gid = int(os.environ['SUDO_GID'])
        broker = pwd.getpwuid(broker_uid)
    except (KeyError, ValueError, OSError) as error:
        raise RuntimeError('fixed Agent dispatcher lacks the sudo Broker identity') from error
    if broker_uid <= 0 or broker_uid == os.getuid() or broker.pw_gid != broker_gid:
        raise RuntimeError('fixed Agent dispatcher received an invalid sudo Broker identity')
    return broker_uid


def check_identity() -> None:
    try:
        agent = pwd.getpwnam(AGENT_USER)
        shared = grp.getgrnam(SHARE_GROUP)
        groups = sorted(set(os.getgroups()))
    except (KeyError, OSError) as error:
        raise RuntimeError('dedicated Agent account or shared group is unavailable') from error
    uid, euid, suid = os.getresuid()
    gid, egid, sgid = os.getresgid()
    if (uid, euid, suid) != (agent.pw_uid, agent.pw_uid, agent.pw_uid):
        raise RuntimeError('Agent dispatcher must run as ptobench-agent before reading a request')
    if (gid, egid, sgid) != (shared.gr_gid, shared.gr_gid, shared.gr_gid):
        raise RuntimeError('Agent dispatcher must run as ptobench-share before reading a request')
    if agent.pw_gid != shared.gr_gid or groups not in ([], [shared.gr_gid]):
        raise RuntimeError('ptobench-agent must have only ptobench-share as an effective group')
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, separator, value = line.partition(':')
        if separator:
            fields[key] = value.strip()
    if any(int(fields.get(name, '-1'), 16) for name in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb')):
        raise RuntimeError('Agent dispatcher must have no capabilities')
    status_groups = [int(value) for value in fields.get('Groups', '').split()]
    if status_groups not in ([], [shared.gr_gid]):
        raise RuntimeError('Agent dispatcher has unauthorized supplementary groups')
    _sudo_identity()


def _directory(path: Path, broker_uid: int, share_gid: int) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError('Agent launch request directory is unavailable') from error
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != broker_uid
            or metadata.st_gid != share_gid or stat.S_IMODE(metadata.st_mode) != 0o2710):
        raise RuntimeError('Agent launch request directory must be Broker-owned mode 2710')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read_request(path: str):
    request_path = Path(path)
    if (not request_path.is_absolute() or not _REQUEST_NAME.fullmatch(request_path.name)
            or request_path.parent.name != 'launcher'
            or request_path.parent.parent.name != 'agent'
            or request_path.parent.is_symlink() or request_path.parent.parent.is_symlink()):
        raise RuntimeError('Agent launch request must use the fixed session_root/agent/launcher path')
    broker_uid = _sudo_identity()
    try:
        share_gid = grp.getgrnam(SHARE_GROUP).gr_gid
    except KeyError as error:
        raise RuntimeError('ptobench-share is unavailable') from error
    _directory(request_path.parent.parent, broker_uid, share_gid)
    _directory(request_path.parent, broker_uid, share_gid)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(request_path, flags)
    except OSError as error:
        raise RuntimeError('Agent launch request must be a regular file') from error
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != broker_uid
                or metadata.st_gid != share_gid or stat.S_IMODE(metadata.st_mode) != 0o640):
            raise RuntimeError('Agent launch request must be Broker-owned mode 0640 in ptobench-share')
        if metadata.st_nlink != 1:
            raise RuntimeError('Agent launch request must have a single link')
        if metadata.st_size > MAX_REQUEST_BYTES:
            raise RuntimeError('Agent launch request exceeds the maximum size')
        chunks, size = [], 0
        while True:
            chunk = os.read(descriptor, min(65536, MAX_REQUEST_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise RuntimeError('Agent launch request exceeds the maximum size')
    finally:
        os.close(descriptor)
    try:
        request = json.loads(b''.join(chunks), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError('Agent launch request is malformed') from error
    if (not isinstance(request, dict) or set(request) != {'schema_version', 'argv', 'cwd'}
            or type(request['schema_version']) is not int or request['schema_version'] != 1):
        raise RuntimeError('Agent launch request has an invalid schema')
    command = request['argv']
    cwd = request['cwd']
    if (not isinstance(command, list) or not command
            or any(not isinstance(value, str) or not value or '\x00' in value for value in command)):
        raise RuntimeError('Agent launch request argv is invalid')
    if not Path(command[0]).is_absolute() or Path(command[0]).name != 'bwrap':
        raise RuntimeError('Agent launch request must execute the system Bubblewrap binary')
    if not isinstance(cwd, str) or not Path(cwd).is_absolute() or '\x00' in cwd:
        raise RuntimeError('Agent launch request working directory is invalid')
    return command, cwd


def dispatch(request_path: str):
    check_identity()
    command, cwd = read_request(request_path)
    os.chdir(cwd)
    os.umask(0o007)
    os.execve(command[0], command, {'PATH': os.defpath})
    raise RuntimeError('Bubblewrap exec unexpectedly returned')


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        print('ptobench-agent-dispatch: this entry accepts no arguments', file=sys.stderr)
        return 2
    request_path = os.environ.get(REQUEST_ENV)
    if not request_path:
        print('ptobench-agent-dispatch: request path is missing', file=sys.stderr)
        return 2
    try:
        dispatch(request_path)
    except Exception as error:
        print(f'ptobench-agent-dispatch: {error}', file=sys.stderr)
        return 126
    return 126


if __name__ == '__main__':
    raise SystemExit(main())
