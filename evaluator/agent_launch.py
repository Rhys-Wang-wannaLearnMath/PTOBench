"""Fail-closed preparation for the unprivileged Bubblewrap Agent dispatcher."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import secrets
import shutil
import stat
import subprocess
import sys


DISPATCHER = Path(__file__).resolve().parents[1] / 'ops/ptobench-agent-dispatch.py'
REQUEST_ENV = 'PTOBENCH_AGENT_LAUNCH_REQUEST'
MAX_REQUEST_BYTES = 1024 * 1024


def _host_credentials():
    status = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, separator, value = line.partition(':')
        if separator and key in {'Uid', 'Gid', 'Groups', 'CapEff', 'NoNewPrivs'}:
            status[key] = value.strip()
    uid, euid, _ = os.getresuid()
    gid, egid, _ = os.getresgid()
    if uid != euid or gid != egid:
        raise RuntimeError('Agent launch requires matching real and effective UID/GID')
    groups = os.getgroups()
    status_groups = [int(value) for value in status['Groups'].split()]
    if groups != status_groups:
        raise RuntimeError('host supplementary groups differ from /proc/self/status')
    return {'uid': uid, 'gid': gid, 'groups': groups,
            'cap_eff': int(status['CapEff'], 16), 'no_new_privs': status['NoNewPrivs'] == '1'}


def parse_subid_ranges(text: str, account: str, uid: int):
    ranges = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split(':')
        if len(fields) != 3:
            raise RuntimeError(f'invalid subordinate-ID entry at line {line_number}')
        owner, start_text, count_text = fields
        if owner not in {account, str(uid)}:
            continue
        try:
            start, count = int(start_text), int(count_text)
        except ValueError as error:
            raise RuntimeError(f'invalid subordinate-ID range at line {line_number}') from error
        if start < 0 or count < 1 or start + count > 2**32:
            raise RuntimeError(f'invalid subordinate-ID range at line {line_number}')
        ranges.append((start, count))
    return ranges


def mapping_plan(*, uid: int, gid: int, uid_ranges, gid_ranges):
    if not uid_ranges or not gid_ranges:
        raise RuntimeError('usable subordinate UID/GID ranges are required for Agent isolation')
    if uid <= 0 or gid <= 0 or uid >= 2**32 - 1 or gid >= 2**32 - 1:
        raise RuntimeError('Agent isolation requires a non-root Broker UID/GID with room for namespace IDs')
    agent_uid = next((start + offset for start, count in uid_ranges
                      for offset in range(count) if start + offset != uid), None)
    agent_gid = next((start + offset for start, count in gid_ranges
                      for offset in range(count) if start + offset != gid), None)
    if agent_uid is None or agent_gid is None:
        raise RuntimeError('subordinate UID/GID ranges overlap the Broker identity')
    agent_ns_uid = uid + 1
    if agent_ns_uid >= 2**32 - 1:
        raise RuntimeError('Broker UID leaves no safe namespace ID for the Agent')
    return {'agent_uid': agent_uid, 'agent_gid': agent_gid,
            'agent_ns_uid': agent_ns_uid,
            'uid_map': sorted([(uid, uid, 1), (agent_ns_uid, agent_uid, 1)]),
            'gid_map': [(0, agent_gid, 1), (gid, gid, 1)],
            'broker_ns_uid': uid, 'broker_ns_gid': gid}


def _setpriv_clears_groups(setpriv_path, uid, gid):
    if not setpriv_path:
        return False
    probe = ('import json,os; print(json.dumps({"uid":os.geteuid(),"gid":os.getegid(),'
             '"groups":os.getgroups()}))')
    try:
        result = subprocess.run(
            [setpriv_path, f'--reuid={uid}', f'--regid={gid}', '--clear-groups', '--',
             sys.executable, '-c', probe], capture_output=True, text=True, timeout=5,
            env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
        observed = json.loads(result.stdout)
        return (result.returncode == 0 and observed == {'uid': uid, 'gid': gid, 'groups': []})
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return False


def _trusted_binary(path):
    if not path:
        return None
    candidate = Path(path)
    try:
        candidate = candidate.resolve(strict=True)
        metadata = candidate.stat()
    except OSError:
        return None
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0
            or metadata.st_mode & 0o022 or not os.access(candidate, os.X_OK)):
        return None
    for parent in candidate.parents:
        try:
            parent_stat = parent.stat()
        except OSError:
            return None
        if parent_stat.st_uid != 0 or parent_stat.st_mode & 0o022:
            return None
    return str(candidate.resolve())


def _trusted_launcher(path):
    return _trusted_binary(path)


def _request_directory(path):
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077):
        raise RuntimeError('Agent launcher request directory must be caller-owned mode 0700')
    path.chmod(0o700)
    return path


def _write_request(command, cwd, request_dir, credentials):
    directory = _request_directory(request_dir)
    request = {'schema_version': 1, 'command': list(map(str, command)), 'cwd': str(Path(cwd).absolute()),
               'uid': credentials['uid'], 'gid': credentials['gid'],
               'groups': list(credentials['groups']), 'environment': {'PATH': os.defpath}}
    path = directory / f'{secrets.token_hex(16)}.json'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        payload = (json.dumps(request, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if len(payload) > MAX_REQUEST_BYTES:
            raise RuntimeError('Agent launcher request is too large')
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return str(path)


def prepare(command: list[str], cwd: Path, request_dir: Path, *, path: str):
    credentials = _host_credentials()
    if credentials.get('no_new_privs', False):
        raise RuntimeError(
            'Agent launch refused: host NoNewPrivs=1 prevents newuidmap/newgidmap from establishing '
            'the required subordinate identity map; launch the Broker before NoNewPrivs is enabled')
    setpriv_path = _trusted_binary(shutil.which('setpriv'))
    sudo_path = _trusted_binary(shutil.which('sudo'))
    if credentials['cap_eff'] == 0 and not credentials['groups']:
        launcher = [sys.executable, str(DISPATCHER)]
    elif (credentials['cap_eff'] == 0 and setpriv_path
          and _setpriv_clears_groups(setpriv_path, credentials['uid'], credentials['gid'])):
        launcher = [setpriv_path, f"--reuid={credentials['uid']}", f"--regid={credentials['gid']}",
                    '--clear-groups', '--', sys.executable, str(DISPATCHER)]
    else:
        trusted = _trusted_launcher(path)
        if trusted is None or sudo_path is None:
            reason = 'supplementary groups' if credentials['groups'] else 'effective capabilities'
            raise RuntimeError(
                f'Agent launch refused: UID/GID={credentials["uid"]}/{credentials["gid"]}, '
                f'supplementary groups={credentials["groups"]}, CapEff={credentials["cap_eff"]:#x}, '
                f'NoNewPrivs={int(credentials.get("no_new_privs", False))}; '
                f'{reason} cannot be cleared by '
                'an unprivileged setpriv path. Install the fixed root-owned launcher and exact '
                'per-user no-argument sudoers rule, then verify it with sudo -n -l.')
        launcher = [sudo_path, '-n', trusted]
    request_path = _write_request(command, cwd, request_dir, credentials)
    return launcher, {'PATH': os.defpath, REQUEST_ENV: request_path}
