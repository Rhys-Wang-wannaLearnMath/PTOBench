#!/usr/bin/python3
"""Run Pi's Bash tool in a child mount and PID namespace without Pi-private files."""
import json
import os
from pathlib import Path
import sys


ALLOWED_NAMES = {
    'PATH', 'LANG', 'LC_ALL', 'TZ', 'TERM', 'LD_LIBRARY_PATH', 'HOME', 'TMPDIR',
    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy',
    'NODE_OPTIONS', 'PTOBENCH_RPC_SOCKET', 'PTOBENCH_SANDBOX',
    'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE',
}


def main():
    config = json.loads((Path(__file__).resolve().parent / 'shell.json').read_text())
    bwrap = config['bwrap']
    argv = [
        bwrap, '--die-with-parent', '--unshare-user', '--unshare-pid', '--unshare-ipc',
        '--unshare-uts', '--cap-drop', 'ALL', '--ro-bind', '/', '/', '--proc', '/proc',
        '--dev', '/dev', '--tmpfs', '/tmp', '--tmpfs', '/home/ptobench',
        '--bind', '/ptobench/candidates', '/ptobench/candidates',
        '--tmpfs', '/ptobench/pi-private',
        '--tmpfs', '/ptobench/agent-sessions',
    ]
    environment = {key: value for key, value in os.environ.items()
                   if key in ALLOWED_NAMES or key.startswith(('ASCEND_', 'HCCL_'))}
    environment.update(HOME='/home/ptobench', TMPDIR='/tmp', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONNOUSERSITE='1', PTOBENCH_SANDBOX='1')
    for key, value in sorted(environment.items()):
        argv.extend(['--setenv', key, value])
    argv.extend(['--chdir', os.getcwd(), '--', '/bin/bash', *sys.argv[1:]])
    os.execve(bwrap, argv, {'PATH': os.defpath})


if __name__ == '__main__':
    main()
