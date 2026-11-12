#!/usr/bin/python3
"""Run Pi's Bash tool in a child mount and PID namespace on the same Workspace."""
import json
import os
from pathlib import Path
import sys


ALLOWED_NAMES = {
    'PATH', 'LANG', 'LC_ALL', 'TZ', 'TERM', 'LD_LIBRARY_PATH', 'HOME', 'TMPDIR',
    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy',
    'NODE_OPTIONS', 'PTOBENCH_RPC_SOCKET', 'PTOBENCH_CANNBOT_SOCKET', 'PTOBENCH_SANDBOX',
    'PTOBENCH_AGENT_DISPATCH_ID', 'PTOBENCH_AGENT_PARENT_TOOL_CALL_ID',
    'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE',
}


def sandbox_argv(command, *, cwd, environment, config):
    workspace_root = Path(config['workspace_root']).resolve(strict=True)
    working_directory = Path(cwd).resolve(strict=True)
    try:
        working_directory.relative_to(workspace_root)
    except ValueError as error:
        raise RuntimeError('Pi Shell must start inside the current Session Workspace') from error
    argv = [
        config['bwrap'], '--die-with-parent', '--unshare-user', '--unshare-pid', '--unshare-ipc',
        '--unshare-uts', '--cap-drop', 'ALL', '--ro-bind', '/', '/', '--proc', '/proc',
        '--dev', '/dev', '--tmpfs', '/tmp', '--tmpfs', '/home/ptobench',
        '--bind', str(workspace_root), str(workspace_root),
        '--tmpfs', '/ptobench/pi-config', '--tmpfs', '/ptobench/pi-tools',
        '--tmpfs', '/ptobench/agent-sessions',
    ]
    # The tools tmpfs also removes the CANNBot extension file from this view.
    # The Broker, Gateway, and CLI sockets remain at their individually mounted paths.
    filtered = {key: value for key, value in environment.items()
                if key in ALLOWED_NAMES or key.startswith(('ASCEND_', 'HCCL_'))}
    filtered.update(HOME='/home/ptobench', TMPDIR='/tmp', PYTHONDONTWRITEBYTECODE='1',
                    PYTHONNOUSERSITE='1', PTOBENCH_SANDBOX='1')
    for key, value in sorted(filtered.items()):
        argv.extend(['--setenv', key, value])
    argv.extend(['--chdir', str(working_directory), '--', '/bin/bash', *map(str, command)])
    return argv


def main():
    config = json.loads((Path(__file__).resolve().parent / 'shell.json').read_text())
    argv = sandbox_argv(sys.argv[1:], cwd=os.getcwd(), environment=os.environ, config=config)
    os.execve(config['bwrap'], argv, {'PATH': os.defpath})


if __name__ == '__main__':
    main()
