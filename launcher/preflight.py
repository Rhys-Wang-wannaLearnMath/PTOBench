"""Read-only readiness checks before creating any Session."""
import os
from pathlib import Path
import shutil
import subprocess

import common
import session
from launcher.adapters import BackendUnavailable, get_adapter
from launcher.config import ConfigError, load_experiment, load_profile


def preflight(path):
    experiment = load_experiment(path)
    profile = load_profile(experiment.optimizer['backend'], experiment.optimizer['profile'])
    for old in ('bench', 'evaluator/bench.py'):
        if os.path.lexists(common.BENCHMARK_ROOT / old):
            raise ConfigError('old benchmark entry still exists')
    command = common.BENCHMARK_ROOT / 'ptobench'
    if not command.is_file() or not os.access(command, os.X_OK):
        raise BackendUnavailable('ptobench is not executable')
    _, _, case, environment = session.preflight_case(experiment.case)
    if environment['platform'] != 'cpu':
        python = common.profile_path(environment.get('python', ''))
        if not python.is_file() or not os.access(python, os.X_OK):
            raise BackendUnavailable('Case Python is unavailable')
        ptoas_root = environment.get('env', {}).get('PTOAS_ROOT')
        if ptoas_root and not (common.profile_path(ptoas_root) / 'bin/ptoas').is_file():
            raise BackendUnavailable('Case PTOAS compiler is unavailable')
        for script in environment.get('setup_scripts', []):
            # Host toolchain installations commonly use a version symlink.
            # Canonical Case/output paths retain S1's no-follow policy.
            session._asset_file(common.profile_path(script).resolve())
        if shutil.which('npu-smi') is None:
            raise BackendUnavailable('npu-smi is unavailable')
        try:
            device = environment.get('device')
            if type(device) is not int or device < 0:
                raise BackendUnavailable('Case NPU device is invalid')
            result = subprocess.run(['npu-smi', 'info', '-t', 'health', '-i', str(device)], shell=False, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BackendUnavailable('NPU query failed') from error
        if result.returncode != 0 or b'OK' not in result.stdout:
            raise BackendUnavailable('NPU is unavailable')
    adapter = get_adapter(experiment.optimizer['backend'])
    info = adapter.preflight(profile)
    return experiment, profile, adapter, info, case
