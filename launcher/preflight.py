"""Read-only readiness checks before creating any Session."""
import os
from pathlib import Path
import shutil
import subprocess

import common
import session
from launcher.methods import MethodError, load_method, call
from launcher.config import ConfigError, load_experiment


def preflight(path):
    experiment = load_experiment(path)
    method = load_method(experiment.mode)
    for old in ('bench', 'evaluator/bench.py'):
        if os.path.lexists(common.BENCHMARK_ROOT / old):
            raise ConfigError('old benchmark entry still exists')
    command = common.BENCHMARK_ROOT / 'ptobench'
    if not command.is_file() or not os.access(command, os.X_OK):
        raise MethodError('ptobench is not executable', 'infrastructure_error')
    _, _, case, environment = session.preflight_case(experiment.case)
    if environment['platform'] != 'cpu':
        python = common.profile_path(environment.get('python', ''))
        if not python.is_file() or not os.access(python, os.X_OK):
            raise MethodError('Case Python is unavailable', 'infrastructure_error')
        ptoas_root = environment.get('env', {}).get('PTOAS_ROOT')
        if ptoas_root and not (common.profile_path(ptoas_root) / 'bin/ptoas').is_file():
            raise MethodError('Case PTOAS compiler is unavailable', 'infrastructure_error')
        for script in environment.get('setup_scripts', []):
            # Host toolchain installations commonly use a version symlink.
            # Canonical Case/output paths retain S1's no-follow policy.
            session._asset_file(common.profile_path(script).resolve())
        try:
            exports = common.load_profile_environment(environment)
            common.validate_toolchain(environment, exports)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            raise MethodError(f'Case compiler environment is unavailable: {error}',
                              'infrastructure_error') from error
        if shutil.which('npu-smi') is None:
            raise MethodError('npu-smi is unavailable', 'infrastructure_error')
        try:
            device = environment.get('device')
            if type(device) is not int or device < 0:
                raise MethodError('Case NPU device is invalid', 'infrastructure_error')
            result = subprocess.run(['npu-smi', 'info', '-t', 'health', '-i', str(device)], shell=False, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MethodError('NPU query failed', 'infrastructure_error') from error
        if result.returncode != 0 or b'OK' not in result.stdout:
            raise MethodError('NPU is unavailable', 'infrastructure_error')
    ready = call(method, 'preflight', {'schema_version': 1, 'experiment': experiment.to_dict()})
    return experiment, method, ready, case
