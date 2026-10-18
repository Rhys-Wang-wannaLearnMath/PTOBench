"""Read-only metadata probe in the actual Case Python/toolchain environment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import common


def command(argv, cwd=None):
    try:
        result = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=10)
        return {'argv': argv, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'argv': argv, 'error': type(error).__name__}


def git_source(path):
    return {'path': str(path), 'revision': command(['git', 'rev-parse', 'HEAD'], path),
            'working_tree': command(['git', 'status', '--porcelain'], path)}


def collect(case):
    packages = {}
    for name in ('pypto', 'simpler', 'torch', 'PyYAML'):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {'installed': False}
            continue
        direct = distribution.read_text('direct_url.json')
        packages[name] = {'version': distribution.version, 'root': str(distribution.locate_file('')),
                          'direct_url': json.loads(direct) if direct else None}
        # Package version strings are insufficient for local runtime builds.
        # Hash installed PyPTO/runtime files, without importing or executing them.
        if name in ('pypto', 'simpler'):
            files = {}
            for relative in distribution.files or []:
                path = distribution.locate_file(relative)
                if path.suffix in ('.py', '.so', '.json', '.pin') and path.is_file():
                    files[str(relative)] = common.sha256_file(path)
            packages[name]['installed_files_sha256'] = files
    profile = common._environment_profile(case)
    result = {'schema_version': 1, 'python': {'executable': sys.executable, 'version': sys.version},
              'host': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
              'case_environment': profile, 'packages': packages,
              'sources': {'ptobench': git_source(common.BENCHMARK_ROOT), 'pypto_lib': git_source(common.REPO_ROOT)},
              'environment': {key: os.environ.get(key) for key in (
                  'ASCEND_HOME_PATH', 'PTOAS_ROOT', 'PTO_ISA_ROOT', 'LD_LIBRARY_PATH', 'PYTHONPATH',
                  'ASCEND_RT_VISIBLE_DEVICES', 'PTOBENCH_REPO_ROOT', 'PTOBENCH_SEALED_ROOT')}}
    if profile['platform'] != 'cpu':
        result['hardware_query'] = command(['npu-smi', 'info'])
        result['driver_version'] = command(['npu-smi', 'info', '-t', 'info'])
        roots = [Path(os.environ[key]) for key in ('ASCEND_HOME_PATH', 'PTOAS_ROOT') if os.environ.get(key)]
        versions = {}
        for root in roots:
            for pattern in ('version.info', '*version*.txt', '*/version.info', '*/version.cfg'):
                for path in root.glob(pattern):
                    if path.is_file():
                        versions[str(path)] = path.read_text(errors='replace')
        result['toolchain_version_files'] = versions
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    _, _, case = common.resolve_case(args.case)
    if common._environment_profile(case).get('python'):
        common.bootstrap_environment(case)
    common.write_json(args.result, collect(case))


if __name__ == '__main__':
    main()
