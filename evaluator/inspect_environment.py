"""Read-only metadata probe in the actual Case Python/toolchain environment."""
import argparse
from datetime import datetime, timezone
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


def _lscpu_fields(stdout):
    try:
        rows = json.loads(stdout).get('lscpu', [])
    except (TypeError, ValueError, AttributeError):
        return {}
    return {row.get('field', '').rstrip(':'): row.get('data')
            for row in rows if isinstance(row, dict)}


def _npu_fields(stdout, allowed):
    fields = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition(':')
        key, value = key.strip(), value.strip()
        if separator and key in allowed and value:
            if value == 'NA':
                value = None
            elif value.replace(',', '').isdigit():
                value = int(value.replace(',', ''))
            fields[allowed[key]] = value
    return fields


def collect_hardware(profile, *, proc_root=Path('/proc')):
    """Return stable host hardware specifications for the public Agent context."""
    cpu_fields = {}
    lscpu = command(['lscpu', '--json'])
    if lscpu.get('exit_code') == 0:
        cpu_fields = _lscpu_fields(lscpu.get('stdout', ''))

    def integer(value):
        try:
            return int(str(value).split()[0])
        except (TypeError, ValueError, IndexError):
            return None

    sockets = integer(cpu_fields.get('Socket(s)'))
    cores_per_socket = integer(cpu_fields.get('Core(s) per socket'))
    memory_bytes = None
    try:
        for line in (Path(proc_root) / 'meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                memory_bytes = integer(line.partition(':')[2])
                if memory_bytes is not None:
                    memory_bytes *= 1024
                break
    except OSError:
        pass

    hardware = {
        'schema_version': 1,
        'scope': 'Host compute hardware; NPU entries describe the Case-assigned devices.',
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'system': {
            'architecture': platform.machine(),
            'os': platform.system(),
            'kernel': platform.release(),
            'cpu': {
                'model': cpu_fields.get('Model name') or platform.processor() or None,
                'vendor': cpu_fields.get('Vendor ID'),
                'logical_processors': integer(cpu_fields.get('CPU(s)')) or os.cpu_count(),
                'sockets': sockets,
                'cores_per_socket': cores_per_socket,
                'physical_cores': sockets * cores_per_socket if sockets and cores_per_socket else None,
                'threads_per_core': integer(cpu_fields.get('Thread(s) per core')),
                'numa_nodes': integer(cpu_fields.get('NUMA node(s)')),
                'cache': {key.lower().replace(' ', '_'): cpu_fields[key]
                          for key in ('L1d cache', 'L1i cache', 'L2 cache', 'L3 cache')
                          if cpu_fields.get(key)},
            },
            'memory': {'total_bytes': memory_bytes},
        },
        'accelerators': {'npu': {'status': 'not_configured', 'devices': []}},
    }

    if profile.get('platform') == 'cpu':
        return hardware

    runtime_ids = profile.get('runtime', {}).get('device_ids')
    device_ids = runtime_ids if isinstance(runtime_ids, list) and runtime_ids else [profile.get('device', 0)]
    device_ids = list(dict.fromkeys(device for device in device_ids
                                    if type(device) is int and device >= 0))
    board_fields = {'Product Name': 'product_name', 'Model': 'model',
                    'Manufacturer': 'manufacturer', 'Software Version': 'software_version',
                    'Firmware Version': 'firmware_version', 'PCIe Bus Info': 'pcie_bus_id',
                    'Chip Count': 'chip_count'}
    memory_fields = {'DDR Capacity(MB)': 'ddr_capacity_mb',
                     'DDR Clock Speed(MHz)': 'ddr_clock_mhz',
                     'HBM Capacity(MB)': 'hbm_capacity_mb',
                     'HBM Clock Speed(MHz)': 'hbm_clock_mhz'}
    common_fields = {'Aicore Count': 'ai_core_count'}
    chips = {}
    inventory = command(['npu-smi', 'info'])
    if inventory.get('exit_code') == 0:
        for line in inventory.get('stdout', '').splitlines():
            columns = [column.strip() for column in line.split('|')[1:-1]]
            if not columns:
                continue
            parts = columns[0].split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                chips[int(parts[0])] = parts[1]
    devices = []
    for device_id in device_ids:
        board = command(['npu-smi', 'info', '-t', 'board', '-i', str(device_id)])
        memory = command(['npu-smi', 'info', '-t', 'memory', '-i', str(device_id)])
        common_info = command(['npu-smi', 'info', '-t', 'common', '-i', str(device_id)])
        entry = {'device_id': device_id}
        board_ok = board.get('exit_code') == 0
        memory_ok = memory.get('exit_code') == 0
        if board.get('exit_code') == 0:
            entry.update(_npu_fields(board.get('stdout', ''), board_fields))
        if memory.get('exit_code') == 0:
            entry['memory'] = _npu_fields(memory.get('stdout', ''), memory_fields)
        if common_info.get('exit_code') == 0:
            entry['compute'] = _npu_fields(common_info.get('stdout', ''), common_fields)
        if device_id in chips:
            entry['chip_name'] = chips[device_id]
        has_board = bool(entry.get('product_name') or entry.get('model'))
        has_memory = bool(entry.get('memory'))
        entry['status'] = ('available' if board_ok and memory_ok and has_board and has_memory
                           else 'partial' if has_board or has_memory else 'unavailable')
        devices.append(entry)
    hardware['accelerators']['npu'] = {
        'status': 'available' if devices and all(item['status'] == 'available' for item in devices)
                  else 'partial' if any(item['status'] != 'unavailable' for item in devices)
                  else 'unavailable',
        'devices': devices,
    }
    return hardware


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
