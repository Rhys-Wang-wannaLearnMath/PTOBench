from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATOR_DIR))
import inspect_environment


class PublicHardwareSummaryTest(unittest.TestCase):
    def test_collects_host_and_case_device_specs_without_serials(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        proc_root = Path(temporary.name)
        (proc_root / 'meminfo').write_text('MemTotal: 8192 kB\n')

        def run(argv, cwd=None):
            if argv == ['lscpu', '--json']:
                return {'exit_code': 0, 'stdout': json.dumps({'lscpu': [
                    {'field': 'Model name:', 'data': 'Example CPU'},
                    {'field': 'Vendor ID:', 'data': 'Example Vendor'},
                    {'field': 'CPU(s):', 'data': '16'},
                    {'field': 'Socket(s):', 'data': '2'},
                    {'field': 'Core(s) per socket:', 'data': '4'},
                    {'field': 'Thread(s) per core:', 'data': '2'},
                    {'field': 'NUMA node(s):', 'data': '2'},
                    {'field': 'L2 cache:', 'data': '8 MiB'},
                ]})}
            if argv == ['npu-smi', 'info']:
                return {'exit_code': 0, 'stdout': (
                    '| NPU   Name                | Health        |\n'
                    '| 3     Example910B         | OK            |\n')}
            if argv == ['npu-smi', 'info', '-t', 'board', '-i', '3']:
                return {'exit_code': 0, 'stdout': (
                    'Product Name : Example NPU\nManufacturer : Example\n'
                    'Software Version : 25.5.0\nSerial Number : PRIVATE\n')}
            if argv == ['npu-smi', 'info', '-t', 'memory', '-i', '3']:
                return {'exit_code': 0, 'stdout': (
                    'HBM Capacity(MB) : 32768\nHBM Clock Speed(MHz) : 1600\n')}
            if argv == ['npu-smi', 'info', '-t', 'common', '-i', '3']:
                return {'exit_code': 0, 'stdout': (
                    'Aicore Count : 24\nTemperature(C) : 45\n')}
            raise AssertionError(argv)

        with mock.patch.object(inspect_environment, 'command', side_effect=run):
            summary = inspect_environment.collect_hardware({
                'name': 'test', 'platform': 'a2a3', 'device': 3,
                'runtime': {'device_ids': [3]},
            }, proc_root=proc_root)

        self.assertEqual(summary['system']['cpu']['model'], 'Example CPU')
        self.assertEqual(summary['system']['cpu']['physical_cores'], 8)
        self.assertEqual(summary['accelerators']['npu']['devices'][0]['device_id'], 3)
        self.assertEqual(
            summary['accelerators']['npu']['devices'][0]['memory']['hbm_capacity_mb'], 32768,
        )
        self.assertEqual(summary['accelerators']['npu']['devices'][0]['chip_name'], 'Example910B')
        self.assertEqual(summary['accelerators']['npu']['devices'][0]['compute']['ai_core_count'], 24)
        self.assertNotIn('PRIVATE', json.dumps(summary))
        self.assertEqual(summary['system']['memory']['total_bytes'], 8 * 1024 * 1024)
        self.assertNotIn('Temperature', json.dumps(summary))


if __name__ == '__main__':
    unittest.main()
