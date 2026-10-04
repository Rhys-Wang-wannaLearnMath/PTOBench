"""Renamed standalone checkout, external assets, and host CLI regressions."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest import mock

import test_launcher as launcher_tests
import common

SOURCE_BENCHMARK = launcher_tests.SOURCE_BENCHMARK


class PortabilityTests(unittest.TestCase):
    def test_external_checkout_imports_without_inherited_pythonpath(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = root / 'PTOBench'
            (bench / 'evaluator').mkdir(parents=True)
            shutil.copyfile(SOURCE_BENCHMARK / 'evaluator/common.py', bench / 'evaluator/common.py')
            source = root / 'external'
            (source / 'golden').mkdir(parents=True)
            (source / 'golden/__init__.py').write_text('VALUE = "selected checkout"\n')
            environment = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            environment['PTOBENCH_REPO_ROOT'] = str(source)
            output = subprocess.check_output(
                [sys.executable, '-c', 'import sys; sys.path.insert(0, sys.argv[1]); '
                 'import common, golden; print(golden.VALUE)', str(bench / 'evaluator')],
                cwd='/', env=environment, text=True,
            )
            self.assertEqual(output.strip(), 'selected checkout')

    def test_profile_python_preserves_virtual_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            venv.EnvBuilder(symlinks=True, with_pip=False).create(root / '.venv')
            with mock.patch.object(common, 'REPO_ROOT', root):
                python = common.profile_path('.venv/bin/python')
                prefix = subprocess.check_output(
                    [str(python), '-c', 'import sys; print(sys.prefix)'], text=True,
                ).strip()
                self.assertEqual(Path(prefix), root / '.venv')

    def test_root_detection_and_toolchain_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bench = root / 'benchmarks/PTOBench'
            (bench / 'evaluator').mkdir(parents=True)
            shutil.copyfile(SOURCE_BENCHMARK / 'evaluator/common.py', bench / 'evaluator/common.py')

            def load():
                spec = importlib.util.spec_from_file_location('portable_common', bench / 'evaluator/common.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            clean = {key: value for key, value in os.environ.items() if not key.startswith('PTOBENCH_')}
            with mock.patch.dict(os.environ, clean, clear=True):
                module = load()
                self.assertEqual(module.REPO_ROOT, bench)
                self.assertEqual(module.profile_values()['ptoas_root'], '')
                (root / 'golden').mkdir()
                self.assertEqual(load().REPO_ROOT, root)
                external = root / 'external'
                os.environ.update(PTOBENCH_REPO_ROOT=str(external),
                                  PTOBENCH_SEALED_ROOT=str(root / 'assets'),
                                  PTOBENCH_PTOAS_ROOT=str(root / 'compiler'),
                                  PTOBENCH_CANN_ROOT=str(root / 'cann'),
                                  PTOBENCH_MINIFORGE_ROOT=str(root / 'conda'))
                module = load()
                self.assertEqual(module.REPO_ROOT, external)
                self.assertEqual(module.sealed_root(), root / 'assets')
                for placeholder, expected in [('repo_root', external), ('benchmark_root', bench),
                                               ('ptoas_root', root / 'compiler'), ('cann_root', root / 'cann'),
                                               ('miniforge_root', root / 'conda')]:
                    self.assertEqual(module.profile_path('{' + placeholder + '}/bin'), expected / 'bin')

    def test_standalone_sessions_external_assets_and_backend_fallback(self):
        # Reuse the controlled CPU Case, but launch the real CLI from / after
        # moving the entire control tree to PTOBench and assets elsewhere.
        fixture = launcher_tests.LauncherTests('test_four_adapters_real_cli_all_five_operations')
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        bench = fixture.repo / 'PTOBench'
        fixture.bench.rename(bench)
        fixture.bench = bench
        common.BENCHMARK_ROOT = bench
        assets = fixture.repo / 'assets/sealed_assets'
        assets.parent.mkdir()
        (bench / 'sealed_assets').rename(assets)
        tools = fixture.repo / 'tools'
        tools.mkdir()
        fixture.env['PATH'] = str(tools) + os.pathsep + fixture.env['PATH']
        for backend in ('pi', 'opencode'):
            shutil.copy2(fixture.tool, tools / backend)
        shutil.copytree(fixture.repo / 'golden', bench / 'golden')
        fixture.env.pop('PTOBENCH_REPO_ROOT', None)
        fixture.env['PTOBENCH_SEALED_ROOT'] = str(assets)

        for backend, external_source in [('pi', False), ('opencode', True)]:
            with self.subTest(backend=backend, external_source=external_source):
                if external_source:
                    fixture.env['PTOBENCH_REPO_ROOT'] = str(fixture.repo)
                    common.REPO_ROOT = fixture.repo
                else:
                    common.REPO_ROOT = bench
                fixture.launcher_config(backend, mode='all')
                profile = bench / f'launcher/profiles/{backend}/default.yaml'
                shutil.copyfile(SOURCE_BENCHMARK / f'launcher/profiles/{backend}/default.yaml', profile)
                with mock.patch.dict(os.environ, {'PTOBENCH_SEALED_ROOT': str(assets)}):
                    process, value = fixture.launch()
                    self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
                    root, final = fixture.assert_final(value, 4)
                    self.assertEqual(final['finish_actions_used'], 1)
                    record = json.loads((root / 'session.json').read_text())
                    self.assertEqual(Path(record['sealed_dir']).parents[1], assets)
                    actual = json.loads((root / 'profile.json').read_text())
                    self.assertEqual(actual['argv'][0], backend)
                    self.assertNotIn('model', actual)


if __name__ == '__main__':
    unittest.main()
