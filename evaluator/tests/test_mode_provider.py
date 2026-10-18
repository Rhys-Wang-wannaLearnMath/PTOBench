"""Mode-local Pi configuration, actual launcher processes, and secret-free records."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from action_test_support import SOURCE_BENCHMARK
import test_launcher
import test_cannbot
import integrity

sys.path.insert(0, str(SOURCE_BENCHMARK))
from modes._shared import provider, recording, provenance


ENV = '''LLM_PROVIDER=test-provider
LLM_BASE_URL=http://127.0.0.1:8765/v1
LLM_API=openai-completions
LLM_MODEL=test-model
LLM_REASONING=high
LLM_API_KEY=local-provider-test-key
'''


class ProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mode = self.root / 'mode'
        self.mode.mkdir()
        (self.mode / '.env').write_text(ENV)
        self.profile = {'backend': 'pi', 'env': {}, 'model': 'old/old-model', 'reasoning': 'low',
                        'argv': ['{repo_root}/benchmarks/mig_workflow/agent_provider/launch.py',
                                 'pi', '--print', '--provider=old', '--model', 'old-model', '{prompt}']}

    def test_overrides_isolated_configs_and_secret_free_archives(self):
        profile = recording.prepare_profile(provider.configure(self.profile, self.mode))
        self.assertEqual(profile['argv'][0], 'pi')
        self.assertEqual(recording.option(profile['argv'], '--provider'), 'test-provider')
        self.assertEqual(recording.option(profile['argv'], '--model'), 'test-model')
        self.assertEqual(recording.option(profile['argv'], '--thinking'), 'high')
        first = provider.prepare(profile, self.root / 'run-one')
        second = provider.prepare(profile, self.root / 'run-two')
        self.assertNotEqual(first['env']['PI_CODING_AGENT_DIR'], second['env']['PI_CODING_AGENT_DIR'])
        directory = Path(first['env']['PI_CODING_AGENT_DIR'])
        config = json.loads((directory / 'models.json').read_text())['providers']['test-provider']
        self.assertEqual(config['baseUrl'], 'http://127.0.0.1:8765/v1')
        self.assertEqual(config['api'], 'openai-completions')
        self.assertEqual(config['apiKey'], '$PYPTO_PROVIDER_API_KEY')
        snapshot = provenance.freeze_agent(self.root / 'run-one', first, self.mode, {**os.environ, **first['env']})
        self.assertEqual(snapshot['config_directory'], str(directory))
        self.assertEqual(provider.credentials(first)['PYPTO_PROVIDER_API_KEY'], 'local-provider-test-key')
        manifest = integrity.capture_protection(self.root / 'protection', {
            'mode': integrity.ProtectionRoot(self.mode, 'fixed'),
            'configuration': integrity.ProtectionRoot(self.root / 'run-one/agent/config', 'agent_archive', readonly=True)})
        self.assertNotIn('.env', manifest['roots']['mode']['entries'])
        # Pi creates native caches next to models.json during normal startup.
        (directory / 'models-store.json').write_text('{}')
        self.assertTrue(integrity.check_only(self.root / 'protection').passed)
        for p in self.root.rglob('*'):
            if p.is_file() and p.name != '.env':
                self.assertNotIn(b'local-provider-test-key', p.read_bytes(), str(p))
        self.assertNotIn('local-provider-test-key', json.dumps(profile))

    def test_absent_empty_env_and_other_backends_preserve_profiles(self):
        self.assertIs(provider.configure({**self.profile, 'backend': 'codex'}, self.mode)['env'], self.profile['env'])
        (self.mode / '.env').write_text('# no overrides\n')
        self.assertIs(provider.configure(self.profile, self.mode), self.profile)
        (self.mode / '.env').unlink()
        self.assertIs(provider.configure(self.profile, self.mode), self.profile)

    def test_literal_values_and_errors_do_not_expose_keys(self):
        (self.mode / '.env').write_text('export LLM_API_KEY="literal $(command) $VARIABLE" # comment\n')
        self.assertEqual(provider.read_env(self.mode / '.env')['LLM_API_KEY'], 'literal $(command) $VARIABLE')
        for source in ('LLM_API_KEY="private-test-key', 'LLM_API_KEY=one\nLLM_API_KEY=two', 'UNKNOWN=value'):
            with self.subTest(source=source):
                (self.mode / '.env').write_text(source)
                with self.assertRaises(provider.ConfigError) as error:
                    provider.read_env(self.mode / '.env')
                self.assertNotIn('private-test-key', str(error.exception))

    def test_blank_key_uses_inherited_credentials(self):
        (self.mode / '.env').write_text(ENV.replace('local-provider-test-key', ''))
        with mock.patch.dict(os.environ, {'PYPTO_PROVIDER_API_KEY': 'inherited-test-key'}):
            profile = provider.configure(self.profile, self.mode)
            self.assertEqual(provider.credentials(profile)['PYPTO_PROVIDER_API_KEY'], 'inherited-test-key')


CHECK_NATIVE = '''
if '--version' not in sys.argv and '--help' not in sys.argv:
    provider_dir=Path(os.environ['PI_CODING_AGENT_DIR'])
    provider_models=json.loads((provider_dir/'models.json').read_text())['providers']
    assert provider_models['test-provider']['baseUrl']=='http://127.0.0.1:8765/v1'
    assert provider_models['test-provider']['api']=='openai-completions'
    assert os.environ['PYPTO_PROVIDER_API_KEY']=='local-provider-test-key'
    assert sys.argv[sys.argv.index('--model')+1]=='test-model'
    assert sys.argv[sys.argv.index('--provider')+1]=='test-provider'
'''


class ProviderLauncherTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_launcher.LauncherTests('test_four_adapters_real_cli_all_five_operations')
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def assert_private_records(self, root):
        profile = json.loads((root / 'profile.json').read_text())
        self.assertEqual(profile['model'], 'test-provider/test-model')
        self.assertEqual(profile['env']['PI_CODING_AGENT_DIR'], str(root / 'agent/pi'))
        manifest = json.loads((root / 'protection/manifest.json').read_text())
        self.assertNotIn('basic/.env', manifest['roots']['public_modes']['entries'])
        self.assertNotIn('cannbot/.env', manifest['roots']['public_modes']['entries'])
        for path in root.rglob('*'):
            if path.is_file():
                self.assertNotIn(b'local-provider-test-key', path.read_bytes(), str(path))

    def test_basic_mode_env_reaches_native_agent_and_not_records(self):
        fixture = self.fixture
        fixture.launcher_config('pi', mode='provider_test')
        (fixture.bench / 'modes/basic/.env').write_text(ENV)
        fixture.tool.write_text('#!' + fixture.python + '\n' + test_launcher.FAKE_TOOL.replace(
            "if '--version' in sys.argv:", CHECK_NATIVE + "\nif '--version' in sys.argv:", 1))
        process, result = fixture.launch()
        self.assertEqual(process.returncode, 0, (result, process.stderr.decode()))
        root, _ = fixture.assert_final(result, 1)
        self.assert_private_records(root)

    def test_cannbot_mode_env_reaches_every_task_stage(self):
        node = test_cannbot.shutil.which('node')
        if not node or int(subprocess.check_output([node, '--version']).decode().split('.')[0][1:]) < 22:
            self.skipTest('Pi TypeScript bridge requires Node 22+')
        fixture = self.fixture
        fixture.launcher_config('pi', budget=6, experiment_mode='cannbot', node=node)
        (fixture.bench / 'modes/cannbot/.env').write_text(ENV)
        fixture.tool.write_text('#!' + fixture.python + '\n' + test_cannbot.FAKE_NATIVE.replace(
            "if '--version' in sys.argv:", CHECK_NATIVE + "\nif '--version' in sys.argv:", 1))
        process, result = fixture.launch()
        self.assertEqual(process.returncode, 0, (result, process.stderr.decode()))
        root, _ = fixture.assert_final(result, 4)
        states = [json.loads(path.read_text()) for path in (root / 'method/dispatches').glob('*/dispatch.json')]
        self.assertEqual(len(states), 3)
        self.assertTrue(all(state['status'] == 'completed' for state in states))
        self.assert_private_records(root)


if __name__ == '__main__':
    unittest.main()
