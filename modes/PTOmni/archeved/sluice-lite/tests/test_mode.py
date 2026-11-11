"""Software checks for the independent SLUICE Lite mode."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

BENCHMARK = Path(__file__).resolve().parents[3]
MODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK), str(BENCHMARK / 'evaluator'), str(MODE)]

from launcher import methods
import runner


class SluiceLiteTests(unittest.TestCase):
    def test_launcher_loads_hyphenated_mode_and_rejects_bad_names(self):
        loaded = methods.load_method('sluice-lite')
        self.assertEqual(loaded['id'], 'sluice-lite')
        self.assertEqual(Path(loaded['path']), MODE / 'runner.py')
        for name in ('-sluice', 'sluice-', 'sluice--lite', '../sluice-lite'):
            with self.subTest(name=name), self.assertRaises(methods.MethodError):
                methods.load_method(name)

    def test_resource_snapshots_include_knowledge_and_skills(self):
        runner._validate_sources()
        knowledge = {path.relative_to(runner.KNOWLEDGE_SOURCE).as_posix()
                     for path in runner._resource_files(runner.KNOWLEDGE_SOURCE)}
        self.assertIn('index/search.json', knowledge)
        self.assertIn('cards/task_granularity_and_dispatch_overhead.json', knowledge)
        skills = runner._resource_files(runner.SKILLS_SOURCE)
        self.assertTrue(any(path.name == 'SKILL.md' for path in skills))

    def test_smoke_profile_selects_deepseek_v41_flash_low(self):
        profile = runner.load_profile('pi', 'deepseek-flash-low', root=MODE / 'profiles')
        self.assertEqual(profile['model'], 'provider-default/deepseek-flash')
        self.assertEqual(profile['reasoning'], 'low')
        self.assertIn('--thinking', profile['argv'])
        self.assertEqual(profile['argv'][profile['argv'].index('--thinking') + 1], 'low')

    def test_preflight_uses_local_pi_profile(self):
        profile = {'argv': ['pi', '{prompt}'], 'sha256': 'profile'}
        adapter = mock.Mock()
        adapter.preflight.return_value.to_dict.return_value = {'version': 'test'}
        request = {'experiment': {'optimizer': {'backend': 'pi', 'profile': 'deepseek-flash-low'}}}
        with mock.patch.object(runner, '_validate_sources'), \
             mock.patch.object(runner, 'load_profile', return_value=profile) as load_profile, \
             mock.patch.object(runner.provider, 'configure', return_value=profile) as configure, \
             mock.patch.object(runner.recording, 'prepare_profile', side_effect=lambda value: value), \
             mock.patch.object(runner, 'get_adapter', return_value=adapter):
            runner.preflight(request)
        load_profile.assert_called_once_with('pi', 'deepseek-flash-low', root=MODE / 'profiles')
        configure.assert_called_once_with(profile, MODE, prefer_profile=True)

    def test_prepare_archives_resources_and_initializes_pi_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / 'workspace'
            workspace.mkdir()
            request = {
                'experiment': {'mode': 'sluice-lite', 'case': 'sample.case',
                               'optimizer': {'backend': 'pi', 'profile': 'test'}},
                'context': {'session_root': str(root), 'workspace': str(workspace),
                            'case': {'module': 'candidate.py', 'entry': 'run'},
                            'budget': {'max_hardware_actions': 8}},
                'preflight': {'profile': {'argv': ['pi', '{prompt}'], 'sha256': 'profile'},
                              'backend': {'version': 'test'}},
            }
            with mock.patch.object(runner.provider, 'prepare', side_effect=lambda profile, _: profile), \
                 mock.patch.object(runner, 'render', return_value=(['pi', 'prompt'], {})), \
                 mock.patch.object(runner.provenance, 'freeze_agent', return_value={}):
                prepared = runner.prepare(request)
            self.assertEqual(prepared['archives']['structured_knowledge'], 'method/resources/knowledge')
            self.assertEqual(set(prepared['archives']), {
                'agent_configuration', 'structured_knowledge', 'adapted_skills'})
            self.assertTrue((root / 'method/resources/knowledge/index/search.json').is_file())
        self.assertNotIn('method/resources/knowledge/index/search.json', prepared['records'])
        self.assertIn('method/README.md', prepared['records'])
        self.assertIn('ptobench evaluate .', prepared['records']['prompt.txt'])
        self.assertIn('Run each `ptobench` action in the foreground', prepared['records']['prompt.txt'])
        self.assertIn('do not launch actions with `nohup`, `&`', prepared['records']['prompt.txt'])
        self.assertIn('at least once before stopping', prepared['records']['prompt.txt'])


if __name__ == '__main__':
    unittest.main()
