"""Basic exposes the public Case environment without injecting a wiki bundle."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

BENCHMARK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCHMARK), str(BENCHMARK / 'evaluator')]

from modes.basic.prompt import make_prompt
from modes.basic import runner


class BasicModeTests(unittest.TestCase):
    def test_prompt_keeps_public_case_guidance_without_knowledge_injection(self):
        prompt = make_prompt(
            {'case': 'test.case'},
            {'budget': {'max_hardware_actions': 8}},
            {'module': 'candidate.py', 'entry': 'run'},
            'Normal optimization instructions.',
        )
        for expected in ('.benchmark_context/README.md', '.benchmark_context/tuning_capabilities.json',
                         'candidate/', 'candidate/tuning.json', 'ptobench evaluate .',
                         'Normal optimization instructions.'):
            self.assertIn(expected, prompt)
        self.assertNotIn('reference knowledge', prompt)
        self.assertNotIn('raw_wiki', prompt)

    def test_preflight_does_not_access_knowledge_bundle(self):
        experiment = {'optimizer': {'backend': 'pi', 'profile': 'selected'}}
        adapter = mock.Mock()
        adapter.preflight.return_value.to_dict.return_value = {'version': 'fake'}
        selected = {'argv': ['pi', '{prompt}']}
        with mock.patch.object(runner, 'load_profile', return_value=selected), \
             mock.patch.object(runner.provider, 'configure', return_value=selected) as configure, \
             mock.patch.object(runner.recording, 'prepare_profile', side_effect=lambda value: value), \
             mock.patch.object(runner, 'get_adapter', return_value=adapter):
            result = runner.preflight({'experiment': experiment})
        self.assertEqual(result['profile'], selected)
        configure.assert_called_once_with(selected, Path(runner.__file__).parent)

    def test_prepare_has_no_wiki_files_or_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = {'backend': 'pi', 'argv': ['pi', '{prompt}'], 'sha256': 'test-profile'}
            request = {
                'experiment': {'mode': 'basic', 'case': 'test.case'},
                'context': {'session_root': str(root), 'workspace': str(root / 'workspace'),
                            'case': {'module': 'candidate.py', 'entry': 'run'},
                            'budget': {'max_hardware_actions': 8}},
                'preflight': {'profile': selected, 'backend': {'version': 'fake'}},
            }
            with mock.patch.object(runner.provider, 'prepare', return_value=selected), \
                 mock.patch.object(runner, 'render', return_value=(['pi'], {})), \
                 mock.patch.object(runner.provenance, 'freeze_agent', return_value={}):
                result = runner.prepare(request)
        self.assertNotIn('method/knowledge.json', result['records'])
        self.assertNotIn('method/README.md', result['records'])
        self.assertNotIn('reference_knowledge', result['archives'])
        self.assertFalse((root / 'method/resources/raw_wiki').exists())


if __name__ == '__main__':
    unittest.main()
