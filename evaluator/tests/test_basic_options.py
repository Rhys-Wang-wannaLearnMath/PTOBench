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
        normalized_prompt = ' '.join(prompt.split())
        for expected in ('.benchmark_context/README.md', '.benchmark_context/tuning_capabilities.json',
                         'candidate/', 'candidate/tuning.json', 'ptobench evaluate .',
                         'Normal optimization instructions.',
                         'Evaluate allowance: 8 accepted `ptobench evaluate` actions',
                         'interrupted accepted Evaluate still consumes one',
                         'Profile budget cost is 0', 'recorded as a hardware action',
                         'remains available after the Evaluate',
                         'allowance is exhausted',
                         'Fork budget cost is 0', 'Extra Evaluate requests are rejected by the CLI',
                         'continue using Profile/Fork'):
            self.assertIn(expected, normalized_prompt)
        self.assertNotIn('accepted hardware actions', normalized_prompt)
        self.assertNotIn('Stop when you are finished or the exploration budget is exhausted', normalized_prompt)
        self.assertNotIn('reference knowledge', normalized_prompt)
        self.assertNotIn('raw_wiki', normalized_prompt)

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

    def test_pi_run_fails_closed_when_sandbox_is_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = {'backend': 'pi', 'mode_provider': {'base_url': 'https://example.invalid'},
                       'argv': ['pi', '{prompt}']}
            (root / 'profile.json').write_text(__import__('json').dumps(profile))
            (root / 'prompt.txt').write_text('prompt')
            request = {'context': {'session_root': str(root), 'workspace': str(root)},
                       'preflight': {'profile': profile}}
            with mock.patch.dict('os.environ', {'PTOBENCH_SANDBOX': '0'}), \
                 mock.patch.object(runner, 'render', return_value=(['pi'], {})), \
                 mock.patch.object(runner.provider, 'credentials', return_value={}), \
                 mock.patch.object(runner.os, 'execvpe') as execute:
                with self.assertRaisesRegex(RuntimeError, 'Pi Agent requires filesystem isolation'):
                    runner.run(request)
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
