"""Original measurements and failed action evidence, without NPU execution."""
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from action_test_support import ActionTestBase, SOURCE_BENCHMARK
import common
import execution_record
import session

sys.path.insert(0, str(SOURCE_BENCHMARK))
from modes._shared import recording


class RecordingProfilesTest(unittest.TestCase):
    def profile(self, backend, *argv, **settings):
        return {'backend': backend, 'argv': [backend, *argv, '{prompt}'], 'env': {}, **settings}

    def test_explicit_codex_model_and_reasoning_overrides_are_preserved(self):
        for flag in ('-c', '--config', '--config='):
            with self.subTest(flag=flag), mock.patch.object(recording, 'host_settings', return_value={
                    'model': 'host-model', 'model_reasoning_effort': 'low'}):
                overrides = ['model="explicit-model"', 'model_reasoning_effort="high"']
                argv = ([flag + item for item in overrides] if flag.endswith('=') else
                        [item for override in overrides for item in (flag, override)])
                prepared = recording.prepare_profile(self.profile('codex', 'exec', *argv))['argv']
                self.assertIn('--json', prepared)
                self.assertNotIn('--model', prepared)
                self.assertEqual(recording.codex_override(prepared, 'model'), 'explicit-model')
                self.assertEqual(recording.codex_override(prepared, 'model_reasoning_effort'), 'high')

    def test_pi_default_provider_model_and_thinking_are_pinned(self):
        with mock.patch.object(recording, 'host_settings', return_value={
                'defaultProvider': 'fixture-provider', 'defaultModel': 'fixture-model',
                'defaultThinkingLevel': 'medium'}):
            prepared = recording.prepare_profile(self.profile('pi', '--print'))['argv']
        self.assertEqual(recording.option(prepared, '--provider'), 'fixture-provider')
        self.assertEqual(recording.option(prepared, '--model'), 'fixture-model')
        self.assertEqual(recording.option(prepared, '--thinking'), 'medium')
        self.assertEqual(recording.option(prepared, '--session-dir'), '{session_root}/agent/native')

    def test_pi_cli_model_thinking_suffix_takes_precedence_over_host_default(self):
        with mock.patch.object(recording, 'host_settings', return_value={'defaultThinkingLevel': 'low'}):
            prepared = recording.prepare_profile(self.profile('pi', '--print', '--model', 'fixture-model:high'))['argv']
        self.assertEqual(recording.option(prepared, '--model'), 'fixture-model:high')
        self.assertEqual(recording.option(prepared, '--thinking'), 'high')


class MeasurementRecordsTest(unittest.TestCase):
    def test_invalid_timing_slots_are_retained_in_original_order(self):
        bench = SimpleNamespace(per_round=lambda metric: [71.0, 0.0, None, -1.0, math.nan, math.inf, 72.0])
        result = SimpleNamespace(bench=bench)
        self.assertEqual(common.raw_bench_samples(result), [71.0, 0.0, None, -1.0, 'nan', 'inf', 72.0])
        self.assertEqual(common.bench_samples(result), [71.0, 72.0])

    def test_runtime_boundary_records_entry_return_and_original_error(self):
        import golden.runner as runner
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'execution.json'
            execution_record.write(path, {'events': []})
            runtime = SimpleNamespace(platform='a2a3', device_id=2,
                                      distributed_config=SimpleNamespace(device_ids=(2, 3)))
            with mock.patch.dict(os.environ, {'PTOBENCH_EXECUTION_RECORD': str(path)}), \
                    mock.patch.object(runner, '_dispatch', return_value='completed') as original:
                with execution_record.observe_runtime(runtime):
                    self.assertEqual(runner._dispatch(), 'completed')
                self.assertIs(runner._dispatch, original)
            record = json.loads(path.read_text())
            self.assertEqual(record['effective_target'], {'platform': 'a2a3', 'device': 2, 'device_ids': [2, 3]})
            events = record['events']
            self.assertEqual([event['stage'] for event in events], ['runtime_dispatch_entered', 'runtime_dispatch_returned'])
            execution_record.write(path, {'events': []})
            with mock.patch.dict(os.environ, {'PTOBENCH_EXECUTION_RECORD': str(path)}), \
                    mock.patch.object(runner, '_dispatch', side_effect=RuntimeError('device setup failed')):
                with execution_record.observe_runtime(), self.assertRaisesRegex(RuntimeError, 'device setup failed'):
                    runner._dispatch()
            events = json.loads(path.read_text())['events']
            self.assertEqual([event['stage'] for event in events], ['runtime_dispatch_entered', 'runtime_dispatch_raised'])
            self.assertEqual(events[-1]['message'], 'device setup failed')


class FailedActionRecordsTest(ActionTestBase):
    def test_failed_candidate_result_and_stderr_have_protected_originals(self):
        self.configure(fail=True)
        process, result = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 1, result)
        event = self.events()[0]
        index = json.loads((self.handle.session_root/event['artifact_index']).read_text())
        archived = self.handle.session_root/event['artifact_index']
        result_file = archived.parent/'result.json'
        self.assertFalse(json.loads(result_file.read_text())['runs'][0]['passed'])
        self.assertIn('fake evaluator stderr', (archived.parent/'stderr.log').read_text())
        process_record = json.loads((archived.parent/'process_execution.json').read_text())
        self.assertEqual(process_record['exit_code'], 1)
        self.assertTrue(process_record['spawned'])
        snapshot = self.handle.session_root/index['candidate_snapshot']
        self.assertEqual((snapshot/'helper.py').read_text(), 'VALUE = 1\n')
        self.assertIsNone(event['snapshot_id'])
        self.assertEqual(event['hardware_actions_used'], 1)
        manifest = session._read(self.handle.session_root/'protection/manifest.json')
        self.assertIn('action_artifacts_1', manifest['roots'])


if __name__ == '__main__':
    unittest.main()
