"""CPU execution and managed-Session regressions; no device dispatch."""
import contextlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_execution
import session
import snapshots
from action_test_support import ActionTestBase


class OutputValidationTest(unittest.TestCase):
    def test_wrong_outputs_skip_only_comparison_and_keep_cached_input_and_state(self):
        import torch
        import golden.runner as runner
        from golden import ScalarSpec, TensorSpec

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('in', 'out'):
                (root / name).mkdir()
            torch.save(torch.tensor([3., 4.]), root / 'in/x.pt')
            torch.save(torch.tensor([5., 6.]), root / 'in/state.pt')
            torch.save(torch.tensor(7, dtype=torch.int32), root / 'in/count.pt')
            for name in ('state', 'out'):
                torch.save(torch.zeros(2), root / f'out/{name}.pt')
            specs = [TensorSpec('x', [2], torch.float32),
                     TensorSpec('state', [2], torch.float32),
                     TensorSpec('out', [2], torch.float32),
                     ScalarSpec('count', torch.int32, 99)]
            for spec, direction in zip(specs, ('in', 'inout', 'out')):
                spec.direction = direction
            dispatches = []

            def dispatch(_compiled, _specs, tensors, scalars, _config):
                self.assertTrue(torch.equal(tensors['x'], torch.tensor([3., 4.])))
                self.assertTrue(torch.equal(tensors['state'], torch.tensor([5., 6.])))
                self.assertEqual(scalars['count'].to_python(), 7)
                dispatches.append(True)
                tensors['out'].fill_(123.)
                tensors['state'].fill_(456.)

            original_golden = runner._compute_golden
            for check in (True, False, True):
                with self.subTest(check=check), \
                        mock.patch.dict(os.environ, {'PYPTO_BENCH': '0', 'PYPTO_BENCH_RESET_INPUTS': 'none'}), \
                        mock.patch.object(runner, '_normalize_config', return_value=SimpleNamespace(enable_chip_swimlane=0)), \
                        mock.patch.object(runner, '_validate_compiled_spec_abi') as abi, \
                        mock.patch.object(runner, '_dispatch', side_effect=dispatch), \
                        profile_execution.output_validation(enabled=check):
                    result = runner._run_pipeline(
                        specs, compile_step=lambda *_: SimpleNamespace(output_dir=root),
                        compile_label='cpu-profile-policy', prologue=None,
                        golden_fn=None, golden_data=str(root), config=None,
                        rtol=0, atol=0, compare_fn={}, compile_only=False,
                        runtime_dir=None, save_data=False, capture_outputs=False,
                    )
                    abi.assert_called_once()
                    self.assertEqual(result.passed, not check)
                self.assertIs(runner._compute_golden, original_golden)
            self.assertEqual(len(dispatches), 3)

    def test_scoped_hook_is_restored_on_execution_error(self):
        import golden.runner as runner
        original = runner._compute_golden
        with self.assertRaisesRegex(RuntimeError, 'runtime failed'):
            with profile_execution.output_validation(enabled=False):
                self.assertIsNone(runner._compute_golden())
                raise RuntimeError('runtime failed')
        self.assertIs(runner._compute_golden, original)


class AgentProfilePolicyTest(ActionTestBase):
    def test_old_sessions_remain_readable_but_cannot_restore_the_old_evaluator(self):
        marker = self.bench / 'evaluator/profile_execution.py'
        current_bytes = marker.read_bytes()
        for backend in ('codex', 'maintenance'):
            for previous_policy in (None, current_bytes.replace(
                    b'output correctness checking disabled.', b'output correctness NOT_CHECKED.')):
                if previous_policy is None:
                    marker.unlink()
                else:
                    marker.write_bytes(previous_policy)
                try:
                    old = self.create(budget=2, optimizer=backend)
                finally:
                    marker.write_bytes(current_bytes)
                self.assertEqual(session.read_session(old.session_root)['optimizer']['backend'], backend)
                environment = {**self.env, 'PTOBENCH_SESSION_ROOT': str(old.session_root)}
                process, result = self.cli(['profile', 'pmu'], cwd=old.initial_workspace, env=environment)
                self.assertEqual(process.returncode, 5, result)
                self.assertIn('fresh Session', result['message'])
                self.assertEqual(marker.read_bytes(), current_bytes)

    def test_agent_profiles_cannot_publish_correctness_score_or_selectable_snapshot(self):
        for mode in ('runtime', 'pmu', 'compiler'):
            self.configure(median=0.001)
            process, result = self.cli(['profile', mode])
            self.assertEqual(process.returncode, 0, result)
            self.assertNotIn('correctness', result)
            self.assertNotIn('_omit_correctness', result)
            self.assertNotIn('correctness', self.events()[-1]['observation'])
            self.assertNotIn('correctness', self.events()[-1]['result'])
            self.assertIsNone(result['latency_us'])
            self.assertIsNone(result['snapshot_id'])
            self.assertTrue(self.calls()[-1]['skip_correctness'])
            self.assertIsNone(snapshots.select_best_snapshot(self.handle.session_root))
        process, checked = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 0, checked)
        self.assertEqual(checked['correctness'], 'PASS')
        self.assertFalse(self.calls()[-1]['skip_correctness'])
        self.assertEqual(snapshots.select_best_snapshot(self.handle.session_root).snapshot_id,
                         checked['snapshot_id'])
        with mock.patch.dict(os.environ, self.env):
            final = snapshots.finalize_session(self.handle.session_root, 'optimizer_exit')
        self.assertEqual(final.status, 'completed')
        self.assertEqual(final.final_verified_correctness, 'passed')
        self.assertFalse(self.calls()[-1]['skip_correctness'])
        self.assertEqual((self.calls()[-1]['phase'], self.calls()[-1]['profile_mode']), ('all', 'clean'))

    def test_runtime_errors_and_integrity_failures_still_fail_profiles(self):
        for config, code in (({'fail': True}, 1), ({'crash': True}, 1), ({'tamper': 'source'}, 4)):
            self.configure(**config)
            process, result = self.cli(['profile', 'pmu'])
            self.assertEqual(process.returncode, code, result)
            self.assertNotIn('correctness', result)
            self.assertIsNone(result['snapshot_id'])
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'], 3)

    def test_maintenance_profiles_keep_original_correctness_and_snapshot_policy(self):
        self.handle = self.create(budget=4, optimizer='maintenance')
        self.env['PTOBENCH_SESSION_ROOT'] = str(self.handle.session_root)
        for mode in ('runtime', 'pmu', 'compiler'):
            process, result = self.cli(['profile', mode])
            self.assertEqual(process.returncode, 0, result)
            self.assertEqual(result['correctness'], 'PASS')
            self.assertFalse(self.calls()[-1]['skip_correctness'])
            self.assertIsNotNone(result['snapshot_id'])
        self.configure(fail=True)
        process, result = self.cli(['profile', 'runtime'])
        self.assertEqual(process.returncode, 1, result)


if __name__ == '__main__':
    unittest.main()
