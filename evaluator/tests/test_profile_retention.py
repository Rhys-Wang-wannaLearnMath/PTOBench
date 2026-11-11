"""Durable action originals survive replacement and interrupted saves."""
import errno
import json
from unittest import mock

from action_test_support import ActionTestBase
import action_records
import file_tree
import integrity
import session


class ProfileRetentionTest(ActionTestBase):
    def setUp(self):
        super().setUp()
        # Archive-path assertions below target the explicit legacy split mode.
        self.handle = self.create(budget=20, profile_strategy='split')
        self.env['PTOBENCH_SESSION_ROOT'] = str(self.handle.session_root)

    def test_multi_round_restart_retains_first_capture_and_repairs_archive(self):
        for generation in (1, 2, 3):
            self.configure(generation=generation)
            process, result = self.cli(['profile', 'runtime'])
            self.assertEqual(process.returncode, 0, result)
            self.assertNotIn('unavailable', result.get('message') or '')
        latest = self.handle.initial_workspace / '.benchmark_outputs/runtime'
        self.assertEqual((latest / 'generation.txt').read_text(), '3')
        self.assertFalse((latest / 'first_only.txt').exists())
        first = self.events()[0]
        saved = action_records.verified_archive(self.handle.session_root, first['artifact_index'])
        original = saved / '.benchmark_outputs/runtime/generation.txt'
        original.chmod(0o644)
        original.write_text('corrupted')
        # A fresh CLI process performs recovery; historical references never use latest.
        process, _ = self.cli(['profile', 'compiler'])
        self.assertEqual(process.returncode, 0)
        saved = action_records.verified_archive(self.handle.session_root, first['artifact_index'], event_seq=1)
        self.assertEqual((saved / '.benchmark_outputs/runtime/generation.txt').read_text(), '1')
        self.assertEqual((saved / '.benchmark_outputs/runtime/first_only.txt').read_text(), 'first')
        receipt = json.loads((latest / 'capture_receipt.json').read_text())
        self.assertEqual(receipt['event_seq'], 3)

    def test_save_faults_never_remove_frozen_originals_or_prior_public_capture(self):
        process, result = self.cli(['profile', 'runtime'])
        self.assertEqual(process.returncode, 0, result)
        root = self.handle.session_root
        public = self.handle.initial_workspace / '.benchmark_outputs/runtime/generation.txt'
        action = root / 'staging/action_000999'
        source = action / 'workspace/.benchmark_outputs/runtime'
        source.mkdir(parents=True)
        (source / 'trace.json').write_text('sole new original')
        # Include an empty directory and arbitrary nested rank files in the inventory.
        (source / 'rank0/empty').mkdir(parents=True)
        (source / 'rank1').mkdir()
        (source / 'rank1/trace.bin').write_bytes(b'\x00\xff')
        for error in (OSError(errno.ENOSPC, 'disk full'), InterruptedError('interrupted')):
            with mock.patch.object(file_tree, 'copy_descriptor', side_effect=error):
                with self.assertRaises(type(error)):
                    action_records.archive(root, 999, action)
            self.assertEqual((source / 'trace.json').read_text(), 'sole new original')
            self.assertEqual(public.read_text(), '1')
            self.assertNotIn('action_artifacts_999', session._read(root / 'protection/manifest.json')['roots'])
        index = action_records.archive(root, 999, action)
        saved = action_records.verified_archive(root, index)
        self.assertTrue((saved / '.benchmark_outputs/runtime/rank0/empty').is_dir())
        self.assertEqual((saved / '.benchmark_outputs/runtime/rank1/trace.bin').read_bytes(), b'\x00\xff')

    def test_late_added_file_rejects_partial_inventory_and_allows_retry(self):
        root = self.handle.session_root
        action = root / 'staging/action_000999'
        source = action / 'workspace/.benchmark_outputs/runtime'
        source.mkdir(parents=True)
        (source / 'trace').write_bytes(b'first')
        real_copy = file_tree.copy_descriptor
        def copy_then_add(*args):
            real_copy(*args)
            (source / 'late').write_bytes(b'late')
        with mock.patch.object(file_tree, 'copy_descriptor', side_effect=copy_then_add):
            with self.assertRaisesRegex(integrity.InfrastructureError, 'tree changed'):
                action_records.archive(root, 999, action)
        self.assertTrue((source / 'late').is_file())
        index = action_records.archive(root, 999, action)
        self.assertEqual((root / index).parent.joinpath('.benchmark_outputs/runtime/late').read_bytes(), b'late')

    def test_unpreserved_modified_public_capture_is_not_overwritten(self):
        process, result = self.cli(['profile', 'runtime'])
        self.assertEqual(process.returncode, 0, result)
        original = self.handle.initial_workspace / '.benchmark_outputs/runtime/generation.txt'
        original.chmod(0o644)
        original.write_text('only public version')
        self.configure(generation=2)
        process, result = self.cli(['profile', 'runtime'])
        self.assertEqual(process.returncode, 0, result)
        self.assertIn('originals; retained', result['message'])
        self.assertEqual(original.read_text(), 'only public version')
        # The new capture is still archived and the consumed action is honest.
        saved = action_records.verified_archive(self.handle.session_root, self.events()[-1]['artifact_index'])
        self.assertEqual((saved / '.benchmark_outputs/runtime/generation.txt').read_text(), '2')

    def test_raw_export_is_bounded_historical_and_capture_bound(self):
        import base64
        for generation in (1, 2):
            self.configure(generation=generation)
            process, result = self.cli(['profile', 'runtime'])
            self.assertEqual(process.returncode, 0, result)
        root = self.handle.session_root
        first = action_records.read_profile_archive(root, 1)
        entry = next(f for f in first['files'] if f['name'] == 'generation.txt')
        data = action_records.read_profile_archive(root, 1, file_ref=entry['file_ref'])
        self.assertEqual(base64.b64decode(data['data']), b'1')
        self.assertIsNone(data['next_offset'])
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            action_records.read_profile_archive(root, 2, file_ref=entry['file_ref'])
        with self.assertRaises(ValueError):
            action_records.read_profile_archive(root, 1, file_ref='../../sealed/golden')
        with self.assertRaises(ValueError):
            action_records.read_profile_archive(root, 1, file_ref=entry['file_ref'], offset=2)
        with self.assertRaises(ValueError):
            action_records.read_profile_archive(root, 999)
