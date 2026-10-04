from __future__ import annotations

import copy
import json
import os
import multiprocessing
from pathlib import Path
import shutil
from unittest import mock

from session_test_support import SessionTestBase, write_json
import common
import integrity
import session
import workspace


def crash_fork(bench, repo, root, after_manifest):
    common.BENCHMARK_ROOT, common.REPO_ROOT = Path(bench), Path(repo)
    original_commit = session._commit
    original_write = integrity._write_manifest
    def commit(protection_root, record, events=None):
        if len(record['workspaces']) > 1:
            if not after_manifest:
                os._exit(23)
            def crash_write(path, manifest):
                original_write(path, manifest)
                os._exit(23)
            integrity._write_manifest = crash_write
        original_commit(protection_root, record, events)
    session._commit = commit
    session.fork_workspace(root, Path(root) / 'workspaces/worker_000')


def crash_creation(bench, repo, output):
    common.BENCHMARK_ROOT, common.REPO_ROOT = Path(bench), Path(repo)
    original = integrity._replace_record
    def crash(*args):
        original(*args)
        os._exit(23)
    integrity._replace_record = crash
    session.create_session('test.case', 'codex', output)


class SessionTest(SessionTestBase):
    def test_creation_independent_layout_schema_and_original_assets(self):
        self.make_case('test.other')
        with mock.patch.object(workspace, 'init_workspace', wraps=workspace.init_workspace) as init, \
             mock.patch('subprocess.run', side_effect=AssertionError('no subprocess in S2')):
            first = self.create()
            second = self.create(5, 'pi', 'test.other')
        self.assertEqual(init.call_count, 2)
        for handle in (first, second):
            record = session.read_session(handle.session_root)
            self.assertEqual(record['status'], 'running')
            self.assertEqual(record['usage'], {'hardware_actions': 0, 'finish_actions': 0})
            self.assertEqual(record['next_event_seq'], 1)
            self.assertEqual(record['workspaces']['worker_000']['path'], 'workspaces/worker_000')
            self.assertRegex(record['session_id'], r'^\d{8}T\d{6}Z_[a-f0-9]{12}$')
            self.assertEqual(record['profiling_modes'], ['runtime', 'pmu', 'compiler'])
            for name in ('workspaces', 'staging', 'final', 'protection'):
                self.assertTrue((handle.session_root / name).is_dir())
            self.assertTrue((handle.session_root / 'process.log').is_file())
            self.assertTrue(handle.snapshots_root.is_dir())
            self.assertFalse((handle.session_root / 'final_result.json').exists())
            self.assertEqual((handle.session_root / 'events.jsonl').read_bytes(), b'')
            manifest = common.read_json(handle.protection_root / 'manifest.json')
            self.assertTrue({'case', 'sealed', 'environment', 'evaluator', 'golden', 'public_ptobench', 'public_launcher'} <= manifest['roots'].keys())
            self.assertTrue(any(n.startswith('family_') for n in manifest['roots']))
            self.assertEqual((handle.initial_workspace / '.benchmark_context/config.py').stat().st_mode & 0o777, 0o444)
            self.assertTrue(integrity.check_only(handle.protection_root).passed)
        self.assertNotEqual(first.session_root, second.session_root)
        self.assertEqual(session.read_session(first.session_root)['budget']['max_hardware_actions'], 2)
        self.assertEqual(session.read_session(second.session_root)['budget']['max_hardware_actions'], 5)

    def test_preflight_rejects_every_incomplete_certified_asset_without_preparing(self):
        cases = [
            ('pending', 'certification.json', lambda p: write_json(p, {'status': 'pending'})),
            ('wrong_case', 'certification.json', lambda p: write_json(p, {**common.read_json(p), 'case_id': 'test.forged'})),
            ('hash', 'certification.json', lambda p: write_json(p, {**common.read_json(p), 'comparator_sha256': '0' * 64})),
            ('input_y', 'fixtures/fixture_000/in/y.pt', lambda p: p.unlink()),
            ('state', 'fixtures/fixture_000/in/out.pt', lambda p: p.unlink()),
            ('scalar', 'fixtures/fixture_000/in/count.pt', lambda p: p.unlink()),
            ('output', 'fixtures/fixture_000/out/out.pt', lambda p: p.unlink()),
            ('empty_input', 'fixtures/fixture_000/in/x.pt', lambda p: p.write_bytes(b'')),
            ('wrong_manifest', 'fixtures/fixture_000/manifest.json', lambda p: write_json(p, {**common.read_json(p), 'inputs_dir': '../outside'})),
            ('baseline', 'ref_performance.json', lambda p: p.unlink()),
            ('evaluator', 'evaluate.py', lambda p: p.unlink()),
            ('ref', 'ref/ref_npu.py', lambda p: p.unlink()),
            ('oracle', 'oracle/ref_cpu.py', lambda p: p.unlink()),
            ('builder', 'fixture_builder.py', lambda p: p.unlink()),
        ]
        for label, relative, mutate in cases:
            with self.subTest(asset=label):
                case_id = 'test.' + label
                _, sealed = self.make_case(case_id)
                mutate(sealed / relative)
                with self.assertRaises(session.SessionError), mock.patch('subprocess.run', side_effect=AssertionError('must not prepare Ref')):
                    self.create(case_id=case_id)
                self.assertFalse((self.output['runs_root'] / 'codex' / case_id).exists())
        for index, invalid in enumerate((0, -1, float('inf'), float('nan'), True, None)):
            with self.subTest(latency=invalid):
                case_id = f'test.latency{index}'
                _, sealed = self.make_case(case_id)
                write_json(sealed / 'ref_performance.json', {'status': 'ready', 'fixtures': {'fixture_000': {'median_effective_us': invalid}}})
                with self.assertRaises(session.SessionError):
                    self.create(case_id=case_id)
        case_dir, _ = self.make_case('test.no_seed')
        (case_dir / 'public/candidate_seed/candidate.py').unlink()
        with self.assertRaises(session.SessionError):
            self.create(case_id='test.no_seed')

    def test_invalid_parameters_output_overlap_and_link_components(self):
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(session.SessionError):
                self.create(budget=budget)
        for value in ({'finish_actions': 2}, {'finish_actions': True}, {'unknown': 1}, [], False):
            with self.subTest(value=value), self.assertRaises(session.SessionError):
                session.create_session(self.case_id, 'codex', self.output, value)
        for runs, snapshots in ((self.bench / 'cases/new', self.repo / 'snap'), (self.repo / 'run', self.repo / 'run/sub'),
                                (self.repo, self.repo / 'snap'), (self.repo / 'run', self.bench / 'evaluator/snap'),
                                (self.repo / 'run', self.repo / 'golden/snap')):
            with self.subTest(runs=runs), self.assertRaises(session.SessionError):
                self.create(output={'runs_root': runs, 'snapshots_root': snapshots})
        (self.repo / 'link').symlink_to(self.repo, target_is_directory=True)
        with self.assertRaises(session.SessionError):
            self.create(output={'runs_root': self.repo / 'link/../runs', 'snapshots_root': self.repo / 'snap'})
        with self.assertRaises(session.SessionError):
            self.create(optimizer='../escape')
        with self.assertRaises(session.SessionError):
            self.create(case_id='../test.case')

    def test_metadata_cannot_change_identity_budget_module_or_modes(self):
        handle = self.create()
        context = handle.initial_workspace / '.benchmark_context'
        write_json(context / 'workspace.json', {'case_id': 'evil.other', 'candidate_module': 'outside.py', 'entry': 'evil'})
        write_json(context / 'profiling_modes.json', {'modes': {'hardware': {}, 'unlimited': {}}})
        name, target, case_id = session.resolve_workspace(handle.session_root, handle.initial_workspace)
        self.assertEqual((name, target, case_id), ('worker_000', handle.initial_workspace, self.case_id))
        self.assertEqual(session.read_session(handle.session_root)['profiling_modes'], ['runtime', 'pmu', 'compiler'])
        child_id = session.fork_workspace(handle.session_root, handle.initial_workspace)
        child = handle.session_root / 'workspaces' / child_id
        self.assertEqual(common.read_json(child / '.benchmark_context/workspace.json')['case_id'], self.case_id)
        self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 0)

    def test_registered_roots_only_cross_session_subdirectory_and_links(self):
        handle = self.create()
        other = self.create(optimizer='pi')
        (handle.initial_workspace / 'sub').mkdir()
        (handle.session_root / 'workspaces/fake').mkdir()
        link = handle.session_root / 'workspaces/link'
        link.symlink_to(handle.initial_workspace, target_is_directory=True)
        for value in (other.initial_workspace, handle.initial_workspace / 'sub', handle.session_root / 'workspaces/fake',
                      link, link / '../worker_000', handle.initial_workspace / '../../../outside', self.repo):
            with self.subTest(value=value), self.assertRaises(session.SessionError) as caught:
                session.resolve_workspace(handle.session_root, value)
            self.assertEqual(caught.exception.error_code, 'invalid_workspace')
        relative = os.path.relpath(handle.initial_workspace, Path.cwd())
        self.assertEqual(session.resolve_workspace(handle.session_root, relative)[0], 'worker_000')
        session_link = self.repo / 'session_link'
        session_link.symlink_to(handle.session_root, target_is_directory=True)
        with self.assertRaises(session.SessionError):
            session.read_session(session_link)
        with session.execution_lock(), self.assertRaises(session.SessionError):
            session.register_workspace(handle.session_root, other.initial_workspace, 'worker_000')
        self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 0)

    def test_reads_repair_latest_records_without_touching_context_or_assets(self):
        handle = self.create()
        with session.execution_lock():
            self.reserve(handle)
            event = session.complete_action(handle.session_root, 1, exit_code=1, error_code='action_failed')
        case_config = self.bench / 'cases/test/case/public/config.py'
        case_config.write_text('CHANGED = True\n')
        (handle.initial_workspace / 'helper.py').write_text('VALUE = 9\n')
        context_config = handle.initial_workspace / '.benchmark_context/config.py'
        context_config.chmod(0o600)
        context_config.write_text('ROGUE = True\n')
        write_json(handle.session_root / 'session.json', {'case_id': 'test.other', 'usage': {'hardware_actions': 0}})
        (handle.session_root / 'events.jsonl').write_text('forged\n')
        with self.assertRaises(integrity.InfrastructureError):
            session.read_session(handle.session_root, repair=False)
        restored = session.read_session(handle.session_root)
        self.assertEqual(restored['usage']['hardware_actions'], 1)
        self.assertEqual(json.loads((handle.session_root / 'events.jsonl').read_text()), event)
        self.assertEqual(case_config.read_text(), 'CHANGED = True\n')
        self.assertEqual(context_config.read_text(), 'ROGUE = True\n')
        with session.execution_lock():
            repaired = session.repair_assets(handle.session_root)
        self.assertTrue(repaired.restored_paths)
        self.assertEqual(case_config.read_text(), 'SIZE = 2\n')
        self.assertEqual(context_config.read_text(), 'SIZE = 2\n')
        self.assertEqual((handle.initial_workspace / 'helper.py').read_text(), 'VALUE = 9\n')
        self.assertEqual((handle.initial_workspace / '.benchmark_context/starting_point/helper.py').read_text(), 'VALUE = 1\n')
        self.assertTrue(integrity.check_only(handle.protection_root).passed)

    def test_corrupt_manifest_backup_and_wrong_binding_fail_without_rebaseline(self):
        handle = self.create()
        manifest = common.read_json(handle.protection_root / 'manifest.json')
        backup = handle.protection_root / manifest['roots']['session']['entries']['']['backup']
        backup.chmod(0o600)
        backup.write_bytes(b'corrupt')
        (handle.session_root / 'session.json').write_text('{}')
        with self.assertRaises(integrity.InfrastructureError):
            session.read_session(handle.session_root)
        self.assertEqual((handle.session_root / 'session.json').read_text(), '{}')
        other = self.create(optimizer='pi')
        (other.protection_root / 'manifest.json').write_text('invalid')
        with self.assertRaises(integrity.InfrastructureError):
            session.read_session(other.session_root)
        third = self.create(optimizer='opencode')
        original = (third.session_root / 'session.json').read_bytes()
        foreign = common.read_json(third.protection_root / 'manifest.json')
        foreign['roots']['session']['path'] = str(handle.session_root / 'session.json')
        write_json(third.protection_root / 'manifest.json', foreign)
        with self.assertRaises(integrity.InfrastructureError):
            session.read_session(third.session_root)
        self.assertEqual((handle.session_root / 'session.json').read_text(), '{}')
        self.assertEqual((third.session_root / 'session.json').read_bytes(), original)

    def test_fork_copies_helpers_and_initial_snapshot_without_budget(self):
        handle = self.create()
        (handle.initial_workspace / 'helper.py').write_text('VALUE = 2\n')
        (handle.initial_workspace / '.benchmark_outputs/cache').mkdir(parents=True)
        (handle.initial_workspace / '.benchmark_outputs/cache/junk').write_text('excluded')
        first = session.fork_workspace(handle.session_root, handle.initial_workspace)
        child = handle.session_root / 'workspaces' / first
        self.assertEqual(first, 'worker_001')
        self.assertEqual((child / 'helper.py').read_text(), 'VALUE = 2\n')
        self.assertEqual((child / '.benchmark_context/starting_point/helper.py').read_text(), 'VALUE = 2\n')
        self.assertFalse((child / '.benchmark_outputs').exists())
        second = session.fork_workspace(handle.session_root, child)
        record = session.read_session(handle.session_root)
        self.assertEqual(second, 'worker_002')
        self.assertEqual(record['workspaces'][second]['parent'], first)
        self.assertEqual(record['usage']['hardware_actions'], 0)
        self.assertTrue(integrity.check_only(handle.protection_root).passed)

    def test_failed_fork_has_no_registry_or_partial_and_never_reuses_allocated_id(self):
        handle = self.create()
        with mock.patch.object(workspace, 'write_context', side_effect=RuntimeError('context failure')):
            with self.assertRaises(RuntimeError):
                session.fork_workspace(handle.session_root, handle.initial_workspace)
        record = session.read_session(handle.session_root)
        self.assertEqual(set(record['workspaces']), {'worker_000'})
        self.assertEqual(record['next_workspace_index'], 2)
        self.assertEqual(list((handle.session_root / 'workspaces').iterdir()), [handle.initial_workspace])
        self.assertEqual(session.fork_workspace(handle.session_root, handle.initial_workspace), 'worker_002')

    def test_creation_failure_cleans_only_owned_unpublished_dirs(self):
        (self.output['runs_root'] / 'existing').mkdir(parents=True)
        sentinel = self.output['runs_root'] / 'existing/sentinel'
        sentinel.write_text('keep')
        with mock.patch.object(workspace, 'init_workspace', side_effect=RuntimeError('copy failed')):
            with self.assertRaises(RuntimeError):
                self.create()
        self.assertEqual(sentinel.read_text(), 'keep')
        self.assertEqual(list((self.output['runs_root'] / 'codex' / self.case_id).iterdir()), [])
        self.assertEqual(list((self.output['snapshots_root'] / 'codex' / self.case_id).iterdir()), [])

    def test_schema_validation_even_for_a_bench_committed_bad_record(self):
        handle = self.create()
        record = session.read_session(handle.session_root)
        for key, invalid in (('schema_version', True), ('next_event_seq', 0), ('status', 'unknown'),
                             ('profiling_modes', ['hardware']), ('usage', {'hardware_actions': 3, 'finish_actions': 0})):
            with self.subTest(key=key):
                bad = {**record, key: invalid}
                integrity.commit_records(handle.protection_root, {'session': json.dumps(bad).encode()})
                with self.assertRaises(integrity.InfrastructureError):
                    session.read_session(handle.session_root)
                integrity.commit_records(handle.protection_root, {'session': json.dumps(record).encode()})

    def test_creation_record_publication_survives_real_process_crash(self):
        process = multiprocessing.get_context('fork').Process(target=crash_creation, args=(self.bench, self.repo, self.output))
        process.start()
        process.join(10)
        self.assertEqual(process.exitcode, 23)
        roots = list((self.output['runs_root'] / 'codex/test.case').iterdir())
        self.assertEqual(len(roots), 1)
        record = session.read_session(roots[0])
        self.assertEqual(record['usage'], {'hardware_actions': 0, 'finish_actions': 0})
        self.assertEqual(set(record['workspaces']), {'worker_000'})
        self.assertEqual((roots[0] / 'events.jsonl').read_bytes(), b'')
        self.assertTrue(integrity.check_only(roots[0] / 'protection').passed)
        print('S2 creation crash: committed manifest -> complete Session and empty event log restored')

    def test_fork_crash_keeps_registry_atomic_and_never_reuses_published_number(self):
        for after_manifest in (False, True):
            with self.subTest(after_manifest=after_manifest):
                handle = self.create()
                process = multiprocessing.get_context('fork').Process(target=crash_fork, args=(self.bench, self.repo, handle.session_root, after_manifest))
                process.start()
                process.join(10)
                self.assertEqual(process.exitcode, 23)
                record = session.read_session(handle.session_root)
                self.assertEqual(record['next_workspace_index'], 2)
                published = handle.session_root / 'workspaces/worker_001'
                self.assertTrue((published / '.benchmark_context/starting_point/helper.py').is_file())
                self.assertEqual('worker_001' in record['workspaces'], after_manifest)
                if not after_manifest:
                    with self.assertRaises(session.SessionError):
                        session.resolve_workspace(handle.session_root, published)
                self.assertEqual(session.fork_workspace(handle.session_root, handle.initial_workspace), 'worker_002')
                self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 0)
                self.assertTrue(integrity.check_only(handle.protection_root).passed)
                print(f'S2 fork crash: registry_manifest={after_manifest}, next accepted worker=002, usage=0')

    def test_record_symlink_replaced_itself_and_external_sentinel_untouched(self):
        handle = self.create()
        sentinel = self.repo / 'external.json'
        sentinel.write_bytes(b'external sentinel')
        record_path = handle.session_root / 'session.json'
        record_path.unlink()
        record_path.symlink_to(sentinel)
        self.assertEqual(session.read_session(handle.session_root)['case_id'], self.case_id)
        self.assertFalse(record_path.is_symlink())
        self.assertEqual(sentinel.read_bytes(), b'external sentinel')
        lock = handle.session_root / '.session.lock'
        lock.unlink()
        lock.symlink_to(sentinel)
        with self.assertRaises(integrity.InfrastructureError):
            session.read_session(handle.session_root)
        self.assertEqual(sentinel.read_bytes(), b'external sentinel')

    def test_failed_asset_recovery_marks_session_failed_and_preserves_budget(self):
        handle = self.create()
        with session.execution_lock():
            self.reserve(handle)
            session.complete_action(handle.session_root, 1, exit_code=1, error_code='action_failed')
        manifest = common.read_json(handle.protection_root / 'manifest.json')
        entry = manifest['roots']['case']['entries']['public/config.py']
        backup = handle.protection_root / entry['backup']
        backup.chmod(0o600)
        backup.write_bytes(b'corrupt recovery source')
        (self.bench / 'cases/test/case/public/config.py').write_text('tampered')
        with session.execution_lock(), self.assertRaises(integrity.InfrastructureError):
            session.repair_assets(handle.session_root)
        record = session.read_session(handle.session_root)
        self.assertEqual(record['status'], 'failed')
        self.assertTrue(record['exploration_closed'])
        self.assertEqual(record['usage']['hardware_actions'], 1)
        self.assertEqual(record['close_reason'], 'asset_recovery_failed')

    def test_foreign_extra_record_and_fixed_binding_are_rejected_before_recovery(self):
        handle = self.create()
        sentinel = self.repo / 'external.json'
        sentinel.write_bytes(b'external sentinel')
        manifest = common.read_json(handle.protection_root / 'manifest.json')
        foreign = copy.deepcopy(manifest)
        foreign['roots']['foreign_record'] = {**foreign['roots']['session'], 'path': str(sentinel)}
        write_json(handle.protection_root / 'manifest.json', foreign)
        with self.assertRaises(integrity.InfrastructureError):
            session.read_session(handle.session_root)
        self.assertEqual(sentinel.read_bytes(), b'external sentinel')
        wrong_fixed = copy.deepcopy(manifest)
        wrong_fixed['roots']['environment']['path'] = str(sentinel)
        write_json(handle.protection_root / 'manifest.json', wrong_fixed)
        with session.execution_lock(), self.assertRaises(integrity.InfrastructureError):
            session.repair_assets(handle.session_root)
        self.assertEqual(sentinel.read_bytes(), b'external sentinel')
