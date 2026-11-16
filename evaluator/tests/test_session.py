from __future__ import annotations

import copy
import grp
import hashlib
import json
import os
import multiprocessing
from pathlib import Path
import shutil
import stat
from unittest import mock

from session_test_support import SessionTestBase, write_json
import common
import integrity
import session
import workspace


def tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob('*')):
        relative = str(path.relative_to(root)).encode()
        metadata = path.lstat()
        digest.update(relative + b'\0' + str(metadata.st_mode).encode() + b'\0')
        if stat.S_ISREG(metadata.st_mode):
            digest.update(path.read_bytes())
        elif stat.S_ISLNK(metadata.st_mode):
            digest.update(os.readlink(path).encode())
    return digest.hexdigest()


def crash_fork(bench, repo, root, crash_at):
    common.BENCHMARK_ROOT, common.REPO_ROOT = Path(bench), Path(repo)
    original_commit = session._commit
    original_phase = session._set_fork_phase
    original_rename = session._rename_fork_stage
    original_register = session._context_root_registered
    original_publish = workspace.publish_workspace_root
    original_write_context = workspace.write_context
    original_fchown = os.fchown

    def commit(protection_root, record, events=None, record_updates=None):
        original_commit(protection_root, record, events, record_updates)
        pending = record.get('pending_forks', {}).get('worker_001')
        if crash_at == 'reservation' and pending and pending['phase'] == 'reservation':
            os._exit(23)
        if crash_at == 'registered-locked' and pending and pending['phase'] == 'registered-locked':
            os._exit(23)
        if crash_at == 'registered-open' and pending and pending['phase'] == 'registered-open':
            os._exit(23)

    def phase(root_path, name, value):
        original_phase(root_path, name, value)
        if crash_at == 'complete-staging' and value == 'staged':
            os._exit(23)
        if crash_at == 'renamed-locked' and value == 'renamed-locked':
            os._exit(23)

    def rename(stage, target):
        original_rename(stage, target)
        if crash_at == 'renamed-before-phase':
            os._exit(23)

    def register(root_path, name, target):
        original_register(root_path, name, target)
        if crash_at == 'context-registered':
            os._exit(23)

    def publish(target):
        original_publish(target)
        if crash_at == 'registered-open-before-phase':
            os._exit(23)

    def write_context(*args, **kwargs):
        if crash_at == 'partial-staging':
            (Path(args[3]) / '.benchmark_context').mkdir()
            os._exit(23)
        return original_write_context(*args, **kwargs)

    def fchown(descriptor, uid, gid):
        if crash_at == 'pre-root-normalization':
            stage = Path(root) / 'staging/forks/worker_001'
            try:
                stage_metadata = stage.stat()
            except FileNotFoundError:
                pass
            else:
                metadata = os.fstat(descriptor)
                if ((metadata.st_dev, metadata.st_ino) ==
                        (stage_metadata.st_dev, stage_metadata.st_ino)
                        and all((stage / name).exists() for name in
                                ('candidate', '.benchmark_context', '.benchmark_outputs'))):
                    # Crash immediately before _normalize_workspace's final root
                    # chown, after all required entries have been normalized.
                    os._exit(23)
        return original_fchown(descriptor, uid, gid)

    session._commit = commit
    session._set_fork_phase = phase
    session._rename_fork_stage = rename
    session._context_root_registered = register
    workspace.publish_workspace_root = publish
    workspace.write_context = write_context
    session.os.fchown = fchown
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
    def test_create_rejects_runs_root_the_agent_cannot_traverse(self):
        denied_runs = self.repo / 'unapproved_runs'
        denied_runs.mkdir(mode=0o700)
        denied_runs.chmod(0o700)
        denied_snapshots = self.repo / 'denied_snapshots'

        with self.assertRaises(session.SessionError) as caught:
            self.create(output={'runs_root': denied_runs, 'snapshots_root': denied_snapshots})

        self.assertEqual(caught.exception.error_code, 'invalid_request')
        self.assertIn('runs_root', str(caught.exception))
        self.assertIn('ptobench-agent', str(caught.exception))
        self.assertEqual(list(denied_runs.iterdir()), [])
        self.assertFalse(denied_snapshots.exists())

    def test_retired_schema_is_rejected_before_repair_without_changing_session_tree(self):
        handle = self.create()
        record = session.read_session(handle.session_root)
        retired = {key: value for key, value in record.items() if key != 'pending_forks'}
        retired['schema_version'] = 1
        integrity.commit_records(handle.protection_root, {'session': json.dumps(retired).encode()})
        (handle.session_root / 'session.json').write_text('{"damaged": true}\n')
        before = tree_digest(handle.session_root)

        with self.assertRaises(session.SessionError) as caught:
            session.read_session(handle.session_root)

        self.assertEqual(caught.exception.error_code, 'invalid_request')
        self.assertIn('fresh Session', str(caught.exception))
        self.assertEqual(tree_digest(handle.session_root), before)

    def test_exported_workspace_marker_is_rejected_before_record_repair(self):
        handle = self.create()
        marker = self.repo / 'retired_candidate_pool'
        marker.mkdir()
        manifest = common.read_json(handle.protection_root / 'manifest.json')
        manifest['roots']['candidate_pool'] = {
            'path': str(marker), 'group': 'retired', 'readonly': False,
            'entries': {'': {'kind': 'directory', 'mode': 0o755}},
        }
        write_json(handle.protection_root / 'manifest.json', manifest)
        (handle.session_root / 'session.json').write_text('{"damaged": true}\n')
        before = tree_digest(handle.session_root)

        with self.assertRaises(session.SessionError) as caught:
            session.read_session(handle.session_root)

        self.assertEqual(caught.exception.error_code, 'invalid_request')
        self.assertEqual(tree_digest(handle.session_root), before)

    def test_preflight_rejects_missing_or_nondefault_baseline_tuning(self):
        _, sealed, _ = common.case_paths(self.case_id)
        baseline = common.read_json(sealed / 'ref_performance.json')
        for tuning in (None, {'memory_planner': 'PTOAS', 'analyze_auto_scopes_for_deps': False},
                       {'memory_planner': 'PYPTO', 'analyze_auto_scopes_for_deps': True}):
            value = dict(baseline)
            if tuning is None:
                value.pop('compiler_tuning')
            else:
                value['compiler_tuning'] = tuning
            write_json(sealed / 'ref_performance.json', value)
            with self.subTest(tuning=tuning), self.assertRaisesRegex(
                    session.SessionError, 'must explicitly record PYPTO'):
                self.create()
        write_json(sealed / 'ref_performance.json', baseline)
        self.create()

    def test_method_identity_does_not_load_a_solver_or_set_evaluator_policy(self):
        # No such method directory exists. The evaluator accepts only its label,
        # and no method instructions or native tool are needed to create/fork.
        handle = session.create_session(self.case_id, 'external_tool', self.output,
                                        mode='independent_participant')
        record = session.read_session(handle.session_root)
        self.assertEqual(record['mode'], {'id': 'independent_participant'})
        self.assertEqual(record['profiling_modes'], ['combined'])
        fork_id = session.fork_workspace(handle.session_root, handle.initial_workspace)
        fork = handle.session_root / 'workspaces' / fork_id
        for location in (handle.initial_workspace, fork):
            self.assertFalse((location / '.benchmark_context/mode.json').exists())

    def test_creation_initializes_independent_layouts_without_running_case(self):
        self.make_case('test.other')
        with mock.patch.object(workspace, 'init_workspace', wraps=workspace.init_workspace) as init:
            first = self.create()
            second = self.create(5, 'pi', 'test.other')
        self.assertEqual(init.call_count, 2)
        for handle in (first, second):
            record = session.read_session(handle.session_root)
            self.assertEqual(record['status'], 'running')
            self.assertEqual(record['schema_version'], 2)
            self.assertEqual(record['pending_forks'], {})
            self.assertEqual(record['usage'], {'hardware_actions': 0, 'finish_actions': 0})
            self.assertEqual(record['next_event_seq'], 1)
            self.assertEqual(record['workspaces']['worker_000']['path'], 'workspaces/worker_000')
            self.assertRegex(record['session_id'], r'^\d{8}T\d{6}Z_[a-f0-9]{12}$')
            self.assertEqual(record['profiling_modes'], ['combined'])
            for name in ('workspaces', 'staging', 'final', 'protection'):
                self.assertTrue((handle.session_root / name).is_dir())
            self.assertEqual(stat.S_IMODE(handle.session_root.stat().st_mode), 0o2710)
            self.assertEqual(stat.S_IMODE((handle.session_root / 'workspaces').stat().st_mode), 0o2750)
            self.assertEqual(handle.session_root.stat().st_gid, grp.getgrnam('ptobench-share').gr_gid)
            self.assertTrue((handle.session_root / 'process.log').is_file())
            self.assertTrue(handle.snapshots_root.is_dir())
            self.assertFalse((handle.session_root / 'final_result.json').exists())
            self.assertEqual((handle.session_root / 'events.jsonl').read_bytes(), b'')
            manifest = common.read_json(handle.protection_root / 'manifest.json')
            self.assertTrue({'case', 'sealed', 'environment', 'evaluator', 'golden', 'public_ptobench', 'public_launcher'} <= manifest['roots'].keys())
            self.assertTrue({'public_modes_package', 'public_modes_basic', 'public_modes_shared',
                             'public_modes_cannbot'} <= manifest['roots'].keys())
            self.assertNotIn('public_modes', manifest['roots'])
            self.assertFalse(any(Path(root['path']).is_relative_to(
                common.BENCHMARK_ROOT / 'modes/ptomni') for root in manifest['roots'].values()))
            self.assertTrue(any(n.startswith('family_') for n in manifest['roots']))
            self.assertEqual(stat.S_IMODE(handle.initial_workspace.stat().st_mode), 0o3770)
            self.assertEqual(handle.initial_workspace.stat().st_gid, grp.getgrnam('ptobench-share').gr_gid)
            self.assertEqual((handle.initial_workspace / '.benchmark_context/config.py').stat().st_mode & 0o777, 0o640)
            self.assertTrue(integrity.check_only(handle.protection_root).passed)
        self.assertNotEqual(first.session_root, second.session_root)
        self.assertEqual(session.read_session(first.session_root)['budget']['max_hardware_actions'], 2)
        self.assertEqual(session.read_session(second.session_root)['budget']['max_hardware_actions'], 5)

    def test_published_workspace_root_uses_sticky_mode(self):
        handle = self.create()
        child_id = session.fork_workspace(handle.session_root, handle.initial_workspace)
        for root in (handle.initial_workspace, handle.session_root / 'workspaces' / child_id):
            metadata = root.stat()
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o3770)
            self.assertEqual(metadata.st_uid, os.getuid())
            self.assertEqual(metadata.st_gid, grp.getgrnam('ptobench-share').gr_gid)
        self.assertEqual(stat.S_IMODE((handle.session_root / 'workspaces').stat().st_mode), 0o2750)

    def test_unrelated_mode_tree_changes_do_not_enter_session_integrity_checks(self):
        handle = self.create()
        unrelated = common.BENCHMARK_ROOT / 'modes/ptomni/runner.py'
        unrelated.write_text('changed while Session is active\n')

        report = integrity.check_and_restore(handle.protection_root)

        self.assertTrue(report.passed)
        self.assertFalse(any('ptomni' in path for path in report.changed_paths))

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
        self.assertEqual(session._budget(None), {'max_hardware_actions': 20, 'finish_actions': 1})
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(session.SessionError):
                self.create(budget=budget)
        for value in ({'finish_actions': 2}, {'finish_actions': True},
                      {'max_profile_actions': 1}, {'unknown': 1}, [], False):
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
        self.assertEqual(session.read_session(handle.session_root)['profiling_modes'], ['combined'])
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
        (handle.initial_workspace / 'candidate/helper.py').write_text('VALUE = 9\n')
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
        self.assertEqual((handle.initial_workspace / 'candidate/helper.py').read_text(), 'VALUE = 9\n')
        self.assertEqual((handle.initial_workspace / '.benchmark_context/starting_point/candidate/helper.py').read_text(), 'VALUE = 1\n')
        self.assertTrue(integrity.check_only(handle.protection_root).passed)

    def test_context_integrity_repair_restores_shared_group_access(self):
        handle = self.create()
        context = handle.initial_workspace / '.benchmark_context'
        config = context / 'config.py'
        config.write_text('ROGUE = True\n')
        shutil.rmtree(context / 'starting_point')

        with session.execution_lock():
            report = session.repair_assets(handle.session_root)

        share_gid = grp.getgrnam('ptobench-share').gr_gid
        self.assertIn('context_worker_000/config.py', report.restored_paths)
        self.assertIn('context_worker_000/starting_point', report.restored_paths)
        self.assertEqual(config.read_text(), 'SIZE = 2\n')
        for path in (config, context / 'starting_point',
                     context / 'starting_point/candidate',
                     context / 'starting_point/candidate/helper.py'):
            self.assertEqual(path.stat().st_gid, share_gid, path)

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
        (handle.initial_workspace / 'candidate/helper.py').write_text('VALUE = 2\n')
        (handle.initial_workspace / '.benchmark_outputs/cache').mkdir(parents=True)
        (handle.initial_workspace / '.benchmark_outputs/cache/junk').write_text('excluded')
        first = session.fork_workspace(handle.session_root, handle.initial_workspace)
        child = handle.session_root / 'workspaces' / first
        self.assertEqual(first, 'worker_001')
        self.assertEqual((child / 'candidate/helper.py').read_text(), 'VALUE = 2\n')
        self.assertEqual((child / '.benchmark_context/starting_point/candidate/helper.py').read_text(), 'VALUE = 2\n')
        self.assertTrue((child / '.benchmark_outputs').is_dir())
        (child / 'candidate/helper.py').write_text('VALUE = 3\n')
        self.assertEqual((handle.initial_workspace / 'candidate/helper.py').read_text(), 'VALUE = 2\n')
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

    def test_fork_crash_recovery_keeps_every_intermediate_worker_closed(self):
        stages = ('reservation', 'partial-staging', 'pre-root-normalization', 'complete-staging',
                  'renamed-before-phase', 'renamed-locked', 'context-registered',
                  'registered-locked', 'registered-open-before-phase', 'registered-open')
        for crash_at in stages:
            with self.subTest(crash_at=crash_at):
                handle = self.create()
                process = multiprocessing.get_context('fork').Process(
                    target=crash_fork, args=(self.bench, self.repo, handle.session_root, crash_at))
                process.start()
                process.join(10)
                self.assertEqual(process.exitcode, 23)
                record = common.read_json(handle.session_root / 'session.json')
                self.assertEqual(record['next_workspace_index'], 2)
                published = handle.session_root / 'workspaces/worker_001'
                staging = handle.session_root / 'staging/forks/worker_001'
                if crash_at in ('reservation', 'partial-staging', 'pre-root-normalization'):
                    with self.assertRaises(session.SessionError):
                        session.read_session(handle.session_root)
                    if crash_at == 'pre-root-normalization':
                        share_gid = grp.getgrnam('ptobench-share').gr_gid
                        self.assertFalse(published.exists())
                        self.assertEqual(stat.S_IMODE(staging.stat().st_mode), 0o700)
                        self.assertEqual(staging.stat().st_gid, share_gid)
                        for name, mode in (('candidate', 0o2770),
                                           ('.benchmark_context', 0o750),
                                           ('.benchmark_outputs', 0o750)):
                            entry = (staging / name).stat()
                            self.assertEqual(entry.st_gid, share_gid)
                            self.assertEqual(stat.S_IMODE(entry.st_mode), mode)
                elif crash_at == 'complete-staging':
                    self.assertFalse(published.exists())
                    self.assertEqual(stat.S_IMODE(staging.stat().st_mode), 0o700)
                else:
                    self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o700
                                     if crash_at in ('complete-staging', 'renamed-before-phase',
                                                     'renamed-locked', 'context-registered',
                                                     'registered-locked') else 0o3770)
                    with self.assertRaises(session.SessionError):
                        session.resolve_workspace(handle.session_root, published)
                with session.execution_lock():
                    recovered = session.recover_pending_forks(handle.session_root)
                record = session.read_session(handle.session_root)
                self.assertEqual(record['pending_forks'], {})
                if crash_at in ('reservation', 'partial-staging'):
                    self.assertEqual(recovered, [])
                    self.assertNotIn('worker_001', record['workspaces'])
                    self.assertFalse(published.exists())
                else:
                    self.assertIn('worker_001', record['workspaces'])
                    self.assertTrue((published / '.benchmark_context/starting_point/candidate/helper.py').is_file())
                    self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o3770)
                self.assertEqual(list((handle.session_root / 'staging/forks').iterdir()), [])
                self.assertEqual(session.fork_workspace(handle.session_root, handle.initial_workspace), 'worker_002')
                self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 0)
                self.assertTrue(integrity.check_only(handle.protection_root).passed)
                print(f'S2 fork crash: {crash_at} recovered, next accepted worker=002, usage=0')

    def test_launcher_poll_can_read_session_during_pending_fork(self):
        handle = self.create()
        with session._session_lock(handle.session_root) as root:
            record, _ = session._load_locked(root)
            record['next_workspace_index'] = 2
            record['pending_forks']['worker_001'] = {
                'worker_id': 'worker_001', 'parent': 'worker_000',
                'staging': 'staging/forks/worker_001', 'phase': 'reservation',
            }
            session._commit(root, record)

        with self.assertRaisesRegex(session.SessionError, 'unfinished Workspace fork'):
            session.read_session(handle.session_root)
        state = session.read_session(handle.session_root, allow_pending=True)
        self.assertEqual(state['pending_forks']['worker_001']['phase'], 'reservation')

    def test_fork_recovery_rejects_target_for_unstaged_reservation(self):
        handle = self.create()
        process = multiprocessing.get_context('fork').Process(
            target=crash_fork, args=(self.bench, self.repo, handle.session_root, 'complete-staging'))
        process.start()
        process.join(10)
        self.assertEqual(process.exitcode, 23)

        stage = handle.session_root / 'staging/forks/worker_001'
        target = handle.session_root / 'workspaces/worker_001'
        with session.execution_lock(), session._session_lock(handle.session_root) as root:
            record, _ = session._load_locked(root, allow_pending=True)
            record['pending_forks']['worker_001']['phase'] = 'reservation'
            session._commit(root, record)
        os.rename(stage, target)

        with session.execution_lock(), self.assertRaises(integrity.InfrastructureError):
            session.recover_pending_forks(handle.session_root)

        record = common.read_json(handle.session_root / 'session.json')
        manifest = common.read_json(handle.protection_root / 'manifest.json')
        self.assertEqual(record['pending_forks']['worker_001']['phase'], 'reservation')
        self.assertNotIn('worker_001', record['workspaces'])
        self.assertNotIn('context_worker_001', manifest['roots'])
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)

    def test_fork_recovery_discards_invalid_unpublished_reservation(self):
        handle = self.create()
        process = multiprocessing.get_context('fork').Process(
            target=crash_fork, args=(self.bench, self.repo, handle.session_root, 'complete-staging'))
        process.start()
        process.join(10)
        self.assertEqual(process.exitcode, 23)

        stage = handle.session_root / 'staging/forks/worker_001'
        with session.execution_lock(), session._session_lock(handle.session_root) as root:
            record, _ = session._load_locked(root, allow_pending=True)
            record['pending_forks']['worker_001']['phase'] = 'reservation'
            session._commit(root, record)
        (stage / '.benchmark_context').chmod(0o700)

        with session.execution_lock():
            recovered = session.recover_pending_forks(handle.session_root)

        record = session.read_session(handle.session_root)
        manifest = common.read_json(handle.protection_root / 'manifest.json')
        self.assertEqual(recovered, [])
        self.assertEqual(record['pending_forks'], {})
        self.assertNotIn('worker_001', record['workspaces'])
        self.assertNotIn('context_worker_001', manifest['roots'])
        self.assertFalse(stage.exists())

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
