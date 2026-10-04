from __future__ import annotations

import json
import multiprocessing
import os
import queue as queue_module
import subprocess
import sys
from pathlib import Path
import time
from unittest import mock

from session_test_support import SessionTestBase
import common
import integrity
import session
import workspace


def configure(bench, repo):
    common.BENCHMARK_ROOT = Path(bench)
    common.REPO_ROOT = Path(repo)


def contender(bench, repo, root, ready, queue, waiting=None):
    configure(bench, repo)
    ready.wait(10)
    if waiting is not None:
        waiting.set()
    try:
        with session.execution_lock():
            record = session.read_session(root)
            source = Path(root) / 'workspaces/worker_000'
            manifest = workspace.copy_candidate_tree(source, Path(root) / f'staging/try_{os.getpid()}')
            usage = session.reserve_action(root, record['next_event_seq'], 'evaluate', workspace_path=source, source_manifest=manifest)
            session.complete_action(root, record['next_event_seq'], exit_code=1, error_code='action_failed')
            queue.put(('accepted', usage['hardware_actions']))
    except session.SessionError as error:
        queue.put(('rejected', error.error_code))


def hold_last_action(bench, repo, root, accepted, release):
    configure(bench, repo)
    with session.execution_lock():
        record = session.read_session(root)
        source = Path(root) / 'workspaces/worker_000'
        manifest = workspace.copy_candidate_tree(source, Path(root) / 'staging/frozen')
        session.reserve_action(root, record['next_event_seq'], 'evaluate', workspace_path=source, source_manifest=manifest)
        accepted.set()
        if not release.wait(10):
            raise RuntimeError('test release timed out')
        session.complete_action(root, record['next_event_seq'], exit_code=0, integrity_passed=True, result={'verdict': 'PASS'}, snapshot_id='snapshot_000001')


def lock_probe(bench, repo, entered):
    configure(bench, repo)
    with session.execution_lock():
        entered.set()


def status_reader(bench, repo, root, queue):
    configure(bench, repo)
    queue.put(session.read_session(root))


def child_wait(release, ready=None, done=None):
    os.setsid()
    if ready is not None:
        ready.set()
    release.wait(10)
    if done is not None:
        done.set()


def leave_dead_action(bench, repo, root):
    configure(bench, repo)
    context = multiprocessing.get_context('fork')
    release, ready = context.Event(), context.Event()
    with session.execution_lock():
        source = Path(root) / 'workspaces/worker_000'
        manifest = workspace.copy_candidate_tree(source, Path(root) / 'staging/frozen_dead')
        session.reserve_action(root, 1, 'evaluate', workspace_path=source, source_manifest=manifest)
        session.mark_action_starting(root, 1)
        child = context.Process(target=child_wait, args=(release, ready))
        child.start()
        if not ready.wait(5):
            raise RuntimeError('fake child did not establish its process group')
        session.set_action_process(root, 1, child.pid)
        release.set()
        child.join(5)
        if child.exitcode != 0:
            raise RuntimeError('fake child did not finish')
    # Owner and the recorded child both die with a reserved action left behind.


def leave_unbound_action(bench, repo, root, starting):
    configure(bench, repo)
    with session.execution_lock():
        source = Path(root) / 'workspaces/worker_000'
        session.reserve_action(root, 1, 'evaluate', workspace_path=source, source_manifest=workspace.source_manifest(source))
        if starting:
            session.mark_action_starting(root, 1)


def leave_live_action(bench, repo, root, input_fd):
    configure(bench, repo)
    with session.execution_lock():
        source = Path(root) / 'workspaces/worker_000'
        session.reserve_action(root, 1, 'evaluate', workspace_path=source, source_manifest=workspace.source_manifest(source))
        session.mark_action_starting(root, 1)
        child = subprocess.Popen([sys.executable, '-c',
            "import sys; print('ready', flush=True); sys.stdin.buffer.read(1)"],
            stdin=input_fd, stdout=subprocess.PIPE, start_new_session=True)
        if child.stdout.readline() != b'ready\n':
            raise RuntimeError('fake child not ready')
        session.set_action_process(root, 1, child.pid)
    os._exit(23)


def crash_during_completion(bench, repo, root, point):
    configure(bench, repo)
    with session.execution_lock():
        source = Path(root) / 'workspaces/worker_000'
        manifest = workspace.copy_candidate_tree(source, Path(root) / 'staging/frozen_crash')
        session.reserve_action(root, 1, 'evaluate', workspace_path=source, source_manifest=manifest)
        original = integrity._write_manifest
        def crash(protection, manifest):
            if point == 'before_manifest':
                os._exit(23)
            original(protection, manifest)
            os._exit(23)
        integrity._write_manifest = crash
        session.complete_action(root, 1, exit_code=1, error_code='action_failed')


class BudgetTest(SessionTestBase):
    def process(self, target, args):
        process = multiprocessing.get_context('fork').Process(target=target, args=args)
        process.start()
        self.addCleanup(lambda: process.kill() if process.is_alive() else None)
        return process

    def join(self, process):
        process.join(10)
        self.assertFalse(process.is_alive(), 'test process did not finish')
        self.assertEqual(process.exitcode, 0)

    def test_multiprocess_last_allowance_never_exceeds_limit(self):
        context = multiprocessing.get_context('fork')
        for limit in (1, 3):
            with self.subTest(limit=limit):
                handle = self.create(budget=limit)
                ready, queue = context.Event(), context.Queue()
                processes = [self.process(contender, (self.bench, self.repo, handle.session_root, ready, queue)) for _ in range(5)]
                ready.set()
                outcomes = [queue.get(timeout=10) for _ in processes]
                for process in processes:
                    self.join(process)
                accepted = [v for k, v in outcomes if k == 'accepted']
                self.assertEqual(sorted(accepted), list(range(1, limit + 1)))
                self.assertEqual(sum(k == 'rejected' for k, _ in outcomes), 5 - limit)
                self.assertEqual({v for k, v in outcomes if k == 'rejected'}, {'budget_exceeded'})
                record = session.read_session(handle.session_root)
                self.assertEqual(record['usage']['hardware_actions'], limit)
                self.assertEqual(record['status'], 'exhausted')
                self.assertIsNone(record['active_action'])
                self.assertEqual(len((handle.session_root / 'events.jsonl').read_text().splitlines()), limit)
                print(f'S2 concurrency: {len(processes)} processes, limit={limit}, accepted={len(accepted)}, rejected={5-limit}')

    def test_status_and_close_do_not_wait_for_active_global_measurement(self):
        handle = self.create(budget=1)
        context = multiprocessing.get_context('fork')
        accepted, release, entered = context.Event(), context.Event(), context.Event()
        worker = self.process(hold_last_action, (self.bench, self.repo, handle.session_root, accepted, release))
        self.assertTrue(accepted.wait(10))
        # A reader finishes while the writer still owns the execution lock.
        queue = context.Queue()
        reader = self.process(status_reader, (self.bench, self.repo, handle.session_root, queue))
        record = queue.get(timeout=5)
        self.join(reader)
        self.assertEqual(record['status'], 'exhausted')
        self.assertIsNotNone(record['active_action'])
        self.assertTrue(worker.is_alive())
        session.close_exploration(handle.session_root, 'budget_exhausted')
        probe = self.process(lock_probe, (self.bench, self.repo, entered))
        self.assertFalse(entered.wait(0.1), 'custom output must use the same fixed global lock')
        release.set()
        self.join(worker)
        self.join(probe)
        record = session.read_session(handle.session_root)
        self.assertIsNone(record['active_action'])
        self.assertTrue(record['exploration_closed'])
        event = json.loads((handle.session_root / 'events.jsonl').read_text())
        self.assertEqual(event['result']['verdict'], 'PASS')
        self.assertEqual(event['snapshot_id'], 'snapshot_000001')
        print('S2 last action: exhausted + active visible during lock; close preserved PASS completion')

    def test_invalid_prechecks_do_not_charge_and_global_lock_is_required(self):
        handle = self.create()
        manifest = workspace.source_manifest(handle.initial_workspace)
        with self.assertRaises(integrity.InfrastructureError):
            session.reserve_action(handle.session_root, 1, 'evaluate', workspace_path=handle.initial_workspace, source_manifest=manifest)
        with session.execution_lock():
            cases = [('evaluate', None, self.repo, manifest, 1), ('profile', 'hardware', handle.initial_workspace, manifest, 1),
                     ('profile', 'compile', handle.initial_workspace, manifest, 1), ('evaluate', None, handle.initial_workspace, {}, 1),
                     ('unknown', None, handle.initial_workspace, manifest, 1), ('evaluate', None, handle.initial_workspace, manifest, 99)]
            for kind, mode, path, value, seq in cases:
                with self.subTest(kind=kind, mode=mode, seq=seq), self.assertRaises(session.SessionError):
                    session.reserve_action(handle.session_root, seq, kind, workspace_path=path, mode=mode, source_manifest=value)
            rogue = handle.initial_workspace / 'link.py'
            rogue.symlink_to(self.repo / 'outside')
            with self.assertRaises(workspace.InvalidWorkspaceError):
                session.reserve_action(handle.session_root, 1, 'evaluate', workspace_path=handle.initial_workspace, source_manifest=manifest)
            rogue.unlink()
            (self.bench / 'cases/test/case/public/config.py').write_text('tampered\n')
            with self.assertRaises(session.SessionError) as caught:
                session.reserve_action(handle.session_root, 1, 'evaluate', workspace_path=handle.initial_workspace, source_manifest=manifest)
            self.assertEqual(caught.exception.error_code, 'integrity_failed')
            session.repair_assets(handle.session_root)
        self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 0)

    def test_failures_after_reservation_never_refund_and_completion_is_idempotent(self):
        handle = self.create(budget=5)
        for seq, error in enumerate(('action_failed', 'integrity_failed', 'interrupted', 'infrastructure_error', 'action_failed'), 1):
            with self.subTest(error=error), session.execution_lock():
                used = self.reserve(handle)
                self.assertEqual(used['hardware_actions'], seq)
                event = session.complete_action(handle.session_root, seq, exit_code=1, error_code=error, action_started=seq != 4)
                repeated = session.complete_action(handle.session_root, seq, exit_code=0, integrity_passed=True, result={'forged': True})
                self.assertEqual(repeated, event)
                self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], seq)
        self.assertEqual(session.read_session(handle.session_root)['status'], 'exhausted')
        with self.assertRaises(session.SessionError) as caught:
            session.fork_workspace(handle.session_root, handle.initial_workspace)
        self.assertEqual(caught.exception.error_code, 'budget_exceeded')

    def test_finish_is_independent_requires_closed_exploration_and_cannot_repeat(self):
        handle = self.create(budget=1)
        with session.execution_lock():
            self.reserve(handle)
            session.complete_action(handle.session_root, 1, exit_code=0, integrity_passed=True)
        workspace.copy_candidate_tree(handle.initial_workspace, handle.session_root / 'final/workspace')
        with session.execution_lock(), self.assertRaises(session.SessionError):
            self.reserve(handle, 'finish')
        self.assertEqual(session.read_session(handle.session_root)['usage']['finish_actions'], 0)
        session.close_exploration(handle.session_root, 'optimizer_exit')
        with session.execution_lock():
            usage = self.reserve(handle, 'finish')
            self.assertEqual(usage, {'hardware_actions': 1, 'finish_actions': 1})
            session.complete_action(handle.session_root, 2, exit_code=0, integrity_passed=True, result={'verdict': 'PASS'})
            record = session.read_session(handle.session_root)
            self.assertEqual(record['status'], 'completed')
            self.assertIsNotNone(record['completed_at'])
            with self.assertRaises(session.SessionError):
                session.reserve_action(handle.session_root, 3, 'finish', workspace_path=handle.session_root / 'final/workspace', source_manifest=workspace.source_manifest(handle.session_root / 'final/workspace'))
        self.assertEqual(session.read_session(handle.session_root)['usage'], usage)

    def test_close_preserves_active_action_and_finish_waits_for_it(self):
        handle = self.create()
        workspace.copy_candidate_tree(handle.initial_workspace, handle.session_root / 'final/workspace')
        with session.execution_lock():
            self.reserve(handle)
            active = session.read_session(handle.session_root)['active_action']
            session.close_exploration(handle.session_root, 'optimizer_exit')
            session.close_exploration(handle.session_root, 'different_reason')
            record = session.read_session(handle.session_root)
            self.assertEqual(record['active_action'], active)
            self.assertEqual(record['close_reason'], 'optimizer_exit')
            with self.assertRaises(session.SessionError):
                session.reserve_action(handle.session_root, 2, 'finish', workspace_path=handle.session_root / 'final/workspace', source_manifest=workspace.source_manifest(handle.session_root / 'final/workspace'))
            session.complete_action(handle.session_root, 1, exit_code=130, error_code='interrupted')
            self.reserve(handle, 'finish')
            session.complete_action(handle.session_root, 2, exit_code=1, error_code='action_failed')
        self.assertEqual(session.read_session(handle.session_root)['status'], 'failed')

    def test_disabled_profile_is_frozen_in_session_and_not_from_metadata(self):
        with mock.patch.object(session.profile_modes, 'enabled_modes', return_value={'runtime': {'description': 'Runtime diagnostics'}}):
            handle = self.create()
        with session.execution_lock():
            with self.assertRaises(session.SessionError):
                self.reserve(handle, 'profile', 'pmu')
            # Failed staging precheck wasn't accepted; use a fresh, valid manifest.
            record = session.read_session(handle.session_root)
            self.assertEqual(record['usage']['hardware_actions'], 0)
            session.reserve_action(handle.session_root, 1, 'profile', mode='runtime', workspace_path=handle.initial_workspace,
                                   source_manifest=workspace.source_manifest(handle.initial_workspace))
            session.complete_action(handle.session_root, 1, exit_code=0, integrity_passed=True)
        self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 1)

    def test_lock_order_rejected_and_state_reads_need_no_global_lock(self):
        handle = self.create()
        with session._session_lock(handle.session_root):
            with self.assertRaises(integrity.InfrastructureError):
                with session.execution_lock():
                    pass
        with session.execution_lock():
            with self.assertRaises(integrity.InfrastructureError):
                with session.execution_lock():
                    pass
        self.assertEqual(session.read_session(handle.session_root)['status'], 'running')
        lock = self.bench / '.ptobench/execution.lock'
        lock.unlink()
        lock.symlink_to(handle.session_root / 'process.log')
        with self.assertRaises(integrity.InfrastructureError):
            with session.execution_lock():
                pass
        self.assertEqual((handle.session_root / 'process.log').read_bytes(), b'')

    def test_dead_owner_and_child_reconciled_once_without_refund(self):
        handle = self.create(budget=1)
        worker = self.process(leave_dead_action, (self.bench, self.repo, handle.session_root))
        self.join(worker)
        record = session.read_session(handle.session_root)
        self.assertIsNotNone(record['active_action']['process'])
        with session.execution_lock():
            event = session.reconcile_active_action(handle.session_root)
            repeated = session.reconcile_active_action(handle.session_root)
        self.assertIsNone(repeated)
        self.assertEqual(event['error_code'], 'interrupted')
        self.assertFalse(event['integrity_passed'])
        record = session.read_session(handle.session_root)
        self.assertIsNone(record['active_action'])
        self.assertTrue(record['exploration_closed'])
        self.assertEqual(record['usage']['hardware_actions'], 1)
        self.assertEqual(len((handle.session_root / 'events.jsonl').read_text().splitlines()), 1)
        print('S2 residual: dead owner + child -> one interrupted event; budget retained')

    def test_live_action_is_preserved_and_unattributed_spawn_blocks_finish(self):
        handle = self.create()
        with session.execution_lock():
            self.reserve(handle)
            with self.assertRaises(session.SessionError):
                session.reconcile_active_action(handle.session_root)
        record = session.read_session(handle.session_root)
        self.assertEqual(record['status'], 'running')
        self.assertIsNotNone(record['active_action'])
        self.assertIsNotNone(session.process_identity(os.getpid()))
        other = self.create(optimizer='pi')
        worker = self.process(leave_unbound_action, (self.bench, self.repo, other.session_root, True))
        self.join(worker)
        with session.execution_lock(), self.assertRaises(integrity.InfrastructureError):
            session.reconcile_active_action(other.session_root)
        record = session.read_session(other.session_root)
        self.assertEqual(record['status'], 'failed')
        self.assertIsNotNone(record['active_action'])
        self.assertEqual(record['close_reason'], 'unresolved_active_process')
        self.assertIsNotNone(session.process_identity(os.getpid()))
        self.assertEqual(record['usage']['hardware_actions'], 1)

    def test_dead_unstarted_action_records_interrupted_without_refund(self):
        handle = self.create()
        worker = self.process(leave_unbound_action, (self.bench, self.repo, handle.session_root, False))
        self.join(worker)
        with session.execution_lock():
            event = session.reconcile_active_action(handle.session_root)
        self.assertEqual(event['error_code'], 'interrupted')
        self.assertFalse(event['action_started'])
        self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 1)

    def test_real_crash_completion_recovers_one_consistent_generation(self):
        for point in ('before_manifest', 'after_manifest'):
            with self.subTest(point=point):
                handle = self.create()
                process = self.process(crash_during_completion, (self.bench, self.repo, handle.session_root, point))
                process.join(10)
                self.assertEqual(process.exitcode, 23)
                record = session.read_session(handle.session_root)
                lines = (handle.session_root / 'events.jsonl').read_text().splitlines()
                self.assertEqual(record['usage']['hardware_actions'], 1)
                if point == 'before_manifest':
                    self.assertIsNotNone(record['active_action'])
                    self.assertEqual(lines, [])
                else:
                    self.assertIsNone(record['active_action'])
                    self.assertEqual(json.loads(lines[0])['error_code'], 'action_failed')
                print(f'S2 completion crash: {point}, used=1, active={record["active_action"] is not None}, events={len(lines)}')

    def test_all_public_profiles_charge_once_and_old_modes_never_reserve(self):
        handle = self.create(budget=3)
        for seq, mode in enumerate(('runtime', 'pmu', 'compiler'), 1):
            with session.execution_lock():
                self.reserve(handle, 'profile', mode)
                event = session.complete_action(handle.session_root, seq, exit_code=0, integrity_passed=True)
                self.assertEqual(event['mode'], mode)
                self.assertEqual(event['hardware_actions_used'], seq)
        self.assertEqual(session.read_session(handle.session_root)['usage'], {'hardware_actions': 3, 'finish_actions': 0})

    def test_close_during_prechecks_prevents_reservation_without_charge(self):
        handle = self.create()
        original = workspace.source_manifest
        def concurrent_close(path):
            session.close_exploration(handle.session_root, 'optimizer_exit')
            return original(path)
        manifest = original(handle.initial_workspace)
        with session.execution_lock(), mock.patch.object(workspace, 'source_manifest', side_effect=concurrent_close):
            with self.assertRaises(session.SessionError) as caught:
                session.reserve_action(handle.session_root, 1, 'evaluate', workspace_path=handle.initial_workspace, source_manifest=manifest)
        self.assertEqual(caught.exception.error_code, 'session_closed')
        self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 0)

    def test_idempotent_old_completion_preserves_the_next_active_action(self):
        handle = self.create()
        with session.execution_lock():
            self.reserve(handle)
            first = session.complete_action(handle.session_root, 1, exit_code=1, error_code='action_failed')
            self.reserve(handle)
            active = session.read_session(handle.session_root)['active_action']
            self.assertEqual(session.complete_action(handle.session_root, 1, exit_code=0), first)
            self.assertEqual(session.read_session(handle.session_root)['active_action'], active)
            session.complete_action(handle.session_root, 2, exit_code=1, error_code='action_failed')

    def test_two_sessions_with_different_outputs_wait_on_one_lock_without_charging(self):
        self.make_case('test.other')
        first = self.create(budget=1)
        second = self.create(budget=1, optimizer='pi', case_id='test.other',
                             output={'runs_root': self.repo / 'another_runs', 'snapshots_root': self.repo / 'another_snapshots'})
        context = multiprocessing.get_context('fork')
        accepted, release, ready, waiting = (context.Event() for _ in range(4))
        holder = self.process(hold_last_action, (self.bench, self.repo, first.session_root, accepted, release))
        self.assertTrue(accepted.wait(10))
        queue = context.Queue()
        waiter = self.process(contender, (self.bench, self.repo, second.session_root, ready, queue, waiting))
        ready.set()
        self.assertTrue(waiting.wait(5))
        self.assertEqual(session.read_session(second.session_root)['usage']['hardware_actions'], 0)
        with self.assertRaises(queue_module.Empty):
            queue.get(timeout=0.1)
        release.set()
        self.assertEqual(queue.get(timeout=10), ('accepted', 1))
        self.join(holder)
        self.join(waiter)
        for handle in (first, second):
            self.assertEqual(session.read_session(handle.session_root)['usage']['hardware_actions'], 1)
        self.assertTrue((self.bench / '.ptobench/execution.lock').is_file())
        self.assertFalse((self.output['runs_root'] / '.ptobench').exists())
        print('S2 cross-Session lock: distinct Optimizer/Case/output roots; waiter usage=0 before release, 1 afterwards')

    def test_actual_living_orphan_process_blocks_final_measurement(self):
        handle = self.create()
        input_fd, release_fd = os.pipe()
        worker = self.process(leave_live_action, (self.bench, self.repo, handle.session_root, input_fd))
        worker.join(10)
        os.close(input_fd)
        self.assertEqual(worker.exitcode, 23)
        identity = session.read_session(handle.session_root)['active_action']['process']
        self.assertEqual(session.process_identity(identity['pid']), identity)
        try:
            with session.execution_lock():
                with self.assertRaises(integrity.InfrastructureError):
                    session.reconcile_active_action(handle.session_root)
                record = session.read_session(handle.session_root)
                self.assertEqual(record['status'], 'failed')
                self.assertIsNotNone(record['active_action'])
                self.assertEqual(record['usage']['hardware_actions'], 1)
                with self.assertRaises(session.SessionError):
                    session.reserve_action(handle.session_root, 2, 'finish', workspace_path=handle.session_root / 'final/workspace',
                                           source_manifest=workspace.source_manifest(handle.initial_workspace))
            self.assertEqual(session.process_identity(identity['pid']), identity, 'S2 must not kill the fake orphan')
        finally:
            os.write(release_fd, b'x')
            os.close(release_fd)
            deadline = time.monotonic() + 5
            while session.process_identity(identity['pid']) is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNone(session.process_identity(identity['pid']))
        print('S2 living orphan: real PID/group remains alive; infrastructure failure blocks finish; test releases child')
