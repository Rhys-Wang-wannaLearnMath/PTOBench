"""Narrow Broker routing tests; no Case, Evaluate, Profile, or model process."""
import io
import grp
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch

from session_test_support import SessionTestBase
import integrity
import sandbox_supervisor
import session
from test_session import tree_digest


class BrokerWorkspaceRoutingTests(unittest.TestCase):
    def test_supervisor_fails_closed_without_supported_pi_isolation(self):
        with tempfile.TemporaryDirectory(prefix='supervisor-fail-closed-') as temporary:
            root = Path(temporary)
            request_path = root / 'request.json'
            request = {'context': {'session_root': str(root)},
                       'preflight': {'profile': {'backend': 'pi',
                                                 'mode_provider': {'base_url': 'https://example.invalid'}}}}
            request_path.write_text(json.dumps(request))
            with patch.dict(os.environ, {'PTOBENCH_SANDBOX': '0'}), \
                 patch.object(sandbox_supervisor, 'run_private_pi') as isolated:
                with self.assertRaisesRegex(RuntimeError, 'Pi Agent requires filesystem isolation'):
                    sandbox_supervisor.run('method.py', str(request_path))
                isolated.assert_not_called()

            request['preflight']['profile']['backend'] = 'codex'
            request_path.write_text(json.dumps(request))
            with patch.dict(os.environ, {'PTOBENCH_SANDBOX': '1'}), \
                 patch.object(sandbox_supervisor, 'run_private_pi') as isolated:
                with self.assertRaisesRegex(RuntimeError, 'only Pi Agent is supported'):
                    sandbox_supervisor.run('method.py', str(request_path))
                isolated.assert_not_called()

    def test_recovers_pending_forks_before_reading_session_or_opening_socket(self):
        with tempfile.TemporaryDirectory(prefix='broker-fork-recovery-') as temporary:
            root = Path(temporary) / 'session'
            private = Path(temporary) / 'private'
            root.mkdir()
            private.mkdir()
            events = []

            def recover(path):
                events.append(('recover', Path(path)))
                return []

            def read(path):
                events.append(('read', Path(path)))
                return {'workspaces': []}

            with patch.object(sandbox_supervisor.session, 'recover_pending_forks', side_effect=recover), \
                 patch.object(sandbox_supervisor.session, 'read_session', side_effect=read):
                broker = sandbox_supervisor.Broker(root, private)
            try:
                self.assertEqual(events, [('recover', root), ('read', root)])
            finally:
                broker.server.close()
                broker.ipc.cleanup()

    def test_fork_recovery_failure_prevents_session_read_broker_and_pi_start(self):
        with tempfile.TemporaryDirectory(prefix='broker-fork-recovery-failure-') as temporary:
            base = Path(temporary)
            root = base / 'session'
            root.mkdir()
            (base / 'bench/.ptobench/isolation').mkdir(parents=True)
            request = {'context': {'session_root': str(root), 'workspace': str(root)},
                       'preflight': {'profile': {'backend': 'pi'}}}
            failure = RuntimeError('pending fork recovery failed')
            private = base / 'bench/.ptobench/isolation' / root.name

            with patch.object(sandbox_supervisor.common, 'BENCHMARK_ROOT', base / 'bench'), \
                 patch('modes._shared.provider.api_key', return_value=None), \
                 patch.object(sandbox_supervisor.session, 'recover_pending_forks', side_effect=failure), \
                 patch.object(sandbox_supervisor.session, 'read_session') as read, \
                 patch.object(sandbox_supervisor.socket, 'socket') as create_socket, \
                 patch.object(sandbox_supervisor.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'pending fork recovery failed'):
                    sandbox_supervisor.run_private_pi('method.py', 'request.json', request)

            read.assert_not_called()
            create_socket.assert_not_called()
            spawn.assert_not_called()
            self.assertFalse(private.exists(), 'failed Broker startup left a retry-blocking directory')

    def test_broker_socket_uses_shared_group_without_listing_parent(self):
        with tempfile.TemporaryDirectory(prefix='broker-socket-permissions-') as temporary:
            root = Path(temporary) / 'session'
            private = Path(temporary) / 'private'
            root.mkdir()
            private.mkdir()
            share_gid = grp.getgrnam('ptobench-share').gr_gid
            with patch.object(sandbox_supervisor.session, 'recover_pending_forks', return_value=[]), \
                 patch.object(sandbox_supervisor.session, 'read_session', return_value={'workspaces': []}):
                broker = sandbox_supervisor.Broker(root, private)
            try:
                socket_mode = broker.sockpath.stat()
                directory_mode = broker.sockpath.parent.stat()
                self.assertEqual(socket_mode.st_uid, os.getuid())
                self.assertEqual(socket_mode.st_gid, share_gid)
                self.assertEqual(stat.S_IMODE(socket_mode.st_mode), 0o660)
                self.assertEqual(directory_mode.st_gid, share_gid)
                self.assertEqual(stat.S_IMODE(directory_mode.st_mode), 0o710)
            finally:
                broker.server.close()
                broker.ipc.cleanup()

    def test_initializes_without_exported_workspace_or_candidate_trees(self):
        with tempfile.TemporaryDirectory(prefix='broker-direct-workspace-') as temporary:
            root = Path(temporary) / 'session'
            private = Path(temporary) / 'private'
            root.mkdir()
            private.mkdir()
            calls = []

            def read(path):
                calls.append(('read', Path(path)))
                return {'workspaces': []}

            with patch.object(sandbox_supervisor.session, 'recover_pending_forks', return_value=[]), \
                 patch.object(sandbox_supervisor.session, 'read_session', side_effect=read):
                broker = sandbox_supervisor.Broker(root, private)
            try:
                self.assertEqual(calls, [('read', root)])
                self.assertFalse((private / 'view').exists())
                self.assertFalse((private / 'public').exists())
                self.assertFalse((private / 'candidates').exists())
                self.assertFalse(hasattr(broker, 'exports'))
                self.assertFalse(hasattr(broker, 'candidates'))
            finally:
                broker.server.close()
                broker.ipc.cleanup()

    def test_cli_help_routes_directly_without_legacy_workspace_methods(self):
        class Child:
            stdout = io.BytesIO(b'help output\n')

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory(prefix='broker-cli-routing-') as temporary:
            root = Path(temporary) / 'session'
            private = Path(temporary) / 'private'
            workspace = root / 'workspaces/worker_000'
            workspace.mkdir(parents=True)
            private.mkdir()
            with patch.object(sandbox_supervisor.session, 'recover_pending_forks', return_value=[]), \
                 patch.object(sandbox_supervisor.session, 'read_session', return_value={'workspaces': []}), \
                 patch.object(sandbox_supervisor.session, 'resolve_workspace',
                              return_value=('worker_000', workspace, {})) as resolve, \
                 patch.object(sandbox_supervisor.subprocess, 'Popen', return_value=Child()) as spawn:
                broker = sandbox_supervisor.Broker(root, private, cli_environment={'PATH': '/usr/bin'})
                for name in ('export', 'sync_source', 'feedback'):
                    self.assertFalse(hasattr(broker, name), f'legacy Broker method remains: {name}')
                thread = threading.Thread(target=broker.serve, daemon=True)
                thread.start()
                try:
                    with socket.socket(socket.AF_UNIX) as client:
                        client.connect(str(broker.sockpath))
                        client.sendall((json.dumps({'argv': ['--help'], 'cwd': str(workspace)}) + '\n').encode())
                        response = [json.loads(line) for line in client.makefile('rb')]
                    broker.stop.set()
                    thread.join(5)
                    self.assertEqual(response, [{'output': 'help output\n'}, {'exit_code': 0}])
                    resolve.assert_called_once_with(root, str(workspace))
                    spawn.assert_called_once()
                    self.assertEqual(spawn.call_args.args[0][-1], '--help')
                finally:
                    broker.stop.set()
                    thread.join(5)
                    broker.server.close()
                    broker.ipc.cleanup()


class RetiredSessionBrokerTests(SessionTestBase):
    def test_broker_rejects_retired_session_before_repair(self):
        handle = self.create()
        root = handle.session_root
        record = session.read_session(root)
        retired = dict(record)
        retired['schema_version'] = 1
        integrity.commit_records(handle.protection_root,
                                 {'session': session._bytes(retired)})
        (root / 'session.json').write_text('{"damaged": true}\n')
        before = tree_digest(root)
        private = self.repo / 'private'
        private.mkdir()

        with self.assertRaises(session.SessionError) as caught:
            sandbox_supervisor.Broker(root, private)

        self.assertEqual(caught.exception.error_code, 'invalid_request')
        self.assertEqual(tree_digest(root), before)
        self.assertFalse(list(private.iterdir()))


if __name__ == '__main__':
    unittest.main()
