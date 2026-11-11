"""Dedicated Agent launch and one-shot request contracts."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent_launch


class AgentLaunchTests(unittest.TestCase):
    def setUp(self):
        self.credentials = {
            'uid': os.getuid(), 'gid': os.getgid(), 'groups': [os.getgid()],
            'cap_eff': 0, 'no_new_privs': False,
        }

    def _prepare(self, root, *, credentials=None):
        try:
            return agent_launch.prepare(
                ['/usr/bin/bwrap', '--unshare-user', '--cap-drop', 'ALL', '--', '/bin/true'],
                Path('/workspace'), root / 'agent' / 'launcher',
            )
        except TypeError as error:
            self.fail(f'prepare must no longer require an external launch-path override: {error}')

    def test_prepare_uses_locked_python_and_exact_agent_sudo_argv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locked_python = Path('/opt/locked/python')
            dispatcher = Path('/checkout/evaluator/agent_dispatch.py')
            with patch.object(agent_launch, '_host_credentials', return_value=self.credentials), \
                 patch.object(agent_launch, '_agent_identity', return_value=(947, os.getgid()), create=True), \
                 patch.object(agent_launch.environment_lock, 'locked_python_executable', return_value=locked_python), \
                 patch.object(agent_launch, '_system_binary', return_value='/usr/bin/sudo', create=True), \
                 patch.object(agent_launch, '_prepare_dispatcher', create=True), \
                 patch.object(agent_launch, 'DISPATCHER', dispatcher):
                command, environment = self._prepare(root)

            self.assertEqual(command, [
                '/usr/bin/sudo', '-n', '-u', 'ptobench-agent', '-g', 'ptobench-share', '--',
                str(locked_python), '-I', '-S', str(dispatcher),
            ])
            self.assertEqual(set(environment), {'PATH', agent_launch.REQUEST_ENV})
            self.assertEqual(environment['PATH'], os.defpath)

    def test_request_is_broker_owned_share_readonly_random_and_minimal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(agent_launch, '_host_credentials', return_value=self.credentials), \
                 patch.object(agent_launch, '_agent_identity', return_value=(947, os.getgid()), create=True), \
                 patch.object(agent_launch.environment_lock, 'locked_python_executable', return_value=Path('/opt/python')), \
                 patch.object(agent_launch, '_system_binary', return_value='/usr/bin/sudo', create=True), \
                 patch.object(agent_launch, '_prepare_dispatcher', create=True):
                first, first_env = self._prepare(root)
                second, second_env = self._prepare(root)

            agent_directory = root / 'agent'
            request_directory = agent_directory / 'launcher'
            self.assertEqual(stat.S_IMODE(agent_directory.stat().st_mode), 0o2710)
            self.assertEqual(stat.S_IMODE(request_directory.stat().st_mode), 0o2710)
            self.assertEqual(first[:7], second[:7])
            request_paths = [Path(first_env[agent_launch.REQUEST_ENV]),
                             Path(second_env[agent_launch.REQUEST_ENV])]
            self.assertNotEqual(*request_paths)
            for request_path in request_paths:
                metadata = request_path.stat()
                self.assertEqual(metadata.st_uid, os.getuid())
                self.assertEqual(metadata.st_gid, os.getgid())
                self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o640)
                self.assertEqual(metadata.st_nlink, 1)
                request = json.loads(request_path.read_text())
                self.assertEqual(set(request), {'schema_version', 'argv', 'cwd'})
                self.assertEqual(request['argv'][0], '/usr/bin/bwrap')
                self.assertEqual(request['cwd'], '/workspace')
                self.assertNotIn('API_KEY', request_path.read_text())
                self.assertNotIn('uid', request)
                self.assertNotIn('groups', request)

    def test_invalid_broker_identity_fails_closed_before_request_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = dict(self.credentials, groups=[])
            with patch.object(agent_launch, '_host_credentials', return_value=bad), \
                 patch.object(agent_launch, '_agent_identity', return_value=(947, os.getgid()), create=True):
                with self.assertRaisesRegex(RuntimeError, 'ptobench-share'):
                    self._prepare(root)
            self.assertFalse((root / 'agent').exists())

    def test_untrusted_sudo_or_missing_locked_python_fails_closed(self):
        for python, sudo in ((None, '/usr/bin/sudo'), (Path('/opt/python'), None)):
            with self.subTest(python=python, sudo=sudo), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.object(agent_launch, '_host_credentials', return_value=self.credentials), \
                     patch.object(agent_launch, '_agent_identity', return_value=(947, os.getgid()), create=True), \
                     patch.object(agent_launch.environment_lock, 'locked_python_executable', return_value=python), \
                     patch.object(agent_launch, '_system_binary', return_value=sudo, create=True), \
                     patch.object(agent_launch, '_prepare_dispatcher', create=True):
                    with self.assertRaises(RuntimeError):
                        self._prepare(root)
                self.assertFalse((root / 'agent').exists())

    def test_trusted_binary_rejects_effective_acl_write_permission(self):
        parser = getattr(agent_launch, '_acl_output_has_write', None)
        self.assertTrue(callable(parser), 'trusted binary ACL validation is missing')
        output = ('# file: /usr/bin/example\nuser::rwx\nuser:1001:rwx\n'
                  'group::r-x\nmask::rwx\nother::r-x\n')
        self.assertTrue(parser(output))
        safe = ('user::rwx\nuser:1001:r-x\ngroup::r-x\nmask::r-x\nother::r-x\n')
        self.assertFalse(parser(safe))

    def test_agent_acl_named_user_permission_overrides_other_entry(self):
        resolver = getattr(agent_launch, '_acl_effective_permissions', None)
        self.assertTrue(callable(resolver), 'dispatcher parent ACL validation is missing')
        metadata = SimpleNamespace(st_uid=1001, st_gid=1001)
        named_agent = ('user::rwx\nuser:947:--x\ngroup::rwx\nmask::rwx\n'
                       'other::r-x\n')
        self.assertEqual(resolver(named_agent, metadata, 947, {940}), '--x')
        no_named_agent = 'user::rwx\ngroup::rwx\nother::r-x\n'
        self.assertEqual(resolver(no_named_agent, metadata, 947, {940}), 'r-x')
        shared_group = ('user::rwx\ngroup::rwx\ngroup:940:--x #effective:--x\n'
                        'mask::rwx\nother::rwx\n')
        self.assertEqual(resolver(shared_group, metadata, 947, {940}), '--x')

    def test_dispatcher_parents_allow_only_agent_traversal(self):
        validate = getattr(agent_launch, '_validate_agent_dispatcher_path', None)
        self.assertTrue(callable(validate), 'dispatcher parent ACL validation is missing')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dispatcher = root / 'private/evaluator/agent_dispatch.py'
            dispatcher.parent.mkdir(parents=True)

            def permissions(path, _uid, _gid):
                path = Path(path)
                if path == dispatcher:
                    return 'r--'
                return 'r-x' if path.stat().st_uid == 0 else '--x'

            with patch.object(agent_launch, 'DISPATCHER', dispatcher), \
                 patch.object(agent_launch, '_agent_directory_permissions', side_effect=permissions):
                validate(947, 940)

            with patch.object(agent_launch, 'DISPATCHER', dispatcher), \
                 patch.object(agent_launch, '_agent_directory_permissions',
                              side_effect=lambda path, uid, gid:
                              'r-x' if Path(path).is_relative_to(root) and Path(path) != dispatcher
                              else permissions(path, uid, gid)):
                with self.assertRaisesRegex(RuntimeError, 'dispatcher parent'):
                    validate(947, 940)

            with patch.object(agent_launch, 'DISPATCHER', dispatcher), \
                 patch.object(agent_launch, '_agent_directory_permissions',
                              side_effect=lambda _path, _uid, _gid: 'rwx'):
                with self.assertRaisesRegex(RuntimeError, 'dispatcher path|dispatcher parent'):
                    validate(947, 940)

            with patch.object(agent_launch, 'DISPATCHER', dispatcher), \
                 patch.object(agent_launch, '_agent_directory_permissions',
                              side_effect=lambda path, uid, gid:
                              'rw-' if Path(path) == dispatcher else permissions(path, uid, gid)):
                with self.assertRaisesRegex(RuntimeError, 'dispatcher path'):
                    validate(947, 940)

    def test_existing_broker_agent_directory_is_prepared_for_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            session_root = Path(temporary) / 'session'
            session_root.mkdir(mode=0o700)
            agent_directory = session_root / 'agent'
            agent_directory.mkdir(mode=0o750)
            config = agent_directory / 'config.json'
            config.write_text('{}\n')
            config.chmod(0o600)

            share_gid = agent_launch.grp.getgrnam(agent_launch.SHARE_GROUP).gr_gid
            launcher = agent_launch._request_directories(
                agent_directory / 'launcher', os.getuid(), share_gid)

            for directory in (agent_directory, launcher):
                metadata = directory.stat()
                self.assertEqual((metadata.st_uid, metadata.st_gid),
                                 (os.getuid(), share_gid))
                self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o2710)
            self.assertEqual(config.read_text(), '{}\n')
            self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o600)

    def test_broker_cleans_stale_requests_only_after_session_children_stop(self):
        cleanup = getattr(agent_launch, 'cleanup_stale_requests', None)
        self.assertTrue(callable(cleanup), 'Broker crash-recovery request cleanup is missing')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'session'
            request_dir = root / 'agent' / 'launcher'
            request_dir.mkdir(parents=True, mode=0o2710)
            request_dir.parent.chmod(0o2710)
            request_dir.chmod(0o2710)
            requests = [request_dir / (token * 32 + '.json') for token in ('a', 'b')]
            for path in requests:
                path.write_text('{}\n')
                path.chmod(0o640)
            unrelated = request_dir / 'notes.txt'
            unrelated.write_text('keep')
            with patch.object(agent_launch.grp, 'getgrnam',
                              return_value=SimpleNamespace(gr_gid=os.getgid())):
                removed = cleanup(root)
            self.assertEqual(removed, 2)
            self.assertTrue(all(not path.exists() for path in requests))
            self.assertEqual(unrelated.read_text(), 'keep')

    def test_broker_cleans_request_after_child_exit_and_launch_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'session'
            request_directory = root / 'agent' / 'launcher'
            request_directory.mkdir(parents=True, mode=0o2710)
            request_directory.parent.chmod(0o2710)
            request_directory.chmod(0o2710)
            path = request_directory / ('12' * 16 + '.json')
            path.write_text('{}\n')
            path.chmod(0o640)
            environment = {'PATH': os.defpath, agent_launch.REQUEST_ENV: str(path)}
            with patch.object(agent_launch.grp, 'getgrnam',
                              return_value=SimpleNamespace(gr_gid=os.getgid())):
                code = agent_launch.run_process(
                    [os.sys.executable, '-c', 'raise SystemExit(7)'],
                    cwd=temporary, environment=environment)
            self.assertEqual(code, 7)
            self.assertFalse(path.exists())

            path.write_text('{}\n')
            path.chmod(0o640)
            with patch.object(agent_launch.grp, 'getgrnam',
                              return_value=SimpleNamespace(gr_gid=os.getgid())), \
                 patch.object(agent_launch.subprocess, 'Popen', side_effect=OSError('launch failed')):
                with self.assertRaisesRegex(OSError, 'launch failed'):
                    agent_launch.run_process(['/usr/bin/false'], cwd=temporary,
                                             environment=environment)
            self.assertFalse(path.exists())


class DispatcherRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(agent_launch.DISPATCHER)
        spec = importlib.util.spec_from_file_location('ptobench_agent_dispatch', source)
        cls.dispatcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.dispatcher)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.agent = self.root / 'agent'
        self.launcher = self.agent / 'launcher'
        self.agent.mkdir(mode=0o2710)
        self.agent.chmod(0o2710)
        self.launcher.mkdir(mode=0o2710)
        self.launcher.chmod(0o2710)
        os.chown(self.agent, os.getuid(), os.getgid())
        os.chown(self.launcher, os.getuid(), os.getgid())

    def tearDown(self):
        self.temporary.cleanup()

    def _request(self, payload=None):
        path = self.launcher / ('ab' * 16 + '.json')
        value = {'schema_version': 1, 'argv': ['/usr/bin/bwrap', '--', '/bin/true'],
                 'cwd': '/tmp'} if payload is None else payload
        path.write_text(json.dumps(value))
        path.chmod(0o640)
        return path

    def test_reader_validates_request_and_leaves_cleanup_to_broker(self):
        path = self._request()
        with patch.dict(self.dispatcher.os.environ,
                        {'SUDO_UID': str(os.getuid()), 'SUDO_GID': str(os.getgid())}, clear=True), \
             patch.object(self.dispatcher.os, 'getuid', return_value=os.getuid() + 1), \
             patch.object(self.dispatcher, 'grp', SimpleNamespace(
                 getgrnam=lambda _name: SimpleNamespace(gr_gid=os.getgid())), create=True):
            try:
                command, cwd = self.dispatcher.read_request(str(path))
            except RuntimeError as error:
                self.fail(f'valid Broker request was rejected: {error}')
        self.assertTrue(path.exists())
        self.assertEqual(command, ['/usr/bin/bwrap', '--', '/bin/true'])
        self.assertEqual(cwd, '/tmp')

    def test_reader_rejects_symlink_hardlink_wrong_mode_and_extra_fields(self):
        real = self._request()
        link = self.launcher / ('cd' * 16 + '.json')
        link.symlink_to(real)
        with patch.dict(self.dispatcher.os.environ,
                        {'SUDO_UID': str(os.getuid()), 'SUDO_GID': str(os.getgid())}, clear=True), \
             patch.object(self.dispatcher.os, 'getuid', return_value=os.getuid() + 1), \
             patch.object(self.dispatcher, 'grp', SimpleNamespace(
                 getgrnam=lambda _name: SimpleNamespace(gr_gid=os.getgid())), create=True):
            with self.assertRaisesRegex(RuntimeError, 'regular file'):
                self.dispatcher.read_request(str(link))
            hardlink = self.launcher / ('ef' * 16 + '.json')
            os.link(real, hardlink)
            with self.assertRaisesRegex(RuntimeError, 'single link'):
                self.dispatcher.read_request(str(real))

            hardlink.unlink()
            real.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, 'mode'):
                self.dispatcher.read_request(str(real))

            real.unlink()
            extra = self._request({'schema_version': 1,
                                   'argv': ['/usr/bin/bwrap', '--', '/bin/true'],
                                   'cwd': '/tmp', 'PYPTO_PROVIDER_API_KEY': 'secret'})
            with self.assertRaisesRegex(RuntimeError, 'schema'):
                self.dispatcher.read_request(str(extra))

    def test_reader_rejects_request_owned_by_agent_or_wrong_group(self):
        path = self._request()
        with patch.dict(self.dispatcher.os.environ,
                        {'SUDO_UID': str(os.getuid()), 'SUDO_GID': str(os.getgid())}, clear=True), \
             patch.object(self.dispatcher.os, 'getuid', return_value=os.getuid() + 1), \
             patch.object(self.dispatcher, 'grp', SimpleNamespace(
                 getgrnam=lambda _name: SimpleNamespace(gr_gid=os.getgid() + 1)), create=True):
            with self.assertRaisesRegex(RuntimeError, '2710'):
                self.dispatcher.read_request(str(path))

    def test_dispatch_executes_only_with_minimal_environment_and_no_request_path(self):
        request_env = {agent_launch.REQUEST_ENV: '/tmp/request', 'SUDO_UID': str(os.getuid()),
                       'SUDO_GID': str(os.getgid())}
        with patch.dict(self.dispatcher.os.environ, request_env, clear=True), \
             patch.object(self.dispatcher, 'check_identity'), \
             patch.object(self.dispatcher, 'read_request', return_value=(['/usr/bin/bwrap', '--', '/bin/true'], '/tmp')) as reader, \
             patch.object(self.dispatcher.os, 'execve', side_effect=AssertionError('execve reached')) as execve:
            try:
                self.dispatcher.dispatch('/tmp/request')
            except AssertionError as error:
                if str(error) != 'execve reached':
                    raise
            except Exception as error:
                self.fail(f'dispatcher took a legacy launch path: {error}')
            else:
                self.fail('dispatcher did not exec Bubblewrap')
        reader.assert_called_once_with('/tmp/request')
        execve.assert_called_once_with('/usr/bin/bwrap',
                                       ['/usr/bin/bwrap', '--', '/bin/true'],
                                       {'PATH': os.defpath})


if __name__ == '__main__':
    unittest.main()
