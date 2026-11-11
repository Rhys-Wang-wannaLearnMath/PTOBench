"""Agent launcher contracts: dynamic host identities and one-shot requests."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import agent_launch


class AgentLaunchTests(unittest.TestCase):
    def test_clean_credentials_use_existing_unprivileged_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(agent_launch, '_host_credentials', return_value={
                    'uid': 4201, 'gid': 5201, 'groups': [], 'cap_eff': 0}), \
                 patch.object(agent_launch, '_setpriv_clears_groups', side_effect=AssertionError), \
                 patch.object(agent_launch, '_trusted_launcher', return_value=None):
                command, environment = agent_launch.prepare(
                    ['/path/to/bwrap', '--', '/bin/true'], Path('/workspace'),
                    Path(temporary), path='/opt/ptobench-agent-launch')

            self.assertEqual(command[0], agent_launch.sys.executable)
            self.assertEqual(Path(command[1]), agent_launch.DISPATCHER)
            self.assertEqual(set(environment), {'PATH', 'PTOBENCH_AGENT_LAUNCH_REQUEST'})

    def test_dirty_groups_use_verified_unprivileged_setpriv(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(agent_launch, '_host_credentials', return_value={
                    'uid': 4201, 'gid': 5201, 'groups': [17, 6012], 'cap_eff': 0}), \
                 patch.object(agent_launch, '_setpriv_clears_groups', return_value=True), \
                 patch.object(agent_launch.shutil, 'which', side_effect=['/usr/bin/setpriv', '/usr/bin/sudo']), \
                 patch.object(agent_launch, '_trusted_launcher', return_value=None):
                command, _ = agent_launch.prepare(
                    ['/path/to/bwrap', '--', '/bin/true'], Path('/workspace'),
                    Path(temporary), path='/opt/ptobench-agent-launch')

            self.assertEqual(command[:2], ['/usr/bin/setpriv', '--reuid=4201'])
            self.assertIn('--clear-groups', command)

    def test_dirty_groups_fail_closed_without_a_verified_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(agent_launch, '_host_credentials', return_value={
                    'uid': 4201, 'gid': 5201, 'groups': [17], 'cap_eff': 0}), \
                 patch.object(agent_launch, '_setpriv_clears_groups', return_value=False), \
                 patch.object(agent_launch, '_trusted_launcher', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'supplementary groups'):
                    agent_launch.prepare(
                        ['/path/to/bwrap', '--', '/bin/true'], Path('/workspace'),
                        Path(temporary), path='/opt/ptobench-agent-launch')

    def test_capabilities_also_require_a_verified_clean_launch_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(agent_launch, '_host_credentials', return_value={
                    'uid': 4201, 'gid': 5201, 'groups': [], 'cap_eff': 1}), \
                 patch.object(agent_launch, '_setpriv_clears_groups', return_value=True), \
                 patch.object(agent_launch, '_trusted_launcher', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'capabilit'):
                    agent_launch.prepare(
                        ['/path/to/bwrap', '--', '/bin/true'], Path('/workspace'),
                        Path(temporary), path='/opt/ptobench-agent-launch')

    def test_no_new_privs_fails_closed_before_subordinate_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(agent_launch, '_host_credentials', return_value={
                    'uid': 4201, 'gid': 5201, 'groups': [], 'cap_eff': 0, 'no_new_privs': True}):
                with self.assertRaisesRegex(RuntimeError, 'NoNewPrivs'):
                    agent_launch.prepare(
                        ['/path/to/bwrap', '--', '/bin/true'], Path('/workspace'),
                        Path(temporary), path='/opt/ptobench-agent-launch')

    def test_each_request_is_private_unique_and_contains_no_environment_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'launcher'
            credentials = {'uid': os.getuid(), 'gid': os.getgid(), 'groups': [], 'cap_eff': 0}
            with patch.object(agent_launch, '_host_credentials', return_value=credentials), \
                 patch.object(agent_launch, '_setpriv_clears_groups', return_value=False), \
                 patch.object(agent_launch, '_trusted_launcher', return_value=None):
                first, first_env = agent_launch.prepare(
                    ['/path/to/bwrap', '--', '/bin/true'], Path('/workspace'), root,
                    path='/opt/ptobench-agent-launch')
                second, second_env = agent_launch.prepare(
                    ['/path/to/bwrap', '--', '/bin/true'], Path('/workspace'), root,
                    path='/opt/ptobench-agent-launch')

            first_path = Path(first_env['PTOBENCH_AGENT_LAUNCH_REQUEST'])
            second_path = Path(second_env['PTOBENCH_AGENT_LAUNCH_REQUEST'])
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            for request_path, command in ((first_path, first), (second_path, second)):
                self.assertEqual(stat.S_IMODE(request_path.stat().st_mode), 0o600)
                self.assertEqual(request_path.stat().st_nlink, 1)
                request = json.loads(request_path.read_text())
                self.assertEqual(request['uid'], os.getuid())
                self.assertEqual(request['gid'], os.getgid())
                self.assertEqual(request['groups'], [])
                self.assertNotIn('API_KEY', request_path.read_text())
                self.assertTrue(command)

    def test_subordinate_mapping_uses_detected_ranges_and_primary_group(self):
        plan = agent_launch.mapping_plan(
            uid=41001, gid=52007,
            uid_ranges=[(230000, 4096), (400000, 1024)],
            gid_ranges=[(510000, 4096)])

        self.assertEqual(plan['agent_uid'], 230000)
        self.assertEqual(plan['agent_ns_uid'], 41002)
        self.assertEqual(plan['uid_map'], [(41001, 41001, 1), (41002, 230000, 1)])
        self.assertEqual(plan['gid_map'], [(0, 510000, 1), (52007, 52007, 1)])

    def test_subordinate_mapping_rejects_missing_dynamic_ranges(self):
        with self.assertRaisesRegex(RuntimeError, 'subordinate'):
            agent_launch.mapping_plan(uid=41001, gid=52007, uid_ranges=[], gid_ranges=[(50000, 10)])


class RequestReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(agent_launch.DISPATCHER)
        spec = importlib.util.spec_from_file_location('ptobench_agent_dispatch', source)
        cls.dispatcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.dispatcher)

    def test_request_reader_unlinks_before_returning_valid_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            request = root / 'request.json'
            request.write_text(json.dumps({'command': ['/bin/true'], 'cwd': '/tmp',
                                           'environment': {'PATH': os.defpath},
                                           'uid': os.getuid(), 'gid': os.getgid(), 'groups': [],
                                           'schema_version': 1}))
            request.chmod(0o600)

            command, environment, cwd = self.dispatcher.read_request(str(request))

            self.assertFalse(request.exists())
            self.assertEqual(command, ['/bin/true'])
            self.assertEqual(environment, {'PATH': os.defpath})
            self.assertEqual(cwd, '/tmp')

    def test_dispatcher_rejects_nonzero_active_capability_sets(self):
        fields = {'CapEff': '0', 'CapPrm': '0', 'CapInh': '0', 'CapBnd': '0',
                  'CapAmb': '0', 'NoNewPrivs': '1'}
        for key in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb'):
            with self.subTest(capability=key):
                observed = dict(fields, **{key: '1'})
                with patch.object(self.dispatcher, '_proc_credentials', return_value=observed):
                    with self.assertRaisesRegex(RuntimeError, 'active capabilities'):
                        self.dispatcher.check_identity(os.getuid(), os.getgid())

    def test_dispatcher_mapping_uses_dynamic_subordinate_ranges(self):
        with patch.object(self.dispatcher, '_subid_ranges', side_effect=[[(230000, 4096)], [(510000, 4096)]]):
            mapping = self.dispatcher._mapping_plan(41001, 52007)

        self.assertEqual(mapping['agent_uid'], 230000)
        self.assertEqual(mapping['agent_gid'], 510000)
        self.assertEqual(mapping['uid_map'], [(41001, 41001, 1), (41002, 230000, 1)])
        self.assertEqual(mapping['gid_map'], [(0, 510000, 1), (52007, 52007, 1)])

    def test_request_reader_rejects_symlink_and_hardlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            real = root / 'real.json'
            real.write_text('{}')
            real.chmod(0o600)
            link = root / 'link.json'
            link.symlink_to(real)
            with self.assertRaisesRegex(RuntimeError, 'regular file'):
                self.dispatcher.read_request(str(link))

            hardlink = root / 'hardlink.json'
            os.link(real, hardlink)
            with self.assertRaisesRegex(RuntimeError, 'single link'):
                self.dispatcher.read_request(str(real))

    def test_request_reader_rejects_wrong_owner_mode_and_oversize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            request = root / 'request.json'
            request.write_bytes(b'{}')
            request.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, 'mode'):
                self.dispatcher.read_request(str(request))

            request.chmod(0o600)
            with patch.object(self.dispatcher, 'MAX_REQUEST_BYTES', 1):
                with self.assertRaisesRegex(RuntimeError, 'size'):
                    self.dispatcher.read_request(str(request))

if __name__ == '__main__':
    unittest.main()
