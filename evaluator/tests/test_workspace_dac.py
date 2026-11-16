"""Real Broker/Agent DAC checks through the fixed dispatcher and Bubblewrap."""
import grp
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import time
from unittest import mock

from action_test_support import ActionTestBase
import agent_launch
import isolation
import session
import session_actions
import sandbox_supervisor
import snapshots
import workspace


class WorkspaceDACTest(ActionTestBase):
    def snapshot(self, result):
        event = next(item for item in self.events() if item['seq'] == result['event_seq'])
        self.assertTrue(event['snapshot_id'])
        return self.handle.snapshots_root / event['snapshot_id']

    def _run_agent(self, command, *, mount_workspace=None, guest_path='/workspace', guest_cwd=None):
        mount_workspace = Path(mount_workspace or self.handle.initial_workspace)
        view = isolation.View().system().native_agent()
        view.add(mount_workspace, guest_path, write=True)
        environment = isolation.clean_environment()
        bubblewrap = view.argv(command, cwd=guest_cwd or guest_path, environment=environment)
        launch, launch_environment = agent_launch.prepare(
            bubblewrap, mount_workspace, self.handle.session_root / 'agent/launcher')
        try:
            return subprocess.run(launch, cwd=mount_workspace,
                                  env=launch_environment, capture_output=True,
                                  text=True, timeout=30)
        finally:
            agent_launch.cleanup_request(launch_environment)

    def _python_agent(self, script, *, workspace_path=None):
        return self._run_agent([sys.executable, '-c', script], mount_workspace=workspace_path)

    def test_long_lived_shell_observes_worker_published_after_launch(self):
        workspace_path = self.handle.initial_workspace
        ready = workspace_path / 'candidate/live_shell_ready.py'
        release = workspace_path / 'candidate/live_shell_release.py'
        complete = workspace_path / 'candidate/live_shell_complete.py'
        probe = workspace_path / 'candidate/live_shell_probe.py'
        probe.write_text(r'''import json, os, pathlib, time
workspace = pathlib.Path.cwd()
workspaces = workspace.parent
ready = workspace / 'candidate/live_shell_ready.py'
release = workspace / 'candidate/live_shell_release.py'
complete = workspace / 'candidate/live_shell_complete.py'
ready.write_text(str(os.getpid()))
deadline = time.monotonic() + 20
while not release.exists() and time.monotonic() < deadline:
    time.sleep(.02)
assert release.exists(), 'Broker did not publish a Worker while Shell was alive'
workers = [path for path in workspaces.iterdir() if path != workspace]
assert len(workers) == 1, workers
worker = workers[0]
candidate = worker / 'candidate/candidate.py'
assert candidate.read_text() == 'from helper import VALUE\n'
assert (worker / '.benchmark_context/config.py').read_text() == 'SIZE = 2\n'
output = worker / '.benchmark_outputs/clean/latest_session_result.json'
assert json.loads(output.read_text()) == {'event_seq': 23}
candidate.write_text('VALUE = 77\n')
def denied(operation):
    try:
        operation()
    except OSError:
        return
    raise AssertionError('published Workspace contract was writable')
denied(lambda: (worker / '.benchmark_context/config.py').write_text('changed'))
denied(lambda: output.write_text('{}'))
denied(lambda: worker.rename(worker.with_name('worker-replaced')))
assert (workspace / 'candidate/helper.py').read_text() == 'VALUE = 1\n'
complete.write_text(json.dumps({'pid': os.getpid(), 'worker': str(worker)}))
''')
        shell = 'python3 candidate/live_shell_probe.py'
        outer = (
            'import subprocess\n'
            f"result = subprocess.run(['/ptobench/pi-tools/ptobench-shell', '-c', {shell!r}], "
            'capture_output=True, text=True)\n'
            'assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)\n'
        )

        endpoint_root = self.handle.session_root / 'broker-endpoints'
        endpoint_root.mkdir(parents=True, exist_ok=True)
        rpc, api = endpoint_root / 'rpc.sock', endpoint_root / 'api.sock'
        endpoints = []
        for path in (rpc, api):
            endpoint = socket.socket(socket.AF_UNIX)
            endpoint.bind(str(path))
            endpoint.listen(1)
            endpoints.append(endpoint)
        transport_directory, transport = sandbox_supervisor._copy_cli_transport(
            self.handle.session_root)
        process = None
        launch_environment = None
        try:
            argv, launch_environment = isolation.agent_command(
                [str(Path(sys.prefix) / 'bin/python'), '-c', outer], cwd=workspace_path,
                environment={'PATH': os.environ['PATH'], 'PTOBENCH_RPC_SOCKET': str(rpc),
                             'PTOBENCH_API_PROXY_SOCKET': str(api),
                             'PTOBENCH_CLIENT_PATH': str(transport)},
                workspace=workspace_path, session_root=self.handle.session_root)
            process = subprocess.Popen(argv, cwd=workspace_path, env=launch_environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 20
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(ready.exists(), 'nested Shell did not reach its publication barrier')
            shell_pid = int(ready.read_text())
            worker_id = session.fork_workspace(self.handle.session_root, workspace_path)
            worker = self.handle.session_root / 'workspaces' / worker_id
            output_directory = worker / '.benchmark_outputs/clean'
            output_directory.mkdir()
            share_gid = grp.getgrnam('ptobench-share').gr_gid
            os.chown(output_directory, -1, share_gid)
            output_directory.chmod(0o750)
            output = output_directory / 'latest_session_result.json'
            output.write_text('{"event_seq":23}\n')
            os.chown(output, -1, share_gid)
            output.chmod(0o440)
            release.write_text('published')
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stdout + stderr)
            observed = json.loads(complete.read_text())
            self.assertEqual(observed, {'pid': shell_pid, 'worker': str(worker)})
            self.assertEqual((worker / 'candidate/candidate.py').read_text(), 'VALUE = 77\n')
            self.assertEqual((workspace_path / 'candidate/helper.py').read_text(), 'VALUE = 1\n')
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if launch_environment is not None:
                    agent_launch.cleanup_request_after_process(launch_environment, process)
            for endpoint in endpoints:
                endpoint.close()
            shutil.rmtree(transport_directory, ignore_errors=True)

    def test_agent_candidate_writes_are_broker_readable_and_snapshot_ready(self):
        node = shutil.which('node')
        self.assertIsNotNone(node)
        node_path = str(Path(node).resolve(strict=True))
        node_script = '''import fs from 'node:fs';
import path from 'node:path';
const file = path.join(process.cwd(), 'candidate/new_helper.py');
fs.writeFileSync(file, 'VALUE = 1\\n');
fs.writeFileSync(file, fs.readFileSync(file, 'utf8').replace('1', '2'));
'''
        node_environment = isolation.clean_environment()
        node_view = (isolation.View().system()
                     .add(node_path, '/ptobench/runtime/node')
                     .add(self.handle.initial_workspace, '/workspace', write=True))
        bubblewrap = node_view.argv(
            ['/ptobench/runtime/node', '--input-type=module', '-e', node_script],
            cwd='/workspace', environment=node_environment)
        launch, launch_environment = agent_launch.prepare(
            bubblewrap, self.handle.initial_workspace, self.handle.session_root / 'agent/launcher')
        try:
            result = subprocess.run(
                launch, cwd=self.handle.initial_workspace, env=launch_environment,
                capture_output=True, text=True, timeout=30)
        finally:
            agent_launch.cleanup_request(launch_environment)
        self.assertEqual(result.returncode, 0, result.stderr)

        atomic_script = r'''import os
from pathlib import Path
root = Path.cwd()
(root / 'candidate/nested').mkdir()
temporary = root / 'candidate/.tuning.json.pending'
temporary.write_text('{"edited": true}\n')
os.replace(temporary, root / 'candidate/tuning.json')
'''
        result = self._python_agent(atomic_script)
        self.assertEqual(result.returncode, 0, result.stderr)
        shell_script = "printf 'SHELL_VALUE = 3\\n' > candidate/shell_helper.py; " \
                       "printf 'temporary\\n' > candidate/.temporary; " \
                       "mv candidate/.temporary candidate/atomic_shell.txt; " \
                       "rm candidate/atomic_shell.txt"
        result = self._run_agent(['/bin/sh', '-c', shell_script])
        self.assertEqual(result.returncode, 0, result.stderr)

        candidate = self.handle.initial_workspace / 'candidate'
        share_gid = grp.getgrnam('ptobench-share').gr_gid
        self.assertEqual((candidate / 'new_helper.py').read_text(), 'VALUE = 2\n')
        self.assertEqual((candidate / 'tuning.json').read_text(), '{"edited": true}\n')
        for path in (candidate / 'new_helper.py', candidate / 'tuning.json'):
            metadata = path.stat()
            self.assertEqual(metadata.st_gid, share_gid)
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o660)
        self.assertTrue((candidate / 'shell_helper.py').is_file())
        self.assertFalse((candidate / 'atomic_shell.txt').exists())
        nested = candidate / 'nested'
        self.assertEqual(nested.stat().st_gid, share_gid)
        self.assertEqual(stat.S_IMODE(nested.stat().st_mode), 0o2770)

        worker_id, _, _ = session.resolve_workspace(self.handle.session_root, self.handle.initial_workspace)
        self.assertEqual(worker_id, 'worker_000')
        frozen = self.handle.session_root / 'staging/dac_snapshot'
        manifest = workspace.copy_candidate_tree(self.handle.initial_workspace, frozen / 'candidate')
        with session.execution_lock():
            snapshot_id = snapshots.create_snapshot(self.handle.session_root, frozen, {
                'seq': 1, 'workspace': str(self.handle.initial_workspace),
                'content_sha256': manifest['content_sha256'], 'exit_code': 0,
                'correctness': 'PASS', 'integrity_passed': True,
                'observation': {'correctness': 'PASS'},
            })
        snapshot_source = self.handle.snapshots_root / snapshot_id / 'source/candidate'
        self.assertEqual((snapshot_source / 'new_helper.py').read_text(), 'VALUE = 2\n')
        self.assertEqual((snapshot_source / 'tuning.json').read_text(), '{"edited": true}\n')

    def test_agent_cannot_write_context_outputs_or_replace_workspace_entries(self):
        outputs = self.handle.initial_workspace / '.benchmark_outputs/clean'
        outputs.mkdir()
        (outputs / 'latest_result.json').write_text('{"event_seq": 1}\n')
        workspace._readonly(self.handle.initial_workspace / '.benchmark_outputs')
        script = r'''import errno, os
from pathlib import Path
root = Path.cwd()
def denied(operation):
    try:
        operation()
    except OSError as error:
        assert error.errno in (errno.EACCES, errno.EPERM, errno.EROFS), error
    else:
        raise AssertionError('protected operation succeeded')
context = root / '.benchmark_context'
outputs = root / '.benchmark_outputs'
context_file = context / 'config.py'
output_file = outputs / 'clean/latest_result.json'
denied(lambda: context_file.write_text('changed'))
denied(lambda: (context / 'new.txt').write_text('changed'))
denied(lambda: context_file.chmod(0o600))
denied(lambda: context_file.unlink())
denied(lambda: output_file.write_text('changed'))
denied(lambda: (outputs / 'clean/new.txt').write_text('changed'))
denied(lambda: output_file.chmod(0o600))
denied(lambda: output_file.unlink())
denied(lambda: root.chmod(0o700))
denied(lambda: (root / 'candidate').chmod(0o700))
denied(lambda: context.chmod(0o700))
denied(lambda: outputs.chmod(0o700))
denied(lambda: (root / 'candidate').rename(root / 'candidate-old'))
denied(lambda: context.rename(root / 'context-old'))
denied(lambda: outputs.rename(root / 'outputs-old'))
context_link = root / 'candidate/.context-link'
outputs_link = root / 'candidate/.outputs-link'
os.symlink('candidate', context_link)
os.symlink('candidate', outputs_link)
denied(lambda: os.replace(context_link, context))
denied(lambda: os.replace(outputs_link, outputs))
context_link.unlink()
outputs_link.unlink()
denied(lambda: os.symlink('candidate', root / '.candidate-link'))
denied(lambda: context.rmdir())
denied(lambda: outputs.rmdir())
assert context_file.read_text() == 'SIZE = 2\n'
assert output_file.read_text() == '{"event_seq": 1}\n'
'''
        result = self._python_agent(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.handle.initial_workspace / '.benchmark_context/config.py').read_text(), 'SIZE = 2\n')
        self.assertEqual((outputs / 'latest_result.json').read_text(), '{"event_seq": 1}\n')

    def test_agent_root_temporaries_respect_sticky_boundary(self):
        root = self.handle.initial_workspace
        share_gid = grp.getgrnam('ptobench-share').gr_gid
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o3770)
        self.assertEqual(root.stat().st_gid, share_gid)
        self.assertEqual(stat.S_IMODE(root.parent.stat().st_mode), 0o2750)
        outputs = root / '.benchmark_outputs/clean'
        outputs.mkdir()
        output = outputs / 'latest_result.json'
        output.write_text('{"event_seq": 1}\n')
        workspace._readonly(root / '.benchmark_outputs')

        script = r'''import errno, os
from pathlib import Path
root = Path.cwd()
def denied(operation):
    try:
        operation()
    except OSError as error:
        assert error.errno in (errno.EACCES, errno.EPERM, errno.EROFS), error
    else:
        raise AssertionError('protected operation succeeded')
note = root / 'notes.md'
scratch = root / 'scratch'
note.write_text('temporary')
scratch.mkdir()
(scratch / 'entry').write_text('temporary')
note.rename(root / 'renamed.md')
(root / 'renamed.md').unlink()
(scratch / 'entry').unlink()
scratch.rmdir()
note.write_text('persistent')
scratch.mkdir()
(scratch / 'entry').write_text('not a Candidate input')
for name in ('candidate', '.benchmark_context', '.benchmark_outputs'):
    protected = root / name
    denied(lambda protected=protected: protected.rename(root / (name + '-renamed')))
    link = root / (name + '-replacement')
    os.symlink('candidate', link)
    denied(lambda link=link, protected=protected: os.replace(link, protected))
    link.unlink()
denied(lambda: root.chmod(0o700))
denied(lambda: (root / '.benchmark_context/config.py').write_text('changed'))
denied(lambda: (root / '.benchmark_outputs/clean/latest_result.json').write_text('{}'))
assert note.read_text() == 'persistent'
assert (scratch / 'entry').read_text() == 'not a Candidate input'
'''
        result = self._python_agent(script)
        self.assertEqual(result.returncode, 0, result.stderr)

        parent_probe = r'''import errno
from pathlib import Path
parent = Path('/workspaces')
try:
    (parent / 'worker_agent_created').mkdir()
except OSError as error:
    assert error.errno in (errno.EACCES, errno.EPERM, errno.EROFS), error
else:
    raise AssertionError('Agent wrote under workspaces/')
'''
        result = self._run_agent(
            [sys.executable, '-c', parent_probe],
            mount_workspace=root.parent, guest_path='/workspaces',
            guest_cwd='/workspaces/worker_000')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((root.parent / 'worker_agent_created').exists())

        process, result = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 0, result)
        snapshot_id = self.events()[-1]['snapshot_id']
        snapshot_source = self.handle.snapshots_root / snapshot_id / 'source'
        self.assertFalse((snapshot_source / 'notes.md').exists())
        self.assertFalse((snapshot_source / 'scratch').exists())

        worker_id = session.fork_workspace(self.handle.session_root, root)
        child = root.parent / worker_id
        self.assertEqual(stat.S_IMODE(child.stat().st_mode), 0o3770)
        self.assertEqual(child.stat().st_gid, share_gid)
        self.assertFalse((child / 'notes.md').exists())
        self.assertFalse((child / 'scratch').exists())

    def test_agent_reads_isolated_outputs_for_each_worker(self):
        process, first = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 0, first)
        process, fork = self.cli(['fork'])
        self.assertEqual(process.returncode, 0, fork)
        child = Path(fork['workspace'])
        process, second = self.cli(['evaluate', str(child)])
        self.assertEqual(process.returncode, 0, second)

        script = r'''import errno, json
from pathlib import Path
root = Path('/workspaces')
first = root / 'worker_000/.benchmark_outputs/clean/latest_session_result.json'
second = root / 'worker_001/.benchmark_outputs/clean/latest_session_result.json'
assert json.loads(first.read_text())['event_seq'] == FIRST_EVENT
assert json.loads(second.read_text())['event_seq'] == SECOND_EVENT
assert (root / 'worker_000/.benchmark_context/config.py').read_text() == 'SIZE = 2\n'
assert (root / 'worker_001/.benchmark_context/config.py').read_text() == 'SIZE = 2\n'
try:
    first.write_text('{"value":"changed"}')
except OSError as error:
    assert error.errno in (errno.EACCES, errno.EPERM, errno.EROFS), error
else:
    raise AssertionError('Agent changed published Output')
workspace = root / 'worker_000'
(workspace / 'candidate/helper.py').write_text('VALUE = 77\n')
assert (root / 'worker_001/candidate/helper.py').read_text() == 'VALUE = 1\n'
(root / 'worker_001/candidate/helper.py').write_text('VALUE = 88\n')
assert (root / 'worker_000/candidate/helper.py').read_text() == 'VALUE = 77\n'
for operation in (lambda: root.chmod(0o700),
                  lambda: (root / 'worker_000').rename(root / 'worker-copy'),
                  lambda: (root / 'worker-new').mkdir()):
    try:
        operation()
    except OSError as error:
        assert error.errno in (errno.EACCES, errno.EPERM, errno.EROFS), error
    else:
        raise AssertionError('Agent changed the Worker registry entry')
'''
        script = script.replace('FIRST_EVENT', str(first['event_seq'])).replace(
            'SECOND_EVENT', str(second['event_seq']))
        result = self._run_agent([sys.executable, '-c', script],
                                 mount_workspace=self.handle.session_root / 'workspaces',
                                 guest_path='/workspaces', guest_cwd='/workspaces/worker_000')
        self.assertEqual(result.returncode, 0, result.stderr)

        first_output = self.handle.initial_workspace / '.benchmark_outputs/clean/latest_session_result.json'
        second_output = child / '.benchmark_outputs/clean/latest_session_result.json'
        process, updated = self.cli(['evaluate'])
        self.assertEqual(process.returncode, 0, updated)
        self.assertNotEqual(updated['event_seq'], first['event_seq'])
        self.assertEqual(json.loads(first_output.read_text())['event_seq'], updated['event_seq'])
        self.assertEqual(json.loads(second_output.read_text())['event_seq'], second['event_seq'])
        self.assertEqual((self.handle.initial_workspace / 'candidate/helper.py').read_text(), 'VALUE = 77\n')
        self.assertEqual((child / 'candidate/helper.py').read_text(), 'VALUE = 88\n')

    def test_agent_private_candidate_file_blocks_snapshot_without_partial_publication(self):
        content_hash = workspace.source_manifest(self.handle.initial_workspace)['content_sha256']
        event = {'seq': 1, 'workspace': str(self.handle.initial_workspace),
                 'content_sha256': content_hash, 'exit_code': 0,
                 'correctness': 'PASS', 'integrity_passed': True,
                 'observation': {'correctness': 'PASS'}}

        def assert_snapshot_rejected(relative):
            before = sorted(path.name for path in self.handle.snapshots_root.iterdir())
            with session.execution_lock(), self.assertRaisesRegex(workspace.InvalidWorkspaceError, relative):
                snapshots.create_snapshot(self.handle.session_root, self.handle.initial_workspace, event)
            self.assertEqual(sorted(path.name for path in self.handle.snapshots_root.iterdir()), before)
            self.assertEqual(list(self.handle.snapshots_root.glob('.snapshot_*')), [])

        script = r'''import os
from pathlib import Path
path = Path('candidate/private.py')
path.write_text('PRIVATE = True\n')
path.chmod(0o600)
'''
        result = self._python_agent(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        assert_snapshot_rejected('private.py')
        (self.handle.initial_workspace / 'candidate/private.py').unlink()

        result = self._python_agent("from pathlib import Path; p=Path('candidate/private_dir'); p.mkdir(); p.chmod(0o700)")
        self.assertEqual(result.returncode, 0, result.stderr)
        assert_snapshot_rejected('private_dir')
        result = self._python_agent("from pathlib import Path; p=Path('candidate/private_dir'); p.chmod(0o2770); p.rmdir()")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_action_freeze_uses_submitted_candidate_while_agent_edits_live_tree(self):
        ready, release, inspected = (self.repo / name for name in ('dac_ready', 'dac_release', 'dac_inspected.json'))
        self.configure(pause=True, ready=str(ready), release=str(release), inspect_before=str(inspected))
        process = self.start_cli(['evaluate'])
        self.wait_file(ready, process)

        result = self._python_agent("from pathlib import Path; Path('candidate/helper.py').write_text('VALUE = 17\\n')")
        self.assertEqual(result.returncode, 0, result.stderr)
        release.write_text('continue')
        stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, stderr.decode())
        action_result = json.loads(stdout.decode().splitlines()[-1])
        snapshot = self.snapshot(action_result)
        self.assertEqual(json.loads(inspected.read_text())['helper'], 'VALUE = 1\n')
        self.assertEqual((snapshot / 'source/candidate/helper.py').read_text(), 'VALUE = 1\n')
        self.assertEqual((self.handle.initial_workspace / 'candidate/helper.py').read_text(), 'VALUE = 17\n')

    def test_agent_edit_during_candidate_scan_retries_to_a_stable_snapshot(self):
        original = workspace._copy_source_file
        source_copies = []

        def edit_during_copy(source, target, entry):
            if Path(source).name == 'candidate.py':
                source_copies.append(source)
                if len(source_copies) == 1:
                    result = self._python_agent(
                        "from pathlib import Path; Path('candidate/helper.py').write_text('VALUE = 23\\n')")
                    self.assertEqual(result.returncode, 0, result.stderr)
            return original(source, target, entry)

        with mock.patch.dict(os.environ, self.env), \
             mock.patch.object(workspace, '_copy_source_file', side_effect=edit_during_copy):
            result = session_actions.run_evaluate(self.handle.session_root, self.handle.initial_workspace)

        self.assertEqual((result.exit_code, result.error_code), (0, None))
        self.assertGreaterEqual(len(source_copies), 2)
        snapshot = self.snapshot(result.to_dict())
        self.assertEqual((snapshot / 'source/candidate/helper.py').read_text(), 'VALUE = 23\n')
        self.assertEqual(workspace.source_manifest(snapshot / 'source')['content_sha256'],
                         self.events()[-1]['content_sha256'])

    def test_agent_keeps_changing_candidate_until_freeze_fails_without_snapshot(self):
        original = workspace._copy_source_tree
        rewrites = []

        def change_after_copy(source, staging, entries):
            original(source, staging, entries)
            value = len(rewrites) + 30
            result = self._python_agent(
                f"from pathlib import Path; Path('candidate/helper.py').write_text('VALUE = {value}\\\\n')")
            self.assertEqual(result.returncode, 0, result.stderr)
            rewrites.append(value)

        with mock.patch.dict(os.environ, self.env), \
             mock.patch.object(workspace, '_copy_source_tree', side_effect=change_after_copy):
            result = session_actions.run_evaluate(self.handle.session_root, self.handle.initial_workspace)

        self.assertEqual((result.exit_code, result.error_code), (2, 'invalid_workspace'))
        self.assertEqual(len(rewrites), 4)
        self.assertEqual(self.calls(), [])
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'], 0)
        self.assertEqual(self.events()[-1]['state'], 'rejected')
        self.assertEqual(list(self.handle.snapshots_root.iterdir()), [])
