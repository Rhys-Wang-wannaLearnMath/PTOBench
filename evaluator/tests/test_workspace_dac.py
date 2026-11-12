"""Real Broker/Agent DAC checks through the fixed dispatcher and Bubblewrap."""
import grp
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from unittest import mock

from action_test_support import ActionTestBase
import agent_launch
import isolation
import session
import session_actions
import snapshots
import workspace


class WorkspaceDACTest(ActionTestBase):
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

    def test_agent_candidate_writes_are_broker_readable_and_snapshot_ready(self):
        node = shutil.which('node')
        package = Path.home() / '.local/npm/lib/node_modules/@earendil-works/pi-coding-agent'
        self.assertIsNotNone(node)
        self.assertTrue((package / 'dist/core/tools/write.js').is_file())
        node_script = f'''import {{ createWriteToolDefinition }} from {json.dumps((package / 'dist/core/tools/write.js').as_uri())};
import {{ createEditToolDefinition }} from {json.dumps((package / 'dist/core/tools/edit.js').as_uri())};
const context = {{ cwd: process.cwd() }};
const write = createWriteToolDefinition(process.cwd());
const edit = createEditToolDefinition(process.cwd());
await write.execute('write', {{ path: 'candidate/new_helper.py', content: 'VALUE = 1\\n' }}, undefined, undefined, context);
await edit.execute('edit', {{ path: 'candidate/new_helper.py', edits: [{{ oldText: '1', newText: '2' }}] }}, undefined, undefined, context);
'''
        result = self._run_agent([node, '--input-type=module', '-e', node_script])
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
denied(lambda: (root / 'new-entry').mkdir())
assert context_file.read_text() == 'SIZE = 2\n'
assert output_file.read_text() == '{"event_seq": 1}\n'
'''
        result = self._python_agent(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.handle.initial_workspace / '.benchmark_context/config.py').read_text(), 'SIZE = 2\n')
        self.assertEqual((outputs / 'latest_result.json').read_text(), '{"event_seq": 1}\n')

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
