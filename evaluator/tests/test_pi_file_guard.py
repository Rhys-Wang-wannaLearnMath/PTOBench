"""Pi file-tool, frozen Skill and private-resource boundary tests."""
import grp
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import isolation
from modes.cannbot import resources as cann_bot_resources
from modes.cannbot import tools as cann_bot_tools


class PiFileGuardTests(unittest.TestCase):
    def test_skill_tool_reads_only_indexed_frozen_resources(self):
        with tempfile.TemporaryDirectory(prefix='cannbot-frozen-skills-') as temporary:
            root = Path(temporary)
            upstream = root / 'method/resources/upstream'
            upstream.mkdir(parents=True)
            cann_bot_resources.freeze(upstream)
            skills = cann_bot_resources.skills(upstream)
            name = next(key for key in skills if '/' not in key)
            source = Path(skills[name])
            original = source.read_text()
            config = {'run_root': str(root), 'profile': {'backend': 'pi'}, 'skills': skills}

            response = cann_bot_tools.call(
                config, 'dispatch', 'skill', {'name': name}, 'call-1', False)
            self.assertFalse(response['isError'])
            self.assertEqual(response['details']['source'], str(source))
            self.assertIn(original, response['content'][0]['text'])
            self.assertTrue(source.is_relative_to(upstream))

            for requested in ('/etc/passwd', 'https://example.invalid/skill.md'):
                with self.subTest(requested=requested):
                    response = cann_bot_tools.call(
                        config, 'dispatch', 'skill', {'name': requested}, 'call-2', False)
                    self.assertTrue(response['isError'])
                    self.assertNotIn('root:', response['content'][0]['text'])

    def test_basic_and_cannbot_preparation_do_not_copy_raw_wiki(self):
        from modes.basic import runner as basic_runner
        from modes.cannbot import runner as cann_bot_runner

        with tempfile.TemporaryDirectory(prefix='agent-no-raw-wiki-') as temporary:
            root = Path(temporary)
            run_root = root / 'session'
            workspace = run_root / 'workspaces/worker_000'
            workspace.mkdir(parents=True)
            profile = {'backend': 'codex', 'argv': ['codex', '{prompt}'], 'env': {},
                       'sha256': 'fixture-profile'}
            context = {'session_root': str(run_root), 'workspace': str(workspace),
                       'case': {'case_id': 'fixture.case', 'module': 'fixture', 'entry': 'run'},
                       'budget': {'max_hardware_actions': 1}}
            backend = {'backend': 'codex', 'version': 'fixture'}

            basic_request = {'context': context, 'preflight': {'profile': profile, 'backend': backend},
                             'experiment': {'mode': 'basic', 'case': 'fixture.case'}}
            with patch.object(basic_runner, 'render', return_value=(['codex', 'prompt'], {})), \
                 patch.object(basic_runner.provenance, 'freeze_agent', return_value={}):
                basic = basic_runner.prepare(basic_request)

            cann_bot_request = {
                'context': context, 'preflight': {'profile': profile, 'backend': backend},
                'experiment': {'mode': 'cannbot', 'case': 'fixture.case',
                               'optimizer': {'id': 'fixture'}}}
            with patch.object(cann_bot_runner.provenance, 'freeze_agent', return_value={}):
                cann_bot = cann_bot_runner.prepare(cann_bot_request)

            for result in (basic, cann_bot):
                self.assertNotIn('raw_wiki', json.dumps(result))
            self.assertFalse(any('raw_wiki' in path.relative_to(run_root).as_posix()
                                 for path in run_root.rglob('*')))

    def test_native_file_tools_cannot_access_private_roots_or_aliases(self):
        with tempfile.TemporaryDirectory(prefix='pi-file-guard-') as temporary:
            root = Path(temporary)
            private = root / 'private'
            tools = root / 'tools'
            sessions = root / 'sessions'
            workspace = root / 'workspace'
            (private / 'extensions').mkdir(parents=True)
            tools.mkdir()
            sessions.mkdir()
            (workspace / 'candidate').mkdir(parents=True)
            (private / 'settings.json').write_text('private config')
            (private / 'extensions/security.mjs').write_text('private extension')
            (tools / 'cannbot-extension.ts').write_text('private CANNBot extension')
            (sessions / 'session.jsonl').write_text('private session')
            (workspace / 'candidate/candidate.py').write_text('candidate')
            (workspace / 'private-link').symlink_to(private, target_is_directory=True)
            os.link(private / 'settings.json', workspace / 'candidate/private-hardlink.json')

            script = r'''import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import pathModule from 'node:path';
const extension = await import(pathToFileURL(process.env.PI_FILE_GUARD));
let guard;
extension.default({ on: (name, handler) => {
    assert.equal(name, 'tool_call');
    guard = handler;
} });
const context = { cwd: process.env.PI_GUARD_CWD };
const privateFile = pathModule.join(process.env.PI_CODING_AGENT_DIR, 'settings.json');
const toolsFile = pathModule.join(process.env.PTOBENCH_PI_TOOLS_DIR, 'cannbot-extension.ts');
const sessionFile = pathModule.join(process.env.PTOBENCH_AGENT_SESSION_DIR, 'session.jsonl');
for (const toolName of ['read', 'write', 'edit', 'grep', 'find', 'ls']) {
    for (const path of [privateFile, toolsFile, sessionFile]) {
        const blocked = await guard({ toolName, input: { path } }, context);
        assert.equal(blocked?.block, true, `${toolName}: ${path}`);
        assert.match(blocked.reason, /private Pi runtime/);
    }
}
for (const path of [
    process.env.PI_CODING_AGENT_DIR,
    process.env.PTOBENCH_PI_TOOLS_DIR,
    sessionFile,
    pathModule.join(context.cwd, 'private-link/extensions/security.mjs'),
    pathModule.relative(context.cwd, privateFile),
    'candidate/private-hardlink.json',
]) {
    const blocked = await guard({ toolName: 'read', input: { path } }, context);
    assert.equal(blocked?.block, true, path);
}
assert.equal(await guard({ toolName: 'read', input: { path: 'candidate/candidate.py' } }, context), undefined);
assert.equal((await guard({ toolName: 'grep', input: { path: '.' } }, context))?.block, true);
'''
            result = subprocess.run(
                [shutil.which('node'), '--input-type=module', '-e', script],
                cwd=workspace,
                env={**os.environ, 'PI_FILE_GUARD': str(Path(isolation.__file__).with_name('pi_file_guard.mjs')),
                     'PI_CODING_AGENT_DIR': str(private), 'PTOBENCH_AGENT_SESSION_DIR': str(sessions),
                     'PTOBENCH_PI_TOOLS_DIR': str(tools),
                     'PI_GUARD_CWD': str(workspace)},
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
