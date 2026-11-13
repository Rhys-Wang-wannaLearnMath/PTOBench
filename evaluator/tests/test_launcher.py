"""S5: real launcher/Optimizer/ptobench subprocesses with a CPU Evaluator."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from action_test_support import ActionTestBase, SOURCE_BENCHMARK, write_json
from test_session import tree_digest
import integrity
import session
sys.path.insert(0, str(SOURCE_BENCHMARK))
from launcher import runner as launcher_runner


def prepare_fake_pi_bundle(test_case):
    """Ensure CANNBot's prepared CPU fake Pi bundle exists before launch."""
    builder = getattr(test_case, 'pi_bundle', None)
    if callable(builder):
        bundle = test_case.handle.session_root / 'agent/pi-tools/fake-pi/dist/bundle/cli.js'
        if not bundle.is_file():
            builder()
        return
    import yaml
    profile_path = test_case.bench / 'modes/_shared/profiles/pi/default.yaml'
    profile = yaml.safe_load(profile_path.read_text())
    if profile['argv'][0] != str(test_case.tool):
        return
    package = test_case.handle.session_root / 'agent/pi-tools/pi-coding-agent'
    bundle = package / 'dist/bundle/cli.js'
    bundle.parent.mkdir(parents=True)
    (package / 'node_modules/jiti').mkdir(parents=True)
    shutil.copyfile(test_case.tool, bundle.parent / 'fake-native.py')
    bundle.write_text(
        '#!/usr/bin/env node\n'
        "const {spawnSync} = require('node:child_process');\n"
        f"const result = spawnSync({test_case.python!r}, "
        "[__dirname + '/fake-native.py', ...process.argv.slice(2)], {stdio:'inherit'});\n"
        "if (result.error) throw result.error;\n"
        "process.exit(result.status === null ? 1 : result.status);\n"
    )
    bundle.chmod(0o550)
    profile['argv'][0] = str(bundle)
    profile_path.write_text(yaml.safe_dump(profile))


FAKE_TOOL = r'''
import json, os, signal, subprocess, sys, time, uuid
from pathlib import Path
if '--version' in sys.argv:
    print('fake-tool 1.0'); sys.exit()
if '--help' in sys.argv:
    print('--skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --dangerously-skip-permissions --print --json --model --provider --thinking --mode --session-dir --test-control-json'); sys.exit()
c=(json.loads(sys.argv[sys.argv.index('--test-control-json')+1])
   if '--test-control-json' in sys.argv else json.loads(Path(os.environ['S5_CONTROL']).read_text()))
root=Path(os.environ['PTOBENCH_SESSION_ROOT']) if 'PTOBENCH_SESSION_ROOT' in os.environ else None
if root is not None and (root/'method/README.md').is_file():
    assert str(root/'method/README.md') in sys.argv[-1]
    references=json.loads((root/'method/knowledge.json').read_text())
    assert (root/references['snapshot']/'README.md').read_text()
backend=os.environ.get('PTOBENCH_AGENT_BACKEND', 'pi' if '--session-dir' in sys.argv else 'codex')
identity=str(uuid.uuid4())
if backend in ('codex','pi') and not c.get('missing_session'):
    if backend=='codex':
        native=Path(os.environ['CODEX_HOME'])/'sessions'/('rollout-'+identity+'.jsonl')
        header={'type':'session_meta','payload':{'id':identity,'cwd':os.getcwd()}}
        print(json.dumps({'type':'thread.started','thread_id':identity}),flush=True)
    else:
        native=Path(sys.argv[sys.argv.index('--session-dir')+1])/(identity+'.jsonl')
        header={'type':'session','id':identity,'cwd':os.getcwd()}
        print(json.dumps(header),flush=True)
    native.parent.mkdir(parents=True,exist_ok=True)
    native.write_text(json.dumps(header)+'\n'+json.dumps({'type':'message','message':{'role':'assistant','content':'native history','usage':{'input':12,'output':3}}})+'\n')
if c.get('seen'):
    Path(c['seen']).write_text(json.dumps({'argv':sys.argv,'cwd':os.getcwd(),
        'environment':{k:v for k,v in os.environ.items() if k.startswith('PTOBENCH_')},
        'cli':shutil.which('ptobench') if False else str(Path(os.environ['PATH'].split(os.pathsep)[0])/'ptobench')}))
print('optimizer stdout',flush=True); print('optimizer stderr',file=sys.stderr,flush=True)
if c.get('huge'):
    os.write(1,b'O'*262144+b'\n'); os.write(2,b'E'*262144+b'\n')
def call(args):
    p=subprocess.run(['ptobench',*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    os.write(1,p.stdout); os.write(2,p.stderr)
    return p
def profile(target='.'):
    if os.environ.get('S5_PROFILE_STRATEGY')=='combined':
        return call(['profile',target])
    result=None
    for mode in ('runtime','pmu','compiler'): result=call(['profile',mode,target])
    return result
if c.get('child'):
    child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
    Path(c['child']).write_text(str(child.pid))
if c.get('mode')=='empty': sys.exit(c.get('exit',0))
if c.get('mode') != 'profiles_only':
    if c.get('pause'):
        Path('candidate/helper.py').write_text('VALUE = 1  # PTOBENCH_TEST_PAUSE\n')
    call(['evaluate'])
if c.get('mode')=='s6':
    fork=call(['fork'])
    target=Path(json.loads(fork.stdout.decode().splitlines()[-1])['workspace'])
    (target/'candidate/helper.py').write_text('VALUE = 2\n')
    call(['profile','runtime',str(target)])
    call(['evaluate',str(target)])
if c.get('mode')=='profiles_only':
    profile()
if c.get('mode')=='all':
    fork=call(['fork'])
    data=json.loads(fork.stdout.decode().splitlines()[-1])
    target=data['workspace']
    profile(target)
if c.get('mode')=='active':
    Path(c['ready']).write_text('ready')
    while not Path(c['go']).exists(): time.sleep(.01)
    call(['evaluate'])
if c.get('mode')=='budget':
    call(['profile'])
    call(['evaluate'])
    sys.exit(c.get('exit',0))
if c.get('mode') in ('interrupt','active'):
    Path(c['ready']).write_text('ready')
    while True: time.sleep(.02)
if c.get('tamper_method'):
    method=Path(os.environ['PATH'].split(os.pathsep)[0])/'modes/basic/runner.py'
    method.write_text("from pathlib import Path\nPath("+repr(c['tamper_method'])+").write_text('executed tampered collector')\n")
    if c.get('corrupt_method_backup'):
        manifest=json.loads((root/'protection/manifest.json').read_text())
        entry=manifest['roots']['public_modes_basic']['entries']['runner.py']
        backup=root/'protection'/entry['backup']
        backup.chmod(0o600)
        backup.write_bytes(b'corrupt method recovery source')
sys.exit(c.get('exit',0))
'''


class LauncherTests(ActionTestBase):
    def setUp(self):
        super().setUp()
        # Keep the venv path: resolving its symlink loses local PyYAML.
        self.python = os.path.abspath(sys.executable)
        shutil.rmtree(self.bench / 'launcher')
        shutil.copytree(SOURCE_BENCHMARK / 'launcher', self.bench / 'launcher', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(SOURCE_BENCHMARK / 'pto-launcher', self.bench / 'pto-launcher')
        for name in ('ptobench', 'pto-launcher'):
            (self.bench / name).chmod(0o755)
        self.tool = self.repo / 'fake-tool'
        self.tool.write_text('#!' + self.python + '\n' + FAKE_TOOL)
        self.tool.chmod(0o755)
        self.config = self.repo / 'experiment.yaml'
        self.launcher_control = self.repo / 'optimizer.json'
        self.seen = self.repo / 'seen.json'
        self.ready = self.repo / 'ready'
        self.go = self.repo / 'go'
        self.env.update(S5_CONTROL=str(self.launcher_control))
        self.env['PATH'] = str(Path(self.python).parent) + os.pathsep + self.env['PATH']
        self.env['CODEX_HOME'] = str(self.repo/'codex_home')
        self.env['PI_CODING_AGENT_DIR'] = str(self.repo/'pi_home')
        # Diagnostic methods are test-only; Ours remains reserved.
        method = self.bench/'modes/diagnostic_test'
        method.mkdir()
        (method/'mode.yaml').write_text('schema_version: 2\nid: diagnostic_test\nimplemented: true\ndescription: Test diagnostic policy\nentrypoint: runner.py\nprofile_strategy: split\n')
        (method/'instructions.md').write_text('Test diagnostics.\n')
        shutil.copyfile(self.bench/'modes/basic/runner.py', method/'runner.py')
        shutil.copyfile(self.bench/'modes/basic/.env', method/'.env')
        self.launcher_config()

    def launcher_config(self, backend='codex', budget=4, **control):
        import yaml
        profile = yaml.safe_load((self.bench / 'modes/_shared/profiles' / backend / 'default.yaml').read_text())
        # Fake native transports have no provider registry or API token limits.
        # Actual limit handling is exercised independently in test_mode_provider.
        profile.pop('context_window', None)
        profile.pop('max_output_tokens', None)
        for key in ('argv','version_argv','help_argv'):
            profile[key][0] = str(self.tool)
        (self.bench / 'modes/_shared/profiles' / backend / 'default.yaml').write_text(yaml.safe_dump(profile))
        write_json(self.launcher_control, {'seen':str(self.seen),'ready':str(self.ready),'go':str(self.go),**control})
        self.runs = self.repo / 'launcher_runs'
        self.snapshots = self.repo / 'launcher_snapshots'
        if backend == 'pi':
            self.runs.mkdir(exist_ok=True)
            self.snapshots.mkdir(exist_ok=True)
            self.runs.chmod(0o755)
            self.snapshots.chmod(0o755)
        experiment_mode = control.get('experiment_mode', 'diagnostic_test' if control.get('mode') in ('all','s6','profiles_only') else 'basic')
        self.env['S5_PROFILE_STRATEGY'] = yaml.safe_load(
            (self.bench / 'modes' / experiment_mode / 'mode.yaml').read_text()
        ).get('profile_strategy', 'combined')
        self.config.write_text(yaml.safe_dump({'case':self.case_id,'mode':experiment_mode,
            'optimizer':{'id':'fake_'+backend,'backend':backend},
            'output':{'runs_root':'launcher_runs','snapshots_root':'launcher_snapshots'},
            'budget':{'max_hardware_actions':budget,'finish_actions':1}}))

    def isolate_fake_agent(self):
        import yaml
        self.env['PTOBENCH_SANDBOX'] = '1'
        control = json.loads(self.launcher_control.read_text())
        for name in ('seen', 'ready', 'go', 'child', 'child_pid', 'worker_pid'):
            control.pop(name, None)
        profile_path = self.bench / 'modes/_shared/profiles/pi/default.yaml'
        profile = yaml.safe_load(profile_path.read_text())
        position = profile['argv'].index('{prompt}')
        profile['argv'][position:position] = [
            '--test-control-json', json.dumps(control, separators=(',', ':')),
        ]
        profile_path.write_text(yaml.safe_dump(profile))

    def launch(self, asynchronous=False):
        prepare_fake_pi_bundle(self)
        command = [self.python,str(self.bench / 'pto-launcher'),'run',str(self.config)]
        if asynchronous:
            p=subprocess.Popen(command,cwd='/',env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.addCleanup(self.cleanup_launcher,p)
            return p
        p=subprocess.run(command,cwd='/',env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
        self.assertTrue(p.stdout,p.stderr.decode())
        return p,json.loads(p.stdout.decode().splitlines()[-1])

    def cleanup_launcher(self, process):
        if process.poll() is None:
            self.configure()
            process.terminate()
            try:
                process.communicate(timeout=45)
            except subprocess.TimeoutExpired:
                process.send_signal(signal.SIGINT)
                process.communicate(timeout=20)
        for pipe in (process.stdout,process.stderr):
            if pipe is not None:
                pipe.close()

    def result(self, process):
        out,err=process.communicate(timeout=45)
        value=json.loads(out.decode().splitlines()[-1])
        return value,out,err

    def assert_final(self, value, used=1):
        root=Path(value['session_root'])
        self.handle = session.SessionHandle(root,Path(session.read_session(root)['snapshots_root']),root/'protection',root/'workspaces/worker_000')
        final=json.loads((root/'final_result.json').read_text())
        self.assertEqual(final['status'],'completed',final)
        self.assertEqual(final['hardware_actions_used'],used)
        self.assertEqual(final['finish_actions_used'],1)
        self.assertIsNone(session.read_session(root)['active_action'])
        return root,final

    def test_launcher_records_can_be_registered_together_before_commit(self):
        names = ('experiment.json', 'method.json', 'method_request.json',
                 'baseline.json', 'environment.json', 'method_collection.json',
                 'launcher_result.json')
        handle = session.create_session(self.case_id, {'id': 'fake_codex', 'backend': 'codex'},
            self.output, {'max_hardware_actions': 2, 'finish_actions': 1},
            initial_record_paths=names)
        root = handle.session_root
        launcher_runner.save(root, 'experiment.json', {'case': self.case_id})
        manifest = json.loads((root / 'protection/manifest.json').read_text())
        for name in names:
            logical = 'launcher_' + name.replace('.', '_').replace('/', '_')
            self.assertIn(logical, manifest['roots'])
            self.assertEqual(manifest['roots'][logical]['group'], 'records')
        self.assertEqual(json.loads((root / 'experiment.json').read_text()), {'case': self.case_id})

    def test_initial_launcher_and_method_records_share_one_registration(self):
        root = self.create(optimizer='pi').session_root
        resources = root / 'method/resources'
        resources.mkdir(parents=True)
        (resources / 'knowledge.json').write_text('{}\n')
        initial = {'experiment.json': {'case': self.case_id}, 'method.json': {'id': 'test'},
                   'method_request.json': {'schema_version': 1}, 'baseline.json': {'median': 1.0},
                   'environment.json': {'platform': 'cpu'}}
        launcher_runner.retain_method_reply(root,
            {'schema_version': 1, 'records': {'method/knowledge.json': {'enabled': True}},
             'archives': {'knowledge': 'method/resources'}}, initial_records=initial)
        manifest = json.loads((root / 'protection/manifest.json').read_text())
        for name in (*initial, 'method/knowledge.json'):
            logical = 'launcher_' + name.replace('.', '_').replace('/', '_')
            self.assertIn(logical, manifest['roots'])
            self.assertEqual(manifest['roots'][logical]['group'], 'records')
        self.assertEqual(manifest['roots']['participant_knowledge']['group'], 'agent_archive')
        self.assertEqual(json.loads((root / 'method/knowledge.json').read_text()), {'enabled': True})

    def test_three_adapters_real_cli_all_five_operations(self):
        for backend in ('codex','claudecode','opencode'):
            with self.subTest(backend=backend):
                self.launcher_config(backend,mode='all')
                p,value=self.launch()
                self.assertEqual(p.returncode,0,(value,p.stderr.decode()))
                root,final=self.assert_final(value,4)
                seen=json.loads(self.seen.read_text())
                self.assertEqual(seen['cwd'],str(root/'workspaces/worker_000'))
                self.assertEqual(seen['environment']['PTOBENCH_SESSION_ROOT'],str(root))
                self.assertEqual(seen['environment']['PTOBENCH_OPTIMIZER_ID'],'fake_'+backend)
                self.assertTrue(Path(seen['cli']).is_file())
                prompt=(root/'prompt.txt').read_text()
                self.assertEqual(seen['argv'][-1],prompt)
                self.assertIn('.benchmark_context/README.md', prompt)
                self.assertNotIn(str(root/'method/README.md'), prompt)
                guide=(root/'workspaces/worker_000/.benchmark_context/README.md').read_text()
                for command in ('evaluate', 'profile runtime', 'profile pmu', 'profile compiler', 'fork'):
                    self.assertIn('ptobench '+command, guide)
                self.assertNotIn('profile hardware',prompt)
                self.assertNotIn('profile compile ',prompt)
                self.assertIn('optimizer stdout',(root/'process.log').read_text())
                self.assertIn('optimizer stderr',(root/'process.log').read_text())
                self.assertEqual(json.loads((root/'backend.json').read_text())['backend']['version'],'fake-tool 1.0')

    def test_basic_codex_profiles_baseline_and_native_history(self):
        backend = 'codex'
        self.launcher_config(backend, mode='all', experiment_mode='basic')
        p, value = self.launch()
        self.assertEqual(p.returncode, 0, (value, p.stderr.decode()))
        root, final = self.assert_final(value, 2)
        record = session.read_session(root)
        self.assertEqual(record['mode']['id'], 'basic')
        self.assertEqual(record['profiling_modes'], ['combined'])
        self.assertEqual(json.loads((root/'baseline.json').read_text())['fixtures']['fixture_000']['median_effective_us'], 1.0)
        self.assertNotIn('First run ptobench evaluate', (root/'prompt.txt').read_text())
        protection=json.loads((root/'protection/manifest.json').read_text())
        self.assertNotIn('participant_reference_knowledge', protection['roots'])
        self.assertFalse((root/'method/resources/raw_wiki').exists())
        self.assertFalse((root/'method/README.md').exists())
        self.assertFalse((root/'method/knowledge.json').exists())
        self.assertNotIn('automatically selects', (root/'prompt.txt').read_text())
        native = json.loads((root/'agent/session.json').read_text())
        self.assertEqual(native['status'], 'recorded')
        self.assertEqual(native['run_id'], root.name)
        history = root/native['files'][0]['path']
        self.assertIn('native history', history.read_text())
        self.assertIn('"input": 12', history.read_text())
        self.assertTrue((root/'agent/events.jsonl').is_file())
        self.assertIn('python', json.loads((root/'environment.json').read_text()))
        for worker in record['workspaces'].values():
            context = root/worker['path']/'.benchmark_context'
            self.assertFalse((context/'profiling_modes.json').exists())
            self.assertFalse((context/'fixture_catalog.json').exists())
            self.assertIn('ptobench profile .', (context/'README.md').read_text())
            self.assertFalse((root/worker['path']/'raw_wiki').exists())
        env = {**self.env, 'PTOBENCH_SESSION_ROOT': str(root)}
        help_result = subprocess.run([self.python, str(self.bench/'ptobench'), '--help'], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('profile', help_result.stdout.decode())
        events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertTrue(all(event['run_id'] == root.name for event in events))
        measured = [event for event in events if event['hardware_action']]
        self.assertEqual(len(measured), 3)  # evaluate, combined profile and finish
        attempts = [event for event in events if event.get('attempt_id') is not None]
        self.assertEqual([event['attempt_id'] for event in attempts], [1, 2, 3])
        self.assertEqual(next(event for event in events if event['action'] == 'fork')['seq'], 2)
        self.assertNotIn('attempt_id', next(event for event in events if event['action'] == 'fork'))
        run = json.loads((root/'launcher_result.json').read_text())['run']
        status=launcher_runner.status(root)
        self.assertEqual(status['remaining_hardware_actions'],3)
        self.assertEqual((run['agent'], run['backend']), ('basic', backend))
        self.assertEqual(run['counts'], {
            'attempts': 3, 'check_calls': 2, 'bench_calls': 2,
            'profile_calls': 1, 'fork_events': 1,
        })
        self.assertEqual(run['hardware_usage']['total_actions'], 3)
        self.assertEqual(run['hardware_usage']['budgeted_actions'], 1)
        self.assertEqual(run['token_usage']['input_tokens'], 12)
        self.assertEqual(run['token_usage']['output_tokens'], 3)
        self.assertEqual(run['token_usage']['reported_total_tokens'], 15)
        for event in measured:
            self.assertIn('started_at', event)
            index = json.loads((root/event['artifact_index']).read_text())
            self.assertTrue((root/index['candidate_snapshot']/'candidate/candidate.py').is_file())

    def test_disabled_sandbox_stops_before_mode_runner_launch(self):
        from launcher import methods

        request = self.repo / 'method-request.json'
        runner = self.repo / 'mode_runner.py'
        marker = self.repo / 'mode-runner-started'
        request.write_text(json.dumps({'preflight': {'profile': {'backend': 'pi'}}}))
        runner.write_text(f"from pathlib import Path; Path({str(marker)!r}).write_text('started')\n")

        with patch.dict(os.environ, {'PTOBENCH_SANDBOX': '0'}), \
             patch.object(methods.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'Pi Agent requires filesystem isolation'):
                methods.start({'path': str(runner)}, str(request), str(self.repo), {})

        spawn.assert_not_called()
        self.assertFalse(marker.exists())

    def test_retired_session_rejected_before_pi_mode_runner_launch(self):
        self.launcher_config('pi', experiment_mode='basic')
        root = self.handle.session_root
        record = session.read_session(root)
        retired = dict(record)
        retired['schema_version'] = 1
        integrity.commit_records(self.handle.protection_root,
                                 {'session': session._bytes(retired)})
        (root / 'session.json').write_text('{"damaged": true}\n')
        before = tree_digest(root)

        with patch.dict(os.environ, {**self.env, 'PTOBENCH_SANDBOX': '0'}), \
             patch.object(launcher_runner.session, 'create_session', return_value=self.handle) as create, \
             patch.object(launcher_runner.methods, 'start') as start:
            with self.assertRaises(session.SessionError) as caught:
                launcher_runner.run(str(self.config))

        self.assertEqual(caught.exception.error_code, 'invalid_request')
        create.assert_called_once()
        start.assert_not_called()
        self.assertEqual(tree_digest(root), before)

    def test_reserved_methods_fail_before_creating_run(self):
        for mode in ('ours',):
            self.launcher_config(experiment_mode=mode)
            p, value = self.launch()
            self.assertEqual(p.returncode, 2)
            self.assertIn('not implemented', value['message'])
            self.assertFalse(self.runs.exists())

    def test_independent_method_runs_without_basic_instructions_or_known_backend(self):
        import yaml
        method = self.bench / 'modes/independent_test'
        method.mkdir()
        (method / 'mode.yaml').write_text('schema_version: 2\nid: independent_test\nimplemented: true\ndescription: Independent participant\nentrypoint: runner.py\n')
        # This participant uses no Basic helper, agent profile or instruction file.
        (method / 'runner.py').write_text('''import json, subprocess, sys
operation = sys.argv[1]
response = {'schema_version': 1}
if operation == 'run':
    result = subprocess.run(['ptobench', 'evaluate'])
    raise SystemExit(result.returncode)
elif operation == 'prepare':
    response['records'] = {'method/policy.txt': 'Independent participant policy.'}
elif operation == 'collect':
    response['recording'] = {'status': 'recorded'}
    response['records'] = {'method/completion.json': {'completed': True}}
print(json.dumps(response))
''')
        config = yaml.safe_load(self.config.read_text())
        config['mode'] = 'independent_test'
        config['optimizer']['backend'] = 'arbitrary_tool'
        self.config.write_text(yaml.safe_dump(config))
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        root, _ = self.assert_final(value)
        self.assertEqual(session.read_session(root)['mode'], {'id': 'independent_test'})
        self.assertEqual((root / 'method/policy.txt').read_text(), 'Independent participant policy.')
        self.assertTrue((root / 'method/completion.json').is_file())
        self.assertFalse((root / 'prompt.txt').exists())

    def test_missing_native_history_reports_incomplete_and_keeps_performance(self):
        self.launcher_config(missing_session=True)
        process, value = self.launch()
        self.assertEqual(process.returncode, 5, value)
        self.assertEqual(value['error_code'], 'recording_incomplete')
        self.assertEqual(value['agent_recording']['status'], 'missing')
        self.assert_final(value)

    def test_method_code_change_is_recorded_without_restore_or_execution_during_collection(self):
        sentinel = self.repo / 'tampered_collector_executed'
        original = (self.bench / 'modes/basic/runner.py').read_bytes()
        self.launcher_config(tamper_method=str(sentinel))
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        root, _ = self.assert_final(value)
        self.assertFalse(sentinel.exists())
        self.assertNotEqual((self.bench / 'modes/basic/runner.py').read_bytes(), original)
        self.assertEqual(json.loads((root / 'method_collection.json').read_text())['recording']['status'],
                         'unavailable_mode_source_changed')
        report = json.loads((root / 'launcher_result.json').read_text())['integrity_report']
        self.assertIn('public_modes_basic/runner.py', report['blocked_paths'])

    def test_unavailable_mode_backup_does_not_block_collection(self):
        sentinel = self.repo / 'tampered_collector_executed'
        self.launcher_config(tamper_method=str(sentinel), corrupt_method_backup=True)
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        self.assertFalse(sentinel.exists())
        root = Path(value['session_root'])
        self.assertTrue((root / 'final_result.json').exists())
        self.assertEqual(json.loads((root / 'method_collection.json').read_text())['recording']['status'],
                         'unavailable_mode_source_changed')
        report = json.loads((root / 'launcher_result.json').read_text())['integrity_report']
        self.assertIn('public_modes_basic/runner.py', report['blocked_paths'])
        self.assertIn('thread.started', (root / 'stdout.log').read_text())
        self.assertTrue(list((self.repo / 'codex_home/sessions').glob('*.jsonl')))

    def test_effective_codex_configuration_is_frozen_without_credentials(self):
        directory = Path(self.env['CODEX_HOME'])
        directory.mkdir()
        (directory/'config.toml').write_text('model = "fixture-model"\nmodel_reasoning_effort = "high"\napi_key = "fixture-private-value"\n[compaction]\nkeep_recent_tokens = 1234\n')
        (directory/'AGENTS.md').write_text('Frozen global instructions.\n')
        p, value = self.launch()
        self.assertEqual(p.returncode, 0, value)
        root, _ = self.assert_final(value)
        argv = json.loads((root/'backend.json').read_text())['argv']
        self.assertEqual(argv[argv.index('--model')+1], 'fixture-model')
        self.assertIn('model_reasoning_effort="high"', argv)
        config = json.loads((root/'agent/configuration.json').read_text())
        captured = '\n'.join((root/item['path']).read_text() for item in config['files'])
        self.assertIn('Frozen global instructions.', captured)
        self.assertIn('keep_recent_tokens', captured)
        self.assertNotIn('fixture-private-value', captured)
        self.assertIn('[REDACTED]', captured)
        (directory/'AGENTS.md').write_text('Changed later.\n')
        retained = '\n'.join((root/item['path']).read_text() for item in config['files'])
        self.assertIn('Frozen global instructions.', retained)
        self.assertNotIn('Changed later.', retained)

    def test_s6_fork_helper_edit_profile_evaluate_exit_provenance(self):
        self.launcher_config('pi',budget=4,mode='s6')
        self.isolate_fake_agent()
        p,value=self.launch()
        self.assertEqual(p.returncode,0,(value,p.stderr.decode()))
        root,final=self.assert_final(value,3)
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['action'] for event in events],['evaluate','fork','profile','evaluate','finish'])
        self.assertTrue(all(event['run_id'] == root.name for event in events))
        first,profile,improved=events[0],events[2],events[3]
        self.assertEqual([event['attempt_id'] for event in (first,profile,improved,events[4])], [1,2,3,4])
        self.assertNotIn('attempt_id', events[1])
        self.assertEqual(events[1]['parent_attempt_id'], first['attempt_id'])
        self.assertEqual(profile['parent_attempt_id'], first['attempt_id'])
        self.assertEqual(improved['parent_attempt_id'], profile['attempt_id'])
        self.assertNotEqual(first['content_sha256'],improved['content_sha256'])
        self.assertEqual(profile['content_sha256'],improved['content_sha256'])
        self.assertIsNone(profile['snapshot_id'])
        self.assertNotIn('correctness', profile['observation'])
        self.assertEqual(final['selected_snapshot'],improved['snapshot_id'])
        selected=Path(session.read_session(root)['snapshots_root'])/improved['snapshot_id']
        self.assertEqual((selected/'source/candidate/helper.py').read_text(),'VALUE = 2\n')
        self.assertEqual((root/'final/workspace/candidate/helper.py').read_text(),'VALUE = 2\n')
        self.assertEqual(events[-1]['result']['runs'][0]['median_effective_us'],7.0)
        self.assertFalse((root/'final/workspace/__pycache__').exists())
        status=launcher_runner.status(root)
        self.assertEqual(status['remaining_hardware_actions'],2)
        run=json.loads((root/'launcher_result.json').read_text())['run']
        self.assertEqual(run['hardware_usage']['budgeted_actions'],2)
        self.assertEqual(run['hardware_usage']['total_actions'],4)
        self.assertEqual(run['counts']['profile_calls'],1)

    def test_s6_profiles_only_cannot_supply_a_correctness_verified_candidate(self):
        self.launcher_config(budget=4,mode='profiles_only')
        p,value=self.launch()
        self.assertEqual(p.returncode,1,(value,p.stderr.decode()))
        root=Path(value['session_root'])
        self.assertEqual(value['error_code'], 'no_valid_snapshot')
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertTrue(all(event['observation']['latency_us'] is None for event in events[:3]))
        self.assertTrue(all(event['snapshot_id'] is None for event in events))
        self.assertTrue(all('correctness' not in event['observation'] for event in events))
        self.assertEqual(session.read_session(root)['usage'], {'hardware_actions': 3, 'finish_actions': 0})
        self.assertEqual(len(self.calls()), 3)

    def test_s6_all_failed_or_protected_tamper_never_finishes_seed(self):
        for control in ({'fail':True},{'tamper':'fixed'}):
            with self.subTest(control=control):
                self.launcher_config(budget=2)
                self.configure(**control)
                p,value=self.launch()
                self.assertEqual(p.returncode,1,value)
                record=session.read_session(value['session_root'])
                self.assertEqual(record['usage'],{'hardware_actions':1,'finish_actions':0})
                self.assertEqual(value['error_code'],'no_valid_snapshot')
                self.assertFalse(list(Path(record['snapshots_root']).iterdir()))
                self.assertEqual((self.bench/'cases/test/case/public/config.py').read_text(),'SIZE = 2\n')

    def test_nonzero_exit_still_finalizes_best_history(self):
        self.launcher_config(exit=7)
        p,value=self.launch()
        self.assertEqual(p.returncode,0,value)
        self.assertEqual(value['optimizer_exit_code'],7)
        self.assertEqual(value['close_reason'],'optimizer_nonzero_exit')
        self.assert_final(value)

    def test_early_exit_without_snapshot_does_not_measure(self):
        self.launcher_config(mode='empty')
        p,value=self.launch()
        self.assertEqual(p.returncode,1,value)
        self.assertEqual(value['error_code'],'no_valid_snapshot')
        self.assertEqual(len(self.calls()),0)
        self.assertEqual(session.read_session(value['session_root'])['usage']['finish_actions'],0)

    def test_evaluate_budget_allows_profile_and_dual_large_logs(self):
        self.launcher_config('pi',budget=1,mode='budget',huge=True)
        self.isolate_fake_agent()
        p,value=self.launch()
        self.assertEqual(p.returncode,0,value)
        root,_=self.assert_final(value,2)
        self.assertEqual(value['close_reason'],'optimizer_exit')
        self.assertIn(b'O'*262144,(root/'stdout.log').read_bytes())
        self.assertIn(b'E'*262144,(root/'stderr.log').read_bytes())
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['action'] for event in events],
                         ['evaluate','profile','evaluate','finish'])
        self.assertTrue(events[1]['hardware_action'])
        self.assertFalse(events[2]['hardware_action'])
        self.assertFalse(events[2]['action_started'])
        self.assertEqual(len(list((root/'staging').glob('action_*/result.json'))),2)
        run=json.loads((root/'launcher_result.json').read_text())['run']
        self.assertEqual(run['hardware_usage']['budgeted_actions'],1)
        self.assertEqual(run['hardware_usage']['total_actions'],3)

    def test_blocked_terminal_does_not_stall_finalization_or_exit(self):
        self.launcher_config('pi',budget=1,mode='budget',huge=True)
        self.isolate_fake_agent()
        p=self.launch(True)
        # Intentionally leave both terminal pipes undrained until run exits.
        self.assertEqual(p.wait(timeout=35),0)
        root=next(self.runs.glob('*/*/*'))
        final=json.loads((root/'final_result.json').read_text())
        self.assertEqual(final['status'],'completed')
        self.assertEqual(final['finish_actions_used'],1)
        self.assertIn(b'O'*262144,(root/'stdout.log').read_bytes())
        self.assertIn(b'E'*262144,(root/'stderr.log').read_bytes())

    def test_leftover_worker_is_killed(self):
        child=self.repo/'worker.pid'
        self.launcher_config(child=str(child))
        p,value=self.launch()
        self.assertEqual(p.returncode,0,value)
        self.assert_final(value)
        self.assertIsNone(session.process_identity(int(child.read_text())))

    def test_user_interrupt_finalizes_and_returns_130(self):
        self.launcher_config(mode='interrupt')
        p=self.launch(True)
        self.wait_file(self.ready,p)
        p.send_signal(signal.SIGTERM)
        value,_,_=self.result(p)
        self.assertEqual(p.returncode,130,value)
        self.assert_final(value)

    def test_interrupt_active_evaluator_and_child(self):
        self.launcher_config(mode='active')
        p=self.launch(True)
        self.wait_file(self.ready,p)
        action_ready=self.repo/'action.ready'; child=self.repo/'eval.child'
        self.configure(hang=True,ready=str(action_ready),child=True,child_pid=str(child))
        self.go.write_text('go')
        self.wait_file(action_ready,p)
        self.configure()
        p.send_signal(signal.SIGINT)
        value,_,_=self.result(p)
        self.assertEqual(p.returncode,130,value)
        self.assert_final(value,2)
        self.assertIsNone(session.process_identity(int(child.read_text())))

    def test_evaluate_budget_exhaustion_allows_profile_after_active_action(self):
        self.launcher_config('pi',budget=1,mode='budget')
        write_json(self.launcher_control, {**json.loads(self.launcher_control.read_text()), 'pause': True})
        self.isolate_fake_agent()
        p=self.launch(True)
        deadline=time.monotonic()+10
        roots=list(self.runs.glob('*/*/*'))
        while not roots and p.poll() is None and time.monotonic()<deadline:
            time.sleep(.01)
            roots=list(self.runs.glob('*/*/*'))
        self.assertTrue(roots, f'Launcher did not create a Session: exit={p.poll()}')
        root=roots[0]
        action_dir=root/'staging/action_000001'
        action_ready=action_dir/'test-ready'
        release=action_dir/'test-release'
        self.wait_file(action_ready,p)
        time.sleep(1.2)
        self.assertIsNone(p.poll())
        record=session.read_session(root)
        self.assertIsNotNone(record['active_action'])
        self.assertFalse(record['exploration_closed'])
        release.write_text('go')
        value,_,_=self.result(p)
        self.assertEqual(p.returncode,0,value)
        root,_=self.assert_final(value,2)
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['action'] for event in events],
                         ['evaluate','profile','evaluate','finish'])
        self.assertTrue(events[1]['hardware_action'])
        self.assertFalse(events[2]['hardware_action'])
        self.assertFalse(events[2]['action_started'])
        self.assertEqual(len(list((root/'staging').glob('action_*/result.json'))),2)

    def test_status_no_hardware_and_repairs_records(self):
        p,value=self.launch()
        root,_=self.assert_final(value)
        count=len(self.calls())
        (root/'session.json').write_text('{}')
        p=subprocess.run([self.python,str(self.bench/'pto-launcher'),'status',str(root)],env=self.env,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr.decode())
        status=json.loads(p.stdout)
        self.assertEqual(status['usage'],{'hardware_actions':1,'finish_actions':1})
        self.assertEqual(status['final_result_path'],str(root/'final_result.json'))
        self.assertEqual(len(self.calls()),count)

    def test_invalid_configuration_fails_before_session(self):
        original=self.config.read_text()
        for bad in ('unknown: true\n','case: test.case\ncase: test.case\noptimizer: {id: x, backend: codex}\n',
                    'case: !!python/object/apply:os.system [false]\n',
                    'case: test.case\noptimizer: {id: ../x, backend: codex}\n',
                    'case: test.case\noptimizer: {id: x, backend: missing}\n',
                    'case: test.case\noptimizer: {id: x, backend: codex, profile: missing}\n',
                    'case: test.case\noptimizer: {id: x, backend: codex}\nbudget: {max_hardware_actions: true}\n',
                    f'case: test.case\noptimizer: {{id: x, backend: codex}}\noutput: {{runs_root: {self.bench}/launcher}}\n'):
            self.config.write_text(bad)
            p,value=self.launch()
            self.assertNotEqual(p.returncode,0,value)
            self.assertFalse(self.runs.exists())
        self.config.write_text(original)
        (self.bench/'bench').write_text('old')
        p,value=self.launch()
        self.assertNotEqual(p.returncode,0,value)
        self.assertFalse(self.runs.exists())

    def test_sensitive_profile_rejected_and_host_secret_not_recorded(self):
        import yaml
        profile_path=self.bench/'modes/_shared/profiles/codex/default.yaml'
        original=profile_path.read_text()
        for env in ({'API_KEY':'private'},{'PATH':'/bad'},{'PTOBENCH_SESSION_ROOT':'/bad'}):
            profile=yaml.safe_load(original); profile['env']=env
            profile_path.write_text(yaml.safe_dump(profile))
            p,value=self.launch()
            self.assertNotEqual(p.returncode,0,value)
            self.assertFalse(self.runs.exists())
        profile_path.write_text(original)
        self.env['API_KEY']='S5_HOST_SECRET_NEVER_RECORD'
        p,value=self.launch()
        self.assertEqual(p.returncode,0,value)
        root,_=self.assert_final(value)
        for name in ('experiment.json','profile.json','backend.json','session.json','optimizer_process.json','launcher_result.json'):
            self.assertNotIn(self.env['API_KEY'],(root/name).read_text())

    def test_yaml_defaults_and_paths_ignore_startup_cwd(self):
        sys.path.insert(0,str(SOURCE_BENCHMARK))
        from launcher.config import load_experiment
        self.config.write_text('case: test.case\noptimizer: {id: x, backend: codex}\n')
        config=load_experiment(self.config)
        self.assertEqual(config.optimizer['profile'],'default')
        self.assertEqual(config.budget,{'max_hardware_actions':20,'finish_actions':1})
        self.assertEqual(config.output['runs_root'],str(self.repo/'runs'))
        self.assertEqual(config.output['snapshots_root'],str(self.repo/'snapshots'))

    def test_infrastructure_failure_stops_owned_evaluator_without_final(self):
        self.launcher_config(mode='active')
        p=self.launch(True)
        self.wait_file(self.ready,p)
        action_ready=self.repo/'action.ready'; child=self.repo/'eval.child'
        self.configure(hang=True,ready=str(action_ready),child=True,child_pid=str(child))
        self.go.write_text('go')
        self.wait_file(action_ready,p)
        root=next(self.runs.glob('*/*/*'))
        record=session.read_session(root)
        evaluator=record['active_action']['process']
        manifest=json.loads((root/'protection/manifest.json').read_text())
        backup=manifest['roots']['session']['entries']['']['backup']
        (root/'protection'/backup).chmod(0o600)
        (root/'protection'/backup).write_bytes(b'corrupt recovery source')
        (root/'session.json').write_text('{}')
        value,_,_=self.result(p)
        self.assertEqual(p.returncode,5,value)
        self.assertEqual(value['error_code'],'infrastructure_error')
        self.assertFalse((root/'final_result.json').exists())
        self.assertIsNone(session.process_identity(evaluator['pid']))
        self.assertIsNone(session.process_identity(int(child.read_text())))
        self.assertEqual(len(self.calls()),2)

    def test_user_interrupt_during_final_measurement(self):
        self.launcher_config(mode='interrupt')
        p=self.launch(True)
        self.wait_file(self.ready,p)
        final_ready=self.repo/'final.ready'
        self.configure(hang=True,ready=str(final_ready))
        # Normal Optimizer exit triggers finalization; signal the Launcher while
        # its independently reserved finish action is actually running.
        root=next(self.runs.glob('*/*/*'))
        identity=json.loads((root/'optimizer_process.json').read_text())
        os.killpg(identity['pgid'],signal.SIGTERM)
        self.wait_file(final_ready,p)
        p.send_signal(signal.SIGINT)
        value,_,_=self.result(p)
        self.assertEqual(p.returncode,130,value)
        final=json.loads((root/'final_result.json').read_text())
        self.assertEqual(final['error_code'],'interrupted')
        self.assertEqual(final['finish_actions_used'],1)


if __name__ == '__main__':
    import unittest
    unittest.main()
