"""Native tool transports and shared-budget integration; no LLM or NPU calls."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from action_test_support import ActionTestBase, SOURCE_BENCHMARK
import agent_launch
import session
import test_launcher as launcher_support
from modes.cannbot import runtime as cann_bot_runtime


FAKE_NATIVE = r'''
import json, os, signal, subprocess, sys, time, uuid
from pathlib import Path
if '--version' in sys.argv:
    print('fake-cannbot-native 1.0'); sys.exit()
if '--help' in sys.argv:
    print('--skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --print --json --model --provider --thinking --mode --session-dir --test-control-json'); sys.exit()
backend='pi' if '--extension' in sys.argv else 'codex'
if backend=='pi':
    control=json.loads(sys.argv[sys.argv.index('--test-control-json')+1])
    primary=os.environ['PTOBENCH_CANNBOT_ALLOW_TASK']=='1'
    assert os.environ.get('PTOBENCH_CANNBOT_SOCKET') == '/ptobench/cannbot.sock', repr(os.environ.get('PTOBENCH_CANNBOT_SOCKET'))
    system_prompt=Path(sys.argv[sys.argv.index('--append-system-prompt')+1])
    assert system_prompt.is_file()
else:
    config=json.loads(Path(os.environ['CANNBOT_CONFIG']).read_text())
    control=json.loads(Path(os.environ['S5_CONTROL']).read_text())
    primary=os.environ['CANNBOT_ALLOW_TASK']=='1'
identity=str(uuid.uuid4())
native=None
if not (control.get('missing_child') and not primary):
    if backend=='codex':
        native=Path(os.environ['CODEX_HOME'])/'sessions'/('rollout-'+identity+'.jsonl')
        header={'type':'session_meta','payload':{'id':identity,'cwd':os.getcwd()}}
        print(json.dumps({'type':'thread.started','thread_id':identity}),flush=True)
    else:
        native=Path(sys.argv[sys.argv.index('--session-dir')+1])/(identity+'.jsonl')
        header={'type':'session','id':identity,'cwd':os.getcwd()}
        print(json.dumps(header),flush=True)
    native.parent.mkdir(parents=True,exist_ok=True)
    native.write_text(json.dumps(header)+'\n')
def record(value):
    print(json.dumps(value),flush=True)
    if native:
        with native.open('a') as stream: stream.write(json.dumps(value)+'\n')
def cli(args):
    p=subprocess.run(['ptobench',*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    value={'argv':args,'exit_code':p.returncode,'stdout':p.stdout.decode(),'stderr':p.stderr.decode()}
    record({'type':'public_cli','value':value})
    return value
def reply(text,code=0):
    event={'type':'message_end','message':{'role':'assistant','content':[{'type':'text','text':text}],'stopReason':'stop','usage':{'input':31,'output':7}}} if backend=='pi' else {'type':'item.completed','item':{'type':'agent_message','text':text}}
    record(event)
    sys.exit(code)
if primary:
    requests=[{'name':'skill','arguments':{'name':'pypto-op-perf-tune'}},
              {'name':'todowrite','arguments':{'todos':[{'id':'original-todo','content':'opaque upstream todo','status':'in_progress'}]}},
              {'name':'todoread','arguments':{}},
              {'name':'Task','arguments':{'subagent_type':'pypto-op-optimizer','prompt':'evaluate','description':'unchanged input'}},
              {'name':'Task','arguments':{'subagent_type':'pypto-op-optimizer','prompt':'profiles'}}]
    if control.get('fork_child'):
        fork=cli(['fork'])
        child=json.loads(fork['stdout'].splitlines()[-1])['workspace']
        requests[-1]['arguments']['workspace']=child
    if backend=='pi':
        script="""
import { pathToFileURL } from 'node:url';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
const {default: extension} = await import(pathToFileURL(process.argv[1]).href);
const registered = new Map();
extension({registerTool: tool => registered.set(tool.name,tool)});
const requests=JSON.parse(process.argv[2]);
const control=JSON.parse(process.argv[3]);
if (registered.has('Task') !== (process.env.PTOBENCH_CANNBOT_ALLOW_TASK==='1'))
  throw new Error('CANNBot Task availability does not match the role');
for (let i=0;i<requests.length;i++) {
 const request=requests[i];
 const controller=new AbortController();
 let interval, fallback;
 if (control.cancel_child && request.name==='Task' && request.arguments.prompt==='evaluate') {
   fallback=setTimeout(()=>controller.abort(),15000);
   interval=setInterval(()=>{
     if (existsSync(join(process.cwd(),'candidate','.fake-agent-control.py'))) {
       clearTimeout(fallback);
       controller.abort();
     }
   },20);
 }
 const value=await registered.get(request.name).execute(String(i+1),request.arguments,controller.signal);
 if (interval) clearInterval(interval);
 if (fallback) clearTimeout(fallback);
 console.log(JSON.stringify({request,value}));
}
"""
        extension=sys.argv[sys.argv.index('--extension')+1]
        p=subprocess.run([control['node'],'--experimental-strip-types','--input-type=module','-e',script,extension,json.dumps(requests),json.dumps(control)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode: reply(p.stderr.decode(),p.returncode)
        results=[json.loads(line) for line in p.stdout.decode().splitlines()]
    else:
        overrides={}
        for i,arg in enumerate(sys.argv[:-1]):
            if arg=='-c' and '=' in sys.argv[i+1]:
                key,value=sys.argv[i+1].split('=',1)
                overrides[key]=json.loads(value)
        server=subprocess.Popen([overrides['mcp_servers.cannbot.command'],*overrides['mcp_servers.cannbot.args']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        def rpc(method,params,index):
            server.stdin.write(json.dumps({'jsonrpc':'2.0','id':index,'method':method,'params':params})+'\n'); server.stdin.flush()
            if control.get('cancel_child') and method=='tools/call' and params['name']=='Task' and params['arguments']['prompt']=='evaluate':
                deadline=time.monotonic()+15
                while not Path(control['ready']).exists() and time.monotonic()<deadline: time.sleep(.02)
                server.stdin.write(json.dumps({'jsonrpc':'2.0','method':'notifications/cancelled','params':{'requestId':index,'reason':'original cancellation'}})+'\n'); server.stdin.flush()
            value=json.loads(server.stdout.readline())
            assert 'error' not in value, value
            return value['result']
        initialized=rpc('initialize',{'protocolVersion':'2024-11-05'},0)
        assert initialized['serverInfo']['name']=='cannbot'
        listed=rpc('tools/list',{},1)
        assert 'Task' in [tool['name'] for tool in listed['tools']]
        results=[{'request':request,'value':rpc('tools/call',request,i+2)} for i,request in enumerate(requests)]
        server.stdin.close(); server.wait(timeout=10)
    record({'type':'bridge_results','results':results})
    if control.get('native_error_main'):
        record({'type':'turn.failed','error':{'message':'Original native model failure'}})
        sys.exit(0)
    reply('Original main answer')
else:
    workspace=Path.cwd()
    candidate=workspace/'candidate'
    assert candidate.is_dir(), 'Candidate workspace link is unavailable in the Agent sandbox'
    probe=candidate/'.sandbox-write-probe.py'
    probe.write_text('pass\n')
    probe.unlink()
    if os.environ.get('PTOBENCH_SANDBOX') == '1':
        try:
            (workspace/'.sandbox-root-write-probe').write_text('must fail\n')
        except OSError:
            pass
        else:
            raise AssertionError('Agent workspace root is writable')
        for name in ('candidate', '.benchmark_context', '.benchmark_outputs'):
            path = workspace/name
            moved = workspace/(name + '.replacement-probe')
            try:
                path.rename(moved)
            except OSError:
                continue
            moved.rename(path)
            raise AssertionError('Agent replaced protected workspace entry: '+name)
        tuning = candidate/'tuning.json'
        tuning_contents = tuning.read_text()
        tuning.write_text(tuning_contents)
    if backend=='pi':
        task='profiles' if sys.argv[-1].strip().endswith('profiles') else 'evaluate'
    else:
        task=Path(config['run_root'],'method/dispatches',os.environ['CANNBOT_DISPATCH'],'task.txt').read_text()
        listed=subprocess.run([config['python'],os.environ['CANNBOT_TOOLS'],'list','--config',os.environ['CANNBOT_CONFIG'],'--dispatch',os.environ['CANNBOT_DISPATCH']],capture_output=True,text=True,check=True)
        assert 'Task' not in [tool['name'] for tool in json.loads(listed.stdout)]
    if task=='evaluate':
        (Path.cwd()/'candidate/helper.py').write_text('VALUE = 42\n')
        cli(['evaluate'])
        if os.environ.get('PTOBENCH_SANDBOX') == '1':
            result = Path.cwd()/'.benchmark_outputs/clean/latest_session_result.json'
            assert result.is_file(), 'Broker feedback did not appear in the live Agent sandbox'
            result_contents = result.read_text()
            for action in (lambda: result.write_text('{}\n'), lambda: result.unlink(),
                           lambda: result.rename(result.with_name('replacement.json'))):
                try:
                    action()
                except OSError:
                    continue
                raise AssertionError('Agent modified read-only result feedback')
            assert result.read_text() == result_contents
        if control.get('hang_child') or control.get('budget_child') or control.get('cancel_child'):
            if control.get('cancel_child'):
                if backend == 'pi':
                    heartbeat=native.with_suffix('.worker-heartbeat')
                    ready=Path.cwd()/'candidate/.fake-agent-control.py'
                    worker_code=('import pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                        'p=pathlib.Path('+repr(str(heartbeat))+'); p.write_text("started"); '
                        'pathlib.Path('+repr(str(ready))+').write_text("ready"); '
                        '\nwhile True: p.write_text(str(time.monotonic())); time.sleep(.05)')
                    worker=subprocess.Popen([sys.executable,'-c',worker_code])
                else:
                    worker=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
                    Path(control['worker_pid']).write_text(str(worker.pid))
            if backend != 'pi':
                Path(control['ready']).write_text('ready')
                Path(control['child_pid']).write_text(str(os.getpid()))
            while True: time.sleep(.02)
        if control.get('fail_child'): reply('Original failed stage answer',7)
        reply('Original evaluate stage answer')
    if task=='profiles':
        cli(['profile'])
        reply('Original profile stage answer')
    reply('Unknown test Task',3)
'''


class CANNBotTests(ActionTestBase):
    launcher_config = launcher_support.LauncherTests.launcher_config
    launch = launcher_support.LauncherTests.launch
    result = launcher_support.LauncherTests.result
    cleanup_launcher = launcher_support.LauncherTests.cleanup_launcher
    assert_final = launcher_support.LauncherTests.assert_final

    def setUp(self):
        super().setUp()
        self.python = os.path.abspath(sys.executable)
        shutil.rmtree(self.bench / 'launcher')
        shutil.copytree(SOURCE_BENCHMARK / 'launcher', self.bench / 'launcher', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(SOURCE_BENCHMARK / 'pto-launcher', self.bench / 'pto-launcher')
        for name in ('ptobench', 'pto-launcher'):
            (self.bench / name).chmod(0o755)
        self.tool = self.repo / 'fake-tool'
        self.tool.write_text('#!' + self.python + '\n' + FAKE_NATIVE)
        self.tool.chmod(0o755)
        self.config = self.repo / 'experiment.yaml'
        self.launcher_control = self.repo / 'optimizer.json'
        self.seen, self.ready, self.go = (self.repo / name for name in ('seen.json', 'ready', 'go'))
        self.env.update(S5_CONTROL=str(self.launcher_control), CODEX_HOME=str(self.repo / 'codex_home'),
                        PI_CODING_AGENT_DIR=str(self.repo / 'pi_home'))
        self.env['PATH'] = str(Path(self.python).parent) + os.pathsep + self.env['PATH']
        self.node = shutil.which('node')

    def cannbot_config(self, backend='codex', budget=6, **options):
        import yaml
        self.launcher_config(backend, budget, experiment_mode='cannbot', node=self.node, **options)
        if backend == 'pi':
            self.env['PTOBENCH_SANDBOX'] = '1'
            control = json.loads(self.launcher_control.read_text())
            for key in ('seen', 'ready', 'go', 'child_pid', 'worker_pid'):
                control.pop(key, None)
            profile_path = self.bench / 'modes/_shared/profiles/pi/default.yaml'
            profile = yaml.safe_load(profile_path.read_text())
            control = json.dumps(control, separators=(',', ':'))
            position = profile['argv'].index('{prompt}')
            profile['argv'][position:position] = ['--test-control-json', control]
            profile_path.write_text(yaml.safe_dump(profile))

    def dispatches(self, root):
        return [json.loads(path.read_text()) for path in (root / 'method/dispatches').glob('*/dispatch.json')]

    def test_agent_request_cleanup_waits_for_cannbot_child_exit(self):
        cleanup = getattr(cann_bot_runtime, '_cleanup_agent_request', None)
        self.assertTrue(callable(cleanup), 'CANNBot Agent request cleanup is missing')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'agent' / 'launcher'
            directory.mkdir(parents=True, mode=0o2710)
            directory.parent.chmod(0o2710)
            directory.chmod(0o2710)
            request = directory / ('34' * 16 + '.json')
            request.write_text('{}\n')
            request.chmod(0o640)
            environment = {agent_launch.REQUEST_ENV: str(request)}
            with patch.object(agent_launch.grp, 'getgrnam',
                              return_value=SimpleNamespace(gr_gid=os.getgid())):
                cleanup(environment, SimpleNamespace(poll=lambda: None))
                self.assertTrue(request.exists())
                cleanup(environment, SimpleNamespace(poll=lambda: 0))
            self.assertFalse(request.exists())

    def test_pi_dispatch_fails_closed_when_sandbox_is_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_root, workspace, upstream = (root / name for name in ('session', 'workspace', 'upstream'))
            session_root.mkdir()
            workspace.mkdir()
            upstream.mkdir()
            role = upstream / 'role.md'
            role.write_text('role')
            profile = {'backend': 'pi', 'argv': ['pi', '{prompt}'], 'env': {},
                       'mode_provider': {'base_url': 'https://example.invalid'}}
            config = {'profile': profile, 'run_root': str(session_root), 'workspace': str(workspace),
                      'upstream': str(upstream), 'roles': {'pypto-op-orchestrator': 'role.md'},
                      'skills': {}, 'python': sys.executable, 'agent_extension': None,
                      'hardware_budget': 1}
            with patch.dict(os.environ, {'PTOBENCH_SANDBOX': '0'}), \
                 patch.object(cann_bot_runtime.subprocess, 'Popen',
                              side_effect=AssertionError('unisolated Pi process was started')) as spawn:
                with self.assertRaisesRegex(RuntimeError, 'Pi Agent requires filesystem isolation'):
                    cann_bot_runtime.run_dispatch(
                        config, 'pypto-op-orchestrator', 'task', parent=None, tool_call_id=None)
            spawn.assert_not_called()

    def test_codex_mcp_fresh_sessions_same_budget_and_original_resources(self):
        self.check_native('codex')

    def test_pi_extension_fresh_sessions_same_budget_and_original_resources(self):
        if not self.node or int(subprocess.check_output([self.node, '--version']).decode().split('.')[0][1:]) < 22:
            self.skipTest('Pi TypeScript bridge smoke requires Node 22+')
        self.check_native('pi')

    def check_native(self, backend):
        self.cannbot_config(backend)
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        root, _ = self.assert_final(value, 2)
        self.assertEqual(session.read_session(root)['mode']['id'], 'cannbot')
        lock = json.loads((root / 'method/upstream.json').read_text())
        for relative, digest in lock['files'].items():
            self.assertEqual(hashlib.sha256((root / 'method/resources/upstream' / relative).read_bytes()).hexdigest(), digest)
        self.assertNotIn('First run ptobench evaluate', (root / 'prompt.txt').read_text())
        protection=json.loads((root/'protection/manifest.json').read_text())
        self.assertNotIn('raw_wiki', protection['roots']['participant_cannbot_resources']['entries'])
        self.assertFalse((root/'method/knowledge.json').exists())
        self.assertFalse((root/'method/README.md').exists())
        self.assertFalse((root/'method/resources/raw_wiki').exists())
        states = self.dispatches(root)
        self.assertEqual(len(states), 3)
        parent = next(state for state in states if state['parent_dispatch_id'] is None)
        dispatch_by_id = {state['id']: state for state in states}
        for state in states:
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['session_root'], str(root))
            self.assertEqual(state['process']['pgid'], parent['process']['pgid'])
            self.assertEqual(state['profile'], parent['profile'])
            original = Path(state['role_path']).read_text()
            self.assertTrue((root / 'method/dispatches' / state['id'] / 'system.md').read_text().startswith(original))
            prompt=(root/'method/dispatches'/state['id']/'prompt.txt').read_text()
            self.assertIn('.benchmark_context/README.md', prompt)
            self.assertIn('.benchmark_context/tuning_capabilities.json', prompt)
            self.assertIn('complete defaults vary by Case', prompt)
            self.assertIn('PYPTO + false', prompt)
            self.assertIn('ptobench evaluate .', prompt)
            self.assertNotIn(str(root/'method/README.md'), prompt)
            self.assertNotIn('5 + 10', prompt)
            if state is not parent:
                self.assertEqual(state['parent_dispatch_id'], parent['id'])
                self.assertIsNotNone(state['tool_call_id'])
                self.assertIn((root / 'method/dispatches' / state['id'] / 'task.txt').read_text(), ('evaluate', 'profiles'))
            self.assertIsNone(session.process_identity(state['process']['pid']))
        events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
        linked = [event for event in events if event['action'] in {'evaluate', 'profile'}]
        self.assertTrue(linked)
        for event in linked:
            dispatch = dispatch_by_id[event['agent_dispatch_id']]
            self.assertEqual(dispatch['session_root'], str(root))
            if dispatch['tool_call_id'] is None:
                self.assertEqual(event['agent_parent_tool_call_id_status'], 'unavailable')
            else:
                self.assertEqual(event['agent_parent_tool_call_id'], dispatch['tool_call_id'])
                self.assertEqual(event['agent_parent_tool_call_id_status'], 'available')
        archive = json.loads((root / 'agent/session.json').read_text())
        self.assertEqual(archive['status'], 'recorded')
        self.assertEqual(len(set(archive['native_session_ids'])), 3)
        for row in archive['dispatches']:
            self.assertTrue((root / row['path'] / row['recording']['files'][0]['path']).is_file())
            self.assertTrue((root / row['path'] / 'agent/events.jsonl').is_file())
        logs = (root / 'method/dispatches' / parent['id'] / 'stdout.log').read_text()
        self.assertIn('Original evaluate stage answer', logs)
        self.assertIn('Original profile stage answer', logs)
        self.assertEqual(json.loads((root / 'method/work/todos.json').read_text())[0]['content'], 'opaque upstream todo')
        # Isolated CLI children intentionally receive a scrubbed environment;
        # assert the public event records instead of relying on the test-only
        # S3_COUNTER variable being forwarded into that boundary.
        measured = [(event['action'], event['result'].get('profiling_mode'))
                    for event in events
                    if event['action'] in {'evaluate', 'profile', 'finish'}]
        self.assertEqual(measured, [('evaluate', 'clean'), ('profile', 'combined'), ('finish', 'clean')])
        self.assertEqual(len(list((root / 'method/tools').glob('*/input.json'))), 5)
        self.assertEqual(len(list((root / 'method/tools').glob('*/result.json'))), 5)

    def test_task_can_use_public_fork_without_replacing_shared_run(self):
        self.cannbot_config(fork_child=True)
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        root, _ = self.assert_final(value, 2)
        self.assertIn('candidate/tuning.json', (root / 'prompt.txt').read_text())
        states = self.dispatches(root)
        self.assertEqual(len({state['workspace'] for state in states}), 2)
        self.assertEqual({state['session_root'] for state in states}, {str(root)})

    def test_failed_stage_returns_original_error_and_retains_native_history(self):
        self.cannbot_config(fail_child=True)
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        root, _ = self.assert_final(value, 2)
        failed = next(state for state in self.dispatches(root) if state['status'] == 'failed')
        result = json.loads((root / 'method/dispatches' / failed['id'] / 'result.json').read_text())
        self.assertEqual(result['answer'], 'Original failed stage answer')
        self.assertEqual(result['exit_code'], 7)
        self.assertTrue(result['is_error'])
        tool = next(json.loads(path.read_text()) for path in (root / 'method/tools').glob('*/result.json') if json.loads(path.read_text())['isError'])
        self.assertIn('Original failed stage answer', tool['content'][0]['text'])
        self.assertEqual(json.loads((root / 'agent/session.json').read_text())['status'], 'recorded')

    def test_missing_child_history_is_not_hidden_by_main_history(self):
        self.cannbot_config(missing_child=True)
        process, value = self.launch()
        self.assertNotEqual(process.returncode, 0, value)
        root, _ = self.assert_final(value, 2)
        self.assertEqual(json.loads((root / 'agent/session.json').read_text())['status'], 'missing')
        self.assertEqual(value['error_code'], 'recording_incomplete')

    def test_native_error_is_not_reported_as_success_even_with_zero_exit(self):
        self.cannbot_config(native_error_main=True)
        process, value = self.launch()
        # The common finish can still score a verified historical candidate.
        self.assertEqual(process.returncode, 0, value)
        self.assertEqual(value['optimizer_exit_code'], 1)
        self.assertEqual(value['close_reason'], 'optimizer_nonzero_exit')
        root, _ = self.assert_final(value, 2)
        main = next(state for state in self.dispatches(root) if state['parent_dispatch_id'] is None)
        result = json.loads((root / 'method/dispatches' / main['id'] / 'result.json').read_text())
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['error'], {'message': 'Original native model failure'})
        self.assertTrue(result['is_error'])

    def test_interrupt_stops_child_and_retains_pending_task_input(self):
        child = self.repo / 'child.pid'
        self.cannbot_config(hang_child=True, child_pid=str(child))
        process = self.launch(True)
        self.wait_file(self.ready, process)
        self.wait_file(child, process)
        process.send_signal(signal.SIGINT)
        value, _, error = self.result(process)
        self.assertEqual(process.returncode, 130, (value, error.decode()))
        root, _ = self.assert_final(value, 1)
        self.assertIsNone(session.process_identity(int(child.read_text())))
        self.assertTrue(any(state['status'] == 'interrupted' for state in self.dispatches(root)))
        self.assertTrue(list((root / 'method/tools').glob('*/interrupted.json')))
        self.assertEqual(json.loads((root / 'agent/session.json').read_text())['status'], 'recorded')

    def test_budget_exhaustion_stops_active_stage_without_new_budget(self):
        child = self.repo / 'child.pid'
        self.cannbot_config(budget=1, budget_child=True, child_pid=str(child))
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        root, _ = self.assert_final(value, 1)
        self.assertEqual(session.read_session(root)['close_reason'], 'budget_exhausted')
        for state in self.dispatches(root):
            if state.get('process'):
                self.assertIsNone(session.process_identity(state['process']['pid']))
        self.assertTrue(any(state['status'] == 'interrupted' for state in self.dispatches(root)))

    def test_task_cancellation_propagates_in_both_transports_and_next_stage_runs(self):
        for backend in ('codex', 'pi'):
            with self.subTest(backend=backend):
                if backend == 'pi' and (not self.node or int(subprocess.check_output([self.node, '--version']).decode().split('.')[0][1:]) < 22):
                    continue
                self.ready.unlink(missing_ok=True)
                child, worker = self.repo / 'child.pid', self.repo / 'worker.pid'
                self.cannbot_config(backend, cancel_child=True, child_pid=str(child), worker_pid=str(worker))
                process, value = self.launch()
                self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
                root, _ = self.assert_final(value, 2)
                cancelled = next(state for state in self.dispatches(root) if state['status'] == 'cancelled')
                if backend == 'pi':
                    heartbeat, = (root / 'method/dispatches' / cancelled['id'] / 'agent/native').glob(
                        '*.worker-heartbeat')
                    last_value, last_update = heartbeat.read_text(), heartbeat.stat().st_mtime_ns
                    time.sleep(.15)
                    self.assertEqual(heartbeat.read_text(), last_value)
                    self.assertEqual(heartbeat.stat().st_mtime_ns, last_update)
                else:
                    child_pid, worker_pid = int(child.read_text()), int(worker.read_text())
                result = json.loads((root / 'method/dispatches' / cancelled['id'] / 'result.json').read_text())
                self.assertTrue(result['cancelled'])
                self.assertTrue(result['is_error'])
                self.assertIsNone(session.process_identity(cancelled['process']['pid']))
                if backend == 'codex':
                    self.assertIsNone(session.process_identity(child_pid))
                    self.assertIsNone(session.process_identity(worker_pid))
                self.assertEqual(len(self.dispatches(root)), 3)


if __name__ == '__main__':
    unittest.main()
