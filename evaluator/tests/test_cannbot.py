"""Native tool transports and shared-budget integration; no LLM or NPU calls."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import unittest

from action_test_support import ActionTestBase, SOURCE_BENCHMARK
import session
import test_launcher as launcher_support


FAKE_NATIVE = r'''
import json, os, signal, subprocess, sys, time, uuid
from pathlib import Path
if '--version' in sys.argv:
    print('fake-cannbot-native 1.0'); sys.exit()
if '--help' in sys.argv:
    print('--skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --print --json --model --provider --thinking --mode --session-dir'); sys.exit()
config=json.loads(Path(os.environ['CANNBOT_CONFIG']).read_text())
control=json.loads(Path(os.environ['S5_CONTROL']).read_text())
root=Path(config['run_root'])
references=json.loads((root/'method/knowledge.json').read_text())
assert (root/references['snapshot']/'README.md').read_text()
assert str(root/'method/README.md') in Path(root/'method/dispatches',os.environ['CANNBOT_DISPATCH'],'prompt.txt').read_text()
dispatch=os.environ['CANNBOT_DISPATCH']
primary=os.environ['CANNBOT_ALLOW_TASK']=='1'
backend=config['profile']['backend']
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
const {default: extension} = await import(pathToFileURL(process.argv[1]).href);
const registered = new Map();
extension({registerTool: tool => registered.set(tool.name,tool)});
const requests=JSON.parse(process.argv[2]);
const control=JSON.parse(process.argv[3]);
for (let i=0;i<requests.length;i++) {
 const request=requests[i];
 const controller=new AbortController();
 let interval;
 if (control.cancel_child && request.name==='Task' && request.arguments.prompt==='evaluate') {
   interval=setInterval(()=>{if(existsSync(control.ready)) controller.abort();},30);
 }
 const value=await registered.get(request.name).execute(String(i+1),request.arguments,controller.signal);
 if (interval) clearInterval(interval);
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
    task=Path(root/'method/dispatches'/dispatch/'task.txt').read_text()
    listed=subprocess.run([config['python'],os.environ['CANNBOT_TOOLS'],'list','--config',os.environ['CANNBOT_CONFIG'],'--dispatch',dispatch],capture_output=True,text=True,check=True)
    assert 'Task' not in [tool['name'] for tool in json.loads(listed.stdout)]
    if task=='evaluate':
        (Path.cwd()/'helper.py').write_text('VALUE = 42\n')
        cli(['evaluate'])
        if control.get('hang_child') or control.get('budget_child') or control.get('cancel_child'):
            if control.get('cancel_child'):
                worker=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
                Path(control['worker_pid']).write_text(str(worker.pid))
            Path(control['ready']).write_text('ready')
            Path(control['child_pid']).write_text(str(os.getpid()))
            while True: time.sleep(.02)
        if control.get('fail_child'): reply('Original failed stage answer',7)
        reply('Original evaluate stage answer')
    if task=='profiles':
        for mode in ('runtime','pmu','compiler'): cli(['profile',mode])
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
        self.python = str(Path(sys.executable).absolute())
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
        self.launcher_config(backend, budget, experiment_mode='cannbot', node=self.node, **options)

    def dispatches(self, root):
        return [json.loads(path.read_text()) for path in (root / 'method/dispatches').glob('*/dispatch.json')]

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
        root, _ = self.assert_final(value, 4)
        self.assertEqual(session.read_session(root)['mode']['id'], 'cannbot')
        lock = json.loads((root / 'method/upstream.json').read_text())
        for relative, digest in lock['files'].items():
            self.assertEqual(hashlib.sha256((root / 'method/resources/upstream' / relative).read_bytes()).hexdigest(), digest)
        self.assertNotIn('First run ptobench evaluate', (root / 'prompt.txt').read_text())
        references=json.loads((root/'method/knowledge.json').read_text())
        knowledge_root=root/references['snapshot']
        self.assertEqual(references['content_sha256'], json.loads(
            (self.bench/'modes/cannbot/knowledge/knowledge.lock.json').read_text())['content_sha256'])
        self.assertEqual(knowledge_root.stat().st_mode & 0o222, 0)
        protection=json.loads((root/'protection/manifest.json').read_text())
        self.assertIn('raw_wiki/README.md', protection['roots']['participant_cannbot_resources']['entries'])
        states = self.dispatches(root)
        self.assertEqual(len(states), 3)
        parent = next(state for state in states if state['parent_dispatch_id'] is None)
        for state in states:
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['session_root'], str(root))
            self.assertEqual(state['process']['pgid'], parent['process']['pgid'])
            self.assertEqual(state['profile'], parent['profile'])
            original = Path(state['role_path']).read_text()
            self.assertTrue((root / 'method/dispatches' / state['id'] / 'system.md').read_text().startswith(original))
            prompt=(root/'method/dispatches'/state['id']/'prompt.txt').read_text()
            self.assertIn('.benchmark_context/README.md', prompt)
            self.assertIn(str(root/'method/README.md'), prompt)
            self.assertNotIn('5 + 10', prompt)
            if state is not parent:
                self.assertEqual(state['parent_dispatch_id'], parent['id'])
                self.assertIsNotNone(state['tool_call_id'])
                self.assertIn((root / 'method/dispatches' / state['id'] / 'task.txt').read_text(), ('evaluate', 'profiles'))
            self.assertIsNone(session.process_identity(state['process']['pid']))
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
        calls = [json.loads(line) for line in self.counter.read_text().splitlines()]
        # One requested evaluation + three profiles + common final; no extra baseline.
        self.assertEqual([call['profile_mode'] for call in calls], ['clean', 'runtime', 'hardware', 'compile', 'clean'])
        self.assertEqual(len(list((root / 'method/tools').glob('*/input.json'))), 5)
        self.assertEqual(len(list((root / 'method/tools').glob('*/result.json'))), 5)

    def test_task_can_use_public_fork_without_replacing_shared_run(self):
        self.cannbot_config(fork_child=True)
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        root, _ = self.assert_final(value, 4)
        states = self.dispatches(root)
        self.assertEqual(len({state['workspace'] for state in states}), 2)
        self.assertEqual({state['session_root'] for state in states}, {str(root)})

    def test_failed_stage_returns_original_error_and_retains_native_history(self):
        self.cannbot_config(fail_child=True)
        process, value = self.launch()
        self.assertEqual(process.returncode, 0, (value, process.stderr.decode()))
        root, _ = self.assert_final(value, 4)
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
        root, _ = self.assert_final(value, 4)
        self.assertEqual(json.loads((root / 'agent/session.json').read_text())['status'], 'missing')
        self.assertEqual(value['error_code'], 'recording_incomplete')

    def test_native_error_is_not_reported_as_success_even_with_zero_exit(self):
        self.cannbot_config(native_error_main=True)
        process, value = self.launch()
        # The common finish can still score a verified historical candidate.
        self.assertEqual(process.returncode, 0, value)
        self.assertEqual(value['optimizer_exit_code'], 1)
        self.assertEqual(value['close_reason'], 'optimizer_nonzero_exit')
        root, _ = self.assert_final(value, 4)
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
                root, _ = self.assert_final(value, 4)
                cancelled = next(state for state in self.dispatches(root) if state['status'] == 'cancelled')
                result = json.loads((root / 'method/dispatches' / cancelled['id'] / 'result.json').read_text())
                self.assertTrue(result['cancelled'])
                self.assertTrue(result['is_error'])
                self.assertIsNone(session.process_identity(int(child.read_text())))
                self.assertIsNone(session.process_identity(int(worker.read_text())))
                self.assertEqual(len(self.dispatches(root)), 3)


if __name__ == '__main__':
    unittest.main()
