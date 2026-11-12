"""Native tool transports and shared-budget integration; no LLM or NPU calls."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import grp
import stat
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from action_test_support import ActionTestBase, SOURCE_BENCHMARK
from session_test_support import SessionTestBase
import agent_launch
import session
import test_launcher as launcher_support
from modes.cannbot import runtime as cann_bot_runtime
from modes.cannbot import resources as cann_bot_resources
from modes.cannbot import tools as cann_bot_tools


FAKE_NATIVE = r'''
import json, os, signal, subprocess, sys, time, uuid, shutil, stat
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
 let interval, fallback, cancelledByMarker=false;
 if (control.cancel_child && request.name==='Task' && request.arguments.prompt==='evaluate') {
   fallback=setTimeout(()=>controller.abort(),10000);
   interval=setInterval(()=>{
     if (existsSync(join(process.cwd(),'candidate','.fake-agent-control.py'))) {
       cancelledByMarker=true;
       clearTimeout(fallback);
       controller.abort();
     }
   },20);
 }
 const value=await registered.get(request.name).execute(String(i+1),request.arguments,controller.signal);
 if (interval) clearInterval(interval);
 if (fallback) clearTimeout(fallback);
 if (control.cancel_child && request.name==='Task' && request.arguments.prompt==='evaluate' && !cancelledByMarker)
   throw new Error('Task cancellation fallback expired before the shared Workspace marker appeared');
 console.log(JSON.stringify({request,value}));
}
"""
        extension=sys.argv[sys.argv.index('--extension')+1]
        node=shutil.which('node')
        if not node:
            raise AssertionError('Pi bridge cannot resolve Node: PATH=' + repr(os.environ.get('PATH')) +
                                 ', mounted=' + repr(Path('/ptobench/runtime/node').exists()))
        p=subprocess.run([node,'--experimental-strip-types','--input-type=module','-e',script,extension,json.dumps(requests),json.dumps(control)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
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
        tuning = candidate/'tuning.json'
        tuning_contents = tuning.read_text()
        tuning.write_text(tuning_contents)
    if backend=='pi':
        task='profiles' if any(argument.strip().endswith('profiles') for argument in sys.argv) else 'evaluate'
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
            assert stat.S_IMODE(result.stat().st_mode) == 0o640, 'Broker feedback DAC mode changed'
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


class FrozenResourceTests(unittest.TestCase):
    def test_cannbot_extension_is_prepared_in_pi_tools(self):
        from modes.cannbot import runner as cann_bot_runner

        with tempfile.TemporaryDirectory(prefix='cannbot-extension-location-') as temporary:
            root = Path(temporary) / 'session'
            workspace = root / 'workspaces/worker_000'
            workspace.mkdir(parents=True)
            config_directory = root / 'agent/pi-config'
            config_directory.mkdir(parents=True)
            profile = {'backend': 'pi', 'argv': ['pi', '{prompt}'], 'env': {
                'PI_CODING_AGENT_DIR': str(config_directory)}, 'sha256': 'fixture-profile'}
            context = {'session_root': str(root), 'workspace': str(workspace),
                       'case': {'case_id': 'fixture.case', 'module': 'fixture', 'entry': 'run'},
                       'budget': {'max_hardware_actions': 1}}
            request = {'context': context,
                       'preflight': {'profile': profile, 'backend': {'backend': 'pi'}},
                       'experiment': {'mode': 'cannbot', 'case': 'fixture.case',
                                      'optimizer': {'id': 'fixture'}}}
            with patch.object(cann_bot_runner, 'provider') as provider, \
                 patch.object(cann_bot_runner.provenance, 'freeze_agent', return_value={}), \
                 patch('modes._shared.config.render', return_value=([], {})):
                provider.prepare.side_effect = lambda value, *_args, **_kwargs: value
                result = cann_bot_runner.prepare(request)

            extension = root / 'agent/pi-tools/cannbot-extension.ts'
            self.assertTrue(extension.is_file())
            self.assertEqual(extension.stat().st_gid, grp.getgrnam('ptobench-share').gr_gid)
            self.assertEqual(stat.S_IMODE(extension.stat().st_mode), 0o440)
            self.assertFalse((config_directory / 'cannbot-tools.ts').exists())
            self.assertEqual(result['records']['method/config.json']['agent_extension'], str(extension))

    def test_tool_socket_uses_shared_group_without_listing_parent(self):
        with tempfile.TemporaryDirectory(prefix='cannbot-tool-socket-') as temporary:
            path = Path(temporary) / 'tools.sock'
            server = cann_bot_tools.ToolServer({}, 'dispatch', path, False)
            try:
                endpoint = path.stat()
                directory = path.parent.stat()
                self.assertEqual(endpoint.st_uid, os.getuid())
                self.assertEqual(endpoint.st_gid, grp.getgrnam('ptobench-share').gr_gid)
                self.assertEqual(stat.S_IMODE(endpoint.st_mode), 0o660)
                self.assertEqual(directory.st_gid, grp.getgrnam('ptobench-share').gr_gid)
                self.assertEqual(stat.S_IMODE(directory.st_mode), 0o710)
            finally:
                server.close()

class TaskWorkspaceAuthorizationTests(SessionTestBase):
    def setUp(self):
        super().setUp()
        self.handle = self.create()
        self.config = {
            'run_root': str(self.handle.session_root),
            'workspace': str(self.handle.initial_workspace),
            'profile': {'backend': 'pi'},
            'work_dir': str(self.handle.session_root / 'method/work'),
        }

    def _call_task(self, workspace=None):
        arguments = {'subagent_type': 'pypto-op-optimizer', 'prompt': 'child task'}
        if workspace is not None:
            arguments['workspace'] = str(workspace)
        return cann_bot_tools.call(
            self.config, 'parent-dispatch', 'Task', arguments, 'task-call', True)

    def test_rejects_unregistered_task_workspaces_before_metadata_reads_or_dispatch(self):
        other_session = self.create(optimizer='other')
        self.make_case('test.other')
        other_case = self.create(case_id='test.other', optimizer='other-case')

        unregistered = self.handle.session_root / 'workspaces/unregistered'
        unregistered.mkdir()
        (unregistered / '.benchmark_context').mkdir()
        (unregistered / '.benchmark_context/workspace.json').write_text(json.dumps({
            'case_id': self.case_id, 'workspace': str(unregistered), 'parent_workspace': None,
        }))

        external = Path(self.temporary.name) / 'external-workspace'
        (external / '.benchmark_context').mkdir(parents=True)
        (external / '.benchmark_context/workspace.json').write_text(json.dumps({
            'case_id': self.case_id, 'workspace': str(external), 'parent_workspace': None,
        }))
        external_link = self.handle.session_root / 'workspaces/external-link'
        external_link.symlink_to(external, target_is_directory=True)

        targets = {
            'other Session': other_session.initial_workspace,
            'different Case': other_case.initial_workspace,
            'unregistered root': unregistered,
            'external symlink': external_link,
        }
        read_metadata = Path.read_text
        for label, target in targets.items():
            with self.subTest(target=label):
                reads = []

                def tracked_read(path, *args, **kwargs):
                    if path.name == 'workspace.json' and path.parent.name == '.benchmark_context':
                        reads.append(path)
                    return read_metadata(path, *args, **kwargs)

                with patch.object(Path, 'read_text', tracked_read), \
                     patch.object(cann_bot_tools, 'run_dispatch', return_value={
                         'answer': 'unexpected child answer', 'is_error': False,
                     }) as dispatch:
                    response = self._call_task(target)

                self.assertTrue(response['isError'])
                self.assertEqual(reads, [])
                dispatch.assert_not_called()

    def test_rejects_registry_case_mismatch_before_metadata_reads_or_dispatch(self):
        self.make_case('test.other')
        other_case = self.create(case_id='test.other', optimizer='other-case')
        resolve = session.resolve_workspace
        reads = []

        def mismatched_resolve(session_root, value):
            if Path(value) == other_case.initial_workspace:
                return 'worker_001', other_case.initial_workspace, 'test.other'
            return resolve(session_root, value)

        read_metadata = Path.read_text

        def tracked_read(path, *args, **kwargs):
            if path.name == 'workspace.json' and path.parent.name == '.benchmark_context':
                reads.append(path)
            return read_metadata(path, *args, **kwargs)

        with patch.object(session, 'resolve_workspace', mismatched_resolve), \
             patch.object(Path, 'read_text', tracked_read), \
             patch.object(cann_bot_tools, 'run_dispatch', return_value={
                 'answer': 'unexpected child answer', 'is_error': False,
             }) as dispatch:
            response = self._call_task(other_case.initial_workspace)

        self.assertTrue(response['isError'])
        self.assertEqual(reads, [])
        dispatch.assert_not_called()

    def test_omitted_workspace_is_resolved_before_dispatch(self):
        events = []
        resolve = session.resolve_workspace

        def tracked_resolve(session_root, value):
            events.append(('resolve', Path(value)))
            return resolve(session_root, value)

        read_metadata = Path.read_text

        def tracked_read(path, *args, **kwargs):
            if path.name == 'workspace.json' and path.parent.name == '.benchmark_context':
                events.append(('metadata', path))
            return read_metadata(path, *args, **kwargs)

        with patch.object(session, 'resolve_workspace', tracked_resolve), \
             patch.object(Path, 'read_text', tracked_read), \
             patch.object(cann_bot_tools, 'run_dispatch', return_value={
                 'answer': 'child answer', 'is_error': False,
             }) as dispatch:
            response = self._call_task()

        self.assertFalse(response['isError'])
        self.assertEqual(events[0], ('resolve', self.handle.initial_workspace))
        metadata_indexes = [index for index, (kind, _) in enumerate(events) if kind == 'metadata']
        resolve_indexes = [index for index, (kind, _) in enumerate(events) if kind == 'resolve']
        self.assertTrue(metadata_indexes)
        self.assertLess(max(resolve_indexes), min(metadata_indexes))
        self.assertEqual(dispatch.call_args.args[0]['workspace'], str(self.handle.initial_workspace))

    def test_explicit_registered_worker_is_resolved_before_dispatch(self):
        worker_id = session.fork_workspace(
            self.handle.session_root, self.handle.initial_workspace)
        worker = self.handle.session_root / 'workspaces' / worker_id
        resolve = session.resolve_workspace
        events = []

        def tracked_resolve(session_root, value):
            events.append(Path(value))
            return resolve(session_root, value)

        with patch.object(session, 'resolve_workspace', tracked_resolve), \
             patch.object(cann_bot_tools, 'run_dispatch', return_value={
                 'answer': 'child answer', 'is_error': False,
             }) as dispatch:
            response = self._call_task(worker)

        self.assertFalse(response['isError'])
        self.assertEqual(events, [worker, self.handle.initial_workspace])
        self.assertEqual(dispatch.call_args.args[0]['workspace'], str(worker))

    def test_workspace_metadata_case_must_match_registry_before_dispatch(self):
        metadata = self.handle.initial_workspace / '.benchmark_context/workspace.json'
        binding = json.loads(metadata.read_text())
        binding['case_id'] = 'other.case'
        metadata.chmod(0o600)
        metadata.write_text(json.dumps(binding))

        with patch.object(cann_bot_tools, 'run_dispatch', return_value={
            'answer': 'unexpected child answer', 'is_error': False,
        }) as dispatch:
            response = self._call_task()

        self.assertTrue(response['isError'])
        dispatch.assert_not_called()


class LiveTaskChildWorkspacePublicationTests(ActionTestBase):
    def test_pi_task_child_observes_worker_published_by_broker(self):
        import sandbox_supervisor

        root = self.handle.session_root
        workspace = self.handle.initial_workspace
        agent_uid, share_gid = agent_launch._agent_identity()
        fake_package = self.repo / 'fake-node-modules/@earendil-works/pi-coding-agent'
        bundle = fake_package / 'dist/bundle'
        bundle.mkdir(parents=True)
        fake_pi = bundle / 'cli.js'
        fake_pi.write_text(r'''#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const { execFileSync } = require('node:child_process');
const args = process.argv.slice(2);
const sessionIndex = args.indexOf('--session-dir');
if (sessionIndex < 0) throw new Error('Pi did not receive its native Session directory');
for (const extension of ['/ptobench/pi-tools/ptobench-file-guard.mjs',
                         '/ptobench/pi-tools/cannbot-extension.ts']) {
  const index = args.indexOf('--extension');
  if (index < 0 || !args.includes(extension)) throw new Error('Pi is missing trusted extension ' + extension);
}
if (process.getuid() !== Number('__AGENT_UID__')) throw new Error('Pi child has the wrong UID');
const status = fs.readFileSync('/proc/self/status', 'utf8');
if (!status.includes('CapEff:\t0000000000000000')) throw new Error('Pi has effective capabilities');
const identity = require('node:crypto').randomUUID();
const native = path.join(args[sessionIndex + 1], identity + '.jsonl');
fs.mkdirSync(path.dirname(native), {recursive:true});
const header = {type:'session', id:identity, cwd:process.cwd()};
fs.writeFileSync(native, JSON.stringify(header) + '\n');
function answer(text) {
  const event = {type:'message_end', message:{role:'assistant', content:[{type:'text', text}], stopReason:'stop', usage:{input:1, output:1}}};
  fs.appendFileSync(native, JSON.stringify(event) + '\n');
  process.stdout.write(JSON.stringify(event) + '\n');
}
function tool(request) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection(process.env.PTOBENCH_CANNBOT_SOCKET);
    let response = '';
    socket.once('connect', () => socket.write(JSON.stringify(request) + '\n'));
    socket.on('data', chunk => {
      response += chunk.toString('utf8');
      const index = response.indexOf('\n');
      if (index < 0) return;
      socket.end();
      try { resolve(JSON.parse(response.slice(0, index))); }
      catch (error) { reject(error); }
    });
    socket.once('error', reject);
  });
}
function assertDenied(action) {
  try { action(); }
  catch (error) {
    if (['EACCES', 'EPERM', 'EROFS'].includes(error.code)) return;
    throw error;
  }
  throw new Error('protected Workspace entry was writable');
}
(async () => {
  const listed = await tool({action:'list'});
  const names = listed.tools.map(item => item.name);
  const primary = process.env.PTOBENCH_CANNBOT_ALLOW_TASK === '1';
  if (names.includes('Task') !== primary) throw new Error('Task availability does not match Pi role');
  if (primary) {
    const result = await tool({action:'call', tool_call_id:'live-worker-task', name:'Task', arguments:{
      subagent_type:'pypto-op-optimizer', prompt:'publish one Worker and inspect it',
    }});
    if (result.isError) throw new Error(result.content[0].text);
    answer(result.content[0].text);
    return;
  }
  if (process.env.PTOBENCH_SANDBOX !== '1') throw new Error('Pi child is not sandboxed');
  if (process.env.PTOBENCH_API_PROXY_SOCKET !== '/ptobench/network/api.sock') {
    throw new Error('Pi child received the wrong Gateway socket path');
  }
  await new Promise((resolve, reject) => {
    const gateway = net.createConnection(process.env.PTOBENCH_API_PROXY_SOCKET);
    gateway.once('connect', () => { gateway.destroy(); resolve(); });
    gateway.once('error', reject);
  });
  const output = execFileSync('ptobench', ['fork'], {encoding:'utf8'});
  const workerPath = JSON.parse(output.trim().split('\n').at(-1)).workspace;
  const candidate = path.join(workerPath, 'candidate/candidate.py');
  if (!fs.statSync(workerPath).isDirectory()) throw new Error('published Worker is not visible');
  if (!fs.readFileSync(candidate, 'utf8').includes('from helper import VALUE')) throw new Error('Candidate was not copied');
  fs.writeFileSync(path.join(workerPath, 'candidate/live_child.py'), 'CHILD = True\\n');
  fs.readFileSync(path.join(workerPath, '.benchmark_context/config.py'), 'utf8');
  assertDenied(() => fs.writeFileSync(path.join(workerPath, '.benchmark_context/config.py'), 'changed'));
  assertDenied(() => fs.renameSync(workerPath, workerPath + '-replaced'));
  const shellCode = 'python3 -c "import pathlib; assert not pathlib.Path(\'/ptobench/pi-config/settings.json\').exists(); assert not pathlib.Path(\'/ptobench/pi-tools/cannbot-extension.ts\').exists(); assert not pathlib.Path(\'/ptobench/agent-sessions/session.jsonl\').exists()"';
  execFileSync('/ptobench/pi-tools/ptobench-shell', ['-c', shellCode], {encoding:'utf8'});
  answer('child published Worker ' + workerPath);
})().catch(error => { process.stderr.write(String(error.stack || error) + '\\n'); process.exit(1); });
'''.replace('__AGENT_UID__', str(agent_uid)))
        fake_pi.chmod(0o755)
        package_jiti = fake_package / 'node_modules/jiti'
        package_jiti.parent.mkdir(parents=True)
        real_pi = Path(shutil.which('pi')).resolve()
        package_jiti.symlink_to(real_pi.parents[2] / 'node_modules/jiti', target_is_directory=True)
        fake_bin = self.repo / 'fake-bin'
        fake_bin.mkdir()
        (fake_bin / 'pi').symlink_to(fake_pi)

        upstream = root / 'method/resources/upstream'
        upstream.mkdir(parents=True)
        (upstream / 'orchestrator.md').write_text('Original orchestrator role.\n')
        (upstream / 'optimizer.md').write_text('Original optimizer role.\n')
        (upstream / 'perf-tune.md').write_text('Frozen performance skill.\n')
        extension = root / 'agent/pi-tools/cannbot-extension.ts'
        extension.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cann_bot_resources.DIRECTORY / 'bridge/pi_task.ts', extension)
        os.chown(extension, -1, share_gid)
        extension.chmod(0o440)
        profile = {
            'backend':'pi', 'argv':['pi', '{prompt}'], 'env':{}, 'sha256':'test-profile',
            'mode_provider':{'base_url':'http://127.0.0.1:18081/v1', 'api':'openai-responses'},
        }
        config = {
            'profile':profile, 'run_root':str(root), 'workspace':str(workspace),
            'upstream':str(upstream),
            'roles':{'pypto-op-orchestrator':'orchestrator.md', 'pypto-op-optimizer':'optimizer.md'},
            'skills':{'pypto-op-perf-tune':str(upstream / 'perf-tune.md')},
            'python':sys.executable, 'agent_extension':str(extension),
            'work_dir':str(root / 'method/work'), 'hardware_budget':1,
        }

        private = self.repo / 'broker-private'
        private.mkdir()
        broker = sandbox_supervisor.Broker(root, private)
        broker.cli_environment = cann_bot_runtime.isolation.clean_environment({
            'PATH':os.environ['PATH'], 'PTOBENCH_SESSION_ROOT':str(root), 'PTOBENCH_SANDBOX':'1',
        })
        broker_thread = threading.Thread(target=broker.serve, daemon=True)
        broker_thread.start()
        gateway_directory = root / 'broker-endpoints'
        gateway_directory.mkdir(parents=True, exist_ok=True)
        gateway_path = gateway_directory / 'api.sock'
        gateway = socket.socket(socket.AF_UNIX)
        gateway.bind(str(gateway_path))
        gateway.listen(1)
        transport_directory, transport = sandbox_supervisor._copy_cli_transport(root)
        manifests = []
        original_agent_command = cann_bot_runtime.isolation.agent_command

        def capture_manifest(*args, **kwargs):
            value = original_agent_command(*args, **kwargs)
            manifests.append(json.loads((root / 'filesystem_isolation.json').read_text()))
            return value

        try:
            with patch.dict(os.environ, {
                'PATH':str(fake_bin) + os.pathsep + os.environ['PATH'],
                'PTOBENCH_SANDBOX':'1', 'PTOBENCH_RPC_SOCKET':str(broker.sockpath),
                'PTOBENCH_API_PROXY_SOCKET':str(gateway_path),
                'PTOBENCH_CLIENT_PATH':str(transport),
            }), patch.object(cann_bot_runtime.isolation, 'agent_command', side_effect=capture_manifest):
                result = cann_bot_runtime.run_dispatch(
                    config, 'pypto-op-orchestrator', 'run a child Worker publication probe',
                    parent=None, tool_call_id=None)

            self.assertFalse(result['is_error'], result)
            self.assertIn('child published Worker', result['answer'])
            self.assertEqual(len(manifests), 2)
            dynamic_mounts = {'/ptobench/runtime/bridge', '/ptobench/cannbot.sock'}

            def stable_manifest(manifest):
                mounts = []
                for item in manifest['mounts']:
                    if item['destination'] in dynamic_mounts:
                        item = {**item, 'source': '<dispatch-specific>'}
                    mounts.append(item)
                return {**manifest, 'mounts': mounts}

            self.assertEqual(stable_manifest(manifests[0]), stable_manifest(manifests[1]))
            for manifest in manifests:
                self.assertEqual(manifest['network'], 'model_api_only')
                self.assertTrue(any(item['source'] == str(root / 'workspaces')
                                    and item['destination'] == str(root / 'workspaces')
                                    and item['access'] == 'rw' for item in manifest['mounts']))
                bridge = next(item for item in manifest['mounts']
                              if item['destination'] == '/ptobench/runtime/bridge')
                self.assertEqual(Path(bridge['source']).parent, root / 'agent/.runtime')
                tool_socket = next(item for item in manifest['mounts']
                                   if item['destination'] == '/ptobench/cannbot.sock')
                self.assertEqual(Path(tool_socket['source']).name, 'tools.sock')
                self.assertTrue(Path(tool_socket['source']).parent.name.startswith('ptb-cannbot-'))
            record = session.read_session(root)
            worker_id = next(name for name in record['workspaces'] if name != 'worker_000')
            worker = root / record['workspaces'][worker_id]['path']
            self.assertTrue((worker / 'candidate/live_child.py').is_file())
            self.assertEqual((worker / 'candidate/live_child.py').read_text(), 'CHILD = True\\n')
            dispatches = [json.loads(path.read_text()) for path in (root / 'method/dispatches').glob('*/dispatch.json')]
            self.assertEqual(len(dispatches), 2)
            self.assertTrue(all(state['status'] == 'completed' for state in dispatches))
            self.assertTrue(all(session.process_identity(state['process']['pid']) is None for state in dispatches))
            self.assertFalse(list((root / 'agent/launcher').glob('*.json')))
        finally:
            broker.stop.set()
            broker_thread.join(timeout=5)
            broker.server.close()
            broker.ipc.cleanup()
            gateway.close()
            shutil.rmtree(transport_directory, ignore_errors=True)

    def test_real_pi_task_child_uses_native_tools_and_cli_transport_after_live_broker_publication(self):
        import sandbox_supervisor
        import shlex
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from modes._shared import provider

        pi = shutil.which('pi')
        self.assertIsNotNone(pi)
        root = self.handle.session_root
        workspace = self.handle.initial_workspace
        agent_uid, share_gid = agent_launch._agent_identity()
        other_session = self.repo / 'other-session'
        other_session.mkdir()
        (other_session / 'private.txt').write_text('OTHER_SESSION_CANARY')
        extension = root / 'agent/pi-tools/cannbot-extension.ts'
        extension.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cann_bot_resources.DIRECTORY / 'bridge/pi_task.ts', extension)
        os.chown(extension, -1, share_gid)
        extension.chmod(0o440)

        child_waiting = threading.Event()
        release_child = threading.Event()
        worker_path = {}
        captured = {'requests': [], 'auth': [], 'attempts': {'parent': 0, 'child': 0},
                    'child_tools': []}
        captured_lock = threading.Lock()

        def completion(tool_calls=None, content=None):
            if tool_calls:
                deltas = [
                    {'id': 'chatcmpl-live-task', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'test-model', 'choices': [{'index': 0,
                     'delta': {'role': 'assistant', 'tool_calls': tool_calls},
                     'finish_reason': None}]},
                    {'id': 'chatcmpl-live-task', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'test-model', 'choices': [{'index': 0, 'delta': {},
                     'finish_reason': 'tool_calls'}]},
                ]
            else:
                deltas = [
                    {'id': 'chatcmpl-live-task', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'test-model', 'choices': [{'index': 0,
                     'delta': {'role': 'assistant', 'content': content or 'probe complete'},
                     'finish_reason': None}]},
                    {'id': 'chatcmpl-live-task', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'test-model', 'choices': [{'index': 0, 'delta': {},
                     'finish_reason': 'stop'}]},
                ]
            return b''.join(b'data: ' + json.dumps(item).encode() + b'\n\n' for item in deltas) + b'data: [DONE]\n\n'

        class Provider(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({'data': [{'id': 'test-model'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
                serialized = json.dumps(payload, ensure_ascii=False)
                child = 'CHILD_LIVE_PUBLICATION_PROBE' in serialized and 'PARENT_LIVE_PUBLICATION_PROBE' not in serialized
                role = 'child' if child else 'parent'
                captured['auth'].append(self.headers.get('Authorization'))
                captured['requests'].append(payload)
                captured['attempts'][role] += 1
                attempt = captured['attempts'][role]
                if child and attempt == 1:
                    captured['child_tools'] = [item.get('function', {}).get('name')
                                               for item in payload.get('tools', [])]
                    child_waiting.set()
                    if not release_child.wait(20):
                        body = completion(content='child publication barrier timed out')
                    else:
                        worker = worker_path['path']
                        probe = (
                            'import json\nfrom pathlib import Path\n'
                            f'p=Path({str(worker)!r})\n'
                            'native=Path(\'/ptobench/agent-sessions\')\n'
                            'candidate=p/\'candidate/candidate.py\'\n'
                            'checks={\'candidate_readable\':candidate.is_file() and candidate.stat().st_size > 0, '
                            '\'context_readable\':(p/\'.benchmark_context/config.py\').is_file(), '
                            '\'pi_config_hidden\':not Path(\'/ptobench/pi-config/settings.json\').exists(), '
                            '\'pi_tools_hidden\':not Path(\'/ptobench/pi-tools/cannbot-extension.ts\').exists(), '
                            '\'native_sessions_empty\':not native.exists() or '
                            '(native.is_dir() and not any(native.iterdir()))}\n'
                            'print(json.dumps(checks))'
                        )
                        command = 'python3 -c ' + shlex.quote(probe)
                        calls = [
                            ('read', {'path': str(Path(worker) / 'candidate/candidate.py')}),
                            ('read', {'path': str(other_session / 'private.txt')}),
                            ('read', {'path': str(SOURCE_BENCHMARK / 'evaluator/sandbox_supervisor.py')}),
                            ('bash', {'command': command}),
                            ('bash', {'command': 'ptobench evaluate . && ptobench profile .'}),
                        ]
                        tool_calls = [{'index': index, 'id': f'child-live-{index}', 'type': 'function',
                                       'function': {'name': name, 'arguments': json.dumps(arguments)}}
                                      for index, (name, arguments) in enumerate(calls)]
                        body = completion(tool_calls=tool_calls)
                elif child and attempt == 2:
                    call = {'index': 0, 'id': 'child-live-edit', 'type': 'function',
                            'function': {'name': 'edit', 'arguments': json.dumps({
                                'path': str(Path(worker_path['path']) / 'candidate/candidate.py'),
                                'edits': [{'oldText': 'from helper import VALUE',
                                           'newText': 'from helper import CHILD_VALUE'}],
                            })}}
                    body = completion(tool_calls=[call])
                elif not child and attempt == 1:
                    task = {'subagent_type': 'pypto-op-optimizer',
                            'prompt': 'CHILD_LIVE_PUBLICATION_PROBE: inspect the Worker published while you are running'}
                    call = {'index': 0, 'id': 'parent-live-task', 'type': 'function',
                            'function': {'name': 'Task', 'arguments': json.dumps(task)}}
                    body = completion(tool_calls=[call])
                else:
                    body = completion(content='CHILD_LIVE_TOOLS_COMPLETE' if child else 'PARENT_LIVE_TASK_COMPLETE')
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        self.addCleanup(upstream.shutdown)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream_thread.join, 5)

        profile = {
            'backend': 'pi', 'reasoning': 'off', 'env': {},
            'argv': [str(Path(pi).resolve()), '--print', '--mode', 'json',
                     '--tools', 'read,edit,bash,grep,find,ls,Task,skill,todowrite,todoread',
                     '--provider', 'test-provider', '--model', 'test-model',
                     '--thinking', 'off', '{prompt}'],
            'mode_provider': {
                'provider': 'test-provider',
                'base_url': f'http://127.0.0.1:{upstream.server_port}/v1',
                'api': 'openai-completions',
                'model': {'id': 'test-model', 'contextWindow': 8192, 'maxTokens': 1024},
            },
        }
        profile = provider.prepare(profile, root, gateway=True)
        upstream_roles = root / 'method/resources/upstream'
        upstream_roles.mkdir(parents=True, exist_ok=True)
        (upstream_roles / 'orchestrator.md').write_text('Use the Task tool for child work.\n')
        (upstream_roles / 'optimizer.md').write_text('Inspect the supplied Worker only.\n')
        (upstream_roles / 'perf-tune.md').write_text('No performance work is requested.\n')
        config = {
            'profile': profile, 'run_root': str(root), 'workspace': str(workspace),
            'upstream': str(upstream_roles),
            'roles': {'pypto-op-orchestrator': 'orchestrator.md',
                      'pypto-op-optimizer': 'optimizer.md'},
            'skills': {'pypto-op-perf-tune': str(upstream_roles / 'perf-tune.md')},
            'python': sys.executable, 'agent_extension': str(extension),
            'work_dir': str(root / 'method/work'), 'hardware_budget': 1,
        }

        private = self.repo / 'real-pi-broker-private'
        private.mkdir()
        broker = sandbox_supervisor.Broker(root, private)
        broker.cli_environment = cann_bot_runtime.isolation.clean_environment({
            'PATH': os.environ['PATH'], 'PTOBENCH_SESSION_ROOT': str(root), 'PTOBENCH_SANDBOX': '1',
        })
        broker_thread = threading.Thread(target=broker.serve, daemon=True)
        broker_thread.start()
        gateway_path = root / 'broker-endpoints/api.sock'
        gateway = subprocess.Popen([
            sys.executable, str(Path(cann_bot_runtime.isolation.__file__).with_name('network_proxy.py')),
            'serve-api', '--socket', str(gateway_path),
            '--base-url', profile['mode_provider']['base_url'], '--api', 'openai-completions',
        ], env={**os.environ, 'PTOBENCH_UPSTREAM_API_KEY': 'TEST_REAL_PI_GATEWAY_KEY'},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 5
        while not gateway_path.exists() and time.monotonic() < deadline:
            if gateway.poll() is not None:
                self.fail(gateway.stderr.read().decode())
            time.sleep(.02)
        self.assertTrue(gateway_path.exists())
        transport_directory, transport = sandbox_supervisor._copy_cli_transport(root)
        manifests = []
        original_agent_command = cann_bot_runtime.isolation.agent_command

        def capture_manifest(*args, **kwargs):
            result = original_agent_command(*args, **kwargs)
            manifests.append(json.loads((root / 'filesystem_isolation.json').read_text()))
            return result

        dispatch_result = {}
        dispatch_failure = []

        def run_parent():
            try:
                dispatch_result.update(cann_bot_runtime.run_dispatch(
                    config, 'pypto-op-orchestrator',
                    'PARENT_LIVE_PUBLICATION_PROBE: dispatch one child Pi and wait for its inspection.',
                    parent=None, tool_call_id=None))
            except BaseException as error:
                dispatch_failure.append(error)

        dispatch_thread = threading.Thread(target=run_parent, daemon=True)
        try:
            with patch.dict(os.environ, {
                'PTOBENCH_SANDBOX': '1', 'PTOBENCH_RPC_SOCKET': str(broker.sockpath),
                'PTOBENCH_API_PROXY_SOCKET': str(gateway_path),
                'PTOBENCH_CLIENT_PATH': str(transport),
            }), patch.object(cann_bot_runtime.isolation, 'agent_command', side_effect=capture_manifest):
                dispatch_thread.start()
                reached_gateway = child_waiting.wait(30)
                if not reached_gateway:
                    details = {
                        'dispatch_failure': [repr(error) for error in dispatch_failure],
                        'gateway_attempts': captured['attempts'],
                        'dispatches': [],
                    }
                    for directory in (root / 'method/dispatches').glob('*'):
                        if not directory.is_dir():
                            continue
                        state_path = directory / 'dispatch.json'
                        details['dispatches'].append({
                            'state': json.loads(state_path.read_text()) if state_path.exists() else None,
                            'stdout': (directory / 'stdout.log').read_text(errors='replace')
                                      if (directory / 'stdout.log').exists() else '',
                            'stderr': (directory / 'stderr.log').read_text(errors='replace')
                                      if (directory / 'stderr.log').exists() else '',
                        })
                    self.fail('real Task child did not reach Gateway barrier: ' +
                              json.dumps(details, ensure_ascii=False))
                live_dispatches = [json.loads(path.read_text()) for path in
                                   (root / 'method/dispatches').glob('*/dispatch.json')]
                child_dispatch = next(state for state in live_dispatches
                                      if state.get('role') == 'pypto-op-optimizer')
                child_pid = child_dispatch['process']['pid']
                child_proc = Path('/proc') / str(child_pid)
                launcher_children = [int(value) for value in
                                     (child_proc / 'task' / str(child_pid) / 'children').read_text().split()]
                self.assertTrue(launcher_children, 'real Pi dispatcher has no live Agent child')
                trusted_bwrap = Path(cann_bot_runtime.isolation.executable()).resolve()
                agent_bwrap_seen = False
                for pid in launcher_children:
                    process = Path('/proc') / str(pid)
                    status = {line.partition(':')[0]: line.partition(':')[2].strip()
                              for line in (process / 'status').read_text().splitlines()
                              if ':' in line}
                    argv = (process / 'cmdline').read_bytes().split(b'\0')
                    if ([int(value) for value in status['Uid'].split()] == [agent_uid] * 4
                            and [int(value) for value in status['Gid'].split()] == [share_gid] * 4
                            and int(status['CapEff'], 16) == 0
                            and Path(os.fsdecode(argv[0])).resolve() == trusted_bwrap):
                        agent_bwrap_seen = True
                        break
                self.assertTrue(agent_bwrap_seen,
                                'live Pi child did not enter Bubblewrap as the dedicated Agent')
                with socket.socket(socket.AF_UNIX) as client:
                    client.connect(str(broker.sockpath))
                    client.sendall((json.dumps({'argv': ['fork'], 'cwd': str(workspace)}) + '\n').encode())
                    events = [json.loads(line) for line in client.makefile('rb')]
                self.assertEqual(events[-1]['exit_code'], 0, events)
                broker_output = ''.join(item.get('output', '') for item in events)
                worker_path['path'] = json.loads(broker_output.strip().splitlines()[-1])['workspace']
                self.assertTrue(Path(worker_path['path']).is_dir())
                release_child.set()
                dispatch_thread.join(timeout=45)

            self.assertFalse(dispatch_thread.is_alive(), 'parent or Task child remained alive')
            self.assertFalse(dispatch_failure, dispatch_failure)
            self.assertFalse(dispatch_result.get('is_error'), dispatch_result)
            self.assertIn('PARENT_LIVE_TASK_COMPLETE', dispatch_result.get('answer', ''))
            self.assertEqual((Path(worker_path['path']) / 'candidate/candidate.py').read_text(),
                             'from helper import CHILD_VALUE\n')
            self.assertEqual(len(manifests), 2)
            dynamic = {'/ptobench/runtime/bridge', '/ptobench/cannbot.sock'}
            stable = lambda manifest: {**manifest, 'mounts': [
                {**item, 'source': '<dispatch-specific>'} if item['destination'] in dynamic else item
                for item in manifest['mounts']]}
            self.assertEqual(stable(manifests[0]), stable(manifests[1]))
            self.assertTrue(all(manifest['network'] == 'model_api_only' for manifest in manifests))
            self.assertEqual(captured['attempts'], {'parent': 2, 'child': 3})
            self.assertNotIn('Task', captured['child_tools'])
            child_requests = [request for request in captured['requests']
                              if 'CHILD_LIVE_PUBLICATION_PROBE' in json.dumps(request)
                              and 'PARENT_LIVE_PUBLICATION_PROBE' not in json.dumps(request)]
            self.assertEqual(len(child_requests), 3)
            tool_results = {message.get('tool_call_id'): message.get('content', '')
                            for message in child_requests[1].get('messages', [])
                            if message.get('role') == 'tool'}
            self.assertIn('ENOENT', tool_results['child-live-1'])
            self.assertIn('ENOENT', tool_results['child-live-2'])
            self.assertEqual(json.loads(tool_results['child-live-3']), {
                'candidate_readable': True, 'context_readable': True, 'pi_config_hidden': True,
                'pi_tools_hidden': True, 'native_sessions_empty': True})
            edit_results = {message.get('tool_call_id'): message.get('content', '')
                            for message in child_requests[2].get('messages', [])
                            if message.get('role') == 'tool'}
            self.assertIn('Successfully replaced', edit_results['child-live-edit'])
            self.assertNotIn('OTHER_SESSION_CANARY', json.dumps(child_requests))
            self.assertNotIn('class Broker:', json.dumps(child_requests))
            self.assertTrue(captured['auth'])
            self.assertTrue(all(value == 'Bearer TEST_REAL_PI_GATEWAY_KEY' for value in captured['auth']))
            self.assertNotIn('TEST_REAL_PI_GATEWAY_KEY', json.dumps(captured['requests']))
            dispatches = [json.loads(path.read_text()) for path in
                          (root / 'method/dispatches').glob('*/dispatch.json')]
            self.assertEqual(len(dispatches), 2)
            self.assertTrue(all(state['status'] == 'completed' for state in dispatches))
            self.assertTrue(all(session.process_identity(state['process']['pid']) is None
                                for state in dispatches))
            session.read_session(root)
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines() if line.strip()]
            action_events = [event for event in events if event.get('action') in ('evaluate', 'profile')]
            self.assertEqual(
                [(event['action'], event['result'].get('profiling_mode'),
                  event.get('agent_dispatch_id'))
                 for event in action_events],
                [('evaluate', 'clean', child_dispatch['id']),
                 ('profile', 'combined', child_dispatch['id'])],
            )
            self.assertFalse(list((root / 'agent/launcher').glob('*.json')))
        finally:
            release_child.set()
            broker.stop.set()
            broker_thread.join(timeout=5)
            broker.server.close()
            broker.ipc.cleanup()
            if gateway.poll() is None:
                gateway.terminate()
            gateway.wait(timeout=5)
            shutil.rmtree(transport_directory, ignore_errors=True)


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

    def pi_bundle(self):
        package = self.handle.session_root / 'agent/pi-tools/fake-pi'
        bundle = package / 'dist/bundle'
        bundle.mkdir(parents=True)
        (package / 'node_modules/jiti').mkdir(parents=True)
        shutil.copyfile(self.tool, bundle / 'fake-native.py')
        (bundle / 'cli.js').write_text(
            '#!/usr/bin/env node\n'
            "const {spawnSync} = require('node:child_process');\n"
            f"const result = spawnSync({self.python!r}, "
            "[__dirname + '/fake-native.py', ...process.argv.slice(2)], {stdio:'inherit'});\n"
            "if (result.error) throw result.error;\n"
            "process.exit(result.status === null ? 1 : result.status);\n"
        )
        (bundle / 'cli.js').chmod(0o550)
        return bundle / 'cli.js'

    def cannbot_config(self, backend='pi', budget=6, **options):
        import yaml
        self.launcher_config(backend, budget, experiment_mode='cannbot', **options)
        self.runs.mkdir(exist_ok=True)
        self.snapshots.mkdir(exist_ok=True)
        if backend == 'pi':
            self.env['PTOBENCH_SANDBOX'] = '1'
            control = json.loads(self.launcher_control.read_text())
            for key in ('seen', 'ready', 'go', 'child_pid', 'worker_pid'):
                control.pop(key, None)
            profile_path = self.bench / 'modes/_shared/profiles/pi/default.yaml'
            profile = yaml.safe_load(profile_path.read_text())
            profile['argv'][0] = str(self.pi_bundle())
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

    def test_codex_dispatch_fails_closed_when_sandbox_is_disabled(self):
        with tempfile.TemporaryDirectory(prefix='cannbot-unsupported-backend-') as temporary:
            root = Path(temporary)
            session_root, workspace, upstream = (root / name for name in ('session', 'workspace', 'upstream'))
            session_root.mkdir()
            workspace.mkdir()
            upstream.mkdir()
            (upstream / 'role.md').write_text('role')
            profile = {'backend': 'codex', 'argv': ['codex', '{prompt}'], 'env': {}}
            config = {'profile': profile, 'run_root': str(session_root), 'workspace': str(workspace),
                      'upstream': str(upstream), 'roles': {'pypto-op-orchestrator': 'role.md'},
                      'skills': {}, 'python': sys.executable, 'agent_extension': None}
            with patch.dict(os.environ, {'PTOBENCH_SANDBOX': '0'}), \
                 patch.object(cann_bot_runtime, 'render', return_value=(['codex', '{prompt}'], {})), \
                 patch.object(cann_bot_runtime.subprocess, 'Popen',
                              side_effect=AssertionError('unisolated Codex process was started')) as spawn:
                with self.assertRaisesRegex(RuntimeError, 'only Pi Agent is supported'):
                    cann_bot_runtime.run_dispatch(
                        config, 'pypto-op-orchestrator', 'task', parent=None, tool_call_id=None)
            spawn.assert_not_called()

    def test_codex_mcp_fresh_sessions_same_budget_and_original_resources(self):
        self.skipTest('Plan 03 supports Pi only; equivalent Pi integration is covered separately')

    def test_pi_extension_fresh_sessions_same_budget_and_original_resources(self):
        if not self.node or int(subprocess.check_output([self.node, '--version']).decode().split('.')[0][1:]) < 22:
            self.skipTest('Pi TypeScript bridge smoke requires Node 22+')
        self.check_native('pi')

    def check_native(self, backend):
        self.cannbot_config(backend)
        process, value = self.launch()
        if process.returncode:
            root = Path(value['session_root'])
            diagnostics = []
            for dispatch in self.dispatches(root):
                directory = root / 'method/dispatches' / dispatch['id']
                diagnostics.append({
                    'dispatch': dispatch,
                    'stdout': (directory / 'stdout.log').read_text(errors='replace')
                              if (directory / 'stdout.log').exists() else '',
                    'stderr': (directory / 'stderr.log').read_text(errors='replace')
                              if (directory / 'stderr.log').exists() else '',
                    'result': json.loads((directory / 'result.json').read_text())
                              if (directory / 'result.json').exists() else None,
                })
            self.fail(f'CANNBot {backend} launcher failed: {value}; dispatches={diagnostics}')
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
            directory = root / 'method/dispatches' / state['id']
            self.assertEqual(state['status'], 'completed', {
                'dispatch': state,
                'stdout': (directory / 'stdout.log').read_text(errors='replace'),
                'stderr': (directory / 'stderr.log').read_text(errors='replace'),
                'result': json.loads((directory / 'result.json').read_text()),
            })
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

    def test_pi_task_cancellation_propagates_and_next_stage_runs(self):
        if not self.node or int(subprocess.check_output([self.node, '--version']).decode().split('.')[0][1:]) < 22:
            self.skipTest('Pi TypeScript bridge smoke requires Node 22+')
        self.ready.unlink(missing_ok=True)
        child, worker = self.repo / 'child.pid', self.repo / 'worker.pid'
        self.cannbot_config('pi', cancel_child=True, child_pid=str(child), worker_pid=str(worker))
        process, value = self.launch()
        if process.returncode:
            root = Path(value['session_root'])
            diagnostics = []
            for dispatch in self.dispatches(root):
                directory = root / 'method/dispatches' / dispatch['id']
                diagnostics.append({
                    'id': dispatch['id'], 'status': dispatch['status'],
                    'stdout': (directory / 'stdout.log').read_text(errors='replace'),
                    'stderr': (directory / 'stderr.log').read_text(errors='replace'),
                    'recording': json.loads((directory / 'recording.json').read_text()),
                })
            self.fail((value, process.stderr.decode(), diagnostics,
                       [path.name for path in (root / 'agent/native').glob('*')]))
        try:
            root, _ = self.assert_final(value, 2)
        except AssertionError:
            root = Path(value['session_root'])
            self.fail({
                'final': json.loads((root / 'final_result.json').read_text()),
                'events': [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()],
                'dispatches': [(state['id'], state['status'],
                                (root / 'method/dispatches' / state['id'] / 'result.json').read_text(),
                                (root / 'method/dispatches' / state['id'] / 'stdout.log').read_text()[-2000:],
                                (root / 'method/dispatches' / state['id'] / 'stderr.log').read_text()[-2000:])
                               for state in self.dispatches(root)],
                'cancellations': [path.read_text() for path in (root / 'method/tools/cancellations').glob('*.json')],
            })
        dispatches = self.dispatches(root)
        cancelled_rows = [state for state in dispatches if state['status'] == 'cancelled']
        self.assertTrue(cancelled_rows, {
            'dispatches': [(state['id'], state['status'],
                            (root / 'method/dispatches' / state['id'] / 'result.json').read_text(),
                            (root / 'method/dispatches' / state['id'] / 'stdout.log').read_text()[-1200:])
                           for state in dispatches],
            'cancellations': [path.read_text() for path in (root / 'method/tools/cancellations').glob('*.json')],
            'marker': (root / 'workspaces/worker_000/candidate/.fake-agent-control.py').exists(),
        })
        cancelled = cancelled_rows[0]
        heartbeat, = (root / 'agent/native').glob('*.worker-heartbeat')
        last_value, last_update = heartbeat.read_text(), heartbeat.stat().st_mtime_ns
        time.sleep(.15)
        self.assertEqual(heartbeat.read_text(), last_value)
        self.assertEqual(heartbeat.stat().st_mtime_ns, last_update)
        result = json.loads((root / 'method/dispatches' / cancelled['id'] / 'result.json').read_text())
        self.assertTrue(result['cancelled'])
        self.assertTrue(result['is_error'])
        self.assertIsNone(session.process_identity(cancelled['process']['pid']))
        self.assertEqual(len(self.dispatches(root)), 3)


if __name__ == '__main__':
    unittest.main()
