"""S5: real launcher/Optimizer/ptobench subprocesses with a CPU Evaluator."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from action_test_support import ActionTestBase, SOURCE_BENCHMARK, write_json
import session

FAKE_TOOL = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
if '--version' in sys.argv:
    print('fake-tool 1.0'); sys.exit()
if '--help' in sys.argv:
    print('--skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --dangerously-skip-permissions --print'); sys.exit()
c=json.loads(Path(os.environ['S5_CONTROL']).read_text())
root=Path(os.environ['PTOBENCH_SESSION_ROOT'])
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
if c.get('child'):
    child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
    Path(c['child']).write_text(str(child.pid))
if c.get('mode')=='empty': sys.exit(c.get('exit',0))
if c.get('mode') != 'profiles_only': call(['evaluate'])
if c.get('mode')=='s6':
    fork=call(['fork'])
    target=Path(json.loads(fork.stdout.decode().splitlines()[-1])['workspace'])
    (target/'helper.py').write_text('VALUE = 2\n')
    control=Path(os.environ['S3_CONTROL'])
    control.write_text(json.dumps({'median':5.0}))
    call(['profile','runtime',str(target)])
    call(['evaluate',str(target)])
    control.write_text(json.dumps({'median':7.0}))
if c.get('mode')=='profiles_only':
    for mode in ('runtime','pmu','compiler'): call(['profile',mode])
if c.get('mode')=='all':
    fork=call(['fork'])
    data=json.loads(fork.stdout.decode().splitlines()[-1])
    target=data['workspace']
    for mode in ('runtime','pmu','compiler'): call(['profile',mode,target])
if c.get('mode')=='active':
    Path(c['ready']).write_text('ready')
    while not Path(c['go']).exists(): time.sleep(.01)
    call(['evaluate'])
if c.get('mode') in ('budget','interrupt','active'):
    Path(c['ready']).write_text('ready')
    while True: time.sleep(.02)
sys.exit(c.get('exit',0))
'''


class LauncherTests(ActionTestBase):
    def setUp(self):
        super().setUp()
        # Keep the venv path: resolving its symlink loses local PyYAML.
        self.python = str(Path(sys.executable).absolute())
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
        self.launcher_config()

    def launcher_config(self, backend='codex', budget=4, **control):
        import yaml
        profile = yaml.safe_load((self.bench / 'launcher/profiles' / backend / 'default.yaml').read_text())
        for key in ('argv','version_argv','help_argv'):
            profile[key][0] = str(self.tool)
        (self.bench / 'launcher/profiles' / backend / 'default.yaml').write_text(yaml.safe_dump(profile))
        write_json(self.launcher_control, {'seen':str(self.seen),'ready':str(self.ready),'go':str(self.go),**control})
        self.runs = self.repo / 'launcher_runs'
        self.snapshots = self.repo / 'launcher_snapshots'
        self.config.write_text(yaml.safe_dump({'case':self.case_id,
            'optimizer':{'id':'fake_'+backend,'backend':backend},
            'output':{'runs_root':'launcher_runs','snapshots_root':'launcher_snapshots'},
            'budget':{'max_hardware_actions':budget,'finish_actions':1}}))

    def launch(self, asynchronous=False):
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

    def test_four_adapters_real_cli_all_five_operations(self):
        for backend in ('codex','claudecode','pi','opencode'):
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
                self.assertEqual(prompt.count('  ptobench '),5)
                self.assertNotIn('profile hardware',prompt)
                self.assertNotIn('profile compile ',prompt)
                self.assertIn('optimizer stdout',(root/'process.log').read_text())
                self.assertIn('optimizer stderr',(root/'process.log').read_text())
                self.assertEqual(json.loads((root/'backend.json').read_text())['backend']['version'],'fake-tool 1.0')

    def test_s6_fork_helper_edit_profile_evaluate_exit_provenance(self):
        self.launcher_config(budget=4,mode='s6')
        p,value=self.launch()
        self.assertEqual(p.returncode,0,(value,p.stderr.decode()))
        root,final=self.assert_final(value,3)
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['action'] for event in events],['evaluate','fork','profile','evaluate','finish'])
        first,profile,improved=events[0],events[2],events[3]
        self.assertNotEqual(first['content_sha256'],improved['content_sha256'])
        self.assertEqual(profile['content_sha256'],improved['content_sha256'])
        self.assertEqual(final['selected_snapshot'],profile['snapshot_id'])  # equal latency: earlier event
        selected=Path(session.read_session(root)['snapshots_root'])/profile['snapshot_id']
        self.assertEqual((selected/'source/helper.py').read_text(),'VALUE = 2\n')
        self.assertEqual((root/'final/workspace/helper.py').read_text(),'VALUE = 2\n')
        self.assertEqual(events[-1]['result']['runs'][0]['median_effective_us'],7.0)
        self.assertFalse((root/'final/workspace/__pycache__').exists())

    def test_s6_all_null_profiles_select_latest_and_clean_finish(self):
        self.launcher_config(budget=4,mode='profiles_only')
        p,value=self.launch()
        self.assertEqual(p.returncode,0,(value,p.stderr.decode()))
        root,final=self.assert_final(value,3)
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertTrue(all(event['observation']['latency_us'] is None for event in events[:3]))
        self.assertEqual(final['selected_snapshot'],events[2]['snapshot_id'])
        self.assertEqual(final['selection_reason'],'latest_pass_without_latency')
        self.assertEqual(events[-1]['result']['profiling_mode'],'clean')

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

    def test_budget_stop_and_dual_large_logs(self):
        self.launcher_config(budget=1,mode='budget',huge=True)
        p,value=self.launch()
        self.assertEqual(p.returncode,0,value)
        root,_=self.assert_final(value)
        self.assertEqual(value['close_reason'],'budget_exhausted')
        self.assertIn(b'O'*262144,(root/'stdout.log').read_bytes())
        self.assertIn(b'E'*262144,(root/'stderr.log').read_bytes())

    def test_blocked_terminal_does_not_stall_finalization_or_exit(self):
        self.launcher_config(budget=1,mode='budget',huge=True)
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

    def test_budget_waits_for_last_active_action(self):
        self.launcher_config(budget=1,mode='budget')
        action_ready=self.repo/'action.ready'; release=self.repo/'release'
        self.configure(pause=True,ready=str(action_ready),release=str(release))
        p=self.launch(True)
        self.wait_file(action_ready,p)
        time.sleep(1.2)
        self.assertIsNone(p.poll())
        root=next(self.runs.glob('*/*/*'))
        record=session.read_session(root)
        self.assertIsNotNone(record['active_action'])
        self.assertFalse(record['exploration_closed'])
        self.configure()
        release.write_text('go')
        value,_,_=self.result(p)
        self.assertEqual(p.returncode,0,value)
        self.assert_final(value)

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
        profile_path=self.bench/'launcher/profiles/codex/default.yaml'
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
