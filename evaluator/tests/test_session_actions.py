import json
import os
from pathlib import Path
import signal
import subprocess
import time
from unittest import mock

from action_test_support import ActionTestBase
import common
import integrity
import session
import session_actions
import workspace


class SessionActionsTest(ActionTestBase):
    def test_invalid_results_and_exceptions_charge_once_never_snapshot(self):
        for config in ({'fail':True},{'crash':True},{'missing':True},{'bad_case':True},{'bad_fixture':True},
                       {'bad_phase':True},{'missing_speedup':True},{'latency':'nan'},{'latency':'zero'},{'nonzero':True}):
            with self.subTest(config=config):
                self.configure(**config)
                process,result=self.cli(['evaluate'])
                self.assertEqual(process.returncode,1,result)
                self.assertEqual(result['error_code'],'action_failed')
                self.assertIsNone(result['snapshot_id'])
                event=self.events()[-1]
                self.assertTrue(event['hardware_action'])
                self.assertTrue(event['integrity_passed'])
                self.assertEqual(event['state'],'completed')
        self.assertEqual(len(self.calls()),10)
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],10)
        self.assertFalse(list(self.handle.snapshots_root.iterdir()))

    def test_protected_and_frozen_changes_invalid_even_if_result_pass(self):
        sentinel=self.repo/'sentinel'; sentinel.write_text('outside')
        for kind in ('fixed','context','source','source_link','record'):
            with self.subTest(kind=kind):
                self.configure(tamper=kind,sentinel=str(sentinel),session_root=str(self.handle.session_root))
                process,result=self.cli(['evaluate'])
                self.assertEqual(process.returncode,4,result)
                self.assertEqual(result['error_code'],'integrity_failed')
                self.assertIsNone(result['snapshot_id'])
                self.assertFalse(self.events()[-1]['integrity_passed'])
                self.assertEqual(sentinel.read_text(),'outside')
                self.assertEqual((self.bench/'cases/test/case/public/config.py').read_text(),'SIZE = 2\n')
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],5)
        self.assertFalse(list(self.handle.snapshots_root.iterdir()))

    def test_original_edits_after_freeze_do_not_change_snapshot_or_hash(self):
        ready,release=self.repo/'ready',self.repo/'release'
        self.configure(pause=True,ready=str(ready),release=str(release))
        process=self.start_cli(['evaluate'])
        self.wait_file(ready,process)
        (self.handle.initial_workspace/'helper.py').write_text('VALUE = 500\n')
        release.write_text('release')
        output,error=process.communicate(timeout=30)
        self.assertEqual(process.returncode,0,error.decode())
        result=json.loads(output.decode().splitlines()[-1])
        snapshot=self.snapshot(result)
        self.assertEqual((snapshot/'source/helper.py').read_text(),'VALUE = 1\n')
        self.assertEqual((self.handle.initial_workspace/'helper.py').read_text(),'VALUE = 500\n')
        manifest=workspace.source_manifest(snapshot/'source')
        self.assertEqual(manifest['content_sha256'],self.events()[-1]['content_sha256'])
        self.assertEqual(manifest['content_sha256'],common.read_json(snapshot/'metadata.json')['content_sha256'])

    def test_dual_large_logs_are_preserved_and_forwarded_without_deadlock(self):
        self.configure(huge_logs=True)
        process,result=self.cli(['evaluate'])
        self.assertEqual(process.returncode,0,result)
        action=self.handle.session_root/'staging'/f"action_{result['event_seq']:06d}"
        stdout=(action/'stdout.log').read_bytes(); stderr=(action/'stderr.log').read_bytes()
        self.assertTrue(stdout.startswith(b'O'*1048576))
        self.assertTrue(stderr.startswith(b'E'*1048576))
        self.assertIn(b'fake evaluator stdout',process.stdout)
        self.assertIn(b'fake evaluator stderr',process.stderr)
        self.assertTrue(process.stdout.startswith(stdout))
        self.assertEqual(process.stderr,stderr)

    def test_blocked_terminal_consumers_do_not_block_logs_or_completion(self):
        self.configure(huge_logs=True)
        process=self.start_cli(['evaluate'])
        # Deliberately read neither terminal pipe until the CLI has exited.
        deadline=time.monotonic()+15
        while process.poll() is None and time.monotonic()<deadline:
            time.sleep(.02)
        self.assertEqual(process.poll(),0,'blocked terminal prevented completion')
        process.communicate(timeout=5)
        event=self.events()[-1]
        self.assertIsNone(event['error_code'])
        action=self.handle.session_root/f"staging/action_{event['seq']:06d}"
        self.assertTrue((action/'stdout.log').read_bytes().startswith(b'O'*1048576))
        self.assertTrue((action/'stderr.log').read_bytes().startswith(b'E'*1048576))
        self.assertTrue(event['snapshot_id'])

    def test_status_repair_during_execution_preserves_tamper_witness(self):
        ready,release=self.repo/'ready',self.repo/'release'
        self.configure(pause=True,ready=str(ready),release=str(release))
        process=self.start_cli(['evaluate']); self.wait_file(ready,process)
        (self.handle.session_root/'session.json').write_text('{}')
        state=session.read_session(self.handle.session_root)
        self.assertTrue(state['active_action']['integrity_violations'])
        self.assertTrue(integrity.check_only(self.handle.protection_root,'records').passed)
        release.write_text('go')
        stdout,stderr=process.communicate(timeout=30)
        result=json.loads(stdout.decode().splitlines()[-1])
        self.assertEqual((process.returncode,result['error_code']),(4,'integrity_failed'),stderr.decode())
        self.assertFalse(self.events()[-1]['integrity_passed'])
        self.assertFalse(list(self.handle.snapshots_root.iterdir()))

    def test_change_after_check_only_is_detected_by_restoration(self):
        real_check=integrity.check_only
        fixed=self.bench/'cases/test/case/public/config.py'
        injected=[]
        def check_then_change(protection,groups=None):
            report=real_check(protection,groups)
            if groups is None and not injected:
                fixed.write_text('changed between check and restore\n')
                injected.append(True)
            return report
        with mock.patch.dict(os.environ,self.env), mock.patch.object(integrity,'check_only',side_effect=check_then_change):
            result=session_actions.run_evaluate(self.handle.session_root,self.handle.initial_workspace)
        self.assertTrue(injected)
        self.assertEqual((result.exit_code,result.error_code),(4,'integrity_failed'))
        self.assertEqual(fixed.read_text(),'SIZE = 2\n')
        self.assertIsNone(self.events()[-1]['snapshot_id'])

    def test_same_mode_publish_replaces_complete_set_and_public_result_is_not_authority(self):
        process,first=self.cli(['evaluate']); self.assertEqual(process.returncode,0,first)
        output=self.handle.initial_workspace/'.benchmark_outputs/clean'
        self.assertTrue((output/'first_only.txt').exists())
        (output/'latest_result.json').write_text('{"verdict":"FAIL","runs":[]}')
        (output/'latest_session_result.json').write_text('{"remaining_hardware_actions":999}')
        (output/'rogue.txt').write_text('old')
        self.configure(generation=2)
        process,second=self.cli(['evaluate']); self.assertEqual(process.returncode,0,second)
        self.assertFalse((output/'first_only.txt').exists())
        self.assertFalse((output/'rogue.txt').exists())
        self.assertEqual((output/'generation.txt').read_text(),'2')
        self.assertEqual(self.events()[0]['result']['verdict'],'PASS')
        self.assertEqual(common.read_json(output/'latest_session_result.json')['event_seq'],second['event_seq'])
        self.assertTrue(common.read_json(output/'latest_session_result.json')['integrity_passed'])
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],2)
        self.assertTrue(Path(self.calls()[0]['candidate']).exists(), 'staging remains for build/report references')

    def test_output_root_or_mode_link_inserted_after_freeze_never_writes_external(self):
        for location in ('root','mode'):
            with self.subTest(location=location):
                ready=self.repo/f'ready_{location}'; release=self.repo/f'release_{location}'
                outside=self.repo/f'outside_{location}'; outside.mkdir()
                sentinel=outside/'sentinel'; sentinel.write_bytes(b'outside sentinel')
                self.configure(pause=True,ready=str(ready),release=str(release))
                process=self.start_cli(['evaluate']); self.wait_file(ready,process)
                output=self.handle.initial_workspace/'.benchmark_outputs'
                if location=='root':
                    output.symlink_to(outside,target_is_directory=True)
                else:
                    import shutil
                    shutil.rmtree(output/'clean')
                    (output/'clean').symlink_to(outside,target_is_directory=True)
                release.write_text('go')
                stdout,stderr=process.communicate(timeout=30)
                result=json.loads(stdout.decode().splitlines()[-1])
                self.assertEqual(process.returncode,0,(result,stderr.decode()))
                self.assertEqual(sentinel.read_bytes(),b'outside sentinel')
                self.assertEqual(list(outside.iterdir()),[sentinel])
                self.assertFalse(output.is_symlink())
                self.assertFalse((output/'clean').is_symlink())
                self.assertTrue((output/'clean/latest_session_result.json').exists())

    def test_spawn_failure_still_postchecks_commits_without_refund(self):
        with mock.patch.dict(os.environ,self.env), mock.patch.object(session_actions.subprocess,'Popen',side_effect=OSError('spawn failed')):
            result=session_actions.run_evaluate(self.handle.session_root,self.handle.initial_workspace)
        self.assertEqual((result.exit_code,result.error_code),(5,'infrastructure_error'))
        record=session.read_session(self.handle.session_root)
        self.assertEqual(record['usage']['hardware_actions'],1)
        self.assertIsNone(record['active_action'])
        self.assertEqual(record['status'],'failed')
        event=self.events()[-1]
        self.assertFalse(event['action_started'])
        self.assertTrue(event['integrity_passed'])
        self.assertIsNone(event['snapshot_id'])
        self.assertFalse((self.bench/'.ptobench/active_action.json').exists())

    def test_sigterm_stops_group_then_postchecks_and_records_once(self):
        ready=self.repo/'ready'; child_pid=self.repo/'child_pid'
        self.configure(hang=True,child=True,ready=str(ready),child_pid=str(child_pid))
        process=self.start_cli(['evaluate']); self.wait_file(ready,process)
        child=int(child_pid.read_text())
        self.assertIsNotNone(session.process_identity(child))
        process.send_signal(signal.SIGTERM)
        output,error=process.communicate(timeout=30)
        result=json.loads(output.decode().splitlines()[-1])
        self.assertEqual(process.returncode,130,(result,error.decode()))
        self.assertEqual(result['error_code'],'interrupted')
        self.assertIsNone(session.process_identity(child))
        record=session.read_session(self.handle.session_root)
        self.assertEqual(record['usage']['hardware_actions'],1)
        self.assertIsNone(record['active_action'])
        self.assertTrue(record['exploration_closed'])
        self.assertEqual(len(self.events()),1)
        self.assertFalse((self.bench/'.ptobench/active_action.json').exists())

    def test_repeated_signals_cannot_interrupt_term_then_kill_cleanup(self):
        ready=self.repo/'ready'
        self.configure(hang=True,ignore_term=True,ready=str(ready))
        process=self.start_cli(['evaluate']); self.wait_file(ready,process)
        leader=common.read_json(self.bench/'.ptobench/active_action.json')['process']['pid']
        process.send_signal(signal.SIGTERM)
        time.sleep(.2)
        process.send_signal(signal.SIGINT)
        stdout,stderr=process.communicate(timeout=25)
        result=json.loads(stdout.decode().splitlines()[-1])
        self.assertEqual((process.returncode,result['error_code']),(130,'interrupted'),stderr.decode())
        self.assertIsNone(session.process_identity(leader))
        self.assertEqual(len(self.events()),1)
        self.assertIsNone(session.read_session(self.handle.session_root)['active_action'])
        self.assertFalse((self.bench/'.ptobench/active_action.json').exists())

    def test_unconfirmed_process_stop_blocks_postcheck_and_keeps_evidence(self):
        with mock.patch.object(session_actions,'_run_process',side_effect=session_actions.ProcessCleanupError('still live')), \
             mock.patch.object(session_actions,'_postcheck_action') as postcheck:
            result=session_actions.run_evaluate(self.handle.session_root,self.handle.initial_workspace)
        self.assertEqual((result.exit_code,result.error_code),(5,'infrastructure_error'))
        postcheck.assert_not_called()
        record=session.read_session(self.handle.session_root)
        self.assertEqual(record['status'],'failed')
        self.assertTrue(record['exploration_closed'])
        self.assertEqual(record['usage']['hardware_actions'],1)
        self.assertIsNotNone(record['active_action'])
        self.assertEqual(self.events(),[])

    def test_reused_pid_is_never_signaled(self):
        identity={'pid':12345,'pgid':12345,'start_ticks':100}
        reused={**identity,'start_ticks':200}
        with mock.patch.object(session,'process_identity',return_value=reused), \
             mock.patch.object(session_actions.os,'killpg') as kill:
            with self.assertRaises(integrity.InfrastructureError):
                session_actions._stop_group(identity)
        kill.assert_not_called()

    def test_real_cli_competition_accepts_only_last_budget_once(self):
        other=self.create(budget=1,optimizer='pi')
        env={**self.env,'PTOBENCH_SESSION_ROOT':str(other.session_root)}
        processes=[subprocess.Popen([self.python,str(self.bench/'ptobench'),'evaluate',str(other.initial_workspace)],
                   env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(4)]
        for process in processes:
            self.addCleanup(lambda p=process: p.kill() if p.poll() is None else None)
        results=[]
        for process in processes:
            stdout,stderr=process.communicate(timeout=30)
            results.append((process.returncode,json.loads(stdout.decode().splitlines()[-1])))
        self.assertEqual(sorted(code for code,_ in results),[0,3,3,3])
        self.assertEqual(len(self.calls()),1)
        self.assertEqual(session.read_session(other.session_root)['usage']['hardware_actions'],1)
        self.assertEqual(len(list(other.snapshots_root.iterdir())),1)

    def test_killed_parent_recovers_global_marker_before_next_session_executes(self):
        ready=self.repo/'ready'; child_pid=self.repo/'child_pid'
        self.configure(hang=True,child=True,ready=str(ready),child_pid=str(child_pid))
        process=self.start_cli(['evaluate']); self.wait_file(ready,process)
        marker=common.read_json(self.bench/'.ptobench/active_action.json')
        child=int(child_pid.read_text()); leader=marker['process']['pid']
        process.kill(); process.communicate(timeout=10)
        self.assertIsNotNone(session.process_identity(leader))
        # A separate optimizer Session triggers global recovery first.
        other=self.create(budget=2,optimizer='pi')
        env={**self.env,'PTOBENCH_SESSION_ROOT':str(other.session_root)}
        self.configure()
        process,result=self.cli(['evaluate',str(other.initial_workspace)],env=env)
        self.assertEqual(process.returncode,0,result)
        self.assertIsNone(session.process_identity(leader))
        self.assertIsNone(session.process_identity(child))
        old=session.read_session(self.handle.session_root)
        self.assertEqual(old['usage']['hardware_actions'],1)
        self.assertTrue(old['exploration_closed'])
        self.assertIsNone(old['active_action'])
        self.assertEqual(self.events()[0]['error_code'],'interrupted')
        self.assertEqual(len(self.events()),1)
        self.assertFalse((self.bench/'.ptobench/active_action.json').exists())

    def test_owner_killed_after_reservation_before_marker_recovers_once(self):
        code="""import os
from unittest import mock
import session_actions
with mock.patch.object(session_actions, '_run_process', side_effect=lambda *a: os._exit(23)):
    session_actions.run_evaluate(os.environ['PTOBENCH_SESSION_ROOT'], '.')
"""
        env={**self.env,'PYTHONPATH':str(self.bench/'evaluator')}
        process=subprocess.run([self.python,'-c',code],cwd=self.handle.initial_workspace,env=env,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
        self.assertEqual(process.returncode,23,process.stderr.decode())
        self.assertFalse((self.bench/'.ptobench/active_action.json').exists())
        record=session.read_session(self.handle.session_root)
        self.assertEqual(record['active_action']['process_state'],'not_started')
        for _ in range(2):
            process,result=self.cli(['evaluate'])
            self.assertEqual((process.returncode,result['error_code']),(3,'session_closed'))
        accepted=[event for event in self.events() if event['hardware_action']]
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0]['error_code'],'interrupted')
        self.assertFalse(accepted[0]['action_started'])
        self.assertEqual(len(self.calls()),0)
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['hardware_actions'],1)
