import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from unittest import mock

from action_test_support import ActionTestBase
import common
import integrity
import session
import session_actions
import snapshots
import workspace


FINALIZE_PROCESS = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
import integrity, snapshots, session_actions
fault=os.environ.get('S4_FAULT')
if fault=='after_reserve':
    session_actions._run_process=lambda *a: os._exit(23)
if fault in ('before_manifest','after_manifest'):
    original=integrity._write_manifest
    def write(path,manifest):
        final=manifest['roots']['final_result']['entries']['']['kind']=='file'
        if final and fault=='before_manifest': os._exit(23)
        original(path,manifest)
        if final and fault=='after_manifest': os._exit(23)
    integrity._write_manifest=write
result=snapshots.finalize_session(os.environ['PTOBENCH_SESSION_ROOT'],'optimizer_exit')
print(json.dumps(result.to_dict()),flush=True)
sys.exit(0 if result.status=='completed' else 1)
'''


class FinalizeTest(ActionTestBase):
    def create(self, budget=2, optimizer='maintenance', case_id=None, output=None):
        # Maintenance qualification retains correctness-checked profiles and
        # their historical Snapshot/final-selection behavior.
        return super().create(budget, optimizer, case_id, output)

    def observe(self, mode=None, median=10.0, **control):
        self.configure(median=median, **control)
        process, result = self.cli(['profile', mode] if mode else ['evaluate'])
        self.assertEqual(process.returncode, 0, result)
        return result

    def finish(self, reason='optimizer_exit'):
        with mock.patch.dict(os.environ, self.env):
            return snapshots.finalize_session(self.handle.session_root, reason)

    def start_final(self, fault=None):
        env = {**self.env}
        if fault:
            env['S4_FAULT'] = fault
        process = subprocess.Popen([self.python, '-c', FINALIZE_PROCESS, str(self.bench/'evaluator')],
                    cwd=self.handle.initial_workspace,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        return process

    def final_output(self, process, timeout=30):
        stdout,stderr=process.communicate(timeout=timeout)
        value=json.loads(stdout.decode().splitlines()[-1])
        return process.returncode,value,stderr.decode()

    def next_session(self):
        self.handle=self.create(budget=20,optimizer='pi')
        self.env['PTOBENCH_SESSION_ROOT']=str(self.handle.session_root)

    def test_mixed_metrics_choose_lowest_then_earliest_without_new_execution(self):
        first=self.observe(median=40)
        best=self.observe('pmu',median=8)
        self.observe('runtime',median=None)
        self.observe(median=8)
        selected=snapshots.select_best_snapshot(self.handle.session_root)
        self.assertEqual(selected.snapshot_id,best['snapshot_id'])
        self.assertEqual((selected.observation['kind'],selected.observation['mode']),('profile','pmu'))
        self.assertEqual(selected.selection_reason,'lowest_observed_latency')
        self.assertEqual(selected.content_sha256,common.read_json(self.snapshot(first)/'metadata.json')['content_sha256'])
        lower=self.observe('compiler',median=7)
        self.assertEqual(snapshots.select_best_snapshot(self.handle.session_root).snapshot_id,lower['snapshot_id'])
        self.assertEqual(len(self.calls()),5)
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['finish_actions'],0)

    def test_all_null_or_invalid_profile_metrics_choose_latest_pass(self):
        for mode,median in [('runtime',None),('pmu',0),('compiler',-1),('pmu',True)]:
            latest=self.observe(mode,median=median)
        selected=snapshots.select_best_snapshot(self.handle.session_root)
        self.assertEqual(selected.snapshot_id,latest['snapshot_id'])
        self.assertEqual(selected.selection_reason,'latest_pass_without_latency')
        self.assertIsNone(selected.observation['latency_us'])
        self.assertIsNone(selected.observation['metric_name'])
        self.assertIsNone(selected.observation['unit'])
        self.assertEqual(len(self.calls()),4)

    def test_no_snapshot_never_falls_back_to_current_or_seed_and_is_idempotent(self):
        (self.handle.initial_workspace/'helper.py').write_text('VALUE = 999\n')
        self.assertIsNone(snapshots.select_best_snapshot(self.handle.session_root))
        result=self.finish()
        self.assertEqual((result.status,result.error_code),('failed','no_valid_snapshot'))
        self.assertEqual((result.hardware_actions_used,result.finish_actions_used),(0,0))
        self.assertIsNone(result.selected_snapshot)
        self.assertEqual(self.finish('different_reason').to_dict(),result.to_dict())
        self.assertEqual(self.calls(),[])
        self.assertEqual(self.events(),[])
        self.assertFalse((self.handle.session_root/'final/workspace').exists())

    def test_success_uses_clean_selected_source_and_new_verified_numbers(self):
        candidate=self.observe('pmu',median=2)
        self.observe(median=10)
        (self.handle.initial_workspace/'helper.py').write_text('VALUE = 900\n')
        stale=self.handle.session_root/'final/workspace'
        (stale/'.benchmark_outputs/clean').mkdir(parents=True)
        (stale/'.benchmark_outputs/clean/old').write_text('old')
        (stale/'__pycache__').mkdir(); (stale/'__pycache__/old.pyc').write_bytes(b'old')
        inspected=self.repo/'inspection.json'
        self.configure(median=123.0,speedup=3.25,inspect_before=str(inspected),generation=2)
        result=self.finish('optimizer_nonzero_exit')
        self.assertEqual(result.status,'completed',result)
        self.assertEqual(result.selected_snapshot,candidate['snapshot_id'])
        self.assertEqual(result.best_observed_latency_us,2)
        self.assertEqual((result.best_observed_source,result.best_observed_mode),('profile','pmu'))
        self.assertEqual((result.final_verified_latency_us,result.final_verified_speedup),(123.0,3.25))
        self.assertEqual(result.final_verified_correctness,'passed')
        self.assertEqual((result.hardware_actions_used,result.finish_actions_used),(2,1))
        self.assertEqual(result.close_reason,'optimizer_nonzero_exit')
        self.assertEqual(common.read_json(inspected),{'outputs':False,'cache':False,'helper':'VALUE = 1\n','context_config':'SIZE = 2\n'})
        self.assertEqual((stale/'helper.py').read_text(),'VALUE = 1\n')
        self.assertEqual((self.handle.initial_workspace/'helper.py').read_text(),'VALUE = 900\n')
        self.assertEqual(self.calls()[-1]['phase'],'all'); self.assertEqual(self.calls()[-1]['profile_mode'],'clean')
        self.assertEqual(self.calls()[-1]['measurement_mode'],'formal')
        self.assertEqual(self.calls()[-1]['candidate'],str(stale))
        self.assertIsNone(self.events()[-1]['snapshot_id'])
        self.assertEqual(len(list(self.handle.snapshots_root.iterdir())),2)
        self.assertEqual(common.read_json(self.handle.session_root/'final/authority_result.json')['runs'][0]['speedup'],3.25)
        self.assertTrue(integrity.check_only(self.handle.protection_root).passed)

    def test_all_null_selection_still_receives_one_clean_finish(self):
        self.observe('runtime',median=None)
        latest=self.observe('compiler',median=None)
        self.configure(median=17,speedup=4)
        result=self.finish()
        self.assertEqual(result.selected_snapshot,latest['snapshot_id'])
        self.assertIsNone(result.best_observed_latency_us)
        self.assertEqual(result.final_verified_latency_us,17)
        self.assertEqual(len(self.calls()),3)

    def test_history_source_metadata_events_and_authority_are_restored(self):
        original=self.observe(median=11)
        snap=self.snapshot(original)
        sentinel=self.repo/'sentinel'; sentinel.write_text('outside')
        (snap/'source').chmod(0o755)
        (snap/'source/helper.py').unlink(); (snap/'source/helper.py').symlink_to(sentinel)
        metadata=snap/'metadata.json'; metadata.chmod(0o600); metadata.write_text('{}')
        (self.handle.session_root/'events.jsonl').write_text('[]\n')
        (self.handle.session_root/'staging/action_000001/authority_result.json').write_text('{}')
        selected=snapshots.select_best_snapshot(self.handle.session_root)
        self.assertEqual(selected.snapshot_id,original['snapshot_id'])
        self.assertEqual(sentinel.read_text(),'outside')
        self.assertEqual((snap/'source/helper.py').read_text(),'VALUE = 1\n')
        self.assertEqual((snap/'source').stat().st_mode & 0o777,0o555)
        self.configure()
        self.assertEqual(self.finish().status,'completed')
        self.assertEqual(len(self.calls()),2)

    def test_corrupt_history_recovery_blocks_finish_without_charge(self):
        original=self.observe()
        manifest=common.read_json(self.handle.protection_root/'manifest.json')
        entry=manifest['roots'][original['snapshot_id']]['entries']['source/helper.py']
        backup=self.handle.protection_root/entry['backup']; backup.chmod(0o600); backup.write_text('corrupt')
        source=self.snapshot(original)/'source/helper.py'; source.chmod(0o600); source.write_text('needs recovery')
        with self.assertRaises(integrity.InfrastructureError):
            snapshots.select_best_snapshot(self.handle.session_root)
        result=self.finish()
        self.assertEqual((result.status,result.error_code),('failed','infrastructure_error'))
        self.assertEqual(result.finish_actions_used,0)
        self.assertEqual(len(self.calls()),1)
        self.assertTrue(session.read_session(self.handle.session_root)['exploration_closed'])

    def test_unregistered_orphan_never_enters_selection(self):
        original=self.observe(median=12)
        orphan=self.handle.snapshots_root/'snapshot_999999'
        shutil.copytree(self.snapshot(original),orphan)
        metadata=orphan/'metadata.json'; metadata.chmod(0o600)
        value=common.read_json(metadata); value['observation']['latency_us']=.001; metadata.write_text(json.dumps(value))
        self.assertEqual(snapshots.select_best_snapshot(self.handle.session_root).snapshot_id,original['snapshot_id'])
        self.assertEqual(len(self.calls()),1)

    def test_inconsistent_committed_history_is_rejected_before_finish(self):
        self.observe()
        with session.execution_lock(),session._session_lock(self.handle.session_root):
            record,events=session._load_locked(self.handle.session_root)
            events[0]['observation']['unit']='ms'
            session._commit(self.handle.session_root,record,events)
        result=self.finish()
        self.assertEqual(result.error_code,'infrastructure_error')
        self.assertEqual(len(self.calls()),1)
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['finish_actions'],0)

    def test_clean_failure_never_uses_observed_score_or_tries_second_place(self):
        selected=self.observe(median=1)
        self.observe(median=2)
        self.configure(fail=True)
        result=self.finish()
        self.assertEqual((result.status,result.error_code),('failed','action_failed'))
        self.assertEqual(result.selected_snapshot,selected['snapshot_id'])
        self.assertEqual(result.best_observed_latency_us,1)
        self.assertIsNone(result.final_verified_latency_us); self.assertIsNone(result.final_verified_speedup)
        self.assertIsNone(result.final_verified_correctness)
        self.assertEqual(result.finish_actions_used,1)
        self.assertEqual(common.read_json(self.handle.session_root/'final/authority_result.json')['verdict'],'FAIL')
        self.configure()
        self.assertEqual(self.finish().to_dict(),result.to_dict())
        self.assertEqual(len(self.calls()),3); self.assertEqual(len(self.events()),3)

    def test_bad_or_missing_final_results_are_failed_without_replay(self):
        for control in ({'missing':True},{'bad_case':True},{'latency':'nan'},{'nonzero':True}):
            with self.subTest(control=control):
                self.next_session(); self.observe()
                before=len(self.calls()); self.configure(**control)
                result=self.finish()
                self.assertEqual(result.error_code,'action_failed')
                self.assertIsNone(result.final_verified_latency_us)
                self.assertEqual(result.finish_actions_used,1)
                self.configure(); self.assertEqual(self.finish().to_dict(),result.to_dict())
                self.assertEqual(len(self.calls()),before+1)

    def test_final_integrity_changes_fail_even_when_evaluator_passes(self):
        for kind in ('source','context','record'):
            with self.subTest(kind=kind):
                self.next_session(); self.observe()
                self.configure(tamper=kind,session_root=str(self.handle.session_root))
                result=self.finish()
                self.assertEqual(result.error_code,'integrity_failed')
                self.assertFalse(result.final_integrity_passed)
                self.assertIsNone(result.final_verified_latency_us)
                self.assertEqual(result.finish_actions_used,1)
                self.assertEqual(len(list(self.handle.snapshots_root.iterdir())),1)

    def test_preparation_failure_is_terminal_idempotent_and_costs_no_finish(self):
        observed=self.observe()
        with mock.patch.object(workspace,'write_context',side_effect=OSError('context fault')):
            result=self.finish()
        self.assertEqual(result.error_code,'infrastructure_error')
        self.assertEqual(result.selected_snapshot,observed['snapshot_id'])
        self.assertEqual(result.finish_actions_used,0)
        self.assertEqual(self.finish().to_dict(),result.to_dict())
        self.assertEqual(len(self.calls()),1)

    def test_committed_final_and_authority_are_restored_without_reading_diagnostics(self):
        self.observe()
        self.configure(median=19,speedup=7)
        result=self.finish()
        self.assertEqual(result.status,'completed',result)
        sentinel=self.repo/'sentinel'; sentinel.write_text('outside')
        final=self.handle.session_root/'final_result.json'; final.unlink(); final.symlink_to(sentinel)
        (self.handle.session_root/'final/authority_result.json').write_text('{}')
        (self.handle.session_root/'final/result.json').write_text('{"median_effective_us":99999}')
        (self.handle.session_root/'events.jsonl').write_text('[]\n')
        self.configure(fail=True)
        repeated=self.finish('changed_reason')
        self.assertEqual(repeated.to_dict(),result.to_dict())
        self.assertEqual(sentinel.read_text(),'outside')
        self.assertEqual(len(self.calls()),2); self.assertEqual(len(self.events()),2)

    def test_invalid_committed_final_metric_is_rejected_without_measurement(self):
        self.observe(); self.configure(median=1)
        original=self.finish(); self.assertEqual(original.status,'completed',original)
        malformed=original.to_dict(); malformed['final_verified_latency_us']=True
        with session.execution_lock(),session._session_lock(self.handle.session_root):
            session._load_locked(self.handle.session_root)
            integrity.commit_records(self.handle.protection_root,{'final_result':session._bytes(malformed)})
        result=self.finish()
        self.assertEqual((result.status,result.error_code),('failed','infrastructure_error'))
        self.assertIsNone(result.final_verified_latency_us)
        self.assertEqual(session.read_session(self.handle.session_root)['status'],'failed')
        self.assertEqual(self.finish().to_dict(),result.to_dict())
        self.assertEqual(len(self.calls()),2); self.assertEqual(len(self.events()),2)

    def test_concurrent_finalize_processes_commit_one_finish(self):
        self.observe(); self.configure(median=29)
        processes=[self.start_final() for _ in range(3)]
        results=[self.final_output(process) for process in processes]
        self.assertTrue(all(code==0 for code,_,_ in results),results)
        self.assertTrue(all(value==results[0][1] for _,value,_ in results))
        self.assertEqual(len(self.calls()),2); self.assertEqual(len(self.events()),2)
        self.assertEqual(session.read_session(self.handle.session_root)['usage']['finish_actions'],1)

    def test_sigterm_finish_stops_group_and_publishes_one_failed_result(self):
        self.observe()
        ready,child_pid=self.repo/'ready',self.repo/'child_pid'
        self.configure(hang=True,child=True,ready=str(ready),child_pid=str(child_pid))
        process=self.start_final(); self.wait_file(ready,process)
        child=int(child_pid.read_text()); process.send_signal(signal.SIGTERM)
        code,result,error=self.final_output(process)
        self.assertEqual((code,result['error_code']),(1,'interrupted'),error)
        self.assertIsNone(session.process_identity(child))
        self.assertIsNone(result['final_verified_latency_us'])
        self.assertEqual(result['finish_actions_used'],1)
        self.configure(); self.assertEqual(self.finish().to_dict(),result)
        self.assertEqual(len(self.calls()),2); self.assertEqual(len(self.events()),2)

    def test_killed_finish_owner_recovers_final_process_group_without_remeasurement(self):
        self.observe()
        ready,child_pid=self.repo/'ready',self.repo/'child_pid'
        self.configure(hang=True,child=True,ready=str(ready),child_pid=str(child_pid))
        process=self.start_final(); self.wait_file(ready,process)
        marker=common.read_json(self.bench/'.ptobench/active_action.json')
        child=int(child_pid.read_text()); process.kill(); process.communicate(timeout=10)
        self.configure()
        result=self.finish()
        self.assertEqual(result.error_code,'interrupted',result)
        self.assertIsNone(session.process_identity(marker['process']['pid']))
        self.assertIsNone(session.process_identity(child))
        self.assertEqual(result.finish_actions_used,1)
        self.assertIsNone(result.final_verified_latency_us)
        self.assertEqual(self.finish().to_dict(),result.to_dict())
        self.assertEqual(len(self.calls()),2); self.assertEqual(len(self.events()),2)

    def test_finish_reservation_crash_before_marker_never_replays(self):
        self.observe(); self.configure()
        process=self.start_final('after_reserve'); process.communicate(timeout=20)
        self.assertEqual(process.returncode,23)
        result=self.finish()
        self.assertEqual(result.error_code,'interrupted')
        self.assertEqual(result.finish_actions_used,1)
        self.assertEqual(len(self.calls()),1)
        self.assertEqual(len(self.events()),2)
        self.assertEqual(self.finish().to_dict(),result.to_dict())

    def test_terminal_transaction_crashes_restore_all_old_or_all_new(self):
        for boundary,expected in [('before_manifest','failed'),('after_manifest','completed')]:
            with self.subTest(boundary=boundary):
                self.next_session(); self.observe()
                self.configure(median=31,speedup=5)
                before=len(self.calls())
                process=self.start_final(boundary); process.communicate(timeout=20)
                self.assertEqual(process.returncode,23)
                result=self.finish()
                self.assertEqual(result.status,expected,result)
                self.assertEqual(result.final_verified_latency_us,31 if expected=='completed' else None)
                self.assertEqual(result.finish_actions_used,1)
                self.assertEqual(len(self.calls()),before+1)
                self.assertEqual(len(self.events()),2)
                self.assertEqual(self.finish().to_dict(),result.to_dict())

    def test_failed_infrastructure_session_never_consumes_finish(self):
        self.observe()
        with session.execution_lock():
            session.fail_session(self.handle.session_root,'infrastructure_failure')
        result=self.finish()
        self.assertEqual(result.error_code,'infrastructure_error')
        self.assertEqual(result.finish_actions_used,0)
        self.assertEqual(len(self.calls()),1)

    def test_last_exhausted_action_finishes_before_independent_finish(self):
        self.handle=self.create(budget=1,optimizer='pi')
        self.env['PTOBENCH_SESSION_ROOT']=str(self.handle.session_root)
        ready,release=self.repo/'ready',self.repo/'release'
        self.configure(pause=True,ready=str(ready),release=str(release),median=40)
        exploring=self.start_cli(['evaluate']); self.wait_file(ready,exploring)
        finishing=self.start_final()
        deadline=time.monotonic()+5
        while not session.read_session(self.handle.session_root)['exploration_closed'] and time.monotonic()<deadline:
            time.sleep(.01)
        record=session.read_session(self.handle.session_root)
        self.assertEqual(record['status'],'exhausted')
        self.assertTrue(record['exploration_closed'])
        self.assertIsNotNone(record['active_action'])
        self.assertEqual(record['usage'],{'hardware_actions':1,'finish_actions':0})
        self.configure(median=50); release.write_text('go')
        exploring.communicate(timeout=30)
        self.assertEqual(exploring.returncode,0)
        code,result,error=self.final_output(finishing)
        self.assertEqual((code,result['status']),(0,'completed'),error)
        self.assertEqual(result['best_observed_latency_us'],40)
        self.assertEqual(result['final_verified_latency_us'],50)
        self.assertEqual((result['hardware_actions_used'],result['finish_actions_used']),(1,1))
        self.assertEqual(len(self.calls()),2); self.assertEqual(len(self.events()),2)
