"""S7 control tests: real Sessions/CLI, tiny assets, controlled CPU Evaluator."""
import contextlib
import csv
import io
import json
from pathlib import Path
from unittest import mock

from action_test_support import ActionTestBase, FAKE_EVALUATOR
import acceptance_artifacts
import accept_all_cases as acceptance
import profile_modes
import session


ARTIFACTS = r'''
if not control.get('skip_artifacts'):
    d=artifacts/'build_output'/'fake_kernel'/'dfx_outputs'
    d.mkdir(parents=True,exist_ok=True)
    def put(name,value):
        (d/name).write_text(json.dumps(value)+'\n')
    if a.profile_mode=='runtime':
        put('chip_swimlane_records.json',{'aicore_tasks':[{}],'aicpu_tasks':[],
            'aicpu_orchestrator_phases':[],'aicpu_scheduler_phases':[],
            'chip_swimlane_level':1,'metadata':{}})
        put('deps.json',{'tasks':[{}],'edges':[],'tensors':[]})
        put('merged_swimlane_1.json',{'traceEvents':[{'name':'task'}]})
        header={'fatal':False,'dropped':0,'total':2,'heap_max':[100],
            'dep_pool_max':[100],'task_window_max':[100],'tensormap_max':100}
        event={k:0 for k in ('depth','ring','heap_start','heap_end','dep_pool_start',
            'dep_pool_end','task_window_start','task_window_end','tensormap')}
        (d/'scope_stats.jsonl').write_text(json.dumps(header)+'\n'+
            '\n'.join(json.dumps(dict(event,site='kernel.cpp:1',phase=p)) for p in ('begin','end'))+'\n')
    elif a.profile_mode=='hardware':
        columns=('thread_id','core_id','task_id','func_id','core_type','pmu_total_cycles',
            'vec_busy_cycles','cube_busy_cycles','scalar_busy_cycles','mte1_busy_cycles',
            'mte2_busy_cycles','mte3_busy_cycles','icache_miss','icache_req','event_type')
        (d/'pmu.csv').write_text(','.join(columns)+'\n'+','.join('1' for _ in columns)+'\n')
    elif a.profile_mode=='compile':
        put('pipeline_profile.json',{'stages':[{'name':'compile'}],'total_seconds':0.01})
        (d/'00_frontend.py').write_text('IR frontend\n')
        (d/'01_after_Compile.py').write_text('IR compiled\n')
if control.get('unsupported'):
    print('ProfileNotSupportedError: pmu profiling is not supported by this test backend',file=sys.stderr)
    control['fail']=True
'''


class FullAcceptanceTest(ActionTestBase):
    def setUp(self):
        super().setUp()
        source = FAKE_EVALUATOR.replace(
            "control=json.loads(Path(os.environ['S3_CONTROL']).read_text())",
            "control=json.loads(Path(os.environ['S3_CONTROL']).read_text())\n"
            "control.update(control.get('modes',{}).get(a.profile_mode,{}))")
        source = source.replace("run={'fixture_id'", ARTIFACTS + "\nrun={'fixture_id'")
        source = source.replace("result={'schema_version':1", "result={'artifact_dir':str(artifacts),"
                                "'build_dir':str(artifacts/'build_output'),'schema_version':1")
        source = source.replace("Path(a.result).write_text", "if control.get('unsupported'): result['runs']=[]\nPath(a.result).write_text")
        (self.bench / 'evaluator/evaluate.py').write_text(source)
        patcher = mock.patch.dict('os.environ', self.env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def batch(self, **kwargs):
        self.runner = acceptance.AcceptanceRunner(self.repo / 'batch', **kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.runner.run()
        self.summary = json.loads((self.repo / 'batch/summary.json').read_text())
        return code, self.runner.rows

    def test_all_cases_pending_independent_profiles_and_matrix(self):
        _, sealed = self.make_case('test.pending')
        cert = json.loads((sealed / 'certification.json').read_text())
        cert['status'] = 'pending'
        (sealed / 'certification.json').write_text(json.dumps(cert))
        code, rows = self.batch()
        self.assertEqual(code, 1)
        self.assertEqual(self.summary['case_count'], 2)
        self.assertEqual(len(rows), 8)
        self.assertTrue(self.summary['recording_complete'])
        self.assertTrue(self.summary['canonical_integrity']['passed'])
        good = [r for r in rows if r['case_id'] == 'test.case']
        self.assertEqual([r['status'] for r in good], ['PASS']*4)
        self.assertEqual([r['correctness'] for r in good], ['passed']*4)
        self.assertTrue(all(r['supported'] is True and r['latency_us'] is None for r in good[1:]))
        self.assertEqual([r['status'] for r in rows if r['case_id'] == 'test.pending'], ['BLOCKED']*4)
        self.assertEqual(self.runner.finals[0]['value']['finish_actions_used'], 1)
        with (self.repo / 'batch/matrix.csv').open(newline='') as stream:
            matrix = list(csv.DictReader(stream))
        self.assertEqual(len(matrix), 2)
        self.assertEqual(matrix[0]['compiler_supported'], 'True')
        self.assertEqual(len(self.calls()), 5)  # four real CLI dispatches and separate finish

    def test_complete_success_returns_zero_and_four_same_seed_workers(self):
        code, rows = self.batch()
        self.assertEqual(code, 0)
        self.assertTrue(self.summary['all_pass'])
        self.assertEqual(len({r['source_hash'] for r in rows}), 1)
        record = session.read_session(rows[0]['session_root'])
        self.assertEqual(len(record['workspaces']), 4)
        self.assertEqual(record['usage'], {'hardware_actions': 4, 'finish_actions': 1})

    def test_clean_failure_does_not_replace_profile_correctness(self):
        self.configure(modes={'clean': {'fail': True}, 'hardware': {'skip_artifacts': True}})
        code, rows = self.batch()
        self.assertEqual(code, 1)
        self.assertEqual([r['status'] for r in rows], ['FAIL', 'PASS', 'FAIL', 'PASS'])
        self.assertEqual(rows[0]['correctness'], 'failed')
        self.assertIsNotNone(rows[0]['artifact_check'])
        self.assertEqual(rows[2]['correctness'], 'passed')
        self.assertIsNone(rows[2]['supported'])
        self.assertEqual(rows[2]['reason'], 'artifact_missing_or_invalid')

    def test_explicit_unsupported_distinct_from_generic_failure(self):
        self.configure(modes={'hardware': {'unsupported': True}, 'compile': {'crash': True}})
        _, rows = self.batch()
        self.assertEqual(rows[2]['status'], 'UNSUPPORTED')
        self.assertIs(rows[2]['supported'], False)
        self.assertEqual(rows[3]['status'], 'FAIL')
        self.assertIsNone(rows[3]['supported'])
        self.assertIsNone(acceptance.unsupported_evidence({}, 'pmu', 'ValueError: unsupported dtype'))

    def test_device_blocked_preserves_four_records_without_dispatch(self):
        _, rows = self.batch(device_probe=lambda env, directory: ['device 1 missing, unavailable'])
        self.assertEqual([r['status'] for r in rows], ['BLOCKED']*4)
        self.assertEqual(self.calls(), [])
        self.assertTrue(all('device 1 missing' in r['reason'] for r in rows))
        with (self.repo / 'batch/matrix.csv').open(newline='') as stream:
            row = next(csv.DictReader(stream))
        self.assertIn('missing, unavailable', row['clean_reason'])

    def test_invalid_duplicate_and_unparseable_ids_are_not_lost(self):
        case_dir, _ = self.make_case('test.other')
        value = json.loads((case_dir / 'case.json').read_text())
        value['case_id'] = 'test.case'
        (case_dir / 'case.json').write_text(json.dumps(value))
        bad = self.bench / 'cases/test/bad/case.json'
        bad.parent.mkdir(parents=True)
        bad.write_text('{broken')
        code, rows = self.batch()
        self.assertEqual(code, 1)
        self.assertEqual(self.summary['case_count'], 3)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r['status'] == 'BLOCKED' for r in rows))
        self.assertEqual(self.calls(), [])

    def test_disabled_mode_is_blocked_without_changing_global_profile(self):
        with mock.patch.object(profile_modes, 'enabled_modes', return_value={
                'runtime': {'description': 'runtime'}, 'compile': {'description': 'compiler'}}):
            _, rows = self.batch()
        self.assertEqual(rows[2]['status'], 'BLOCKED')
        self.assertEqual(rows[2]['reason'], 'mode_disabled')
        self.assertNotIn('hardware', [r['profile_mode'] for r in self.calls()])
        inventory = json.loads((self.repo / 'batch/inventory.json').read_text())
        self.assertEqual(inventory['profiling_open'], ['runtime', 'compiler'])

    def test_multidevice_preflight_queries_every_requested_device(self):
        env = {'python': str(Path(__import__('sys').executable)), 'runtime': {'device_ids': [6, 7]}}
        with mock.patch.object(acceptance.subprocess, 'run') as run:
            run.return_value = mock.Mock(returncode=0, stdout=b'OK', stderr=b'')
            self.assertEqual(acceptance.device_preflight(env, self.repo / 'devices'), [])
            self.assertEqual([call.args[0][-1] for call in run.call_args_list], ['6', '7'])

    def test_bad_environment_remains_four_finite_records(self):
        path = self.bench / 'environments/cpu_test.json'
        # Use the actual profile named in this temporary Case.
        value = json.loads((self.bench / 'cases/test/case/case.json').read_text())
        path = self.bench / 'environments' / (value['environment']+'.json')
        path.write_text('[]')
        _, rows = self.batch()
        self.assertEqual([r['status'] for r in rows], ['BLOCKED']*4)
        self.assertTrue(self.summary['recording_complete'])
        self.assertEqual(json.loads((self.repo / 'batch/finalizations.json').read_text()), [])

    def test_timeout_cleanup_stops_children_before_next_action(self):
        self.configure(modes={'clean': {'hang': True, 'ready': str(self.repo / 'hanging')}})
        _, rows = self.batch(timeout=2)
        self.assertEqual(rows[0]['status'], 'TIMEOUT')
        self.assertTrue(self.summary['recording_complete'])
        self.assertIsNone(self.summary['stop_reason'])
        record = session.read_session(rows[0]['session_root'])
        self.assertIsNone(record['active_action'])
        self.assertFalse((self.bench / '.ptobench/active_action.json').exists())

    def test_integrity_failure_cannot_pass_and_current_candidate_is_not_repaired(self):
        self.configure(modes={'runtime': {'tamper': 'source'}})
        _, rows = self.batch()
        self.assertEqual(rows[1]['status'], 'FAIL')
        self.assertIs(rows[1]['integrity_passed'], False)
        self.assertEqual(rows[1]['error_code'], 'integrity_failed')
        self.assertTrue(self.summary['canonical_integrity']['passed'])

    def test_existing_output_is_never_overwritten(self):
        target = self.repo / 'existing'
        target.mkdir()
        sentinel = target / 'keep'
        sentinel.write_text('original')
        with self.assertRaises(FileExistsError):
            acceptance.AcceptanceRunner(target).run()
        self.assertEqual(sentinel.read_text(), 'original')

    def test_public_code_changed_after_last_case_invalidates_batch(self):
        original = acceptance.AcceptanceRunner.run_case
        def change_after_final(runner, case, index):
            original(runner, case, index)
            (runner.bench / 'evaluator/late_change.txt').write_text('changed after last action')
        with mock.patch.object(acceptance.AcceptanceRunner, 'run_case', change_after_final):
            code, rows = self.batch()
        self.assertEqual([r['status'] for r in rows], ['PASS']*4)
        self.assertEqual(code, 1)
        self.assertFalse(self.summary['public_code_integrity']['passed'])
        self.assertTrue(self.summary['canonical_integrity']['passed'])

    def test_hash_freeze_rejects_links_and_reports_read_races(self):
        target = self.repo / 'linked'
        target.symlink_to(self.bench / 'cases')
        self.assertTrue(acceptance.hash_roots({'linked': target})['errors'])
        with mock.patch.object(acceptance.file_tree, 'scan_tree', side_effect=RuntimeError('read race')):
            self.assertIn('read race', acceptance.hash_roots({'cases': self.bench / 'cases'})['errors'][0]['error'])


    def test_native_schema_requires_rows_and_actual_ir(self):
        _, rows = self.batch()
        pmu = Path(rows[2]['artifact_check']['root'])
        report = next(pmu.rglob('pmu.csv'))
        report.write_text(report.read_text().splitlines()[0]+'\n')
        self.assertFalse(acceptance_artifacts.inspect_artifacts(pmu, 'pmu')['passed'])
        compiler = Path(rows[3]['artifact_check']['root'])
        for path in compiler.rglob('*_after_*.py'):
            path.unlink()
        self.assertFalse(acceptance_artifacts.inspect_artifacts(compiler, 'compiler')['passed'])
        runtime = Path(rows[1]['artifact_check']['root'])
        scope = next(runtime.rglob('scope_stats.jsonl'))
        data = [json.loads(line) for line in scope.read_text().splitlines()]
        self.assertTrue(acceptance_artifacts.inspect_artifacts(runtime, 'runtime')['passed'])
        data[0]['total'] += 1
        scope.write_text('\n'.join(json.dumps(row) for row in data))
        self.assertFalse(acceptance_artifacts.inspect_artifacts(runtime, 'runtime')['passed'])
