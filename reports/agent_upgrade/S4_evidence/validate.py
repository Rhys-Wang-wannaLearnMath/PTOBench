"""Read-only S4 requirement and preservation audit; only writes its summary."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

repo=Path(__file__).resolve().parents[5]
bench=repo/'benchmarks/pypto_lib_bench-v0.2'
evidence=Path(__file__).parent
before=json.loads((evidence/'before.json').read_text())
preserved=json.loads((evidence/'preservation.json').read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def functions(path):
    return {node.name:ast.dump(node,include_attributes=False)
            for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}

allowed={str((bench/name).relative_to(repo)) for name in (
    'README.md','evaluator/README.md','evaluator/SESSION.md','evaluator/AGENT_CLI.md',
    'evaluator/session.py','evaluator/session_actions.py','evaluator/snapshots.py',
    'evaluator/tests/action_test_support.py')}
assert preserved['index_unchanged'] and not preserved['tracked_changed_paths']
assert set(preserved['previous_public_and_history_changed_paths'])<=allowed
history={name:value for name,value in before['preserved'].items() if '/reports/agent_upgrade/' in name}
assert all(digest(repo/name)==value for name,value in history.items())
unchanged=('integrity.py','file_tree.py','workspace.py','common.py','evaluate.py','prepare_ref.py',
           'generate_fixtures.py','workflow.py','profile_modes.py','agent_cli.py')
for name in unchanged:
    path=bench/'evaluator'/name
    assert digest(path)==before['preserved'][str(path.relative_to(repo))],name
assert digest(bench/'ptobench')==before['preserved'][str((bench/'ptobench').relative_to(repo))]
old=functions(evidence/'session_actions.py.before')
new=functions(bench/'evaluator/session_actions.py')
shared=('_stop_group','_run_process','_postcheck_action','validate_result','_signals')
assert all(old[name]==new[name] for name in shared)
assert functions(evidence/'snapshots.py.before')['create_snapshot']==functions(bench/'evaluator/snapshots.py')['create_snapshot']
assert {'select_best_snapshot','finalize_session'}<=functions(bench/'evaluator/snapshots.py').keys()
assert not os.path.lexists(bench/'bench') and not os.path.lexists(bench/'evaluator/bench.py')
assert not (bench/'pto-launcher').exists()

log=(evidence/'unittest_first_full.log').read_text()
match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
assert match and int(match[1])==122 and '\nOK\n' in log and '\nFAILED' not in log
test_counts={}
for path in sorted((bench/'evaluator/tests').glob('test_*.py')):
    test_counts[path.name]=sum(isinstance(node,ast.FunctionDef) and node.name.startswith('test_')
                             for cls in ast.parse(path.read_text()).body if isinstance(cls,ast.ClassDef) for node in cls.body)
assert sum(test_counts.values())==122 and test_counts['test_finalize.py']==22
requirements={
    'history_validation_recovery': ['test_history_source_metadata_events_and_authority_are_restored',
        'test_corrupt_history_recovery_blocks_finish_without_charge','test_inconsistent_committed_history_is_rejected_before_finish',
        'test_unregistered_orphan_never_enters_selection'],
    'deterministic_lowest_earliest_else_latest': ['test_mixed_metrics_choose_lowest_then_earliest_without_new_execution',
        'test_all_null_or_invalid_profile_metrics_choose_latest_pass'],
    'no_candidate_no_finish_no_fallback': ['test_no_snapshot_never_falls_back_to_current_or_seed_and_is_idempotent'],
    'clean_workspace_fresh_context': ['test_success_uses_clean_selected_source_and_new_verified_numbers'],
    'one_clean_independent_finish_postcheck': ['test_last_exhausted_action_finishes_before_independent_finish',
        'test_final_integrity_changes_fail_even_when_evaluator_passes','test_all_null_selection_still_receives_one_clean_finish'],
    'only_verified_final_metrics_no_retry': ['test_clean_failure_never_uses_observed_score_or_tries_second_place',
        'test_bad_or_missing_final_results_are_failed_without_replay','test_invalid_committed_final_metric_is_rejected_without_measurement'],
    'idempotence_atomic_interruption': ['test_concurrent_finalize_processes_commit_one_finish',
        'test_committed_final_and_authority_are_restored_without_reading_diagnostics',
        'test_sigterm_finish_stops_group_and_publishes_one_failed_result',
        'test_killed_finish_owner_recovers_final_process_group_without_remeasurement',
        'test_finish_reservation_crash_before_marker_never_replays',
        'test_terminal_transaction_crashes_restore_all_old_or_all_new'],
    'infrastructure_failure_no_measurement': ['test_preparation_failure_is_terminal_idempotent_and_costs_no_finish',
        'test_failed_infrastructure_session_never_consumes_finish'],
}
for names in requirements.values():
    for name in names:
        assert f'{name} (test_finalize.FinalizeTest) ...' in log,name

archive=evidence/'fake_finalizations'
observations={}
for name in ('completed','no_candidate','clean_failure'):
    root=archive/name
    result=json.loads((root/'final_result.json').read_text())
    record=json.loads((root/'session.json').read_text())
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    assert result==json.loads((root/'returned.json').read_text())
    assert result['status']==record['status'] and record['active_action'] is None
    assert result['hardware_actions_used']==record['usage']['hardware_actions']
    assert result['finish_actions_used']==record['usage']['finish_actions']
    finishes=[event for event in events if event['action']=='finish']
    assert len(finishes)==result['finish_actions_used']
    assert all(event['snapshot_id'] is None for event in finishes)
    if name=='completed':
        assert result['selected_snapshot']=='snapshot_000002' and result['best_observed_mode']=='pmu'
        assert result['best_observed_latency_us']==3 and result['final_verified_latency_us']==55 and result['final_verified_speedup']==4
        assert result['final_integrity_passed'] and result['close_reason']=='optimizer_nonzero_exit'
        raw=json.loads((root/'final/authority_result.json').read_text())
        assert raw['phase']=='all' and raw['profiling_mode']=='clean' and raw==finishes[0]['result']
        source=root/'final/workspace'
        assert (source/'helper.py').read_text()=='VALUE = 1\n'
        assert not (source/'__pycache__').exists()
        assert (source/'.benchmark_context/config.py').read_text()=='SIZE = 2\n'
        assert (source/'.benchmark_context/starting_point/helper.py').read_text()=='VALUE = 1\n'
        manifest=json.loads((root/'final/source_manifest.json').read_text())
        selection=json.loads((root/'final/selection.json').read_text())
        assert selection['content_sha256']==manifest['content_sha256']==result['selected_content_sha256']
        files=[{'path':item['path'],'sha256':digest(source/item['path'])} for item in manifest['files']]
        encoded=json.dumps(files,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        assert hashlib.sha256(encoded).hexdigest()==manifest['content_sha256']
    else:
        assert result['status']=='failed'
        assert all(result[key] is None for key in ('final_verified_correctness','final_verified_latency_us','final_verified_speedup'))
        assert result['error_code']==('no_valid_snapshot' if name=='no_candidate' else 'action_failed')
        assert result['finish_actions_used']==(0 if name=='no_candidate' else 1)
    observations[name]={key:result[key] for key in ('status','error_code','selected_snapshot','best_observed_latency_us',
        'final_verified_latency_us','final_verified_speedup','hardware_actions_used','finish_actions_used')}
calls=[json.loads(line) for line in (archive/'evaluator_calls.jsonl').read_text().splitlines()]
assert len(calls)==6
final_calls=[call for call in calls if '/final/workspace' in call['candidate']]
assert len(final_calls)==2 and all(call['phase']=='all' and call['profile_mode']=='clean' for call in final_calls)
remaining=[]
for path in Path('/proc').iterdir():
    if path.name.isdecimal():
        try:
            command=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
        if any(re.fullmatch(rb'/tmp/s2_[^/]+/benchmarks/pypto_lib_bench-v0\.2/evaluator/evaluate\.py',arg) for arg in command):
            remaining.append(int(path.name))
assert not remaining,remaining
report=bench/'reports/agent_upgrade/S4_finalize.md'
for link in re.findall(r'\]\(([^)]+)\)',report.read_text()):
    if link!='S4_evidence/validation_summary.json':
        assert (report.parent/link).exists(),link
sources=('evaluator/snapshots.py','evaluator/finalize.py','evaluator/session.py','evaluator/session_actions.py',
    'evaluator/tests/test_finalize.py','evaluator/tests/action_test_support.py','evaluator/FINALIZE.md',
    'README.md','evaluator/README.md','evaluator/SESSION.md','evaluator/AGENT_CLI.md')
summary={'stage':'S4','status':'PASS','audit_time_utc':datetime.now(timezone.utc).isoformat(),
    'full_suite':{'exit_code':0,'tests':int(match[1]),'seconds':float(match[2]),'skipped':0},
    'existing_tests':100,'s4_tests':22,'test_counts':test_counts,'requirement_tests':requirements,
    'historical_files_checked':len(history),'historical_files_unchanged':True,'index_unchanged':True,'tracked_unchanged':True,
    's1_tools_and_measurement_files_unchanged':list(unchanged),'shared_execution_functions_ast_unchanged':list(shared),
    'snapshot_creation_ast_unchanged':True,'old_entries_absent':True,'launcher_not_implemented':True,
    'canonical_files_at_baseline':preserved['canonical_file_count'],'canonical_delta_observed':len(preserved['canonical_changed_paths']),
    'canonical_delta_groups':dict(Counter('/'.join(p.split('/')[:3]) for p in preserved['canonical_changed_paths'])),
    'canonical_delta_provenance':'Concurrent external Case work; S4 writes only public implementation/docs/reports and temporary fake assets.',
    'tensor_verification':preserved['tensor_verification'],'archived_fake_results':observations,
    'archived_evaluator_calls':6,'archived_finish_calls':2,'remaining_fake_evaluators':remaining,
    'source_sha256':{name:digest(bench/name) for name in sources},
    'evidence_sha256':{str(path.relative_to(evidence)):digest(path) for path in sorted(evidence.rglob('*'))
                      if path.is_file() and path.name not in {'validation_summary.json','validation_final.log'}},
    'report_sha256':digest(report),'real_hardware':'NOT_RUN','real_optimizer':'NOT_RUN','s5_ready':True,'s5_implemented':False}
(evidence/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({key:summary[key] for key in ('stage','status','full_suite','historical_files_checked',
    'canonical_delta_observed','remaining_fake_evaluators','old_entries_absent','s5_ready')},indent=2))
