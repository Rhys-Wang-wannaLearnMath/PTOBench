"""Archive S4 fake CPU finalization evidence, without real Case execution."""
import json
import os
from pathlib import Path
import shutil
import sys
from unittest import mock

repo=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(repo/'benchmarks/pypto_lib_bench-v0.2/evaluator/tests'))
from action_test_support import ActionTestBase
import snapshots

destination=Path(__file__).parent/'fake_finalizations'
destination.mkdir()  # Never overwrite an earlier capture.
fixture=ActionTestBase('runTest')

def archive(label,handle,result,repeated):
    target=destination/label; target.mkdir()
    for name in ('session.json','events.jsonl','final_result.json'):
        shutil.copyfile(handle.session_root/name,target/name)
    shutil.copytree(handle.session_root/'final',target/'final')
    shutil.copytree(handle.snapshots_root,target/'snapshots')
    (target/'returned.json').write_text(json.dumps(result.to_dict(),indent=2)+'\n')
    assert result.to_dict()==repeated.to_dict()

try:
    fixture.setUp()
    for argv,median in [(['evaluate'],20),(['profile','pmu'],3),(['profile','compiler'],None)]:
        fixture.configure(median=median)
        process,result=fixture.cli(argv)
        assert process.returncode==0,result
    (fixture.handle.initial_workspace/'helper.py').write_text('VALUE = 777\n')
    fixture.configure(median=55,speedup=4)
    with mock.patch.dict(os.environ,fixture.env):
        selected=snapshots.select_best_snapshot(fixture.handle.session_root)
        result=snapshots.finalize_session(fixture.handle.session_root,'optimizer_nonzero_exit')
        repeated=snapshots.finalize_session(fixture.handle.session_root,'changed_reason')
    assert result.status=='completed' and result.best_observed_latency_us==3 and result.final_verified_latency_us==55
    archive('completed',fixture.handle,result,repeated)
    (destination/'completed/selection.json').write_text(json.dumps(selected.to_dict(),indent=2)+'\n')

    empty=fixture.create(optimizer='pi')
    result=snapshots.finalize_session(empty.session_root,'optimizer_exit')
    repeated=snapshots.finalize_session(empty.session_root,'again')
    assert result.error_code=='no_valid_snapshot' and result.finish_actions_used==0
    archive('no_candidate',empty,result,repeated)

    failed=fixture.create(optimizer='opencode')
    fixture.env['PTOBENCH_SESSION_ROOT']=str(failed.session_root)
    fixture.configure()
    process,result=fixture.cli(['profile','runtime',str(failed.initial_workspace)])
    assert process.returncode==0,result
    fixture.configure(fail=True)
    with mock.patch.dict(os.environ,fixture.env):
        result=snapshots.finalize_session(failed.session_root,'optimizer_exit')
        repeated=snapshots.finalize_session(failed.session_root,'again')
    assert result.error_code=='action_failed' and result.finish_actions_used==1 and result.final_verified_latency_us is None
    archive('clean_failure',failed,result,repeated)
    shutil.copyfile(fixture.counter,destination/'evaluator_calls.jsonl')
    assert len(fixture.calls())==6,fixture.calls()
    print('S4 fake CPU evidence: completed, no_candidate, clean_failure; 6 Evaluator calls total; reentries add none')
finally:
    fixture.doCleanups()
