"""Serial canonical-Candidate collection through the normal PTOBench Profile CLI.

Session/Event records are authoritative; stdout is only a collection progress log.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
BENCH=Path(__file__).resolve().parents[5]
RUNS=Path(os.environ.get('PTO_STAGE5_PROFILE_RUNS', str(BENCH/'runs'/'stage5_profile_fill')))
SNAPS=Path(os.environ.get('PTO_STAGE5_PROFILE_SNAPSHOTS', str(BENCH/'snapshots'/'stage5_profile_fill')))

def expected_devices(case_id):
    fam,name=case_id.split('.',1)
    case=json.loads((BENCH/'cases'/fam/name/'case.json').read_text())
    env=json.loads((BENCH/'environments'/f"{case['environment']}.json").read_text())
    rt={**env.get('runtime',{}),**case.get('runtime',{})}
    return list(rt.get('device_ids') or ([rt['device']] if rt.get('device') is not None else [0]))

def run_one(case_id, index):
    sys.path.insert(0,str(BENCH/'evaluator')); sys.path.insert(0,str(BENCH))
    import session
    root=None
    try:
        handle=session.create_session(case_id, {'id':'stage5_profile_fill','backend':'pi','profile':'default'}, {'runs_root':str(RUNS),'snapshots_root':str(SNAPS)}, {'max_hardware_actions':1,'finish_actions':1}, mode='basic', profile_strategy='combined')
        root=Path(handle.session_root); workspace=Path(handle.initial_workspace)
        env=dict(os.environ)
        env.update({
          'PATH':os.pathsep.join((str(BENCH.parents[1]/'.venv-official/bin'),str(BENCH),env.get('PATH',''))),
          'PTOBENCH_SESSION_ROOT':str(root),'PTOBENCH_SESSION_ID':root.name,
          'PTOBENCH_ROOT':str(BENCH),'PTOBENCH_REPO_ROOT':str(BENCH.parents[1]),
          'PYTHONPATH':str(BENCH/'evaluator')+os.pathsep+str(BENCH),
          'PTOBENCH_CANN_ROOT':env.get('PTOBENCH_CANN_ROOT','/usr/local/Ascend/cann-9.0.0/cann'),
          'PTOBENCH_PTOAS_ROOT':env.get('PTOBENCH_PTOAS_ROOT','/data1/home/ziruiwang/ptoas-v0.60'),
          'PTOBENCH_MINIFORGE_ROOT':env.get('PTOBENCH_MINIFORGE_ROOT','/data1/home/ziruiwang/miniforge3'),
        })
        log=Path('/tmp')/f'stage5_profile_{index:03d}_{case_id.replace(".","__")}.log'
        print(json.dumps({'starting':case_id,'session_root':str(root),'devices':expected_devices(case_id),'log':str(log)}),flush=True)
        with log.open('w') as stream:
            cp=subprocess.run([str(BENCH/'ptobench'),'profile','.'],cwd=workspace,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
        rows=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines() if x.strip()]
        ev=rows[-1] if rows else {}
        result=ev.get('result') or {}
        print(json.dumps({'finished':case_id,'return_code':cp.returncode,'session_root':str(root),'event_seq':ev.get('seq'),'state':ev.get('state'),'mode':ev.get('mode'),'compile_status':ev.get('compile_status'),'runtime_status':ev.get('runtime_status'),'error_code':ev.get('error_code'),'content_sha256':ev.get('content_sha256'),'devices':expected_devices(case_id),'artifact_index':str(root/(ev.get('profile_artifacts_path',''))/'index.json')},sort_keys=True),flush=True)
        return cp.returncode,root,ev
    except Exception as e:
        print(json.dumps({'failed':case_id,'error':f'{type(e).__name__}: {e}','session_root':str(root) if root else None}),flush=True)
        return 2,root,None

if __name__=='__main__':
    cases=sys.argv[1:]
    if not cases: raise SystemExit('usage: script CASE ...')
    rc=0
    for i,c in enumerate(cases,1):
        r,_,_=run_one(c,i); rc=max(rc,r)
    raise SystemExit(rc)
