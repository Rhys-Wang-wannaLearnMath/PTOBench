"""Independent, read-only audit of a completed S7 batch; saves its findings."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, sys

batch=Path(sys.argv[1]).absolute()
evidence=Path(__file__).parent
inventory=json.loads((batch/'inventory.json').read_text())
summary=json.loads((batch/'summary.json').read_text())
rows=[json.loads(l) for l in (batch/'results.jsonl').read_text().splitlines()]
issues=[]
keys=Counter((r['case_path'],r['item']) for r in rows)
expected={(c['case_path'],m) for c in inventory['cases'] for m in ('clean','runtime','pmu','compiler')}
if set(keys)!=expected or any(n!=1 for n in keys.values()):issues.append('coverage mismatch')
with (batch/'matrix.csv').open(newline='') as f:matrix=list(csv.DictReader(f))
if Counter(r['case_path'] for r in matrix)!=Counter(c['case_path'] for c in inventory['cases']):issues.append('CSV coverage mismatch')
for r in rows:
 if r['status'] not in {'PASS','FAIL','BLOCKED','UNSUPPORTED','TIMEOUT'}:issues.append('invalid status')
 if r['status']=='PASS':
  if r['correctness']!='passed' or r['integrity_passed'] is not True or not r['artifact_check']['passed']:issues.append('invalid PASS:'+r['case_id']+':'+r['item'])
  if r['item']!='clean' and r['supported'] is not True:issues.append('PASS support missing')
  if not r['source_hash'] or r['source_hash']!=next(c for c in inventory['cases'] if c['case_path']==r['case_path']).get('source_hash'):issues.append('seed mismatch')
 if r['status']!='PASS' and not r['reason']:issues.append('failure reason missing')
 if r['status']=='UNSUPPORTED' and (r['supported'] is not False or not r.get('unsupported_evidence')):issues.append('unsupported evidence missing')
 for k in ('stdout_path','stderr_path','result_path','raw_result_path'):
  if r.get(k) and not Path(r[k]).is_file():issues.append('evidence missing:'+r[k])
 if r.get('session_root') and r.get('event_seq') is not None:
  events=[json.loads(l) for l in (Path(r['session_root'])/'events.jsonl').read_text().splitlines()]
  matched=[e for e in events if e['seq']==r['event_seq']]
  if len(matched)!=1:issues.append('event missing or duplicate')
  elif matched[0]['mode']!=r['public_mode']:issues.append('event mode mismatch')
commands=[]
for p in (batch/'evidence').glob('*/*/process_result.json'):
 v=json.loads(p.read_text());commands.append(v)
 if not v.get('cleanup_confirmed'):issues.append('process cleanup unconfirmed:'+str(p))
sessions=[]
for p in (batch/'runs').glob('acceptance/*/*/session.json'):
 v=json.loads(p.read_text());sessions.append(v)
 if v.get('active_action') is not None:issues.append('active action remains:'+str(p))
 if len(v['workspaces'])!=4:issues.append('worker count:'+str(p))
 if v['usage']['hardware_actions']>4 or v['usage']['finish_actions']>1:issues.append('budget overrun')
comparison=json.loads((batch/'canonical_comparison.json').read_text())
code=json.loads((batch/'code_comparison.json').read_text())
for name,data in [('canonical',comparison),('code',code)]:
 if not data['passed']:issues.append(name+' hash mismatch')
counts={m:dict(Counter(r['status'] for r in rows if r['item']==m)) for m in ('clean','runtime','pmu','compiler')}
for m,states in counts.items():
 for state,n in states.items():
  if summary['by_item'][m][state]!=n:issues.append('summary count mismatch')
all_pass=bool(inventory['cases']) and not issues and all(r['status']=='PASS' for r in rows)
if summary['all_pass']!=all_pass:issues.append('all_pass mismatch')
report={'audit_passed':not issues,'issues':issues,'cases':len(inventory['cases']),'records':len(rows),'sessions':len(sessions),'commands':len(commands),'by_item':counts,'recording_complete':set(keys)==expected and all(n==1 for n in keys.values()),'all_pass':all_pass}
(evidence/'batch_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(0 if not issues else 1)
