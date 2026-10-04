from pathlib import Path
from collections import Counter
import json
b=Path(__file__).parent.parent/'full_v02/20261004_S7_01'
p=b/'results.jsonl'
rows=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
x={'records':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
   'completed_case_records':sum(n==4 for n in Counter(r['case_path'] for r in rows).values()),
   'last_records':[{k:r[k] for k in ('case_id','item','status','reason')} for r in rows[-1:]],
   'summary_ready':(b/'summary.json').exists()}
print(json.dumps(x,ensure_ascii=False,separators=(',',':')))
