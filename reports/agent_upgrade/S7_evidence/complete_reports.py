"""Wait once for the existing batch process, then archive its final reports.

No hardware actions, retries, polling or canonical/public source changes.
"""
from pathlib import Path
from collections import Counter
import csv, ctypes, errno, hashlib, json, os, select, subprocess, sys

repo=Path('/data1/home/ziruiwang/pypto-lib')
bench=repo/'benchmarks/pypto_lib_bench-v0.2'
reports=bench/'reports/agent_upgrade'
evidence=reports/'S7_evidence'
batch=reports/'full_v02/20261004_S7_01'
# pidfd pins this process, avoiding PID reuse and periodic monitoring.
try:
    if hasattr(os,'pidfd_open'):
        fd=os.pidfd_open(int(sys.argv[1]))
    else:
        # Linux aarch64/x86_64 syscall number; this Python build lacks the wrapper.
        libc=ctypes.CDLL(None,use_errno=True)
        fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(int(sys.argv[1])),ctypes.c_uint(0))
        if fd<0:
            code=ctypes.get_errno()
            if code==errno.ESRCH:raise ProcessLookupError(code,os.strerror(code))
            raise OSError(code,os.strerror(code))
except ProcessLookupError:
    fd=None
if fd is not None:
    try:
        select.select([fd],[],[])
    finally:
        os.close(fd)

if not (batch/'summary.json').is_file():
    failure={'state':'batch_terminated_without_summary','recording_complete':False,
             'batch':str(batch),'action':'Preserve existing evidence; diagnose before any new batch.'}
    (evidence/'completion_status.json').write_text(json.dumps(failure,indent=2)+'\n')
    (reports/'S7_full_v02_acceptance.md').write_text('''# S7：全部 v0.2 Case 验收

**批次进程已退出，但没有完整 summary.json，验收尚未完成。**

保留 [原始日志](S7_evidence/full_batch_01.log) 和 [逐项记录](full_v02/20261004_S7_01/results.jsonl)。没有自动重跑、修改 Case 或拼接成功记录。
''')
    p=reports/'final_acceptance.md'
    text=p.read_text().replace('S7 全量 Case 验收已启动，正在执行','S7 批次异常退出，完整验收尚未完成').replace('RUNNING：198 个 Case','INCOMPLETE：批次异常退出；198 个 Case')
    p.write_text(text)
    raise SystemExit(1)

completed=subprocess.run([sys.executable,str(evidence/'audit_batch.py'),str(batch)],
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
(evidence/'batch_audit.log').write_bytes(completed.stdout)
summary=json.loads((batch/'summary.json').read_text())
inventory=json.loads((batch/'inventory.json').read_text())
rows=[json.loads(l) for l in (batch/'results.jsonl').read_text().splitlines()]
finals=json.loads((batch/'finalizations.json').read_text())

# Preserve existing sources and all original S0-S6 reports. Only README was edited.
before=json.loads((evidence/'preservation_before.json').read_text())
changes=[p for p,h in before['files'].items() if p not in before['allowed_changes'] and
    (not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h)]
diff=subprocess.run(['git','diff','HEAD','--binary'],cwd=repo,capture_output=True,check=True)
preservation={'existing_source_and_stage_reports_unchanged':not changes,'changed_paths':changes,
    'tracked_diff_unchanged':hashlib.sha256(diff.stdout).hexdigest()==before['tracked_diff_sha256']}
(evidence/'preservation_final.json').write_text(json.dumps(preservation,indent=2)+'\n')

# Read only the captured logs; identify observed failures without assigning all
# runtime exceptions to Case kernels or changing any result status.
issues=[]
for r in rows:
    if r['status']=='PASS':continue
    detail=[]
    p=Path(r['stderr_path']) if r.get('stderr_path') else None
    if p and p.is_file():
        with p.open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-32768));tail=f.read().decode('utf-8',errors='replace')
        detail=[line.strip() for line in tail.splitlines() if
            ('Error:' in line or 'Exception:' in line or 'failed' in line.lower())][-6:]
    issues.append({k:r.get(k) for k in ('case_id','case_path','item','status','reason','message',
        'evaluator_errors','issue_owner','stdout_path','stderr_path','result_path','raw_result_path')}|
        {'captured_error_lines':detail})
(evidence/'unresolved_issues.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2)+'\n')

counts=summary['by_item']
status_names=('PASS','FAIL','BLOCKED','UNSUPPORTED','TIMEOUT')
count_table=['| 项目 | PASS | FAIL | BLOCKED | UNSUPPORTED | TIMEOUT |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
for item in ('clean','runtime','pmu','compiler'):
    count_table.append('| '+item+' | '+' | '.join(str(counts[item][s]) for s in status_names)+' |')
complete=summary['recording_complete']
all_pass=summary['all_pass']
audit_ok=completed.returncode==0
recording_ok=complete and audit_ok
full_cases=Counter(r['case_path'] for r in rows if r['status']=='PASS')
fully_pass=sum(n==4 for n in full_cases.values())
final_counts=dict(Counter(f.get('value',{}).get('status','no_valid_final_result') if isinstance(f.get('value'),dict)
    else 'no_valid_final_result' for f in finals))
stage_conclusion='验收已完整执行并记录' if recording_ok else '已有记录；覆盖或事后审计未全部通过'
lines=['# S7：全部 v0.2 Case 验收','', '日期：2026-10-04。仅执行 S7。', '',
    f'**记录结论：{stage_conclusion}。全部项目通过：{all_pass}。**', '',
    f'冻结清单 {summary["case_count"]} 个 Case；记录 {summary["actual_records"]}/{summary["expected_records"]}；'
    f'四项全部 PASS 的 Case：{fully_pass}。FAIL、BLOCKED、UNSUPPORTED、TIMEOUT 均不算通过。', '',
    '## 实现与前置', '',
    'S6 公共层与真实硬件入口已通过。新增 `evaluator/accept_all_cases.py`、'
    '`acceptance_artifacts.py` 及 15 项 CPU 控制测试；只增加维护者工具，不增加 Agent CLI，'
    '不改 Case、Seed、Fixture、Ref、Oracle、comparator、Case-local evaluator 或 RunConfig，'
    '不自动 prepare-ref。不使用旧 bench、Docker 或独立仓库。', '',
    '完整公共 CPU 回归：159 项 / 418.046s / OK（含当时 13 项 S7 用例）；'
    '最终 S7 工具：15 项 / 102.774s / OK。S6 的九组真实 native profile 产物全部通过检查器复核。', '',
    '[完整回归日志](S7_evidence/full_regression.log)、'
    '[最终工具测试](S7_evidence/control_tests_release.log)、'
    '[native 产物复核](S7_evidence/S6_native_artifact_recheck.json)、'
    '[实现与 hash](S7_evidence/implementation_manifest.json)。', '',
    '## 实际批次', '',
    '批次 `20261004_S7_01` 冻结全部 `cases/*/*/case.json`，包括 pending 和 ID 异常路径。'
    '读取所有资产实际内容做 SHA-256，包括大型 .pt；冻结公共代码 hash 和环境版本。'
    '就绪 Case 创建四个相同原始 Seed 的独立 Workspace，通过真实 ptobench 依次执行 '
    'clean evaluate、runtime、PMU、compiler。每项完成即保存日志、JSON、Session/event ID、'
    '正确性、完整性和产物检查。单命令超时 1800 秒，清理未确认时阻止后续执行。', '',
    f'基线读取错误：{len(inventory["hash_errors"])}；canonical 内容一致：{summary["canonical_integrity"]["passed"]}；'
    f'公共代码内容一致：{summary["public_code_integrity"]["passed"]}；停止原因：{summary["stop_reason"]}。', '',
    *count_table, '',
    f'独立 finish 记录 {len(finals)} 份：{json.dumps(final_counts,ensure_ascii=False)}。'
    '它们不替代四项探索记录；全失败且无有效 Snapshot 时无法产生最终成绩。', '',
    '## 数据与审计', '',
    '- [完整清单与版本](full_v02/20261004_S7_01/inventory.json)',
    '- [逐项详细结果](full_v02/20261004_S7_01/results.jsonl)',
    '- [全部 Case 矩阵](full_v02/20261004_S7_01/matrix.csv)',
    '- [机器汇总](full_v02/20261004_S7_01/summary.json)',
    '- [批次报告与未通过项](full_v02/20261004_S7_01/report.md)',
    '- [独立 finish](full_v02/20261004_S7_01/finalizations.json)',
    '- [canonical hash 对比](full_v02/20261004_S7_01/canonical_comparison.json)',
    '- [公共代码 hash 对比](full_v02/20261004_S7_01/code_comparison.json)',
    '- [独立覆盖及证据审计](S7_evidence/batch_audit.json)',
    '- [历史报告与原源码保留检查](S7_evidence/preservation_final.json)',
    '- [原始命令和完整批次日志](S7_evidence/commands.txt)', '',
    f'独立审计通过：{audit_ok}；既有公共源码及 S0～S6 报告一致：'
    f'{preservation["existing_source_and_stage_reports_unchanged"]}；原 tracked diff 一致：'
    f'{preservation["tracked_diff_unchanged"]}。', '',
    '## 未解决问题与后续处理', '',
    '[逐项问题、原始错误和排查归属](S7_evidence/unresolved_issues.json) 保留每个非 PASS 项。'
    '资产/认证/ID 问题归 Case 迁移维护；设备/模式关闭归环境配置；运行和产物错误需依据原始日志排查'
    '公共加载协议、Case 适配及 runtime/toolchain，不能仅凭 FAIL 判为 Case kernel 错误。', '',
    '`combine_flash/pro` 已观察到 Case-local evaluator 的 build_tensor_specs 延迟 '
    '`from config import ...` 失败；公共加载器的 config alias 为临时作用域。'
    '后续专项应核对两者的加载约定，决定公共作用域或 Case 适配修复。本阶段只保留错误，没有修改它们。', '',
    '修复后用新的完整批次复测，保留本次失败证据；不能将不同版本的成功行拼成“全量通过”。', '',
    '只有每个 Case 的四项都为 PASS 且内容 hash/覆盖审计通过，才声明全部 v0.2 验收通过。', '']
(reports/'S7_full_v02_acceptance.md').write_text('\n'.join(lines))

old=(evidence/'final_acceptance.md.before').read_text()
old_table=old[old.index('| 范围 |'):old.index('\n按用户确认')]
old_table=old_table.replace('| 全部 v0.2 Case 的 clean/runtime/PMU/compiler | NOT_RUN：属于 S7 |',
    f'| 全部 v0.2 Case 的 clean/runtime/PMU/compiler | {stage_conclusion}；全部通过={all_pass}；'
    f'{summary["actual_records"]}/{summary["expected_records"]} 项 |')
final=['# 最终验收结果','', f'日期：2026-10-04。S6 公共层与集成验收已通过；S7 {stage_conclusion}。'
       f'**全部 v0.2 Case 四项通过：{all_pass}。**','',old_table,'',
       f'S7 冻结 {summary["case_count"]} 个 Case，四项均 PASS 的 Case 为 {fully_pass}；统计如下。','',
       *count_table,'',
       f'记录覆盖完整={complete}，独立审计通过={audit_ok}，canonical 内容一致='
       f'{summary["canonical_integrity"]["passed"]}，公共代码一致={summary["public_code_integrity"]["passed"]}。','',
       'S6 历史全库 hash 相等的 FAIL 仍保留：当时其他 Case 的并发修缮产生 42 条差异，'
       '不是本次 hash 结论。缺 Seed Case、pending 和运行缺陷均按 S7 原始记录保存，未自动修复资产。','',
       '[S7 完整报告](S7_full_v02_acceptance.md)、[批次矩阵](full_v02/20261004_S7_01/matrix.csv)、'
       '[逐项结果](full_v02/20261004_S7_01/results.jsonl)、'
       '[未解决问题](S7_evidence/unresolved_issues.json)、'
       '[S6 历史报告](S6_integration_acceptance.md)、'
       '[Provider 补充验收](S6_provider_followup/provider_acceptance.md)。','']
(reports/'final_acceptance.md').write_text('\n'.join(final))

# Update current status only; S0/S1 prompts and historical evidence are untouched.
for relative in ('docs/10_stage_prompts.md','Benchmark升级/README.md',
                 'Benchmark升级/plan/README.md','Benchmark升级/plan/08_execution_steps.md'):
    p=repo/relative
    text=p.read_text()
    status=('S7 已执行并记录；全部 Case 四项尚未全部通过' if recording_ok and not all_pass else
            'S7 全量验收通过' if recording_ok and all_pass else
            'S7 已产生验收记录，覆盖或事后审计存在问题；尚未完整结项')
    text=text.replace('S7 尚未执行',status)
    p.write_text(text)

completion={'state':'reports_saved','recording_complete':complete,'independent_audit_passed':audit_ok,
    'all_pass':all_pass,'case_count':summary['case_count'],'records':summary['actual_records'],
    'report':str(reports/'S7_full_v02_acceptance.md'),'batch':str(batch)}
(evidence/'completion_status.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(completion,ensure_ascii=False),flush=True)
raise SystemExit(0 if recording_ok else 1)
