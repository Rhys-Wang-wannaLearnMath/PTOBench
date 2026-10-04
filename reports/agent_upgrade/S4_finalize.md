# S4：候选选择与最终复核

阶段：S4。日期：2026-10-03（Asia/Shanghai）。结论：**PASS（公共实现、CPU 假 Evaluator 验收）**。最终 **122 tests / 182.167s，OK，exit 0**：原有 100 项 S0～S3 回归 + 新增 22 项 S4 测试，无 skip/expectedFailure。满足进入 S5 的公共前置条件，本轮没有实施 Launcher/Adapter 或自动触发逻辑，没有执行真实 Optimizer、NPU、真实 Case prepare-ref、真实 profiling 或正式测量验收。

依据：`docs/10_stage_prompts.md` 共同约束、`Benchmark升级/plan/{README,10_cli_cutover,05_finalize,08_execution_steps}.md`。已核对 S0～S3 报告及实际 API；原阶段报告和证据均保留，未重做 S1 保护/复制/context 或 S0 comparator 修缮。

## 实际修改与接口

| 文件 | S4 行为 |
| --- | --- |
| [snapshots.py](../../evaluator/snapshots.py) | Selection、历史登记/事件/权威结果/source/metadata 校验，确定性选择，两个内部公开 Python API |
| [finalize.py](../../evaluator/finalize.py) | FinalResult、干净 final workspace/context、独立 finish、共享执行/事后检查、正式结果、失败与幂等恢复 |
| `session.py` | `_commit` 支持额外保护 records；`complete_action` 一次提交 finish event、Session 终态/时间、清除 active、完整 Evaluator bytes 和 final_result |
| `session_actions.py` | 全局残留恢复根据 action kind 使用 final/workspace 或探索 staging；执行/停止/信号/后检/结果验证 helper 本身不变 |
| [test_finalize.py](../../evaluator/tests/test_finalize.py) | 22 项选择、最终成绩、失败、篡改、并发、真实假进程/崩溃恢复测试 |
| `tests/action_test_support.py` | 假 Evaluator 可配置已有 median/speedup，记录执行前目录状态；S3 默认行为保留 |
| [FINALIZE.md](../../evaluator/FINALIZE.md) 与当前公共 README/SESSION/AGENT_CLI | 使用说明、S5 caller 前提、返回 schema、幂等/中断与正式成绩来源 |

按设计提供：

```python
snapshots.select_best_snapshot(session_root) -> Selection | None
snapshots.finalize_session(session_root, close_reason) -> FinalResult
```

finalize 实现分在公共 `finalize.py`，由 snapshots 的内部入口调用。两个入口自己取得全局锁；内部执行全程使用 caller-held S3 helper，不调用公开 CLI、不递归取得同一执行锁。不新增 Agent finalize 命令；旧 bench/parser 继续不存在。选择无硬件调用，正式复核直接运行现有 evaluate.py。

## 历史候选与确定性选择

先恢复 Session/events/authority records 及只读 Snapshot source/metadata，核对 Snapshot 保护名称/绝对路径/readonly、Session/Case/Optimizer、event_seq、来源 workspace、source manifest/hash、完整 Evaluator JSON 与 observation。只考虑来源 completed event 成功、correctness PASS、integrity PASS、正式保护登记完成的 Snapshot。meta/observation 必须与来源 authority 和事件一致；源码须与该动作保护清单一致。

未登记的发布目录或未被成功事件引用的 orphan 不参与选择；不一致的已提交证据或无法恢复的来源明确失败，不将它们静默当作可信候选。外部改写/deletion/link 会从原保护内容恢复；当前工作副本不回退。

策略固定为 `lowest_observed_latency_else_latest_pass`：

1. 有有限正数 latency_us，选最低值；同值选更早 event_seq。null 候选不挤掉有数值候选。
2. 全部候选无有效耗时，选最近 PASS，reason=`latest_pass_without_latency`。
3. 无有效候选，持久化 failed/no_valid_snapshot，finish_usage=0，不使用当前 Candidate 或 Seed 代替成功候选。

evaluate/profile 的已存在 observation 都能用于筛选。保留 kind、公开 mode、metric_name、unit、原 latency、事件/hash 与 selection reason。不添加 profile 执行、warmup、性能采样或 CLI wall-clock 推算，不伪造 speedup。

## 最终执行与成绩

内部 API 用短 Session 更新关闭探索，再等待全局锁；探索耗尽时已接受的最后动作仍可完整提交 Snapshot/事件。S5 调用方仍必须停止/回收 Optimizer 和剩余 worker，并确认硬件动作已收尾；S4 不声称管理尚未登记的 Optimizer 进程。进入 finalize 时先通过 S3 恢复已归属的残留组，无法确认 attribution/stop 则阻塞而不测量。

从选中 Snapshot source 重建 final/workspace，不复制旧 context/output/cache；复制后的源码 hash 与选择一致，canonical context/starting_point 重新生成。原只读 Snapshot 保留；新的 workspace 可写自己的编译产物。保护 final/source_manifest.json、selection.json、context 和 authority_result 位置，再预占唯一 finish action。

调用现有 Evaluator，固定 `phase=all`、`profile-mode=clean`、独立 `final/result.json`，双路日志保存在 final。沿用 S3 gate/PID identity/进程组/信号清理与 finally postcheck。事后变化使本次失效。异常或中断不退还 finish，也不换第二名重测；无成功探索 Snapshot 新增。

final_verified 字段仅取本次校验通过的 fixture_000 clean result：correctness=`passed`、有限正数 median 与 speedup。失败字段为 null，即使历史 observation 或原始失败 JSON 有数值也不填入。best_observed 字段只说明选择信号。完整原始 Evaluator bytes（包括失败/坏 JSON）保存在 final/authority_result.json 并受保护；无 result 时保持保护的 absence。

## 原子终态、幂等和中断

一次 S1 `commit_records` 同步发布 Session status/completed_at、final_result、finish event、active 清除及完整原始结果的保护记录。重入先恢复/校验已提交结果，返回同一 payload，不重复测量、计费或写 finish event。可编辑 raw/diagnostic 文件不能提供权威正式值；不合法的已提交 final metric 也会被拒绝并转为无 verified 成绩的 infrastructure failure。

finish 已消费但没有已提交 final_result 时，清理已登记进程、补齐一次 interrupted 或已知失败，再写失败结果，不用遗留 PASS raw result 充当成功，也不重跑。预占后 marker 前崩溃、Evaluator 期间 kill parent、终态 manifest 前/后崩溃均有真实假进程测试：manifest 前保留 consumed finish 并失败；manifest 后恢复已提交完整成功；两种情况都是一次 finish event/charge。

停止不确认或权威/恢复基础设施不可用时保留 active/marker 证据，阻止资产修复下的并行执行及进一步硬件操作。可可靠读取/提交的 Session 写 failed；无法可靠提交时返回 infrastructure_error 而不伪造成功终态。

## 测试命令与实际结果

所有命令从仓库根目录使用 `.venv-official/bin/python`；tee 使用 pipefail。

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_finalize.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S4_evidence/finalize_second.log

set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_*.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S4_evidence/unittest_first_full.log
```

| 原始日志 | 退出码 | 实际结果 |
| --- | ---: | --- |
| [finalize_first.log](S4_evidence/finalize_first.log) | 1 | 20 / 82.183s，1 FAIL（故障测试未破坏工作源码以触发恢复，见下） |
| [finalize_second.log](S4_evidence/finalize_second.log) | 0 | 21 / 85.084s，OK；后续增加 final schema 数值类型故障测试并完善校验 |
| [unittest_first_full.log](S4_evidence/unittest_first_full.log) | 0 | **122 / 182.167s，OK**；最终代码下的 22 项 S4 与原有 100 项全部通过 |
| [capture_final.log](S4_evidence/capture_final.log) | 0 | 三种归档：completed / no_candidate / clean_failure；总 6 次 fake Evaluator，重复调用增加 0 次 |

首轮唯一失败位于 `test_corrupt_history_recovery_blocks_finish_without_charge`：只破坏备份、工作文件保持完整，S1 正确地不需要恢复，所以未抛异常。补充破坏对应 Snapshot helper 使其确实需要恢复后，测试证明损坏恢复源阻止 finish、finish_usage=0、无第二次执行。保留首轮日志，未改 S1 的“只验证所需恢复源”协议，也未降低恢复失败验收。

完整回归 comparator 构造读取当时 inventory=196、ready=159、constructed=190；6 个 non-ready 缺 evaluator 明确 NOT_RUN。该 CPU 构造并非真实 Case/NPU 认证；动态计数变化来自其他任务的并行 Case 工作，原 S0～S3 报告数值未改写。

## 逐项证据

| 要求 | 当前测试行为 |
| --- | --- |
| 混合/null/并列/重复源码选择 | `test_mixed_metrics_choose_lowest_then_earliest_without_new_execution`：profile 8us 优于 evaluate 40us，同 8us 取更早 profile，后来 7us 才替换；无额外调用 |
| 全 null 最近 PASS | `test_all_null_or_invalid_profile_metrics_choose_latest_pass`：None/0/-1/bool 诊断值按 S3 规则为 null；选择最新 PASS，无额外测量 |
| 无候选、不回 Seed/current、不耗 finish | `test_no_snapshot_never_falls_back_to_current_or_seed_and_is_idempotent`：有 Seed/current 仍 failed，usage=0/0，重复返回一致，无 final workspace |
| 干净 source/context、正式新值、非零 Optimizer reason | `test_success_uses_clean_selected_source_and_new_verified_numbers`：执行前无 output/cache，helper 来源 Snapshot，不取 current=900；观测 2us、复核 123us/3.25，完成 optimizer_nonzero_exit |
| null 也执行一次正式 clean | `test_all_null_selection_still_receives_one_clean_finish`：选最新 compiler，best null、最终 17us |
| 历史 source/meta/event/authority 恢复与 link 安全 | `test_history_source_metadata_events_and_authority_are_restored`：恢复原 helper/metadata/event/authority、权限复原、外部哨兵不变，成功复核 |
| 恢复损坏/不一致历史拒绝 | `test_corrupt_history_recovery_blocks_finish_without_charge`；`test_inconsistent_committed_history_is_rejected_before_finish`：finish=0，只有原探索调用 |
| 未登记 orphan 不选 | `test_unregistered_orphan_never_enters_selection`：伪造 .001us orphan 不能替代已登记 12us |
| 最终失败不重选、不沿用 observation | `test_clean_failure_never_uses_observed_score_or_tries_second_place`：两候选，最低 1us 的 clean FAIL，verified 全 null、只执行一次 finish、重入不测第二名 |
| 空/错 Case/NaN/nonzero 最终结果 | `test_bad_or_missing_final_results_are_failed_without_replay`：四类均 action_failed、finish=1，重入无新调用 |
| 事后 integrity | `test_final_integrity_changes_fail_even_when_evaluator_passes`：final source/context/record 三类篡改均失效，无最终 Snapshot |
| 准备失败、已有 infrastructure failure | 两项测试分别证明 failed 可幂等、finish=0，已有 PASS 不强行测量 |
| FinalResult/authority/event 恢复、忽略未保护 raw | `test_committed_final_and_authority_are_restored_without_reading_diagnostics`：link 外哨兵不变，恢复同一最终 19us/7，无新调用 |
| 不合法已提交 final 类型 | `test_invalid_committed_final_metric_is_rejected_without_measurement`：bool 不能充当 1us，改为 failed/infrastructure_error、verified null，重入无执行 |
| 多进程幂等 | `test_concurrent_finalize_processes_commit_one_finish`：三进程返回完全相同 payload，1 finish、1 finish event |
| SIGTERM 与 descendants | `test_sigterm_finish_stops_group_and_publishes_one_failed_result`：回收 child，一次 interrupted，之后不重测 |
| kill parent 恢复 | `test_killed_finish_owner_recovers_final_process_group_without_remeasurement`：按 final 路径做 postcheck，清 leader/child，保留一次 finish 消费 |
| reservation/marker 窗口 | `test_finish_reservation_crash_before_marker_never_replays`：真实 owner exit 23，未启动 Evaluator、1 finish、一次 interrupted event |
| 终态原子提交 | `test_terminal_transaction_crashes_restore_all_old_or_all_new`：真实进程 manifest 前/后 exit 23；分别 failed/completed、无重测/多计费/重复事件 |
| exhausted 最后动作自然完成 | `test_last_exhausted_action_finishes_before_independent_finish`：closed+active 时 finish=0，释放最后探索后保留 PASS 40us，再完成独立 clean 50us，usage=1/1 |

完整测试方法、断言与日志均在上述文件；验收映射、源码/日志/报告 hash 另见 [validation_summary.json](S4_evidence/validation_summary.json)。

## 可检查的假运行归档

运行 `S4_evidence/capture_final.py` 在 TemporaryDirectory 小型假 Case 中执行并归档；拒绝覆盖既有证据。

- [completed final_result](S4_evidence/fake_finalizations/completed/final_result.json)：来源 PMU 3us，正式 clean 55us、speedup=4，exploration/finish usage=3/1，保留 optimizer_nonzero_exit 原因。`returned.json` 与重复调用相同。
- [completed events](S4_evidence/fake_finalizations/completed/events.jsonl)：三次探索、一次 finish；finish 不引用新 Snapshot。
- [selection](S4_evidence/fake_finalizations/completed/final/selection.json)、[authority_result](S4_evidence/fake_finalizations/completed/final/authority_result.json)：保留选择 provenance 与独立正式数据；相邻目录保存 source manifest、新 context/starting_point、完整 final workspace 和双路日志。
- [no_candidate](S4_evidence/fake_finalizations/no_candidate/final_result.json)：failed/no_valid_snapshot、usage=0/0，未执行。
- [clean_failure](S4_evidence/fake_finalizations/clean_failure/final_result.json)：已有 null profile PASS，最后 clean FAIL，usage=1/1、verified 全 null；原始 FAIL JSON 仍受保护归档。
- [evaluator_calls](S4_evidence/fake_finalizations/evaluator_calls.jsonl)：总 6 次 fake Evaluator，只有 2 次 final/workspace all/clean；三类重复 finalize 都增加 0 次。

所有数值来自可控 fake，不能称作实际性能、真实 profiling 或真实认证。归档中的绝对 /tmp 路径记录当时 provenance；临时运行目录已清理，归档源码/结果可独立检查。保护机制是 Bench 完整性协议，不是 OS 权限隔离。

## 保留性与范围审计

[before.json](S4_evidence/before.json)、前后 git status 与 [preservation.json](S4_evidence/preservation.json) 记录：

- 原 297 个 tracked 文件内容 hash 与 Git index 保持；未 reset/clean/stage/commit。
- S0～S3 报告及原始证据共 **178 个文件**内容 SHA-256 保持，未重写历史 gate、日志或资产统计。
- 原 public/history 差异只有本报告表列的 S4 公共模块、fake support 与当前说明；新 final 模块/测试/说明/报告为新增。
- `integrity.py`、`file_tree.py`、`workspace.py`、`common.py`、`evaluate.py`、`workflow.py`、`profile_modes.py`、维护 prepare_ref/generate_fixtures、Agent parser 与 ptobench 的内容 hash 不变。S3 `_run_process`/`_stop_group`/`_postcheck_action`/`validate_result`/`_signals` 与 create_snapshot AST 不变；只扩展 finish 残留目录选择与原子 records 提交。
- 基线 canonical 共 **7219 个文件**，3952 个非 tensor 做内容 hash，3267 个 `.pt` 做 size/mtime_ns/mode；未读取全量 tensor 字节 hash。
- 结束审计观察到 **64 个 canonical 新增/变化路径**，来自并行 moe_ep2 demo/pro、prefill_layer_ep2_csa demo/pro 工作；完整列表保留，不写“全仓库 canonical 未变化”。本次所有写操作仅公共实现/说明/S4 证据与临时假资产，没有写真实 Case、Fixture、Ref、Oracle、comparator、Case-local evaluator 或 environment，也没有回退其他任务资产。64 是本次审计时刻的观测值。

最终审计 [validate.py](S4_evidence/validate.py) 核对全部历史 hash、允许 public 差异、S1/测量/共享执行不变、旧入口不存在、归档 source/hash/结果/预算、22 项验收映射、无残留 fake Evaluator，并保存当前源码、证据和报告 SHA-256。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python \
  benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S4_evidence/preserve.py after \
  > benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S4_evidence/preservation_final.log

set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python \
  benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S4_evidence/validate.py \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S4_evidence/validation_final.log
```

## 进入 S5 的条件

**S4 无未解决公共实现阻塞，可进入 S5。** 内部 selection/finalization 可用于 Launcher 的正常退出、非零退出、预算耗尽或人工终止路径。S5 仍须实现 YAML/preflight、真实工具/Adapter、Optimizer 生命周期、停止并回收剩余 worker 后调用 finalize，以及自动触发/status；本轮不添加这些行为。

真实 NPU/测量等价性、真实 Optimizer、代表与全量 Case 验收均 **NOT_RUN（后续阶段）**。原阶段真实资产阻塞仍按原责任归属处理；并行 Case 文件变化和这组 fake 测试不证明这些资产已重新认证或全量就绪。
