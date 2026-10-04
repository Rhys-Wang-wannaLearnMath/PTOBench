# S3：CLI、事件、冻结源码与 Snapshot

结论：**PASS（公共实现与假 Evaluator CPU 验收）**。最终 **100 tests / 94.742s，OK，exit 0**，包含原有 74 项 S0/S1/S2 回归与 26 项 S3 测试，无 skip/expectedFailure。S3 门槛已满足，可进入 S4；本次没有实施 S4，也没有执行真实 NPU、真实 profiling、真实 Optimizer、真实 Case prepare-ref 或正式测量。

执行日期：2026-10-03（Asia/Shanghai）。依据 `docs/10_stage_prompts.md` 共同约束及 `Benchmark升级/plan/{README,10_cli_cutover,01_interfaces_and_layout,04_cli_events_and_snapshots,08_execution_steps}.md`。已核对 S0、S1、S2 报告与实际 API，保留原报告/证据，不重做 S1 文件保护或源码复制。

## 实际修改

| 文件 | S3 行为 |
| --- | --- |
| `ptobench`、`evaluator/agent_cli.py` | 仅五种操作；位置参数默认 `.`；强制 Session 环境；JSON 结果/固定退出码；不做 bootstrap、不 import torch/PyPTO |
| `evaluator/session_actions.py` | 全局恢复、事前恢复/检查、稳定冻结、预占、现有 Evaluator 子进程、双路日志、进程组清理、事后校验、结果验证、权威提交/事件/诊断发布 |
| `evaluator/snapshots.py` | 从同一冻结源码创建唯一 readonly Snapshot，验证 metadata/source，再登记 S1 保护，最后允许事件引用 |
| `evaluator/session.py` | 提供已持锁 fork、非硬件事件、Session failure；completion observation/state；状态读取修复记录时保存 active integrity witness；拒绝会恢复旧入口/parser 的历史 Session |
| `evaluator/common.py`、`evaluate.py` | 仅增加内部启动 pipe gate 与 main 首行调用；先持久绑定 PID/records/logs，再允许 Evaluator 解析/加载 Case。维护调用没有 gate 时照常执行 |
| `profile_modes.py`、`workspace.py` | 增加 runtime/pmu/compiler 公开映射与 context 展示；更新生成 README 的 CLI、Session、额度、诊断与 fork 说明。内部模式/参数/产物语义保留 |
| 新测试/假 Evaluator support | 真实 CLI subprocess + TemporaryDirectory 假 ready Case；实现文件与 tensor generation 入口设置为禁止执行，真实资产不被复制或准备 |
| `README.md`、Evaluator README/SESSION、新 `AGENT_CLI.md` | 当前 CLI 与维护接口说明、日志背压限制、恢复与 S4 交接 |
| 删除 `bench`、`evaluator/bench.py` | 无兼容链接、别名、转发或旧 parser；原字节仅存 S3 历史证据 `.before` 文件 |

S2 测试的非功能命令 stub 与固定根断言从 `bench/public_bench` 改为 `ptobench/public_ptobench`，继续验证公共命令保护，没有降低测试标准。S1 的 `integrity.py`、`file_tree.py`、workspace 源码复制/hash 函数未改写；`workflow.py`、全部原有 common 函数、profile RunConfig 与 warmup/rounds 未改变。最终审计以 SHA-256/AST 核对这些事实，见 [validation_summary.json](S3_evidence/validation_summary.json)。

## 执行与记录语义

公开接口固定为：

```text
ptobench evaluate [WORKSPACE]
ptobench profile runtime [WORKSPACE]
ptobench profile pmu [WORKSPACE]
ptobench profile compiler [WORKSPACE]
ptobench fork [PARENT_WORKSPACE]
```

`evaluate` 只调用一次现有 `evaluate.py --phase all --profile-mode clean`；profile 分别使用 `--phase correctness --profile-mode runtime/hardware/compile`。以 argv 数组、shell=False、独立进程组启动，cwd 为冻结 workspace。删除继承的 `PYPTO_BENCH_ENV` 与 `PYPTO_BENCH_PROFILE`，原 Evaluator 继续负责 Case environment bootstrap；gate 在 bootstrap re-exec 前消耗并移除。没有新增 profiling collector、第二次性能执行或 CLI 总耗时指标。

在 caller-held 全局锁内恢复残留全局标记、恢复资产与记录、校验 Session/registry/开放模式、调用 S1 稳定复制、重建冻结 context。冻结源码清单、context 与独立 authority_result 纳入保护，预占与 active_action 一次提交。检查/冻结失败为非硬件 rejected event，不扣预算；预占后任何失败/中断不退还。

子进程原始双路日志同时保存到 `stdout.log`/`stderr.log`。终端转发使用磁盘日志作为 spool、非阻塞写与有界最终 flush；终端不读取时磁盘仍完整增长，动作能结束并释放锁。持续阻塞/断开的终端可能丢失其显示尾部和最终 JSON，权威事件与磁盘日志仍保留；不是声称不可读终端也能收到全部字节。

SIGINT/SIGTERM 先终止已拥有的组，最多等十秒，再 KILL 存活成员、回收 leader。重复信号在清理/提交期间延迟处理。只有确认已停止才能事后校验/恢复；无法确认时关闭 Session、保留 active/marker，不发布成功。PID 与 birth ticks 不符时禁止 signaling。下一 Session 执行前先清理旧 Session 的已登记残留组；预占后尚未写 spawn marker 的死亡 owner 通过 S2 lifecycle 只收尾一次；starting 无 PID 的不确定状态明确阻塞。

事后先检查固定资产/context/records/冻结源码，再恢复。状态读取只持短 Session 锁，记录修复留下 active witness，不能抹掉本次篡改证据；恢复操作新发现的变化也使本次失效。原 Candidate 编辑保留，Snapshot 来源一直是 staging。有效结果必须严格匹配 Case、冻结 candidate 路径、phase/mode、official/diagnostic flags、唯一 fixture_000、PASS/正确性与零退出码；evaluate 额外要求有限正数 latency/speedup。profile 没有有效 latency 时 observation 三个 metric 字段为 null。

有效 authority result 经 S1 原子提交后，Snapshot 临时创建、hash/metadata 核对、原子发布、保护登记完成，事件才引用其 ID。Snapshot source 排除 context/outputs/caches，文件 0444、目录 0555；同源码多次成功也保留独立 observation。登记失败或 commit 失败关闭 Session，不把 orphan 或不可靠结果公开为成功。完成事件与清除 active 一次提交；无效 raw result 仅用于诊断。

权威结果与 events 不读取可编辑 `latest_result.json`/`latest_session_result.json`。诊断发布在权威 completion 后，将对应内部 mode 的完整产物集替换到原 workspace，删除 root/mode 链接本身，不沿链接写外部。重复同一模式不会混入旧文件。latest_session_result 包含 event/status/integrity/Snapshot。staging 保留，已有 build/report 路径不改写。诊断发布失败不改变已提交观察值。

## 命令、结果与原始证据

所有测试从仓库根目录使用 `.venv-official/bin/python`，保存日志的 shell 启用 pipefail。最终命令：

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_*.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S3_evidence/unittest_final.log
```

| 证据 | 退出码 | 实际结果 |
| --- | ---: | --- |
| [unittest_final.log](S3_evidence/unittest_final.log) | 0 | **100 / 94.742s，OK**；6 CLI + 17 actions + 3 snapshots + 原有 74 |
| [session_actions_gate.log](S3_evidence/session_actions_gate.log) | 0 | 9 / 35.563s，启动门控后篡改回归通过 |
| [session_actions_hardened.log](S3_evidence/session_actions_hardened.log) | 0 | 16 / 58.730s，背压、重复信号、恢复 witness、停止失败、CLI 最后额度竞争通过 |
| [agent_cli_hardened.log](S3_evidence/agent_cli_hardened.log) | 0 | 5 / 10.826s；最终新增旧 Session 恢复阻止测试由完整 100 项日志覆盖 |
| [snapshots_hardened.log](S3_evidence/snapshots_hardened.log) | 0 | 3 / 10.727s，authority/Snapshot 登记故障不发布成功 |
| [unittest_99_pass.log](S3_evidence/unittest_99_pass.log) | 0 | 最终 legacy 恢复防护之前，99 / 93.743s，OK |
| [fake_actions_capture.log](S3_evidence/fake_actions_capture.log) | 0 | 7 次 CLI、5 次假 Evaluator、4 个成功 Snapshot |

定向命令使用相同 discover 参数，将 `-p` 分别设为 `test_agent_cli.py`、`test_session_actions.py`、`test_snapshots.py`，对应 tee 到上表日志；最终完整日志覆盖每项测试在最终源码下的结果。

保留可检查的小型假运行归档：

- [outcomes.json](S3_evidence/fake_actions/outcomes.json)：evaluate PASS 10us、三种 profile PASS null、fork、旧 hardware 拒绝、evaluate FAIL。
- [events.jsonl](S3_evidence/fake_actions/events.jsonl)：seq 1～7；hardware usage=5；fork/rejection 不扣费；失败无 Snapshot。
- [evaluator_calls.jsonl](S3_evidence/fake_actions/evaluator_calls.jsonl)：固定映射下恰好五次执行，无 profile 补测。
- [Snapshot metadata](S3_evidence/fake_actions/snapshots/snapshot_000003/metadata.json)：PMU observation 的 latency/metric/unit 为 null。相邻 source 为实际归档的只读 Candidate/helper。
- `fake_actions/actions/`：冻结源码清单、authority/raw result、双路日志；`fake_actions/context/`：实际新建公开 context。

归档明确来自受控 CPU fake Evaluator；其中绝对临时路径用于记录当时 provenance，临时运行目录已清理。`capture_actions.py` 只复制小型假资产/证据，拒绝覆盖已有归档，不作为真实正确性/性能认证。归档捕获于最终 legacy 防护之前，最终源码另由完整 100 项测试与 hash 审计确认。

## 验收映射

| S3 要求 | 行为证据 |
| --- | --- |
| 五种操作、默认根、固定映射/null | `test_five_operations_default_workspace_fixed_modes_and_null_profiles`；归档 calls/events/metadata |
| 旧参数/mode/命令拒绝且无预算/执行/Snapshot | `test_rejects_old_parser_modes_parameters_and_commands_without_execution`（12 种请求） |
| 缺 Session/非法根/未登记/子目录无搜索回退 | `test_missing_session_invalid_root_unregistered_and_no_upward_search` |
| 旧入口不存在、维护能力仍在、公开模板 | `test_new_context_public_names_and_old_entry_absence`；S1 context 无模式测试；最终 hash/入口审计 |
| 旧 Session 不能通过恢复重装 parser | `test_legacy_session_cannot_restore_removed_parser`；零调用/零扣费/无 Snapshot |
| 失败/异常/空/坏 Case/fixture/phase/NaN/zero/nonzero | `test_invalid_results_and_exceptions_charge_once_never_snapshot`，十次失败各扣一次，无 Snapshot |
| 原 Candidate 并发修改保留，Snapshot 同冻结源 | `test_original_edits_after_freeze_do_not_change_snapshot_or_hash` |
| 固定资产/context/冻结源码/源码 link/records 篡改拒绝 | `test_protected_and_frozen_changes_invalid_even_if_result_pass`，五类均 integrity_failed |
| 查询修复仍保留篡改 witness、check/restore 窗口变化 | `test_status_repair_during_execution_preserves_tamper_witness`、`test_change_after_check_only_is_detected_by_restoration` |
| 源码链接事前拒绝且外部哨兵不变 | `test_linked_source_is_rejected_before_reservation`，evaluate/profile/fork 均不扣费 |
| 大双路日志/终端背压 | `test_dual_large_logs_are_preserved_and_forwarded_without_deadlock`（每路 1MiB）；`test_blocked_terminal_consumers_do_not_block_logs_or_completion`（两路均不读直到 CLI 退出） |
| 诊断替换与公开 result 非权威 | `test_same_mode_publish_replaces_complete_set_and_public_result_is_not_authority` |
| freeze 后输出 root/mode 链接安全 | `test_output_root_or_mode_link_inserted_after_freeze_never_writes_external`，外部目录仅保留原哨兵 |
| spawn 异常仍 postcheck/不退款 | `test_spawn_failure_still_postchecks_commits_without_refund`，1 charge、无 active、failed、无 Snapshot |
| 中断/后代清理/一次收尾 | `test_sigterm_stops_group_then_postchecks_and_records_once`；`test_repeated_signals_cannot_interrupt_term_then_kill_cleanup`（忽略 TERM 后 KILL） |
| kill parent 后先恢复跨 Session 进程组 | `test_killed_parent_recovers_global_marker_before_next_session_executes` |
| marker 前死亡 reservation 恢复一次 | `test_owner_killed_after_reservation_before_marker_recovers_once`，一次 charged interrupted，后续拒绝，无假执行 |
| PID reuse/不能确认停止 | `test_reused_pid_is_never_signaled`；`test_unconfirmed_process_stop_blocks_postcheck_and_keeps_evidence` |
| 实际 CLI 最后预算并发 | `test_real_cli_competition_accepts_only_last_budget_once`，4 个进程：1 PASS/3 budget_exceeded、1 调用、1 Snapshot |
| 独立 readonly Snapshot、历史保护恢复 | 两个原 Snapshot 测试：同源码三次成功独立 ID；source/metadata/authority 篡改于下次恢复 |
| authority/登记故障无可靠成功 | `test_authority_or_snapshot_registration_failure_never_publishes_success`，各保留扣费、failed/closed、无成功事件引用，恢复仅 interrupted |

S0 comparator 回归最终读取当时 inventory=196、ready=157、constructed=190；6 个 non-ready 缺 evaluator 明确 NOT_RUN。测试没有跳过 ready Case 的缺陷，也不把 comparator 构造视为真实 Case/NPU 认证。动态清单变化来自同时进行的外部 Case 工作，旧报告中的 192/183 等历史数值未改写。

## 发现与修复

保留失败证据：`session_actions_first/second/third.log` 各 9 项/2 FAIL，`unittest_first_full.log` 89 项/2 FAIL。最初 records 篡改在多动作场景中可被漏报：Evaluator 打开旧 session.json inode 时，父进程正用原子替换绑定 PID，子进程写入了已 unlink 的旧 inode。仅添加 repair witness 不足以消除这个执行顺序竞态。内部 gate 将实际 Case 行为延迟到 PID/records/logs 就绪后；最终记录篡改与并发状态修复测试均通过，未改变测量流程或放宽完整性标准。

`unittest_disabled_readme.log` 保留 99 项/1 FAIL：新增通用说明让无开放模式的 README 仍出现 profile 文本。将该说明移入条件生成的 Diagnostics 部分后，原 S1 断言保持，99 项及最终 100 项均通过。

审计另补齐：重复信号不能打断 TERM/KILL 清理；新发现的 restoration 变化参与 integrity；record restore 始终持短 Session 锁；无法确认停止不进入资产修复；终端背压不能拖住 completion；旧 Session 固定备份不能复活已删除 CLI。对应故障测试保留实际行为，没有用 skip 或降低期望制造 PASS。

## 原有工作与 canonical 保护

[before.json](S3_evidence/before.json)、前后 git status、[preservation.json](S3_evidence/preservation.json) 保存开始基线与结束审计：

- 开始时 297 个 tracked 文件的内容 SHA-256 未改变，Git index 未改变；未 reset/clean/stage/commit。
- S0/S1/S2 原报告及证据共 77 个文件 SHA-256 保持；原 public/history 差异全部属于上表 S3 授权文件/删除，包含必要的两个 S2 fixture/assertion 更新。
- 基线 canonical cases/sealed/environments 为 6924 个文件：3932 个非 tensor 保存内容 hash，2992 个 `.pt` 保存 size/mtime_ns/mode。**未读取全量 tensor 字节 hash。**
- 最后保留性审计观察到 **264 个 canonical 新增或变化路径**，完整列表见 preservation.json：主要涉及并行的 moe_ep2/ep4 demo/pro、prefill_attention_hca_pro、prefill_layer_ep2_csa demo/pro 及两个新 environment profile。没有把这些差异隐藏或写成“全仓库 canonical 未变化”。
- 本任务写操作仅覆盖公共代码、当前公共说明、S3 报告/证据及 TemporaryDirectory 假资产。未写任何真实 Case/config/comparator/Seed/Fixture/Ref/Oracle/Case-local evaluator 或 environment。外部并行资产工作保留，未恢复到本任务开始基线。264 是审计时刻的观测值，后续外部任务仍可改变它。

公共文件变更前字节在 S3_evidence 的 `.before` 中保存；common/evaluate 的原字节从仅添加 gate 的差异重建，重建 hash 与开始基线完全匹配。最终源码、日志、归档、报告的 SHA-256 和验收摘要见 validation_summary.json；审计脚本 [validate.py](S3_evidence/validate.py) 检查历史保护、S1 AST、measurement AST、入口删除、归档事件/hash/权限与无残留假 Evaluator。

审计命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python \
  benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S3_evidence/preserve.py after \
  > benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S3_evidence/preservation_final.log

set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python \
  benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S3_evidence/validate.py \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S3_evidence/validation_final.log
```

## S4 前置条件与未运行项目

**S3 无未解决公共实现阻塞，可进入 S4。** 可复用冻结/执行/完整性/validated result/authority/event/Snapshot 机制；S4 必须在已持全局锁的管理路径使用共享 helper，不能递归调用公开 CLI 重新取锁。候选历史校验、确定性选择、干净 final workspace、独立 finish 预占、正式 clean 复核与 final_result/幂等仍待 S4 实施，本次没有添加选择或 finalize 行为。

真实 NPU、真实 profile/测量等价性、真实 Optimizer/Launcher、代表 Case 与全量 Case 验收均 **NOT_RUN（后续阶段）**。S0/S2 记录的真实资产阻塞保留原责任归属；本轮并未重新认证它们，也不能从外部并行文件变化或假测试 PASS 推断真实资产已全部就绪。
