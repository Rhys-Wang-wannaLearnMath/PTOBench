# S2：Session、路径、锁和预算

阶段：S2。日期：2026-10-03（Asia/Shanghai）。结论：**PASS，满足进入 S3 的条件。**

依据：已读取 `docs/10_stage_prompts.md` 的共同约束、`Benchmark升级/plan/README.md`、`10_cli_cutover.md`、`01_interfaces_and_layout.md`、`03_session_and_budget.md` 和 `08_execution_steps.md`；核对 [S0 报告](S0_baseline.md)、[S1 报告](S1_file_integrity.md) 与实际 `workspace`、`file_tree`、`integrity`、`common`、`profile_modes` API。未重新实施 S0/S1，未改写原报告或证据。

本次实现 S2 内部 Python API、临时资产测试、使用说明与报告。没有实现新 Agent CLI、Evaluator 执行器、真实 Snapshot 发布、最终候选选择或 Launcher；旧 `bench` 和 `evaluator/bench.py` 留给 S3 删除。没有运行真实 Optimizer、硬件或真实 Case 的 Ref 准备。

## 修改和接口

| 新文件 | 行为与用途 |
| --- | --- |
| [session.py](../../evaluator/session.py) | 创建/读取 Session、冻结路径/schema、workspace registry、全局/Session 锁、保护记录提交、探索/finish 预占、active_action、关闭探索和残留记录处理 |
| [SESSION.md](../../evaluator/SESSION.md) | 内部 API 参数、完整调用约定、锁顺序、预算和中断语义，以及 S3/S4/S5 接入边界 |
| [test_session.py](../../evaluator/tests/test_session.py) | 16 项 Session/资产/路径/恢复/创建和 fork 中断测试 |
| [test_budget.py](../../evaluator/tests/test_budget.py) | 17 项额度、锁、收尾、finish、并发和残留进程测试 |
| [session_test_support.py](../../evaluator/tests/session_test_support.py) | 假 ready Case、预备小文件、假官方 spec builder；禁止 tensor 生成与实现入口执行 |
| [preserve.py](S2_evidence/preserve.py) | 只读保存前后 canonical、tracked、原公共文件及 S0/S1 证据摘要 |

固定设计 API 已提供：

```python
create_session(case_id, optimizer, output=None, budget=None) -> SessionHandle
read_session(session_root, repair=True) -> SessionRecord
register_workspace(session_root, workspace_path, parent_id=None) -> workspace_id
reserve_action(session_root, event_seq, kind, *, workspace_path, mode=None,
               source_manifest) -> usage
close_exploration(session_root, reason) -> None
```

额外的内部生命周期 API：`execution_lock`、`resolve_workspace`、`repair_assets`、`fork_workspace`、`mark_action_starting`、`set_action_process`、`complete_action`、`reconcile_active_action`。它们未增加公开 CLI，也未给旧 parser 添加 Session 路由、搜索、别名或回退。

`optimizer` 接受 backend 字符串或 `{id?, backend, profile?, command?, tool_version?}`；`output` 接受 `runs_root`/`snapshots_root`，默认 benchmark 下的 runs/snapshots；预算默认 20 次探索和 1 次 finish。探索额度要求正整数，finish 固定为 1，拒绝 bool 与额外预算字段。实际命令、工具版本可以由 Launcher 提供；未提供时明确保存 null，不伪造版本或命令。

## 创建与预检查

创建先检查参数和输出路径，取得固定全局锁后检查资产；分配独占的新 Session/Snapshot 目录，捕获固定保护范围和应不存在的记录；**直接调用 `workspace.init_workspace`** 创建 worker_000，登记只读 context，再通过 S1 一次提交 Session 与空 events。测试用 wraps 断言两次 Session 创建恰好调用两次 init，没有旧 bench 子进程。

资产检查覆盖：

- Case/certification schema 与身份、ready certification、ready Ref cache；public comparator SHA-256 必须与认证记录匹配，比较输出声明与 correctness 输出一致。
- Seed 入口与可安全复制的完整源码树、config、Case evaluator、Ref、Oracle、官方 Fixture builder 和所选 environment。
- 恰好 fixture_000，manifest 的身份及固定 in/out 目录；调用既有官方 builder **仅描述 spec 名称**，逐个检查普通输入、标量输入、mutating 初始输出状态输入和声明的 Ref 输出为普通非空文件。
- ref_performance ready，fixture_000 median 必须为有限正数；0、负数、Inf、NaN、bool、null 均拒绝。

没有调用 `create_tensor`、`golden.run`、Candidate/Ref/Oracle 入口或 prepare-ref。该检查验证现有资产准备和已记录认证，**没有重新证明 tensor 内容/shape 的正确性或执行新的 Ref 认证**。本次 preflight 实际只运行于小型假 Case，不创建真实 Case Session。

保护范围复用 S1：当前 Case/sealed 全树、sealed family 的 Python helper、所选 environment、公共 evaluator/launcher、现有命令和 golden；context 单独分组。Session/events/final_result 为 records，final_result 初始为 expected absence。Candidate 当前源码、输出、日志和锁不被冻结回退。

handled 创建失败只删除本次创建且尚未发布的目录；已有哨兵保留。开始记录发布后的异常保留恢复材料，不删除可能已由 manifest 提交的 Session。真实子进程在首次工作记录替换后退出，状态读取恢复出完整初始 Session 和空事件文件。

## 身份、路径和合法更新

SessionHandle 保存创建时的绝对 Session、Snapshot、保护和初始 workspace 路径。记录包含固定 schema 字段、Case/Optimizer 身份、状态、探索关闭标记、预算/usage、registry、next workspace/event counters、active_action、环境、公开 modes、时间及固定资产 hash 摘要。

读取先使用 S1 验证 manifest 与记录恢复来源，核对本 Session 的记录路径，再修复 records；schema、固定资产范围/hash 摘要和路径绑定、registry context、事件编号与预算 ledger 都需一致。额外 records 不能指向其他 Session，context 只能绑定本 Session 对应 workspace。控制 manifest/backups 仍是受信 Bench 控制数据，不把 hash/chmod 描述为操作系统隔离。

workspace 只接受 registry 中的根目录。拒绝子目录、未登记目录、另一 Session、越界路径和所有链接组件；先检查 raw 路径再规范化，因此 `link/../worker_000` 也被拒绝。合法相对路径可定位兄弟 workspace。workspace.json/profiling metadata 被伪造后，Case、入口和权限仍取受保护 Session/恢复后的 canonical 内容。

状态读取只持短 `.session.lock`，只修 records，不获取执行锁、不修固定资产/context。测试在另一进程长期持全局锁且 active_action 已预占时完成读取和关闭探索，证明查询无需等待测量收尾。

records 改写后从**最新 Bench 提交**恢复，usage/event 不退回初始状态。状态、usage、event counter 和 active_action 预占在一次 `commit_records` 内保存；完成事件和 active_action 清除也在一次提交内保存。合法提交后 `check_only` 一致。普通资产恢复保留 worker_000 原始 starting_point，同时保留当前 Candidate/helper 编辑。

## 锁、预算和最后一次 action

全局锁固定为 `<benchmark_root>/.ptobench/execution.lock`，不随 runs_root 改变。create/fork 自动获取；reserve/process identity/complete/reconcile/asset repair 要求调用者已持全局锁。固定顺序为全局锁→短 Session 锁→S1 内部完整性锁；Session 锁下等待全局锁或递归锁会明确失败。完整复制和固定文件检查在 Session 锁外完成。

调用方按 [SESSION.md](../../evaluator/SESSION.md) 先修复保护内容、检查请求/registry/mode，再通过 S1 复制得到稳定 frozen source manifest，才预占一次 action。S2 校验 manifest 格式/hash、路径、profile 开放、保护状态和事件编号，在重取短锁后再次检查状态，防止检查期间关闭探索后仍扣费。S3 负责实际冻结目录和执行后检查；本阶段不启动 Evaluator。

evaluate 与 runtime/pmu/compiler 每次各扣一次探索。检查失败、等待全局锁和 fork 不扣；预占后的 action_failed、integrity_failed、interrupted、启动失败/infrastructure_error 都不退还。多进程竞争由固定执行锁与短记录锁共同串行化。

达到探索上限立即 exhausted，但 active_action 保留，收尾不再经过“接受新探索”判断。close_exploration 也保留 active_action，并保留第一次 close_reason。真实多进程测试在 exhausted+active 状态关闭探索，随后仍提交 PASS result 与 Snapshot 引用；这里保存的是**假 action 的记录与引用**，不是宣称本阶段已发布真实 Snapshot。

finish 只供内部路径使用，要求已关闭探索、没有 active action、有 clean final/workspace 和有效源码清单。它只扣独立的 finish_actions。探索耗尽不阻止一次 finish；finish 失败不退还，重复调用不再次接受。完成 finish 根据调用方证据设置 completed/failed；候选选择、真实 clean evaluate、final_result 和完整幂等 finalize 属于 S4。

重复 action completion 返回原事件，包括下一次 action 已 active 的情况，不改写新 action、不新增事件或扣费。只读 schema 检查还验证 accepted events+active 与两种 usage 完全一致。

## fork 和残留记录

fork 从 Session Case 记录取得 canonical config，复用 `copy_candidate_tree`/`write_context`，不调用依赖 parent metadata 身份的旧 fork 管理路径。先持久消耗 Bench 分配的编号，再在临时目录完整复制/context 生成、rename 发布、登记 context、提交 registry。正常失败清除临时树；已发布但 registry 尚未提交的完整目录留作受控 orphan，未登记就不能执行，编号不复用。registry manifest 已提交后的中断则由 S1 恢复完整新 registry。

active_action 记录 owner PID/start ticks/group、source hash、操作/模式/workspace、event_seq 和启动时间，并区分 `not_started`、`starting`、`running`。执行方在 spawn 前提交 starting，spawn 后绑定独立进程组身份。

- 活跃 owner 的操作保持 pending，不因 exhausted 或查询而清除。
- owner 已死亡且 child 确认死亡，或明确尚未 spawn：写一次 interrupted、保留扣费、关闭探索并清除 active。
- 在 spawn/PID 绑定之间死亡，或真实 child/已拥有进程组成员仍存活：标记 infrastructure failure，保留 active 证据并阻止 finish。
- S2 不杀未知进程；S3/S5 执行实际进程组清理。测试启动的假 orphan 由测试通过 pipe 释放并确认退出。
- 固定资产恢复来源损坏时，能读取的 Session 标为 failed/closed，usage 不回退；记录自身恢复来源损坏则明确失败，不伪造正常事件或重新建立基线。

## 测试命令和原始结果

所有命令从仓库根目录使用 `.venv-official/bin/python`，保存日志时启用 `pipefail`；未 skip 或 expectedFailure。

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_session.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S2_evidence/session_final.log

set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_budget.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S2_evidence/budget_final.log

set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_*.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S2_evidence/unittest_final.log
```

| 最终证据 | 退出码 | 实际结果 |
| --- | ---: | --- |
| [session_final.log](S2_evidence/session_final.log) | 0 | **16 tests / 3.314s，OK** |
| [budget_final.log](S2_evidence/budget_final.log) | 0 | **17 tests / 5.329s，OK** |
| [unittest_final.log](S2_evidence/unittest_final.log) | 0 | **74 tests / 11.935s，OK**；33 项 S2 + 原有 41 项 |

阶段中发现并修正的失败保留原始日志：

- [session_first.log](S2_evidence/session_first.log)：11 tests、3 ERROR。测试未先 chmod 临时只读 context/backup 就尝试篡改，改为显式处理临时文件权限后测试通过；未降低只读恢复标准。
- [budget_first.log](S2_evidence/budget_first.log)：11 tests、2 ERROR。假 profile 字典缺 description，失败预检查后又复用非空 staging；补全假 profile、使用新的临时 staging 后通过。
- [budget_orphan.log](S2_evidence/budget_orphan.log)：17 tests、1 FAIL。最初的纯 fork 假 child 继承 owner 的 flock descriptor，使恢复等到假 child 退出。改为真实 exec 子进程、独立进程组和 pipe，验证实际 subprocess 关闭锁 FD 的语义；没有修改锁实现或跳过 live orphan 验收。
- 中间通过日志也保存在 S2_evidence；最终源文件/日志 SHA-256、逐项验收映射见 [validation_summary.json](S2_evidence/validation_summary.json)。

关键行为的实际输出与断言位于最终日志和测试：

| 验收 | 实际证据 |
| --- | --- |
| 最后额度竞争 | 两组均 5 进程；limit=1 接受 1/拒绝 4，limit=3 接受 3/拒绝 2，usage 未超限 |
| 跨 Session 全局锁 | 不同 Optimizer/Case/runs/snapshots，waiter 在持锁期间 usage=0，释放后只接受一次 |
| 最后 accepted action | 持锁期间读取 exhausted+active，关闭探索后正常保存 PASS completion，active 清除 |
| 创建崩溃 | 真实进程 exit 23；manifest 已发布后恢复完整初始 Session 与空事件 |
| fork 崩溃 | registry manifest 前/后分别保持旧/新 registry；两种情况下一次均 worker_002，usage=0 |
| completion 崩溃 | manifest 前保持 active+0 events；manifest 后恢复 cleared active+1 event；两者 usage=1 |
| 死亡残留 | owner/child 真正退出后只写 1 interrupted event；重入无重复扣费/事件 |
| 活跃 orphan | 真正存活的独立 PID/group 被保留，Session failed、finish 被拒绝；测试随后释放 child |
| 链接与记录 | 外部 sentinel 保持原字节，Session record 链接只替换其自身；跨 Session/额外外部记录拒绝 |

完整回归继续覆盖 192 个 Case inventory、156 ready、183 个现存 comparator evaluator；9 个 pending 缺 evaluator 明确 NOT_RUN，未将其算作实际 Case 认证。没有执行正式 Fixture builder、Ref 准备、NPU、真实 profiling、Optimizer 或正式最终测量。

## 原有修改与资产保护

[before.json](S2_evidence/before.json) 保存开始时保护基线；[preservation.json](S2_evidence/preservation.json) 与前后 git status 保存结束时核对结果：

- **297 个**开始时 tracked 文件 SHA-256 未变化，index 未变化。未 reset、clean、暂存或提交。
- 原有 evaluator 代码/测试/README、旧命令以及 S0/S1 原报告和证据的内容 hash 全部未变化。本次公共实现与使用说明均为新增文件。
- canonical cases/sealed/environments 共 **6701 个**非缓存文件，没有新增、删除或变化；3709 个非 `.pt` 文件内容 SHA-256 一致。
- 2992 个 `.pt` 共约 216.98 GiB 核对 size/mtime_ns/mode，**未读取全量 tensor 字节 hash**。

所有新运行资产、哨兵、源码修改和假进程均在 TemporaryDirectory；写入正式仓库的仅为新增公共模块/测试/说明、S2 报告与证据。没有改变真实 Case/config/comparator/Seed/Fixture/Ref/Oracle/Case-local evaluator 或测量设置，也未新增 Docker、Supervisor、socket 服务或独立项目。

## 进入 S3 的条件与保留阻塞

S2 创建、准备资产校验、registry/path、状态、两层锁、预算预占、finish 独立、合法记录、最后 action 收尾和残留处理均有当前实现与 CPU 行为证据。**S2 无未解决公共实现阻塞，可以进入 S3；本轮没有执行 S3。**

S3 需要接入五种新 CLI 操作、稳定 staging、实际 Evaluator 子进程和双路日志、事前恢复/事后 check_only、validated result、权威记录/Snapshot、拒绝/fork 事件及诊断发布；按 SESSION.md 使用 caller-held 全局锁、spawn boundary 和 completion。然后删除旧入口/parser，并修改新建 context/README 的公开 mode 展示。本阶段保留的旧说明和 mode 展示属于尚待 S3 的接口切换，不作为新 CLI 验收通过。

S0 记录的 6 个 ready Case 缺 Seed 与 36 个 pending Case 资产缺项继续保留 BLOCKED，归 Case 迁移/认证/发布负责人；本次未修改或重新认证这些真实资产。它们不阻止 S3 的假 Evaluator 公共实现测试，也不能被用于就绪 Session、真实硬件或最终全量验收的 PASS 结论。
