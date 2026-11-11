# Fork 发布与恢复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task.

**Goal:** 使 Fork 在挂载外 staging 中构造并以可恢复的 pending → ready 流程发布真实 Worker。

**Architecture:** 复用 Plan 01 建立的真实目录及 `workspace.apply_permissions` 权限合同；保持既有 Session ready registry、CLI、Action、预算和事件合同。本计划只执行原计划 Task 3。

**Tech Stack:** Python 3、Bubblewrap、Linux user namespace、POSIX UID/GID 与目录权限、现有 unittest 测试。

**Spec:** [`升级/1109/PTOBench真实目录Workspace与沙箱简化修缮计划.md`](../../PTOBench真实目录Workspace与沙箱简化修缮计划.md)

**Source plan:** [`升级/1109/plans/PTOBench真实目录Workspace与沙箱简化Implementation Plan.md`](../PTOBench真实目录Workspace与沙箱简化Implementation Plan.md)

**Execution order:** This is Plan 02 of 05. Plan 01（原计划 Task 0–2）完成并通过验收后，Task 3 使用其 `workspace.apply_permissions(root, shared_gid=...)` 和真实 Worker 权限合同。

**Plan boundary:** Execute only original Task 3; do not implement tasks assigned to another split plan. Task 3 的在线 Fork、故障注入、旧 Session 兼容和 CLI/预算合同验证均通过后停止；不得开始 Sandbox 挂载切换。

## Global Constraints

- 只保留真实目录 Workspace：workspaces/worker_NNN/{candidate,.benchmark_context,.benchmark_outputs}，不以链接、副本或同步目录呈现给 Agent。
- Agent 和 Broker 必须是不同的宿主 UID；继续使用现有 `evaluator/isolation.py::agent_command` / `View.argv` Bubblewrap `--unshare-user` 路径，并依据当前宿主配置验证 subordinate UID/GID 映射和 Broker 主 GID（`os.getgid()`）。只为 Pi Agent 子进程清除未授权附加组；Broker 和受信任 NPU Worker 保持原有 UID/GID/groups 与硬件权限。优先使用通过验证的无特权启动方式；必要时才使用 Plan 01 中固定、无参数、一次性配置的最小权限启动器，调用者 UID/GID 动态检测。不得让 Agent 获得 host sudo/root，也不得以 root 执行用户可修改代码。sudoers 不使用通配参数或通用 root Shell。
- 每个 Agent Sandbox 只对当前 Session 的 workspaces/ 做一个可写 bind mount；Candidate、Context、Outputs 的权限由宿主文件系统 DAC 强制执行。
- 保留 ptobench evaluate/profile/fork 的语法、Combined 默认、显式 split、预算计数、事件记录、正确性检查、正式延迟、Snapshot、Finalize 和 Candidate Worker 执行路径。
- Basic 与 CANNBot 继续使用各自的 `.env`，配置为 DeepSeek Flash 和 High thinking；PTOmni 尚未作为本轮 Agent 后端验收对象。
- 保留 Pi 私有文件隔离、Gateway、Broker Unix Socket、模型 API 通道、Agent 网络隔离及 NPU/HCCL 受信任执行路径。
- 不修改 Case、Fixture、Seed、Reference、Baseline、CV 或 case_selection 数据；验收结果另存到新的 reports 目录。
- lightweight_e2e 不得成为正式 PTOBench 运行、导入、测试或部署依赖。
- Candidate Worker 内部评测文件的可见范围仍是独立问题，本轮不扩展处理范围，也不声称已解决。

## Review Focus

1. 普通登录带附加组时，Plan 01 选择的启动方式是否只清理 Agent 子进程；Broker/NPU Worker 凭据是否保持，Agent user-namespace 映射是否匹配当前宿主配置，真实 DAC canary 是否拒绝未授权访问。由 Task 1 在生产 Bubblewrap 启动路径验证。
2. Agent 新建 0600 文件或不可遍历目录后，Broker 是否明确拒绝 Candidate 快照而不是生成残缺 Snapshot；由 Task 2 和 Task 7 验证。
3. Fork 在 staging、pending、发布和恢复之间中断时，是否出现可被 CLI 使用的半成品或重复 Worker ID；由 Task 3 的故障注入测试验证。
4. Pi Shell 子 Sandbox 是否仍能写 Candidate、不能改 Context/Outputs；必要 Unix Socket 是否允许映射 UID 连接而不开放目录枚举；由 Task 4 的生产 Bubblewrap 集成测试验证。
5. 诊断文件是否仍可读、只读，且路径指向 Agent 可见的 Workspace；由 Task 5 的发布测试和 Combined Profile 抽查验证。

## 原计划任务（正文原样迁移）

### Task 3: 将 Fork 改为挂载外 staging 与可恢复发布

**目标及实现方式：** Fork 的 Candidate、Context、Outputs 在 Session 的 Broker-only `staging/` 下完整生成；不能在 Agent 可见的 `workspaces/` 中创建临时目录。持久化 Worker ID 后按 pending → ready 发布；CLI 只解析 ready Worker。中断恢复只处理 PTOBench 明确命名的 Fork staging/pending 项，保留已消耗 ID，不能覆盖用户目录或复用 Worker ID。旧 Session 中没有状态字段的已登记 Worker 按 ready 读取。

**文件：**

- Modify evaluator/session.py：`_validate_record`、`_load_locked`、`_workspace_path`、`register_workspace`、`repair_assets`、`fork_workspace_locked`。
- Modify evaluator/session_actions.py::fork_workspace：维持现有 CLI 结果和事件合同。
- Test evaluator/tests/test_session.py、evaluator/tests/test_agent_cli.py、evaluator/tests/test_isolation.py。

**发布顺序：** 在 Session record 新增可选 `pending_workspaces` 字段，旧 Session 缺失时按空字典读取。锁内消耗 Worker ID 并记录 `{worker_id: {"staging": "staging/forks/<unique-name>", "target": "workspaces/worker_NNN", "parent": "worker_NNN", "context_sha256": "..."}}`；ready `workspaces` registry 仍保持原 `{path,parent}` 格式。Broker-only staging 下构造三个真实目录、校验 Candidate/Context/Outputs、保存只读 Context 内容 hash，并保持 Worker 根目录为 Agent 不可遍历的 `0700`。确认 staging 与 `workspaces/` 位于同一文件系统后，原子 rename 到目标 Worker 路径；目标仍不可由 Agent 遍历时注册 Context protection，再提交 ready registry 并删除 pending 条目；最后才把 Worker 根目录开放为 Agent 可读可进入的 `0750`。因此 Agent 最多短暂看见一个不可进入的 Worker 名称，不会在未登记状态读取其内容。若 Broker 在 ready 提交后、开放权限前退出，Broker 启动恢复必须先校验并修复 ready Worker 权限，再启动 Agent。`resolve_workspace` 只接受 ready registry。恢复时：只有 staging 的 pending 项可安全删除并清除 pending、保留已消耗 ID；若目标 Worker 已发布但仍 pending，只有三个真实目录、Context hash 和权限均符合记录、且能恢复保护信息时才提交 ready，否则把该 Session 保持不可用并报告 infrastructure error，不递归删除不确定目标。不得恢复出部分 Workspace；旧 Session 缺少该字段时仍可读取。

**Interfaces:** Session 注册仍以 workspaces/worker_NNN 为正式路径，ready entry 继续使用既有 `path`、`parent` 字段；pending 状态只保存在 `pending_workspaces`，不改变 CLI/事件/预算合同。`resolve_workspace` 只返回 ready Worker。

- [ ] 增加在线 Fork 测试：已有 Agent Shell 持续运行且 mount 数不变时，Broker 完整构造并发布 worker_001；Agent 立即看到并可写，Broker 可读取；worker_000 与 worker_001 修改互不影响。
- [ ] 在 staging 创建前、pending 记录后、rename 后、Context protection 注册后、ready 提交后和权限开放后注入进程中断；验证 Agent 在 ready 前不能进入目标 Worker，恢复结果、Context 保护、Workspace 列表、预算和 ID 单调递增。
- [ ] pending Workspace 未 ready 时，ptobench evaluate/profile/fork 均不能把它当成有效目标；它不产生硬件 Action 或 Snapshot。
- [ ] 对 Session record 的 `pending_workspaces` 缺失、空值、无效父 Worker、越界 staging 路径、重复 Worker ID 加入拒绝测试；旧记录无该字段时维持既有 ready Worker 读取行为。
- [ ] 检查 staging 在 Agent mount 下不可见，pending Worker 不可进入；失败清理不得触碰已登记 Worker 和 Session 之外的目录。检查 staging 与 `workspaces/` 的 `st_dev` 一致，不满足则在消耗 Worker ID 前拒绝 Fork。
- [ ] 保持 fork 不消耗硬件预算、事件顺序及返回绝对 Workspace 路径不变。

**验证命令与预期：**

    python -m unittest discover -s evaluator/tests -p 'test_session.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_agent_cli.py' -v
    (cd evaluator/tests && python -m unittest test_isolation.FilesystemBoundaryTests.test_live_agent_observes_broker_fork_without_remount -v)

预期：Fork 成功可立即在原 Agent 进程内使用；所有故障注入无半成品被 CLI 接受、无重复 ID、无预算消耗。

**完成标准及失败处理：** 任一恢复边界可能让 unready Worker 被 CLI 接受或进入、使旧 Session 无法读取、暴露 staging，或让 ready Worker 在 Agent 启动后持续不可访问，则停止并修复发布事务；不得以保留旧导出 Workspace 作为回退。

---
