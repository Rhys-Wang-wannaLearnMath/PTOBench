# PTOBench 真实目录 Workspace 修缮：Implementation Plan 拆分总览

**完整主计划：** [`../PTOBench真实目录Workspace与沙箱简化Implementation Plan.md`](../PTOBench真实目录Workspace与沙箱简化Implementation Plan.md)（保留完整任务序列；宿主身份启动的最新执行要求以 Plan 01 为准）

**设计依据：** [`../PTOBench真实目录Workspace与沙箱简化修缮计划.md`](../../PTOBench真实目录Workspace与沙箱简化修缮计划.md)

**本次修订范围：** 根据已验证的宿主 UID/GID、普通登录附加组和干净 Shell Bubblewrap/DAC canary，Plan 01 增加跨用户、跨 Linux/NPU 主机的可移植要求：运行时检测身份和映射，优先无特权清组，仅在需要时使用一次性配置的最小权限启动器；Broker 与 NPU Worker 保留原凭据。当前服务器测得的数字仅为历史观测，不作为实现常量。仍不修改运行代码，也不扩大通用安装器范围；拆分仍按 Task 0–9 一项一次分配。

## 当前执行门槛

此前的生产 Bubblewrap 探针确认 Agent 会继承宿主附加组，并能读取属于其中一个组、权限为 `0640` 的 Candidate canary，因此 Task 1 未通过。随后在当前服务器上验证了 Broker 身份、`setpriv --clear-groups` 和干净 Shell 下的真实 Bubblewrap 映射/DAC canary。那些 UID/GID、组号和 subordinate range 仅记录为该服务器当时的观测值；Plan 01 的实现必须在运行时读取当前用户、有效组和宿主映射，不得将它们固化为常量。

Plan 01 复用 Basic/CANNBot 已有的 `isolation.agent_command` → `View.argv` → Bubblewrap `--unshare-user` 路径。启动时先验证当前宿主身份、user namespace/subuid/subgid 配置与有效组权限；已满足合同则直接走无特权路径，可用普通用户 `setpriv --clear-groups` 时优先使用它；只有这两种方式无法清组时才要求一次性配置的固定 root-owned 启动器。启动器只对 Agent 子进程清组并永久降权，UID/GID 从当前调用者动态取得，绝不以 root 执行 PTOBench 或用户可修改代码。核心测试覆盖不同 UID/GID、subordinate range 和附加组；宿主缺少配置时须提供明确结果与指引并 fail closed。若需要特权配置，当前 Plan 仅验证专用启动器和当前部署，不建设通用跨发行版安装器；用户配置后仍用原 PTOBench 命令一键启动。普通登录环境下需验证实际 UID/GID map、无未授权附加组、零 capability、动态 DAC canary、Broker/NPU 权限未变及 Basic/CANNBot smoke。以上通过后才可开始 Task 2；Plan 02–05 均依赖 Plan 01 完成。

## 子计划与执行顺序

| 顺序 | 子计划 | 原任务 | 目标 | 前置依赖 |
|---|---|---:|---|---|
| 1 | [01-宿主身份与真实Workspace权限.md](01-宿主身份与真实Workspace权限.md) | Task 0–2 | 建立真实宿主身份映射和 Workspace 权限合同 | 无 |
| 2 | [02-Fork发布与恢复.md](02-Fork发布与恢复.md) | Task 3 | 将 Fork 改为挂载外 staging、pending/ready 发布与恢复 | Plan 01 |
| 3 | [03-Agent挂载与只读反馈.md](03-Agent挂载与只读反馈.md) | Task 4–5 | 切换 Agent/Pi Shell 的真实 Workspace 挂载，并直接发布只读反馈 | Plan 01–02 |
| 4 | [04-旧视图清理与无NPU软件验收.md](04-旧视图清理与无NPU软件验收.md) | Task 6–7 | 按调用关系清理旧视图，完成无 NPU 回归和 75 Workspace 扫描 | Plan 01–03 |
| 5 | [05-Pi与Formal-Held-out最终验收.md](05-Pi与Formal-Held-out最终验收.md) | Task 8–9 | 真实 Pi Smoke、Formal/Held-out Evaluate 与 Combined Profile 验收 | Plan 01–04；Task 9 等 Task 8 通过 |

每个原任务只分配给一份子计划。按表中顺序逐份执行；某计划的前置计划未通过时，不得越级执行。每份子计划都保留了原 Task 的目标、文件/函数、Interfaces、步骤、验证命令、预期结果、完成标准和失败处理。

完整主计划中的约束和审查重点适用于全部子计划。以下两节与当前主计划保持一致；每份子计划也包含副本，便于独立交给执行者。

## Global Constraints

- 只保留真实目录 Workspace：workspaces/worker_NNN/{candidate,.benchmark_context,.benchmark_outputs}，不以链接、副本或同步目录呈现给 Agent。
- Agent 和 Broker 必须是不同的宿主 UID；继续使用现有 `evaluator/isolation.py::agent_command` / `View.argv` Bubblewrap `--unshare-user` 路径，并在每台主机运行时检测当前账户可用的 subordinate UID/GID 配置及 Broker 主 GID（`os.getgid()`）。只为 Pi Agent 子进程清除未授权附加组；优先采用已验证的无特权路径，必要时才由 Plan 01 的固定、无参数、一次性配置启动器清组并立即降权。调用者身份动态检测，不硬编码账号、UID/GID 或 subordinate range。Broker 和受信任 NPU Worker 保持原有身份、组与硬件权限；不得让 Agent 获得 host sudo/root，也不得以 root 执行用户可修改代码。sudoers 不使用通配参数或通用 root Shell。
- 每个 Agent Sandbox 只对当前 Session 的 workspaces/ 做一个可写 bind mount；Candidate、Context、Outputs 的权限由宿主文件系统 DAC 强制执行。
- 保留 ptobench evaluate/profile/fork 的语法、Combined 默认、显式 split、预算计数、事件记录、正确性检查、正式延迟、Snapshot、Finalize 和 Candidate Worker 执行路径。
- Basic 与 CANNBot 继续使用各自的 `.env`，配置为 DeepSeek Flash 和 High thinking；PTOmni 尚未作为本轮 Agent 后端验收对象。
- 保留 Pi 私有文件隔离、Gateway、Broker Unix Socket、模型 API 通道、Agent 网络隔离及 NPU/HCCL 受信任执行路径。
- 不修改 Case、Fixture、Seed、Reference、Baseline、CV 或 case_selection 数据；验收结果另存到新的 reports 目录。
- lightweight_e2e 不得成为正式 PTOBench 运行、导入、测试或部署依赖。
- Candidate Worker 内部评测文件的可见范围仍是独立问题，本轮不扩展处理范围，也不声称已解决。

## Review Focus

1. 普通登录带附加组时，Plan 01 所选启动方式是否只清理 Agent 子进程；Broker/NPU Worker 凭据是否保持，Agent user-namespace 映射是否匹配宿主配置，动态生成的 DAC canary 是否拒绝未授权访问。由 Task 1 在生产 Bubblewrap 启动路径验证。
2. Agent 新建 0600 文件或不可遍历目录后，Broker 是否明确拒绝 Candidate 快照而不是生成残缺 Snapshot；由 Task 2 和 Task 7 验证。
3. Fork 在 staging、pending、发布和恢复之间中断时，是否出现可被 CLI 使用的半成品或重复 Worker ID；由 Task 3 的故障注入测试验证。
4. Pi Shell 子 Sandbox 是否仍能写 Candidate、不能改 Context/Outputs；必要 Unix Socket 是否允许映射 UID 连接而不开放目录枚举；由 Task 4 的生产 Bubblewrap 集成测试验证。
5. 诊断文件是否仍可读、只读，且路径指向 Agent 可见的 Workspace；由 Task 5 的发布测试和 Combined Profile 抽查验证。

## 已核实的代码边界

- evaluator/workspace.py::init_workspace 创建 candidate/ 和 .benchmark_context/，当前不预建 .benchmark_outputs/；_readonly 目前只设文件权限，未统一收紧目录权限。
- evaluator/session.py::fork_workspace_locked 当前把临时目录放在 workspaces/ 内，rename 后才登记 Worker；真实挂载后需要调整 staging 位置和发布状态。
- evaluator/sandbox_supervisor.py::Broker 当前创建 exports、candidates、public 三套视图；`export`、`sync_source`、`feedback` 和部分路径改写服务于此视图。`serve` 还调用 `rewrite` 格式化 CLI 输出，清理时须保留或替换这项 Agent 可见响应路径功能。
- evaluator/isolation.py::agent_command 当前分别挂载 Workspace、Candidate 池、public 和 Pi 私有目录；View.argv 自行建立 user namespace，但没有已验证的不同宿主 UID 映射。
- evaluator/pi_shell_wrapper.py 当前用只读根视图再单独挂载 /ptobench/candidates；Candidate 改为真实目录后需要改为绑定当前 Session 的 workspaces/。
- evaluator/session_actions.py::_publish_diagnostics 已将归档诊断发布到原始 Session Workspace；Broker.feedback 还会复制诊断并重写其中的 staging 路径。
- evaluator/file_tree.py::atomic_bytes 默认创建 0600 文件；权限初始化和诊断发布必须显式设置 Agent 所需的组读权限。
- evaluator/snapshots.py::create_snapshot、Session Action 冻结 Candidate、Session 完整性保护和 Candidate Worker 隔离均有独立职责；保留这些机制，不以 Workspace 变更替代。
- evaluator/sandbox_supervisor.py::run_legacy 也调用 `Broker.export` 并挂载 Candidate/public 视图；若 Broker 共用路径被改，需迁到同一个真实 `workspaces/` 视图，否则会回归已有 Basic/CANNBot 的非 Pi 入口。
- evaluator/tests/test_isolation.py 中多个测试直接断言 Candidate/Context/Outputs 符号链接和导出视图，切换后应替换为真实目录与宿主权限测试。

## 任务依赖

Task 0（变更边界检查）→ Task 1（身份映射）→ Task 2（真实 Workspace 权限）→ Task 3（Fork 发布/恢复）→ Task 4（真实目录挂载与跨 UID 通道）→ Task 5（直接反馈发布）→ Task 6（旧视图清理）→ Task 7（软件验收）→ Task 8（真实 Pi Smoke）→ Task 9（Formal/Held-out 与 Combined 验收）。

Task 2 和 Task 3 的权限及 Worker 状态接口是 Task 4/5 的前置条件；Task 7 只能在 Task 6 全部通过后启动；Task 8 只能在 Task 7 通过后启动；Task 9 还必须等待 Task 8 通过。

---

## 最终发布门槛

- Task 1 选择的无特权或一次性启动器路径已在目标宿主配置完成；普通登录下 Pi Agent 的动态宿主 UID/GID 映射、无未授权附加组、零 `CapEff`、动态 DAC canary 均通过，Broker/NPU Worker 权限保持；Task 2/3 的权限、Fork、崩溃恢复测试通过。
- 新建 Session 的所有 Worker 都是普通目录；外层 Agent 只有当前 Session 的 Workspace RW bind mount；Shell 子 Sandbox 继承同根或显式绑定同根，不存在候选池/公开副本/Workspace 链接和同步实现。
- Pi 私有文件、Native 工具、Shell/Python、Broker/Gateway/CANNBot Socket、网络和 CANNBot Extension 边界均通过确定性测试与 Basic/CANNBot 真实 Pi Smoke。
- 反馈只读且不作为正式记录；CLI、Action、Snapshot、Finalize 和 Candidate Worker 合同保持通过。
- Formal 60 与 Held-out 15 的 Evaluate 均通过；Combined Profile 覆盖矩阵和 Basic/CANNBot Pi Broker 的真实 Evaluate/Profile 路径检查均完成；Case、Fixture、Seed、Reference、Baseline、CV 未改变。
- 若环境与 Case 清单没有可执行的跨 Host 样本，报告多机覆盖缺口，不将 8 卡结果表述为多机验收。
- 任一门槛未通过时，不使用旧导出实现兜底，也不对外宣称已完成切换。

## 拆分完整性

- 子计划覆盖原计划 Task 0–9，且每个 Task 恰好出现一次。
- 完整 Implementation Plan 保留为任务序列来源；Plan 01 负责可移植的 Agent 身份检测和启动选择，其他子计划只继承其通过后的权限合同。当前主机身份观测不得转化为硬编码常量。
- 拆分过程不修改 PTOBench 运行代码、不启动 Pi、不运行 NPU。
