已被 1110 Dedicated Agent User 计划取代，仅供历史参考、不得执行。

# Agent 真实目录挂载与只读反馈 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task.

**Goal:** 让 Agent 及 Pi Shell 使用当前 Session 的唯一真实 `workspaces/` 挂载，并将诊断反馈直接发布为只读 Workspace 内容。

**Architecture:** 复用 Plan 01 的宿主身份/权限合同和 Plan 02 的 Fork 发布合同；保留 Pi 私有文件、Socket、Gateway、网络、Snapshot 与正式记录边界。本计划只执行原计划 Task 4–5。

**Tech Stack:** Python 3、Bubblewrap、Linux user namespace、POSIX UID/GID 与目录权限、现有 unittest 测试。

**Spec:** [`升级/1109/PTOBench真实目录Workspace与沙箱简化修缮计划.md`](../../PTOBench真实目录Workspace与沙箱简化修缮计划.md)

**Source plan:** [`升级/1109/plans/PTOBench真实目录Workspace与沙箱简化Implementation Plan.md`](../PTOBench真实目录Workspace与沙箱简化Implementation Plan.md)

**Execution order:** This is Plan 03 of 05. Plan 01 与 Plan 02（原计划 Task 0–3）已完成并通过；Task 4 依赖真实 Workspace 权限及可恢复 Fork，Task 5 依赖 Task 4 的真实目录路径。

**Plan boundary:** Execute only original Task 4–5; do not implement tasks assigned to another split plan. Task 4–5 的 Sandbox/Socket/私有文件边界与直接反馈测试均通过后停止；不得在本计划中删除旧视图实现。

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

### Task 4: 让 Agent 与 Pi Shell 只挂载真实 workspaces/

**目标及实现方式：** 外层 Agent Sandbox 对当前 Session 的真实 `workspaces/` 只做一个可写 bind mount。内核文件权限继续限制 Candidate、Context、Outputs。Pi Shell 的子 Sandbox 不再挂 Candidate 池；它必须继承或显式 bind 同一个真实 `workspaces/`，不得按 Worker 分别挂载。Pi 私有目录、会话目录、CLI Socket、Gateway Socket 和系统依赖保持现有隔离合同。由于跨 UID 后 DAC 也作用于 Unix Socket 和 Pi 自己的配置/会话文件，必须为这些必要通道设置最小组权限并实际验证。

**文件：**

- Modify evaluator/isolation.py::agent_command、View.argv。
- Modify evaluator/sandbox_supervisor.py::Broker.__init__、Broker.serve、run_private_pi、run_legacy。
- Modify evaluator/pi_shell_wrapper.py::main。
- Modify evaluator/network_proxy.py::serve_api、bridge；modes/cannbot/tools.py::ToolServer.__init__。
- Modify modes/_shared/provider.py::prepare、modes/_shared/recording.py::collect、evaluator/isolation.py::install_pi_private_tools；mapped Agent UID must still use Pi private files, while Broker can collect session files after Agent exit.
- Modify modes/basic/runner.py::run、modes/cannbot/runtime.py::run_dispatch：继续经现有 `isolation.agent_command` 启动；Task 1 的凭据守卫必须覆盖这些调用点，不新增 launcher。
- Test evaluator/tests/test_isolation.py。

**Interfaces:** Broker CLI 仍经原 Unix Socket 接收 evaluate/profile/fork 请求并验证 Session/Workspace；Agent 的 cwd 和 CLI 请求继续使用注册 Workspace 的真实绝对路径。RPC、模型 Gateway 与 Agent 环境变量名称只在删去导出路径变量时调整。

- [ ] 让 `agent_command` 把当前 Session 的 `workspaces/` 以一个 RW bind mount 挂载到原绝对路径；移除 Candidate pool、public mount 和 `PTOBENCH_AGENT_WORKSPACE_ROOT` / `PTOBENCH_PUBLIC_DIR` 依赖。
- [ ] `run_private_pi` 与 `run_legacy` 都只绑定同一真实 `workspaces/`；在 `run_legacy` 中只把既有 Agent 可写记录目录按共享 GID 暴露为可写，其他 method/session 记录仍只读或不挂载。保留非 Workspace 历史归档行为，不因迁移改变 backend 选择、网络代理或记录合同。
- [ ] 给 run_legacy 需要由 Agent 写入的 method/agent 目录设置 Broker 主 GID 和组权限；测试只覆盖现有 fake Codex 流程，不新增 Codex 功能或真实 Codex 验收。
- [ ] Pi Shell 子 Sandbox 移除 `/ptobench/candidates` mount；检查其 `--ro-bind / /` 后能否继承外层 Workspace RW mount。若内核/bwrap 将该挂载也降为只读，只在这个子 Sandbox 中对整个 `workspaces/` 增加一个 RW bind mount；无论哪种路径都不得按 Worker 新增 mount。保留 Pi private 与 agent session 的 tmpfs 遮蔽。
- [ ] 为 `Broker` CLI RPC、Model Gateway 和 CANNBot ToolServer 的 Unix Socket 设置 Broker 主 GID、socket `0660`、不可列目录但可遍历的父目录 `0710`；确认 Agent 能连接，其他无关文件仍不可枚举或读取。覆盖 `Broker.__init__`、`network_proxy.serve_api`、`ToolServer.__init__` 和三种 socket 的建立/清理。
- [ ] 将 `agent/pi`、`agent/native` 和命名 session-dir 设为共享组可写；`settings.json`、`models.json`、session 和配置锁的现存文件让 Broker 可读；`install_pi_private_tools` 的 shell wrapper 可执行、`shell.json` 可读。让 Pi 创建普通新文件时使用组可读权限（在 Agent bridge 子进程处设置 `umask 0002`，不改变 Broker 全局 umask）；验证 Pi 的原子保存和 Broker 退出后归档。
- [ ] 外层、Shell 子 Sandbox 与 Pi Native 文件工具分别验证：Candidate 可写；Context/Outputs、Worker 与 Workspace 根不可写；Pi Shell/Python 不能读写 Pi 私有配置/会话；Native 工具由 File Guard 拒绝越界读写；其他 Session/Case、Evaluator/Launcher/Broker 源码、上游密钥不可访问。
- [ ] 更新 isolation manifest 测试：唯一 Workspace mount 的 source/destination 是当前 Session 的 workspaces/，access 为 rw；不出现 /ptobench/candidates、/ptobench/public 或 Workspace 子目录链接。其他 Gateway/CLI/私有目录 mount 仍按现有合同存在。
- [ ] 验证 model Gateway 请求成功且上游 key 只在 Gateway 进程；CLI RPC 和 CANNBot socket 可用；Agent Web Search/Web Fetch 未开放、公网连接失败；受信任 NPU Runtime/HCCL、管理员 SSH 不在 Agent Sandbox 测试中改动。

**验证命令与预期：**

    (cd evaluator/tests && python -m unittest test_isolation.FilesystemBoundaryTests.test_direct_workspace_mount_and_unix_socket_permissions -v)
    (cd evaluator/tests && python -m unittest test_isolation.FilesystemBoundaryTests.test_pi_private_and_shell_boundary_with_mapped_uid -v)
    (cd evaluator/tests && python -m unittest test_isolation.FilesystemBoundaryTests.test_cannbot_socket_with_mapped_uid -v)

预期：外层 Agent manifest 只有一个当前 Session 的 Workspace RW mount，没有 Candidate/public mount；同一真实目录的动态 Fork 可见。Shell 子 Sandbox 继承该目录或仅有一个同根目录 RW mount。两个 Agent 文件访问面都不能改变 Context/Outputs；CLI、Gateway、CANNBot Socket 可连接，Pi 私有文件归档成功。

**完成标准及失败处理：** 如果 Candidate 只能靠额外挂载才能写，或保护目录可被子 Sandbox 绕过，停止，不调整只读权限或回退到链接视图。

---

---

### Task 5: 直接发布只读反馈并清理导出同步链

**目标及实现方式：** 让现有 Session Action 直接更新真实 Workspace 的 `.benchmark_outputs/`，并在发布时设置 Broker 所有、Agent 组可读而不可写的权限。仅将公开输出副本中 Agent 不能访问的 frozen/staging 路径改写为当前 Workspace 的反馈路径；权威 archive、Action/Event 记录和 Snapshot 原样保留。

**文件：**

- Modify evaluator/session_actions.py::_publish_diagnostics；增加私有 `_rewrite_output_paths(text: str, *, frozen: Path, original: Path) -> str`，只转换本次 frozen Workspace 下 `.benchmark_outputs/` 前缀，不改数值、其他绝对路径或权威归档。
- Modify evaluator/workspace.py::_readme。
- Modify evaluator/sandbox_supervisor.py::Broker.serve 和 Broker 类。
- Test evaluator/tests/test_session_actions.py、evaluator/tests/test_isolation.py、evaluator/tests/test_workspace.py。

**Interfaces:** _publish_diagnostics(original, frozen, internal_mode, action_result, raw_result, *, root, artifact_index) 继续接受现有参数；只改变真实 Workspace 的反馈发布位置/权限，不改变 ActionResult、events、authority_result、archive 和 Snapshot 数据合同。

- [ ] 将当前 `Broker.feedback` 对输出副本的读取/复制改为不再需要；在 `_publish_diagnostics` 临时发布树中执行必要的 JSON 路径转换。单独检查 `Broker.rewrite` 的 `serve()` 调用：若它仍负责 CLI 即时输出中 Agent 可见的路径，保留一个窄用途 formatter 并改为真实 Workspace 路径；没有可见调用后才删除。不能把 authority result、archive 或 Snapshot 路径改写。
- [ ] 在发布 staging 树中把目录设为 `0750`、文件设为 `0640`，保留 Broker 所有权和原子目录替换能力；`latest_result.json`、`latest_session_result.json`、`capture_receipt.json` 创建时显式设为 `0640`。测试 Agent 无法创建、写入、删除、改名或 chmod Outputs 及其子目录。
- [ ] 保留同一 Profile 重跑时完整替换旧反馈集的行为；Broker 原子更新后，正在运行的 Agent 立即能读到新反馈。
- [ ] 验证 Context 的 integrity protection、Action archive、权威结果、Snapshot 选择和 Finalize 仍使用原始受信任文件，不信任 Agent 可读的反馈副本。

**验证命令与预期：**

    python -m unittest discover -s evaluator/tests -p 'test_session_actions.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_snapshots.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_workspace.py' -v
    (cd evaluator/tests && python -m unittest test_isolation.FilesystemBoundaryTests.test_direct_feedback_is_readonly_and_refreshes_in_place -v)

预期：公开诊断路径可读且只读，Action archive/Snapshot 校验通过，重复发布替换完整，原始记录无变化。

**完成标准及失败处理：** 若输出不可读、权限可被 Agent 修改、展示路径仍指向 Agent 看不到的 staging，或公开输出可以改变正式评测结果，停止并修正发布过程。

---
