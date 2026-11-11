已被 1110 Dedicated Agent User 计划取代，仅供历史参考、不得执行。

# PTOBench 真实目录 Workspace 与沙箱简化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task.

**Goal:** 让受限 Pi Agent 直接使用当前 Session 的真实 workspaces/ 目录，并以真实宿主 UID/GID 和文件权限保护 Candidate、Context、Outputs。

**Architecture:** Broker 继续在受信任宿主路径运行，Workspace 与 Session、Snapshot、CLI 的正式记录仍由现有 PTOBench 管理。Pi 外层 Agent Sandbox 对当前 Session 的 `workspaces/` 做一次可写 bind mount；目录权限决定 Candidate 可写、Context/Outputs 只读，Fork 在挂载外构造后发布到该目录。Pi Shell 是单独的子 Sandbox，复用同一真实 Workspace 根，不再挂 Candidate 池或逐 Worker 挂载。移除导出 Workspace 和同步回写，不保留第二种 Workspace View。Basic/CANNBot 共享 Broker 时都沿用这一真实目录实现；本轮只做 Pi 真实运行验收，其他已有入口只做回归。

**Tech Stack:** Python 3、Bubblewrap、Linux user namespace、POSIX UID/GID 与目录权限、现有 unittest 测试。

**Spec:** 升级/1109/PTOBench真实目录Workspace与沙箱简化修缮计划.md

## Global Constraints

- 只保留真实目录 Workspace：workspaces/worker_NNN/{candidate,.benchmark_context,.benchmark_outputs}，不以链接、副本或同步目录呈现给 Agent。
- Agent 和 Broker 必须是不同的宿主 UID；Agent 继续由现有 `evaluator/isolation.py::agent_command` / `View.argv` Bubblewrap 路径通过 `--unshare-user` 启动，并映射到已验证的 subordinate UID 与 Broker 主 GID（`os.getgid()`）作为共享组。实际调用入口必须确认宿主进程 `os.getgroups() == []` 且 `CapEff` 为零；不满足时拒绝启动。不得把附加组带入 Agent，也不得调用 `setgroups`、引入 systemd 服务或 privileged helper 来绕过当前凭据。
- 每个 Agent Sandbox 只对当前 Session 的 workspaces/ 做一个可写 bind mount；Candidate、Context、Outputs 的权限由宿主文件系统 DAC 强制执行。
- 保留 ptobench evaluate/profile/fork 的语法、Combined 默认、显式 split、预算计数、事件记录、正确性检查、正式延迟、Snapshot、Finalize 和 Candidate Worker 执行路径。
- Basic 与 CANNBot 继续使用各自的 `.env`，配置为 DeepSeek Flash 和 High thinking；PTOmni 尚未作为本轮 Agent 后端验收对象。
- 保留 Pi 私有文件隔离、Gateway、Broker Unix Socket、模型 API 通道、Agent 网络隔离及 NPU/HCCL 受信任执行路径。
- 不修改 Case、Fixture、Seed、Reference、Baseline、CV 或 case_selection 数据；验收结果另存到新的 reports 目录。
- lightweight_e2e 不得成为正式 PTOBench 运行、导入、测试或部署依赖。
- Candidate Worker 内部评测文件的可见范围仍是独立问题，本轮不扩展处理范围，也不声称已解决。

## Review Focus

1. 当前 Shell 凭据是否确实到达实际 Agent 启动点：宿主进程 UID/GID 应为 `1001/1001`、`os.getgroups()` 为空、`CapEff` 为零；Agent 的宿主 UID/GID 映射是否正确、无额外宿主组或有效 capability，且组 10/1000 DAC canary 不可读。由 Task 1 在生产 Bubblewrap 路径和计划使用的队列/tmux 上下文验证。
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

### Task 0: 固定本次变更边界

**目标及实现方式：** 先保留当前工作树中的已有修改；按本计划明确的文件清单检查并记录基线差异。不得用清理、回退或全量暂存来制造“干净”状态。实现期间每项只触及该任务列出的 PTOBench 运行代码、测试和本计划要求的文档。

**文件：** 只读检查 git status、目标代码和 Case/Fixture/Seed/Reference/Baseline/CV 文件；不修改它们。

**验证命令与预期：**

    git status --short
    git diff -- evaluator launcher modes/basic modes/cannbot ptobench pto-launcher
    git diff --cached -- evaluator launcher modes/basic modes/cannbot ptobench pto-launcher

预期：能区分既有改动与本计划新增改动；计划后续验收输出路径位于新的 reports/ 子目录。

**完成标准及失败处理：** 若实现工作无法与既有改动区分，先停止并整理逐文件变更清单；不得回退不属于本计划的内容，不得使用 git add -A。

---

### Task 1: 验证并闭锁真实宿主凭据与 Bubblewrap 身份边界

**目标及实现方式：** 复用当前生产路径，不增设宿主服务或另一层启动器。Basic 和 CANNBot 已通过 `evaluator/isolation.py::agent_command` 生成隔离命令，再由现有 Runner 启动；`View.argv` 使用 Bubblewrap `--unshare-user`、`--cap-drop ALL`。当前已验证的 user namespace 映射仍为 Agent 宿主 UID `165536`、共享 GID `1001`。调用者报告已从干净 Shell 重启 Codex，宿主 UID/GID 为 `1001/1001`、附加组为空、`CapEff=0`。在 `agent_command` 构造启动命令前校验实际 Runner 凭据：有效 UID/GID 必须与当前 Broker 账号相符，`os.getgroups() == []` 且 `CapEff` 为零；不符合时 fail closed。运行过程中不调用 `setgroups`，不更改账户/系统配置，不新增 systemd、setuid、capability helper、launch socket 或 bootstrap。仅当生产 Agent 和计划使用的队列/tmux 启动上下文都通过组 canary 后，才允许开始 Task 2。

**文件：**

- Modify `evaluator/isolation.py::agent_command`：在调用现有 `View.argv` 前验证当前宿主有效 UID/GID、`os.getgroups()` 和 `/proc/self/status` 的 `CapEff`；组不为空或 capabilities 非零时抛出清晰错误，不能启动 Pi。
- Modify `evaluator/tests/test_isolation.py`：新增对干净凭据通过、附加组非空时拒绝启动、有效 capability 非零时拒绝启动的测试；扩展现有生产 Bubblewrap 测试以核对 Agent 映射和真实 DAC。
- Do not modify `modes/basic/runner.py::run`、`modes/cannbot/runtime.py::run_dispatch` 或启动 API，除非测试证明现有 `agent_command` 边界未覆盖其中某条 Pi 路径；任何必要接入必须继续调用同一个守卫，不引入独立 launcher。

**Interfaces:** Basic/CANNBot 继续经当前 `agent_command(...) -> argv: list[str]` 和现有进程启动逻辑运行；无新 IPC、FD 或启动参数。共享 GID 仍为 Broker 主 GID `os.getgid()`，subordinate UID/GID 映射和 Bubblewrap user namespace 由当前 `View.argv` 路径负责。

- [ ] **记录当前宿主前置条件：** 在本轮提供的干净 Shell 中记录 `id -u`、`id -g`、`id -G`、`/proc/self/status` 的 `Groups` 和 `CapEff`；预期 UID/GID `1001/1001`、仅主组 1001、`Groups` 为空、`CapEff` 为零。该记录只对应当前 Shell，不自动证明 tmux/队列 Server 使用同一凭据。
- [ ] **写 fail-closed 测试：** Mock `agent_command` 读取到非空 `os.getgroups()` 或非零 `CapEff` 时，断言在构造/运行 bwrap 前报错；读取到当前干净凭据时继续生成现有 Bubblewrap 命令。异常信息指出启动进程附加组或 capability 不符合要求。
- [ ] **运行生产 Bubblewrap 凭据探针：** 通过 `isolation.agent_command` 启动短命探针，读取 Agent 的 `/proc/self/uid_map`、`gid_map`、`/proc/self/status` 中的 `Groups` 和 `CapEff`，并从宿主侧读取运行中子进程身份。验证 Agent 宿主 UID 与 Broker UID 不同、宿主 GID 为 Broker 主 GID `1001`、没有组 10/1000、没有有效 capability；`Groups` 应为空，若内核对映射主组的显示差异导致它列出 1001，则只能接受列表中包含允许的 1001 且不含任何其他 GID，并以 DAC canary 作为最终授权证据。
- [ ] **运行真实 DAC canary：** 在实际 Agent 可访问的 Candidate bind mount 中放置真实宿主 GID 为 10、1000 的 `0640` 文件，并放置共享 GID 1001 的控制文件。Agent 必须无法读取两个 canary、可按既定合同访问控制文件；Broker 能读取普通 Agent 文件。保留对 Agent 创建 `0600` 文件的明确 Snapshot 拒绝测试。核实 canary 的宿主 `stat` GID 和 mode，无法确认时标记 BLOCKED。
- [ ] **核验真实调度上下文：** 用计划中执行 PTOBench 的同一队列和 tmux 启动方式运行不调用 NPU 的启动探针。旧 tmux server 若仍携带附加组，必须从干净 Shell 重启该 server/队列 worker 后复测；不能只凭新客户端 Shell 的 `id` 判为通过。`agent_command` 的守卫应在脏上下文 fail closed。
- [ ] **保持后续任务门槛：** 只有宿主 Shell、正式 Agent 启动进程、生产 Bubblewrap Agent、目标队列/tmux 上下文以及 GID 10/1000 canary 全部符合合同，才解锁 Task 2。若任一不符，停止并定位是哪一层重新加入凭据；不放宽 DAC、不改挂载、不跳过 canary，也不转回提权服务方案。

**验证命令与预期：**

    id -u
    id -g
    id -G
    rg '^(Groups|CapEff):' /proc/self/status
    (cd evaluator/tests && PYTHONPATH=.. python -m unittest test_isolation.FilesystemBoundaryTests.test_agent_namespace_uses_distinct_host_uid_and_shared_host_gid test_isolation.FilesystemBoundaryTests.test_missing_mapping_refuses_to_start_agent -v)
    (cd evaluator/tests && PYTHONPATH=.. python -m unittest test_isolation.FilesystemBoundaryTests.test_inherited_group_canaries_are_unreadable -v)

预期：第一条显示 UID/GID `1001/1001`、主组 1001，`Groups` 为空，`CapEff` 为全零；后两项通过现有生产 Bubblewrap 路径证明 Agent 映射与 canary 行为。队列/tmux 探针应在计划运行任务的同一启动上下文执行；若上下文尚未清理或结果不同，Task 1 不通过。

**完成标准及失败处理：** 当前 Shell 的新实测满足干净启动前置条件，但尚未验证生产 Agent、队列/tmux 和真实 DAC canary，因此 Task 1 仍未完成，Task 2 不得开始。只有生产路径 fail-closed 测试、映射与 capability 检查、组 10/1000 拒绝读取、共享 GID 1001 合法访问以及实际调度上下文验证全部通过后才解除门槛。若 bwrap 子进程仍出现未授权组，先确认启动子进程的 Shell、tmux server、队列 worker 是否继承了旧凭据，再定位已有生产路径中重新带入组的环节；不新增 systemd 服务、setuid/capability helper，不放宽权限、不绕过失败测试。
---

### Task 2: 初始化真实 Workspace 与权限合同

**目标及实现方式：** 让每个新 Worker 直接包含三个真实目录，并在首次向 Agent 开放前设置权限。Workspace/Worker/Context/Outputs 由 Broker 所有，共享 GID 仅有读取和遍历权限；`candidate/` 使用 setgid 目录和共享组读写权限，原有 Candidate 文件也允许 Agent 编辑。Broker 必须能读取普通 Agent 文件；不可读文件在快照前明确拒绝。

**文件：**

- Modify evaluator/workspace.py：`_readonly`、`_ensure_tuning`、`write_context`、`init_workspace`；增加只服务 Worker 的 `apply_permissions(root: Path, *, shared_gid: int) -> None`，预建 `.benchmark_outputs/` 并设置三个真实目录权限。
- Modify evaluator/session.py::create_session：初始 Worker 的权限规范化必须发生在 `source_manifest` 与 Context protection capture 之前。
- Modify evaluator/file_tree.py::atomic_bytes 的调用点，仅在需要 Agent 读取时显式传入文件 mode；不改 `atomic_bytes` 的安全默认值。
- Test evaluator/tests/test_workspace.py、evaluator/tests/test_session.py、evaluator/tests/test_isolation.py。

**权限值：** 共享 GID 使用 Broker 当前批准的 GID，其他 GID 无权限。`workspaces/` 与 `worker_NNN/` 为 Broker 所有、0750；`candidate/` 及其 Broker 创建的子目录为 Broker 所有、setgid 2770，已有 Candidate 文件和 `tuning.json` 为 0660；Context/Outputs 目录为 Broker 所有、0750，文件为 0640。Agent 新建文件继承共享 GID；若文件模式仍为 0600 或目录不可遍历，Broker 在 Snapshot 前报告具体相对路径并拒绝 Action。

**Interfaces:** `shared_gid` 固定为 Broker 主 GID（`os.getgid()`）；Session 初始 Worker 与 Fork staging 都调用 `workspace.apply_permissions(root, shared_gid=shared_gid)`。只允许真实目录，三个 Agent 可见入口都用 `lstat` 检查，不跟随链接。

- [ ] 先增加权限测试：`worker_000` 下 Candidate、Context、Outputs 均为普通目录；Agent 可编辑 Candidate，不能写/删/改名/chmod Context/Outputs、Worker 或 `workspaces/` 根目录。实现前相关权限断言应失败。
- [ ] 新增 Helper/tuning 测试：Agent 可在 Candidate 中创建嵌套 .py 文件、编辑 candidate/tuning.json、原子替换入口及删除文件；Broker 可读取正常权限文件。
- [ ] 调整 Context 权限递归处理：不仅文件只读，目录也不得由 Agent 写、删、改名或 chmod；初始 Outputs 目录预建且 Agent 只读。
- [ ] 将 Agent 创建的 0600 文件和不可遍历目录转为含相对路径的 `InvalidWorkspaceError`；不能略过文件继续生成 manifest 或 Snapshot。
- [ ] 验证新 Workspace 只包含 `candidate/`、`.benchmark_context/`、`.benchmark_outputs/`；不生成 `mode.json`、`profiling_modes.json`、`fixture_catalog.json`。README 只描述真实 `candidate/`、只读 Context/Outputs 和 CLI 用法；保留并扩展 `test_workspace.py` 对 “Do not inspect hidden test/reference assets” 的 `assertNotIn`；不出现链接、导出或同步指引；Profile correctness / evaluate 说明保持中性准确。

**验证命令与预期：**

    python -m unittest discover -s evaluator/tests -p 'test_workspace.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_session.py' -v
    (cd evaluator/tests && python -m unittest test_isolation.FilesystemBoundaryTests.test_direct_workspace_permissions -v)

预期：工作区文件布局/权限和旧的 Case 内容、Seed 内容断言均通过；失败时不发布部分 Workspace。

**完成标准及失败处理：** 任一入口是链接、Context/Outputs 可由 Agent 改动、Broker 读不到普通 Candidate 文件或 0600 错误未明确报告，都阻止后续挂载切换。修正权限实现，不允许在 Snapshot 中跳过不可读文件。

---

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

### Task 6: 删除已失去调用者的 Workspace View 代码

**目标及实现方式：** 只在 Task 4/5 的真实目录路径通过后清理旧导出 Workspace 机制。依据真实调用关系删除，不按名字批量删除；`run_legacy` 必须先改到相同真实目录路径，不能因它仍调用共享 Broker 而留下第二套 view。

**可安全删除的旧代码（确认全仓无 Workspace 调用者后）：**

- sandbox_supervisor.py::Broker.export、Broker.sync_source、Broker.feedback，以及 Broker.exports、Broker.candidates、Broker.public 这些 Workspace 视图字段。`Broker.rewrite` 只有在 Task 5 证明即时 CLI 响应无需路径转换，或已迁为窄用途 formatter 后才删除。
- isolation.py 中从 PTOBENCH_AGENT_WORKSPACE_ROOT 推导 Candidate 池、分别挂载 public/Candidate 的代码，以及对应旧环境变量。
- pi_shell_wrapper.py 中 /ptobench/candidates 的单独挂载和相关配置。
- test_isolation.py 中要求 Workspace 入口为符号链接、要求导出视图/同步回写的测试；用真实目录、宿主 UID/GID 与直接反馈测试替代。

**明确保留：**

- sandbox_supervisor.py 顶层 copy_tree：run_legacy 仍用它归档 method/agent 历史；删除前先确认并替代这些非 Workspace 调用。
- workspace.copy_candidate_tree、copy_tree、source_manifest：仍用于 Case Seed、Fork、Action freeze、Snapshot 和 Candidate Worker 输入。
- file_tree 的 no-follow、稳定扫描、原子写入；Session/Context 完整性和 Candidate 特殊文件拒绝；Pi File Guard、Shell 隔离、Gateway/Socket、Profile Combined/split 和预算逻辑。
- 不删正式 PTOBench 代码或测试以 lightweight_e2e 的行为替代；不处理 Candidate Worker 评测源码边界。

**文件：** evaluator/sandbox_supervisor.py、evaluator/isolation.py、evaluator/pi_shell_wrapper.py、evaluator/tests/test_isolation.py、必要的 Agent-facing README/帮助文档。

- [ ] 在删除前用全仓 rg 确认旧 Workspace 环境变量、方法和链接路径的调用者；逐项清理只与旧 Workspace View 有关的用法。
- [ ] 删除后再 rg 导出目录、旧环境变量和 `/ptobench/candidates`；运行时不得残留旧 Workspace View。另查所有 `rewrite` 调用，确保只保留经 Task 5 验证仍需要的用户可见 CLI 路径格式化。文档不再要求 Agent 检查链接、导出副本或 Broker 同步。
- [ ] 保留 non-Pi 后端原有支持入口，但其 Workspace 也使用唯一真实目录实现；本轮真实 Agent 端到端验收范围仍仅 Pi。

**验证命令与预期：**

    rg -n 'PTOBENCH_AGENT_WORKSPACE_ROOT|PTOBENCH_PUBLIC_DIR|/ptobench/candidates|Broker\.(export|sync_source|feedback)' evaluator launcher modes
    python -m unittest discover -s evaluator/tests -p 'test_isolation.py' -v

预期：第一条搜索无旧 Workspace View 运行路径；任何保留的响应路径格式化不再读取/复制 Candidate 或 Outputs；非 Workspace 历史归档无回归。

**完成标准及失败处理：** 若旧代码仍被实际调用，先迁移该调用再删除；不能通过保留两种 Workspace 后端或全量重构绕过调用关系检查。

---

### Task 7: 无 NPU 软件回归与 75 个 Workspace 扫描

**目标及实现方式：** 扩展现有 maintainer acceptance 工具以支持按已选 Case/action 运行；补齐 Combined artifact 验证。默认无参数行为保持当前“全部 registered Cases + clean/runtime/pmu/compiler split”合同。

**文件：**

- Modify evaluator/accept_all_cases.py：`AcceptanceRunner.__init__`、`run_case`、`blocked_case`、`summarize`、`run`、`main`；新增 `select_cases(rows, selection_csv)` 和 `action_items(actions, profile_strategy)`。
- Modify evaluator/acceptance_artifacts.py::inspect_artifacts：增加 `mode == 'combined'`，在同一 artifact root 上运行 Runtime、PMU、Compiler 三类现有 validator，全部通过才通过。
- Create evaluator/tests/test_workspace_acceptance.py：对 Formal/Held-out CSV 的 75 个 Case 调用 `preflight_case` 和 `workspace.init_workspace`/`apply_permissions`，临时目录中检查三个普通目录及 mode，不创建 Session Action 或启动 NPU。
- Test evaluator/tests/test_accept_all_cases.py、evaluator/tests/test_workspace_acceptance.py、evaluator/tests/test_agent_cli.py、evaluator/tests/test_isolation.py。

- [ ] 先写 selection/action 测试：空 CSV、缺 `case_id`、重复/未知 ID 在设备检查和 Session 创建前拒绝；保留注册 Case 原顺序。`evaluate` 只生成一项；combined profile 只生成一项；split profile 仍三项；默认仍四项。
- [ ] 让 AcceptanceRunner 按选中的 action 数计算 Session budget、Fork worker 数、摘要行数和 Finalize：只跑 evaluate 或 Combined Profile 时各用一个 Worker，不创建未使用的 Fork；默认四项仍保持四个独立 Worker 和原有预算/Finalize 行为。`--selection-csv`、`--actions all|evaluate|profile`、`--profile-strategy split|combined` 缺省时不改变当前行为。
- [ ] 写 Combined artifacts 测试：Runtime、PMU、Compiler 子验证全部通过才 PASS；分别缺失任一类、空文件或 schema 错误都 FAIL；split 既有报告保持不变。
- [ ] 对 75 个 CSV Case 做临时目录扫描：`lstat` 三个 Workspace 入口均为目录且非链接，mode 符合 Task 2；断言不存在 `mode.json`、`profiling_modes.json`、`fixture_catalog.json`。不创建 `runs/`、Snapshot 或 hardware Action，不写回 Case/Seed/Fixture/Reference。
- [ ] 增加正式代码不依赖 `lightweight_e2e` 的静态检查；仅当正式 import、调用或运行路径引用该目录时失败。
- [ ] `accept_all_cases.asset_roots` 纳入两个选择 CSV 所在的 `case_selection/`，让工具批次前后校验 Case、sealed assets、environment 和选择文件内容不变。

**验证命令与预期：**

    python -m unittest discover -s evaluator/tests -p 'test_accept_all_cases.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_workspace_acceptance.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_agent_cli.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_isolation.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_*.py' -v
    rg -n 'lightweight_e2e' evaluator launcher ptobench pto-launcher

预期：测试通过；默认仍为每 Case 四项；75 个临时 Workspace 检查不创建 Session/硬件记录；正式代码扫描无 import/call 依赖。`rg` 无命中时退出码 1 是预期结果，有命中则失败。若有测试失败或输入资产 hash 变化，停止，不进入 Pi 或硬件验收。

**完成标准及失败处理：** Workspace 权限、CLI、Combined/split、Snapshot、Finalize、隔离和轻量 E2E 独立性检查全部通过后才开始 Task 8。任何 `BLOCKED` 不算 PASS。

---

### Task 8: Basic/CANNBot 真实 Pi Smoke（不占 NPU）

**目标及实现方式：** 增加 maintainer-only Smoke 命令，在临时 Session 中复用 Basic/CANNBot 的既有 `preflight`、`prepare`、`run_private_pi`、真实 `agent_command`、Broker 和 Gateway。Smoke harness 不调用 Launcher 自动 Finalize，避免无 NPU Smoke 意外触发 final clean run。只给 Smoke Session 写短 prompt，要求 Pi 创建 helper 并执行 `ptobench fork .`；Broker 在此场景只允许 `fork`/help，Agent sandbox 不挂 NPU device，因此不会启动 Candidate Worker 或硬件 Action。此测试不改正式 Case、prompt、mode instructions 或 Agent-facing CLI。

**文件：** Create evaluator/pi_sandbox_smoke.py；Modify evaluator/sandbox_supervisor.py::Broker.__init__、Broker.serve、run_private_pi（增加仅供 maintainer harness 调用的 `allowed_operations` 参数，生产默认集合不变）；Modify modes/basic/runner.py::prepare、modes/cannbot/runner.py::prepare、modes/cannbot/runtime.py::run_dispatch（只在受信任 harness request 中替换任务 prompt，并关闭 CANNBot Task 子 Agent 工具；不接受 Agent/experiment YAML 传入此标记）；Test evaluator/tests/test_pi_sandbox_smoke.py。复用 evaluator/session.py::create_session、launcher/methods.py::call/retain_method_reply、evaluator/isolation.py::agent_command。

- [ ] Smoke CLI 接受 `--mode basic|cannbot`、`--case-id`、`--operation fork|evaluate-profile`、`--output-dir`；`fork` 为默认操作。Case 必须来自 formal_60 或 heldout_15 CSV。Profile 从对应 mode 的 `.env` 解析，启动前断言模型为 DeepSeek Flash、thinking 为 High。临时 Session/Workspace 建在 `TemporaryDirectory`，报告和日志仅写新的 `reports/agent_upgrade/direct_workspace/pi_smoke/`。
- [ ] Smoke harness 在可丢弃 Session 中，通过对应 mode 的既有 `preflight` 和 `prepare` 传入受信任内部标记，让 Smoke prompt 在 `retain_method_reply` 和完整性保护前生成；之后复用既有 request/provenance/冻结路径并直接调用 `run_private_pi`，不进入 Launcher 自动 Finalize。Basic prompt 由 `modes/basic/runner.py::prepare` 生成，CANNBot prompt 由 `modes/cannbot/runner.py::prepare` 写入 `method/config.json` 和 `prompt.txt`。禁止准备结束后改写 `prompt.txt`，不得覆盖任何正式 Session 或正式 prompt。
- [ ] `fork` Smoke prompt 只要求读取 Workspace README、在 `candidate/` 创建 helper、执行 `ptobench fork .`、读取子 Worker 的 Candidate 并退出。Broker allowlist 只允许 fork/help 并拒绝 evaluate/profile；CANNBot ToolServer 不开放 Task 子 Agent 工具；确认 Agent sandbox 未挂载 NPU device、Session 中没有 hardware action 或 Snapshot。
- [ ] `evaluate-profile` Smoke prompt 只要求对原 Candidate 依次执行 `ptobench evaluate .` 和 `ptobench profile .`，不编辑代码或 tuning。Broker allowlist 只允许 evaluate/profile/help，Session 预算设为两个 hardware action；Candidate Worker 由受信任 Broker 启动，不给 Agent sandbox 挂 NPU device。此操作用于 Task 9 硬件验收。
- [ ] Basic/CANNBot 各覆盖一个 Formal 和一个 Held-out，共四次真实 Pi 启动；检查加载到的是各自 `.env` 的 DeepSeek Flash / High。CLI/Gateway/CANNBot ToolServer（CANNBot 路径）连接成功，Agent 进程没有上游 key。
- [ ] 权限拒绝项由 Task 4 的确定性 Bubblewrap 探针验收，不依赖模型是否照做；真实 Pi Smoke 只验证 Pi 启动、正常 Candidate 写入、Broker fork 可见、父子 Candidate 独立和会话归档。
- [ ] 每次 `fork` 运行后检查 Session 没有 hardware Action/Snapshot/Candidate Worker；同一 Pi 进程内可见 fork，Agent 修改子 Candidate 后 Broker 可读且父 Candidate 不变。Task 8 只运行 `fork` Smoke。

**验证命令与预期：**

    python -m unittest discover -s evaluator/tests -p 'test_pi_sandbox_smoke.py' -v
    python evaluator/pi_sandbox_smoke.py --mode basic --case-id deepseek_v4_flash_mtp.sample_pro --operation fork --output-dir reports/agent_upgrade/direct_workspace/pi_smoke/basic_formal
    python evaluator/pi_sandbox_smoke.py --mode basic --case-id qwen3_14b.topk_select_k1 --operation fork --output-dir reports/agent_upgrade/direct_workspace/pi_smoke/basic_heldout
    python evaluator/pi_sandbox_smoke.py --mode cannbot --case-id deepseek_v4.prefill_layer_ep2_csa_demo --operation fork --output-dir reports/agent_upgrade/direct_workspace/pi_smoke/cannbot_formal
    python evaluator/pi_sandbox_smoke.py --mode cannbot --case-id deepseek_v4_flash_dspark.prefill_indexer_flash --operation fork --output-dir reports/agent_upgrade/direct_workspace/pi_smoke/cannbot_heldout

预期：四次真实 Pi smoke 完成；`hardware_actions_used == 0`、没有 Snapshot/Evaluator/NPU 调用；模型 API 经 Gateway 工作，三类必要 Unix Socket 均可用。

**完成标准及失败处理：** 任一 Pi 无法启动、Socket/Gateway 失败、权限合同失败、fork 不可见、Agent 能调用硬件 Action 或出现 NPU 访问时，停止 Task 9；修复后用新的临时 Session 和 output-dir 重跑四项。

---

### Task 9: Formal/Held-out Evaluate 与 Combined Profile 硬件验收

**目标及实现方式：** 新实现投入前，使用 Task 7 的同一 acceptance 工具对已选 60 个 Formal 和 15 个 Held-out Case 做正确性 Evaluate；再以 Combined profile 覆盖正式 Case 的三种 Granularity、1/2/4/8 卡及 Held-out 可用的 Granularity/1/2/4 卡。Profile coverage 只存入新 reports；不更新 Baseline/CV，不改选择 CSV。

**文件：** 无新增运行代码；使用 evaluator/accept_all_cases.py 及新验收输出目录。可在 `reports/agent_upgrade/direct_workspace/combined_profile_cases.csv` 保存经审核的 Profile coverage 输入清单。

**当前 selection CSV 已核实：** Formal 60 分布为 operator 12、subgraph 43、model 5；卡数为 1 卡 11、2 卡 28、4 卡 11、8 卡 10。Held-out 15 为 operator 3、subgraph 10、model 2；卡数为 1 卡 10、2 卡 3、4 卡 2、无 8 卡。可用的覆盖样例包括：Formal `deepseek_v4_flash_mtp.sample_pro`（operator/1 卡）、`deepseek_v4.synthetic_token_loop_prefill_flash`（model/2 卡）、`deepseek_v4.prefill_layer_ep4_hca_flash`（subgraph/4 卡）、`deepseek_v4.prefill_layer_ep8_csa_demo`（subgraph/8 卡）；Held-out `qwen3_14b.topk_select_k1`（operator/1 卡）、`deepseek_v4_flash_dspark.dspark_prefill_pro`（model/2 卡）、`deepseek_v4.moe_ep_ep4_flash`（subgraph/4 卡）。

- [ ] 先从正式 selection CSV 和环境 JSON 审核覆盖列表；验证 granularity 与 device_ids 和 CSV 一致，并判断是否有明确跨 Host 配置。8 卡只代表 8 个设备，不代表多机。若存在跨 Host Case，加入覆盖列表；若没有，报告“本轮无可验证的多机样本”，不声称已覆盖多机。
- [ ] 分别运行 Formal 60、Held-out 15 的 `--actions evaluate`；要求每个 Case 的所有 declared Fixture correctness、Snapshot、integrity 都 PASS。Acceptance 结果必须恰好每 Case 一条 clean/evaluate 记录。
- [ ] 用经审核的 coverage CSV 运行 `--actions profile --profile-strategy combined`；每个 Case 必须由 `ptobench profile .` 产生同次 Runtime、PMU、Compiler 三类有效 artifact，缺一项即失败。Combined Profile 不替代 Evaluate 或 clean latency 检查。
- [ ] 用真实 Basic Pi + Formal `deepseek_v4_flash_mtp.sample_pro`、CANNBot Pi + Held-out `qwen3_14b.topk_select_k1` 各运行一次 `--operation evaluate-profile`；确认同一真实 Candidate 经 Broker RPC 完成 Evaluate 和 Combined Profile、结果由 Agent 可读且只读、Action/Snapshot/Integrity 记录通过。每个 Session 正好两项 hardware action，不触发 Launcher 自动 Finish，不改 Candidate/tuning；这是验证 Agent 到 Broker 到 Candidate Worker 的路径，不代替 75-case Acceptance。
- [ ] 硬件任务严格串行；每批前按功耗/温度和任务协调确认可运行，不以 AICore 利用率判断设备忙闲。多卡 Case 检查 HCCL 和完整 device_ids；真实跨 Host Case（若存在）单独串行。
- [ ] 批次前后保留 acceptance 工具的 Case、sealed、environment、selection CSV hash 对比；输出必须在新 `reports/agent_upgrade/direct_workspace/` 目录。确认 Candidate/Fixture/Seed/Reference/selection CSV/正式 Baseline/CV 未修改。
- [ ] 若实现改动触及 Evaluate/Profile 的 Runtime 调用、状态复位、编译、数据采集或计时路径，扩大 Profile/Evaluate 覆盖到所有受影响 Case；不能假定 Workspace 改动天然不影响测量。

**验证命令与预期：**

    batch=$(date -u +%Y%m%dT%H%M%SZ)
    python evaluator/accept_all_cases.py --out "reports/agent_upgrade/direct_workspace/formal_60/$batch" --selection-csv case_selection/formal_60/cases.csv --actions evaluate
    python evaluator/accept_all_cases.py --out "reports/agent_upgrade/direct_workspace/heldout_15/$batch" --selection-csv case_selection/heldout_15/cases.csv --actions evaluate
    python evaluator/accept_all_cases.py --out "reports/agent_upgrade/direct_workspace/combined_profile/$batch" --selection-csv reports/agent_upgrade/direct_workspace/combined_profile_cases.csv --actions profile --profile-strategy combined
    python evaluator/pi_sandbox_smoke.py --mode basic --case-id deepseek_v4_flash_mtp.sample_pro --operation evaluate-profile --output-dir "reports/agent_upgrade/direct_workspace/pi_hardware/basic_formal/$batch"
    python evaluator/pi_sandbox_smoke.py --mode cannbot --case-id qwen3_14b.topk_select_k1 --operation evaluate-profile --output-dir "reports/agent_upgrade/direct_workspace/pi_hardware/cannbot_heldout/$batch"

预期：Formal 60 与 Held-out 15 的 75 个 Case 全部 correctness PASS；覆盖清单中的 Combined Profile 都产出三类有效诊断；两个真实 Pi Broker 测试中的 Evaluate/Combined Profile 都 PASS 且反馈只读；对 Case/环境/选择 CSV 的 before/after hash 一致。若环境清单没有跨 Host 表达，报告该项未覆盖，不将 8 卡冒充多机测试。

**完成标准及失败处理：** 任一 correctness 失败、Snapshot/integrity 失败、Combined 三类证据缺失、多卡/HCCL 执行失败或正式输入 hash 变化，均阻止投入使用。记录失败 Case 和原因；不重建 Baseline、不更新 CV、不将 `UNSUPPORTED`/`BLOCKED` 当作 PASS。确认停止与清理完毕后才运行下一项。

## 最终发布门槛

- Task 1 的实际启动上下文凭据检查、真实宿主 UID/GID 映射、空附加组、零 `CapEff` 及组 10/1000 DAC canary 测试通过；Task 2/3 的权限、Fork、崩溃恢复测试通过。
- 新建 Session 的所有 Worker 都是普通目录；外层 Agent 只有当前 Session 的 Workspace RW bind mount；Shell 子 Sandbox 继承同根或显式绑定同根，不存在候选池/公开副本/Workspace 链接和同步实现。
- Pi 私有文件、Native 工具、Shell/Python、Broker/Gateway/CANNBot Socket、网络和 CANNBot Extension 边界均通过确定性测试与 Basic/CANNBot 真实 Pi Smoke。
- 反馈只读且不作为正式记录；CLI、Action、Snapshot、Finalize 和 Candidate Worker 合同保持通过。
- Formal 60 与 Held-out 15 的 Evaluate 均通过；Combined Profile 覆盖矩阵和 Basic/CANNBot Pi Broker 的真实 Evaluate/Profile 路径检查均完成；Case、Fixture、Seed、Reference、Baseline、CV 未改变。
- 若环境与 Case 清单没有可执行的跨 Host 样本，报告多机覆盖缺口，不将 8 卡结果表述为多机验收。
- 任一门槛未通过时，不使用旧导出实现兜底，也不对外宣称已完成切换。
