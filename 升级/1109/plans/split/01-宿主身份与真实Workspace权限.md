已被 1110 Dedicated Agent User 计划取代，仅供历史参考、不得执行。

# 宿主身份映射与真实 Workspace 权限 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task.

**Goal:** 建立生产 Sandbox 可验证的宿主 UID/GID 隔离，并让新 Worker 使用权限正确的真实 Workspace。

**Architecture:** 沿用现有 Broker、Session 与 Workspace 管理路径；先在运行时检测宿主身份和 Bubblewrap 映射，优先使用满足权限合同的无特权启动路径，必要时才经一次性安装的最小权限启动器清理 Pi Agent 的附加组，再初始化真实目录及其 Candidate、Context、Outputs 权限。本计划只执行原计划 Task 0–2。

**Tech Stack:** Python 3、Bubblewrap、Linux user namespace、`setpriv`、必要时使用的精确 sudoers 规则、POSIX UID/GID 与目录权限、现有 unittest 测试。

**Spec:** [`升级/1109/PTOBench真实目录Workspace与沙箱简化修缮计划.md`](../../PTOBench真实目录Workspace与沙箱简化修缮计划.md)

**Source plan:** [`升级/1109/plans/PTOBench真实目录Workspace与沙箱简化Implementation Plan.md`](../PTOBench真实目录Workspace与沙箱简化Implementation Plan.md)

**Task 1 authority:** 宿主身份启动要求以本 Plan 的 Task 1 为准；它替代 Spec 和完整主计划中与身份清理、固定宿主编号及启动方式冲突的旧表述。其他 Workspace、Sandbox、Broker、Snapshot 和验收约束继续适用。

**Execution order:** This is Plan 01 of 05. 无；这是整个实施序列的起点。

**Plan boundary:** Execute only original Task 0–2; do not implement tasks assigned to another split plan. Task 0–2 的完成标准全部满足才结束本计划。当前服务器已验证 Broker 的宿主身份、普通登录附加组、`setpriv` 清组路径及 Bubblewrap/DAC canary；这些数值和组成员仅是该服务器的观测记录，不是代码或测试常量。Task 1 必须自动检测每台主机的调用用户、有效 UID/GID、附加组、capability 和 user-namespace 映射；优先使用可验证的无特权路径，只有无法安全清组时才使用一次性配置的固定启动器。本计划执行者不自行安装特权组件或修改系统 sudoers。通用跨发行版安装器、自动修改账户或 `/etc/subuid`/`/etc/subgid` 的部署工具属于后续部署任务，不纳入本计划。身份映射和 DAC canary 通过前，不得开始 Task 2 或 Fork 状态迁移。

## Global Constraints

- 只保留真实目录 Workspace：workspaces/worker_NNN/{candidate,.benchmark_context,.benchmark_outputs}，不以链接、副本或同步目录呈现给 Agent。
- Agent 和 Broker 必须是不同的宿主 UID；继续使用现有 `evaluator/isolation.py::agent_command` / `View.argv` Bubblewrap `--unshare-user` 路径。运行时从当前账户与宿主实际配置检测 UID/GID、`/etc/subuid` 和 `/etc/subgid` 中可用的 subordinate ranges、user-namespace 支持及实际 `uid_map`/`gid_map`；不得假定某个范围起点或数量。先尝试直接使用已满足合同的凭据，或以普通用户权限验证 `setpriv --clear-groups` 是否可用；只有无特权清组不可用时才要求最小权限的一次性启动器。启动器按 sudo 提供的调用者 UID/GID 动态降权并清空附加组，不在代码中固定账号或数字。Agent 只保留 Workspace 合同需要的共享主组映射；Broker 和受信任 NPU Worker 保持原有 UID/GID/groups 与硬件权限。Pi 每次启动都检查实际身份、组列表、capability、映射和 DAC canary；任一项不符、必要宿主配置缺失或辅助程序不可用时 fail closed 并给出明确检查结果及配置指引。不得让 Agent 获得 sudo/root，不得以 root 执行用户可修改代码；若使用 sudoers，只允许特定安装入口无参数调用，不允许通配参数或通用 root Shell。
- 每个 Agent Sandbox 只对当前 Session 的 workspaces/ 做一个可写 bind mount；Candidate、Context、Outputs 的权限由宿主文件系统 DAC 强制执行。
- 保留 ptobench evaluate/profile/fork 的语法、Combined 默认、显式 split、预算计数、事件记录、正确性检查、正式延迟、Snapshot、Finalize 和 Candidate Worker 执行路径。
- Basic 与 CANNBot 继续使用各自的 `.env`，配置为 DeepSeek Flash 和 High thinking；PTOmni 尚未作为本轮 Agent 后端验收对象。
- 保留 Pi 私有文件隔离、Gateway、Broker Unix Socket、模型 API 通道、Agent 网络隔离及 NPU/HCCL 受信任执行路径。
- 不修改 Case、Fixture、Seed、Reference、Baseline、CV 或 case_selection 数据；验收结果另存到新的 reports 目录。
- lightweight_e2e 不得成为正式 PTOBench 运行、导入、测试或部署依赖。
- Candidate Worker 内部评测文件的可见范围仍是独立问题，本轮不扩展处理范围，也不声称已解决。

## Review Focus

1. 普通登录可能带任意 supplementary groups 时，清组是否只作用于 Agent 子进程；Broker/NPU Worker 凭据是否保留，Agent 实际 user-namespace 映射是否来自当前宿主可用配置，以及当前登录附加组对应的 DAC canary 是否拒绝读取。由 Task 1 在生产 Bubblewrap 和实际 SSH 启动上下文中验证。
2. Agent 新建 0600 文件或不可遍历目录后，Broker 是否明确拒绝 Candidate 快照而不是生成残缺 Snapshot；由 Task 2 和 Task 7 验证。
3. Fork 在 staging、pending、发布和恢复之间中断时，是否出现可被 CLI 使用的半成品或重复 Worker ID；由 Task 3 的故障注入测试验证。
4. Pi Shell 子 Sandbox 是否仍能写 Candidate、不能改 Context/Outputs；必要 Unix Socket 是否允许映射 UID 连接而不开放目录枚举；由 Task 4 的生产 Bubblewrap 集成测试验证。
5. 诊断文件是否仍可读、只读，且路径指向 Agent 可见的 Workspace；由 Task 5 的发布测试和 Combined Profile 抽查验证。

## 原计划任务（正文原样迁移）

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

---

### Task 1: 为 Pi Agent 建立固定的一键受限启动链

**目标及实现方式：** 正常 PTOBench 命令保持不变；Basic 和 CANNBot 继续走现有 `evaluator/isolation.py::agent_command` 与 Bubblewrap `View.argv`。启动时自动读取当前调用者的宿主有效 UID/GID、主组与附加组、capability、user-namespace 配置及 Bubblewrap 实际映射。若当前凭据已满足“无未授权附加组、无有效 capability”的合同，直接使用现有无特权路径；否则先验证普通用户 `setpriv --clear-groups` 是否可用。只有无特权路径不可用时，才经一次性配置的固定最小权限启动器为 Pi Agent 清组并降权。启动器从 sudo 提供且经校验的调用者身份动态取得目标 UID/GID，不固定账号名或数字；降权后才读取一次性启动请求并执行已有 Bubblewrap argv。Broker 和 NPU Worker 不清组、不改身份，保留现有硬件权限。Agent 不获得 sudo/root；任何特权阶段都不导入或执行 PTOBench、Session、用户 Python 或其他可修改代码。宿主缺少 user namespace、有效 subordinate 映射、清组方式或必要工具时，报告检测到的配置和修复指引并 fail closed。

**文件：**

- Create `evaluator/agent_launch.py::prepare(command: list[str], cwd: Path, request_dir: Path, *, path: str) -> tuple[list[str], dict[str, str]]`：生成 Session 私有的一次性请求文件，并按运行时检查结果选择安全启动前缀：干净凭据走现有 Bubblewrap；无特权 `setpriv --clear-groups` 验证成功则用它启动；否则使用已安装的最小权限启动器。请求目录为 `session_root/agent/launcher/`，不得挂载到 Agent Sandbox；目录权限 `0700`、文件权限 `0600`，每次启动使用独立文件名。
- Modify `evaluator/isolation.py::agent_command`：保留现有 `View.argv` 生成的完整 Bubblewrap 命令；在启动前通过 `agent_launch.py` 检测当前有效身份、附加组、capability、user-namespace 支持及 subordinate UID/GID 配置。使用系统实际检测到的范围，不写入固定 subordinate 起点或 UID/GID；实际 Agent 映射必须验证宿主 Agent UID 与 Broker UID 不同、共享主组是 Broker 当前主 GID。检测失败或启动方式未配置时给出明确错误，不得回退为未清组启动。
- Modify `modes/basic/runner.py::run`、`modes/cannbot/runtime.py::run_dispatch`：接收 `agent_command` 的新返回值并按返回的最小环境启动；不改 CLI 命令、Broker、Gateway、Pi 工具配置或 NPU Worker 启动方式。
- Create `ops/ptobench-agent-launch`：若运行时检测确认无特权清组不可用，才使用安装后 root-owned、不可被普通用户修改的固定入口；只接受零参数。入口仅从 sudo 的调用上下文取得原调用者 UID/GID，校验其为有效账户且与受限 Agent 请求所属 PTOBench 用户一致，然后以 `env -i` 传递必要请求信息，调用安装时发现并验证的绝对 `setpriv` 路径，动态指定调用者 UID/GID、`--clear-groups`、清除 capability 和 `--no-new-privs`，再运行固定 dispatcher。代码中不得固定 UID/GID、用户名、subordinate 范围或工具安装目录；禁止在特权阶段解析请求、启动 Python/PTOBench 或执行动态命令。
- Create `ops/ptobench-agent-dispatch.py::check_identity(expected_uid: int, expected_gid: int) -> None` and `read_request(path: str) -> tuple[list[str], dict[str, str], str]`：只在 setpriv 已降权后运行。验证实际 UID/GID 与本次调用者一致、`os.getgroups() == []`、所有 capability 为零和 `NoNewPrivs=1`；用 `O_NOFOLLOW` 打开请求，校验 owner、mode `0600`、普通文件、单链接和大小上限；读取后先 unlink，再校验 schema 并以普通 UID/GID `execve` 请求中的 Bubblewrap argv。
- Create `ops/ptobench-agent-launch.sudoers` 模板：只有无特权清组无法满足合同的部署才安装。模板由一次性主机配置步骤按当前 PTOBench 运行账户生成，限制为无参数调用固定启动器，并只保留必需的 `PTOBENCH_AGENT_LAUNCH_REQUEST` 与 sudo 自动提供的调用者身份信息；不写死账号名或 UID/GID，不允许通配参数、任意 sudo 命令或通用 root Shell。每个需要特权清组的运行账户分别获得精确授权；其他账户不能借用该规则。通用跨发行版安装器和自动配置账户/subuid/subgid 不属于本 Plan，留待后续部署任务。
- Create `evaluator/tests/test_agent_launch.py`；Modify `evaluator/tests/test_isolation.py`：覆盖请求文件安全校验、身份失败、无参数启动、sudo 失败时 fail closed，以及真实 Bubblewrap UID/GID、groups、capability 和 DAC canary。新增 `test_agent_launcher_fails_closed_when_unavailable` 与需管理员安装后执行的 `test_pi_launcher_clears_groups_and_preserves_candidate_access`。既有 `agent_command` 集成测试在普通 CI 中用 test-only mock 替代 privileged launcher、仍真实运行 Bubblewrap；mock 不是生产 fallback。更新 Basic/CANNBot 启动调用点测试。仅当 `sandbox_launcher_environment` 不再有调用且回归测试通过后，才可删除这个旧 helper。

**Interfaces:** `agent_command(...) -> tuple[list[str], dict[str, str]]`。返回命令由启动时检测结果决定：现有 Bubblewrap argv、已验证的无特权 `setpriv --clear-groups` 前缀，或已安装的固定 sudo 启动器。运行请求仅含 `PATH` 和 `PTOBENCH_AGENT_LAUNCH_REQUEST` 等必要字段，不携带模型密钥；目标 UID/GID 从当前 PTOBench 调用者动态取得并校验。调用者 Broker 的 `os.getgroups()` 不作修改。Bubblewrap 的实际 subordinate UID/GID 范围从当前宿主配置和实际 `uid_map`/`gid_map` 验证，不得假定固定起点；共享 GID 始终是 Broker 当前主 GID（`os.getgid()`）。`--cap-drop ALL`、挂载、Gateway、Broker Socket 和网络隔离合同保持不变。`common.await_action_start` / `session_actions._run_process` 的现有 FD gate 不改；验收时确认 Agent 启动不依赖会被 sudo 关闭的额外继承 FD，Gateway 与 Broker Socket 仍按当前路径连通。

- [ ] **记录本机探测结果而不固化编号：** 从普通 SSH Shell 和 PTOBench 实际服务/队列上下文分别记录 `id -u`、`id -g`、`id -G`、`/proc/self/status` 的 `Groups`/`CapEff`、`/proc/self/uid_map`/`gid_map`、`/etc/subuid`/`/etc/subgid` 是否有当前账户对应的范围、`bwrap`/`setpriv` 的实际路径及版本。记录哪些值是当前主机事实；测试断言不得要求特定用户名、UID/GID、附加组号、subordinate 起点或二进制安装路径。确认启动 Agent 前后 Broker 组列表未变，NPU Worker 相关启动代码和身份未改。
- [ ] **先写无特权单元测试：** 测试每次生成唯一请求文件，目录/文件权限为 `0700/0600`；拒绝 symlink、hardlink、错误 owner/mode、超限或损坏请求；启动选择逻辑对干净组、普通 `setpriv --clear-groups` 成功、失败且无管理员启动器、失败但启动器可用分别验证直接路径、无特权路径、fail closed 与专用启动器路径。dispatcher 对当前 UID/GID 不符、清组失败、附加组非空、capability 非零或 `NoNewPrivs` 未置位均拒绝；shell 入口拒绝任何参数。测试不能要求 root、不能安装 sudoers。
- [ ] **验证动态宿主映射：** 用 mock 提供不同用户/组 ID、不同 subordinate range 起点及不同 supplementary groups，证明启动器和映射检查均使用检测值而不是本机常量；再在可用 Linux 环境检查实际 `uid_map`/`gid_map`。若 user namespace 或可用 subordinate 范围缺失，预期得到包含当前检测结果和宿主配置指引的明确错误，不得自动创建范围或降低隔离。
- [ ] **实现固定的特权边界（仅在无特权方式不可用时）：** 证明 root-owned 入口只调用已检测并验证的绝对系统二进制和固定参数；目标 UID/GID 来自并校验 sudo 的调用者身份。setpriv 完成组清除、永久降权、清除 capability 和设置 `no_new_privs` 后，才运行 dispatcher。检查 sudoers 对当前部署账户精确限定路径和无参数匹配，并且安装文件及父目录均 root-owned、普通用户不可写。dispatcher 在降权后验证动态身份，读完即删除请求，再运行原有 Bubblewrap 命令。任何特权阶段不执行 PTOBench 或用户可修改代码。
- [ ] **接入现有 Pi 路径：** Basic、CANNBot 均继续调用同一个 `agent_command`，按已验证启动方式自动启动；用户继续使用原 PTOBench 命令，不需每次手动输入 sudo/setpriv。清组、user namespace、映射或身份检查失败时必须报错并停止；不得直接以带附加组的身份启动。PTOBench 的用户侧命令、Broker 和 NPU 环境保持原样。
- [ ] **一次性配置（只针对当前部署，不做通用安装器）：** 若无特权路径不可用，准备并验证当前宿主可执行的专用启动器安装步骤及精确 sudoers 项；用户名、调用者 UID/GID 和系统工具路径均在配置时从宿主检测并验证，不作为源码常量。每个需要该能力的运行账户单独授权。安装命令不自动修改用户组或 `/etc/subuid`、`/etc/subgid`。通用跨发行版安装命令、自动检测/配置器留待后续部署任务。本计划执行者不得运行安装命令、修改系统 sudoers 或安装特权组件；需要管理员操作时提供命令后停止等待调用者。
- [ ] **启动验收：** 从普通登录环境及实际队列/tmux 上下文启动生产 Pi 探针，不运行 NPU。检查 Agent 的 `/proc/self/uid_map`、`gid_map`、`Groups`、`CapEff`、`NoNewPrivs` 及宿主 PID 身份：宿主 Agent UID 必须与当前 Broker UID 不同；共享 GID 必须匹配当前 Broker 主 GID；不得存在任何未授权附加组；capability 为零。将映射与该主机实际 subordinate 配置对照，不能比较固定范围起点。确认 Broker 组列表未变，NPU Worker 权限未改。
- [ ] **Agent 内部的提权拒绝检查：** 从真实 Pi Sandbox 内尝试 `sudo -n -l` 及执行固定 launcher；Agent 必须不能以 Broker/Root 身份启动命令，宿主侧确认它仍是映射给当前账户 subordinate range 的非 Broker UID、无 capability。若 `sudo` 或 launcher 在挂载视图中不可见，也记录这一结果；安全验收的重点是宿主身份不变且无法调用 root 规则。
- [ ] **真实 DAC 与 CLI smoke：** 为当前登录环境中每个 supplementary GID 建立 `0640` 拒读 canary（若无附加组，则用当前宿主可安全创建的未授权组 canary），并创建共享 GID `os.getgid()` 的控制文件；Agent 不得读取任何未授权组 canary，必须可按权限合同读写 Candidate，Broker 必须可读取 Agent 新建文件。保留 `0600` Candidate 文件导致 Snapshot 明确拒绝的检查。经正常 PTOBench 路径分别抽查 Basic 和 CANNBot 的 Pi 启动，不调用 NPU；检查 CLI、Gateway、Broker Socket 和现有 FD gate 仍工作。
- [ ] **解锁后续任务：** 只有所选启动路径的前置配置已完成，且普通登录启动、附加组清空、动态 UID/GID 映射、零 capability、DAC canary、Candidate 读写、Basic/CANNBot smoke 全部通过，才允许开始 Task 2。无特权路径通过时无需管理员安装；需要固定启动器时必须先完成受限的一次性配置。若任一项失败，停止修复；不得放宽 sudoers、允许 Agent sudo/root、以 root 执行用户代码、跳过 canary 或回退为带未授权组的 Bubblewrap 启动。

**验证命令与预期：**

    (cd evaluator/tests && PYTHONPATH=.. python -m unittest test_agent_launch -v)
    (cd evaluator/tests && PYTHONPATH=.. python -m unittest test_isolation.FilesystemBoundaryTests.test_agent_launcher_fails_closed_when_unavailable -v)
    visudo -cf /tmp/ptobench-agent-launch.sudoers

预期：单元测试不使用特权并覆盖启动选择、请求安全、动态身份和 fail-closed。`/tmp/ptobench-agent-launch.sudoers` 是仅在无特权清组路径不可用时，由一次性主机配置步骤生成的精确 sudoers 文件；执行 `visudo` 前检查其调用用户、启动器绝对路径和无参数限制与本机配置一致。若主机可无特权清组，则无需生成或验证 sudoers。以上只验证本地代码和模板，不能作为系统已安装或生产 Agent 已通过的证据。必要的一次性管理员配置后，另运行 `sudo -n -l`、`visudo -c`，并执行 `(cd evaluator/tests && PYTHONPATH=.. python -m unittest test_isolation.FilesystemBoundaryTests.test_pi_launcher_clears_groups_and_preserves_candidate_access -v)`、普通登录下的真实 Pi/Bubblewrap 探针及 DAC canary。

**完成标准及失败处理：** 先在当前宿主通过自动探测、无特权优先的启动选择、实际 UID/GID 映射、未授权组 canary、Candidate 访问和 Basic/CANNBot smoke。若本机必须使用启动器，还需完成一次性管理员配置并通过生产验收；之后同一 PTOBench 命令自动选择安全路径。其他 Linux/NPU 服务器只需适配各自用户 namespace/subuid/subgid 与工具路径，不得修改源码常量。任何宿主配置缺失或验收失败时，输出具体检测项和配置指引并 fail closed；Task 2 不得开始。不得改宿主账号组、放宽权限、动 Case/Fixture/Seed/Baseline/CV 或运行 NPU。

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
