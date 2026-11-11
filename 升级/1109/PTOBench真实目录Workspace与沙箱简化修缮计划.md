# PTOBench 真实目录 Workspace 与沙箱简化修缮计划

**更新日期：2026-11-10**
**状态（2026-11-10）：文件系统探针和 UID/GID 映射已通过。此前附加组泄漏导致组 10 canary 可读；现在调用者已在清除附加组的 Shell 中重启 Codex，并实测 UID/GID 为 `1001/1001`、附加组为空、`CapEff=0`。这使现有直接 Bubblewrap 启动路径具备重新验证条件，但尚未证明实际 Agent 也继承了干净凭据；Task 1 仍须通过生产启动路径的 canary 验收，Task 2 尚未开始。正式 PTOBench 代码尚未切换，尚未启动 Pi、Case 或 NPU。**

## 目标

PTOBench 的 Pi Agent 只使用当前 Session 中的真实 Workspace。最终只保留一种 Workspace 实现：一个真实的 `workspaces/` 目录、一个可写 bind mount、Broker 与 Agent 的宿主 UID/GID 权限隔离。

```text
workspaces/
├── worker_000/
│   ├── candidate/           # Agent 可读写
│   ├── .benchmark_context/  # Agent 只读
│   └── .benchmark_outputs/  # Agent 只读，Broker 更新
└── worker_001/
    ├── candidate/
    ├── .benchmark_context/
    └── .benchmark_outputs/
```

不再把 Candidate、Context 或 Outputs 导出为链接、副本或同步视图，也不保留两套 Workspace View 并行运行。旧的导出和同步代码只作为迁移对象，完成切换后按调用关系删除；最终 Pi 路径只有上述真实目录实现。本计划当前只覆盖 Pi Agent 后端，PTOmni 尚未建成，不作为本轮运行验收对象。

权限合同如下：

| 内容 | Agent 权限 | Broker 权限 |
|---|---|---|
| 当前 Session 各 Worker 的 `candidate/` | 读取、修改、创建、删除 `.py` 和辅助文件；修改 `candidate/tuning.json` | 读取、复制、快照和执行前校验 |
| `.benchmark_context/` | 读取 | 创建、更新和保护 |
| `.benchmark_outputs/` | 读取；不能写、删、改名或 `chmod` | 发布和更新反馈 |
| Workspace 根及 Worker 入口 | 可列出和进入；不能写、删、改名或替换 | 管理 fork 和发布状态 |
| 系统依赖、Python/Node/PyPTO | 按需执行和读取；不能改写 | 由受信任运行路径管理 |
| `/tmp` | 临时读写 | 不纳入 Candidate 或 Snapshot |

Agent 可以创建新的 Python helper；Broker 通过稳定扫描和 Snapshot 收录它们。Agent 只应修改 `candidate/` 和 `candidate/tuning.json`。结果目录是反馈副本，正式记录仍由 Broker 的 Session、Action 和 Event 记录维护。

Workspace 中不生成与当前任务无关的 `mode.json`、`profiling_modes.json` 或 `fixture_catalog.json`。当前 Case 的 `tuning_capabilities.json` 可以作为公开任务资料；全局调优策略和状态驻留清单不提供给 Agent。`state_residency_inventory` 相关策略不属于本实现目标。`lightweight_e2e` 不成为正式 PTOBench 的依赖。

生成的 README、任务说明和 Profile 说明只描述真实的 `candidate/`、只读 Context/Outputs、`ptobench evaluate/profile/fork` 用法和当前 Case 合同；不再保留会把链接目标、导出副本、同步过程或隐藏评测实现写给 Agent 的措辞。Profile 缺省仍为 Combined，显式 `split` 继续由 CLI 支持。

## 当前状态与探针结论

当前正式代码仍有旧的导出 Candidate 池和链接式 Agent 视图；这只是待替换的现状，不是目标设计。正式切换前不得用旧 Session 证明新布局已经生效，验收必须使用新建 Session。

本轮在 `/tmp/ptobench_probe/` 完成了最小 Sandbox + 真实 Workspace 探针。探针不启动 Pi、不加载模型、不调用 PTOBench Case、不访问 NPU；bwrap 路径由生产代码 `evaluator/isolation.py::executable()` 解析得到。探针中的文件均为普通目录和普通文件，没有 Symbolic Link。

探针结果：

| 检查 | 结果 |
|---|---|
| Agent 与 Broker 身份 | 部分通过，但安全门槛失败。Broker 为宿主 UID/GID `1001/1001`；Agent namespace UID `0` 映射到宿主 UID `165536`，GID `0` 映射到共享 GID `1001`，且无有效 capability。Agent 仍继承宿主附加组 `10` 和 `1000`；真实 DAC 测试中，它成功读取了 Candidate 挂载内属于宿主组 `10`、权限为 `0640` 的文件。 |
| Candidate 创建、替换、删除 | 通过。Agent 可创建 helper、原子替换入口文件并删除文件。 |
| Context/Outputs 保护 | 通过。写入、删除、文件或目录 `chmod`、目录改名、目录替换和 Workspace 根写入均被拒绝。 |
| Broker 读取 Candidate | 通过。Broker 能读取 Agent 新建的普通 `0664` 文件和原子替换后的入口文件。Agent 以 `0600` 新建的文件对 Broker 不可读，探针明确得到拒绝；正式 Snapshot 必须据此报错，不得生成半份快照。 |
| 动态 fork | 通过。Agent Shell 保持运行且没有增加 mount 时，Broker 在挂载外的 staging 目录完整构造并原子 rename 出 `worker_001/` 后，Worker 立即可见；Agent 修改其 Candidate 后，Broker 能读取，`worker_000` 不受影响；staging 不可见且发布后为空。 |
| Broker 更新反馈 | 通过。Broker 原子更新 `worker_001/.benchmark_outputs/result.json` 后，同一个 Agent 进程能读取；Agent 不能改写该反馈。 |
| 单次 Workspace mount | 通过。Agent mount 视图中只有一个针对临时 `workspaces/` 的可写 bind mount，动态 fork 不需要重启或追加 mount。 |

这证明了真实目录与单次可写 bind mount 的文件系统机制可行，也验证了 UID/GID 映射数值；但当时运行探针的 Shell 带有附加组，DAC 安全合同未通过。调用者随后在清除附加组的 Shell 中重启 Codex，实测宿主 Shell 为 UID/GID `1001/1001`、`/proc/self/status` 的 `Groups` 为空、`CapEff` 全零。该新证据只说明当前 Shell 凭据干净，尚未验证实际 Agent 的 Bubblewrap 子进程、队列或 tmux 启动上下文。因此下一步是在现有生产 `agent_command`/`View.argv` 路径上重跑身份和 DAC canary 探针；不需要另建 systemd 服务或 privileged launcher。通过前仍不能切换 Workspace 权限。Pi 的私有设置、会话、工具、Gateway、CLI socket 和 Candidate Worker 仍需单独验收。

## 唯一的正式实现方案

### 1. Workspace 创建与权限

Broker 在当前 Session 的 `workspaces/` 下创建真实的 `worker_NNN/`，再创建三个真实子目录。发布前在 Broker-only staging 目录完成 Candidate、Context、Outputs、权限和保护清单；校验完成后原子 rename 到 Worker 路径并登记 `pending`，最后设置 Agent 可见权限并登记 `ready`。CLI 只接受 `ready` Worker。

目标权限为：Workspace/Worker/Context/Outputs 目录由 Broker 所有、Agent 组可读可进入但无写权限；`candidate/` 为 setgid 共享目录，Broker 和 Agent 组可读写；普通 Candidate 文件至少让 Broker 可读。不能只依靠 `umask`，因为现有 `file_tree.atomic_bytes()` 默认创建 `0600` 文件；初始化、fork 和 Context/Outputs 发布都要显式设置权限。

Agent 新建的 `0600` 文件必须在 Snapshot 前被明确识别为 Broker 不可读并拒绝该 Action，给出可修复的错误。不能静默跳过该文件，也不能生成不完整 Snapshot。

### 2. Sandbox 与身份

Pi Agent 的外层 sandbox 只挂当前 Session 的 `workspaces/`，以可写 bind mount 呈现；不再挂 Candidate 池、副本目录或链接目标。复用当前正式启动链：Basic/CANNBot 调用 `evaluator/isolation.py::agent_command`，由 `View.argv` 通过 Bubblewrap `--unshare-user` 建立现有 user namespace。Agent 使用独立的宿主映射 UID，并只通过 Broker 主 GID 访问共享文件；启动 `agent_command` 的宿主进程必须具有与当前已实测 Shell 相同的干净凭据：UID/GID 为 `1001/1001`、`os.getgroups() == []`、`CapEff` 为零。入口在启动前检查这些条件，不满足就拒绝启动；不尝试调用 `setgroups([])`，也不在 GID 映射后修改组列表。无需安装 systemd unit、常驻服务、setuid/capability helper 或新增 Broker-to-launcher socket。队列或 tmux 里的实际运行也必须继承并通过同一检查；由旧 tmux server 保留的脏凭据不能因客户端 Shell 已清理而视为通过。实际 Agent 还要验证宿主 UID/GID 映射、`/proc/<pid>/status` 的 `Groups`、`CapEff`，并用组 10/1000 的真实 DAC canary 确认它们不可读、共享 GID `1001` 仍按合同可访问。若此生产路径仍携带未授权组，Task 1 停止在这里继续定位启动链；不得通过放宽文件权限、改挂载范围或跳过 canary 进入 Task 2。Agent 没有 Broker UID、root 能力或绕过 DAC 的 capability。Broker、PTOBench CLI、Candidate Worker 和 NPU/HCCL 仍运行在既有受信任宿主路径，不套用 Agent 的 user namespace。

动态 fork 由 Broker 完成：复制父 Worker 的 Candidate、生成新的只读 Context/Outputs、规范化权限、校验后原子发布。由于整个 `workspaces/` 已经挂载，存活的 Agent 自动看到新 Worker，不需要重启 Pi 或增加 mount。Worker Candidate 是独立副本，不是多个 Worker 指向同一份 Candidate 内容。

### 3. Pi 私有文件与网络

Pi 的设置、会话、扩展和配置锁文件仍由 Pi 读取和写入；它们不属于 Candidate。Pi 原生文件工具使用受信任访问拦截，Shell/Python 使用受信任的子 sandbox 隐藏 Pi 私有路径和对应的 `/proc` 视图。Basic 和 CANNBot 各自使用自己的 `.env`，内容一致；Agent Shell 不读取这些文件，也不接触上游模型密钥。模型请求经固定 Gateway，PTOBench CLI 经受限 Unix socket。

Pi 运行配置不开放 Web Search/Web Fetch；Shell 的公网 HTTP、`curl`、`wget`、`git clone/fetch` 和搜索用途的 SSH 连接继续被现有网络隔离拒绝。该限制只作用于交互式 Agent 工具，不改变受信任的 NPU Runtime、HCCL、管理员 SSH 或模型 Gateway 通道。

Candidate Worker 内部可见哪些评测实现文件是另一个边界问题，按当前决定暂不处理；本计划不把它写成已解决条件。

### 4. 旧实现清理

完成 Pi 新 Workspace 实现并通过验收后，逐项删除仍无调用者的 Candidate 导出池、public 副本、链接创建、同步逻辑、旧 mount 参数和只验证旧布局的测试。不得仅按函数名删除：先检查 Launcher、Pi Basic、Pi CANNBot、Session/Action、Snapshot、CLI、脚本和文档的调用关系。保留必要的 Candidate 特殊文件拒绝、路径校验、Snapshot 稳定性检查、Session 完整性检查和显式 split Profile。

## 实施顺序

1. **建立干净基线。** 当前工作树混有大量公共代码、Case、Fixture、结果和报告改动；不得使用 `git add -A`。先按明确的 PTOBench 公共代码清单形成可审阅基线，再改 Workspace 路径。
2. **实现单一 Pi Workspace 路径。** 集中修改 Workspace 创建、fork 发布、Agent sandbox mount、权限初始化和 Snapshot 入口；不保留另一种 View 作为运行时后备。
3. **补齐 Pi 边界。** 验证私有配置、会话锁文件、Pi 文件工具、Shell/Python、Gateway、CLI socket 和网络限制；不把 Candidate Worker 内部评测源码纳入本轮。
4. **迁移并清理旧路径。** 用调用关系确认旧导出/同步机制无调用后删除；更新 Basic/CANNBot README、任务 handoff、CLI 帮助和测试。PTOmni 只保留未来接入点，不做真实运行。
5. **完成分层验收。** 先做不占 NPU 的软件和真实 Pi smoke test，再做 Formal/Held-out 的正确性与 Combined Profile 抽查；不把临时探针结果冒充正式 Case 验收。

## 验收计划

### A. 不占 NPU 的软件回归

- Workspace 初始化、fork、三个真实目录、Candidate helper、`candidate/tuning.json`、权限继承、Snapshot、Action、反馈发布、Session 恢复和 finalize。
- 对每个新 Worker 使用 `lstat` 确认三个入口都是普通目录；Agent 可以读写 Candidate，不能写删改名 Workspace 根、Context 或 Outputs；Broker 可以读取 Candidate 并更新 Outputs。
- Agent 存活期间连续创建至少两个 fork；无需重启或追加 mount 即可看到；父子 Candidate 后续修改相互独立。
- 注入 fork 发布各持久化边界的中断，确认恢复不会暴露未注册 Worker、不会重复 Worker ID、不会留下未保护 Context。
- 同时修改 Candidate 和启动 CLI Action，确认得到一致 Snapshot 或明确失败；覆盖原子替换、嵌套 helper、删除和权限异常文件。
- 只挂当前 Session；确认其他 Session/Case、Evaluator/Launcher/Broker、其他 mode、全局策略、Pi 私有设置和上游密钥对 Agent 工具不可见。Candidate Worker 内部文件不在本项结论内。
- Gateway 请求、CLI socket、系统依赖、`/tmp` 和 Combined 默认/显式 split Profile 回归通过；网络限制不影响受信任执行路径。
- 运行相关 `evaluator/tests`、Basic/CANNBot 隔离测试和 75 个 Formal/Held-out 的新 Workspace 初始化扫描；不复制或记录 Case 源码。

### B. Basic 与 CANNBot 真实 Pi smoke test

两种 mode 分开验收，各至少覆盖一个 Formal 和一个 Held-out；PTOmni 暂不测。检查：

- 真实 Pi 加载各自 `.env` 中的 DeepSeek Flash 和 High thinking，并能正常保存会话和配置锁文件。
- Pi 原生文件工具不能读写私有设置/会话；Shell、Python、路径遍历、符号链接、硬链接别名和 `/proc` 也不能绕过隔离。
- Agent 能创建 `.py` helper、修改 `candidate/tuning.json`、执行 `ptobench evaluate .`、`ptobench profile .` 和 `ptobench fork .`；反馈可读但不可修改。
- 新 fork 在同一 Pi 进程中出现，父子 Candidate 独立；Broker 更新反馈后 Agent 能看到。
- 工具清单没有 Web Search/Web Fetch；公网访问失败；固定模型 Gateway、PTOBench CLI 和 CANNBot 所需扩展/socket 正常。

### C. 正式 Case 验收

- Formal 60 与 Held-out 15 先做正确性 Evaluate；结果只作为本轮验收记录，不写回 Seed、Ref、baseline 或 CV。
- Combined Profile 覆盖两层 Granularity、单卡、多卡、8 卡和实际跨 host 多机；分别覆盖 Basic 与 CANNBot。没有多机样本时报告缺口，不用 8 卡代替多机。
- 硬件任务串行运行；按温度、功耗和任务协调判断设备可用，不用 AICore 利用率作为阻挡条件。
- 如果 Workspace 改造触及 Profile 采集、状态复位、Runtime 编译/执行或测量代码，取消“无需重测”的假设，另行扩大正确性和性能复验范围。

## 发布门槛

只有以下条件全部满足，才把新 Workspace 用于 Basic/CANNBot 的 Formal 和 Held-out 正式优化：

1. UID/GID、capability、文件权限和动态 fork 探针通过；探针结果已记录，本机临时目录已清理。
2. 新建真实 Pi Session 中，所有 Worker 的三个入口都是普通目录；Candidate、Context、Outputs 权限和 Snapshot 行为符合合同。
3. Basic/CANNBot 的 Pi 私有文件、Shell/Python、Gateway、CLI socket、网络限制和 `.env` 加载均通过真实 smoke test。
4. Candidate 修改、新建 helper、fork、Evaluate、Combined Profile、只读反馈和 Session/finalize 记录全链路通过。
5. Formal 60、Held-out 15 以及规定的多设备/多机覆盖完成；未改写 Seed、Ref、baseline 或 CV；PTOmni 不作为本轮发布条件。

任一权限映射、fork 恢复、Pi 私有文件、CLI、Gateway、旧路径清理或受信任硬件执行检查失败，都停止发布，不用保留旧链接或另一套 View 来掩盖失败。
