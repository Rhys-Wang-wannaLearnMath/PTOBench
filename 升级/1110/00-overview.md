# PTOBench Dedicated Agent User 修缮计划总览

**设计决定：** Pi Agent 使用独立 Linux 用户 `ptobench-agent`；Broker 与 Candidate Worker/NPU 执行继续使用启动 PTOBench 的原 UID。仅含 Broker 和 Agent 的 `ptobench-share` 组控制真实 Workspace 与必要 Unix Socket 的访问；Bubblewrap 继续限制 Agent 可见路径和网络。Broker 增加共享组成员资格，仅用于文件共享；原有硬件组与 NPU 权限不变。Agent 不再使用自定义 subordinate UID/GID 映射。专用账号 + Bubblewrap + POSIX DAC 是唯一支持的 Agent 隔离路径。

**计划范围：** 只处理 PTOBench 公共运行代码、测试、Workspace 和 Pi 沙箱边界。Basic 与 CANNBot 纳入；PTOmni 尚未建成，本轮不做真实运行验收。Candidate Worker 中评测器实现文件的可见范围仍暂不处理。

**旧方案处理：** `升级/1109/` 只作为历史设计记录，不是兼容实现或备用运行指令。新实现不读取、迁移或兼容旧身份映射配置、旧 Workspace 布局、旧 Agent runner 或旧 Session 的可执行状态；不保留适配器、双路径、feature flag 或自动回退。未迁移的旧 Session 文件可原样留作静态历史，但不能被恢复为可运行 Session。实现完成后，PTOBench 根目录不保留 `ops/`；仍需要的固定 dispatcher 放在现有 `evaluator/` 中。

## 子计划与顺序

| 顺序 | 子计划 | 目标 | 依赖 |
|---|---|---|---|
| 1 | [01-专用账号与启动链](01-专用账号与启动链.md) | 一次性创建 Agent 用户/共享组/精确 sudoers 规则；PTOBench 原命令自动以 Agent 身份启动 Bubblewrap | 无 |
| 2 | [02-真实 Workspace 与 DAC](02-真实Workspace与DAC.md) | 初始化真实 `candidate/`、只读 Context/Outputs，并让在线 Fork 直接发布到同一 Workspace 树 | 01 |
| 3 | [03-Pi 沙箱与运行通道](03-Pi沙箱与运行通道.md) | Agent 只挂当前 Session 的 `workspaces/`；接通 Pi 私有目录、Shell、Broker/Gateway/CANNBot Socket 和运行时依赖 | 01、02 |
| 4 | [04-旧路径清理与软件验收](04-旧路径清理与软件验收.md) | 删除映射/导出视图的死代码，完成不占 NPU 的正式软件回归 | 01–03 |
| 5 | [05-真实 Pi 与 Case 验收](05-真实Pi与Case验收.md) | Basic/CANNBot 真实 Pi Smoke、Formal/Held-out Evaluate 与 Combined Profile 验收 | 01–04 |

每份子计划只修改自己列出的代码边界。前置计划未完成或失败时停止，不跳到后续任务。发布后的 Agent 启动只有专用用户 + Bubblewrap + POSIX DAC 路径：配置缺失或不支持的后端均 fail closed。旧隔离方法没有运行兼容要求；不保留旧隔离实现、旧 Workspace 视图、旧配置解析、迁移逻辑或运行时回退。旧 Session 不自动迁移，也不提供继续运行的兼容入口；旧文件不批量删除，需要继续优化时新建 Session。

## 实施前置检查

1. 先记录现有工作树差异；不得回退、不全量暂存、不修改既有 Case、Fixture、Seed、Reference、Baseline、CV 或 `case_selection/` 数据。
2. 在临时目录完成 `sudo -u ptobench-agent` 启动、Bubblewrap 所需的最小用户命名空间设置、共享组 DAC 和必要 FD/标准流传递探针；不配置或依赖 subordinate UID/GID，也不调用 `newuidmap`/`newgidmap`。先确认固定无参数 sudoers 入口可启动动态 Bubblewrap argv；若要依靠通配 sudoers 参数或让可修改 PTOBench 代码以 root 身份运行，停止并修正启动接口。
3. 从 `getent`、`id` 和 `getfacl` 动态读取用户、组、UID/GID、ACL 和运行时实际路径；源码不写入当前主机的 UID/GID、subordinate range、用户名以外的组号、NPU 设备组或安装目录常量。`ptobench-agent` 与 `ptobench-share` 是建议的固定账户/组名；若同名主体已存在但不符合合同，失败退出并给出检查指引，不接管未知账户。
4. 当前 `/data1/home/ziruiwang` 是 `0700`；Agent 用户不能遍历当前模型用户的 HOME。仅对 Bubblewrap 所需的确切 Python、Node、PyPTO 与 PTOBench 客户端路径授予共享组所需的 ACL；不得开放整个 HOME 或挂载 Broker 源码。
5. 目标顶层代码布局仍限于 `sealed_assets/`、`modes/`、`launcher/`、`examples/`、`evaluator/`、`environments/`、`cases/`；本次不新增顶层运行模块。只清理 PTOBench 根目录 `ops/`，不触碰 vendor、Skills 或 EAT 资料里的嵌套 `ops/`。

## 全局不变量

- `ptobench-agent` 不能属于 `wheel`、`HwHiAiUser` 或其他非必要组；其可用组仅为共享组。Agent 中无 sudo 授权、root 身份、capability 或上游模型 API Key。
- Broker 仅增加访问共享文件所需的共享组成员资格；其 UID 及硬件组不变。Candidate Worker 仍以 Broker UID 和原硬件权限运行，可能继承共享组，但不得因此获得新的 NPU/HCCL 权限。Evaluate/Profile、Snapshot 和 Gateway 均不切换为 Agent 用户。
- `workspaces/worker_NNN/` 下的 Candidate、Context、Outputs 都是真实目录。Candidate 可由共享组读写；Context/Outputs 只读。目录权限阻止 Agent 替换、删除或改名 Worker、Context、Outputs 入口。
- Bubblewrap 每个 Agent 只暴露当前 Session 的 `workspaces/`，以单个 RW bind mount 显示；DAC 决定其中各目录可否写入。同一 Session 的所有 Worker Candidate 都属于 Agent 可读写的 Candidate 池；Context/Outputs 都可读但只读。Fork 创建时复制父 Candidate，创建后各 Worker 的 Candidate 仍是独立真实目录，不做硬链接或自动双向同步。在线 Fork 由 Broker 原子创建，已运行 Agent 无需重启即可看到。
- Pi 读取自身所需的设置与会话文件；Pi 原生文件工具继续由 File Guard 限制，Shell/Python 子 Sandbox 继续隐藏 Pi 私有路径。必要 Broker、Gateway 和 CANNBot Socket 只向共享组开放。
- 保留 `ptobench evaluate/profile/fork` 合同、Combined 默认、显式 `split`、预算、Action/Event、Snapshot、Finalize、Correctness、正式 latency 与现有 NPU/HCCL 路径。
- Basic、CANNBot 各自使用自己的 `.env`，两者按当前要求配置 DeepSeek Flash 与 High thinking；Agent 工具不开放 Web Search/Web Fetch，公网搜索不能绕过网络隔离。
- Basic 和 CANNBot 的新 Session 需验证 Workspace 与 README；不生成与优化任务无关的 `mode.json`、`profiling_modes.json`、`fixture_catalog.json`。保留当前 Case 的 tuning capabilities。Profile 和 Evaluate 的说明维持客观、中性。
- 本轮支持 Pi Agent 后端。沙箱启动遇到非 Pi/不支持配置时明确失败；不得转交旧 `run_legacy` 或无隔离启动路径。
- `lightweight_e2e` 不得成为 PTOBench 正式运行、导入、测试或部署依赖。Candidate Worker 内评测实现的可见性在本轮明确不处理。

## 最终验收条件

1. 正常 PTOBench 命令从普通登录及队列/tmux 上下文运行时，Pi 进程实际 UID 是 `ptobench-agent`；实际组权限只有共享组；`CapEff` 为零；Agent 用户自身不能通过 sudo 提权。
2. Agent 可创建、编辑、原子替换和删除 `candidate/` 中的 Python/helper 文件及 `candidate/tuning.json`；Broker 可读取普通文件。Agent 创建的 Broker 不可读文件会在 Snapshot 前明确拒绝，不生成不完整 Snapshot。
3. 同 Session 所有 Worker 的 Candidate 对 Agent 可读写；Context/Outputs 可读但不可改；Worker/Workspace 根不可写，其他 Session 不可见。Broker 能更新反馈，Agent 能读而不能改；Fork 在同一个 Pi 进程中立即可用，且 Fork 后各 Candidate 相互独立。
4. Basic 与 CANNBot 的 Pi 文件工具、Shell/Python、CLI、Gateway、CANNBot Socket、模型凭据和网络边界均验证通过；Broker 与受信任 NPU Worker 保持既有身份和硬件能力。
5. 旧身份映射、旧 Agent runner、旧 Session 执行兼容、Candidate/public 导出/链接/同步运行路径已删除；`evaluator/` 内的专用账号 dispatcher 是唯一入口，PTOBench 顶层无 `ops/`。其余 Snapshot 稳定性、Session 完整性、权限检查和非 Workspace 归档行为仍通过。
6. Basic/CANNBot 真实 Smoke、Formal 60 与 Held-out 15 正确性 Evaluate、Combined Profile 覆盖和指定多卡/多机样本通过。设备任务串行；缺失的多机样本如实记录为覆盖缺口，不能以多卡替代多机。
7. Case/Fixture/Seed/Reference/Baseline/CV 内容未改写。硬件最终运行只在所有不占 NPU 的软件、真实 Pi 和 DAC 门槛通过后开始。
