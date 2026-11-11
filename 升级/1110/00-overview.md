# PTOBench Dedicated Agent User 修缮计划总览

**设计决定：** Pi Agent 使用独立 Linux 用户 `ptobench-agent`；Broker 与 Candidate Worker/NPU 执行继续使用启动 PTOBench 的原 UID。仅含 Broker 和 Agent 的 `ptobench-share` 组控制真实 Workspace 与必要 Unix Socket 的访问；Bubblewrap 继续限制 Agent 可见路径和网络。Broker 增加共享组成员资格，仅用于文件共享；原有硬件权限不变。Agent 不再使用自定义 subordinate UID/GID 映射。专用账号 + Bubblewrap + POSIX DAC 是唯一支持的 Agent 隔离路径。

**Workspace 解释：** 同一 Session 内 Pi 可以查看并修改每个 Worker 的 `candidate/`，也可以比较各 Worker 的只读 Context/Outputs。Fork 会把父 Candidate 复制成新的真实目录；Worker 之间互不自动同步。“共享 Candidate 池”表示同一 Session 的 Agent 可访问全部 Worker 的 Candidate，不表示它们共用同一份文件。

**计划范围：** 只处理 PTOBench 公共运行代码、测试、Workspace 和 Pi 沙箱边界。Basic 与 CANNBot 纳入。本计划定义可供 Pi Agent 方法共用的 Workspace/Agent 合同；未来 PTOmni Pi 应使用同一合同，但 PTOmni 本轮不实现、不验收，也不声称已验证兼容。只读使用 Formal/Held-out 选择清单是最终验收输入，不代表修改 Case；本计划不写入具体 Case ID，不改 Case、Fixture、Seed、Reference、Baseline、CV 或选择数据。Agent 的 Pi 原生文件工具与 Shell/Python 文件视图不暴露 Evaluator/Launcher/Broker；Candidate Worker 中运行的候选代码可能看到更多评测文件，此边界按此前决定暂不处理。

**旧方案处理：** `升级/1110/` 是唯一可执行计划。1109 的 subordinate UID/GID、`setpriv`、导出 Workspace 和旧 Session 恢复方案均被取代，不保留兼容或回退。旧 Session 只允许被识别为退役格式并在任何恢复/写入前明确拒绝；不加载、不迁移、不恢复、不继续执行，这只是拒绝旧格式，不是兼容层。1109 执行计划只加“已被 1110 取代、不得执行”的短标记，正文保留作历史记录；既有审计结果不清理。实现完成后删除 PTOBench 根目录 `ops/` 中的旧 dispatcher 与缓存，并将固定 dispatcher 放入现有 `evaluator/`；不改 CANNBot vendored `modes/cannbot/upstream/ops/`。

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

**用户准备门槛：** 本计划整理完成后，先由用户一次性完成下列宿主配置并回报检查结果，再开始 Plan 01 的代码修缮。实施开始后不再安排需要中断任务的管理员步骤；预检未通过时停在计划阶段，先补齐准备。

1. **实施前先冻结干净代码基线。** 记录 `git rev-parse HEAD`、完整 staged/unstaged/untracked 清单及差异；由用户在开始前明确哪些既有改动构成 PTOBench 基线，并将其单独提交或放入隔离工作树。当前工作树有混合暂存和大量无关数据，不能在此状态直接开始实施或创建所谓基线提交；不运行 `git add -A`、不覆盖/清空现有 index、不回退或 stash 用户改动。若无法建立干净且包含指定基线的实施工作树，停止在实施前，不让执行中的 Agent 自行整理这些改动。基线确认后，实施提交仅选择本计划列出的公共代码、测试、文档和七份 1109 退役标记；不暂存 Case/Fixture/Seed/Reference/Baseline/CV/`case_selection/`/测量结果。Plan 04 的 `legacy_ops_inventory.md` 是本机运行迁移回执，不纳入 Git；本计划文档本身保持跟踪。
2. **开始代码修缮前，用户一次性完成宿主机准备：** 管理员创建锁定且不可交互登录的 `ptobench-agent`，其主组设为 `ptobench-share`，并确保 NSS 不把它加入任何其他组；将 Broker 加入共享组但保留其硬件组。Agent 不需要可写的宿主 HOME；Pi 配置、可信工具与会话目录由 Broker 在每个新 Session 下创建。若同名主体已存在但不符合合同，停止核实，不接管未知主体。
3. 确认目标主机的已安装环境与当前有效 `environments/LOCK.json` 及 Case profiles 一致。当前锁文件包含主机绝对运行时路径；其他主机须先按现有环境部署流程准备兼容环境锁/profile 并通过锁校验，不能把当前主机路径写入启动代码、降低 hash 检查或在本计划中增加通用环境安装框架。若现有环境流程无法为目标主机提供有效锁，先单独解决环境部署，再开始本计划。
4. 安装系统 Bubblewrap，并确认 `setfacl`、`getfacl` 与非特权 user namespace 可用。Agent 启动只使用系统 PATH 上的 Bubblewrap，不依赖 Broker HOME 下的 npm/Codex 副本。Bubblewrap 缺失、Agent 身份无法创建 user namespace 或 ACL 工具缺失时，在代码修缮前先解决。
5. 按 Plan 01 安装精确 sudoers 规则。Broker 只能无密码地以 `ptobench-agent:ptobench-share` 执行有效环境锁指定的、解析后的 Python executable 加 `-I -S <checkout>/evaluator/agent_dispatch.py`；不得附加参数、运行其他命令、保留 `SETENV` 或给 Agent sudo 权限。Broker argv 和 sudoers 必须使用同一个规范化绝对路径；运行时不能被 PATH 或 Agent 环境变量覆盖。只保留 Request 路径变量，并验证 sudo 后 dispatcher 的实际环境。Dispatcher 文件创建后由 Broker canary 验证固定入口，不需要再次管理员介入。
6. **在用户配置主机 ACL 前冻结 Agent mount manifest。** 从 `environments/LOCK.json`、Basic/CANNBot Pi 命令、`View.system()`/`View.native_agent()`、Pi Shell wrapper 和 CLI transport shim 路径，整理宿主路径、符号链接最终目标、只读/可执行用途与 Sandbox 挂载点，保存到 `reports/agent_upgrade/dedicated_agent_user/host_preflight.md`，不提交到 Git。清单覆盖锁定 Python、实际挂载的 PyPTO runtime、Node/Pi、系统 Bubblewrap、CLI shim、CANNBot 冻结 Skills、Pi 配置/扩展与原生 Session 挂载、必要 Socket 及已批准 `runs_root`。不把 Candidate Worker 专用 `View.toolchain()` 路径加入 Agent ACL；CANN/PTOAS/编译器和 NPU Worker 动态库仍由 Broker/Worker 按原身份访问。用 `readlink -f` 解析每个宿主路径并逐级检查父目录及 ACL mask；记录每个路径 `getfacl -e` 的变更前/后快照，新增 `ptobench-share` 权限后，其他用户/组条目的有效权限不得扩大；若 mask 变化会扩大既有授权，停止并重新设计 ACL，不接受“总 ACL 仍看起来合理”作为通过依据。不递归 `chmod/chown`，不开放整个 HOME。Python/Node 加载目录可能需要读取/列举权限，这只授权宿主挂载访问，不扩大 Bubblewrap 显示范围。代码开始前的宿主探针只验证身份、namespace、Bubblewrap、DAC、存活 Agent 动态看到 Broker 发布的新 Worker，以及 NPU 设备拒绝访问；探针用临时目录和手工最小 mount，不创建 Pi Session、不运行 PTOBench production mount 或 CLI。精确生产 mount、锁定 Runtime 导入、transport CLI stub 与 dispatcher 检查由 Plan 01/03 在新代码具备后执行。所有预期路径和 ACL 必须在开工门槛一次准备完；实现发现 manifest 外路径时停止，不在运行中追加授权。
7. 从实际 Launcher、队列配置和启动命令盘点所有会使用的 `runs_root`。当前 `LAUNCHER.md` 默认是 PTOBench `runs/`，Session API 也接受自定义绝对路径；历史报告路径不自动视为活动 root。用户在开工前确认一份预批准 root 清单，并为每个 root 的完整父路径配置 Agent 最小遍历 ACL；运行时不得自动扩大这份清单或改写任意父目录 ACL。未预授权 root 在创建 Session 前明确失败并给出管理员预配置指引。以后新增自定义 root 时，先将该 root 纳入宿主 ACL 配置再运行；这不改变现有 CLI 合同，也不导致当前 Session 中途等待管理员。
8. Broker 加入共享组后，重新登录 SSH；在不打断运行任务的前提下等旧 tmux/队列退出，再重建 tmux/队列 server。确认新 Broker 进程获得共享组但保留硬件组；Agent 的有效组只有共享组，`/proc/self/status` 的 `Groups` 为空或只含共享组 GID、无 sudo/capability，能建立 Bubblewrap 所需的 user namespace，且无法打开 NPU 设备节点。未通过则先处理宿主准备，不走旧隔离路径。
9. 用户在开工前完成并回报上述准备检查，并确认实际启动工作流和活动 `runs_root`。用临时目录完成 Broker/Agent 身份不同、Candidate 可写、Context/Outputs 只读、Broker 可读普通 Agent 文件、存活 Agent 在单个真实 Workspace bind 上可见 Broker 新建 Worker，以及 Agent 无法打开 NPU 设备节点；这些仅为宿主能力探针，不得宣称生产 mount/dispatcher/CLI 已验证。不创建 Pi 会话、不加载模型、不运行 PTOBench Case/NPU。该门槛通过后才开始代码修缮；Plan 01/03 再验证生产启动链和真实 mount，不新增 ACL 或管理员配置。若实现暴露出未预批准的路径需求，停止整项实施，不临时放宽权限。
10. 从 `getent`、`id`、`readlink -f` 和 `getfacl` 动态读取用户、组、UID/GID、ACL 和运行时路径；源码不固化数字 UID/GID、subordinate range、硬件组名/编号或安装目录。若目标系统无法满足合同，停止并报告要求，不静默降低隔离。
11. 目标顶层代码布局仍限于 `sealed_assets/`、`modes/`、`launcher/`、`examples/`、`evaluator/`、`environments/`、`cases/`；本次不新增顶层运行模块，也不修改 `cases/` 或 `sealed_assets/` 内容。只清理 PTOBench 根目录 `ops/` 中已迁移且确认属于 PTOBench Agent 启动的文件；先清点其内容并保留必要审计记录，不递归删除未知或用户文件；不触碰 vendor、Skills 或 EAT 资料里的嵌套 `ops/`。
## 当前宿主准备状态

本轮只读复核：当前用户为 `1001/1001` 且只有当前主组；`ptobench-agent` 与 `ptobench-share` 不存在；系统未找到 `bwrap`；`setfacl`/`getfacl` 已安装；Broker 身份下 `unshare -Ur` 成功；`/data1/home/ziruiwang` 为 `0700`。这些是当前主机观测，不是实现常量；新建 Agent 后必须重新验证 user namespace 和 ACL。

因此，代码修缮开始前用户需要一次性完成并回报：确认目标主机有效环境锁与 Case profiles；冻结 Agent 目标 mount manifest 和活动 `runs_root` 清单；创建并核实专用账号/共享组；安装系统 Bubblewrap；安装精确 sudoers 规则；为固定 dispatcher 路径、锁定运行时、Pi/Node、CLI、冻结 Skills 读取路径和预批准 `runs_root` 设置最小 ACL；重新登录并重建 tmux/队列；完成临时 namespace/DAC/动态发布/NPU 拒绝探针。不得递归开放 Broker HOME 或整个仓库。固定 dispatcher 的真实执行 canary 在 Plan 01 建立后运行，不需要再次管理员安装。计划中 Python 测试命令均使用 `PTOBENCH_LOCKED_PYTHON`，由 `environment_lock.locked_python_executable()` 从目标主机有效 `environments/LOCK.json` 解析；不得依赖当前主机 `.venv-official` 路径。

## 全局不变量

- `ptobench-agent` 的有效组只有配置的共享组，附加组不得含其他 GID；不得拥有其他组权限、sudo、root 身份、capability、上游模型 API Key 或对 NPU 设备节点的直接访问权限。以实际 NPU 设备节点的拒绝访问测试验证，而不依赖特定硬件组名称。
- Broker 仅增加访问共享文件所需的共享组成员资格；其 UID 及硬件组不变。Candidate Worker 仍以 Broker UID 和原硬件权限运行，可能继承共享组，但不得因此获得新的 NPU/HCCL 权限。Evaluate/Profile、Snapshot 和 Gateway 均不切换为 Agent 用户。
- `workspaces/worker_NNN/` 下的 Candidate、Context、Outputs 都是真实目录。Workspace/Worker 根和 Context/Outputs 入口不授予 Agent 目录写权限；只有 `candidate/` 目录由共享组读写。这样 Agent 能管理 Candidate 内文件，但不能替换、删除或改名 Worker、Candidate、Context、Outputs 入口。
- Bubblewrap 每个 Agent 只暴露当前 Session 的 `workspaces/`，以单个 RW bind mount 显示；DAC 决定其中各目录可否写入。同一 Session 的所有 Worker `candidate/` 都可读写，Context/Outputs 都可读但只读。Fork 创建时复制父 Candidate，创建后各 Worker 的 Candidate 仍是独立真实目录，不做硬链接或自动双向同步。在线 Fork 由 Broker 原子创建，已运行 Agent 无需重启即可看到。
- Pi 读取自身所需的设置与会话文件；Pi 原生文件工具继续由 File Guard 限制，Shell/Python 子 Sandbox 继续隐藏 Pi 私有路径。必要 Broker、Gateway 和 CANNBot Socket 只向共享组开放。
- 保留 `ptobench evaluate/profile/fork` 合同、Combined 默认、显式 `split`、预算、Action/Event、Snapshot、Finalize、Correctness、正式 latency 与现有 NPU/HCCL 路径。
- Basic、CANNBot 各自使用自己的 `.env`，两者按当前要求配置 DeepSeek Flash 与 High thinking；Agent 工具不开放 Web Search/Web Fetch，公网搜索不能绕过网络隔离。
- CANNBot 保留原生冻结 Skills；通过受控 `skill` 工具读取，不给 Basic/CANNBot 注入 `raw_wiki`。
- Basic 和 CANNBot 的新 Session 需验证 Agent Workspace 与 README；`.benchmark_context/` 与 Agent 文件视图不包含/不暴露 `mode.json`、`profiling_modes.json`、`fixture_catalog.json`。若 Broker 内部仍需 Run 级 `mode.json` 等元数据，可留在可信目录，不挂载、复制或提供给 Agent；不要为隐藏 Agent 信息而删除仍被可信运行器使用的记录。保留当前 Case 的 tuning capabilities。Profile 和 Evaluate 的说明维持客观、中性。
- Gateway 的本地 `PYPTO_PROVIDER_API_KEY` 占位值不是认证凭据；当前访问控制由 Broker-owned Unix Socket 的 owner/group/mode 与 Bubblewrap 可见性提供。上游 Provider Key 不得进入 Pi Agent 的环境、启动 Request、Pi 配置、Session 配置或日志。该保证针对 Pi Agent 的文件与进程视图；Candidate Worker 与 Broker 同 UID，且其内部隔离已明确不在本轮范围，不对 Worker 进程状态作保证。
- 本轮支持 Pi Agent 后端。沙箱启动遇到非 Pi/不支持配置时明确失败；不得转交旧 `run_legacy` 或无隔离启动路径。
- 正式代码模块不新增 PTOBench 顶层运行模块，最终根目录没有 PTOBench 自有 `ops/`。`reports/`、`case_selection/`、`升级/` 与独立 `lightweight_e2e/` 是结果、选择或审计资料，不是运行模块。
- `lightweight_e2e` 不得成为 PTOBench 正式运行、导入、测试或部署依赖。Candidate Worker 内评测实现的可见性在本轮明确不处理。

## 最终验收条件

1. 正常 PTOBench 命令从普通 SSH 登录及队列/tmux 上下文运行时，Pi 进程实际宿主 UID 是 `ptobench-agent`；实际有效/附加组仅有共享组；`CapEff` 为零；Agent 用户自身不能通过 sudo 提权，且不能直接打开 NPU 设备节点。Broker 与可信 NPU Worker 仍使用原身份和权限。
2. Agent 可创建、编辑、原子替换和删除 `candidate/` 中的 Python/helper 文件及 `candidate/tuning.json`；Broker 可读取普通文件。Agent 创建的 Broker 不可读文件会在 Snapshot 前明确拒绝，不生成不完整 Snapshot。
3. 同 Session 所有 Worker 的 Candidate 对 Agent 可读写；Context/Outputs 可读但不可改；Worker/Workspace 根不可写，其他 Session 不可见。Broker 能更新真实 Workspace 内各 Worker 的 `.benchmark_outputs/`，Agent 能读而不能改；Fork 后新 Worker 在同一个 Pi/Shell 进程中立即可见，且各 Candidate 相互独立。
4. Basic 与 CANNBot 的 Pi 文件工具、Shell/Python、CLI、Gateway、CANNBot Socket、模型凭据和网络边界均验证通过；另经 Agent Bubblewrap/Pi Shell 内的真实 `ptobench` CLI 与 Broker socket 各完成一次 Evaluate 和 Combined Profile canary；Broker 与受信任 NPU Worker 保持既有身份和硬件能力。
5. 旧身份映射、旧 Agent runner、旧 Session 执行兼容、Candidate/public 导出/链接/同步运行路径已删除；`evaluator/` 内的专用账号 dispatcher 是唯一入口，PTOBench 顶层无 `ops/`。其余 Snapshot 稳定性、Session 完整性、权限检查和非 Workspace 归档行为仍通过。
6. Basic/CANNBot 真实 Smoke、Formal 60 与 Held-out 15 正确性 Evaluate、按已存在 `(granularity, device_count)` 类别选出的 8 个 Combined Profile 代表样本，以及 selection 内能够由有效运行配置证明的多卡和真实跨 host 多机样本通过。验收只读使用选择清单，不包含或修改具体 Case 实现内容。Profile 不是对全部 75 个 Case 逐一采集。多机需由实际运行配置中的 host topology 证明；`device_count` 或 8 卡不能单独证明多机。若 Formal/Held-out selection 没有可验证多机样本，则如实记录未覆盖，不能声称多机目标已通过；其余门槛通过时，只能签署“单机部署已验收”，不能签署“多机部署已验收”。如果实际部署目标包含多机，必须先提供可验证的 host topology 与可运行 Case，再完成该项验收。
7. Case/Fixture/Seed/Reference/Baseline/CV、`case_selection/` 和 `sealed_assets/` 内容未改写；验收开始与结束的内容哈希 manifest 一致。硬件最终运行只在所有不占 NPU 的软件、真实 Pi 和 DAC 门槛通过后开始。
