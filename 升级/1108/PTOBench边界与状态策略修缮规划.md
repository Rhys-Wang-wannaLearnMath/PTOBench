# PTOBench 边界与状态策略修缮规划

**更新日期：2026-11-09**

## 目标与本轮范围

完成本规划的发布门槛后，PTOBench 可供已建成的 **Basic 和 CANNBot** 直接用于 Formal 与 Held-out 优化任务。验收包括软件回归、全部 Case 的 workspace 创建检查、Basic/CANNBot 的真实 sandbox 抽查、311 个官方 Candidate Seed 的 correctness Evaluate、两次真实 Pi Candidate-edit Evaluate，以及代表性 Combined Profile。

**本轮只验收 Basic 和 CANNBot。PTOmni 当前尚未达到正式 Agent mode 就绪状态，不做真实 sandbox 抽查，也不作为 Basic/CANNBot 投入使用的前置条件。** PTOmni 可用后另做接入验收。Candidate Worker 执行 Candidate 代码时可见 PTOBench 运行代码及 `evaluate.py` 的问题，按既定决定暂不处理，不列入本轮实现或验收范围。Basic/CANNBot 当前使用的 Pi 路径是本轮网络、文件视图与凭据验收对象；不据此宣称其他 Agent backend 也通过相同隔离验收。

Agent workspace 的 Candidate 工作目录统一为小写 `candidate/`；workspace 根目录仍是 `ptobench evaluate .` 和 `ptobench profile .` 的工作入口。

## 当前进展

### 已完成并有记录的部分

- 正式 PTOBench 已去除 `state_residency_inventory_20261028.csv` 驱动的共享状态驻留策略，也已清除正式代码、测试和文档对 `lightweight_e2e` 的依赖。Case-local resident ABI、Bench Launch 状态恢复、Runtime L3 两 pass 间恢复和执行契约继续保留。
- Basic/CANNBot 的运行准备不再校验、复制或注入 `raw_wiki`；CANNBot 原生 Skill 不改造。PTOmni 达到正式 mode 就绪状态后继续保留完整三部分知识库，不受 Basic/CANNBot 调整影响。新 workspace 不生成 `mode.json`、`fixture_catalog.json`、`profiling_modes.json`；Run/Session 的方法归档继续保留。Combined 是 CLI 与启动器的共同默认 Profile；需要分体模式时仍可显式选择 `split`。
- Basic 与 CANNBot 分别持有自己的 `.env`，当前两份文件的非密钥配置为模型 `deepseek-flash`、思考程度 High；真实 Pi 是否分别加载了对应 mode 的文件，仍须在 sandbox 抽查中确认。
- 受限本地 Model API Gateway、Agent 网络隔离和 PTOBench Unix socket CLI 通道已有自动化覆盖，包括模型请求转发及普通公网连接被拒绝。真实 Pi 工具清单、联网拒绝和 Basic/CANNBot 端到端流程仍须实测。
- 基线和状态修缮结果已完成：48 个受影响 Case 重认证通过；`all_case_boundary_repair_20261108` 的 311 个 Case 为 311/311 PASS、0 失败；代表性 Combined Profile 为 7/7 PASS。它们证明此前状态修缮和基线队列已完成，不替代本轮 Agent workspace 边界验收。
- 最新工作树已加入小写 `candidate/` workspace 布局、Run-local Candidate 池、Broker 固定链接导出、sandbox 单独挂载 Candidate 池，以及提交前 Candidate 同步。首次 Basic/CANNBot 真实 Pi sandbox 启动在 CLI 执行前失败：Broker 把自己创建的受保护链接当成普通 Candidate 内容处理；`sync_source()` 后续已改为先校验这些链接，再只同步 Run 授权的 Candidate 树。修复后的定向测试通过，但 Basic/CANNBot 的真实 Pi 流程尚未重跑，不能把修复视为真实验收通过。
- 对当前 322 份 Case 定义做了 Candidate Seed 文件清点，未发现非 Python 源文件；唯一额外文件是一个 `.gitkeep`，复制逻辑将其排除。Candidate 文件合同限定为 Python 文件与 `candidate/tuning.json`。Agent 可以在 `candidate/` 中新建 `.py` 文件作为 helper；递归扫描、Snapshot 和入口模块搜索路径均支持这类文件。真实 Pi 创建 helper 后再 Evaluate 的端到端链路尚未验证。
- `no_valid_snapshot` 的 Candidate 挂载问题已修复：Agent workspace exports 只读挂载，当前 Run 的 Candidate 池单独挂入 `/ptobench/candidates` 并允许写入。定向测试覆盖 Candidate 可写、workspace 根只读、受保护链接校验和动态 fork 反馈刷新。
- `modes/` 不再作为整棵目录参与 Session 的完整性校验；运行依赖明确限于 `modes/__init__.py`、`basic/`、`_shared/` 和 `cannbot/`。运行时复制也只复制这些显式声明的 mode 子目录，不会因新增 PTOmni、`profile_tool` 或其他 mode 而自动纳入。定向测试已覆盖不相关 mode 目录变更不进入运行中 Session 的检查，以及运行时复制排除未选 mode；完整软件回归已通过。
- 最新完整软件回归已通过：`PYTHONPATH=evaluator ../../.venv-official/bin/python -m pytest -q evaluator/tests` **340 passed、4364 subtests passed**（1311.23 秒）。该次重跑覆盖 Pi 私有配置目录可写调整，包含 Workspace/Snapshot/Session/finalize、Pi 文件工具拦截、嵌套 Shell sandbox、Broker/Gateway、网络限制、CANNBot 和 CLI 测试；不会调用真实 NPU。
- 完整回归起初有一个失败，原因是网络隔离测试把虚拟环境外部 Python 安装路径传给 bubblewrap，而测试视图只挂载 `.venv-official`。该测试现在使用 `sys.prefix/bin/python` 下的已挂载解释器；单项复测 **1 passed**，随后完整回归通过。此次只调整测试解释器路径，没有改变 Agent/runtime 行为。
- 已完成的 sandbox 自动化用例覆盖：Agent Bash/Python 看不到 Pi 私有设置和会话目录，但能写 `candidate/` helper 与 `candidate/tuning.json`，能读只读任务上下文，通过 Unix socket 使用 CLI；Pi 原生文件工具的受信任扩展会拦截私有路径访问；模型 Gateway 能转发到测试 Provider，公网连接失败。它们是实际 bubblewrap 测试和受控 Provider 测试，不等同于真实 Pi 进程使用真实模型的验收。
- `candidate/` 新增 `.py` 的契约已由源码与自动化测试覆盖：真实 bubblewrap Shell/Python 测试可新建 helper，入口模块的本地导入范围包含 `candidate/`，Snapshot 递归收录 `.py` 文件及 `candidate/tuning.json`。真实 Pi 亲自创建 helper 后调用 Evaluate 并进入 Snapshot 的端到端抽查仍待完成。
- Basic 与 CANNBot 的无硬件 workspace 检查均已完成：311 个 Case × 2 个 mode，共 **622/622** 个新 workspace，无遗漏或失败。2026-11-09 又在临时目录中独立复核了 622 份新 workspace：正式 Case 唯一性、Formal 60/Held-out 15 清单、Candidate Seed Python 源文件哈希、入口文件、`candidate/tuning.json`、不生成 workspace `mode.json`/`fixture_catalog.json`/`profiling_modes.json`，以及 README 的 Candidate 路径、Combined Profile 命令和中性说明均通过。临时 workspace 已自动清理；这是 workspace 初始化检查，不替代真实 Pi sandbox、Broker/fork 权限或硬件 Evaluate/Profile 验收。Granularity、设备数与 host 分布已汇总。
- 当前尚未启动本轮硬件队列。旧的 `combined_profile_queue_20261106` tmux 会话存在但不在运行。用户指定以温度和功耗判断设备可用，不以 AICore 利用率作阻挡；最近检查为约 34–36°C、83–88 W，NPU 进程表为空，因此按该判据可开始串行硬件验收。此前 311/311 PASS 与 7/7 Profile 是已有基线及状态修缮记录，不等同于本轮新队列结果。本次完整回归没有调用 NPU。

### 尚未通过发布验收的部分

- Basic 与 CANNBot 的真实 Pi sandbox 中，连续创建至少两个 fork 的生产流程仍待抽查：确认新 workspace 的 `candidate/` 指向当前 Run 的 Candidate、Candidate 和 `candidate/tuning.json` 可写、workspace 根及 `.benchmark_context/` / `.benchmark_outputs/` 不可替换、Broker 更新的反馈可见、其他 Run 的 Candidate 不可访问。
- Candidate 池挂载源已收口：`agent_command()` 从 Broker 导出的 workspace 根目录推导同一 Run 的 Candidate 池，并拒绝缺失或符号链接目录；不再接受 `PTOBENCH_AGENT_CANDIDATES_ROOT` 覆盖。伪造环境变量的定向 bubblewrap 回归已通过。仍须在真实 Pi Session/fork 流程中确认 Broker 导出的路径、worker 身份和 Candidate Snapshot/finalize 全链路一致；完成最后代码改动后重跑相关回归。
- Pi 首次启动会在其配置目录创建 `settings.json.lock`，因此该目录不能只读挂载。当前 `agent_command()` 将 Pi 私有目录可写地挂到 `/ptobench/pi-private`，并显式加载受信任 `tool_call` 文件访问拦截扩展；Pi Shell 使用 `pi_shell_wrapper.py` 在嵌套 bubblewrap 中隐藏 Pi 私有目录、Agent 会话目录及对应 `/proc` 进程视图。Pi 配置锁文件、只读反馈和私有路径拦截的三项定向测试通过；完整 evaluator 回归也通过。真实 Basic Pi 已经通过受限 Gateway 完成无工具模型请求；CANNBot 真实 Pi、真实模型请求及扩展/CLI 联合流程仍须验证。
- Basic 与 CANNBot 当前 `.env` 的非密钥值均已读到 `deepseek-flash`、`high`。这确认配置文件内容，不证明生产 Pi Session 实际加载了对应配置，也不证明真实模型请求成功。
- Basic 与 CANNBot 真实 Pi 会话仍须核对工具清单未开放 Web Search/Web Fetch；从 Shell/Python 访问公网、`curl`、`wget`、`git clone/fetch` 和搜索用途的 SSH 连接均失败；固定 Model Gateway 与 PTOBench CLI RPC 可用。Pi 实际加载各自 `.env` 的模型与 reasoning 也须在会话中确认。
- Basic 与 CANNBot 各自至少一个 Formal 和一个 Held-out 的真实 Pi sandbox 抽查尚未完成。PTOmni 尚未建成，本轮不测 PTOmni，也不把它作为 Basic/CANNBot 发布条件。
- 本轮 311 个官方 Seed Evaluate、两次 Candidate-edit Evaluate 和至少 11 个 Combined Profile 硬件队列尚未启动；须在真实 Pi sandbox 预检及动态 fork 检查通过后串行执行并保留完整日志。若 11 个基础 Profile Case 中没有实际跨 host 的多机 Case，则增加 1 个多机 Profile，Profile 总数为 12。队列不得更新 Ref baseline、Candidate Seed、认证数据或 CV 统计。
- `sandbox_supervisor.run_legacy()` 仍为旧 backend 合并运行时代码和 Agent 文件视图；本轮验收范围是 Basic/CANNBot 实际使用的 Pi 路径，其他 backend 不据此宣称已通过同等边界验收。

## 已确定的设计

### Workspace 内容与权限

| 路径或内容 | Agent 可见性 | Agent 可执行的操作 |
| --- | --- | --- |
| 当前 Case 任务说明与公开任务上下文 | 可见，只读 | 阅读当前任务与接口约定 |
| `candidate/` 中 Candidate 入口及 Python helper | 可见，可写 | 编辑、创建和删除当前 Candidate 的 Python 文件及目录；新建的 `.py` helper 可随 Candidate 一起评测 |
| `candidate/tuning.json` | 可见，可写 | 调整 Case 声明允许的 tuning 参数 |
| `ptobench` CLI 客户端与本地 Unix socket | 客户端只读可见，socket 可调用 | 调用 `evaluate`、`profile`、`fork`；请求由工作区外的 Broker 处理 |
| 系统、Python、Node 与已安装 PyPTO 运行依赖 | 只读可见，可执行 | 启动 Pi、运行必要的本地工具；不能改写已安装环境 |
| CANNBot 自身的原生 Skill/extension | 仅 CANNBot 可见，只读 | 支持 CANNBot 工作流；Basic 不注入，CANNBot 不做替换或改造 |
| `.benchmark_context/` | 可见，只读 | 阅读当前 Case 的公开上下文；不提供 `mode.json` |
| `.benchmark_outputs/` | 可见，只读反馈副本 | 阅读、复制和分析 Evaluate/Profile 结果；不能改写或替换路径 |
| workspace 根目录及其他 PTOBench 元数据 | 可见性限于 CLI 工作所需，不可写 | 以 workspace 根调用 CLI；不能借根目录更换受保护入口 |
| `fixture_catalog.json`、`profiling_modes.json` | 不生成到 Agent workspace | 无需提供输入摘要清单或 Profile 模式枚举 |
| `evaluator/`、`launcher/`、其他 mode 实现、全局调优策略、其他 Run 的 Candidate | Basic/CANNBot Pi Agent Shell 不可见 | 不属于当前 Case 的优化资料 |
| Basic/CANNBot mode `.env` 与上游模型密钥 | 目标是 Agent Shell 不可见；模型密钥由宿主 Gateway 持有，Agent 只得到本地占位 key | 通过固定 Gateway 推理，不直接读取 mode `.env` 或 Provider 密钥；真实 Pi 仍须验收 |
| Pi 设置、模型配置、会话与扩展私有文件 | 目标是 Pi 可使用、Agent 工具不可读写 | Pi 文件工具拦截与 Shell 嵌套文件视图已实现，自动化测试通过；真实 Pi 启动、推理、会话读写和 CANNBot 扩展兼容性仍待验收 |
| 模型 Gateway | Agent 可用固定模型服务，并能看到模型名、Gateway 地址及本地占位 key；上游 key 不可见 | 通过固定 Gateway 推理，不直接访问 Provider 凭据 |
| Pi Web Search/Web Fetch | 不开放 | Agent 使用给定的资料和固定模型 API |

`.benchmark_outputs/` 是 CLI 结果的反馈副本；Session/受保护归档仍是正式 Action/Event 记录。README 只客观说明这些路径、权限和命令用途，不包含 “Do not inspect hidden test/reference assets” 或相近提醒。说明 `profile` 提供诊断信息，不检查输出正确性，也不提供 clean latency；正确性与 clean latency 通过 `ptobench evaluate` 检查。

### Candidate 与动态 fork 的文件布局

1. Candidate Seed 可继续保留旧的根级平铺格式；创建 workspace 时迁入小写 `candidate/`。新 workspace 和 fork 的正式 Agent 视图只把 `candidate/` 作为 Candidate 工作目录。Agent 可以在其中新建、修改、删除 `.py` 文件和目录，包括拆分实现所需的本地 helper。
2. Candidate Snapshot 递归收录 `candidate/` 下的 Python 文件及 `candidate/tuning.json`；新建的 `.py` helper 会随 Candidate 一起参加 Evaluate/Profile 与 Snapshot。入口模块通过 `candidate/` 的本地模块搜索路径按 Python 导入规则使用 helper。workspace 根、`.benchmark_context/`、`.benchmark_outputs/` 和其他元数据不隐式进入 Snapshot。非 Python 资源文件不属于当前 Candidate 文件合同，不复制、不接受为提交内容。
3. workspace 集合、每个 workspace 根、`.benchmark_context/` 和 `.benchmark_outputs/` 在 Agent 视图中只读。Broker 可以从受信任通道更新反馈副本，且更新后 Agent 在现存 sandbox 内立即读到新结果。
4. 动态 fork 采用 **Run-local Candidate 池**：Agent 的 workspace 视图只读挂载；每个 workspace 的 `candidate/` 链接指向当前 Run 单独导出的 Candidate 树；同一 Session 中 Broker 可在 Agent 运行期间加入新 Candidate；Agent 仅能编辑当前 Run 授权的 Candidate，不能访问其他 Run。Broker 只接受目标与当前 Run/worker 精确匹配的受保护 Candidate 链接，并从该授权树构建 Snapshot。不得把 PTOBench 全局 Candidate 目录挂成 Agent 可写目录。
5. 上述 Candidate 池是已选定并已有真实 bubblewrap 定向测试支持的方案，但尚未完成端到端发布验收。实现包括 Broker 创建并校验固定目标链接，以及 Agent sandbox 显式挂载当前 Run 的 Candidate 池；动态 fork 用例已验证新 workspace 在存活 sandbox 中出现、候选可写和受保护入口不可替换。完整 Session/fork/Snapshot/finalize、跨 Run 隔离和正式 Pi 会话中的行为仍须验收。Candidate 中除 Broker 创建的固定目标链接外，其他 symlink 和特殊节点仍拒绝。
6. Profile/Evaluate 结果使用稳定的只读 public 反馈挂载点；Broker 在挂载点内更新内容，不替换 Agent 已挂载的根目录。Agent 的只读限制不应阻止 Broker 写正式 Session 记录或刷新反馈。
7. 新建 Session 的 Agent workspace 不提供 `mode.json`、`fixture_catalog.json` 或 `profiling_modes.json`。已有运行目录不批量清理；本轮验收以新建 Session 为准。旧 backend 的历史运行目录行为不作为 Basic/CANNBot Pi 发布行为的依据。

### Mode 依赖范围与运行中修改

- Session 不校验或恢复整棵 `modes/` 目录，只把 `modes/__init__.py`、`modes/basic/`、`modes/_shared/`、`modes/cannbot/` 作为当前 Basic/CANNBot 运行路径的 mode 源依赖。对这些目录之外的 mode（包括尚未建成的 PTOmni）新增、修改或删除文件，不应阻塞已运行 Session，也不应进入本次运行时复制。
- 运行时复制依赖采用显式目录清单，不扫描并复制 `modes/` 下所有 Python 目录。对被排除 mode 的变更，验收同时检查 Session 完整性结果和复制产物；对已声明的 Basic、CANNBot 与共享代码，继续按现有 Session 记录和参与者来源规则处理。
- Candidate Worker 可见其运行时所需的评测实现文件这一边界仍按范围声明暂不处理；mode 目录依赖收窄不等于已隔离 Candidate Worker 的评测器文件视图。

### 网络、模型与运行环境

- Basic 和 CANNBot 各自在自己的 mode 目录持有 `.env`，不是共用或链接到另一 mode 的文件；公共配置为模型 `deepseek-flash`、思考程度 High。验收只检查非密钥字段，不输出或复制密钥。
- Provider 密钥留在宿主 Supervisor/Gateway；Agent 只获得本地 Gateway 地址与不可用于上游认证的占位 key。Agent 的 Bash/Python/Web 工具不能任意访问公网、搜索站点或在线依赖源；Pi Web Search/Web Fetch 未开放。
- 网络隔离的验收对象是交互式 Agent 进程。Candidate Worker 运行 Agent 提交的 Candidate 代码，当前仍使用 PTOBench 的既有网络执行环境；本规划不声称 Candidate 代码本身无法发起网络请求。由于该执行路径还承载 NPU/HCCL，本轮不把 Agent 网络命名空间直接套用到 Candidate Worker。若目标扩展为阻止 Candidate 代码访问外网，需另行设计并验证不破坏 HCCL 的限制方式。
- 固定模型 Gateway 的必要请求和 PTOBench Unix socket CLI RPC 保持可用。管理员对主机的 SSH 管理连接发生在 Agent sandbox 外，不由 Agent 出口策略改写。NPU Runtime/HCCL 在受信任执行路径中运行，不能因 Agent 网络隔离而受影响。
- 本 Goal 不新增 per-Run 模型 API 用量限额；这与禁止外网搜索分开处理。Agent 对现有 Gateway 的模型请求沿用 Provider 计费。
- Pi 私有设置/会话不能仅靠搬到 workspace 外、改名或隐藏 README 路径来隔离。扩展文件工具拦截与 Shell 嵌套 sandbox 已实现并通过自动化测试；Basic/CANNBot 的真实 Pi Session 仍须证明 Pi 能读取所需配置和会话，而 Pi 文件工具及 Shell/Python 无法经绝对路径、目录扫描、环境、`/proc` 或链接读取或修改这些文件。

## 实施顺序

### 1. 完成 Candidate 生产路径与动态 fork 验收

Run-local Candidate 池、Broker 链接导出和 sandbox 挂载已有工作树实现。Codex `no_valid_snapshot` 的缺失挂载原因已修复。一个 bubblewrap 定向用例已覆盖新 workspace 在存活 sandbox 中出现、`candidate/` 可写和受保护入口不可替换；伪造 Candidate 根环境变量的定向测试也已通过。首次真实 Pi sandbox 流程在 CLI 前因 Broker 链接同步检查失败而中止，Broker 链接校验修复后的真实 Pi 流程仍待复验。本步骤先完成不启动 Pi 的 Candidate/Broker/bubblewrap 检查；**真实 Pi 验收必须等第 2 步的 Pi 私有文件边界通过后**，再按第 4 步执行。定向用例不算真实 Pi 验收：

- 每个新 fork 都以小写 `candidate/` 暴露；其中已有和新建的 `.py` Candidate/helper 文件、目录及 `candidate/tuning.json` 可读写。新建 helper 应能由 Candidate 入口正常导入，并出现在 Snapshot manifest 中；Snapshot 只收录 Python 文件及明确允许的 `candidate/tuning.json`，其他非 Python 文件不得混入。当前尚缺真实 sandbox 下新建 helper 后 Evaluate 与 Snapshot 的端到端用例。
- Agent 不能创建、删除、重命名或替换 workspace 根入口、`.benchmark_context/`、`.benchmark_outputs/` 或其根挂载点；Broker 创建的受保护链接目标被篡改时同步拒绝。
- Broker 发布 Evaluate/Profile 反馈后，运行中的 Agent 立即读到新内容，旧结果不会残留；Agent 不能改写结果副本或伪造结果。
- 同一 Run 的 fork 行为符合 Session 语义；独立 Run 的 Candidate 不可见、不可写；授权边界以内的 Candidate 修改才能进入 Snapshot/finalize。
- 正常受保护链接可同步；Candidate 中任意非授权 symlink 和特殊节点被拒绝。

当前契约为：Broker 持有稳定的 `exports`、`candidates`、`public` 目录；Agent 启动时一次性挂载只读 workspace exports、当前 Run 的 Candidate 池和只读 public 结果。sandbox 启动后，Broker 通过这些已挂载父目录发布 fork，因此无需动态追加单 workspace mount；Agent 在存活 sandbox 中应立即看到新 workspace。Workspace 根上的 `candidate/` 链接由 Broker 创建，目标固定到当前 Run/worker；`sync_source()` 校验受保护链接后从授权 Candidate 树读取，不跟随任意 workspace symlink。Candidate 挂载源现在从 Broker workspace export 路径推导，不接受 `PTOBENCH_AGENT_CANDIDATES_ROOT` 覆盖，伪造路径定向测试已通过。池内只包含当前 Run 对 Agent 开放的 Candidate，不能暴露其他 Run 或未授权 worker。Session fork、Candidate Snapshot/finalize、连续多 fork、跨 Run 隔离、修复后的真实 Pi 流程仍须验收。

### 2. 验收 Pi 私有文件与真实 Agent 文件视图

实现已加入 PTOBench：Pi 使用受信任配置读取 sandbox 私有路径中的设置、模型配置、扩展与会话；该 Pi 私有目录在 Agent 主 sandbox 内可写，以支持 Pi 创建配置锁文件，但由受信任 `tool_call` 扩展拦截 Pi 原生文件工具对私有路径的访问，并检查路径规范化、`..`、符号链接和硬链接别名。workspace 自发现扩展关闭，CANNBot 扩展显式加载。Pi 的 `shellPath` 指向受信任 wrapper，由嵌套 bubblewrap 启动 Bash，在 Shell/Python 视图中隐藏 Pi 私有路径和 Agent 会话目录，并隔离对应 `/proc` 进程视图。嵌套视图保留当前 Run 的 `candidate/`、正常 `/tmp` 用途、必要系统依赖、模型 Gateway 和 PTOBench CLI Unix socket。自动化测试覆盖 Pi 文件工具拦截、Shell/Python 隐藏私有文件、Pi 配置锁文件写入、Candidate 写入和 CLI RPC；新增的三项定向隔离测试通过，随后完整 `evaluator/tests` **340 passed、4364 subtests passed**。Basic 已完成一次真实 Pi/Gateway 无工具模型请求；以下仍是生产验收要求，而不是已通过结论：Basic 与 CANNBot 各用真实 Pi、真实模型请求验证 Pi 能启动并保存自己的会话；Pi 文件工具和 Shell/Python 都无法访问私有设置/会话；CANNBot Task/Skill 扩展、CLI、Gateway 和 Candidate 编辑保持可用。Pi 工具清单须确认没有 Web Search/Web Fetch；真实 Bash/Python 对公网 HTTP、`curl`、`wget`、`git clone/fetch` 和搜索用途的 SSH 连接应失败。不要将 Candidate Worker 内部文件视图纳入本轮结论。

### 3. 完成软件回归与 622 个 workspace 检查

按下列顺序完成；任何边界或数据损坏问题先修复，再继续：

| 顺序 | 范围 | 覆盖内容 |
| --- | --- | --- |
| A | Workspace、Snapshot、Session、finalize | 旧平铺 Seed 迁入 `candidate/`、入口/helper 导入、tuning 编辑、fork 复制、Snapshot 只收录 Candidate 内容 |
| B | Broker、Profile、Evaluate、Session Action/Event | 结果状态与路径、预算、历史记录、只读反馈刷新、Profile/Evaluate 的错误状态保持 |
| C | bubblewrap 隔离与 CANNBot/Agent CLI | 新建和动态 fork 的权限、链接校验、结果入口不能替换、候选可写、Broker 固定 Candidate 挂载源、伪造 Candidate 根路径被拒绝、CLI 转发和中断清理；不相关 mode 变更不阻塞 Session，运行时复制排除未选 mode |
| D | Pi 私有目录、网络与 `.env` | Pi 原生文件工具与 Shell/Python 的双路径访问控制；绝对路径、路径遍历、符号链接、环境和 `/proc`；无 Web 搜索；模型/API 路径允许项可用、其他公网拒绝；两份 mode-local `.env` 非密钥设置一致且各自被加载 |
| E | 311 × 2 workspace 初始化 | Basic 与 CANNBot 各生成全部 311 Case 的新 workspace，共 622 个；不启动模型、不占 NPU |

622 项检查逐个验证 Case 无遗漏/重复、正式 Seed 迁入、README 与权限一致、无 workspace `mode.json`/`fixture_catalog.json`/`profiling_modes.json`、Combined 默认且显式 `split` 可用、fork 遵循同一 Candidate 布局。另汇总 Formal 60、Held-out 15 的覆盖，及 Granularity、设备数和实际 host 数；8 卡不等于多机。若正式 Case 清单在执行前变更，以当时的正式清单为准并记录差异。

### 4. 对 Basic 与 CANNBot 做真实 sandbox 抽查

Basic 和 CANNBot 两种当前可用于真实优化的 mode **分别**完成真实 bubblewrap + Pi 抽查；不是用一种 mode 的结果代表另一种。PTOmni 当前尚未建成，本轮不抽查，也不阻挡本轮发布验收。

每个 mode 至少覆盖一个 Formal 和一个 Held-out workspace；用该 mode 自己的 `.env` 启动，核对模型 `deepseek-flash`、High，并完成一次真实模型请求。真实模型与边界预检阶段不调用 NPU；通过后，硬件队列先执行代表性 Pi Session，再继续全量队列。每个 mode 的抽查还要覆盖：

本节先完成无硬件的 Pi、文件边界、网络和动态 fork 预检。列出的 Seed Evaluate/Profile 与 Candidate-edit Evaluate 属于硬件验证，由第 5 节的串行队列先执行；它们不是开启队列前的预检条件。

- Pi 的实际工具列表没有 Web Search/Web Fetch；Shell 的公网搜索、普通 HTTP/SSH 和在线 Git 获取失败；固定 Gateway、PTOBench CLI RPC 正常。
- 在 sandbox 存活时至少创建两个 fork；在其中一个 fork 中验证 Candidate 与 `candidate/tuning.json` 可编辑，并新建一个 `.py` helper，由 Candidate 入口导入后运行 Evaluate，确认 Snapshot 收录该文件。workspace 根和结果/context 路径不可替换，Broker 刷新可见，另一 Run 的 Candidate 不可访问。抽查结束时丢弃临时 Candidate，不 finalize 回官方 Seed。
- 真实 Agent CLI 路径至少调用一次 Evaluate 和一次默认 Combined Profile；README 提示与实际行为一致。Evaluate 检查正确性并测量 clean latency/speedup；Profile 提供三类诊断，不做正确性检查，也不代替 Evaluate。
- 有多卡/多机 Case 时，至少选一个代表测试受信任 NPU/HCCL 执行路径；实际 host 数从 Environment/Launcher 核实。管理员 SSH 管理连接在 Agent namespace 外单独确认仍可用。

Basic 和 CANNBot 各选一个 Formal 与一个 Held-out Case，且使用四个不同 Case；建议 Basic 用 `deepseek_v4_flash_mtp.sample_pro` 和 `qwen3_14b.topk_select_k1`，CANNBot 用 `deepseek_v4.prefill_layer_ep2_csa_demo` 和 `deepseek_v4_flash_dspark.prefill_indexer_flash`。这些 Case 均在第 5 节清单中。真实 Pi Session 在父 workspace 上对官方 Candidate Seed 调用 Evaluate/Profile；另在 fork 中做一次可丢弃的 Candidate 编辑并运行 Evaluate，确认 Pi 改写的版本确实经过 Candidate Snapshot 和 Worker 验证，随后丢弃该 fork，不 finalize 回官方 Seed。每个代表 Case 的 Seed Evaluate/Profile 计入 311 次 Evaluate 与 Profile 清单；两次 fork Candidate-edit Evaluate（Basic、CANNBot 各一次）是额外的端到端验证动作。若需要多机覆盖而既定 Formal/Held-out Case 没有跨 host Case，应另选真实多机 Case；不得把 8 卡代替多机。新建独立 Session，测试过程不改写 Ref、Candidate Seed、认证数据或 CV 记录。

### 5. 串行进行 311 Case Evaluate 与 Combined Profile

完整软件回归、622 个 workspace 扫描和两种 mode 的无硬件真实 sandbox 预检通过后，开启一个串行 tmux 队列。队列使用新 Session/workspace，不复用其他 Case 的输出；先跑第 4 节 Basic/CANNBot 代表 Session，确认真实 Agent 的 Evaluate/Profile 链路后继续剩余 Case。任何预检或代表 Session 失败都停止队列并修复，不继续消耗硬件时间。

- 对当前正式清单的 **311 个官方 Candidate Seed 各运行一次 `ptobench evaluate`**；Formal 60 与 Held-out 15 都包括在内并单独汇总。每次 Evaluate 按既有流程检查 correctness 并产出 clean-latency/speedup 数据；本轮只以 Candidate correctness 和执行契约判定通过，不依据 clean latency 设置性能门槛，也不更新 Ref baseline、认证数据或 CV 统计。若当前有效 Ref/timing contract 缺失或不匹配，暂停该 Case 并报告，不得跳过契约检查或自动重建 baseline。
- 从下表的 **11 个基础 Case** 各额外运行一次默认 Combined `ptobench profile`，检查 Runtime、PMU、Compiler 诊断产物及状态/分布式身份。若这些 Case 不含实际跨 host 的多机 Case，再额外选择 1 个并机运行 Profile；因此总数为 11 或 12。表中 Case 的 Evaluate 已计入上述 311 次；Profile 为额外动作。
- 第 4 节的四个 Basic/CANNBot 代表 Case 的 Seed Evaluate/Profile 通过实际 Pi Session 执行，并分别计入本节的 311 个官方 Seed Evaluate 与 11 个基础 Combined Profile；如果额外多机 Profile 使用了其他 Case，它的 Evaluate 仍计入 311 次官方 Seed Evaluate。四个 Case 使用不同 mode/Case 组合，不能把一种 mode 的结果当作另一种 mode 的通过证据。另有两次 Candidate-edit Evaluate（每种 mode 一次），完成后才能继续剩余队列。若代表 Session 失败，先停止，不得继续全量硬件队列。
- 因此本轮硬件动作总数为 **313 次 Evaluate**（311 个官方 Seed + 2 个编辑后 fork）和 **11 或 12 次 Combined Profile**；编辑后 fork 的两次 Evaluate 只验收 Candidate 提交/执行链路，不更新 Seed、Ref、认证性能数据或 CV。
- 一次 Combined Profile 是一次 CLI Action/预算动作，可同时交付三类诊断；内部 Runtime 仍可能包含多个 pass/dispatch。因此要分别记录 CLI Action 数、实际 Runtime dispatch 数和设备执行时间，不能把一个 Profile Action 等同于一次硬件交互。
- Profile 的正确性和 clean latency 由独立 Evaluate 验证；Profile 自身不做输出正确性检查或 clean-latency 基准测量。

| Cohort | Case | Profile 覆盖点 |
| --- | --- | --- |
| Formal | `deepseek_v4_flash_mtp.sample_pro` | 单卡、Operator-level、无 mutable state |
| Formal | `deepseek_v4.prefill_layer_ep2_csa_demo` | 2 卡 mutating、Runtime L3 两 pass 状态恢复 |
| Formal | `deepseek_v4.prefill_layer_ep4_hca_flash` | 4 卡 mutating 路径 |
| Formal | `deepseek_v4.prefill_layer_ep8_csa_demo` | 8 卡 mutating Prefill Layer |
| Formal | `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | 8 卡 CP 路径，与 EP 家族交叉覆盖 |
| Formal | `deepseek_v4.synthetic_token_loop_decode_flash` | Model-level、多卡、Case-local resident 对照 |
| Held-out | `qwen3_14b.topk_select_k1` | 单卡、Operator-level、无 mutable state |
| Held-out | `deepseek_v4_flash_dspark.prefill_indexer_flash` | 单卡、mutating 子图 |
| Held-out | `deepseek_v4.moe_ep_ep2_pro` | 2 卡、非 mutating 分布式路径 |
| Held-out | `deepseek_v4_flash_dspark.dspark_prefill_pro` | 2 卡、Model-level、mutating 路径 |
| Held-out | `deepseek_v4.moe_ep_ep4_flash` | 4 卡、非 mutating 分布式路径 |

这 11 个基础 Profile 用例覆盖 Formal/Held-out 的三种 Granularity；Formal 覆盖 1/2/4/8 卡，Held-out 覆盖 1/2/4 卡。Profile 输出核对 Runtime swimlane 与 dependency graph 的 dispatch 配对、PMU CSV、Compiler `pipeline_profile.json`、`perf_hints.log` 和可用的 AllocateMemoryAddr 产物；分布式用例核对 rank/host 覆盖。若表中没有实际跨 host 的 Case，增加一个经 Environment/Launcher 确认的多机 Profile 用例，此时总数为 12。

## 性能基线与重测范围

`all_case_boundary_repair_20261108` 的 311/311 PASS 及当前有效 Ref 记录作为既有 baseline 证据。单纯调整 Agent workspace 目录与文件读写边界，不重新测 311 个性能 baseline，也不重算 CV。

如果实现改变 Evaluate/Bench 的测量路径、Runtime 状态复位、编译执行契约或 Profile 的设备调用序列，则暂停发布，按受影响 Case 和执行路径重新评估认证/性能重测范围。任何重测均与 workspace 边界验收分开记录，避免把 correctness 验收写成新的性能结论。

## 发布通过条件

以下条件全部满足后，PTOBench 才可用于 Basic/CANNBot 的 Formal/Held-out 正式优化。本轮 Agent 文件视图与网络隔离验收只针对交互式 Pi Agent 的原生文件工具及 Shell/Python；Candidate Worker 的文件视图和网络行为按本规划明确暂缓，不据本轮结果作出安全隔离结论。

1. `lightweight_e2e` 不可见时，正式 CLI、Case discovery、测试和 Profile 仍可运行；无共享 residency inventory 自动改写，既有状态恢复与执行契约测试通过。无关 mode 目录变更不阻塞运行中 Session，也不进入运行时复制；`basic/`、`_shared/`、`cannbot/` 仍按显式依赖处理。
2. Candidate 生产路径统一为小写 `candidate/`；Agent 可创建、修改和删除其中的 `.py` 实现及 helper；新建 helper 能被入口导入，进入 Evaluate/Profile 输入和 Snapshot manifest。311 个 Seed、Snapshot、fork、tuning、finalize 和 README 使用一致的范围；workspace 根、上下文和结果入口只读；动态 fork 在同一活 sandbox 中可写 Candidate、不可替换受保护入口，跨 Run 隔离通过；挂载源固定为当前 Broker/Run，伪造 `PTOBENCH_AGENT_CANDIDATES_ROOT` 等环境值不能改变 Agent 可写挂载。
3. Basic、CANNBot 的 Agent 工具均不能读取 Pi 私有设置/会话、mode `.env`、上游模型凭据、evaluator/launcher/其他 mode 源码和全局策略；Pi 可读取自身所需私有配置。Pi 原生文件工具与 Shell/Python 的私有路径隔离已实现并通过自动化测试，仍须在真实 Basic/CANNBot Pi Session 验证绝对路径、路径遍历、符号链接及 `/proc` 访问均受限。无 Web Search/Web Fetch；普通公网搜索失败；模型 Gateway 和 PTOBench CLI 正常。Agent 工具只针对本轮范围内的交互式 Pi 路径验收。
4. Workspace、Snapshot、Session、Broker、Isolation、Profile、Evaluate、CANNBot 与 Pi 的软件回归通过；Basic/CANNBot 的 622 个 Case workspace 初始化扫描无遗漏和边界失败。
5. Basic 与 CANNBot 分别完成 Formal/Held-out 的真实 sandbox 抽查、真实模型往返、Candidate 修改与新建 helper、动态 fork、Evaluate/Profile CLI 和只读反馈验证；新建 helper 经 Candidate 入口导入并进入 Snapshot；需要的多设备/多机受信任执行路径可用。
6. 311 个官方 Candidate Seed Evaluate 全部 PASS；两次真实 Pi Candidate-edit Evaluate 均通过 correctness；11 个基础 Combined Profile（如需多机覆盖则为 12 个）成功并具备预期诊断产物；正式 Session/Action 记录与 workspace 反馈一致。若某 Case 未运行或执行契约不匹配，不能宣布验收完成。

Candidate Worker 可见 PTOBench 内部实现的边界按已定范围暂不处理；Candidate Worker 网络也按上文限定，本轮不作为 Agent 搜索隔离已覆盖的内容。PTOmni 达到正式运行状态后，另行对全部 Case workspace 创建、其真实 sandbox、动态 fork 和 CLI 反馈做验收；至少覆盖一个 Formal 与一个 Held-out 真实流程，并复用 Combined Profile 覆盖矩阵。
