# PTOmni 接入 Wiki 与 PTOAnalyzer 的设计

> 状态：后续正式 `ptomni` 的集成构思；当前不代表 PTOmni Host 已接线。
> 依据：PTOmni 主架构正文版本 2026-11-08、`ptomni_no_wiki-tool` 消融设计、PTO Wiki 当前内容、PTOAnalyzer Stage 4 离线验收与 PTOBench 方法入口协议。

## 1. 目标

正式 PTOmni 在 `ptomni_no_wiki-tool` 的动态轻量 Agent Team 和 PTOBench 闭环上，增加两个只读 Guidance 来源：

1. **PTO Wiki**：回答 PyPTO、编译器和目标硬件“有哪些已知机制、约束和适用条件”。
2. **PTOAnalyzer**：把当前 Candidate 的 PTOBench Profile 原始产物整理成有来源、单位、覆盖范围和解释边界的观测事实。

两者都提供信息，不代替 Lead 做团队计划、不替 Tuner 选择代码改动、不替 Assembler 决定候选，也不修改 PTOBench 事实。PTOBench `evaluate` 仍是 correctness 与 Clean Benchmark 的权威来源。

## 2. 现状与约束

- 当前主架构已定义 Lead、Assembler、Tuner、按需 Challenger 和 Curator，以及动态 `RoundPlan`、Runner、Event Ledger、派生 State 与 Case Memory。
- `pto_wiki/opt_wiki` 已有离线整理的知识内容，可被固定版本并按需检索；正式运行不应把整库塞进每个 prompt。
- PTOAnalyzer Stage 4 已有九个离线只读工具，按 Compute / Dataflow / Schedule 专长分配；工具绑定 Profile Event、Candidate 与受保护归档身份。
- Analyzer 当前没有 PTOmni Host 的调用入口。Stage 4 已确认这一缺口，因此不可把离线工具验收误称为 Agent 已能使用。
- PTOBench 方法协议规定方法自有 `preflight / prepare / run / collect` 生命周期；方法使用公共受管 `profile/evaluate` CLI。Profile 是硬件动作，但不建立正确性或 Clean Latency；PTOAnalyzer 离线读档本身不产生硬件动作。

因此集成重点是 Host 中两个窄小的读取接口和记录，不是再造知识服务、Profiler、Agent Harness 或评测账本。

## 3. 建议模块与接口

保持 Team Loop 作为主模块：它只知道状态、RoundPlan、ExperimentResult 和工具权限，不直接解析 Wiki 文件或 Profile 文件格式。信息来源在 RoundContext 组装及 Tuner 原生 Tool 注册处接入。

### 3.1 Wiki 读取口

一次实验运行冻结一份 Wiki 快照，记录来源版本 / 内容摘要。由方法侧准备步骤校验并将需要的只读资源放在 Run 可访问位置；Tuner 或 Lead 根据当前 Case、设备和软件版本提出窄查询，返回短片段、来源定位和适用范围。不要把整库或来源不明的摘要直接注入每个 Agent。

Wiki 只能回答已核验的机制资料。查询结果是知识引用，不是 Candidate 的实测证据；如果片段与当前 PyPTO / CANN / 设备版本不匹配，必须保留适用性警示或返回未命中。Wiki 版本在一个匹配实验组内固定，不能让不同 Candidate 获得不同版本知识。

首版使用现有离线内容、轻量关键词 / 元数据检索或已有索引能力即可；检索接口返回少量定位明确的片段。当前没有证据要求向量数据库、知识图谱、在线联网搜索或自适应重排服务。

### 3.2 Profile 分析口

Runner 仍独占 PTOBench `profile` / `evaluate` 调用、预算检查和 Event 记录。Analyzer 不发起动作。PTOBench 成功 Profile 完成后，Host 以已提交 Event 构造受校验的 `ProfileRef`，再只向本轮获准的 Tuner 开放对应专长 `ToolSession`：

```text
RoundPlan 授权 Tuner 与 Profile 动作
  -> Tuner 按需请求 Runner.profile
  -> PTOBench 产生 Profile Event 并归档原始产物
  -> Host 从该 Event 构造 ProfileRef
  -> open_toolset(当前 Tuner 专长, ProfileRef)
  -> Tuner 使用专属 ToolSpec / call(name, args)
  -> ExperimentResult 引用 Profile Event 与 Analyzer 观测
```

Host 只能将 PTOAnalyzer 已实现的原生 `tool_specs` 和 `call(name, args)` 接入 Agent 自身的工具传输；不新建第二个 MCP / Tool Server。Profile 身份、权限和路径约束必须留在 Analyzer 的 `ProfileRef` / `ToolSession` seam 内，禁止由 Agent 传任意路径或请求“最新档案”。

| 调用方 | 允许的 Profile 信息 |
|---|---|
| Lead | RoundContext 中已有的摘要/来源引用；不直接调用 Analyzer 或 Runner |
| 当前启用的 Tuner | 与其 Compute / Dataflow / Schedule 专长对应的只读工具；计划许可时可向 Runner 请求 Profile |
| Challenger | 来自计划和结果包的只读证据；不获得额外 Analyzer 或 Runner Tool |
| Assembler | Tuner 的 ExperimentResult、PTOBench 验证事实；不直接操作 Analyzer |
| Curator | 本轮已记录事实及 Case Memory；不新增采集或重写原始指标 |

Lead 的 RoundPlan 只授予动作种类与预算，不直接触发硬件。Lead 的 Context 可包含已有 Analyzer 派生结果，但其角色权限不变。是否在每轮调用工具由 Tuner 根据计划和证据缺口决定；不得为了填满工具清单自动多采 Profile。

### 3.3 共享证据契约

RoundContext 与 ExperimentResult 需要区分三种东西：

- **PTOBench 事实**：Candidate / Event 身份、correctness、Clean Benchmark、硬件动作及运行状态；由 PTOBench 产生。
- **Analyzer 观测**：结构化数值、单位、来源覆盖、源文件引用和限制；由 PTOAnalyzer 确定性生成，始终引用原 Profile Event / Candidate。
- **Agent interpretation**：对知识和观测的解释、假设与实验判断；由相应 Agent 产生，不能升级成前两类事实。

丢失、空、损坏或不兼容来源必须分别表达，不填零；跨设备、Shape、Candidate、编译配置和软件环境无法证明可比时，不生成直接差值。Analyzer 的结果是可重建派生数据，不建立第二份事实账本或长期 Profile 数据库。

## 4. 在消融基线上的演进

`ptomni_no_wiki-tool` 与正式 `ptomni` 应共享同一套 Team / Runner / State 行为，差异集中在是否装配 Wiki 与 Analyzer 读取口。保持无 Wiki / 无 Analyzer 分支能在零工具暴露时运行；正式分支才提供资源，避免把核心闭环写成必须依赖指导信息才能启动。

| 层 | 无 Wiki / 无 Analyzer | 正式 PTOmni |
|---|---|---|
| 团队规划和角色生命周期 | 相同 | 相同 |
| Runner、PTOBench CLI、预算和候选身份 | 相同 | 相同 |
| Case Memory、State 和事件记录 | 相同 | 相同 |
| PTO Wiki | 无快照、无检索、无片段暴露 | 固定快照，按需取短片段 |
| 原始 Profile | 保持 PTOBench 一致的 Agent 可见性 | 保持一致，并增加结构化 Analyzer 观测 |
| Analyzer | Tool 集合为空 | 仅向获准 Tuner 注册对应专长工具 |

第一阶段先用无 Wiki / 无 Analyzer mode 验证“动态团队到 PTOBench 正式验证”的垂直闭环。后续通过同一个知识和工具接线位置增加能力，避免复制一套 PTOmni 或维护两份独立团队策略。初期不要为将来所有可能知识源设计插件系统；真实需要的两种只读来源有明确使用者和数据类型，足以用小接口表达。

## 5. 集成顺序

### 阶段 A：建立消融闭环

先实现并验收 `ptomni_no_wiki-tool` 的最小可运行 Team：一个 Lead、一个被启用 Tuner、Assembler、基础 State / Case Memory 和 PTOBench Runner。先证明 Agent 可在方法协议中完成一次受预算控制的 Candidate 探索和验证；不等待 Wiki 与 Analyzer 接线完成。

### 阶段 B：稳定共享信息契约

用一轮真实或受控的端到端轨迹核对 RoundContext、RoundPlan、Candidate / Event 引用、ExperimentResult、State 更新和 collect 记录。把知识暴露、Profile Event 和 Tool 暴露列入可复核记录。只有有真实调用者时才固定接口，不先构建通用容器或多层插件。

### 阶段 C：接入 Wiki

沿方法 `prepare` 的资源冻结流程加入固定 Wiki 快照和轻量检索。先在 CPU-only 查询样例中检索版本适用、来源引用、空结果和上下文过滤；确认 prompt 不会注入整个知识库，Run 记录能够复原 Agent 实际看到的片段。此阶段不改 Runner、硬件预算、角色权限和 Case Memory。

### 阶段 D：接入 PTOAnalyzer Host 工具

在现有 PTOmni Host / Pi 工具注册处加入薄适配。验证 Profile Event 到 `ProfileRef` 的身份绑定、Tuner 专长权限、Tool 调用只读、Cursor 续页、错误传递、来源覆盖和工具暴露记录。Lead、Assembler、Challenger 和 Curator 不获得 Tuner 工具。CPU Fixture 测试不运行 NPU，也不启动新的 Profile。

### 阶段 E：匹配比较

使用同一 Team / Runner / Case / Candidate 初始化 / 预算与停止规则，对比无 Wiki / 无 Analyzer 和正式 PTOmni。正式组与消融组都保留相同 PTOBench 原始 Profile 可见性和可请求次数；唯一计划内差异是 Wiki 检索及 PTOAnalyzer 结构化工具。按 Case 记录 correctness、Clean Benchmark、动作次数、无效/重复尝试、Token 与墙钟成本。

这个对比给出 Wiki 与 PTOAnalyzer 的联合增量。若需要分别估计两者，增加匹配的 Wiki-only 与 Analyzer-only 条件；不改变本架构的核心模块。

## 6. 安全与实验不变量

1. PTOBench Runner 是唯一执行和硬件预算入口；Analyzer 只读已有 Profile。
2. 每项结构化 Profile 结果绑定确切的 Profile Event、Candidate 与兼容环境；不同身份不混比。
3. Wiki 快照在一个对照批次固定，来源和 Agent 实际读取片段可追溯。
4. Lead 制定 RoundPlan，不持有 Runner 或 Analyzer Tool；Tool 授权只针对本轮已启用的 Tuner 专长。
5. PTOAnalyzer 结果、Wiki 片段和 Agent 解释均不能覆盖 PTOBench correctness、Clean Benchmark 或动作记录。
6. Case Memory 继续只保存当前 Case 的短经验，不变成 Wiki 的隐式动态副本。
7. Profile 缺失或工具不可用时明确返回 unavailable；系统可以继续根据已有状态完成本轮，不能因此伪造结论或启动超出预算的动作。

## 7. 后续评审门槛

当前 PTOAnalyzer 已通过离线工具和归档回放，但 Host 接线尚未完成。只有仓库中出现实际 PTOmni Host / Pi 原生工具注册入口后，才实施阶段 D；不要为阶段 D 另建模拟 Host、第二套 Agent Harness 或另一个 Tool Server。若 Host 尚不存在，先把 PTOmni 最小 Host 与 Team Loop 实现到足以承载这两个读取接口，再接入 Analyzer。

触发硬件的配对试验应单独使用已授权的 Case、设备和动作预算。文档设计和 CPU Fixture 验证不启动 `profile`、`evaluate` 或 NPU 工作负载。

## 8. 依据

- [PTOmni 当前架构设计](PTOmni_架构设计_2026-11-03.md)
- [无 Wiki / PTOAnalyzer 消融设计](../../ptomni_no_wiki-tool/design/PTOmni-NoWikiTool_架构设计.md)
- [PTOBench 方法入口协议](../../PROTOCOL.md)
- [PTOBench Agent CLI](../../../evaluator/AGENT_CLI.md)
- [PTO Wiki](../../../pto_wiki/opt_wiki/README.md)
- [PTOAnalyzer 当前接口与离线验收状态](../../ptomni/pto_analyzer/README.md)
- [PTOAnalyzer Stage 4 Host 接线与验收记录](../../ptomni/pto_analyzer/planning/stage4/README.md)
