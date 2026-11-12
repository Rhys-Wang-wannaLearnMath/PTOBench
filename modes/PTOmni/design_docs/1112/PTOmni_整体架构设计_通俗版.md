# PTOmni 整体架构设计：通俗版

> 这份文档是架构设计的快速导读，重点讲“系统怎么配合工作”和“每个角色做什么”。它不替代[完整架构设计](PTOmni_整体架构设计_2026-11-12.md)；接口、字段和未定事项以完整文档为准。

## 先用一句话理解 PTOmni

PTOmni 是一支按任务临时组队的优化小队：它先提出一个可验证的优化想法，再让 Agent 在隔离的代码副本里尝试，最后由 PTOBench 检查正确性和性能，并把有依据的经验留给同一个 Case 的下一轮。

它不是一群一直同时运行的 Agent。角色按任务启动；本次 Run 的第一个优化 Round 固定启用 Compute、Dataflow、Schedule 三个 Tuner。之后每轮由 Lead 自己决定启用哪些、启用几个，系统不暗示多选或少选，也没有默认人数。

## 把它想成一个有实验室规矩的小组

| PTOmni 里的东西 | 可以把它想成 | 它负责什么 |
|---|---|---|
| **PTOBench** | 实验室里的仪器和正式记录本 | 运行代码、检查结果、记录硬件动作和评测事实 |
| **Host** | 实验室管理员 | 安排角色、检查权限、整理上下文、记录过程 |
| **Lead** | 组长 | 判断现在最该回答什么问题，并安排谁做什么 |
| **Tuner** | 实验员 | 改一份自己的代码副本，提出并验证优化想法 |
| **Assembler** | 结果整理者 | 比较实验结果，选出本轮要保留的候选代码 |
| **Challenger** | 临时复核者 | 在有疑点时检查推理和证据，提出反例或补充实验 |
| **Curator** | 实验记录整理者 | 每个正常结束的 Round 更新一次本 Case 的经验摘要 |
| **PTO Wiki** | 技术手册 | 查 PyPTO、编译器、运行时、设备等已有知识 |
| **PTOAnalyzer** | Profile 读数工具 | 把某次真实运行留下的 Profile 文件整理成可读观测 |
| **Case Memory** | 这个 Case 的短实验日志 | 记住试过什么、哪些结论有证据、还有什么没弄清楚 |

表格里的“角色”是工作职责，不代表必须启动同样数量的独立 Agent 进程。Tuner 的数量规则是：首个优化 Round 固定三个专长都启用；之后由 Lead 自主选择任意有效组合，不设推荐人数。角色任务可以按顺序完成。

## 谁来决定结果是真的

PTOBench 是 correctness、评测性能和硬件动作记录的权威来源。PTOmni 可以提出解释，但不能把解释改成测量事实，也不能自己改正式成绩。

这一区分很重要：

- **Profile** 像仪器采集的诊断读数。它能帮助找线索，但不能证明代码正确，也不是正式性能成绩。
- **Evaluate** 会检查候选代码是否正确，并测量探索阶段的性能结果。
- **正式最终成绩** 仍由 PTOBench 的受控完成流程产生。Agent 不能直接宣布成绩。

PTOmni 通过 PTOBench 的公共命令请求这些操作。它不另写一套评测器，也不自行运行 NPU 命令来绕过 PTOBench。

## 一次优化是怎么进行的

假设当前 Case 的 Kernel 有一个性能问题。

1. **准备实验。** PTOmni 检查配置和工具是否可用，固定本次使用的 Wiki 版本，并读取 PTOBench 提供的 Case、代码工作区和剩余预算。
2. **整理已知情况。** Host 看当前代码、以前的评测结果、已有 Profile 摘要和本 Case 的经验记录，挑出少量相关 Wiki 内容。
3. **Lead 定一个问题。** 比如：“数据搬运是不是这次变慢的主要原因？”Lead 把问题、要做的实验、可用角色和硬件动作上限写成计划。Host 会检查计划有没有越权或超过预算。
4. **Host 先采 Profile。** Lead 选好本轮 Tuner 并通过权限、预算校验后，Host 在任何 Tuner 改代码前，对当前 Parent Candidate 采一份新的 `combined` Profile。每个优化 Round 都重新采，不复用历史 Profile；入选的 Tuner 共用这份 Profile 的事件，但各自只拿到对应专长的 PTOAnalyzer 工具。
5. **Tuner 做实验。** 首轮由三个专长 Tuner 分别尝试；后续轮次按 Lead 选出的组合启动。每个 Tuner 只能读取只读的 Parent workspace，并读写自己的 workspace；其他 Tuner 的 workspace 对它不可见、不可访问。Tuner 根据 Profile 读数和相关资料开展实验，并可按授权请求 Evaluate。
6. **Evaluate 候选代码。** Tuner 可以请求 PTOBench 检查正确性和探索性能。每次结果都关联到实际运行的代码版本，不能把不同代码或不同环境的读数混在一起比较。
7. **Assembler 做本轮决定。** 它可以选一个通过验证的候选、保留原代码，或要求一个有明确目的的补充实验。如果它把多个改动组合成新版本，新版本也必须重新验证。
8. **Curator 整理本轮经验。** 每个正常完成的 Round 调用一次 Curator。它把已经提交的结果整理进 Case Memory，并区分“测到了什么”和“我们怎么解释”。它不改 PTOBench 记录。
9. **开始下一轮或停止。** Lead 根据新状态决定下一步。预算用完、结果没有新信息或优化目标已经满足时，可以停止。

每轮这一份 `combined` Profile 会交给所有入选 Tuner 的专长 Analyzer 工具使用，不会按 Tuner 人数重复采集。对同一份 Profile 做多次离线分析不会额外启动硬件测量；新的 Profile（每轮一次）或 Evaluate 仍按 PTOBench 预算记账。Assembler 不需要自己采 Profile，也没有直接调用 Analyzer 的工具。

## Round、Experiment 和 Event 有什么区别

可以用“三层计数”来记：

| 名称 | 通俗解释 | 举例 |
|---|---|---|
| **Round** | 围绕一个主要问题进行的一整轮计划、实验、选择和总结 | “确认是不是数据搬运导致变慢” |
| **Experiment** | 为回答问题而尝试的一种代码改动 | “改变数据布局后再测一次” |
| **PTOBench Event** | PTOBench 实际执行并记账的一次动作 | 一次 Profile、Evaluate 或 workspace fork |

一个 Round 可能有几个 Experiment；每轮开始时有一份 Parent Profile，后续 Evaluate 等也会产生 PTOBench Event。它们不是同一个计数器。Curator 按正常完成的 Round 调用，不是每做一次 Experiment 或 Event 就调用一次。

每份代码版本叫一个 **Candidate**。Profile、正确性和性能结果都要指向实际运行的 Candidate 和 PTOBench Event。这样才能知道“这条结果到底是哪个代码版本跑出来的”。

## 三种知识各解决什么问题

### PTO Wiki：以前已经知道的技术知识

它回答：“这个 API 或硬件机制有什么规则？什么写法可能更合适？”一次 Run 使用固定的只读版本，只把少量相关内容交给 Agent，并记录来源。Wiki 不负责判断当前代码实测快不快。

### PTOAnalyzer：这次运行留下了什么读数

它读取 PTOBench 已经保存的 Profile，整理计算、数据流和调度等方面的观测。它不运行硬件、不改代码，也不自动宣布根因。Tuner 结合读数和其他证据提出解释。

目前 Analyzer 本身已有工具和离线验证；它还需要由 PTOmni Host 接入后，Tuner 才能在优化过程中调用。

### Case Memory：这个 Case 之前试过什么

它保存本 Case 有用的近期经验，比如哪种改动失败、哪些结果已经验证、还有什么问题没解决。它是短摘要，不是完整聊天记录，也不能替代 PTOBench 的原始记录。

## 为什么要有 Host、State 和 Event Ledger

Host 把“Agent 的想法”和“系统实际做过的事”分开管理：

- Agent 提交计划和实验结果；Host 检查格式、角色权限、代码版本和预算。
- PTOBench 记录真实执行过的 Profile、Evaluate 等动作。
- Event Ledger 记录 PTOmni 的计划、角色输出、代码变更引用、错误和决策，并引用 PTOBench 事件。
- State 根据这些记录算出当前候选、已验证最佳候选、预算和 Round 进度。
- Case Memory 只是 Curator 生成的摘要，不能覆盖上面那些事实。

所以即使 Agent 说“我跑过了”，如果没有对应的 PTOBench Event，也不能把它当成已经完成的硬件实验。

## 小队怎么保持轻量

- 首个优化 Round 固定启用 Compute、Dataflow、Schedule 三个 Tuner；此后每轮选几个、选哪些，由 Lead 决定。Host 不为人数设置默认值、建议或自动调整。
- Challenger 不固定出场。出现高成本决定、结果冲突或证据解释不清时才请它复核。
- 角色通常顺序工作，避免同时修改同一份代码和重复消耗上下文。
- 每个 Tuner 只能读 Parent、写自己的代码副本，看不到其他 Tuner 的 workspace；该限制由实际隔离和文件权限执行。Assembler 组合代码后，组合出的新版本也要重新验证。
- Assembler 综合正确性、可比性能、兼容性、证据质量和集成风险选本轮交付候选；它不必选性能数字最高的版本，也不必选自己组合的新版本。Host 根据 PTOBench 事实分别维护本轮 Parent (`current_candidate`) 和全局最好成绩 (`best_candidate`)，二者可以不同。
- 当前 PTOBench 工作流按串行硬件实验设计；不能因为 Agent 可以并行，就同时发起 NPU 测量。
- 首版不需要通用 Agent 平台、数据库、向量搜索或跨 Case 自动学习。

## 现在有什么，接下来还要做什么

- `pto_wiki/opt_wiki` 已整理好，可作为 PTOmni 的只读知识来源；还要做的是 Host 侧的按需检索、版本固定和实际曝光记录。
- `modes/ptomni/pto_analyzer` 已有只读分析工具和离线验证；还要做的是把它接到真实 Host / Agent 工具调用中。
- PTOmni 的 Agent Team、Host、状态记录和 Runner 还需要实现。
- 当前 PTOBench Launcher 能根据 `mode.yaml` 动态加载方法，但隔离运行时的公共源码清单还没有登记 `ptomni`，而且需要把 Analyzer 子包一并带进运行时。因此首次隔离运行仍有一处公共运行时接入工作；PTOmni 的业务逻辑仍可以留在自己的 mode 目录里。

实现可以分阶段进行，早期用 fake Runner / Analyzer 做 CPU 测试；但实际可用的优化流程必须包含固定首轮三个 Tuner、每轮开始时的 Parent Profile 与 Analyzer 分发，以及隔离的 workspace。后续 Round 的团队人数始终由 Lead 决定，不能把单 Tuner 设成实际运行默认。

## 仍需要讨论的设计问题

完整设计还留下了一些需要后续确定的细节：

- 不同角色是用各自独立的会话，还是复用会话？怎样避免权限和上下文串在一起？
- Wiki 由 Host 给 Lead 摘要就够，还是也要给 Tuner 搜索工具？
- Profile 失败或剩余预算不足时，怎样安全恢复或结束 Round？固定规则是不复用历史 Profile，也不在 Profile 未完成时启动本轮 Tuner 修改。
- Lead 应该看到最近一次 Analyzer 摘要，还是有限数量的历史对比？
- Lead 的后续 Tuner 人数决策需要看到哪些历史实验和证据摘要？Host 不负责设置人数偏好或自动改写选择。
- Case Memory 保留多少轮，怎样淘汰过期经验？
- 将来 PTOBench 支持设备绑定和硬件并行时，怎样安排公平的并发实验？
- Token 和运行时间等成本能否从 Agent backend 可靠获得？
- 除了预算用完，什么证据说明优化该停止？

这些问题不妨碍先实现最小闭环；应结合真实运行记录逐步确定，不需要预先做成复杂框架。

## 最值得记住的五点

1. **PTOmni 是按需组队的小队，不是固定常驻的五个 Agent。**
2. **Lead 决定要回答的问题；Tuner 做实验；Assembler 选本轮结果；Curator 整理每轮经验；Challenger 只在需要时复核。**
3. **PTO Wiki 是技术手册，PTOAnalyzer 是 Profile 读数，Case Memory 是本 Case 的短实验日志。**
4. **PTOBench 才是运行、正确性、测量和硬件预算的事实来源。Profile 帮助诊断，Evaluate 验证候选。**
5. **首个优化 Round 固定启动三个 Tuner；此后的 Round 由 Lead 自主决定团队数量和组合。Profile 每轮对 Parent 新采一份并共享给入选 Tuner。**

## 进一步理解：角色怎么协作、谁能改代码

下面是把完整架构落到实际工作流程后的建议。PTOAnalyzer 本身已有可调用的只读接口；Host 接线和角色权限控制仍属于 PTOmni 待实现部分。

### 1. 不同 Agent 角色如何沟通？

建议第一版让角色**通过 Host 交接结构化结果，不让它们直接互相聊天**。Host 像接力员：一个角色做完，把结果交给 Host；Host 检查后，只把下一位角色需要的内容传过去。

一次交接大致是：

    Lead 写 RoundPlan
      → Host 检查并记录
      → Tuner 收到计划、自己的代码副本和所需资料
      → Tuner 返回 ExperimentResult 和证据引用
      → Assembler 收到结果包，选择候选或要求补充实验
      → Host 记录验证结果
      → Curator 收到已提交记录，更新 Case Memory

交接内容可以是带版本号的 JSON 或 Markdown 记录，例如 RoundPlan、ExperimentResult、Assembler 决定和 Round Note，保存在本次 Run 的方法记录中。Host 负责校验格式、角色权限和证据来源，再将必要摘要传给下一角色。首版 Tuner 按计划顺序执行；若未来允许同阶段并行，Host 会等参与者完成后再把结果交给 Assembler。无论串行还是并行，Tuner 都不直接修改彼此文件，也不依赖互相可见的对话历史。

这样比“所有 Agent 共用一段聊天记录”容易追溯：出了问题能查清谁提出了什么、依据是什么、下一角色看到了什么。第一版不需要消息队列或常驻协作服务器。

### 2. 谁能 fork、谁能修改 workspace？

这里要把“请求创建副本”和“真正执行 fork”分开。建议由 PTOmni 的 Host / Runner 执行 PTOBench 的 fork 命令；Agent 只能通过 Host 请求一个已授权的工作区，不能自行挑选任意父目录或创建未登记的 workspace。PTOBench 的 fork 会从已登记的父 workspace 建立子 workspace 并生成新的上下文，具体见 [PTOBench Agent CLI 文档](../../../../evaluator/AGENT_CLI.md)。

| 角色 | 可以修改什么 | fork 由谁执行 |
|---|---|---|
| **Tuner** | 只读 Parent workspace，只修改分配给自己的实验 workspace；看不到其他 Tuner workspace | Host / Runner 按计划为它从 Parent Candidate 创建副本 |
| **Assembler** | 只修改自己的组装 workspace；只有要组合改动时才需要 | Host / Runner 在需要组装时创建；若只选一个现成候选，可以不 fork |
| **Lead** | 不修改 Candidate workspace | 不 fork |
| **Challenger** | 只读，不修改 Candidate | 不 fork |
| **Curator** | 只写 Case Memory 的新版本，不改代码 | 不 fork |

Tuner 不能自行请求 Profile；Host 在每个优化 Round 开始时统一对 Parent 采一份 Profile。Tuner 可在计划授权范围内请求 Evaluate，但这不代表它能把自己的实验结果直接定为本轮交付候选。Assembler 若组合了多个改动，就在自己的 workspace 里形成新 Candidate，并对这个新版本重新 Evaluate；其他 Tuner 的 workspace 保持原样。Assembler 要求补充实验时只能使用 Lead 已选团队和已批准的权限；如果要换 Tuner 组合，由 Lead 开启新 Round 决定。

要让“只能读 Parent、只能改自己的 workspace、看不到其他 Tuner workspace”成为实际限制，不能只写在 Agent 提示里。Host 应只向 Tuner 挂载只读 Parent 和自身可写 workspace，并用文件权限或运行隔离阻断其他 workspace 路径；所有 fork、Profile 和 Evaluate 都由受控 Runner / Host 流程执行并检查预算。Assembler 通过 Host 收到结果包和受控 Candidate 引用，不直接获得所有 Tuner 的工作目录。

### 3. 最终提交由 Assembler 负责，怎样确保只有它能提交？

先区分两种“最终”：Assembler 决定的是**本 Round 交付哪个 Candidate**；实验结束后的正式完成与最终成绩仍由 PTOBench 的受控流程负责。Assembler 不能直接写正式成绩。

建议这样限制本轮候选提交：

1. Tuner 只能交付自己的 ExperimentResult，也可以按计划请求探索性验证；不能改写当前 Best 或宣布自己是最终候选。
2. Assembler 返回选中的 Candidate 和理由。Host 只接受来自当前 Round Assembler 会话的决定，并检查 Candidate 是否存在、代码身份是否匹配、所需验证是否通过。
3. Host 按这个决定调用 Runner 验证；Runner 使用 PTOBench 公共命令执行，不接受 Agent 自己给出的任意路径或 Event 编号。
4. 只有 Host 的状态记录能把 Candidate 标成当前 Round 的交付候选；正式 finish 仍由 PTOBench / 实验管理流程完成。

关键点是**权限靠工具和执行入口限制，不靠提示词自觉**。PTOBench 公共 CLI 能识别 Session 和已登记的 workspace，但不识别 PTOmni 的 Lead、Tuner 或 Assembler 角色；所以光靠 PTOBench CLI 不能区分谁提交了候选。如果每个 Agent 都能直接在 shell 里运行 PTOBench 命令，那么“只有 Assembler 能提交”就只是约定，无法严格保证。要做到强制限制，角色进程应拿不到可绕过 Host 的 PTOBench 直接入口；它们通过 Host 提供的角色受限工具请求动作，Host 代表经授权的角色调用 Runner。这是建议的实现契约，不是目前已经完成的功能。

### 4. PTOAnalyzer 怎样交给三个专长 Tuner？

PTOAnalyzer 已有适合接入的 Python 调用方式。Host 在 PTOBench 成功完成一次 combined Profile 后，拿到确切的 Profile Event，再用它建立 ProfileRef。之后按 Tuner 专长打开对应的只读工具集合：

    同一个 Profile Event
       ├── Compute Tuner  → Compute 专属的 3 个工具
       ├── Dataflow Tuner → Dataflow 专属的 3 个工具
       └── Schedule Tuner → Schedule 专属的 3 个工具

实际接线可以这样理解。首个优化 Round 固定有三个专长 Tuner；之后 Lead 可选择任意有效组合。无论入选几个，Profile 都只采一次：

1. 每个优化 Round 开始时，Host 对 Parent Candidate 采集一份新的 `combined` Profile，不复用历史 Profile；所有 Tuner 改代码前都使用这份数据。
2. Host 根据成功的 Profile Event 创建 ProfileRef，绑定真实事件，而不是让 Agent 自己挑 Profile 文件；随后为本轮每个入选 Tuner 打开对应专长的 ToolSession。
3. Host 把 ToolSession 提供的工具名称、说明和参数格式注册到该 Tuner 的 Agent Session。这些定义告诉 Agent“有哪些工具可以调用”。
4. Tuner 请求某个工具时，Agent Session 把调用交给 Host 的工具回调；回调调用对应 ToolSession，再把结构化结果返回给 Tuner。
5. Host 记录调用和结果引用。所有入选 Tuner 可以分析同一个只读 Profile，但每个 Tuner 只拿到与自己专长对应的工具集合；这不意味着它们能读取彼此的 workspace。Assembler 可读结果摘要和来源引用，不直接使用 Analyzer 工具。

所以集成重点不是重写 Analyzer，也不是给每个 Tuner 拷贝一份 Profile，而是在 PTOmni Host 的 Agent 工具注册和回调处加一层薄 Adapter。现有 Analyzer 的 open_toolset、tool_specs、initial_summary 和 call 已提供主要接入面，见 [ToolSession 实现](../../../ptomni/pto_analyzer/toolset.py)。目前缺少的是 PTOmni Host 对这套接口的调用和 Agent 工具传输接线。

### 5. Challenger 什么时候进入 Round？

Challenger 不应该是每轮固定成员。当前设计列出的触发情形是：即将进行高成本或高风险动作、Tuner 结论冲突、解释与观测不一致、性能回退原因不清，或连续实验没有带来新信息。

最简单的流程是：Lead 在 RoundPlan 中说明是否允许 Challenger、什么情况触发；如果触发条件在 Tuner 返回结果后才出现，Host 可在 Assembler 做昂贵决定前安排一次只读复核。Challenger 只看相关计划、结果和证据，返回反例、证据缺口或区分性实验建议；它不改代码、不直接运行硬件，也没有否决权。

若 RoundPlan 没有预先允许 Challenger，Host 不应悄悄扩大权限；可以将问题交给 Lead 开启一个明确授权的新 Round。是否允许在一个 Round 中途追加 Challenger，是完整设计还需要定下来的细节。建议首版在 RoundPlan 中预留一个可选的只读复核名额，未触发就不调用。

## EAT 对这些问题有什么参考

这里的 EAT 指同目录 references/EAT 保存的 Evolving Agent Teams 设计和代码。它有些问题与 PTOmni 相似，但它的目标、运行环境和权限保证不同；“脚本能运行”不代表所有权限都由系统强制执行。

| 问题 | EAT 怎么做 | 对 PTOmni 的参考和差异 |
|---|---|---|
| **角色沟通** | Supervisor 每轮生成角色提示和 Workflow；同一 Phase 的 Agent 并行，Phase 之间顺序执行。每个 Agent 的输出写到自己的文件，Harness 汇总成 TEAM_OUTPUT.md，供下一轮 Supervisor 阅读。没有专门的 Agent-to-Agent 消息系统。 | 借鉴“用阶段和文件交接就够了”。PTOmni 可让 Host 把结构化 Round 产物交给下一角色，不需要消息队列；同时保留 Candidate / Event 引用。 |
| **各角色工作区** | EAT 的任务说明建议 Agent 在 work_role 这样的副本里工作，由交付新最佳结果的角色把结果放回主目录。启动脚本按 Phase 启动 Agent，但它们从同一个项目目录启动。 | 简单有效，但主要依靠目录约定和提示，不是每个角色各自的硬权限隔离。PTOmni 应使用 PTOBench 登记的独立 workspace 和 fork，避免多个 Tuner 修改同一 Candidate。 |
| **谁做最终提交** | EAT Harness 在团队结束后运行官方评测，再由外层脚本对本轮改动执行 git add 和 git commit。不是由角色身份权限系统保证只有某个角色能提交。 | 借鉴“由外层 Harness 统一评测和记录”。PTOmni 应让 Assembler 决定候选，让 Host / Runner 执行受控验证；正式 finish 仍留给 PTOBench。 |
| **Profile / Analyzer 工具** | EAT 的 Agent 使用 Claude Code 的工具和命令完成优化、评测及分析；这些材料没有 PTOmni 这种按 Profile Event 和专长角色绑定的只读工具集合。 | PTOmni 需要自己接线：Host 根据 Profile Event 打开专长 ToolSession，再把工具定义和回调注册给对应 Tuner。这不是可以直接从 EAT 复制的部分。 |
| **Challenger 何时加入** | EAT Supervisor 阅读上一轮输出和成绩，再为下一轮重设角色、提示和工作阶段；遇到停滞时可以创建针对问题的新角色。 | 借鉴“根据证据变化角色，而不是固定开满团队”。EAT 主要在下一轮调整团队，不等于 PTOmni 在当前 Round 中途插入 Challenger。PTOmni 可按风险、结果冲突或停滞触发一次只读复核。 |

EAT 的简化方法是：Supervisor 写团队配置，Harness 按 Phase 启动 Agent，把输出落盘，再统一评测和总结。它不需要复杂的实时 Agent 通信设施。PTOmni 可以沿用这个简单思路，同时利用 PTOBench 的受管 workspace、事件记录和预算，把重要约束交给 Host 检查，而不是只写在角色提示里。

参考材料：[EAT README](../references/EAT/README.md) 介绍整体流程；[EAT Skill](../references/EAT/SKILL.md) 说明工作区副本和交付习惯；[launch_team.sh](../references/EAT/agent_team/launch_team.sh) 与 [parse_team_config.py](../references/EAT/agent_team/parse_team_config.py) 展示 Phase 启动、提示解析和输出归档；PTOmni Analyzer 接口见[完整架构设计 §7.2](PTOmni_整体架构设计_2026-11-12.md) 与 [ToolSession 实现](../../../ptomni/pto_analyzer/toolset.py)。
