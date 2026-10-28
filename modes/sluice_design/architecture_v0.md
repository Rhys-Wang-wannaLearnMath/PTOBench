# SLUICE 单 Pi 持续会话架构（修订版 v0）

更新日期：2026-10-27。

这份文档是 `benchmarks/PTOBench/modes/sluice_design` 的当前初步架构基线。它取代本目录此前提出的“每个优化轮次关闭 Pi、再启动一个新 Pi”的方案。

**当前决定：一次 PTOBench 优化运行只启动一个 Pi 实例。Pi 在整个运行期间持续工作；优化轮次是逻辑上的工作周期，不是 Pi 进程的生命周期。** 后续仍可以根据真实运行删减或调整模块，但不能把“多 Pi 接力”重新当作本架构的默认前提。

## 先用通俗的话说明

SLUICE 像给同一个工程师配了一本持续更新的工作台：

1. Pi 启动一次，连接到一个 PTOBench Session 和一个 SLUICE Controller。
2. Pi 提出假设、改代码、请求评测或 Profile，并读取结果。
3. 一项优化尝试完成后，SLUICE 把客观结果写入唯一可信状态。
4. Pi 下一次向模型发请求前，SLUICE 从这份状态重新整理一张“当前工作单”，替换掉旧的 SLUICE 状态快照。
5. Pi 在同一个会话中继续下一项尝试，直到明确停止或系统保护条件触发。

SLUICE 负责每次模型请求前刷新当前状态投影，让 Pi 看到最新代码、结果和待办；Pi 自行管理对话历史及其上下文长度。这两类职责不混在一起。

## 设计目标

- 用一个长期运行的 Pi 会话完成多次有反馈的优化尝试，保持模型、工具和 PTOBench Session 的连续性。
- 让每次模型请求都以当前可信状态为准，而不是依赖模型记住旧对话中的最新事实。
- 让 PTOBench 的正确性、性能、预算和 Profile 结果成为事实来源；Pi 的机制解释保持为解释。
- 让知识库和 Profiling Tool 成为可插拔的只读知识/证据入口，不把原始知识或原始 Profile 大量塞进每次 prompt。
- 用一个 Controller、一个 Progress 状态和一个 ContextRenderer 组织运行，避免重复的 Memory、History、Frontier 或 State Store。
- 支持 Pi 根据进展主动停止，也支持在预算或运行约束触发时停止；决策和测量结果通过 PTOBench 运行记录及 Progress State 追溯。中断后恢复暂不列为最小架构要求，待基础闭环稳定后再根据实际需求决定。

## 整体结构

```text
                         一个 PTOBench Session
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
               ▼                                     ▼
       SLUICE Controller                       一个持续运行的 Pi
       与唯一可信状态                           （一次启动）
               │                                     │
       ┌───────┼────────┐              ┌─────────────┼─────────────┐
       │       │        │              │             │             │
       ▼       ▼        ▼              ▼             ▼
   Progress  Evidence  PTOBench    context hook   工具调用
   Reducer   /Profile  Adapter     → 状态投影      → Controller
       │       │        │              │             │             │
       └───────┴────────┴──────────────┴─────────────┴─────────────┘
                                  │
                                  ▼
                   下一次模型请求继续同一个 Pi 会话
```

这些方框表示职责和接口，不要求每个方框都是独立进程。当前实现的主要落点是：

- Pi 接入和 Context hook：`modes/sluice/bridge/pi_sluice.ts`；
- 状态投影：`modes/sluice/engine/context_renderer.py`；
- 工具门面和动作编排：`modes/sluice/engine/controller.py`；
- 状态 Journal/Reducer：`modes/sluice/engine/progress_port.py`；
- Profile 证据入口：`modes/sluice/engine/profile_adapter.py`、`evidence_port.py`；
- 只读知识入口：`modes/sluice/engine/knowledge_port.py`。

## 各模块的职责

### 1. PTOBench 运行底座

PTOBench 创建并管理 Session、Candidate workspace、公开的 evaluate/profile/fork action、硬件动作预算、correctness、clean latency 和运行产物。SLUICE 和 Pi 必须通过这些公开入口取得正式结果，不能绕过 PTOBench 自行宣布结果有效。

Pi 可以读取和修改自己有权限的 Candidate 源码。它不能读取封存答案、隐藏 Fixture 或内部 Oracle，也不能把未经统一评测的数字当成正式结果。`fork` 产生的辅助副本不能自动变成第二个 SLUICE 状态；当前主 Candidate 和版本选择规则仍由 Controller 管理。

### 2. Pi 与桥接层

Agent harness 只启动一次 Pi，并把它连接到一个长期存在的 Controller 工具桥。桥接层负责：

- 注册和转发 SLUICE 工具；
- 在每次模型请求前触发 ContextRenderer；
- 保持消息中的 assistant tool call 与 tool result 配对；
- 在 Context 不可信或 Controller 出错时阻止会改变源码或消耗硬件预算的动作；
- 在 Session 结束、取消或异常时关闭 Pi 和子进程。

桥接层不选择优化方向、不解释 Profile，也不维护一份独立的 Memory。Pi 可以在一个逻辑周期内连续调用多个工具；周期结束并不关闭 Pi。

### 3. Controller 与唯一可信状态

Controller 是 Pi 工具的唯一门面。它把请求路由到 PTOBench、KnowledgePort、EvidencePort 和 ProgressPort，并检查调用顺序、源码身份、假设引用、预算和恢复状态。

Progress Journal/Reducer 是唯一的优化状态来源。它至少表达：

- Optimization Contract、环境和剩余预算；
- 当前 working program、last verified program、best verified program；
- 当前 Decision 和 active Hypothesis；
- 已完成的逻辑周期、客观评测结果和 EvidenceBundle；
- 当前仍待处理的问题或机会；
- stop/finish 状态。

Journal 文件、PTOBench action 记录、Profile 原始工件和 Pi 对话是运行记录或证据来源，不是可以独立修改的第二套状态。`Recent Memory`、`Historical Memory`、`Opportunity Frontier` 如果需要，都只能是从这一份状态派生出的视图；初版不建立同名独立服务。

Pi 提交的机制解释、收益判断和停止理由保留在记录中，但不能覆盖 PTOBench 的事实。Reducer 只能自动接受可核验事实；对复杂机制的解释必须保留其证据引用和不确定性。

### 4. ContextRenderer：每次请求前重建当前工作单

ContextRenderer 不保存第二份状态，也不调用另一个 LLM。它在 Pi 每次进入模型请求前，从 Controller 当前状态、当前主 workspace、相关知识和证据重新生成有界的状态投影。

确定性的处理顺序是：

1. 从 ProgressPort 读取最新 state revision。
2. 从受控 workspace 读取当前源码，确认它与状态中的 working program 身份一致。
3. 读取当前 Hypothesis、最近完成周期、仍相关的历史事实、剩余预算和必要的知识摘录。
4. 从本次请求消息中删除上一版 `SLUICE_CONTEXT` 标记消息，避免旧快照和新快照同时出现。
5. 裁剪不再相关的旧周期对话，但保留合法的 assistant tool call/tool result 配对。
6. 在末尾加入**唯一一份**最新状态投影，再把消息交给 Pi。

因此，“替换旧状态快照”只指替换本次模型请求将看到的动态投影。Pi 的完整会话记录仍可以保存在 Session 中，历史事实也可以按相关性保留；旧快照不会被当成当前状态重复注入。

ContextRenderer 输出的是事实和工作协议，不替 Pi 选择机会。异常时不能静默提供旧状态继续执行：如果无法生成可信投影，应暂时阻止编辑、evaluate、profile 等动作，提示 Pi 等待下一次成功刷新或结束运行。

### 5. Offline Knowledge：只读的通用机制知识

知识库在在线优化开始前由独立构建流程生成并发布。在线 Pi 只能通过 KnowledgePort 检索已发布卡片和引用；知识库不被当前 Session 写入，也不保存当前 Case 的答案或运行结果。

知识条目可以提供机制、适用条件、预期可观测信号、限制和建议的验证问题。它帮助 Pi 形成 Hypothesis，但不替 Pi 作因果判断，不替 PTOBench 评测，也不决定停止。

ContextRenderer 只注入与当前 Hypothesis/机会相关的短摘录；长卡片按需通过工具读取。这样既保留知识作用，也避免每次请求重复传输整个知识库。

### 6. Profiling Tool 与 EvidenceBundle：整理事实，不作主观结论

PTOBench 的原始 Profile 可能很大。Profiling Tool 根据 Pi 登记的具体问题，对受控工件做确定性的筛选、归一化和组织，返回指标、数值、单位、采集模式、适用程序版本以及 unavailable/缺失信息。

它的任务是降低阅读负担，不是替 Pi 判断“真正瓶颈是什么”或“某优化导致了某现象”。原始工件仍由受控运行目录保存；Agent 默认读取结构化事实，只有配置允许且确有需要时才按授权入口读取原始内容。EvidenceBundle 绑定 Hypothesis、逻辑周期、程序版本、问题和 Profile 引用，避免不同版本的证据串线。

## Pi 与 SLUICE 的工具调用（暂定命名）

文档中的 `sluice_*` 名称表示 Pi 可以调用的 SLUICE 工具，不是 Pi 的内置命令，也不是额外的 Agent 或独立服务。它们都由同一个 Controller 接收，并在同一个 PTOBench Session 中执行。名称之后可以调整，方法的核心不依赖这些具体名字。

| 工具 | 通俗含义 | 主要作用 |
|---|---|---|
| `sluice_step` | “我决定下一步做什么” | 提交 `optimize`、`diagnose`、`recheck` 或 `stop` 决策，推进逻辑周期状态 |
| `sluice_register` | “我准备验证这个想法” | 在修改源码前登记 Hypothesis、修改范围、预期信号和验证问题 |
| `sluice_evaluate` | “请按 PTOBench 规则测一下” | 调用公开评测，取得编译、正确性、clean latency 和预算结果 |
| Profile 工具 | “请给我与这个问题有关的测量事实” | 查询结构化 Profile/EvidenceBundle，必要时读取获准的原始工件 |
| `sluice_finish` | “我确认这次运行结束” | 在已有 `stop` 决策后检查未完成工作，提交最终状态并请求 PTOBench 收尾和 collect |

其中，`sluice_step` 的 `stop` 只是 Pi 提出的停止决定；`sluice_finish` 才是 Controller 执行最终收尾的确认动作。这样可以避免 Pi 只说一句“结束了”就绕过未完成的评测、证据或状态写入。

这组接口不是必须拆成五个名字。若实现时发现 `step` 和 `register` 的分离没有带来足够的检查价值，可以合并为一个带明确阶段字段的决策工具；但“提交决策”“执行受控评测”和“最终收尾”这三类职责仍应保持可区分。

## Context 重建机制

### 每次 Context 事件都做什么

每次模型请求前执行轻量、确定性的刷新：

```text
Pi 即将请求模型
        │
        ▼
context hook 调用 sluice/context
        │
        ▼
从唯一状态和当前 workspace 生成最新 projection
        │
        ▼
移除旧 SLUICE_CONTEXT，裁剪无关旧对话，保持 tool 配对
        │
        ▼
附加一份最新 projection
        │
        ▼
同一个 Pi 发起模型请求
```

这一步不需要调用 LLM，因此可以在每次模型请求前执行。它确保新 Context 真正覆盖旧的“当前状态描述”。Pi 如何保留、整理或限制会话对话历史由 Pi 自行管理，不属于 SLUICE 的状态刷新设计。

## 一次完整运行

### 1. Launcher preflight

读取 Case、Candidate、Pi profile、运行选项、知识包和预算，检查知识包与实际软件环境、版本和硬件条件匹配。PTOBench 创建一个 Session；这一步不启动第二个 Pi，也不把知识构建混入在线运行。

### 2. 准备唯一运行状态

创建一个 Controller、一个 ProgressPort、一个 Evidence/Knowledge 入口和一个 AgentBoundary。记录初始 Candidate、working/best verified 关系和预算。所有后续工具调用都经过这个 Controller。

### 3. 启动一次 Pi

Agent harness 启动唯一 Pi 实例，加载 SLUICE bridge，注册工具并连接当前 Controller。Pi 读取初始 Context projection；从此 Pi 进程持续到 `sluice_finish`、取消、硬性预算保护或错误收尾。

### 4. Pi 提交 Decision

Pi 根据当前工作单和可检索知识选择 `optimize`、`diagnose`、`recheck` 或 `stop`，说明理由和相关机会。Controller 检查引用、状态 revision 和调用顺序，但不代替 Pi 选择方向。

### 5. 登记 Hypothesis（需要修改时）

对于 optimize/recheck，Pi 登记修改前的父程序、变更范围、适用前提、预期信号、验证问题、回退条件和知识引用。登记后 Pi 才修改 Candidate Python 源码。

### 6. 通过 PTOBench 验证

Pi 在同一会话中修改源码并调用 `sluice_evaluate`。Controller 将请求交给 PTOBench 公开 action。编译、correctness、clean latency、预算变化和失败原因都作为客观结果保存；失败、变慢和无提升也不能被省略。

### 7. 按问题取得 Profile 证据

如果 Hypothesis 需要硬件证据，Pi 调用对应 Profile 工具。工具返回结构化测量事实或 unavailable，并把 EvidenceBundle 绑定到当前 Hypothesis、cycle 和程序版本。Profile Tool 不把观测自动翻译成机制结论。

### 8. 提交一个逻辑周期并刷新状态

Controller/ProgressPort 写入 Attempt、周期结果、EvidenceBundle、程序版本和剩余预算。Pi 仍然是同一个进程；它通过下一次模型请求前的 Context hook 获得新 projection。下一周期不是重启 Pi，而是同一 Pi 读取新的工作单后继续。

### 9. 选择继续、诊断、重查或停止

Pi 可以根据最新事实提出下一步，也可以明确提交 stop Decision。Controller 的硬性保护优先于模型意见：公共硬件预算耗尽、未完成动作需要恢复、Session 已关闭或可信 Context 无法生成时，系统进入安全停止/等待路径。正常停止必须先提交 stop Decision，再调用 `sluice_finish`，由 PTOBench 完成最终校验、预算结算和 collect。

### 10. 结束唯一 Pi 和 Session

`sluice_finish` 成功后，Agent harness 关闭同一个 Pi 和工具子进程，PTOBench 收集运行记录。最终输出是最佳合法且已验证的 Candidate、对应事实和运行记录；不是某个 Pi 实例的私有记忆。

最短成功链路是：

```text
一次启动 Pi
  → context projection
  → step
  → register
  → edit
  → evaluate
  → profile（按需）
  → 更新唯一状态
  → 下一次 context projection
  → ...重复若干逻辑周期...
  → stop
  → finish
  → 关闭同一个 Pi
```

## 数据和职责边界

| 内容 | 权威来源 | Pi 看到的形式 | 是否由模型解释 |
|---|---|---|---|
| 当前源码 | 受控 Candidate workspace | 当前文件和程序身份 | Pi 可修改 |
| 正确性、延迟、预算 | PTOBench 公共 action | 结构化结果 | 只能解释，不能改写 |
| Profile 指标 | PTOBench Profile 输出 / Profiling Tool | 针对问题的 EvidenceBundle | 只能提出机制假设 |
| 通用机制知识 | 已发布 Knowledge package | 相关卡片/摘录 | 用于形成 Hypothesis |
| 当前进展 | ProgressPort reducer | Context projection | Pi 提议下一步 |
| 完整对话 | Pi Session | 由 Pi 自行管理 | 不是状态权威 |

## 初版明确不建立的东西

- 多个 Pi 实例之间的接力调度；
- 并行 Pi、竞争 Agent 或多个模型会话；
- 独立的 Recent Memory、Long-term Memory、History、Progress 和 Frontier 数据库；
- 由固定分数替 Pi 预测 speedup 或自动判断因果关系的分析器；
- 把在线 Candidate 运行结果回写到离线知识库；
- 把原始 Profile 全量拼进每次模型请求。

如果真实运行暴露出明确缺口，先在唯一状态、ContextRenderer 或 Controller 的现有接口内修复；只有证明现有接口不足时，才增加新的模块或持久化结构。

## 渐进实现顺序

1. **固定单 Pi 生命周期。** 验证一次 Session 只创建一个 Pi/bridge/controller，多个逻辑周期在同一进程中完成。
2. **固定状态刷新。** 验证每次 model request 都从当前状态生成 projection，旧 `SLUICE_CONTEXT` 不重复堆积，tool call/result 始终配对。
3. **固定状态与源码一致性。** 验证 working、last verified、best verified 和实际 Candidate 在程序切换、失败恢复后仍一致。
4. **接入/复核 Profiling Tool。** 验证按问题返回结构化事实、缺失项和正确的程序/周期引用，不产生主观因果判断。
5. **接入/复核离线知识库。** 验证运行时只读已发布包，按当前问题取相关卡片，不把 Case 结果写回知识库。
6. **真实运行和停止验证。** 用真实 PTOBench Session 检查继续、诊断、无提升、预算耗尽、显式 stop、取消和 collect 路径。

完成这些步骤后，SLUICE 的核心闭环就是：**一个持续 Pi + 一个 Controller + 一个可信状态 + 每请求 Context 重建 + PTOBench 受控验证**。这也是后续代码和论文描述应共同遵守的最小架构。
