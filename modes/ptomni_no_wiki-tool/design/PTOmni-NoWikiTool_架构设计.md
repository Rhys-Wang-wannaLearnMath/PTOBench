# PTOmni-NoWikiTool 架构设计

> 状态：设计稿，供 `ptomni_no_wiki-tool` 首个可执行版本实施使用；本文件不代表该 mode 已实现。
> 依据：当前 PTOmni 主架构正文版本 2026-11-08（文件名仍为 `PTOmni_架构设计_2026-11-03.md`）、PTOBench 方法入口协议及 Agent CLI。

## 1. 目标与定位

`ptomni_no_wiki-tool` 是 PTOmni 动态轻量 Agent Team 的消融实现。它保留团队规划、候选探索、受管评测、状态更新和 Case Memory；只关闭 PTOmni 管理的离线 Wiki 检索与 PTOAnalyzer 结构化工具。

本设计将 `profile` 保留为和正式 PTOmni 相同的 PTOBench 操作。Tuner 可以在 RoundPlan 许可和预算范围内请求 Profile，并检查 PTOBench 在注册 workspace 中公开的原始诊断产物；它不能调用 PTOAnalyzer 的 `ToolSession` 或任何 Analyzer Tool，也不能绕过 PTOBench 直接读取受保护归档。这样比较的是 Wiki 与结构化 Profile 整理的合并增量，不是“有无 Profile”。

这不是架构主文档中的完整 `PTOmni-NoGuidance`：完整 NoGuidance 还会移除 Dynamic Case Memory。本消融保留 Case Memory，避免把跨轮经验蒸馏也混入本次处理变量。

## 2. 范围

| 保留，与正式 PTOmni 相同 | 本 mode 中关闭 |
|---|---|
| 动态轻量团队，由 Lead 为每轮选择必要的 Tuner；角色按任务执行，不要求长期驻留进程 | PTO Wiki 快照、检索入口和注入给 Agent 的 Wiki 片段 |
| Lead、Assembler、Tuner、按需 Challenger、Curator 的职责和权限 | PTOAnalyzer `open_toolset` / `ToolSession` 与九个只读工具 |
| Compute、Dataflow、Schedule 专长标签；只启用与本轮问题相关的 Tuner | PTOAnalyzer 汇总结果、来源覆盖和分析摘要进入 RoundContext 的通道 |
| 独立 Tuner workspace；多个候选并行与否由资源和 PTOBench 执行域决定 | 对 PTOBench 受保护 Profile 归档的 Agent 直接路径访问 |
| Runner 作为 `profile` / `evaluate` 的唯一方法侧执行入口；沿用同一 Case、环境、预算与设备约束 | 额外的自研 Profile 解析器或另一套硬件执行/记账链 |
| PTOBench 的 correctness、Clean Benchmark、Candidate 身份和事件作为事实来源 | 对 PTOBench 指标自动给出根因、性能严重度或加速预估的确定性规则 |
| Event Ledger、派生 State、ExperimentResult 和 Curator 维护的 Case Memory | 向 Curator 或 Case Memory 写入全局知识库 |

基础任务说明、Case Contract、Candidate 源码与项目原有可读文件仍按公共协议提供。这些不是 PTOmni 的 PTO Wiki；两种 mode 应保持相同。`pto_wiki/` 不应作为本 mode 的资源复制、索引、检索源或提示附件。

## 3. 核心流程

```text
Case / Baseline / Case Memory / PTOBench 事实
  -> Lead 形成一个 RoundPlan（选择问题、Tuner、权限和预算）
  -> 被选中的 Tuner 在隔离 workspace 探索
       -> 按计划可请求 PTOBench profile，读取 Agent 可见原始诊断产物
       -> 请求 PTOBench evaluate，获得正确性与 Clean Benchmark 事实
  -> Tuner 提交 ExperimentResult（事实引用与解释分开）
  -> 按需由 Challenger 只读质疑假设或证据
  -> Assembler 选候选、组合兼容改动或保留父版本
  -> PTOBench 验证最终候选
  -> 确定性 State 更新，Curator 更新本 Case Memory
  -> Lead 根据新状态继续、重规划或停止
```

Baseline 不算优化 Round。单次 Profile 是 PTOBench 的一次硬件动作，不提供 correctness 或 Clean Latency；正式正确性和性能结论来自 `evaluate`。静态读取 Wiki（本 mode 不启用）、Case Memory 和历史事件不计硬件动作。PTOmni 不自行维护第二本硬件预算账。

## 4. 角色行为

### Lead

Lead 只消费状态层整理出的 RoundContext，不持有 Runner、PTOBench 或 PTOAnalyzer Tool。Context 包含当前 Candidate 身份、PTOBench 结果和事件引用、Case Memory、剩余预算及团队状态；没有 Wiki 片段，也没有 PTOAnalyzer observations。Lead 输出结构化 RoundPlan，指定一个连贯的决策问题、可证伪假设、可改范围、启用的 Tuner、Profile/evaluate 权限与额度，以及是否启用 Challenger。

当已有 Profile 时，Lead 可看到其存在状态、Candidate/Event 引用和由 PTOBench 公开返回的执行状态；本 mode 不为 Lead 生成结构化诊断摘要。若 Tuner 需要解读 Profile，应由它在自身任务中通过公开 workspace 产物检查，而不是让 State 或 Lead 背地里调用 Analyzer。

### Tuner

Lead 每轮只选择必要专长，默认由一个 Tuner 开始。Tuner 只能写入自己的隔离 workspace，且只能通过 Runner 请求 PTOBench 操作。它可以自行检查当前注册 workspace 中 PTOBench Agent CLI 可见的 Profile 诊断文件；这类读取不产生新的硬件动作。文件缺失、不可读、来源不明或与当前 Candidate 不匹配时，Tuner 必须把证据标为 unavailable / incomparable，不能补造数字。

Tuner 不得导入、启动或以复制实现替代 PTOAnalyzer；不得直接访问 `session_root` 下的受保护 Profile archive、修改 PTOBench 记录，或从别的 Candidate 借用 Profile。手工解读原始数据的结论属于 Agent interpretation，必须和 PTOBench 原始事实及其 Event / Candidate 引用分开。

### Assembler、Challenger 与 Curator

- **Assembler** 比较各 Tuner 的 correctness、Clean Benchmark、Profile Event 引用和解释，决定接受、组合或拒绝候选。组合后的新 Candidate 必须重新验证。
- **Challenger** 仅在证据冲突、风险较高或搜索停滞等情况下短时启用；只读审查，不写代码、不运行 PTOBench，也没有否决权。
- **Curator** 在最终候选验证后总结事实、受限解释、失败条件和未解问题，并更新当前 Case Memory。它不改事实，不写入 PTO Wiki。

这五个角色是按职责创建的短时任务，不是五个常驻 Agent。第一版最小闭环通常只需 Lead、一个 Tuner、Assembler 和收尾 Curator；Challenger 与其他 Tuner 由 RoundPlan 按需启用。

## 5. 数据与权限接口

首版不引入独立 Agent 框架、通用插件注册中心或复杂 Guidance 抽象。将团队循环与实验信息来源分开即可：团队只依赖稳定的 RoundContext、RoundPlan、ExperimentResult 和 PTOBench Event / Candidate 引用；本 mode 的信息构造不装配 Wiki 或 Analyzer 结果。后续正式 PTOmni 在同一信息构造位置增加这两种只读信息来源。

`ExperimentResult` 沿用 PTOmni 主设计中的字段：候选及父候选身份、主要/次要机制、假设、改动范围、correctness、Clean Benchmark、Profile 状态与证据引用、解释、硬件动作/时间/Token 成本、失败变体和下一问题。Profile 字段表示“由 PTOBench 采集的证据”，不表示“由 PTOAnalyzer 分析的结果”。

Profile 使用约束：

1. `profile` 的是否可用、`combined` / `split` 策略及每次预算，与对照 PTOmni 保持一致；不得为消融偷偷禁用 Profile 或增加 Profile 次数。
2. Runner 只把受管 PTOBench CLI 暴露给当前 RoundPlan 授权的 Tuner；Lead 不提交执行请求。
3. Tuner 通过 PTOBench 返回的 Event 结果和当前 workspace 内公开诊断产物工作。不得从受保护 Archive 猜路径或扫描“最新 Profile”。
4. Profile 总是绑定准确的 Event、Candidate 和环境身份。它是诊断证据，不能替代 correctness 或 Clean Benchmark。
5. Agent 对原始数据的整理或推断作为 interpretation 留档，不伪装成 PTOBench 事实或 Analyzer 输出。

实验记录应能复现一次运行用了哪些 TeamPlan、Candidate、Profile / evaluate Event、Case Memory 版本、方法文件与提示资源版本，以及实际向 Agent 暴露的资料。知识暴露清单应明确为空；Analyzer 工具暴露清单应明确为空；Profile Event 和 PTOBench 原始诊断产物的实际暴露仍如实记录。

## 6. 与正式 PTOmni 的公平比较

核心比较应固定模型与 Agent backend、PTOBench Case / 初始 Candidate、团队规划逻辑、角色提示（除关闭 Wiki / Analyzer 的必要说明外）、Runner 行为、Profile 策略和次数上限、evaluate 配置、硬件预算、Case Memory 规则以及停止条件。

| 维度 | `ptomni_no_wiki-tool` | 正式 `ptomni` |
|---|---|---|
| Offline Wiki | 不装配、不检索、不向 Agent 提供 | 使用冻结版本的 Wiki，检索结果带来源和适用范围 |
| 原始 Profile | Tuner 可读 PTOBench Agent CLI 暴露的诊断产物 | 同样保留，便于实验条件一致 |
| PTOAnalyzer | 不可调用；Lead 无摘要，Tuner 手工检查其可见的原始产物 | 仅已启用 Tuner 可按专长调用只读 Analyzer Tools |
| Case Memory | 每轮 Curator 更新 | 同一规则、同一作用范围 |
| 动态团队和 PTOBench | 完全相同 | 完全相同 |

本消融将 Wiki 与结构化 Profile 工具作为一个联合因素关闭。若需要分别声称 Wiki 或 PTOAnalyzer 的独立贡献，需追加仅开 Wiki 或仅开 Analyzer 的匹配组，不能从这一对结果拆分归因。

## 7. 实施顺序与验收

先构建这个 mode 的端到端最小版本，尽早验证方法入口和 PTOBench 闭环；不要先把团队、状态、Runner、Wiki、Analyzer 各自完整实现后再做首次集成。首轮范围建议按以下顺序收敛：

1. **执行骨架**：遵循 `modes/PROTOCOL.md` 实现 `preflight / prepare / run / collect`；确认 `mode.yaml` 的 ID 为 `ptomni_no_wiki-tool`，Runner 和调用进程符合 PTOBench 生命周期及进程组规则。
2. **最小 Team Loop**：单 Case 建立 Baseline、RoundContext、Lead RoundPlan、一个 Tuner、ExperimentResult 和 Assembler 决策；各角色可由一个 Pi Session Runner 以不同任务提示执行，不强制开多个长期进程。
3. **真实评测闭环**：通过 Runner 使用公共 `ptobench evaluate/profile` CLI，记录 Candidate、Event、成本和结果；不自行实现评测、设备锁或硬件预算。
4. **状态与记忆闭环**：用 PTOBench Event 作为事实来源，确定性更新必要的 Candidate / Best / Budget 状态；Curator 只维护短小的 Case Memory。
5. **配对验收**：用相同 Profile 策略确认 Profile 权限、原始诊断文件暴露范围和成本；检查任何角色均无法调用 PTOAnalyzer 或访问受保护归档。

实现完成的验收条件：

- 一个 Case 能从 Baseline 走过至少一轮 Tuner 修改、PTOBench 正确性/性能反馈、Assembler 决策、状态更新与 Case Memory 收尾。
- Lead 的 RoundPlan 能动态启用必要角色，团队规模不被写死为所有角色同时启动。
- Agent 无 Wiki 资源、索引或检索入口；运行记录可验证其未暴露。
- Agent 无 PTOAnalyzer `ToolSession` / 工具；运行记录可验证工具暴露为空。
- Profile 和 evaluate 均由 PTOBench 计数、归档和判定；无 Analyzer 时 Profile 仍保持主设计允许的受控采集路径。
- Candidate 写入权、候选身份绑定、Runner 执行约束和清洁评测规则满足主设计与协议。

## 8. 非目标

- 不实现完整 NoGuidance；不删除 Case Memory、基础任务说明、项目源码或 PTOBench 反馈。
- 不关闭 `profile`，也不把它的读取路径伪装成 Analyzer。
- 不实现第二套 Profile parser、Profiler、Tool Server、预算账或设备调度器。
- 不固定整个研究阶段的团队配置；固定团队属于独立 FixedTeam 对照。
- 本设计文档不创建 `mode.yaml`、runner 或 Agent 实现；这里只冻结后续实现的行为和比较约束。

## 9. 依据

- [PTOmni 当前架构设计](../../PTOmni/design_docs/PTOmni_架构设计_2026-11-03.md)
- [PTOBench 独立方法入口协议](../../PROTOCOL.md)
- [PTOBench Agent CLI](../../../evaluator/AGENT_CLI.md)
- [PTOAnalyzer 当前接口与状态](../../ptomni/pto_analyzer/README.md)
