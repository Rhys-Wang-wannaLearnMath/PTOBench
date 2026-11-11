# PTOAnalyzer Stage 2：证据摘要与九个 Tool 修缮计划

> Stage 2 已完成实现与离线验收。本文保留原设计和验收依据；执行结果见 [Stage 2 任务清单](TASK_CHECKLIST.md)、[回放报告](../../reports/offline_replay_stage2_20261108.json) 和 [PTOAnalyzer README](../../README.md)。本次没有运行 PTOBench、Agent 或真机实验。

## 目标与边界

Stage 2 完成了两件事：修缮 PTOAnalyzer 的输出与时间分析，并把 [九个 Tool 的参考分析](../references/ANALYSIS.md) 落到各 Tool 的实现和边界中。

PTOAnalyzer 继续保持轻量、离线、只读、确定性。它的工作是把 Profile 中的原始性能信息整理成**可追溯的客观观测和统计摘要**，不是替 Tuner 判断根因、评估优化方向或选择代码改动。Tuner 结合自己的知识和实验决定下一步；Lead 不调用 Analyzer，也不启动实验。

Stage 2 不新增 LLM、数据库、通用规则引擎、自动采集器或新的硬件动作，不改 PTOBench 采集器。若真实数据缺少必要字段，只记录最小采集需求，不用换 Case 重跑来掩盖字段缺失。

## 输出原则：报告观测，不给性能判决

每个 Tool 的结果只包含三类内容：

1. **观测值**：原始计数、提示、图结构、区间，以及由明确公式直接计算出的统计量。
2. **数据覆盖**：所需来源、实际读到的文件和记录数、参与汇总与未能解析的数据量。
3. **解释边界**：这些数据能说明什么、还不能说明什么，以及要得到更完整观测缺少哪些输入。

Stage 2 已移除 Tool 级诊断标签。结果不使用 `finding`、`no_signal` 或类似状态；缺少文件、空文件、无法读取及坏记录等情况由来源覆盖和解析问题表达。不能解析的部分不填 0，也不丢弃同一来源中的有效记录。`observations: []` 与来源覆盖一起返回，帮助 Tuner 区分没有此类记录与没有读到数据。

PTOAnalyzer 不输出“瓶颈已确认”、根因置信度、严重度、性能评级、统一阈值、建议优化动作或预期加速比。若提供派生统计，必须给出计算方式、范围、单位、样本数和来源引用。跨文件计算只有在身份、单位、采集窗口和关联关系均已验证时才进行。

例如，Tile 编译提示是编译器写出的事实，不等于运行时瓶颈；PMU busy cycles 是计数，不等于利用率；DAG 深度是无权图结构，不等于关键路径时间；同核相邻区间之间的时间差是观测空档，不等于调度开销。

## Tuner 初始摘要：完整覆盖，不截断

`ToolSession.initial_summary` 只返回当前 Tuner 有权访问的三个 Tool，每个 Tool 固定有一份摘要；即使数据缺失或不可解析，也保留该 Tool 的摘要项并说明覆盖问题。不得把另两个 Tuner 的工具内容混入结果。

这里的“完整”指：**所有获准 Tool 都有交代；每个数据源和全部可解析记录都进入覆盖统计；摘要中的统计量由整批数据计算。** 它不要求把数千条原始记录再复制到初始上下文中。单独列几个 Top-K 行或样例不算完整摘要；如提供样例，必须标明样例数和总数。

每个 Tool 摘要使用同一精简结构：

```json
{
  "tool": "compute_unit_activity",
  "source_coverage": [
    {
      "source_family": "pmu",
      "required": true,
      "files_found": 8,
      "files_read": 8,
      "files_empty": 0,
      "files_unreadable": 0,
      "records_seen": 1200,
      "records_parsed": 1200,
      "records_rejected": 0,
      "issues": []
    }
  ],
  "observations": [
    {
      "name": "cube_busy_cycles",
      "value": 123456,
      "unit": "cycles",
      "scope": "build/rank/dispatch/core_type",
      "aggregation": "all parsed records grouped by build/rank/dispatch/core type; per-core distribution summarized",
      "sample_count": 1200,
      "source_refs": ["profile artifact references"]
    }
  ],
  "limitations": ["计数没有运行时间分母，不能解释为利用率。"],
  "details": {
    "total_records": 1200,
    "page_size_default": 20,
    "page_size_max": 50
  }
}
```

字段要求：

- `source_coverage` 对每个预期来源族列出是否必需、找到和读取的文件数、空/不可读文件数、已解析/拒收记录数及具体问题；即使文件数为 0，也保留来源项。它描述数据处理过程，不是性能判断；无需再用总括状态枚举表达这些情况。
- `observations` 汇总该 Tool 的**全部有效数据**。每个值保留单位、聚合口径、作用范围、样本数和来源；不能把不同单位或范围的值合并。
- `limitations` 只写数据契约支持的边界，不写根因猜测或优化处方。
- `details` 说明原始记录总数和可用分页能力。摘要不得按字符数或行数静默截断。

原始明细通过同一 Tool 的按需分页读取：固定排序后使用 `offset` 与 `limit`，默认每页 20 条、最多 50 条，并返回总数、已返回数和下一页位置。九个摘要覆盖所有数据；分页用于查阅具体记录，不替代摘要。只有高基数 Tool 才开放明细分页，其他 Tool 不为接口形式对称而增加查询参数。初始摘要精简但不截断统计，不能把第一页或 Top-K 当作全量结果。

已搜索 PTOBench 中的 `initial_findings`；只发现 PTOAnalyzer 自身 README、工具会话和测试的引用，现已统一为 `initial_summary` 并同步更新文档与测试。

## 九个 Tool：结合参考后的具体规划

下面的摘要都以**全量有效记录**计算；高基数原始记录按需分页。逐项依据、参考文件和边界详见 [ANALYSIS.md](../references/ANALYSIS.md)。

| Tuner / Tool | 初始摘要覆盖内容 | 可借鉴的实现与 Stage 2 修缮 | 必须保留的边界 |
|---|---|---|---|
| **Compute — `compute_mapping_check`** | program、callable、kernel config 等映射类别与记录数；已验证、未验证及无法匹配的关联数量。 | 借鉴 PyPTO-Agent 的分阶段关联，以及 PyPTO-Gym `leafhash_to_code.py` 的映射整理；保留映射来源和 Candidate 身份链。全量映射细节可分页。 | callable/name map 不等于 Candidate 源码位置。只有映射链与当前 Candidate 快照核实时才展示代码位置；否则报告未验证关联数和来源。 |
| **Compute — `compute_unit_activity`** | 每种 PMU counter 的全量计数汇总，按 build/rank/dispatch/core type 分组，并给出 core 数、记录数和 per-core 分布摘要。 | 借鉴 CANNBot 按 AIC/AIV 和 core 分组、保留单位及稳定排序的做法。高基数 per-core 记录分页。 | 计数按原单位保留；不套用外部 utilization、bubble 公式或评级阈值。没有校准分母时不输出百分比或“利用率”。 |
| **Compute — `compute_task_share`** | 全量 task 关联成功/失败数量及失败原因分布；分别汇总已关联 task 的 PMU counters 与 runtime span 分布。 | 借鉴 PyPTO-Agent 的 task/leaf 分组和紧凑报告；摘要完整汇总全部类别，task 明细按需分页。关联仅使用同一 dispatch 内验证过的身份。 | PMU cycles 和 AICore 时间跨度量纲不同，分别呈现、不相除。swimlane 行数不自动等于唯一 task 数；未关联项要计数并说明原因。 |
| **Dataflow — `tile_access_check`** | 全部 compiler hint 类别的计数、涉及的源码/IR 位置数及来源覆盖。 | 借鉴 CANNBot Skill 按问题整理提示与下一步观察的形式；保留原始提示文本、位置、类别和来源，必要时分页查看提示明细。 | 提示是编译器观测，不是运行时瓶颈判决；不自动把提示改写成增大/减小 Tile 的建议。 |
| **Dataflow — `buffer_pressure_check`** | MemoryReuse hint 各类别和数量；Scope 每个原生字段按原单位给出全量记录统计、范围与样本数。 | 借鉴 PyPTO-Agent 把编译事实与运行时事件分层，并为每项数值保留来源。Scope 明细按需分页。 | 未验证容量、分配和生命周期语义时，不计算“压力百分比”，不由单项 hint 推断 buffer 溢出。 |
| **Dataflow — `pipeline_overlap_check`** | 先完整汇报可用运行时阶段区间的类别、窗口、task/core 覆盖和缺失字段；只有契约验证通过后才汇总重叠区间及覆盖量。 | 借鉴 PyPTO-Agent 与 CANNBot 先归一化事件、再按身份和时间范围计算的做法。必须审计该 Tool 是否提供独立于 Buffer Tool 的信息。 | 编译 pass 时间、PMU 汇总计数和普通 task 总区间不能证明 compute/MTE overlap。没有同一 Candidate、设备、时基和可关联阶段区间时，不输出重叠值；若没有独立证据用途，暂缓该 Tool 的对外暴露。 |
| **Schedule — `dependency_path_check`** | 全量图的节点/边数、重复边和不完整端点数、连通结构及已验证的无权深度；有路径明细时分页。 | 借鉴 CANNBot `analyze_aiv_dep_chains.py` 的图规范化、去重和结构化输出，以及 PyPTO-Agent 对 task 关联的组织。 | 无 runtime 权重时只描述图结构；不称为运行时关键路径或延迟瓶颈。 |
| **Schedule — `dispatch_gap_check`** | 报告输入文件/记录覆盖；仅在同 rank/device/core 且时基可信时，汇总相邻区间的观测空档数和全量分布。 | 借鉴 PyPTO-Agent `compactness.py` 先计算可观察区间、再尝试归因的分层；按 artifact index 的实际文件族匹配 merged swimlane，接受已观察到的时间戳后缀。明细分页。 | 未验证 tuple 字段、覆盖范围、同核时间语义和时钟锚点时不计算。观测空档不命名为调度/派发开销；跨 rank 只有时钟归一化经过验证才可比较。 |
| **Schedule — `task_distribution_check`** | 全量 rank/core 的记录数与持续时间分布摘要，并报告唯一 task 身份是否可验证。 | 借鉴 CANNBot `analyze_core_usage.py` 和 PyPTO-Agent 的 per-core 汇总；稳定排序并按需分页。 | swimlane tuple 数不等于唯一执行次数，区间跨度不等于忙碌时间。没有设备 core 基线时不输出 occupancy 或 load-imbalance 结论。 |

## 工具证据组织与参考边界

工具按诊断问题组织证据：每个 Micro-tool 整理一个具体问题相关的性能事实、来源和范围，由对应 Tuner 综合多个工具的结果。PTOAnalyzer 因此完整汇总原始性能信息，不替 Tuner 写“发现/未发现瓶颈”式结论。这种工具边界参考 KernelPro 的 Micro-profiling 思路；当前保存的参考快照没有 KernelPro 源码或输出契约，所以不假定它采用某种状态字段或 schema。

当前可核对的参考实现各有不同取舍：

- CANNBot `perf-analyzer` 会进一步生成评级、瓶颈和优化建议；PTOAnalyzer 只借鉴它的数据解析、分组和报告组织，不移植它的阈值或结论，也不照搬其评级式表达。
- PyPTO-Agent 会把症状逐层关联到 task、源码和编译器行为，并组织 DiagnosisRecord/ActionPlan；PTOAnalyzer 借鉴证据关联和 gap 量化方式，不复制其完整诊断 Agent 或因果判断。
- PyPTO-Gym 的 swimlane 工具提供 task/core 归一化与依赖整理样例；其字段和硬件含义仍须按 PTOBench 的实际数据验证。

因此，Analyzer 的默认返回是“观测摘要 + 覆盖情况 + 限制”；Tuner 才提出可检验假设，PTOBench `evaluate` 才验证优化候选的正确性和 Clean Latency。

## 时间类 Tool 的实施门槛

### `dispatch_gap_check`

先从已有归档审计 raw Level-4 swimlane、merged swimlane 和依赖图：确认真实文件族、schema、单位、rank/device、Task/Block 身份、时钟锚点和 raw/merged 覆盖关系。未知 tuple 字段不能按数组位置猜测。仅对同一 rank/device/core、同一可信时基及覆盖充分的区间排序并计算相邻区间时间差；报告区间、统计公式、来源和未覆盖部分。没有 scheduler/orchestrator 阶段的直接时间证据时，不推断“调度延迟”或“派发开销”。

如果现有数据不支持可靠的区间计算，只报告数据覆盖和无法计算的具体原因；若该工具不能提供有别于文件清单的有效信息，就收窄或暂缓它。增加 Case 不能补出采集器没有输出的时间语义。

### `pipeline_overlap_check`

审计是否存在同一 Candidate、同一设备、可对齐时间基准和可关联 task/block 下的运行时计算与搬运阶段起止区间。只有这种区间才能支持重叠计算；编译 pass 耗时、汇总 PMU 计数、普通 Task 总起止时间都不够。若有完整性不一的区间，分别汇报可覆盖窗口和未覆盖范围，不外推整体 pipeline 效率。

若缺少阶段区间，先确定该 Tool 是否仍能提供与 `buffer_pressure_check` 不重复的 pipeline 事实；没有独立价值则从 Tuner 当前工具集暂缓，不为了维持“每个 Tuner 三个工具”而保留空壳。采集字段若确实缺失，形成 PTOBench 后续需求，不在本阶段扩展采集器。

## 实施与验证顺序

1. **盘点现有调用方与样本。** 阅读 Stage 1 归档回放、toolset/profile/diagnostics/tests，复核 `initial_findings` 消费点；复用已有 16 个 combined Profile，至少覆盖单卡、双卡和 8 卡样本。
2. **固定结果契约。** 引入完整覆盖的 `initial_summary`；用来源覆盖、全量聚合的 `observations`、解释边界和按需 `details` 替代诊断级总状态与“finding”语义。
3. **按九项矩阵修缮 Tool。** 每项都核对参考建议，做到全量计数、单位与范围明确、来源可追溯、跨源 join 有验证；不适用的诊断能力如实收窄或暂缓。
4. **只给高基数明细加分页。** 先做确定排序和完整的总数/分页协议，默认 20、上限 50；确保每个摘要由全量记录计算，而不是分页第一页或 Top-K。
5. **审计并处理两个时间 Tool。** 严格按前述数据门槛决定是否计算；不满足门槛则返回覆盖事实和具体限制，不产出推断性状态或结论。
6. **离线验证。** 用真实归档人工抽查摘要值与原始数据；添加 schema 与字段一致、记录量大的 CPU-only Fixture 验证摘要全量性、分页、排序和来源引用。合成 Fixture 只验证格式和边界，不作为性能证据。
7. **数据缺口再决定是否运行。** 若真实 schema/设备形态在现有 16 个归档中缺失，先写明新增 Profile 能补充什么及判读方法，再提出最小 Formal/Held-out combined Profile。只有要检验优化前后变化或 Analyzer 对 Basic Pi 方向选择的作用时，才考虑 Basic Pi 轨迹；必须绑定 Candidate hash/diff、Profile event 和 evaluate 结果。

任何新增 Profile/evaluate 均由 PTOmni Runner/PTOBench 正常路径执行，遵循设备锁、预算和硬件动作账本；Analyzer 不调用 CLI、不自行选 Case。离线解析和 CPU-only 测试不计硬件动作。本轮没有授权或执行这些运行。

## 完成标准

- 每个 Tuner 的 `initial_summary` 包含其全部三个获准 Tool；没有 Tool 被静默省略。
- 每个 Tool 摘要基于全部可解析记录及必需来源，完整汇总全部类别；不以 Top-K 或样例替代总体统计。不能合理聚合的明细可分页完整查阅。
- 每个数值都有单位、聚合口径、作用范围、样本数及来源引用；缺失不填零。
- 工具结果不含诊断级状态枚举、性能根因判决、严重度、评级、未校准阈值或优化处方；输入覆盖、解析问题、限制与待补证据以事实表达。
- 大型明细可稳定分页，逐页无重复/遗漏，所有合计与全量数据一致。
- dispatch gap 与 pipeline overlap 只在字段身份、时间语义和覆盖经过核实时计算；否则收窄或暂缓。
- 真实归档离线回放与人工抽查相符；CPU-only 测试覆盖缺失、空、坏格式、部分记录、未知字段、高基数、分页和只读边界。
- 未为没有明确数据缺口的问题新增硬件运行，也未修改 PTOBench 原始归档、角色权限或采集器。

## 参考

- [Stage 1 规划与回放](../stage1/TASK_CHECKLIST.md)
- [PTOAnalyzer 九个 Tool 的参考实现分析](../references/ANALYSIS.md)
- [固定版本参考文件索引与许可说明](../references/README.md)
- [PTOAnalyzer 当前接口与角色边界](../../README.md)
