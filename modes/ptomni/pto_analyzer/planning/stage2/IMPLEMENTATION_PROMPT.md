# PTOAnalyzer Stage 2 实施 Prompt

> 历史实施范围 Prompt。Stage 2 已执行；当前状态与结果以 [`README.md`](README.md)、[`TASK_CHECKLIST.md`](TASK_CHECKLIST.md) 和离线回放报告为准。

---

你正在 PTOBench 嵌套 Git 仓库的 `modes/ptomni/pto_analyzer/` 中修缮 PTOAnalyzer。

## 目标

将九个 Micro-tool 做成轻量、离线、只读的性能证据整理工具。每个 Tuner 收到的初始摘要完整覆盖自己获准的三个 Tool 和全部可解析数据；数据太多时摘要使用全量聚合，原始明细按需分页。输出客观观测、覆盖情况、来源和解释边界，不替 Tuner 判定瓶颈、根因或优化方向。

## 开始前阅读

1. `modes/ptomni/pto_analyzer/README.md`
2. `modes/ptomni/pto_analyzer/planning/README.md`
3. `modes/ptomni/pto_analyzer/planning/stage1/TASK_CHECKLIST.md`
4. `modes/ptomni/pto_analyzer/planning/stage1/IMPLEMENTATION_PROMPT.md`
5. `modes/ptomni/pto_analyzer/planning/stage2/README.md`
6. `modes/ptomni/pto_analyzer/planning/stage2/TASK_CHECKLIST.md`
7. `modes/ptomni/pto_analyzer/planning/references/ANALYSIS.md` 和参考材料索引
8. 当前 `toolset.py`、`profile.py`、`diagnostics/`、`tests/` 与离线回放数据

## 架构边界

- PTOAnalyzer 只读调用方指定的 `ProfileRef` 和受保护归档；不扫描目录寻找最新 Profile，不写入或复制原始 Profile。
- PTOAnalyzer 不连接 PTOBench CLI、Runner、设备或 NPU，不启动 `profile` / `evaluate`，不维护硬件动作账本。
- Lead 没有 PTOAnalyzer 工具。Compute、Dataflow、Schedule 只访问各自获准的三个工具；不得扩大权限。
- Assembler 不负责工具分发；Analyzer 不替 Tuner、Lead 或 Assembler 作优化决策。
- 不新增 LLM、数据库、通用规则引擎、自动采集器或自动 Case 选择器。
- 本轮代码与测试改动限于 `modes/ptomni/pto_analyzer/`。不得改 PTOBench 采集器或其他 Mode；确认缺采集字段时另写最小需求说明。

## 输出契约

### 1. 用事实表达输入与观测

- 移除现有 Tool 级总括状态字段，用观测、来源覆盖和解释边界表达结果，避免把数据处理状况与性能判断混在一个标签里。
- 用 `source_coverage` 记录每个预期来源族的必需性、找到/读取/空/不可读文件数、读取记录数、解析/拒收记录数及具体问题。对缺失、空文件、格式不识别、损坏记录分别说明实际情况，不填成零。
- 用 `observations` 记录原始值及明确公式得出的统计量。每项包含单位、作用范围、聚合口径、样本数和来源引用；不同单位、窗口或身份不合并。
- 用 `limitations` 说明字段契约的解释边界和缺少的证据。不要输出根因置信度、严重度、评级、未经校准的阈值、建议优化动作或预期加速比。
- 没有可用观测时，返回空观测集合并保留来源覆盖与具体问题，让调用者区分数据中没有该类记录与输入不可用。
- 所有跨文件统计都要求 Profile/Candidate 身份、设备、单位、时间窗口和关联键已核实；否则分别陈述事实并列出关联缺口，不作 join。

### 2. 完整但精简的 Tuner 摘要

- 将 `ToolSession.initial_findings` 改为 `initial_summary`。实施前复核调用方；目前 PTOAnalyzer 内部 README、会话对象和测试有引用，未发现其他仓库内消费者，需同步迁移。
- 每个 Tuner 的摘要固定包含其三个获准 Tool，每项都说明所处理的数据、来源覆盖、全量统计和解释限制；即使某一来源缺失，也保留 Tool 项并陈述缺失事实。
- “完整”表示所有获准工具、来源和可解析记录均被纳入总体统计，不表示将所有原始行灌入上下文。不能用 Top-K、第一页、几个样例或字符截断代替全量摘要。
- 类别频数或分组统计须覆盖全部类别。极高基数的原始记录使用稳定排序分页查阅；摘要仍基于全量数据计算，并给出全局记录数与聚合口径。
- 仅高基数明细支持分页；默认 20 条、上限 50 条，结果包含总数、已返回数和下一页位置。分页不能改变摘要和统计的全量口径。

建议的精简结果形态：

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
      "aggregation": "all parsed records; grouped by build/rank/dispatch/core type",
      "sample_count": 1200,
      "source_refs": ["profile artifact references"]
    }
  ],
  "limitations": ["缺少经校准的运行时间分母，不能将 cycles 解释为利用率。"],
  "detail_access": {"total_records": 1200, "page_size_default": 20, "page_size_max": 50}
}
```

示例仅说明字段形态；真实工具只返回其必要字段，不为接口对称而制造空分页或无意义状态。

## 九个 Tool 的实施要求

逐项依据和参考位置见 `planning/references/ANALYSIS.md`。据此实施：

| Tool | 全量摘要与关键约束 |
|---|---|
| `compute_mapping_check` | 汇总 program/callable/config 映射与关联覆盖；只有 Candidate 身份链核实后才显示源码位置。 |
| `compute_unit_activity` | 按 build/rank/dispatch/core type 汇总 PMU counter 和 per-core 分布；保留原单位，不移植外部利用率公式或评级。 |
| `compute_task_share` | 只在同 dispatch 身份 join 验证后按 task 汇总；cycles 和时间跨度分别呈现；列出未关联量和原因。 |
| `tile_access_check` | 汇总全部 compiler hint 类别、位置、原文和来源；提示保持编译器观测，不改写成优化处方。 |
| `buffer_pressure_check` | 分开呈现 MemoryReuse hints 和 Scope 原生字段；无容量/生命周期契约时不计算压力百分比。 |
| `pipeline_overlap_check` | 只有同 Candidate、设备、时间基准和关联 task/block 的 Compute/MTE 区间可支持 overlap 统计；否则评估并暂缓无独立证据价值的接口。 |
| `dependency_path_check` | 规范化并汇总节点、边和图结构；无运行时权重时不称为关键路径或延迟。 |
| `dispatch_gap_check` | 仅用同 rank/device/core 且时基可信、覆盖充分的区间计算相邻空档；称“观测空档”，不称调度/派发开销。 |
| `task_distribution_check` | 汇总 rank/core 记录数及区间分布；区分 tuple 行、唯一 task 和忙碌时间，不推断 occupancy 或负载失衡。 |

所有高基数明细应分页；全量摘要必须基于所有可解析记录，不能只汇报排序靠前的项。

## 参考风格的边界

工具按诊断问题组织证据：每个 Micro-tool 整理一个具体问题相关的性能事实、来源和范围，再由对应 Tuner 跨工具综合判断。这一工具边界参考 KernelPro 的 Micro-profiling 思路。可参考 PyPTO-Agent 的证据关联层次，以及 CANNBot/PyPTO-Gym 的解析、分组和时间线整理方式；不复制它们的性能评级、阈值、因果诊断或行动计划。当前保存材料没有 KernelPro 的实现或结果契约，不据此假定它采用特定状态字段。

## 实施顺序

1. 复核调用方、现有数据和九个工具实现，锁定单卡/双卡/8 卡的归档样本。
2. 统一 `initial_summary` 与每个 Tool 的事实输出契约，移除混合诊断状态语义。
3. 按上表逐项实现全量聚合、单位/范围/来源、解释边界和必要分页。
4. 审计 gap/overlap 的字段、身份、窗口和时间语义；证据不够时收窄或暂停对应能力。
5. 用现有 Profile 离线回放；用 CPU-only Fixture 检查大数据聚合、分页、来源引用、异常输入与只读性。
6. 只有离线审计发现具体数据缺口时，再形成最小 Case 运行方案；Basic Pi 轨迹只用于优化前后证据变化或工具效用验证。

任何新增 `profile` / `evaluate` 都必须先形成可审查的 Case 计划，再由 PTOmni Runner/PTOBench 正常执行并遵守设备锁、预算和硬件动作计数。Analyzer 不得自行运行它们。本 Prompt 本身不是新增硬件运行的授权。

## 完成标准

- 每位 Tuner 的 `initial_summary` 完整列出其三个工具，并基于全部可解析数据生成全量统计；不静默截断或省略类别。
- 高基数明细能稳定分页；分页遍历无重无漏，合计与全量数据一致。
- 每个统计可追溯到来源，单位、作用范围、聚合口径和样本数明确；缺失不填零。
- 不输出性能判决状态、根因、评级或优化处方；数据缺陷用覆盖计数和具体问题表述。
- gap/overlap 只使用已核实的时间字段、身份与覆盖；否则明确列出无法计算所缺的信息。
- CPU-only 测试和真实归档离线回放可复核；未改采集器、原始归档、角色权限或硬件动作账本。
