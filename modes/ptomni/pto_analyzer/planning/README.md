# PTOAnalyzer Tool 与样本规划

> Stage 1/2 的九项诊断是历史实现目标。Stage 4 后的当前接口有九个只读 Tool（Compute 3、Dataflow 3、Schedule 3）；`pipeline_overlap_check` 仍暂缓，新增 `tile_memory_allocation_summary`。Stage 4 的实测证据、分页修缮和离线验收见 [Stage 4 报告](../reports/offline_replay_stage4_20261112.json)。

可直接交给后续实现任务的材料：

- [Stage 5 实施 Prompt](stage5/IMPLEMENTATION_PROMPT.md)：限定范围、权限边界、实施顺序和交付物。
- [Stage 5 任务清单](stage5/TASK_CHECKLIST.md)：按阶段推进并在每个阶段设验收门槛。

Stage 1 已完成初版实现与离线回放。Stage 2 修缮与离线验收也已完成；设计依据、执行清单和回放结果见 [Stage 2 总览](stage2/README.md)、[Stage 2 任务清单](stage2/TASK_CHECKLIST.md) 和 [Stage 2 回放报告](../reports/offline_replay_stage2_20261108.json)。原 [Stage 2 实施 Prompt](stage2/IMPLEMENTATION_PROMPT.md) 保留作历史实施范围记录。

Stage 3 已完成工具收敛与 Cursor 分页改造；Stage 4 已完成大文件解析修缮和静态分配摘要：当前对外为九个只读 Tool（Compute 三个、Dataflow 三个、Schedule 三个），其中八个支持高基数明细分页，`tile_memory_allocation_summary` 返回完整聚合摘要。`pipeline_overlap_check` 因现有采集字段不能提供配对的 Compute/MTE 区间而暂缓。Stage 4 离线重放覆盖 16 份归档和 144 个工具摘要首屏校验，5,492 个索引文件哈希前后相同，硬件动作数为 0。Stage 2/3 报告未保留全部 observation 值，故跨阶段只能核对其存下的计数和来源覆盖；当前 `buffer_pressure_check` 的 Scope observations 差异及限制见 [Stage 4 离线回放报告](../reports/offline_replay_stage4_20261112.json)。最大单卡依赖图首屏临时解析峰值约 509 MB，后续 Cursor 页不再重复调用依赖图解析器。Stage 4 当前 CPU 回归为 51 项通过；PTOmni Host 接线、Basic Pi 和 Agent 配对试验尚未启动。

## Stage 5：Basic + Wiki + PTOAnalyzer 消融

75 个 Formal/Held-out Case 的 `combined` Profile 覆盖和 PTOAnalyzer 离线验收视为已完成前置条件，不在 Stage 5 重跑。[Stage 5 规划](stage5/README.md)聚焦独立的 `modes/basic_wiki-tool`：尽量保持 Basic Pi 不变，增加只读 `pto_wiki/opt_wiki` 和复制后的最新 PTOAnalyzer。Basic 处理组不按 Tuner 分配工具，九个工具全部暴露；每次实际调用（包括失败和取消）都记录。Stage 5 依赖另一个 PTOBench 公共代码任务先完成并验收通用 Pi extension/socket 接口；Stage 5 只调用该接口，不修改隔离层。Pi 通过原生 extension 调用 Session-local Unix-socket Python Broker；Stage 5 的 CPU 集成测试还要证明 Pi 侧注册并实际调用九个工具，无需真实 LLM 或 NPU。Broker 授权 Profile、分发工具并归档完整调用，Runner 负责它的启动和清理。最新 Analyzer 本体继续保留在 `modes/ptomni/pto_analyzer/`，接入使用复制体和薄适配层。接口就绪后，Stage 5 用三个 Formal Case 做 Basic 与 Basic+Wiki+PTOAnalyzer 的六个真实 Agent Session 配对；结果仅作为组合消融的可行性和行为证据，不宣称单独归因或普遍提升。完整执行门槛见 [Stage 5 任务清单](stage5/TASK_CHECKLIST.md)，启动说明见 [Stage 5 Prompt](stage5/IMPLEMENTATION_PROMPT.md)。

## Stage 4 实施与验收状态

Stage 4 已按清单完成 CPU-only 修缮和离线验收：

- JSONL/CSV 错误示例最多保留 20 条，同时保留精确拒绝总数和省略数；PMU CSV 单遍解析，`events.jsonl` 逐行读取，Scope 来源按路径索引。
- 依赖图使用受保护原文的轻量偏移索引；同一 `ToolSession` 的后续 Cursor 页不再重新解析完整图。最大单卡 453,375 条明细在 limit=50 时为 9,068 页，完整遍历无重无漏。
- 16 份 Profile 均提供一个可验证的 `*_after_AllocateMemoryAddr.py` 产物。独立 AST 复算得到 4,166 个 `pl.tile.alloc`、70 个 `pld.tensor.alloc_window_buffer`，共 1,228 个分组，和 Tool 输出逐项一致。
- 当前没有仓库内 PTOmni Host 调用 `ToolSession`，因此 Host 接线、Basic Pi 和 Agent 配对试验均未启动；它们属于后续阶段。

完整摘要、首屏、分页计数、资源测量、独立复算和 5,492 个归档条目的前后 SHA-256 见 [Stage 4 回放报告](../reports/offline_replay_stage4_20261112.json)。

## 1. 目标与依据

当前九个 Tool 复用已有真实 Profile；只有离线数据不能回答一个具体问题、且现有采集路径能产生缺失字段时，才增加 PTOBench Profile。Basic Pi 轨迹与 Agent 效用试验仍是有门槛的可选工作，不由本轮 Analyzer 修缮启动。

当前样本池是 [`post_baseline_combined_20261106`](../../../../manual_runs/post_baseline_combined_20261106/queue.json) 中的 16 个成功 `combined` Profile。索引核对结果：每个 Profile 都含 Compiler hint、Compiler pipeline profile、PMU、Runtime swimlane、依赖图和 scope statistics；多卡 Case 按 rank 保存 PMU、swimlane 和依赖文件。它们可以支撑首轮解析和证据组织，无需为开发 Parser 重跑。

样本给出的几个具体事实：

- `pmu.csv` 有 `task_id`、`func_id`、`core_id`，以及 `pmu_total_cycles`、Cube/Vector/Scalar/MTE busy cycles、I-cache 计数等列。
- `chip_swimlane_records.json` 有 AICore/AICPU 任务记录和时钟锚点；`deps.json` 有 tasks、tensors、edges。
- `scope_stats.jsonl` 有 task window、heap、dependency pool 和 TensorMap 的统计。当前 16 份索引中没有独立 Memory Report。
- `perf_hints.log` 的真实样本包含 `PH001 TileInnermostDimGranularity` 和 `PH-MR-001 MemoryReuse` 等提示。
- 当前 `pipeline_profile.json` 记录编译 Pass 的阶段耗时，不等同于 Kernel Runtime 的计算/搬运重叠时间。
- PMU 与 DAG 中的 Task ID 表示形式不同，跨文件 ID 映射尚未确认；没有验证前不得把两者直接合并。

这些 Profile 是起始 Candidate 的单点诊断记录，不含对应的 Agent 优化轨迹，也不提供 Clean Latency。因此它们能确定“怎么读、能看到什么”，不能证明 Finding 是性能根因或建议一定有效。

## 2. 当前 Tool 与历史候选

Tool 按 Tuner 的优化视角组织，同一份 `combined` Profile 共享底层数据。每个 Tool 返回可追溯事实和适用边界，不直接替 Tuner 下最终优化结论。Stage 2 曾实现九项候选；Stage 3 暂缓 `pipeline_overlap_check`，Stage 4 增加静态分配摘要。下面是当前已实现的九个 Tool。

| Tuner | Tool | 当前输出 | 解释边界 |
|---|---|---|---|
| Compute | `compute_mapping_check` | 汇总归档中的 dispatch program、callable name map 与 kernel config 映射，并标出未经验证的 Candidate 源码关联。 | 没有 Candidate 源码清单连接时不猜源码位置；映射信息本身不是运行瓶颈。 |
| Compute | `compute_unit_activity` | 按 build/rank/dispatch/core type 汇总 PMU 计数，并分页查看原始记录。 | PMU 是原始事件计数；没有校准的运行时分母，不据单项计数判定 Compute-bound。 |
| Compute | `compute_task_share` | 仅在同一 dispatch 内 task、callable 和 AICore 关联校验通过时，按 task 汇总 PMU 计数；未关联行保留原因。 | 不把未验证的 ID 直接拼接，也不把 PMU 计数与 AICore 周期合并成时间占比。 |
| Dataflow | `tile_access_check` | 统计编译器 Tile-access 提示类别，并保留提示原文、来源和位置明细。 | 编译提示只是线索；不保证收益，也不映射到未经验证的 Candidate 源码行。 |
| Dataflow | `buffer_pressure_check` | 统计 MemoryReuse 提示与 Scope 中记录的原始字段及聚合值。 | 不是独立容量报告；不从静态 Scope 计数推断运行时内存压力或瓶颈。 |
| Dataflow | `tile_memory_allocation_summary` | 按归档产物、函数、memory space 与操作汇总编译期分配请求数和显式字节数。 | 静态请求之和不表示运行时存活量、峰值占用、容量压力、spill 或瓶颈。 |
| Schedule | `dependency_path_check` | 汇总依赖图 task、node、edge 等结构事实，并分页查看 task/edge 明细。 | 依赖图不含 runtime 权重；结构关系不直接代表延迟或瓶颈。 |
| Schedule | `dispatch_gap_check` | 在身份覆盖通过的同核 AICore 区间间统计正向周期间隔。 | 间隔不直接归因于调度；没有完整 AICore/AICPU 配对覆盖时不生成该项观测。 |
| Schedule | `task_distribution_check` | 按 rank/dispatch/core 汇总 AICore 物理记录、原始 task token 与区间周期。 | 不由 task 数量或区间总和推断忙碌时间或负载均衡。 |

所有当前 Tool 使用同一事实结果契约：Profile 身份、观测与单位、来源覆盖、来源引用及解释限制。结果不使用 `finding` 等诊断状态，不输出统一严重度、根因置信分或预计加速比。高基数明细通过不透明 Cursor 逐页访问，保持摘要完整。

Stage 2 的九项是历史诊断目标。`pipeline_overlap_check` 没有独立的 runtime overlap 观测：PMU 的 MTE busy-cycle 是汇总计数，Runtime 与 merged trace 不提供可配对的 MTE 起止区间，`pipeline_profile.json` 记录编译 Pass 时间。普通 task 起止区间不能代替 compute/MTE 阶段区间。现有 Case 不能补出采集器没有产生的字段；需 PTOBench 先定义共同时间基准和关联身份后再考虑恢复。

## 3. 现有 Profile 的离线核验

1. Stage 3/4 已核对 Profile event、Candidate digest、设备覆盖和归档身份，并在 16 份既有归档上运行当前工具集合。
2. 回放记录了来源覆盖、分页数、摘要事实 digest 和哈希前后检查；单卡、2 卡和 8 卡样本用于检查高明细与多 rank 情况。
3. `pipeline_overlap_check` 仍因缺少可关联的 Compute/MTE 起止区间而停用；`buffer_pressure_check` 相对 Stage 2 的 Scope observations 有变化，历史报告未保存观测名称和值，不能声称逐字段历史一致。
4. Stage 4 CPU-only 回放不产生硬件动作。需要新 Profile 时，必须先确认现有采集器能产生具体缺失字段。

## 4. 根据结果选择下一批 Case

### 4.1 只需 PTOBench Profile 的情况

当缺口是现有采集器可输出的字段格式、设备/rank 变体、单位或特定 Profile 产物时，才考虑 PTOBench `combined profile` 补一个目标样本。不要为满足工具数量或覆盖对称而运行 Case；先确认目标字段已由采集路径产生。

| 待解决的证据缺口 | 候选 Case | 选择理由 |
|---|---|---|
| Operator 级 MatMul 与双卡 PMU/Runtime 形态 | `deepseek_v4.lm_head_tp2_flash`（Formal） | 2 卡、MatMul + communication，可补计算计数和跨 rank 记录的样本。 |
| 单卡 Attention 的 Runtime 与依赖形态 | `deepseek_v4.decode_sparse_attn_demo`（Held-out） | 单卡 Attention，补充与现有多阶段/多卡 Profile 不同的形态。 |
| 模型级、多机制、多卡产物 | `deepseek_v4_flash_dspark.dspark_prefill_pro`（Held-out） | 模型级 MatMul/Attention/Normalization/Selection/Communication，适合检查综合 Profile 结构。 |

这些是按问题选取的候选，不是预先要求全部运行。只有当 16 份现有 Profile 不能覆盖待核对字段时才运行；一次成功 `combined profile` 计一次 PTOBench 硬件动作。

### 4.2 需要 Basic Pi 优化轨迹的情况

只有离线输出看起来可解析，但还无法判断证据是否随代码修改变化、是否能帮助选择优化方向时，才让 Basic Pi 跑真实优化。Basic Pi 的 Case 必须来自 Formal 或 Held-out 清单，并优先选择同时覆盖多个未解决问题的代表性 Case；按缺口逐步选择，不预设固定数量或完整实验矩阵：

| Case | 清单 | 主要覆盖 | 当前可复用的起始 Profile |
|---|---|---|---|
| `deepseek_v4.lm_head_tp2_flash` | Formal | Operator 级 MatMul/通信、Compute 与双卡记录 | 无同 Case Profile；需先 Profile 起始 Candidate |
| `deepseek_v4.prefill_layer_ep2_csa_demo` | Formal | 多阶段 CSA/HCA、Tile/Buffer/Pipeline 线索 | 有同 Case Profile；校验身份后复用 |
| `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | Formal | CP8、跨 rank 数据流/Task/PMU | 有同 Case Profile；校验身份后复用 |
| `deepseek_v4.moe_ep2_demo` | Formal | 2 卡 MoE Task 与依赖 | 有同 Case Profile；校验身份后复用 |
| `deepseek_v4.moe_ep8_demo` | Formal | 8 卡 MoE 分布与并行调度 | 有同 Case Profile；校验身份后复用 |
| `deepseek_v4_flash_dspark.dspark_prefill_pro` | Held-out | 模型级、多机制交互与端到端轨迹 | 无同 Case Profile；需先 Profile 起始 Candidate |

上表仅供按缺口挑选，不要求全部运行。先复用身份一致的起始 Profile；缺少基线或优化候选 Profile 时，才补采所需版本。`deepseek_v4_flash_mtp.sample_demo` 可作为简单控制 Case，用于检查工具面对弱信号时是否能正确返回 `no_signal`/`insufficient_evidence`，但若需要 Basic Pi 真实优化轨迹，也应从 Formal 或 Held-out 清单挑选。

Basic Pi 每个 Case 至少保留起始 Candidate 的 Profile，并在有代表性的优化候选上采集 Profile；不要求每次编辑后都采集。每个 Profile 都要绑定代码快照/hash、diff、PTOBench event、优化轮次和对应 `evaluate` 结果。Profile 是诊断证据，不提供正确性或 Clean Latency；`evaluate` 结果单独记录。已有起始 Profile 只有在 Candidate、设备、编译配置和软件版本一致时才可复用。

从 Basic 轨迹离线运行 PTOAnalyzer 后，逐条检查：Tool 是否指出当时可见的事实、事实是否指向对应的优化机制、后续修改是否检验该线索、正确性/延迟如何变化。单条轨迹只用于检查诊断可读性与可追溯性，不足以证明 PTOAnalyzer 会提升优化结果。

若要验证 PTOAnalyzer 是否减少无效尝试，后续再做小规模配对：同一 Basic Pi、相同起始 Candidate、Case、模型和 PTOBench 预算，只改变 Agent 看到原始 Profile 还是角色化 Tool 结果；比较有效/无效尝试、硬件动作、最终 `evaluate`、Token 和墙钟时间。两组的 Profile 配置与次数保持一致。

## 5. 后续工具与采集变更门槛

- 当前九项 Tool 均已实现；后续新增或重建诊断前，先明确输入证据、来源引用和限制，不隐藏已知的 Task ID、时钟、单位或窗口问题。
- 优先使用已归档且语义可靠的字段；缺失证据应显式记录，不用零值或未经验证的跨源关联替代。
- Parser 只规范化当前 Tool 实际需要的字段；大型 IR、Timeline 按需读取。原始 Profile 保持只读，Tool 不调用 PTOBench、不连接 NPU、不维护硬件账本。
- Basic 或 PTOBench 的额外硬件运行必须由 Runner/PTOBench 正常执行并计数；离线 Analyzer 运行不产生硬件动作。

正式 Case 来源：[`formal_60`](../../../../case_selection/formal_60/cases.csv)、[`heldout_15`](../../../../case_selection/heldout_15/cases.csv)。当前整体架构与 Tool 接入关系见 [PTOAnalyzer 设计与验证计划](../README.md)。
