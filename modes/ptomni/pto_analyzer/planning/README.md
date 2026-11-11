# PTOAnalyzer Tool 与样本规划

> Stage 1/2 的九项诊断是历史实现目标。Stage 3 后的当前接口有八个只读 Tool（Compute 3、Dataflow 2、Schedule 3）；`pipeline_overlap_check` 暂缓。Stage 3 的实测证据、分页实现和离线验收见 [Stage 3 报告](../reports/offline_replay_stage3_20261108.json)。

可直接交给后续实现任务的材料：

- [实现 Prompt](IMPLEMENTATION_PROMPT.md)：限定范围、权限边界、实施顺序和交付物。
- [任务清单](TASK_CHECKLIST.md)：按阶段推进并在每个阶段设验收门槛。

Stage 1 已完成初版实现与离线回放。Stage 2 修缮与离线验收也已完成；设计依据、执行清单和回放结果见 [Stage 2 总览](stage2/README.md)、[Stage 2 任务清单](stage2/TASK_CHECKLIST.md) 和 [Stage 2 回放报告](../reports/offline_replay_stage2_20261108.json)。原 [Stage 2 实施 Prompt](stage2/IMPLEMENTATION_PROMPT.md) 保留作历史实施范围记录。

Stage 3 已完成工具收敛与 Cursor 分页改造：当前对外为八个只读 Tool（Compute 三个、Dataflow 两个、Schedule 三个）；`pipeline_overlap_check` 因现有采集字段不能提供配对的 Compute/MTE 区间而暂缓。高基数明细使用绑定 Profile event、Candidate、Tool 和 Tuner 的不透明 Cursor。2026-11-09 离线重放覆盖 16 份归档和 128 个摘要/分页调用，5,492 个索引文件哈希前后相同，硬件动作数为 0。Stage 2 报告未保留 observation 值，故跨阶段只能核对其存下的计数和来源覆盖；当前 `buffer_pressure_check` 有额外 Scope observations，细节见 [Stage 3 离线回放报告](../reports/offline_replay_stage3_20261108.json)。最大单卡依赖图每页约 29 秒且临时解析峰值约 0.87–0.95 GiB；页候选仍限制为 50 条，若该扫描成本实际阻碍 Agent 查阅再评估 Session 缓存。

## 1. 目标与依据

当前八个 Tool 复用已有真实 Profile；只有离线数据不能回答一个具体问题、且现有采集路径能产生缺失字段时，才增加 PTOBench Profile。Basic Pi 轨迹与 Agent 效用试验仍是有门槛的可选工作，不由本轮 Analyzer 修缮启动。

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

Tool 按 Tuner 的优化视角组织，同一份 `combined` Profile 共享底层数据。每个 Tool 返回可追溯事实和适用边界，不直接替 Tuner 下最终优化结论。Stage 2 曾实现九项候选；Stage 3 暂缓 `pipeline_overlap_check`，Dataflow 当前只暴露下面两个 Tool。

| Tuner | Tool | 证据与初版行为 | 初版边界 |
|---|---|---|---|
| Compute | `compute_mapping_check` | 读取编译提示、可用 IR/源码映射及 PMU task/function 标识，列出能确认的计算路径或映射线索。 | 当前样本中的编译提示偏向 Tile/Memory；缺少可靠映射时返回 `insufficient_evidence`，不从编译 Pass 耗时推断运行瓶颈。 |
| Compute | `compute_unit_activity` | 按 device/rank、core、task/function 摘要 PMU 的 Cube、Vector、Scalar 与 MTE 计数；仅在分母和单位确认后计算比例。 | 展示活动证据，不用未经校准的阈值判定 Compute-bound；单项低利用率也不等同于计算表达低效。 |
| Compute | `compute_task_share` | 在 Task 身份和时间范围可关联时，统计可识别计算 Task 的 Runtime 时间占比，并列出对应 PMU 计数。 | 先验证 PMU、swimlane、DAG 的 Task ID 映射；不能关联时分别输出来源数据，不拼接、不计算占比。 |
| Dataflow | `tile_access_check` | 解析 `perf_hints.log` 中 Tile 形状、访问粒度、搬运大小和编译器建议；初期优先覆盖已见的 `PH001`。 | 保留 hint 原文、位置和适用条件；只有源码快照身份可核实时才给源码行号。Hint 是线索，不是收益保证。 |
| Dataflow | `buffer_pressure_check` | 汇总 `PH-MR-001` 等 Buffer/MemoryReuse 提示，并读取 scope statistics 中的 heap、task window、dep pool、TensorMap 统计。 | 当前无独立 Memory Report；只报告已采集的容量/占用事实和编译器观察，不推断全局内存瓶颈或未采集的容量。 |
| Schedule | `dependency_path_check` | 从 `deps.json` 提取 Task/DAG 数量、依赖深度、扇入/扇出和结构性串行链。 | 没有经过验证的运行时权重时，只报告 DAG 结构；依赖链长不等于性能瓶颈。 |
| Schedule | `dispatch_gap_check` | 从 swimlane 的 Task 时间戳和时钟锚点计算同一设备/执行流上的观测间隔，并保留时间单位与不确定度。 | 未区分的空白不能直接叫调度开销；跨卡比较仅在时钟归一化误差足够小时启用。 |
| Schedule | `task_distribution_check` | 按 rank/device/core 汇总 Task 数、block 数和可用的持续时间分布，列出负载差异事实。 | 不用 Task 多、短或单卡时间较长直接判定负载不均；比较需限定在同一 Profile 和可比执行范围。 |

所有当前 Tool 使用同一事实结果契约：Profile 身份、观测与单位、来源覆盖、来源引用及解释限制。结果不使用 `finding` 等诊断状态，不输出统一严重度、根因置信分或预计加速比。高基数明细通过不透明 Cursor 逐页访问，保持摘要完整。

Stage 2 的九项是历史诊断目标。`pipeline_overlap_check` 没有独立的 runtime overlap 观测：PMU 的 MTE busy-cycle 是汇总计数，Runtime 与 merged trace 不提供可配对的 MTE 起止区间，`pipeline_profile.json` 记录编译 Pass 时间。普通 task 起止区间不能代替 compute/MTE 阶段区间。现有 Case 不能补出采集器没有产生的字段；需 PTOBench 先定义共同时间基准和关联身份后再考虑恢复。

## 3. 先用现有 Profile 检查当前八项

1. 核对 Profile 事件、Candidate 快照、设备、软件和编译配置身份，确定各工具可读取的确切归档。
2. 从 16 份数据中挑选单卡、2 卡和 8 卡样本，运行当前八项离线诊断；记录字段覆盖率、解析稳定性、来源可追溯性、跨文件关联是否成立，以及结果是否能回答对应 Tuner 的问题。
3. 对每项 Tool 标记 `可用`、`需关联验证`、`当前无证据`。用原始 Profile 人工核对输出，不把单个 Case 的数值设成通用阈值。
4. 只有实际结果暴露出具体缺口后，才选择下一批 Case。离线解析、Fixture 和报告复核都不产生硬件动作。

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

## 5. 进入 Tool 实现的条件

- 当前八项 Tool 的输入证据、来源引用和限制都有明确契约；已知的 Task ID、时钟、单位或窗口问题不会被隐藏。
- 先实现字段与证据最可靠的诊断；其余 Tool 可以保留为有明确信息的 `insufficient_evidence`/`unsupported`，待目标 Case 补齐证据再启用。
- Parser 只规范化当前 Tool 实际需要的字段；大型 IR、Timeline 按需读取。原始 Profile 保持只读，Tool 不调用 PTOBench、不连接 NPU、不维护硬件账本。
- Basic 或 PTOBench 的额外硬件运行必须由 Runner/PTOBench 正常执行并计数；离线 Analyzer 运行不产生硬件动作。

正式 Case 来源：[`formal_60`](../../../../case_selection/formal_60/cases.csv)、[`heldout_15`](../../../../case_selection/heldout_15/cases.csv)。当前整体架构与 Tool 接入关系见 [PTOAnalyzer 设计与验证计划](../README.md)。
