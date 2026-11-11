# PTOAnalyzer 实施与后续任务清单

本清单记录首版实现状态及后续 Case 选择。已完成实现与现有样本的离线核验；受控 Agent 效用小试仍是独立后续工作。规划目录是临时材料；实际 Analyzer 代码和测试放在 `modes/ptomni/pto_analyzer/`。

## 0. 预检与样本锁定

- [x] 阅读 PTOAnalyzer 总览、本目录实现 Prompt 和 PTOmni 最新架构文档。
- [x] 从 `post_baseline_combined_20261106` 核对已有 16 个 Profile 的事件、Candidate、设备、编译配置、软件版本和归档完整性。
- [x] 选出 8 个离线审计样本（queue index 1、2、4、6、7、10、12、15），覆盖 1、2、8 卡及多种 Case 形态；其余 8 个用于扩展格式回放。
- [x] 记录真实字段、单位、缺失产物、rank 布局和未验证关联；没有把这些关联写成诊断结论。

**验收：** 能为每份选中 Profile 指向确切的只读归档，并清楚列出它能支持和不能支持的诊断。

## 1. 初版九个 Tool

- [x] Compute：实现 `compute_mapping_check`、`compute_unit_activity`、`compute_task_share`。
- [x] Dataflow：实现 `tile_access_check`、`buffer_pressure_check`、`pipeline_overlap_check`。
- [x] Schedule：实现 `dependency_path_check`、`dispatch_gap_check`、`task_distribution_check`。
- [x] 三类 Tuner 共用 Profile 身份验证、归档读取和最小证据解析；按静态权限只开放本 Tuner 的 Tool。
- [x] 所有结果包含来源、单位、Profile/Candidate/event 身份、采集/解析状态、适用限制和待验证线索。
- [x] 缺失、不支持、无效、无信号分别返回状态；缺失不当作零值，未校准的规则不输出根因分数或预计加速比。

**验收：** 九项都能基于真实样本运行并返回有效状态；有无 Finding 由证据决定，不要求每项都产生诊断。

## 2. 现有数据离线回放与复核

- [x] 在 8 个代表 Profile 上回放九个 Tool，并扩展到现有全部 16 个；没有新增硬件运行。
- [x] 记录每项 Tool 的可用证据与限制：格式覆盖结果见回放报告；工具结果可用于证据输出不等于支持性能根因结论。
- [x] 用真实格式 Fixture 对照 PMU 汇总、依赖图、编译提示和 task join；8 个代表样本的原始 PMU/Task/Scope/Compiler/swimlane 事实均独立核对，Task ID 仅在同一 dispatch scope 内关联。
- [x] 验证重复调用稳定性、Tuner 权限、缺失/无效状态及原始归档只读性。
- [x] 汇总数据缺口，并区分缺少关联语义与需要优化前后变化/Agent 效用证据的问题。

**验收：** 每条结论都有原始证据引用；相关性没有写成确定根因；每个待补样本都对应一个已知问题。

### 当前回放结论

| 范围 | 当前可用性 | 回放结果与限制 |
|---|---|---|
| `tile_access_check` | 可用（编译器提示） | 16/16 有 PH001 提示；不是运行时瓶颈证据。 |
| `buffer_pressure_check` | 可用（提示与原始 Scope 计数） | 13/16 有 MemoryReuse 提示；部分 Scope 字段没有声明单位。 |
| `compute_unit_activity` | 可用（PMU 观察） | 16/16 可解析 PMU 核心计数，按 build/rank/dispatch/core 分组；没有校准阈值，不能判定利用率不足。 |
| `compute_task_share` | 有条件可用 | 累计 24 个 dispatch scope、96,216 行 PMU 通过 Task/kernel/name 一致性校验；AICore tick 与 PMU cycles 不作比例或瓶颈推断。 |
| `dependency_path_check` | 可用（图结构） | 16/16 返回依赖图结构；没有 runtime 权重，路径深度不是关键路径时长。 |
| `task_distribution_check` | 可用（记录计数） | 16/16 返回 swimlane 记录事实；缺少完整负载均衡解释与跨 rank 覆盖保证。 |
| `compute_mapping_check` | 需关联验证 | 16/16 为 `insufficient_evidence`；compiler callable/name 数据未完成 Candidate 源码位置映射。 |
| `dispatch_gap_check` | 需关联验证 | 16/16 为 `insufficient_evidence`；完整派发捕获、依赖映射与间隙归因尚未验证。 |
| `pipeline_overlap_check` | 需关联验证 | 16/16 为 `insufficient_evidence`；未验证 PMU、swimlane 和设备时钟的共同窗口。 |

所有九项都不具备“已证实性能根因”的证据。回放覆盖 5,492 个受保护归档文件；报告及逐 Profile 状态见 [`../reports/offline_replay_20261106.json`](../reports/offline_replay_20261106.json)。

## 3. 根据诊断效果选择下一批 Case

对每个缺口先回答：现有 Profile 为何不足？新数据将区分哪种解释？若答案不明确，不增加 Case。

### A. 只需 PTOBench 原始 Profile

- [ ] 缺口属于字段结构、单位、rank/设备变体或特定产物时，选择覆盖该缺口的最小 Case。
- [ ] 优先从 Formal/Held-out 清单中选与缺口匹配的代表性 Case；一个 Case 可覆盖多个缺口。
- [ ] 复用身份完全一致的已有 Profile；否则通过 Runner/PTOBench 正常执行一次 `combined profile`。
- [ ] 记录 Case、问题、预期新增字段/形态、设备与软件条件，以及硬件动作成本。

### B. 需要 Basic Pi 的真实优化轨迹

- [ ] 只有要判断证据是否随代码改变，或 Analyzer 是否帮助 Tuner 选择优化方向时，才安排 Basic Pi。
- [ ] 从 `formal_60` 或 `heldout_15` 清单选择 Case；优先代表性强、可同时验证多个缺口的 Case。
- [ ] 不按 Dev/Test 划分筛选，不预设固定 Case 数或完整矩阵；每个 Case 都必须对应明确待验证问题。
- [ ] 复用身份一致的基线 Profile；记录起始 Candidate hash、代码 diff、每个 Profile/evaluate 事件和对应优化假设。
- [ ] 只在有代表性的候选版本上 Profile，不要求每次编辑都采集；Profile 不替代正确性和 Clean Latency 检查。
- [ ] 若要验证 Analyzer 是否减少无效尝试，成对固定模型、Case、起始 Candidate、预算和其他 Guidance，只改变 Agent 获得信息的方式。

**验收：** 每个新增运行都有明确问题、选择理由、预期判读方式和预算；没有为了覆盖列表而盲跑。

## 4. 收敛与交付

- [x] 确认九个 Tool 的实际有效范围；无关联证据的能力保留为 `insufficient_evidence`，不虚构结论。
- [x] Fixture 复用真实归档格式并覆盖缺失、空文件、损坏输入和错类型 JSON；测试在 CPU 上运行。
- [x] 输出回放摘要、源数据核对结果、样本选择理由及已知限制。
- [x] 实现没有 PTOBench/硬件调用入口；CPU 测试检查归档只读性和 Lead 无工具权限。
- [x] 当前没有能由泛化新 Profile 回答的明确问题；不增加 Case。若后续锁定可补足的字段或设备变体，再挑选最小样本。

**最终完成条件：** 每个工具说得清依据与边界；额外运行只补明确缺口；Tuner 能据此提出可检验实验，而 Analyzer 不代替 Tuner、Lead 或 Assembler 决策。
