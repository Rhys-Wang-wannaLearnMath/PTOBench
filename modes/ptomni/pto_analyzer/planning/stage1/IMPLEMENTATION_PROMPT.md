# PTOAnalyzer 实现 Prompt

以下内容可直接作为后续实现任务的起始 Prompt。它以 PTOAnalyzer 总览和本目录的规划为准；若实现过程中发现真实数据不支持某个假设，应记录证据并收窄功能，不得补造指标或诊断结论。

```text
你正在 PTOBench 嵌套 Git 仓库中实现 PTOmni 的轻量 PTOAnalyzer。

目标
在 modes/ptomni/pto_analyzer/ 中实现按 Tuner 专长开放的只读诊断 Tools。它们使用 PTOBench 已归档的 combined Profile，向 Tuner 提供可追溯事实、证据限制和待验证线索。PTOAnalyzer 不采集 Profile、不运行 NPU、不改 Kernel，也不判断候选正确性或正式性能。

开始前先读
1. modes/ptomni/pto_analyzer/README.md
2. modes/ptomni/pto_analyzer/planning/README.md
3. modes/ptomni/pto_analyzer/planning/TASK_CHECKLIST.md
4. modes/PTOmni/design_docs/PTOmni_架构设计_2026-11-03.md
5. PTOBench Profile 产物与 manual_runs/post_baseline_combined_20261106/queue.json 中的真实样本索引。

架构边界
- PTOBench 负责采集、原始归档和硬件动作计数；Runner 负责 CLI 调用、设备租约、预算与执行记录。
- PTOAnalyzer 只读确切的 ProfileRef 和其受保护归档；不得扫描 .benchmark_outputs 猜最新结果，不得修改原始产物。
- 工具只开放给当轮启用的 Compute、Dataflow 或 Schedule Tuner。Lead 不持有 PTOAnalyzer Tool，也不启动实验；Assembler 不负责分发 Tool。
- PTOAnalyzer 不连接 NPU、PTOBench CLI、Runner，不改代码，不调用内部 LLM，不维护第二套硬件账本。
- 同一 combined Profile 可供多个 Tuner 复用，但工具结果必须保留 Profile、Candidate 和 PTOBench event 身份。

初始候选工具
Compute: compute_mapping_check, compute_unit_activity, compute_task_share
Dataflow: tile_access_check, buffer_pressure_check, pipeline_overlap_check
Schedule: dependency_path_check, dispatch_gap_check, task_distribution_check

实现顺序
1. 盘点现有 Profile、Candidate 身份、产物字段、单位、设备/rank 变体和缺失情况。先选能覆盖 1/2/8 卡及代表性计算/数据流/调度形态的最小离线样本集；不为开发 Parser 重跑 NPU。
2. 以实际产物实现共享、按需解析；只规范化九个候选 Tool 实际需要的字段。
3. 为九个候选 Tool 提供稳定的结构化结果。证据不足、不支持、数据无效和无信号必须分别表示；没有证据的 Tool 可以返回相应状态，不能伪造 finding。
4. 每条结果都携带单位、原始来源引用、Profile/Candidate/event 身份、采集/解析状态、限制和可验证的后续观察。未经验证的跨文件 Task 映射、时钟窗口或单位不能合并计算。
5. 在 CPU-only Fixtures 上验证权限、来源追溯、缺失语义、重复调用稳定性和原始产物不变；再将结果与对应原始 Profile 人工核对。
6. 根据九个 Tool 在现有样本上的实际输出，建立“可用 / 需关联验证 / 当前无证据”的结果表，并列出尚未回答的具体问题。只有这些问题决定是否增加 Case。

Case 选择规则
- 若缺口只是字段格式、设备/rank、单位或某类原始产物，选择能补足该缺口的最小 Case，用 PTOBench combined profile；不为九个 Tool 各跑一个 Case。
- 若缺口涉及证据是否随代码改动变化，或工具是否帮助 Agent 选择优化方向，才考虑让 Basic Pi 跑真实优化轨迹。
- Basic Pi 的优化 Case 必须来自 formal_60 或 heldout_15 清单，并优先选择能覆盖多个未解决问题的代表性 Case。Dev/Test 划分不作为筛选依据，不预设固定 Case 数或实验矩阵。
- 优先复用 Candidate、设备、编译配置和软件版本均一致的已有 Profile；身份不一致时不得复用。
- 对每个新增 Case，先写清它要回答的问题、所需 Profile/优化轨迹、为何现有数据不足及预期观察结果，再通过 PTOBench/Runner 的正常路径执行。profile/evaluate 均遵守设备锁、预算和 PTOBench 计数。
- 只有在要比较“原始 Profile”与“角色化 Tool 输出”对优化效果的影响时，才设计受控 Basic Pi 配对试验；固定 Case、起始 Candidate、模型、预算和其他 Guidance，只改变信息接口。该小试不自动扩展论文主实验。

判断与报告
- PMU 活动、编译提示、依赖链、短 Task 或观测间隙都只是证据，不可单独写成已确认瓶颈。
- pipeline_profile.json 的编译 Pass 时间不能当作 Kernel Runtime 的计算/搬运重叠时间。
- PMU 与 DAG 的 Task ID、不同 rank 的时钟、采集范围和单位，只有验证后才允许关联或计算。
- 报告应分开说明解析覆盖、证据质量、Tool 对 Tuner 的可用性，以及需要新数据验证的问题；Profile 不提供 correctness 或 Clean Latency，正式性能仍以 evaluate 为准。
- 不新增通用插件框架、数据库、学习阈值、自动 Case 选择器或新的采集器；不为了凑数产生低置信度规则。

完成标准
1. 九个候选 Tool 的可见权限与所属 Tuner 一致。
2. 每个输出都能追溯来源，缺失/不支持/无效/无信号没有被混为零值或正常值。
3. 现有样本离线回放结果经人工抽查与原始数据一致，并形成工具效果与数据缺口摘要。
4. 若提出新 Case，附上上述选择理由；没有明确问题就不启动新硬件运行。
5. PTOAnalyzer 的实现与测试都留在 modes/ptomni/pto_analyzer/；PTOBench 的采集事实和其他 Mode 不被改写。
```
