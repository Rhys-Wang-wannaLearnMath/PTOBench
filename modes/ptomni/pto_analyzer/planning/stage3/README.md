# PTOAnalyzer Stage 3：工具收敛与高基数明细访问

> 状态：Stage 3 实现与离线验收已完成。结果见 [Stage 3 离线回放报告](../../reports/offline_replay_stage3_20261108.json)。

Stage 3 处理了 Stage 2 暴露的两件事：`pipeline_overlap_check` 缺少可配对的运行时区间，因此从当前接口撤下；高基数明细改用绑定 Profile 与工具身份的 Cursor 分页。实现通过 CPU 测试，并用既有 16 份归档完成离线回放；没有启动硬件或 Agent 试验。

## Stage 2 证据

- Stage 2 的 31 项 CPU-only 测试通过；回放覆盖 16 份 combined Profile，抽查单卡、双卡与 8 卡；新增硬件动作数为 0。详见 [Stage 2 任务清单](../stage2/TASK_CHECKLIST.md) 和 [离线回放报告](../../reports/offline_replay_stage2_20261108.json)。
- 16 份回放中，`pipeline_overlap_check` 都没有性能观测。现有 `pipeline_profile.json` 是编译 pass 耗时，尚无能按 Candidate、设备、时间基准与 task/block 身份关联的运行时 Compute/MTE 区间。
- 当前最高明细量来自 `deepseek_v3_2.deepseek_v3_2_prefill_back`：依赖图明细 453,375 条；8 卡 `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` 的 gap/task 明细各 134,936 条。这里的数量说明离线访问规模，不代表瓶颈或性能结论。
- 现有 `DetailRecords` 按 `offset + limit` 保存候选记录。请求远端页面时会保留大量不返回的前序记录；超出末页的请求也可能先保留所有记录。

## Stage 3 目标

1. 把 PTOAnalyzer 对外提供的工具限制在当前能给 Tuner 带来独立、可操作证据的工具；没有实测字段时不保留 overlap 空壳，也不为维持“每位 Tuner 三个工具”而制造替代工具。
2. 让高基数分页的内部明细缓存有固定上限，同时保持全量摘要、稳定顺序和无重无漏的逐页读取。
3. 用现有 16 份归档和 CPU-only 测试验证实现，再按实际证据缺口决定是否 Profile；只有要验证 Agent 是否会使用 Analyzer、并减少无效优化尝试时，才进入真实 Agent 小试。

## 设计决策

### 工具集合

Stage 3 默认将 `pipeline_overlap_check` 暂时从 Dataflow 的 `tool_specs`、`initial_summary` 和调用分发表移除；不以空工具维持 3×3 数量。若后续证据审计确认当前或新归档已有身份和时间基准都可验证的 Compute/MTE 起止区间，则实现真正的 overlap 统计后再评估重新开放。若 PTOBench 当前采集路径不能产生这些字段，不为同一缺口重复运行 Case；先记录最小采集需求。

### 分页接口

把高基数工具的 `offset` 改为不透明的 `cursor`：首次调用省略 cursor，后续把上页返回的 `next_cursor` 传回；每页 `limit` 默认 20、最大 50。Cursor 绑定 Profile 事件、Candidate 身份和 Tool，不能改变归档路径或分析权限。内部只保留当前页所需候选，摘要与总记录数仍按全量数据计算。

分页由 `ToolSession.call(name, args)` 这个外部 seam 提供。`DetailRecords` 或其替代物只作为 PTOAnalyzer 内部实现；不要让调用方了解各 Parser 的排序键，也不要增加数据库、落盘索引或第二套工具服务。Cursor 分页可能每次重新扫描归档；先测量真实高基数样本的时间与内存，再决定是否需要内部缓存，不预先引入它。

## 顺序与门槛

| 阶段 | 交付 | 通过门槛 |
|---|---|---|
| S3.0 现状复核 | 工具价值决定、分页接口契约、压力样本与基线记录 | Stage 2 结果复核；需求不扩展到无证据支持的性能结论 |
| S3.1 工具收敛 | overlap 工具暂停暴露或基于真实区间重建 | 无空壳工具；角色权限和摘要内容与实际工具表一致 |
| S3.2 有界分页 | Cursor API 与内部固定大小的页面选择 | 最多保留 `limit` 条候选；排序稳定，遍历无遗漏/重复 |
| S3.3 离线验证 | CPU-only 边界测试与真实 Profile 回放 | 31 项既有测试及新增测试通过；16 份 Profile 归档哈希不变 |
| S3.4 采集缺口门 | 决定是否需要新 Profile | 仅当当前 combined 采集能补到具体缺失字段时才运行；否则只记录 PTOBench 需求 |
| S3.5 Agent 小试（可选） | Basic Pi 轨迹或配对工具效用试验 | 前置条件、Case、预算与记录字段预先固定；通过 Runner/PTOBench 执行 |

## Case 与实验安排

### 先用现有数据，不先重跑

- 复放 16 份已有 Profile。重点用 `deepseek_v3_2.deepseek_v3_2_prefill_back` 检查最高明细量，用 `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` 检查 8 卡路径，再抽查 `deepseek_v4.prefill_layer_ep2_csa_demo` 的双卡来源覆盖。
- 用 CPU-only 合成数据覆盖大规模稳定分页、重复排序键、末页、空结果、非法/跨 Profile Cursor 和 `limit=1/50`。压力测试验证缓存上界，不用超大文件或 NPU 才能测分页。
- 只有确认真实 combined Profile 缺少某个工具所需的字段、且当前 PTOBench 采集器能输出它，才补一个 Profile。重叠区间若采集器无法提供，重跑任意 Case 都不会验证这个工具。

### 可选的 Basic Pi 优化轨迹

若离线回放无法判断观测是否会随 Candidate 变化，可从 Formal 清单挑 1–2 个 Case 运行 Basic Pi，并保存起始和有代表性的优化候选 Profile。优先使用多机制、多卡样本：`deepseek_v4.prefill_layer_ep2_csa_demo`（Formal，2 卡）与 `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo`（Formal，8 卡）。先验证已有 Profile 的 Candidate digest、设备和软件配置是否匹配；匹配则复用，不匹配才为相应 Candidate 采集 `combined` Profile。

此轨迹只回答“工具能否准确呈现优化前后可观察事实”，不证明 Analyzer 帮助 Agent 提升性能。记录每个 Profile 的 Candidate hash、diff、事件序号和对应 `evaluate` 的正确性/延迟结果；每次 `profile`/`evaluate` 都经 Runner/PTOBench 正常记账并遵守设备锁。

### 可选的 Analyzer 效用配对小试

只有在 `ToolSession` 已能通过 PTOmni/实验 Harness 的原生工具接口提供给 Tuner、离线验收通过后才启动。用相同模型、起始 Candidate、Tuner 角色、其它上下文、实验预算和工具权限，配对比较“读取原始 Profile”与“读取 PTOAnalyzer 摘要并按需分页”。可先用上述一个 2 卡 Formal Case 和一个 8 卡 Formal Case，每个条件各跑一次作为可行性试验；若预算允许再重复种子。顺序随机或交错，所有 `profile` 与 `evaluate` 走 Runner，硬件动作上限一致。

记录最终 `evaluate`、正确性、硬件动作数、有效/无效/重复优化尝试、Profiler/Analyzer 调用、Token 和墙钟时间。这个小试只检查工具能否减少无效尝试和是否值得扩大实验，不用于论文级统计结论。若宿主无法安全提供按角色授权的 Tools，则先记录工具接线缺口，不模拟 Agent 调用，也不启动 NPU 试验。

## 边界

- PTOAnalyzer 仍是只读、离线的证据整理模块；Lead 不获得 Analyzer 或实验 Tool。
- Analyzer 不调用 PTOBench、不连接 NPU、不修改原始归档，也不维护硬件动作账本。
- 新 Profile/evaluate/Agent 运行都由 Runner/PTOBench 调用，受设备锁、预算和 PTOBench 硬件动作计数约束。
- Stage 3 不加入新状态标签、性能评级、自动根因结论、阈值、优化处方、通用分页框架或数据库。
- 只更新 PTOAnalyzer 代码、测试、结果与本目录相关文档；若需要 PTOBench 新采集字段，先单独提交可审查的采集需求，不混入 Analyzer 改造。

## 文件

- [任务清单](TASK_CHECKLIST.md)：逐项执行和验收步骤。
- [实施 Prompt](IMPLEMENTATION_PROMPT.md)：可直接用于后续实现工作的任务说明。
- [Stage 2 设计与验收](../stage2/README.md)：上一阶段背景。
- [PTOAnalyzer 当前接口](../../README.md)：工具权限、角色和归档边界。

## Stage 3 验收状态

- 已完成八个只读 Tool 的角色收敛、Cursor 分页和离线回归；16 份 Profile、128 次摘要/分页调用均纳入回放。
- 5,492 个索引产物在回放前后 SHA-256 一致；新增硬件动作数为 0。
- Stage 2 报告没有保存全部 observation 名称和值，因此只核对其记录的 observation 数量与来源覆盖；不声称跨阶段逐值一致。`buffer_pressure_check` 的额外 Scope observations 及差异原因见 Stage 3 回放报告。
- Basic Pi 和“原始 Profile 对比 PTOAnalyzer”的 Agent 效用试验均为可选后续工作，本阶段没有启动。
