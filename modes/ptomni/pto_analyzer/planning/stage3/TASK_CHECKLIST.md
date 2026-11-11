# PTOAnalyzer Stage 3 任务清单

> 状态：Stage 3 实现与离线验收已完成。Stage 3 的历史工具集合为八个；Stage 4 后续新增 `tile_memory_allocation_summary`，不改写本阶段历史记录。Basic Pi 与 Agent 配对试验仍未启动。

## 目标

移除或重建没有可验证 overlap 观测的工具，并将高基数明细分页改为内存有界的 Cursor 接口。通过已有归档和 CPU-only 测试确认输出完整、分页稳定、权限正确；只有证据缺口能被现有采集路径补齐时才新增 Profile。Agent 优化试验是可选的、有前置门槛的末阶段工作。

## 约束

- PTOAnalyzer 仍只读、离线；不得从 Analyzer 调 PTOBench、Runner 或硬件。
- Lead 无 PTOAnalyzer 工具；不得把专属 Tuner 工具暴露给其它角色。
- 初始摘要仍汇总所有当前获准工具和全部可解析数据；不加 Top-K、静默截断或瓶颈结论。
- `limit` 默认 20、最大 50。页面明细的内部保留量不得随记录总量或页序号增长。
- 不引入数据库、持久索引、新工具服务器或为凑齐 3×3 而新增空工具。
- 修改限制在 `modes/ptomni/pto_analyzer/`；PTOBench 采集器缺字段时只记录最小需求并停在门槛，不擅自扩展采集器。

## Task 0：冻结 Stage 2 基线与当前调用面

**交付：** 本阶段基线记录，证明工作从 Stage 2 的实际实现出发。

- [x] 阅读本目录 README、Stage 2 设计/清单/回放和 `toolset.py`、`profile.py`、`diagnostics/`、测试。
- [x] 从 PTOBench 目录运行 `python -m unittest modes.ptomni.pto_analyzer.tests.test_toolset -v`；记录通过数与任何既有失败。
- [x] 从 Stage 2 JSON 报告复核 16 个 Profile、代表设备数、工具观测/明细计数和归档哈希记录；不得重写原报告。
- [x] 复核 `ToolSession.tool_specs`、`initial_summary`、`call()` 的所有仓库内调用点；记录旧 `offset` 是否有外部宿主依赖。
- [x] 锁定回放样本：最大明细 Profile、8 卡 Profile、双卡 Profile，先确认归档仍可用及身份完整。

**验收：** 基线来自当前代码和既有数据；不运行 NPU、Profile、evaluate 或 Agent。

## Task 1：收敛 `pipeline_overlap_check`

**默认决策：** 当前实现从 Dataflow 对外工具集合暂时撤下，不以“无 overlap 观测”的工具占用 Tuner 工具位。只有 Task 0 发现已有可配对运行时 Compute/MTE 区间时，才改为实现实际 overlap 统计后重新评估。

**文件：** `modes/ptomni/pto_analyzer/toolset.py`、`diagnostics/dataflow.py`、`tests/test_toolset.py`、`README.md`、`planning/README.md`。

- [x] 检查 16 份归档和 PTOBench 当前 combined 输出契约：是否有同一 Candidate/设备/窗口、公共时间单位/时钟锚点及可关联 task/block 的 Compute 与 MTE start/end 区间。
- [x] 写测试确保 Dataflow `tool_specs` 和 `initial_summary` 不暴露 `pipeline_overlap_check`，直接 `call` 返回未知 Tool 错误；其它角色工具和权限不变。
- [x] 搜索 `pipeline_overlap_check`、固定“每位 Tuner 三个 Tool”及九个 Tool 的文档引用，只更新必要的接口描述；文档明确当前 Dataflow 暂为两个有效 Tool。
- [x] 移除无调用方的 overlap 占位实现和仅被它使用的辅助代码；不影响共享 Parser 或其它 Tool。
- [x] 若数据审计发现真实可用区间，暂停本项删除路径，先建立能从原始区间复算 overlap 的 Fixture 和验证契约，再实现真正的统计；否则将缺失字段写入采集需求，不重跑无效 Case。

**验收：** 无空壳 overlap Tool 暴露给 Tuner；不存在为保持工具数对称而新增的替代工具；文档和权限表一致。

## Task 2：固定 Cursor 分页接口与失败行为

**外部接口：**

```python
session.call("compute_unit_activity", {"limit": 20})
session.call("compute_unit_activity", {"cursor": previous_next_cursor, "limit": 20})
```

响应中的 `details` 保留 `total_records` 与 `returned_records`，返回本页 `records` 和 `next_cursor`；最后一页 `next_cursor` 为 `null`。没有明细的 Tool 不接受分页参数。

- [x] 定义 Cursor 为不透明、版本化的 JSON-safe token，绑定 Tool、Tuner、Profile event 和 Candidate digest；Cursor 不含可由调用者指定的文件路径或授权字段。
- [x] 定义排序键和唯一 tie-breaker；同一 Profile/Tool 的逐页遍历必须稳定，即使原始排序键相同也不能漏项或重复。
- [x] 定义错误情况：损坏 Cursor、错误版本、另一 Profile/Tool/Tuner 的 Cursor、非字符串 Cursor、非法 `limit` 均在 Parser 运行前拒绝。
- [x] 保持完整摘要不依赖页面参数；第一页、后续页和 `initial_summary` 的 observations 与 source coverage 必须相同。
- [x] 在工具规格中将 `offset` 改成 `cursor`/`limit`，同步返回字段说明。

**验收：** 调用方只需传 Tool 名、可选 Cursor 和 limit；Cursor 无法改变归档选择或角色权限。

## Task 3：实现固定内存上界的分页选择器

**文件：** `modes/ptomni/pto_analyzer/diagnostics/common.py`、必要时 `toolset.py` 与九个诊断函数；测试位于 `tests/test_toolset.py`。

- [x] 先为内部分页选择器写 CPU-only 测试：100,000 条生成记录、limit=20/50 时，内部候选缓存峰值不超过 limit；请求末页或空末页时也遵守该上界。
- [x] 写重复排序键、乱序输入、多来源、末页、空输入、非法 Cursor、跨 Profile/Tool Cursor 的失败测试；逐页拼接结果与一次性稳定排序结果完全一致。
- [x] 只改共享分页实现和必要调用点；每个诊断继续负责提供原始 detail record 与稳定排序键，不重复实现 Cursor 编码/验证。
- [x] Cursor 用当前页最后一条的稳定复合键继续遍历；当前页以外只保留固定上限候选。全量摘要和 `total_records` 仍由完整扫描计算。
- [x] 保持 `ProfileArchive` 哈希/路径检查、只读行为和工具授权检查不变。
- [x] 记录大 Profile 第一页与连续少量页面的耗时和峰值内存；不在 CI 里加脆弱的墙钟阈值。若重复扫描成本确实阻碍 Agent 按需查阅，再单独评估最小的 Session 内部缓存，不预先引入磁盘索引。

**验收：** 每页至多保留 `limit` 条 detail 候选；全量统计与 Stage 2 一致；分页遍历稳定、无重无漏；原始文件不变。

## Task 4：跑完整离线回归并记录真实样本结果

- [x] 运行 PTOAnalyzer 全部 CPU-only 测试，并新增一个分页器定向测试命令，记录通过数。
- [x] 对当前仍可读取的 16 份归档重放新工具集；工具权限清单、全部摘要值和来源覆盖可复核。
- [x] 用 `deepseek_v3_2.deepseek_v3_2_prefill_back` 检查极高明细量，用 8 卡 `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` 和双卡 `deepseek_v4.prefill_layer_ep2_csa_demo` 检查多卡 rank 覆盖。
- [x] 对回放前后被索引原始文件核对 SHA-256；记录 profile 数、设备覆盖、工具数、分页性能数据及工具暂缓状态到 `reports/`。
- [x] 确认实际执行没有 PTOBench CLI/NPU 硬件动作；若只跑离线分析，硬件动作数记为 0。
- [x] 更新 PTOAnalyzer README 和 planning README：当前 Dataflow Tool 数、Cursor 用法、结果边界、Stage 3 报告链接。

**验收：** 测试与回放无未解释差异；全部 16 份可用归档 SHA-256 不变；摘要保持全量，不因移除工具或换分页接口静默缺数。

## Task 5：按证据缺口决定新 Profile

- [x] 将每个待回答问题写成“缺什么字段、字段来自哪个采集阶段、什么 Case 能产生它、预期如何验证”。
- [x] 先检查当前 PTOBench `combined` Profile 实现和现有 16 份索引，确认目标字段是否已由现有产物提供。
- [x] 若现有字段够用，复用 Profile，不新增运行。
- [x] 若字段存在于采集器且只缺 Case 形态，最多选一个针对性 Formal/Held-out Case。优先候选：2 卡 `deepseek_v4.prefill_layer_ep2_csa_demo`；其相同 Candidate/设备/软件版本 Profile 只有身份和配置核验通过后才可复用。
- [x] 若字段不由采集器输出，停止 Case 运行；形成单独的 PTOBench 最小采集需求，列出单位、身份键、时间锚点和覆盖验证，不假设换 Case 会增加字段。

**验收：** 每次新增 `profile` 都回答一个现有归档不能回答、且当前采集路径能够回答的问题；由 Runner/PTOBench 执行并记账。

## Task 6：可选 Basic Pi 优化轨迹

**状态：未启动。** Stage 3 只完成离线工具收敛；Basic Pi 需要独立预算、设备窗口和匹配的 Candidate/Profile/evaluate 链，留作后续阶段。

**启动条件：** 离线输出已通过人工复核，但仍无法判断工具呈现的事实是否会随 Candidate 优化变化。

- [ ] 从 Formal 清单选 1–2 个与具体证据缺口相符的 Case；优先候选为 2 卡 `deepseek_v4.prefill_layer_ep2_csa_demo` 和 8 卡 `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo`。
- [ ] 先校验起始 Profile 的 Candidate digest、Case、设备、编译配置及软件版本；一致则复用，不一致才为起始 Candidate 运行 `combined profile`。
- [ ] 通过 Basic Pi 运行真实优化；由 PTOBench/Runner 完成 Check、`profile` 和 `evaluate`。Profile 只在起始版本和代表性候选上采集，不要求每次编辑后都采集。
- [ ] 将每条轨迹绑定到 Candidate hash/diff、Profile event、工具摘要、Agent 决策和 `evaluate` 正确性/延迟结果。
- [ ] 人工核对 Analyzer 是否准确呈现变化与限制；不把相关变化表述成工具证明了因果关系。

**验收：** 得到可复核的 Candidate → Profile → Tool 输出 → Agent 改动 → evaluate 链；不把单条轨迹当作性能提升的统计证据。

## Task 7：可选 Agent 工具效用配对小试

**状态：未启动。** 仓库中没有可调用 PTOAnalyzer `ToolSession` 的正式 Host；没有创建模拟 Harness，也没有启动 Agent 或硬件试验。

**启动条件：** PTOmni/实验 Host 已能通过既有原生工具传输安全暴露 `ToolSession`；Stage 3 离线回归通过；用户授权的实验预算和硬件时段已落实。不得为此在 PTOAnalyzer 内创建第二个 Tool Server。

- [ ] 预先固定同一模型、Tuner 角色、Case 起始 Candidate、上下文、工具权限、优化预算和设备安排；唯一变量是 Agent 读取原始 Profile 还是读取 PTOAnalyzer 的完整摘要与分页 Tool。
- [ ] 先做两项配对可行性试验：Formal 2 卡 `deepseek_v4.prefill_layer_ep2_csa_demo` 与 Formal 8 卡 `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo`。两条件共享完全相同且身份有效的起始 Profile；优化候选的 Profile/evaluate 按相同上限计数。
- [ ] 若当前 Agent Harness 不能在保持同一 Tuner/角色行为下切换 Profile 信息接口，则不运行这组比较；记录接线缺口后结束本任务。
- [ ] 记录最终 `evaluate`、正确性、硬件动作、有效/无效/重复优化尝试、Profile/Analyzer 调用、Token 和墙钟时间；同一设备使用 PTOBench/Runner 锁，不并行争用。
- [ ] 将结果标为工程可行性小试；若需论文主张，再另行设计足够重复和统计功效的实验。

**验收：** 两个条件唯一差别是 Profile 信息接口，所有硬件动作有账本记录；未达到可比性时清楚停止，不用不公平结果下结论。

## Stage 3 完成标准

- Dataflow 不再暴露无法产生性能观测的 overlap 空壳；只有真实字段足以算 overlap 时才恢复。
- Cursor API 可完整遍历明细，内部页面候选数始终不超过 `limit`；摘要、来源覆盖和授权行为保持正确。
- CPU-only 测试、现存真实归档回放及只读哈希检查通过。
- 新增 Case 和 Agent 运行都有明确问题、样本理由、预算、工具条件和 PTOBench 记录；没有字段缺口时不为“覆盖齐全”而运行。
- README、角色权限和代码接口一致；PTOAnalyzer 仍不调用硬件、不承担 Lead 决策。
