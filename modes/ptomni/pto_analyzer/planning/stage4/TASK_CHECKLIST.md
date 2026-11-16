# PTOAnalyzer Stage 4 任务清单

> 状态：Stage 4 实现与 CPU-only 离线验收已完成。以下清单记录已完成的离线修缮、回放证据，以及因前置条件不满足而保留到后续阶段的 Host/Agent 工作。

## 约束

- 修改范围以 `modes/ptomni/pto_analyzer/` 为主；仅为校正 Stage 3 完成状态，修改 `planning/stage3/README.md` 与 `planning/stage3/TASK_CHECKLIST.md`。
- 保留 Stage 3 的八个工具及其 Tuner 权限、全量摘要、观测口径、来源覆盖、Cursor 身份绑定和原始归档只读性；新增一个仅供 Dataflow Tuner 使用的 `tile_memory_allocation_summary` 后，接口共九个工具（Compute/Dataflow/Schedule 各三个）。
- `pipeline_overlap_check` 继续从接口移除；不恢复 overlap 推算，也不因此改动其余八个工具。
- Analyzer 不运行 PTOBench/Runner/NPU，不获得硬件预算，不给 Lead 增加工具。
- 不增加第三方依赖、数据库、持久索引或第二个 Tool Server。
- 没有用户另行授权时，不启动 Basic Pi 或 Agent 硬件实验。

## Task 0：冻结基线并校正 Stage 3 状态

**文件：**

- `modes/ptomni/pto_analyzer/planning/stage3/README.md`
- `modes/ptomni/pto_analyzer/planning/stage3/TASK_CHECKLIST.md`
- `modes/ptomni/pto_analyzer/reports/offline_replay_stage3_20261108.json`（只读）

- [x] 对照 Stage 3 报告核对实现/离线验收完成状态、8 个工具、16 份 Profile、128 次摘要/分页调用、5,492 个文件哈希和 0 硬件动作。
- [x] 将 Stage 3 README 顶部“规划完成，尚未执行”改为与验收记录一致的完成状态。
- [x] 按报告更新 Stage 3 清单中已完成的离线任务；Basic Pi 与 Agent 配对试验保持未启动，并标明它们是可选项。
- [x] 保留 Stage 2 observation 名称/值未归档、无法逐项历史比较的限制；不得将“计数和来源覆盖一致”写成“所有观测值一致”。
- [x] 记录 Stage 3 历史基线 43 项测试通过；Stage 4 最新 CPU 回归为 51 项通过，新增行为测试没有通过放宽断言掩盖回归。

**验收：** Stage 3 README、清单和回放报告状态一致；Stage 4 不重写历史报告。

## Task 1：有界保存大文件解析错误

**文件：**

- `modes/ptomni/pto_analyzer/diagnostics/common.py`
- `modes/ptomni/pto_analyzer/diagnostics/dataflow.py`
- `modes/ptomni/pto_analyzer/tests/test_toolset.py`

- [x] 为 JSONL 与 CSV 的大量坏记录建立 CPU-only 测试：有效记录仍正常统计；拒绝总数精确；返回的问题示例数有固定上限。
- [x] 让解析器或共享记录汇总器只保留有限错误示例，并在结果中明确报告错误总数和省略的示例数。
- [x] 让问题示例保留对应来源路径和行/记录号；空文件、缺字段、损坏记录、无观测仍然可区分。
- [x] 验证有效输入的 observation、source coverage 和计数与 Stage 3 行为一致。
- [x] 将验收范围写清为“错误示例存储有界、错误总数精确”；文件字节读取、哈希校验和解析本身仍可能随输入大小增加，单独记录其内存成本，不声称整个分析流程为常量内存。

**验收：** 即使每一行都格式错误，问题文本内存也不会随文件行数线性增长；错误总量仍可准确获知。

## Task 2：大文件单遍处理和低风险重复工作

**文件：**

- `modes/ptomni/pto_analyzer/profile.py`
- `modes/ptomni/pto_analyzer/diagnostics/common.py`
- `modes/ptomni/pto_analyzer/diagnostics/compute.py`
- `modes/ptomni/pto_analyzer/diagnostics/dataflow.py`
- `modes/ptomni/pto_analyzer/tests/test_toolset.py`

- [x] 用逐行迭代读取 `events.jsonl`，保留非法 JSON、目标 event 缺失和重复 event 的原失败语义，不使用 `read_text().splitlines()` 同时保留整份文本和行列表。
- [x] 将 `csv_records` 的“先扫描计数、再重新读 CSV”改为单次解析，不缓存全部 PMU 行；解析结束仍要报告 seen/parsed/rejected、缺少列和有限错误示例。
- [x] 说明并测量 `ProfileArchive` 当前保留原始文件字节的成本；移除 PMU 行对象缓存不等于不再将源文件整体载入内存。
- [x] 用 Scope 路径到来源记录的字典索引替代每个字段都线性搜索 `scope_sources`；保持同样的来源引用和 Scope 聚合。
- [x] 合并 `compute_task_share` 两个未关联 PMU 记录分支的重复明细构造，同时保留具体原因计数与通用格式错误语义。
- [x] 保持 CSV 的 100,000 行压力测试，并验证峰值内存不随“保留所有 CSV 行”增长。
- [x] 测试目标 event 在账本开头、中间、结尾、重复和账本包含非法行时的结果。

**验收：** 同样输入只解析一次 PMU CSV；事件账本不再被拆成整文件行列表；观测值和来源覆盖没有变化。

## Task 3：依赖图摘要与 Cursor 分页扩展性

**文件：**

- `modes/ptomni/pto_analyzer/profile.py`（仅在需要受保护的按需读取时）
- `modes/ptomni/pto_analyzer/diagnostics/schedule.py`
- `modes/ptomni/pto_analyzer/diagnostics/common.py`
- `modes/ptomni/pto_analyzer/toolset.py`
- `modes/ptomni/pto_analyzer/tests/test_toolset.py`

- [x] 先以 Stage 3 的三个归档样本建立当前基线：最大单卡 `deepseek_v3_2.deepseek_v3_2_prefill_back`、双卡 `deepseek_v4.prefill_layer_ep2_csa_demo`、8 卡 `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo`。分别记录摘要、头三页和末页的计数/排序、耗时与峰值内存。
- [x] 通过独立 profiling 确认峰值主要来自 JSON 文本、完整解码后的 task/edge 对象，还是图统计的集合/邻接结构；不要只优化页面候选堆，因为它已被限制为 `limit` 条。
- [x] 实现能保持当前所有 graph observations、detail 字段和稳定排序的最小解析/分页方案：一份 `ToolSession` 的后续 Cursor 页不得再次解码并重建完整依赖图；不能为此长期保存整套展开后的 task/edge 对象。
- [x] 如需要按原始字节位置或轻量索引分页，Cursor 仍须绑定 Profile event、Candidate、Tool 和 Tuner；不得将位置解释成路径或授权字段。
- [x] 用最大单卡真实归档核对优化前后 10 个依赖图摘要字段、453,375 条明细计数、分页首/中/末页及完整遍历；多卡样本逐 rank 核对来源覆盖。
- [x] 记录与 Stage 3 同口径的单页耗时、三页总耗时、进程 RSS 和 `tracemalloc` 峰值。不得添加脆弱的 CI 墙钟断言；CPU fixture 应覆盖游标恢复、排序、重复键、异常输入和无重无漏。
- [x] 如果在不增加依赖、不降低结果完整性且不持有整图对象的前提下无法满足资源目标，停止扩大实现，记录实测瓶颈与可选设计供单独决策；不要悄悄改为 Top-K 或截断摘要。

**验收：** 同一 Session 连续翻页不重复构建全量图；完整观察和明细结果与基线语义一致；最大 Case 的时间/内存变化有实测记录。目标建议为把临时解析峰值降到 512 MiB 以下，若代码结构或真实输入无法达到，必须带数据说明，不以删减指标达标。

## Task 4：让分页测试验证外部行为

**文件：**

- `modes/ptomni/pto_analyzer/tests/test_toolset.py`
- 必要时 `modes/ptomni/pto_analyzer/diagnostics/common.py`

- [x] 保留 100,000 条记录和 `limit=20/50` 的候选内存上界验证，但删除对 `DetailRecords._heap` 的直接读取。
- [x] 以黑盒压力观测验证固定页上限（例如生成大输入并测量受控的 Python 分配峰值）；阈值应由基线实验校准并留有环境余量。
- [x] 删除要求 Cursor 使用某种 Base64 canonical spelling 的测试；保留随机篡改、非法 token、错误 Profile/Tool/Tuner 绑定均在诊断运行前拒绝的测试。
- [x] 保留所有工具全量遍历无重复/无遗漏、空末页、重复排序键和摘要不随页参数变化的测试。

**验收：** 测试覆盖分页合同和资源边界，不依赖具体堆变量或 Cursor 序列化拼写。

## Task 5：新增 `tile_memory_allocation_summary`

**文件：**

- `modes/ptomni/pto_analyzer/diagnostics/dataflow.py`
- `modes/ptomni/pto_analyzer/diagnostics/common.py`（仅在需要共享来源/记录辅助函数时）
- `modes/ptomni/pto_analyzer/toolset.py`
- `modes/ptomni/pto_analyzer/tests/test_toolset.py`
- `modes/ptomni/pto_analyzer/README.md`
- `evaluator/profile_modes.py` and `pto_wiki/pypto_wiki/topics/api-and-operations-reference/source-02_pypto--static_api--pypto.language.tile.md` (read-only references)

- [x] 先审计 Stage 3 的 16 份 Profile 中实际归档的 AllocateMemoryAddr 输出；确认文件形式、调用语法、函数边界、memory space、分配大小单位和源码位置。不得先假设文本格式或编写宽泛正则。
- [x] 以 `evaluator/profile_modes.py` 规定的归档产物为入口，使用 PTOBench 已索引的 `memory_after_AllocateMemoryAddr` 报告或 `*_after_AllocateMemoryAddr.py`；对 `tile.alloc` 等打印语法按 PyPTO API 文档核实 `size` 的字节语义。保持 Profile 身份、来源路径和内容哈希校验，不从 Candidate 工作树或任意路径读取。
- [x] 仅解析真实格式中字段明确且语义经核实的编译器分配记录。按产物、函数（来源提供时）和 memory space 分组，完整报告 `allocation_count` 与 `requested_bytes_sum`，并保留 artifact/source 引用；若来源不含函数边界，明确标为产物级汇总，不推断函数归属。如果同一格式出现不同 allocation op，保留其类型或分别计数。
- [x] 返回完整分组摘要，不返回逐 allocation 明细、不做 Top-K、不添加分页接口；每个值仍须有单位和来源范围。
- [x] `requested_bytes_sum` 只能表示该分组中显式记录的分配大小之和。不得跨产物或函数求和并称为峰值，不据此推断运行时峰值、容量压力、生命周期重叠、spill 或瓶颈。
- [x] 将新工具注册为 Dataflow 专属只读 Tool；保留 `buffer_pressure_check` 的现有职责（MemoryReuse 编译提示与 Scope 原始计数），不修改或重命名其既有观测。Lead 不获得此工具。
- [x] 新增 CPU Fixture 测试覆盖多个 memory space、多个函数、不同 allocation op（若真实输入含有）、字节单位、确定性排序，以及缺失/空/不支持/损坏记录的区分；来源引用须能定位到原始产物和行或记录。
- [x] 用真实 Profile 产物对 Fixture 和解析结果做离线核对；16 份归档均有可验证的 AllocateMemoryAddr Python 样本，因此没有触发缺失样本阻塞，也没有启动新 Profile 或修改 PTOBench Collector。

**验收：** Dataflow 获得一项基于真实编译分配产物的客观摘要；输出可追溯且不包含运行时压力或性能结论。原有八个工具的权限和行为不变，`pipeline_overlap_check` 不重新开放。

## Task 6：离线完整回归与记录

**文件：**

- `modes/ptomni/pto_analyzer/reports/offline_replay_stage4_<date>.json`
- `modes/ptomni/pto_analyzer/README.md`
- `modes/ptomni/pto_analyzer/planning/README.md`

- [x] 运行 `python -m unittest modes.ptomni.pto_analyzer.tests.test_toolset -v`。
- [x] 对已有 16 个 Profile 重放九个工具，记录工具权限、完整 summary facts、source coverage、分页数和可复核摘要 digest；新工具在 16 份归档中均读取到受保护的真实分配产物。
- [x] 回放前后核对所有 5,492 个被索引产物哈希；离线工作记录 `ptobench_cli_invoked=false`、`npu_accessed=false`、`hardware_actions_added=0`。
- [x] 比较 Stage 4 与 Stage 3 当前结果；对 `buffer_pressure_check` 等已知变化逐字段解释，不因 Stage 2 缺少值快照而声称逐值兼容。
- [x] 更新 PTOAnalyzer 当前接口文档和 planning 索引，确保权限表列出九个 Tool，说明新增工具的静态分配语义、`buffer_pressure_check` 的独立范围、Stage 3/4 状态和 Host 缺口；注明 overlap 仍未实现。

**验收：** CPU 测试、16 份归档回放、九个工具的摘要/分页完整性和归档哈希检查通过；无新增硬件动作。

## Task 7：PTOmni Host 接线（有前置条件）

**状态：未启动。** 仓库中没有当前 PTOmni Host 调用 `ToolSession` 的入口；所需 seam 和缺口记录在 Stage 4 回放报告中，不创建模拟 Host 或第二个 Tool Server。

**前置条件：** 仓库中存在明确的当前 PTOmni Host / Pi 工具注册入口。若只有设计文档而没有可运行宿主，本任务停止在接口契约，不创建新的 Agent Harness 或 Tool Server。

- [ ] 核实 Host 提供给本轮 Tuner 的 ProfileRef 来源、候选身份和工具调用生命周期。
- [ ] 在 Host 现有原生工具传输处建立薄适配：仅把 `tool_specs` 与 `call(name, args)` 暴露给当轮授权的 Compute/Dataflow/Schedule Tuner，并验证 `tile_memory_allocation_summary` 仅向 Dataflow 开放。
- [ ] 验证 Lead、Challenger、Assembler 均不能取得 Tuner 专属 Analyzer 工具；Host 不因 Analyzer 调用启动 profile/evaluate。
- [ ] 在 Host 集成测试中用 fixture Profile 验证权限隔离、Cursor 连续分页、错误传递和调用只读；NPU/Runner 调用次数为 0。
- [ ] 若找不到当前 Host 实现，将定位结果与所需最小 Host seam 记录为 PTOmni 主架构依赖，Stage 4 不以模拟 Host 完成验收。

## 可选 Task 8：Agent 数据敏感性与工具效用试验

**状态：未启动。** 该项需要 Host seam、独立预算、匹配的 Candidate/Profile/evaluate 链和用户授权的硬件窗口，属于后续阶段，不是 Stage 4 完成条件。

此项不是 Stage 4 离线修缮的完成条件。试验需要独立预算/设备窗口授权，并始终通过 PTOmni Host、Runner 和 PTOBench 的既有锁与账本。

- [ ] Basic Pi 单轨迹只回答工具呈现的观测是否随真实 Candidate 变化；优先考虑 Formal 2 卡 `deepseek_v4.prefill_layer_ep2_csa_demo` 与 Formal 8 卡 `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo`，先检查并复用完全匹配的起始 Profile。
- [ ] 只有 Task 7 完成后，才考虑配对“读取原始 Profile”与“读取 PTOAnalyzer 工具”的小试；固定模型、Tuner、起始 Candidate、其他上下文和硬件预算，唯一改变分析信息接口。
- [ ] 记录所有 Candidate hash/diff、Profile event、evaluate 正确性/延迟、硬件动作、有效/无效/重复尝试、Token 和墙钟时间。
- [ ] 将结果标为工程可行性证据；单次或少量配对试验不用于论文级因果或统计主张。

## Stage 4 完成标准

- Stage 3 文档状态与报告一致。
- 错误样例存储有界但错误总数精确；PMU CSV 单遍处理，事件账本逐行读取；依赖图分页不重复构造全图。
- Dataflow 共三个工具，新增的 `tile_memory_allocation_summary` 汇总可追溯的静态分配请求，不替代 `buffer_pressure_check`，也不输出运行时峰值或瓶颈结论。
- 依赖图性能/内存以真实单卡和多卡归档数据说明，结果完整性不变。
- 全部 CPU-only 测试和离线回放通过，5,492 个被索引归档产物哈希一致，硬件动作数为 0。
- Host 接线只在真实 Host 存在时完成；Agent 试验可保持未启动。
- `pipeline_overlap_check` 继续停用；不以聚合计数伪造区间重叠。
