# PTOAnalyzer Stage 5 任务清单

## 范围和约束

- 目标是把现有 PTOAnalyzer 接到真实 PTOmni Tuner Host，并完成一个小规模真实 Agent 验证。
- PTOAnalyzer 保持离线、只读、确定性；不得连接设备、调用 PTOBench 或启动 Agent。
- `profile`、`evaluate` 和设备分配只经 PTOmni Runner/PTOBench；所有动作按 PTOBench 原账本计数，并遵守设备租约及 PTOBench 锁。
- 只向本轮启用的 Compute、Dataflow 或 Schedule Tuner 暴露各自三个工具。Lead 不持有 Analyzer 或硬件工具；Assembler 不负责工具分发。其他角色也不获得 Analyzer Tool。
- 使用真实 PTOmni Host 的原生工具入口；不建立模拟 Host 或第二个 Tool Server。
- 不新增诊断工具、不改工具输出语义、不扩大 PTOBench Collector。本阶段开始时不新增 Case。
- 不覆盖工作区已有的 `evaluator/` 改动。

## Task 0：冻结 Stage 4 基线并确认 Host 前置条件

**参考：** `README.md`、PTOAnalyzer 当前 README、`reports/offline_replay_stage4_20261112.json`、PTOmni 架构设计。

- [ ] 核对 16 Profile、9 Tools、51 项 CPU 测试、5,492 个哈希核对、0 新硬件动作等 Stage 4 事实；本任务不重跑任何 NPU 工作。
- [ ] 搜索当前源码中的 PTOmni Host 原生 Tool 注册入口及 `ToolSession` 调用者。Stage 4 记录为没有；若当前仍没有，停止接线实现，将 PTOmni Host/Runner 最小实现列为单独前置事项，不用模拟 Host 代替。
- [ ] 对照 `modes/basic/runner.py` 和 `modes/cannbot/{runner.py,runtime.py,tools.py}` 的 PTOBench Mode 生命周期，找出 PTOmni 应接入的真实原生工具边界；只复用宿主接线模式，不复制其 Agent 工作流。
- [ ] 确认真实 Host 可从 Runner/Event Ledger 取得 `ProfileRef`，且能将一次 Tuner 调用绑定到对应 Round、Candidate 和 Profile event。

**验收：** Host 入口和 `ProfileRef` 来源已指明；或者明确记录 Host 缺失并停在本任务，不开始伪接线。

## Task 1：在真实 Host 原生入口暴露 Tuner 专属工具

**文件：** PTOmni Host 已有的原生工具注册/调用模块（Task 0 先确定实际路径）；必要时只调整 `modes/ptomni/pto_analyzer/` 中的适配接口。

**接口：** 使用既有 `open_toolset(tuner, profile_ref)`；Host 将返回的 `tool_specs`、`initial_summary` 和 `call(name, args)` 映射到其原生工具调用生命周期。

- [ ] 仅由可信 Host 注入角色身份和 ProfileRef；不允许模型参数选择 Tuner、Candidate、事件或任意文件路径。
- [ ] 只把调用方 Tuner 自己的三个 `tool_specs` 注册到当轮会话；工具调用只路由至同一 `ToolSession`。
- [ ] 把工具调用及结果关联到 Host 现有调用记录和当前 Profile event；不另建 Profile 身份或硬件动作账本。
- [ ] 保留 `initial_summary` 的完整事实，不用字符数截断；高基数明细沿用现有 Cursor 分页。
- [ ] 发生身份校验失败、无效 Cursor、工具错误或取消时，向 Host 返回可识别的工具错误，不降级成成功的空摘要。

**验收：** 真实 Tuner 可通过现有原生接口完成工具列表、摘要和按需明细调用；调用不触发 Runner/PTOBench/NPU。

## Task 2：验证权限、身份和只读性

**测试：** 放在真实 Host 现有 Tool 接入测试中，并保留 PTOAnalyzer 自身测试。

- [ ] Compute、Dataflow、Schedule 分别只能看到自己三个 Tool；`tile_memory_allocation_summary` 仅 Dataflow 可见。
- [ ] Lead、Assembler、Challenger、Curator 看不到 PTOAnalyzer 工具；Lead 也没有 Runner/PTOBench 调用权限。
- [ ] 伪造角色、错误 Profile event、Candidate digest 不匹配、跨 Tuner Cursor 和跨 Profile Cursor 均在读取错误数据前拒绝。
- [ ] 同一 ToolSession 的摘要和分页结果完整传递；分页无重复、无遗漏，不被 Host 静默截断。
- [ ] 测试前后归档 SHA-256 不变；Host 的 Analyzer 调用次数不会增加 PTOBench 硬件动作。

**验收：** Host 权限与 PTOAnalyzer 静态权限表一致；只读、身份绑定和 Cursor 合同通过 CPU 集成测试。

## Task 3：确定试点 Case 并核验可复用 Profile

**主试点：** `deepseek_v4.prefill_layer_ep2_csa_demo`（Formal，2 卡）。已有 Profile event `20261106T154609Z_a39baee9eb68:1`，Candidate digest 前缀 `70cb66c15b42`。

**条件扩展：** `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo`（Formal，8 卡）。已有 Profile event `20261106T155904Z_085104f9d3c2:1`，Candidate digest 前缀 `49ab191a3a5b`。

- [ ] 从 Formal 清单确认 Case 身份；通过完整源码/Candidate digest、输入与编译配置、设备、软件版本和归档校验决定是否复用 Stage 4 Profile。
- [ ] 核实对应 baseline `evaluate` 结果是否可比；缺少同环境 Clean Latency 时，列入 Runner 执行预算。
- [ ] 预先固定本次 Agent、Tuner 专长、优化问题、模型配置、代码上下文、PTOBench 预算、设备安排和结果指标。
- [ ] 只有试点结果显示需要多 rank/高基数验证时，才启用 CP8 Case；不因“覆盖更多”而提前运行。

**验收：** 形成一份 Case/ Candidate/ ProfileRef/ evaluate 对照表和预算草案；Profile 复用依据可审查。不得把同 Case 名称当作身份一致的证明。

## Task 4：准备最小受控 Agent 小试

- [ ] 使用同一底层 Agent（优先 Basic Pi）和同一 Formal Case，比较 Raw Profile 与 Analyzer Tool 两种输入条件。
- [ ] 两组从独立但相同的 Candidate 快照开始，固定模型版本、提示、知识、源码工具、优化时间和硬件动作预算；处理顺序交错或随机。
- [ ] Analyzer 组使用预先指定的 Tuner 身份，只见其专属工具。Raw 组可读同一 Profile 的原始文件；两组 Profile/Clean Benchmark 的调用许可和次数规则一致。
- [ ] 规定成本上限后，才由 Runner 执行。Agent 修改后的 Candidate 先过正确性检查，再决定是否 Profile；正式性能由 `evaluate` 给出。
- [ ] 记录完整 Agent 轨迹、工具调用、代码 Diff、正确性、Clean Latency、Profile 与 evaluate event、硬件动作、Token 和 wall-clock。

**验收：** 试验可复现，唯一实验变量是 Profiling 信息接口；Profile 与 Evaluate 都由 PTOBench/Runner 记录。单一匹配对只作为可行性小试，不作统计性或普适性结论。

## Task 5：执行小试并形成证据记录

**执行门槛：** 真实 PTOmni Host 已可用；Task 1–3 的 CPU 验收通过；用户启动实验时给定或确认硬件动作上限和设备窗口。

- [ ] 运行主试点 Case 的最小匹配实验；只有发现身份不匹配或确实需要新的候选 Profile 时，才增加 `combined profile`。
- [ ] 每个候选以 `evaluate` 检查正确性与 Clean Latency；失败、未完成和无可比结果都明确记录。
- [ ] 归档前后核对 Profile 原始产物未被 Analyzer 或 Host 改写；核对 PTOBench 账本中的实际动作与运行记录一致。
- [ ] 保存 Stage 5 结果报告，引用 Case、Candidate digest、Profile/evaluate event、tool-call trace 和计量口径。
- [ ] 根据结果只作下一步选择：停止、对主 Case 重复、或启用 CP8 扩展。若要声称 PTOAnalyzer 减少无效尝试，需要后续增加重复运行和足够的比较样本。

**验收：** 一条真实 Agent 优化轨迹可追溯到 Analyzer 调用与 PTOBench 结果；报告明确区分事实、观察和结论限制。

## Task 6：收尾验证

- [ ] 运行 `python -m unittest modes.ptomni.pto_analyzer.tests.test_toolset -v`。
- [ ] 运行 PTOmni Host 新增的 CPU 集成测试及该 Mode 的现有测试。
- [ ] 检查 9 个 Tool 的权限、PTOAnalyzer 文档、Host 接口、Stage 5 记录和实际运行日志一致。
- [ ] 核对离线测试没有 PTOBench/NPU 动作；真实小试所有动作由 Runner/PTOBench 计数。

**验收：** 工具接线通过 CPU 集成测试，真实小试记录完整；不得用“Agent 能调用工具”代替“工具提升了优化效果”的证据。
