# PTOAnalyzer Stage 5：PTOmni 接线与真实优化小试

> 状态：规划中。本阶段只制定后续工作，不启动 PTOBench、NPU 或 Agent 实验。

## 目标

让实际运行的 PTOmni Tuner 能通过现有原生工具接口，读取自己获准的 PTOAnalyzer Tools；随后用一个已有 Formal Case 做小规模真实 Agent 验证，确认工具能进入优化轨迹，并初步观察它是否帮助 Agent 使用 Profiling 证据。

Stage 5 的顺序是：**真实 Host 前置检查 → Tuner 工具接线 → CPU 集成验收 → Case 身份核验 → 真实 Agent 小试 → 决定是否扩展 Case**。

## Stage 4 检查结论

Stage 4 已完成工具修缮和离线验收，证据见 [回放报告](../../reports/offline_replay_stage4_20261112.json)：

- 16 份现有 `combined` Profile，覆盖 1、2、4、8 卡；当前接口有 9 个只读 Tool。
- 51 项 PTOAnalyzer CPU 测试通过。
- 5,492 个被索引归档产物回放前后 SHA-256 相同；新增硬件动作数为 0。
- 最大单卡依赖图有 453,375 条明细；同一 Session 后续 Cursor 页不再重复构建完整图。
- `tile_memory_allocation_summary` 已用现有 AllocateMemoryAddr 产物独立复算。
- 回放报告明确记录：仓库中未发现 PTOmni Host 调用 `ToolSession`；Basic Pi / Agent 试验没有启动。

因此 Stage 4 的工具实现可以作为 Stage 5 起点，但还没有证据说明 Tuner 实际收到工具结果或在真实优化中使用了这些结果。现有 Profile 是 Candidate 的单点快照，不包含优化轨迹，也不提供 Clean Latency。

## 需要什么 Case 和实测？

初期不需要新增 Case，也不需要为了重建 Parser 再跑一批普通 Profile。已有 Formal Profile 可以作为候选基线，但**同名 Case 不等于同一个 Candidate**：只有源码 digest、输入/编译配置、设备、软件版本及 Profile 事件身份都匹配，才复用。

| 优先级 | Case | 现有数据 | 用途 |
|---|---|---|---|
| 主试点 | `deepseek_v4.prefill_layer_ep2_csa_demo`（Formal，2 卡） | Stage 4 回放记录事件 `20261106T154609Z_a39baee9eb68:1`；Candidate SHA-256 前缀 `70cb66c15b42` | 第一轮优先用它：覆盖多种算子与多卡 Profile，已有完整工具结果，硬件成本比 8 卡 Case 更可控。 |
| 条件扩展 | `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo`（Formal，8 卡） | Stage 4 回放记录事件 `20261106T155904Z_085104f9d3c2:1`；Candidate SHA-256 前缀 `49ab191a3a5b` | 仅当第一轮表明有必要验证多 rank 调度、分页或资源协调时再用；Stage 4 已有 8 卡基线 Profile。 |

Case 来源为 `case_selection/formal_60/cases.csv`。优先选择一个 Case 完成接线和可行性验证，避免一开始铺开多 Case 实验矩阵。若主试点身份检查失败，先对**同一 Case 的基线 Candidate**补一次 PTOBench `combined profile`；若没有可复用的 Clean Latency 基线，再按相同评测配置补一次 `evaluate`。对通过正确性检查的代表性优化 Candidate 采一次 Profile，并用 `evaluate` 验证其正式延迟。所有实际硬件动作由 Runner/PTOBench 执行并计入原账本。

真实 Agent 优化轨迹是必要的：它能提供“工具事实—Agent 假设—代码改动—检查与性能结果”的关联，判断 Agent 是否能够消费工具结果。仅有新的 PTOBench Profile 不能回答这个问题。第一轮可由 Basic Pi 作为固定优化 Agent；它是 Analyzer 工具效用的小试，不等同于 PTOmni Adaptive Team 的完整方法评估。

## 前置条件：真实 PTOmni Host

当前 `modes/ptomni/` 只有 PTOAnalyzer，没有 PTOmni Host/Runner 调用者。Stage 5 不应在 Analyzer 内造一个模拟 Host，也不应新增第二个 Tool Server。实施接线前，PTOmni Runtime 必须提供其真实的原生工具注册和调用入口；如果它仍不存在，先把 Host/Runner 作为独立前置工作完成，再继续本阶段。

PTOBench 已有的 Agent Mode 生命周期可作为接线参考：`modes/basic/runner.py` 展示基本 Pi Mode；`modes/cannbot/runner.py`、`runtime.py` 和 `tools.py` 展示真实的原生工具注册、调用和记录。Stage 5 只借鉴这些现有宿主接口的接法，不搬入 CANNBot 工作流，也不复制出第二套服务。

接线沿用 PTOAnalyzer 现有接口：Host 从 Runner/Event Ledger 获得确切的 `ProfileRef`，按当前 Tuner 身份打开 `open_toolset(tuner, profile_ref)`，只传递该 Tuner 的 `tool_specs`、`initial_summary` 和 `call(name, args)`。Agent 不能传任意 Profile 路径。不得向 Lead 增加 Analyzer 或实验 Tool；Assembler 不负责分发 Tool。

## 实验设计与结论边界

最小效用小试采用同一底层 Agent（优先 Basic Pi）、同一 Case、起始 Candidate、模型配置、知识和代码工具、硬件预算及评测设置，比较两个输入条件：

1. **Raw 条件**：Agent 读取同一份受保护 Profile 的原始数据。
2. **Analyzer 条件**：对应 Tuner 只通过自己的 PTOAnalyzer Tools 读取同一 Profile。

两组使用互相隔离的起始工作区，运行次序交错或随机，实验预算与 PTOBench 硬件动作上限预先固定。记录每个 Agent 的假设、代码 diff、工具调用、Profile / evaluate event、正确性、Clean Latency、硬件动作、Token 和 wall-clock。若只有一对运行，只把结果当作接线与可行性信号，不声称统计显著或证明泛化；只有观察到稳定差异后，才追加重复种子或第二个 Case。

## 交付物

- PTOmni Host 的 Tuner 专属 PTOAnalyzer 工具接线及权限验证；接线改动落在真实 Host 所属目录。
- CPU 集成测试：身份绑定、工具可见性、完整摘要与 Cursor 调用、只读归档和失败路径。
- 一个 Formal Case 的 Agent 优化轨迹；复用条件匹配的基线 Profile，只在需要时补采基线或候选 Profile。
- 一份 Stage 5 结果记录，区分 Host/工具接线事实、Agent 过程和 PTOBench 正式结果；不把 Profile 计数写成性能结论。
- 基于试点证据的下一步决定：结束、补充同 Case 重复，或启用 8 卡条件扩展。

## 文件

- [任务清单](TASK_CHECKLIST.md)：有前置门槛的实施顺序与验收条件。
- [实施 Prompt](IMPLEMENTATION_PROMPT.md)：后续启动 Stage 5 时可直接使用的任务说明。
- [Stage 4 验收记录](../stage4/README.md)：前一阶段的边界、修缮和证据。
- [PTOAnalyzer 接口](../../README.md)：当前 ToolSession、角色权限和只读约束。
- [PTOmni 架构设计](../../../../PTOmni/design_docs/PTOmni_架构设计_2026-11-03.md)：Tuner、Runner、Lead 与 Assembler 的职责边界。
