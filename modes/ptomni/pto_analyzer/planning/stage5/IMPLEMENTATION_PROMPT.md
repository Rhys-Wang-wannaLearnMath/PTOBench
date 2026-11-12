# PTOAnalyzer Stage 5 实施 Prompt

请按本目录的 [Stage 5 任务清单](TASK_CHECKLIST.md) 推进 PTOAnalyzer 的 PTOmni Host 接线和最小 Agent 效用小试。Stage 4 已完成，本轮不要重做其数据解析或工具设计。

## 开始前

1. 阅读本目录 `README.md`、`TASK_CHECKLIST.md`、PTOAnalyzer `README.md`、Stage 4 清单和 [Stage 4 回放报告](../../reports/offline_replay_stage4_20261112.json)。
2. 检查当前工作树和 Stage 4 后的 Host 状态；保留所有无关改动。
3. 先完成 Task 0。当前已知仓库中没有 PTOmni Host 调用 `ToolSession`；若仍然没有真实原生 Host 入口，停止接线代码，不创建假 Host、独立 MCP Server 或 Analyzer 内的实验 Harness。
4. 本 Prompt 供后续实施使用。当前规划回合不执行 PTOBench、NPU 或 Agent 实验。真实硬件小试必须在对应执行任务中先确认预算、设备窗口和 Runner/PTOBench 锁。

PTOBench 当前 `modes/basic/runner.py` 与 `modes/cannbot/runner.py`、`runtime.py`、`tools.py` 是现有 Mode/原生工具生命周期的参考。沿用真实宿主接口，不移植 CANNBot 的优化工作流。

## 实施要求

- PTOAnalyzer 仍为离线、只读模块。用可信 Runner/Event Ledger 提供的 `ProfileRef` 调用现有 `open_toolset(tuner, profile_ref)`；禁止 Agent 传任意路径、Profile event 或 Tuner 权限。
- 在真实 PTOmni Host 的既有原生工具注册和调用入口中接线。每次只向获准 Tuner 注册其三个工具，准确转发完整 `initial_summary` 与 `call(name, args)`；高基数明细使用既有 Cursor。Lead、Assembler、Challenger 和 Curator 不获得 Analyzer 工具；Lead 不获得 Runner/PTOBench 工具。
- 不改变九项 Tool 的职责和输出契约，不新增 Analyzer 诊断，不改 PTOBench Collector，不重写原 Profile，不引入第二套硬件动作账本。
- Stage 4 的 16 个 Profile 只作候选基线。复用前核对 Candidate digest、输入/编译配置、设备、软件版本和归档身份；只比较同身份、同设置的数据。
- 首选 Formal Case `deepseek_v4.prefill_layer_ep2_csa_demo`（2 卡）。8 卡 Formal Case `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` 只作为有证据需求时的扩展。
- 最小 Agent 试点使用同一底层 Agent 和同一 Case，比较 Raw Profile 与 PTOAnalyzer Tool 两种信息接口。隔离起始工作区，固定模型、提示、知识、代码工具、时间与硬件预算；所有 Profile/evaluate 由 Runner/PTOBench 执行并计数。
- 候选先经过正确性检查；Clean Latency 以 `evaluate` 为准。Profile 事实不得代替正式性能结论。一对试验只用于验证接线和可行性，不宣称统计显著或普遍有效。

## 验收与交付

- 完成 Host 原生工具接线和 Tuner 身份权限测试；错误身份和无效 Cursor 在读取错误数据前拒绝。
- PTOAnalyzer 当前 CPU 测试通过，Host 集成测试通过；测试确认摘要/分页结果完整、原始 Profile 只读、Analyzer 调用不启动硬件动作。
- 在用户单独启动真实 Agent 试验并确认预算后，生成一条可追溯的 Formal Case 轨迹，记录 Candidate、Profile/evaluate event、Tool calls、diff、正确性、Clean Latency、资源成本和比较限制。
- 更新 Stage 5 清单与结果记录；不将“工具已注册”或单次成功案例写成性能提升结论。
