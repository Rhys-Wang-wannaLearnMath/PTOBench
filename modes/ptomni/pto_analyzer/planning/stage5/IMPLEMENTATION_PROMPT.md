# PTOAnalyzer Stage 5 实施 Prompt

请按本目录的 [Stage 5 任务清单](TASK_CHECKLIST.md) 完成 `modes/basic_wiki-tool`，并在 Mode 的 CPU 验收通过后执行三个 Formal Case 的 Basic Pi vs Basic+Wiki+PTOAnalyzer 配对试验。

## 已有输入

- Formal 60 + Held-out 15 的 75 个 Case Profile 覆盖已完成。
- `modes/ptomni/pto_analyzer/` 的九个只读工具已构建并通过当前离线验收。
- Wiki 位于 `pto_wiki/pypto_wiki/`。
- Basic Pi 的实现位于 `modes/basic/`。

先核对这些输入和工作树，但不要重跑 75 个 Profile，不要把 PTOmni Host、Tuner 分权或完整 PTOmni 架构带入本任务。

Stage 5 的外部前置条件是 PTOBench 公共代码任务已实现并验收通用 Pi extension/socket 接口。开工时确认 `isolation.agent_command(...)` 已支持成对的 `pi_tool_extension`、`pi_tool_socket`，并能以固定沙箱路径和 `PTOBENCH_PI_TOOL_SOCKET` 向 Pi 暴露 Session-local Broker。若接口未完成或无验收记录，停止 Pi/Broker 接线和后续真实 Agent 试验，报告缺失项；不得自行修改 `evaluator/isolation.py`、公共 Runner 或该接口的测试。

## 必须遵守

- 新 Mode 尽可能沿用 Basic Pi 的 Prompt、Runner 生命周期、PTOBench 接口和预算行为；保持 Basic 对照 Mode 不变。
- 只给处理组增加 Wiki 和 PTOAnalyzer。九个 PTOAnalyzer 工具全部暴露给 Basic Pi，不按三个 Tuner 分配。
- 将 PTOAnalyzer 当前运行时代码复制到 `modes/basic_wiki-tool/pto_analyzer/`。接入适配只能放在新 Mode 或明确需要的通用 Pi 工具接线处；不要修改 PTOAnalyzer 本体来迎合 Basic，也不要用复制体反向覆盖本体。
- Wiki 只读、按需查阅；不全量塞进 Prompt，不新增 Wiki Agent 或复杂检索系统。
- 每次 Analyzer 工具调用都要记录，包括失败和取消；记录至少关联 Agent call ID、工具、参数、Profile identity、状态和结果/错误。
- Pi 接入采用原生 Pi extension + 本次 Agent Session 的本地 Unix socket + Python Tool Broker：extension 注册九个具名工具并转发 tool-call ID、参数和取消信号；Broker 校验授权 Profile ID，路由到该 Profile 对应的 Compute/Dataflow/Schedule `ToolSession`，将完整结果/错误返回给 Pi，并保存完整调用记录。Profile ID 只从 Runner 提供的当前 Case 授权索引中解析；Runner 接受新的 PTOBench Profile Event 后更新索引，Broker 不扫描任意目录或接受路径。
- 只消费 PTOBench 公共接口提供的 `pi_tool_extension`/`pi_tool_socket` seam；通用接口实现、公共隔离验收及 CANNBot 兼容测试由独立任务负责人负责。Stage 5 自己必须通过 CPU 集成测试证明 PTOAnalyzer extension 按 Pi 的 `registerTool` API 注册了全部九个工具，且调用经 Session socket 和 Broker 后能将结果返回给 Pi；可沿用 CANNBot 的 Extension/socket 验证方式，但不采用其 Task/Skill 工作流。测试使用 `registerTool` 测试桩和 CPU fixtures，不启动真实 LLM 或 NPU。
- Pi 已发起但在 extension 分发前取消的调用也必须经 Broker 留下取消记录，不能静默丢弃。
- Runner 在 Agent 启动前开启本次 Session 的 Broker，Agent 正常退出、启动失败或取消时都关闭 Broker 并清理 socket。通过已验收的 PTOBench 通用接口挂载 Broker-owned extension 和 Session-local socket（固定映射为 `/ptobench/pi-tool.sock`，并通过 `PTOBENCH_PI_TOOL_SOCKET` 提供）；新 extension 需与 File Guard、Provider 和现有 Pi extensions 并存。Stage 5 不修改公共隔离代码。
- Pi extension 仅负责工具注册和传输，不增加 Agent 角色、Candidate 写入、额外 PTOBench CLI 权限、真机实验或设备锁操作。
- PTOAnalyzer 不运行 `profile`/`evaluate`，不操作设备锁、不计硬件动作。所有 NPU 动作仍通过 PTOBench Runner/Event Ledger。
- 不增加自动 diff 或 baseline 选择功能。Agent 可分别读取授权的 Profile 并自行比较。
- 配对试验固定 Basic 与处理组的模型、Case、起始 Candidate、软件/硬件条件和预算；三组配对共六个 Agent Session。只有在选中 Case 的 Profile 身份确实不匹配时，才通过 PTOBench 正常补足该试验输入。
- 将 Wiki 与 Analyzer 作为组合消融解释；不声称单独识别两者各自的因果贡献，也不把小样本写成统计显著或普遍收益。

## 交付

完成任务清单里的 Mode、复制体适配、Wiki 快照、CPU 测试、三组真实配对、调用记录与 Stage 5 报告。最终说明通用接口的前置验收依据、变更文件、验证命令和结果、Pi 侧九工具可见/可调用的证据、六个 Session 的追踪位置、工具调用覆盖、Profile/evaluate 硬件动作数及结论边界。保留工作区中与 Stage 5 无关的改动。
