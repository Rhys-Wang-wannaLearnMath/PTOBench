# PTOAnalyzer Stage 5：Basic + Wiki + PTOAnalyzer 消融

> 状态：已完成。Stage 5 已实现 `modes/basic_wiki-tool`，完成 75 个 Formal/Held-out Profile 覆盖和六个真实配对 Session。结果见 [配对实验报告](../../../../basic_wiki-tool/reports/stage5/paired_trials.md)、[覆盖审计](../../../../basic_wiki-tool/reports/stage5/evidence/selected_coverage_audit.json) 和 [验收审计](../../../../basic_wiki-tool/reports/stage5/acceptance_audit.md)。

## 目标

在尽可能保留 Basic Pi 流程的前提下，新增一个对照 Mode：`basic_wiki-tool`。它在 Basic Pi 的工作流上只增加两项能力：

1. 读取知识库 `pto_wiki/pypto_wiki`。
2. 按需调用 PTOAnalyzer 的全部九个只读工具。

与同 Case、同初始 Candidate、同模型和同预算的 Basic Pi 对照，观察组合加入 Wiki 与 PTOAnalyzer 后的实际 Agent 行为和优化结果。该实验是 **Wiki + Analyzer 的组合消融**；结果不能分别归因于 Wiki 或 PTOAnalyzer。

## 已完成的前置条件

Stage 5 不重新安排以下工作：

- Formal 60 + Held-out 15 的 75 个 Case 已完成 `combined` Profile 覆盖。
- PTOAnalyzer 当前九个工具及离线验收已准备好使用。
- `pto_wiki/pypto_wiki` 已作为知识库存在。

实现开始时只需核对这些输入的版本、归档身份和可访问性；不重跑 75 个 Profile，也不把 PTOAnalyzer 的本体修缮混进 Mode 接入。若某个配对试验所需 Profile 的 Candidate 身份不匹配，单独记录并按 PTOBench 正常流程补足该试验输入，不因此重跑全量 Case。

## 接入状态与源码保护范围

PTOBench 通用 Pi extension/socket 接口已接入并通过公共 Pi 接口测试；`basic_wiki-tool` 已使用该接口完成真实 Agent 配对实验。Stage 5 没有修改 PTOBench 公共隔离层、Runner 或通用接口测试。

**本设计不要求 PTOBench 对本次选中的 Mode 源码增加 Session 保护或完整性检查。** 这是 Stage 5 的范围决定，不是待完成的外部前置条件；实验依照正常 PTOBench Session 流程运行，并记录 Profile、Event、Candidate 和运行结果。报告不声称这些 Session 保护了所选 Mode 源码。

## 架构与边界

`basic_wiki-tool` 沿用 [Basic Mode](../../../../basic/README.md) 的 `preflight → prepare → run → collect` 生命周期、Pi 配置和 PTOBench Runner。Basic 本身保持不变。

将 PTOAnalyzer 的当前运行时代码复制到 `modes/basic_wiki-tool/pto_analyzer/`，由 Mode 内的轻量适配层针对每个授权 Profile 创建/复用 Compute、Dataflow、Schedule 三个单 Profile `ToolSession`，并把九个工具全部注册给 Basic Pi。这里不采用 PTOmni 的 Tuner 分配规则，也不运行工具的自动 `initial_summary`；Agent 通过实际工具调用按需获取结果。

Pi 接入使用 **原生 Pi extension + 本次 Agent Session 的本地 Unix socket + Python Tool Broker**。`prepare` 将 extension 放入 Broker 管理的 `agent/pi-tools`；`run` 启动 Broker，再通过已经验收的 PTOBench 通用 Pi extension/socket 接口启动 Agent；Agent 结束或被取消时关闭 Broker。Extension 用 Pi 的 `registerTool` 注册九个具名工具，将 tool-call ID、参数和取消信号转发给 Broker，并把完整结果或错误返回 Pi。Broker 按工具名路由到对应 `ToolSession`；每个会话绑定一个 Profile。

这里借用的是与 CANNBot 相同的 Pi 接入方式，不是复用它的业务工具：CANNBot 注册自己的 Task/Skill 类工具；`basic_wiki-tool` 注册 PTOAnalyzer 的九个工具，由 Broker 调用复制体的 Profile-bound `ToolSession`。CANNBot 的测试只能说明它自己的扩展接线有效，不能替代 Stage 5 对 PTOAnalyzer 工具注册和调用的验证。

Broker 只接受 Runner 为当前 Session/Case 提供的授权 Profile 索引。Agent 可省略 `profile` 使用默认 Profile，也可指定索引中的 Profile ID；Broker 将它解析为已验证的 `ProfileRef`，拒绝路径和未授权事件。Cursor 仍由对应 `ToolSession` 校验。Stage 5 只调用已提供的通用接口，不改它的实现；不得把 Basic Mode 伪装成 CANNBot。

只读暴露 `pto_wiki/pypto_wiki` 的冻结快照，让 Agent 用现有文件读取和搜索能力按需查阅；不增加 Wiki Agent 或复杂检索服务。为保持实验可复现，记录知识库快照身份。

每次 Analyzer 工具调用均须留下可追溯记录，至少包含 Agent tool-call ID、工具名、参数、选用的 Profile 身份、开始/结束状态及完整结果或错误。调用请求和结果保存到本次 PTOBench Session 的 Mode 归档，同时保留 Pi 原生会话记录；失败和取消也要记录。若大结果另存为不可变文件，归档引用和校验哈希。工具只读读取已归档 Profile，不触发 `profile`、`evaluate` 或其它 NPU 动作，不操作设备锁，也不产生硬件动作。首版没有固定 baseline、自动 diff 或额外比较工具。

## 实验方案

先用三个 Formal Case 做一次配对可行性试验，分别覆盖计算/多卡、数据流/多阶段以及高并行调度形态：

| Case | 代表性覆盖 |
|---|---|
| `deepseek_v4.lm_head_tp2_flash` | 双卡 MatMul 与通信相关的计算证据 |
| `deepseek_v4.prefill_layer_ep2_csa_demo` | 多阶段数据流及 Tile、Buffer、静态分配信息 |
| `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | CP8 多 rank、依赖和高基数调度明细 |

每个 Case 各运行 Basic Pi 与 `basic_wiki-tool` 一次，共六个 Agent Session。配对运行使用相同模型、初始 Candidate、PTOBench 配置、评测/Profiling 权限与预算；隔离工作区，配对内交错或随机运行顺序。除 Wiki 与 PTOAnalyzer 能力外，不额外给处理组增加 Agent 角色或代码工具。

记录正确性与 Clean Latency、Candidate Diff、有效/无效优化尝试、Profile/evaluate 硬件动作、九个工具逐次调用、工具结果是否被后续决策引用、Token 与 wall-clock。Wiki/Tool 调用本身不是硬件动作；NPU 动作仍由 PTOBench 原有账本记录。

这三个匹配对用于确认 Mode 接线、调用行为和试验可行性，不能支持统计显著性、普遍性能提升或单独证明 Wiki/Analyzer 的因果贡献。若试点显示值得扩大，下一阶段再预注册更多 Case 和重复种子。

## 文件

- [任务清单](TASK_CHECKLIST.md)：按输入冻结、Mode/Analyzer 接入、Wiki 接入、CPU 验收、Agent 配对试验和报告分步执行。
- [启动 Prompt](IMPLEMENTATION_PROMPT.md)：Stage 5 实施时可直接交给执行 Agent。
- PTOBench 通用 Pi 扩展/socket 接口的实现与测试记录见 [配对实验报告](../../../../basic_wiki-tool/reports/stage5/paired_trials.md) 和 [验收审计](../../../../basic_wiki-tool/reports/stage5/acceptance_audit.md)。
- [Stage 4 离线报告](../../reports/offline_replay_stage4_20261112.json)：当前 PTOAnalyzer 工具基线；Stage 5 不重复 Stage 4 回放。
- [PTOAnalyzer 当前接口](../../README.md)：九个工具、ProfileRef 与只读语义。
