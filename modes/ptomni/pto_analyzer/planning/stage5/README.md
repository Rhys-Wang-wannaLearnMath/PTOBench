# PTOAnalyzer Stage 5：75 Case 轨迹覆盖与 PTOmni Tuner 效用验证

> 状态：规划中。本阶段规划真实 Host 接线、复用 Basic Pi/CANNBot 的 Formal 与 Held-out 运行记录，并进行受控 Tuner 工具效用试点；不在本规划回合启动任何 Profile、Agent 或 NPU 实验。

## 目标

Stage 5 分开回答两个问题：

1. **九个 Tool 能否覆盖全部正式 Case 的真实 Profile？** Formal 60 + Held-out 15 的每个 Case 至少有一份身份匹配的 `combined` Profile；对所有这些 Profile 离线回放九个 Tool。
2. **Tuner 使用这些 Tool 后，是否会改变优化假设和实验选择？** 在真实 PTOmni Host 接线后，针对 Compute、Dataflow、Schedule 各做一个 Raw Profile 与 Analyzer Tool 配对试点。

Stage 5 顺序：**Host 前置检查 → 工具接线与 CPU 验收 → 75 Case Profile 覆盖清单 → 身份核验/缺口补采 → 全量离线回放 → 三个专长配对试点 → 结果报告**。

## Stage 4 基线与当前缺口

Stage 4 在 16 份 `combined` Profile 上回放九个只读 Tool，设备覆盖 1、2、4、8 卡；51 项 CPU 测试通过，5,492 个归档文件哈希未变化，新增硬件动作数为 0。按当前 Formal/Held-out Case ID 精确交集，这 16 份中只有 7 个 Formal Case 和 1 个 Held-out Case；其余 8 个不属于这 75 个 Case，因此不能当作 75 Case 覆盖证据。详见 [Stage 4 回放报告](../../reports/offline_replay_stage4_20261112.json)。

Stage 4 Profile 是单一 Candidate 快照，没有 Agent 优化轨迹，也不提供 correctness 或 Clean Latency。当前仓库仍没有 PTOmni Host 调用 `ToolSession` 的入口。完整 Case 轨迹和 Tuner 实际使用 Analyzer 是两项不同证据，必须分别记录。

## Case 选择

### 全量 Profile 覆盖：Formal 60 + Held-out 15

对 75 个 Case **每个至少准备一份 `combined` Profile**，以 Case 清单中的正式初始 Candidate 和配置作为该 Case 的覆盖目标。Stage 4 或 Basic Pi/CANNBot Session 中已有 Profile 时，只有 Case、Candidate/source digest、输入与编译配置、设备、软件版本、PTOBench event 和受保护归档身份全部匹配，才能计入该 Case；否则由 Runner/PTOBench 对该 Case 的初始 Candidate 补采一次。最终覆盖矩阵必须有 75 行，Formal 60 和 Held-out 15 分开报告，失败或缺失不得从分母移除。

这项任务的目标是让每个正式 Case 都有可解析的 Profile 样本，不要求为每个 Case 的所有候选版本重复采集。已存在且身份匹配的 Profile 可以复用；新增 Profile 数等于 75 个 Case 中缺少有效 Profile 的数量。每个新 `combined profile` 计一次 PTOBench 硬件动作。全量 Case Profile 覆盖只能证明工具在这些 Case 的产物上有无解析与观测覆盖，不能单独证明优化前后的变化或性能收益。

### Baseline 轨迹与候选 Profile

覆盖全集就是现有的 [Formal 60](../../../../../case_selection/formal_60/cases.csv) 与 [Held-out 15](../../../../../case_selection/heldout_15/cases.csv)，不再缩成一个小样本集。纳入每个 Case 的 Basic Pi 与 CANNBot 实际 Session，共最多 150 条方法轨迹。Stage 5 不重跑这些 Agent Session；等既有运行完成后读取其事件、Candidate、代码 Diff、`evaluate` 和已有 `profile` 记录。

每个 Case/方法轨迹至少选择起始 Candidate，以及该轨迹中最终被 `evaluate` 接受的最佳 Candidate。75 Case 覆盖 Profile 若与某条轨迹的起始 Candidate 身份相同，可同时满足该轨迹的起始 Profile 需求。若最佳 Candidate 与起始 Candidate 的源码和配置身份完全相同，只保留一份 Profile。若 Agent Session 中已有对应 `combined profile`，先核验并复用；缺少 Profile 时才由 Runner/PTOBench 对该精确 Candidate 补采。不要为每次编辑或每次失败尝试都 Profile。失败 Candidate 只纳入已有可解析的 Profile；不为无法正确运行的版本强行补采。

以完整 Candidate 身份去重，并把 75 Case 覆盖目标、两种方法的轨迹候选目标合并成唯一 Profile 清单。若 150 条轨迹都没有可复用 Profile，起始与最佳候选最多形成 300 个不同 Candidate Profile；75 Case 覆盖 Profile 与这些候选身份相同的部分不重复采集。最终新增动作数按“所有必需且身份唯一的 Profile 中，扣除已有匹配归档后的缺口数”核算，不预先把 75 次覆盖采集叠加到 300 次上限。

同一 Case 名称不证明 Profile 可复用。至少核对 Candidate/source digest、输入与编译配置、设备、软件版本、PTOBench event 和保护归档身份。成功 `profile` 计一次硬件动作，但不证明正确性或性能；正确性和 Clean Latency 以相应 `evaluate` 事件为准。Profile 采集、设备租约、锁和预算均由 Runner/PTOBench 负责。补采必须发生在已注册的 Session workspace；若 Agent Session 已关闭或 Snapshot 不可直接使用，先走获准的 Runner/Session workspace 流程，不对任意快照目录直接运行 CLI。当前 `ProfileRef` 只接受完整的 `combined` Profile；不得把不同 `split` Event 拼成一份 Profile。

每个有效的不同 Profile 都供全部九个 Tool 共用；按角色分别离线重放 Compute、Dataflow、Schedule 的三个 Tool。输出两份互相关联的矩阵：`Case × Profile` 用于核对 Formal 60 + Held-out 15 的全量覆盖；`Case × Method × Candidate × Tool` 用于记录 Baseline 轨迹候选的 Tool 结果、来源文件覆盖、解析/拒绝情况、观测和值的单位、受限解释以及对应的 Agent Diff 和 `evaluate` 结果。完整摘要不得截断；高基数明细按需分页。Basic/CANNBot 的历史轨迹仅用于覆盖审查和事后关联，不代表它们曾调用 PTOAnalyzer。

Profile 修改前后对比只对同一 Case 中确有身份匹配的起始/最佳候选配对计算；按指标和单位对齐，缺失或口径不兼容时标为不可比较。对比只报告观测变化，不推断根因或性能收益；正确性和 Clean Latency 仍取自对应 `evaluate`。

Held-out Profile 在 Formal 60 的工具语义和解析行为冻结后再做最终回放。Held-out 结果用于记录泛化和数据缺口；若因此修改工具，必须标出版本变化，不能把修改后的结果与先前 Hold-out 结果混为同一轮验收。

### 受控的 Tuner 工具效用试点：三个 Formal Case

Baseline 的 75 Case Session 不会调用 PTOAnalyzer，不能证明工具影响了 Agent 决策。真实 Host 接线后，选择以下三个 Formal Case，各针对一个 Tuner 专长运行 Raw Profile 与 Analyzer Tool 的匹配比较：

| Tuner | Case | 选择理由 |
|---|---|---|
| Compute | `deepseek_v4.lm_head_tp2_flash` | Operator 级 MatMul/communication、双卡，覆盖计算活动与 rank/dispatch 关联。Formal 清单第 7 行。 |
| Dataflow | `deepseek_v4.prefill_layer_ep2_csa_demo` | 多阶段、双卡 Case，适合检查 Tile、Buffer、静态分配等 Dataflow 证据；Stage 4 有同名 Profile，仍须先核验身份。Formal 清单第 20 行。 |
| Schedule | `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | CP8 多 rank、高基数 Task/依赖轨迹，检验 Schedule Tool 的摘要与按需分页。Stage 4 有同名 Profile，仍须先核验身份。Formal 清单第 30 行。 |

每个 Case 内保持 Agent 后端/模型、Tuner、起始 Candidate、知识、源码工具、预算和评测设置一致，只改变 Agent 看到原始 Profile 还是 Analyzer 角色化结果。Raw/Analyzer 两组隔离工作区，运行顺序交错或随机。一个匹配对只能验证接线和可行性；不能据此宣称普遍提升。若后续要评估稳定收益，再按同一设计增加重复种子或 Case。

Held-out 15 用于冻结后的离线全量回放；本阶段不拿 Held-out 反复调试三个受控试点。若需要 Held-out 上的真实效用验证，应在正式记录工具版本和初始结果后另立配对任务。

## PTOmni Host 前置条件

当前 `modes/ptomni/` 尚无真实 Host 调用 `ToolSession`。先确认 PTOmni Runtime 的原生工具注册和调用入口；若不存在，停止 Analyzer 接线，不在 Analyzer 内造模拟 Host、第二个 Tool Server 或新的 Agent Harness。PTOBench 的 `modes/basic/runner.py` 和 `modes/cannbot/` 可供参考其 Session 生命周期与记录方式，但不移植 CANNBot 工作流。

Host 必须从可信 Runner/Event Ledger 取得确切 `ProfileRef`，按当前 Tuner 打开 `open_toolset(tuner, profile_ref)`，并仅暴露 `tool_specs`、完整 `initial_summary` 和 `call(name, args)`。Lead 不获得 Analyzer 或硬件 Tool；Assembler 不负责分发工具。PTOAnalyzer 不启动硬件动作、不写原 Profile、不维护第二套账本。

## 结论边界与交付

- 75 Case Profile 覆盖说明工具能否读取全部正式 Case 的产物；Basic/CANNBot 全量轨迹回放补充候选变化和事后关联，但两者都不能说明 Baseline Agent 使用了 Analyzer。
- 真实 Tuner 配对试点能检查工具是否进入决策轨迹；少量配对不能证明统计显著或跨 Case 泛化。
- Profile 只提供诊断输入；正确性与 Clean Latency 必须引用对应 `evaluate` 事件。
- 记录所有复用/补采 Profile 的身份依据、PTOBench 硬件动作、设备安排、Profile/evaluate Event、Agent Session、Candidate Diff、工具调用和失败情况。
- 不新增 Tool、不扩大 PTOBench Collector、不把相关计数写成性能根因；如 Case 暴露真实解析缺陷，保留可复现归档并按缺陷修正与回归，不将缺失证据补成零。

## Stage 5 文件

- [任务清单](TASK_CHECKLIST.md)：按 Host、75 Case Profile 覆盖、轨迹候选补采、离线回放和 Tuner 配对试点分步验收。
- [启动 Prompt](IMPLEMENTATION_PROMPT.md)：后续实施时可直接使用。
- [Stage 4 回放报告](../../reports/offline_replay_stage4_20261112.json)：工具基线与离线证据。
- [PTOAnalyzer 接口](../../README.md)：Profile 身份、Tuner 权限和只读边界。
