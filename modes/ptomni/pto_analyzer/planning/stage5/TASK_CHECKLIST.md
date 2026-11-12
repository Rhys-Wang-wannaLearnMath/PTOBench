# PTOAnalyzer Stage 5 任务清单

## 范围与约束

- 覆盖 Formal 60 + Held-out 15 中所有 Case 的 Basic Pi、CANNBot 真实 Session 记录；不为这项覆盖工作重跑两种 Baseline Agent。
- 在真实 PTOmni Host 接入九个只读 Tool，并对 Compute、Dataflow、Schedule 各做一个受控 Raw/Analyzer 配对试点。
- PTOAnalyzer 继续离线、只读、确定性；`profile`、`evaluate`、设备租约和硬件动作只通过 Runner/PTOBench 与原账本。
- 只向当前启用的 Tuner 暴露其三个工具。Lead 不调用 Analyzer 或硬件 Tool；Assembler 不分发工具；其他角色不获得 Analyzer Tool。
- 不创建模拟 Host、第二个 Tool Server 或 Agent Harness；不新增诊断 Tool、不改变输出语义、不扩大 PTOBench Collector。
- 不逐次尝试 Profile。每个 Case/方法至少覆盖起始 Candidate 与最终 `evaluate` 接受的最佳 Candidate；同一身份的 Candidate 只需一个 Profile。
- Held-out 结果在 Formal 60 的 Parser/工具语义冻结后再回放；不依据 Held-out 结果迭代调试并仍把它称为独立验收。
- 保留工作区已有改动，尤其不覆盖 `evaluator/` 中的并行修改。

## Task 0：冻结 Stage 4 基线并检查 Host

- [ ] 核对 Stage 4 报告：16 份 Profile、九个 Tool、1/2/4/8 卡覆盖、51 项 CPU 测试、5,492 个哈希核对、0 个新增硬件动作。
- [ ] 搜索当前 PTOmni Host 的原生工具注册入口和 `ToolSession` 调用方；若仍不存在，记录为前置阻塞并停止接线，不创建模拟实现。
- [ ] 指明可信 Runner/Event Ledger 如何提供 Profile event、Candidate 身份和当前 Tuner 身份。
- [ ] 查阅 Basic/CANNBot Session 记录接口，确认如何只读取得 Case、optimizer、action events、source snapshot/hash、Diff、`evaluate` 结果和已有 `profile` events。

**验收：** Host 与 Event Ledger 的实际入口已定位；75 Case × 两种方法的可读取记录格式已列明。Host 不存在时不得继续接线。

## Task 1：在真实 Host 暴露 Tuner 专属 Analyzer Tool

**接口：** `open_toolset(tuner, profile_ref)`；Host 转发 `tool_specs`、完整 `initial_summary` 和 `call(name, args)`。

- [ ] 只由可信 Host 注入 Tuner 与 ProfileRef；Agent 不能选择 Profile 路径、Event 或权限。
- [ ] 每个 Tuner 只注册自己的三个 Tool；一次会话的工具回调固定绑定同一个 `ToolSession`。
- [ ] 工具调用结果关联到原有 Agent tool-call trace、Round、Candidate 和 Profile event；不另建账本。
- [ ] 保留完整摘要；高基数明细沿用 Cursor，错误与取消必须可识别，不能伪装成成功空结果。

**验收：** 真实 Tuner 可调用自己的摘要和分页工具；调用不触发 PTOBench/NPU。

## Task 2：权限、身份和只读集成测试

- [ ] Compute、Dataflow、Schedule 各自只看到对应三个工具；Lead、Assembler、Challenger、Curator 看不到 Analyzer Tool。
- [ ] 伪造角色、错误 Profile event、Candidate digest 不匹配、跨 Tuner/Profile Cursor 均在读取不匹配数据前失败。
- [ ] 测试完整摘要和 Cursor 页的转发无截断、无重复、无遗漏。
- [ ] 测试前后 Profile 原始归档哈希不变；Analyzer 调用不增加 PTOBench 硬件动作。
- [ ] 运行 PTOAnalyzer 与 Host 的 CPU 集成测试。

**验收：** Host 权限与 PTOAnalyzer 静态权限表一致；身份绑定、错误传递和只读性通过。

## Task 3：建立 75 Case × Basic/CANNBot 轨迹清单

**全集：** [`formal_60/cases.csv`](../../../../../case_selection/formal_60/cases.csv) 和 [`heldout_15/cases.csv`](../../../../../case_selection/heldout_15/cases.csv)。不再抽小样本；完整记录最多 150 条 Agent Session 轨迹。

- [ ] 每个 Case 分别登记 Basic Pi、CANNBot 的 Session 身份、开始与结束 Candidate hash、代码 Diff、全部 `evaluate` 结果、已有 `profile` Event、硬件动作、Token 和 wall-clock。
- [ ] 缺失 Session、失败/中断、没有 Candidate 变化、没有有效 `evaluate` 等状态必须显式记录；不得静默从分母中移除。
- [ ] 按实际 Candidate hash/config 关系识别 Basic 与 CANNBot 是否共用起始 Candidate；不能仅按 Case ID 合并。
- [ ] 每条方法轨迹选择起始 Candidate 和最终经 `evaluate` 接受的最佳 Candidate；若两者身份相同，去重。中间候选仅在已有 Profile 或能回答明确的轨迹问题时纳入。
- [ ] 将 60 个 Formal 标为 Parser/语义修正阶段的开发覆盖；Hold-out 15 先封存其工具结果，待 Formal 规则冻结后统一回放。

**验收：** 形成 75 行 Case 主表及 Basic/CANNBot 两条方法记录列；每个缺失状态有明确原因和可追踪的 Session/Event。

## Task 4：补齐候选 Profile 输入

PTOAnalyzer 只消费成功的 `combined profile`。已有 Profile 只有在 Case、Candidate/source digest、输入/编译配置、设备、软件版本和保护归档身份全部匹配时才能复用。

- [ ] 先盘点每个起始/最佳 Candidate 是否已有对应 Profile event，并核对上述身份字段。
- [ ] 对缺失 Profile 的起始 Candidate 和最终最佳 Candidate，由 Runner/PTOBench 对精确 Candidate 执行一次 `combined profile`；若同一 Candidate 已有可复用 Event，不重复运行。
- [ ] 只在已注册的 Session workspace 中补采；若 Agent Session 已关闭或 Snapshot 不可直接使用，先走获准的 Runner/Session workspace 流程，不对任意快照目录直接运行 CLI。
- [ ] 当前 `ProfileRef` 只接受完整 `combined` Profile；遇到历史 `split` 产物时记录不兼容，不拼接不同 Event。
- [ ] 不对每次 Agent 编辑、失败候选或未选中的低价值尝试批量 Profile。正确性失败且无法稳定执行的候选不强行补采。
- [ ] Profile 计入 PTOBench 硬件动作预算；Profile 不证明 correctness 或 Clean Latency，关联的 `evaluate` Event 保持为权威结果。
- [ ] 记录每一项复用/新增 Profile 的 ProfileRef、Candidate hash、来源 Session、硬件动作和复用理由。

在 150 条轨迹都缺少可复用 Profile 的极端情况下，起始与最终候选最多对应 300 个不同 Candidate Profile；同一候选身份去重和复用已有 Event 会降低新增动作数。每个缺失的 `combined profile` 计一次 PTOBench 硬件动作。此为预算规划上限，不为重复 Candidate 重复采集。

**验收：** 每个纳入回放的 Candidate 都绑定一个身份有效的 ProfileRef；Profile 缺口与跳过原因完整可查。

## Task 5：回放九个 Tool 并整理覆盖矩阵

- [ ] 对每个有效 Profile 分别打开 Compute、Dataflow、Schedule ToolSession；九个 Tool 共用同一份 Profile，不重复采集。
- [ ] 保存每个 Tool 的完整摘要、来源覆盖、单位、限制、Profile 身份和 Tool 版本；不把缺失数据补成零或瓶颈判断。
- [ ] 高基数 Tool 先完整读取摘要；只在验证某个 Case/候选差异或审查数据完整性时分页读取明细。至少包含 Stage 4 最大图形态的全页回归样本。
- [ ] 对同 Case 的起始与最佳候选，比较工具观测变化，并并列记录对应代码 Diff、`evaluate` 正确性和 Clean Latency 变化；只陈述关联，不从单一关联断言根因。
- [ ] 先处理 Formal 60 的发现。冻结输出语义后再对 Held-out 15 做最终离线回放；Hold-out 的异常记录为结果，不用来静默调参。
- [ ] 若真实归档暴露解析缺陷，保存可复现样本和来源；只做证据支持的 Parser 修正，保持 Tool API/语义边界，并回归全部受影响 Fixtures 与 Stage 4 样本。不新增未经数据支持的判定规则或 Tool。

**验收：** 输出 `Case × Method × Candidate × Tool` 覆盖矩阵；每个观测可追到 Profile 文件和 Agent Event。Formal/Held-out 覆盖率分别以 60、15 为分母，失败/缺失单列。

## Task 6：三个 Tuner 的 Raw/Analyzer 配对试点

**执行门槛：** 真实 PTOmni Host 接线和 CPU 验收通过；Stage 5 已明确实验预算、设备窗口和 Runner/PTOBench 锁的使用方式。

| Tuner | Formal Case | Profile 身份与覆盖重点 |
|---|---|---|
| Compute | `deepseek_v4.lm_head_tp2_flash` | Operator MatMul/communication、双卡计算和 rank/dispatch 记录；Stage 4 无同 Case Profile，按 Task 4 补齐。 |
| Dataflow | `deepseek_v4.prefill_layer_ep2_csa_demo` | 多阶段、双卡 Tile/Buffer/分配观测；Stage 4 同名 Profile 只有身份核验通过才复用。 |
| Schedule | `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | CP8 多 rank、高基数 Task/依赖与分页；Stage 4 同名 Profile 只有身份核验通过才复用。 |

对每个 Case 单独做 Raw Profile 与 Analyzer Tool 匹配对：固定 Agent 后端/模型、当前 Tuner、起始 Candidate、知识、代码工具、优化预算、Profile/evaluate 许可和硬件设置，只改变 Profiling 信息接口。两组用隔离工作区，运行顺序交错或随机。每个试点记录 Tuner 假设、工具调用、代码 Diff、检查/evaluate、Profile event、正确性、Clean Latency、硬件动作、Token 和 wall-clock。

**验收：** 每个 Tuner 至少一组可追溯的匹配对；三组只作为角色覆盖和可行性证据。单一匹配对不声称统计显著或全 Case 提升；后续重复数根据观察到的效应和波动另定。

## Task 7：结果归档与结论边界

- [ ] 分开报告：Baseline 轨迹/Case 覆盖、工具解析与来源覆盖、候选 Profile 变化、Tuner 实际工具使用、正式 `evaluate` 结果。
- [ ] Basic/CANNBot 事后重放不得写成“Baseline 使用了 Analyzer”；只有受控 PTOmni Tool calls 才是在线使用证据。
- [ ] 记录未完成、Profile 失败、correctness 失败、缺失来源、解析拒绝、未变化候选和预算耗尽，不能只挑成功 Case。
- [ ] 对所有真实硬件事件核对 PTOBench 原账本、设备分配与执行记录；回放前后核对受保护原始 Profile 哈希。
- [ ] 保存 Stage 5 结果报告和更新后的 Case/Tool 覆盖矩阵；对下一步仅提出有证据的修复或追加试验。

**验收：** 结果可从 Case/Session/Candidate/Profile/evaluate/Event 完整追溯；没有把单次可行性信号写成普遍性能收益。
