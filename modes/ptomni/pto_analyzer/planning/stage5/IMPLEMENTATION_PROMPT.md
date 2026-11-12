# PTOAnalyzer Stage 5 实施 Prompt

请按 [Stage 5 任务清单](TASK_CHECKLIST.md) 实施 PTOAnalyzer 的真实 PTOmni Host 接线、Formal 60 + Held-out 15 的 Basic Pi/CANNBot 运行记录覆盖，以及三个 Tuner 专长的受控 Raw/Analyzer 试点。当前 Prompt 只是执行说明；没有在本轮启动 Agent、PTOBench 或 NPU 实验。

## 开始前

1. 阅读本目录 `README.md`、`TASK_CHECKLIST.md`、PTOAnalyzer 当前接口说明、Stage 4 清单和 [Stage 4 回放报告](../../reports/offline_replay_stage4_20261112.json)。
2. 检查当前工作树，保留所有无关改动；不要覆盖 `evaluator/` 的并行工作。
3. 完成 Task 0。已知仓库没有 PTOmni Host 调用 `ToolSession`；若复查仍不存在真实原生 Host 入口，停止接线，不创建模拟 Host、第二个 Tool Server 或新的 Agent Harness。
4. 确认 75 Case 的 Basic Pi/CANNBot 真实 Session 记录可读，先核对其 action events、Candidate snapshot/hash、Diff、`evaluate` Event 和已有 `profile` Event。Agent 轨迹不等于一定已有可供 Analyzer 使用的 Profile。

## Case 与数据要求

- 使用完整的 [Formal 60](../../../../../case_selection/formal_60/cases.csv) 和 [Held-out 15](../../../../../case_selection/heldout_15/cases.csv)，不再抽小样本代替覆盖矩阵。Basic Pi/CANNBot 已有 Session 作为离线回放输入，不为覆盖目标重跑这 150 条 Agent 轨迹。
- 每条 Case/方法轨迹选择起始 Candidate 和最终经 `evaluate` 接受的最佳 Candidate。若两者身份一致则去重；不为每次尝试 Profile。中间候选仅在已有 Profile 或回答明确轨迹问题时纳入。
- 只复用 Case、Candidate/source digest、输入和编译配置、设备、软件版本、Profile Event 与受保护归档身份全部匹配的 `combined profile`。仅 Case ID 相同不足以复用。
- 缺少对应 Profile 时，才由 Runner/PTOBench 在已注册的 Session workspace 对精确 Candidate 执行 `combined profile`。若 Agent Session 已关闭或 Snapshot 不可直接使用，先走获准的 Runner/Session workspace 流程，不对任意快照目录直接运行 CLI。每次成功 Profile 按 PTOBench 原账本计一次硬件动作；正确性与 Clean Latency 仍以对应 `evaluate` 为准。当前 `ProfileRef` 只接受完整的 `combined` Profile，不拼接 `split` Event。
- 若 150 条轨迹均没有可复用事件，起始和最终 Candidate 最多形成 300 个 Profile 输入；基于 Candidate 身份去重，并复用已有 Event 后，实际补采数会降低。不要为重复 Candidate 额外 Profile。
- 每个不同 Profile 都供九个 Tool 共用。Basic/CANNBot 记录用于事后覆盖与关联；除非轨迹含真实 PTOAnalyzer tool-call 记录，否则不能说 Baseline Agent 使用了 Analyzer。
- 先用 Formal 60 发现并修复有真实样本支持的问题；工具语义冻结后再回放 Held-out 15，保留其独立检查意义。

## 工具接线与试验

- 在真实 PTOmni Host 原生工具入口接入现有 `open_toolset(tuner, profile_ref)`。可信 Host 注入 Tuner 与 ProfileRef；每次只注册该 Tuner 的三个工具，并准确转发 `tool_specs`、完整 `initial_summary` 和 `call(name, args)`。
- Lead、Assembler、Challenger、Curator 不获得 Analyzer Tools；Lead 不获得 Runner/PTOBench 工具。错误身份、无效 Cursor、工具失败不能降级为空成功结果。
- 不新增 Tool、不改变九个 Tool 的语义、不改 PTOBench Collector、不写回 Profile、不建第二套硬件账本。基于真实归档修 Parser 时，保留 Fixtures、来源和回归记录。
- 用以下 Formal Case 做三个 Tuner 的 Raw/Analyzer 配对：Compute=`deepseek_v4.lm_head_tp2_flash`；Dataflow=`deepseek_v4.prefill_layer_ep2_csa_demo`；Schedule=`deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo`。现有 Profile 只有完整身份核验通过后才能复用。
- 每个 Case 的两组固定 Agent 后端/模型、Tuner、Candidate、知识、代码工具、预算与评测设置，只改变原始 Profile 与 Analyzer Tool 的信息接口。隔离起始工作区，交错或随机运行顺序；Profile/evaluate 全部经 Runner/PTOBench。
- 三个匹配对仅作每个角色的一次可行性验证，不据此宣称统计显著、普遍提升或跨 75 Case 的因果效应。

## 验收与交付

- Host 原生工具接线、角色隔离、Profile 身份、只读归档和 Cursor CPU 集成测试通过。
- 建立覆盖全部 75 Case、Basic Pi/CANNBot 两种方法的轨迹清单；所有缺失、失败、中断和无 Candidate 变化都保留原因。
- 对所有身份有效的起始/最佳 Candidate Profile 离线回放九个 Tool，形成 `Case × Method × Candidate × Tool` 覆盖矩阵；每项观测可追到源文件、Profile event 和 Agent event。
- Formal 60 与 Held-out 15 分开报告，记录每个来源覆盖、解析拒绝、`evaluate` 结果、Profile 成本和解释边界。
- 完成三个 Tuner 各一组 Raw/Analyzer 匹配对，并保存工具调用、假设、Diff、Profile/evaluate Event、正确性、Clean Latency、硬件动作、Token 和 wall-clock。
- 所有原始 Profile 哈希保持不变；所有真机动作均由 PTOBench 记账。结论区分离线覆盖、Agent 轨迹关联和受控工具效用，不把工具被调用写成性能提升。
