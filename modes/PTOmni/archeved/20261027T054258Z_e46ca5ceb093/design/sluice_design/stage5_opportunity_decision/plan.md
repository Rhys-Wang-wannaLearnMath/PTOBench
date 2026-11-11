# Stage 5：证据驱动的机会和下一步决策

## 目的

把已经积累的 Hypothesis、评测结果、Profile 和知识线索转成下一步可解释的搜索决定。这是 SLUICE 的核心决策机制之一，但必须保持在 Pi + 单一 Progress State 内，不新建 Frontier 服务或评分模型。

## 状态表示

每个 Opportunity 是 Progress State 中的一个轻量记录，至少关联：

- 对应的机制/优化方向和当前问题；
- 适用前提及其来源（Pi 提议或可核验事实）；
- 当前已有的相关 EvidenceBundle；
- 已尝试的 Hypothesis 和客观结果；
- 尚缺的关键证据；
- 下一步建议及其理由。

机会状态只保留少量有实际控制作用的值，例如 `open`、`needs_evidence`、`closed`。变快、变慢、无变化、失败等属于尝试结果，不再重复编码成机会状态。条件变化后是否重查由新证据和 Pi 的决策表达，不额外发明 `recheck` 状态。

## 第一版决策逻辑

Context 给 Pi 提供可解释的判断依据：

- 有重要瓶颈信号、适用条件成立且尚未尝试：优先考虑 optimize；
- 方向可能重要但缺少区分假设的观测：考虑 diagnose；
- 过去无效的前提已变化，或新证据改变了结论：考虑 recheck；
- 没有证据支持且成本合理的剩余方向，或预算/约束不允许继续：考虑 stop。

这些是 Pi 决策的证据框架，不是自动 speedup predictor。Pi 给出的“重要性、适用性、成本”属于带来源的判断提议；Reducer 检查引用和结构，不伪装成硬件事实。Controller 只强制执行 PTOBench 预算和运行安全约束。

每个完整评测周期结束后，Pi 必须先提交一个新 Decision，Controller 才接受下一次源码修改或新的正式评测。没有收益且没有新机制信息的尝试会降低该方向的优先级；如果没有剩余的证据支持、成本合理的机会，Pi 应停止，而不是为了耗尽预算继续搜索。第一版不硬编码“连续失败 N 次就停”的阈值，避免把未经实测的数字伪装成通用规则；PTOBench 预算仍是最终硬上限。

## 与知识和 Profile 的连接

- Knowledge Card 中的预期信号与 EvidenceBundle 中的 Observation 通过机制/信号标识关联。
- 每条结论可追溯到 Hypothesis、ProgramRef、Cycle 和证据引用。
- 结果不支持假设时，记录该结果；不能因为 latency 变快就自动宣称机制因果成立。

## 验收

1. Context 能区分当前有证据的方向、缺少证据的方向和已关闭方向。
2. `optimize`、`diagnose`、`recheck`、`stop` 都要求给出简短理由及相关证据引用（若有）。
3. 不引入固定分数或训练型 speedup predictor。
4. Reducer 不接受没有引用依据的事实性更新。
5. 实现与测试能说明机会记录如何随成功、失败、无提升和前提变化而更新。
6. 一个完整周期后，未提交新 Decision 的 Pi 不能开始下一次源码修改或正式评测。
