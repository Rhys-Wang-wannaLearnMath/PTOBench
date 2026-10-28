# Stage 2：唯一 State 和确定性 Reducer

## 目的

把在线运行中所有重要进展收拢到一个可检查、可重放的 Progress State，消除重复 Memory、History、Frontier 和 Progress 实现。

## 实现内容

1. 定义状态初始值、事件类型和 revision 递增规则。
2. 实现 `apply(event, state) -> state` 的确定性 Reducer。
3. 覆盖决策、假设、源码版本、evaluate 结果、Profile/Evidence、预算、停止和收尾事件。
4. 明确 working、last_verified、best_verified 的切换和回退规则。
5. 让机会信息作为 State 字段或派生视图更新，不建立单独 Frontier 写入器。
6. 为重放、重复事件、过期 revision、非法状态转移提供错误行为。

## 关键规则

- 只有通过规定 correctness 的程序才能成为 `last_verified` 或 `best_verified`。
- Pi 不能直接写事实字段。
- 失败和无提升也产生事件。
- 每个 EvidenceBundle 必须引用 Hypothesis、Cycle 和 ProgramRef。
- 这一步只保证状态正确，不自动判断某个机制是否因果有效。

## 验收

- 用同一事件序列重放得到同一 State。
- 过期或重复事件不会静默覆盖新状态。
- 失败评测不会污染最佳版本。
- 状态可被 ContextRenderer 直接读取，无需拼接多个数据库。
