# Stage 0：共享契约和范围冻结

## 目的

在写在线实现、Profile Tool 或知识库生成器之前，固定三者共同理解的最小数据和职责，防止后续各自发展出互不兼容的状态格式。

## 必须确定的内容

### 运行对象

- `SessionRef`：一次 PTOBench 运行的公开身份。
- `ProgramRef`：Candidate 的工作版本、父版本和验证状态。
- `Budget`：可消费的公开评测/Profile 预算。

### Agent 语义

- `Decision`：`optimize`、`diagnose`、`recheck`、`stop`。
- `Hypothesis`：修改范围、前提、预期信号、验证问题和回退条件。
- `Cycle`：一次从决策到事实写入的完整逻辑周期。

### 观测和状态

- `Observation`：数值、单位、采集方式、程序版本和可用性。
- `EvidenceBundle`：一组与 Hypothesis、Cycle 和 ProgramRef 绑定的 Observation。
- `ProgressState`：唯一可写在线状态及其 revision。

## 设计约束

1. 事实字段与 Pi 的解释字段分开。
2. 缺失或不可用是观测属性，不引入 `unproven`、`deprecated`、`inconclusive` 等多余认识状态。
3. Agent-facing 数据不包含审计哈希、内部路径、commit 编号或隐藏评测对象。
4. Knowledge Card 与 Observation 使用可关联的机制标识和信号名称，但知识卡片不包含当前 Case 答案。
5. 所有可写变化都通过 Reducer 进入 ProgressState。

## 工作步骤

1. 画出状态转移：`running → stopping → finished/failed`。
2. 写出最小字段表和版本规则。
3. 为 evaluate、profile、knowledge query 和 context projection 写请求/结果示例；示例只使用抽象占位值。
4. 明确哪些字段由 PTOBench、Profile Tool、Knowledge Builder 或 Pi 提供。
5. 为每条不变量写一个失败测试或断言。

## 完成条件

- 三个 stage1 子阶段和 stage2 计划可直接引用同一套字段名称和约束。
- 能明确回答“这个值是谁产生的、谁可以修改、谁只能解释”。
- 没有独立 Memory、History、Frontier 或 Audit 状态的第二份可写接口。
