# Stage 4-1：离线知识接入在线 Pi

## 目的

让在线 Pi 能按当前 Hypothesis 或机制信号查询已发布知识卡片，并在后续请求的 Context 中看到相关内容。知识库是只读输入，不成为状态存储或决策器。

## 依赖

- Stage 0 的 Knowledge Card/Knowledge Query 契约；
- Stage 1-3 发布的知识包；
- Stage 2 的 Progress State；
- Stage 3 的 ContextRenderer 和 Pi 持续循环。

本阶段可与 `stage4-2_profile_integration` 并行。

## 实现内容

1. 实现一个小的 Knowledge Port 接口：按机制标识、信号或自然语言查询返回卡片引用和内容。
2. 检查已发布包的 schema 和索引可读；不在在线运行中重建或修改包。
3. ContextRenderer 只放入当前决策所需的短摘录；其余卡片由 Pi 按需读取。
4. 卡片内容作为知识线索，不自动转成当前 Case 的事实或适用性结论。
5. 记录卡片引用供 Hypothesis 和 Progress State 追溯，但不保存审计哈希或 commit 编号。

## 明确不做

- 不要求知识包与运行时软件版本/硬件做匹配门禁。
- 不读取或回写 PTOBench 真实运行记录和在线 Case 结果。
- 不让 Knowledge Port 排序机会、判断瓶颈或决定是否停止。

## 验收

- 同一个已发布包在相同 query 下给出确定、可测试的结果。
- 查询为空或卡片损坏时返回明确错误/空结果，不阻塞其他安全动作或注入旧内容。
- Context 中的知识卡片能追溯到卡片标识，但不泄露原始来源审计信息。
- Pi 可以在不重新启动的情况下查询多张卡片并继续同一个 Session。
