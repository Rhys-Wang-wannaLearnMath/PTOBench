# Stage 3：Context 投影和单 Pi 持续循环

## 目的

让 Pi 在同一个进程中完成多个逻辑周期，并在每次请求前看到当前 State，而不让 SLUICE 接管 Pi 的普通对话历史。

## 实现内容

1. ContextRenderer 根据 State、workspace、相关 Knowledge Card 和 EvidenceBundle 生成动态 `SLUICE_CONTEXT`。
2. 通过 Pi 支持的动态 Context 接口替换上一份 SLUICE 投影。
3. 保持 Pi 自己的 assistant/tool 消息配对、历史和 compact 行为不变。
4. 在投影生成失败时阻止危险工具，而不是注入旧投影。
5. 为上下文规定最小内容：当前程序、最近事实、活动假设、预算、待验证问题、可用工具和停止条件。

## 不做

- 不删除、重排或裁剪 Pi 的完整消息历史。
- 不在 SLUICE 里实现 compact。
- 不启动第二个 Pi 来传递进展。
- 不把整个知识库或整个 Profile 原始结果加入每次请求。

## 验收

- 两个以上逻辑周期使用同一个 Pi 进程完成。
- 每次请求的投影对应最新 State revision。
- 旧的 SLUICE 投影不会和新的动态投影重复生效。
- Context 失败能安全阻止编辑/evaluate/profile。
