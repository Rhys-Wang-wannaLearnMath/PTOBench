# PTOmni：EAT 参考与任务拆分

> 目的：记录从 EAT 借鉴的团队编排机制与当时的任务拆分，是历史讨论稿。当前架构以 [PTOmni_架构设计_2026-11-03.md](PTOmni_架构设计_2026-11-03.md) 为准；EAT 仅作参考。

## 可借鉴的机制

EAT 的最小闭环是 **Reflect → Evolve → Launch → Monitor**：Supervisor 根据前轮结果调整团队配置，Agent 按阶段执行，Harness 统一评测并记录结果与成本，再将这些信息用于下一次规划。

PTOmni 可吸收四点：

1. **让 Supervisor 输出结构化 `TeamPlan`**，由外层校验角色、阶段、权限和协作开销；不让它直接改 Candidate。
2. **显式定义阶段与并行关系**：同一只读分析阶段可并行；阶段间顺序执行；Owner 独占 Candidate 写入。
3. **把评测结果和运行成本回送规划闭环**：正式测量走 PTOBench；Supervisor 只在初始化、证据冲突、瓶颈变化或停滞等事件触发时反思，不必每个小改动都重新组队。
4. **让执行记录可追溯**：优化变更、评测结果、证据和团队计划关联到同一个正式尝试；由确定性状态更新 Best、预算与机会状态，Curator 只维护经验摘要。

建议将 PTOmni 主流程收敛为：

```text
PTOBench Case Contract
  → Supervisor Reflect / Evolve
  → validated TeamPlan
  → phase Runner (specialists advise; Owner alone edits)
  → PTOBench correctness / clean benchmark
  → optional native profiling
  → append Attempt and Evidence; deterministically update State
  → Curator updates Case Memory
  → Monitor; continue or replan on meaningful events
```

`Episode` 可用于团队规划分组，正式优化记录建议统一为 `Attempt`。不要再把 EAT 的 iteration 或 PTOmni 的 Round 引入为第二套持久化记录单位。

## 不应照搬的部分

- EAT 的通用 `eval.sh`、GPU 资源选择和 Harness 不能替代 PTOBench 的 Case Contract、NPU 执行和正式预算。
- EAT 以团队轮次和 Git 变更组织实验；PTOmni 应以 Candidate/ProgramVersion、Attempt 和 Evidence 组织，Git commit 不构成评测事实。
- 不直接接入 EAT 的启动脚本、插件、角色提示、预算账本或 Agent Session 管理；复用 PTOBench 与现有 Pi Runner，仅实现必要的编排适配。
- 不让多个 Specialist 并行修改同一 Candidate，也不因借鉴 EAT 而扩成多个长期 Agent 子系统。

## 工作拆分

整体按 **三个实现工作流**推进，PTOBench 集成与验收是贯穿三者的横切工作，不另建一套评测系统。

| 工作流 | 负责内容 | 最小交付 |
|---|---|---|
| **1. PTOmni 架构与运行闭环** | Supervisor、`TeamPlan`、阶段调度、统一 Runner、Owner 写入权、Attempt/State/Case Memory | 一个 Case 能从基线运行到优化尝试、正式验证、状态更新和继续/停止决策 |
| **2. Offline Knowledge Base** | 固定知识源、轻量索引、按需检索、设备/版本适用性 | Owner 与 Specialist 可按主题获取带来源和适用范围的知识 |
| **3. Profiling Adapter** | 探测可用原生数据，提供短 Summary/Compare，校验候选与环境可比性 | Agent 可获取带来源、版本身份和缺失标记的诊断证据 |

**PTOBench 集成与验收**：工作流 1 调用 PTOBench 的 Case/Workspace/预算/正确性/正式 Benchmark；工作流 3 消费可用 Profile 产物。Clean Benchmark 是正式性能依据，Profile 只用于诊断。验收覆盖一次完整 Case 闭环、候选写入权限、状态恢复和证据可追溯性。

## 建议顺序

先冻结共享契约：`ProgramVersion` 身份、`Attempt` 生命周期、`TeamPlan` 阶段与权限、Evidence 引用、PTOBench 预算来源。然后完成最小运行闭环；Knowledge Base 与 Profiling Adapter 可在接口确定后并行；最后用 PTOBench 端到端验证并据实验结果决定是否扩展角色或记忆能力。

第一版不需要向量数据库、通用 Agent 框架、自研 Profiler、学习型调度器或独立预算系统。
