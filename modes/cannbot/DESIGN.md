# CANNBot PyPTO 原始性能流程接入设计

状态：2026-10-07 已实现独立方法入口，`mode.yaml` 为 `implemented: true`。
Pi 局部扩展与 Codex 局部 MCP 使用同一个原生阶段会话执行器；运行方式与
实际档案布局见 [README.md](README.md)，公共接口遵循 [方法协议](../PROTOCOL.md)。
执行机制使用受控 CPU/原生工具替身验证，没有执行真实 LLM 优化或 NPU 调优，
也没有全局安装插件。下文保留抽取边界与原流程依据。

最新用户要求：**尽可能保持 CANNBot 原始模样，不为 PTOBench 改写方法内容。**
选用上游已有的 PyPTO 性能入口，保留其原始技能、角色、catalog、分析脚本、
案例、阶段规则和交接格式。Pi 直接使用选定版本的原文，不另外制作改进版指南。
实现仅包含独立的执行接线和原始记录。

最新职责边界：旧 API、profile 格式、原脚本和宿主工具的使用问题由 CANNBot
按自己的原始流程处理，不属于我们的接入预检、修复或内容适配任务。
PTOBench CLI 直接向 CANNBot 的 Pi/Codex 会话开放，与 Basic 使用同一接口。
接入侧负责原样资源加载、会话启动/交接、共享 Session 环境、进程清理及原始记录；
验证限于这些执行机制，不设置 CANNBot 方法内部兼容性门槛。

架构约束（按用户最新要求修订）：**方法是考生，PTOBench 是裁判。**
Basic、CANNBot、Ours 的任务说明、知识、分析器、流程控制、资料加载与
Codex/Pi 会话管理，都由方法实现负责。它们可以放在 PTOBench 仓库的 `modes/`
目录，但不成为 evaluator 的业务逻辑，也不要求 PTOBench 替某个方法执行优化步骤。
通用实验启动和档案收集可以复用，不能因复用而把方法决策放进公共层。
Basic 的方法职责已迁到 `modes/basic` 与 `modes/_shared`；
公共 Launcher 调用不透明的四操作入口，evaluator 不加载方法实现或 instructions。

## 1. 方法定义与来源

建议实验方法定义为 **固定上游版本的 CANNBot PyPTO 性能调优流程，由 Pi/Codex 执行**。
仍采用此前约定的性能入口范围；原文和原算法保持原样，环境和宿主工具接线单独说明。
配置仍用 `mode: cannbot`；`optimizer.backend: codex | pi` 选择执行工具。
两者正交，例如 `cannbot + pi` 和 `cannbot + codex` 是同一方法的两个执行配置。

上游 README 指向 [GitCode cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)。
本次实际检查的是公开 GitHub 仓库 `hicann/cannbot-skills` 的固定快照：

- commit：`94216ba2f0b569cfad22c7794ad99426a07ab7b6`。
- commit 时间：`2026-09-30T09:48:59Z`。
- [完整快照](https://github.com/hicann/cannbot-skills/tree/94216ba2f0b569cfad22c7794ad99426a07ab7b6)。
- [LICENSE](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/LICENSE)：CANN Open Software License Agreement Version 2.0；已保留许可证和原始声明。

不能将这个 GitHub 快照宣称为已验证与 GitCode 当前 HEAD 完全一致。
实现已用 `upstream.lock.json` 冻结所采用的实际内容，运行时不跟随上游更新。

主要依据：

| 原始资料 | 对设计的作用 |
| --- | --- |
| [pypto-op-perf-tune/SKILL.md](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/ops/pypto-op-perf-tune/SKILL.md) | 性能调优入口、三个领域阶段、实测与回退原则、配置先验搜索 |
| [tune-orchestrator/SKILL.md](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/ops/pypto-op-perf-tune/tune-orchestrator/SKILL.md) | 阶段顺序、阶段内部迭代、Todo、退出条件、重新采集要求 |
| [phase-handoff.md](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/ops/pypto-op-perf-tune/tune-orchestrator/references/phase-handoff.md) | 交接摘要和新会话的上下文边界 |
| [optimization_catalog.md](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/ops/pypto-op-perf-tune/shared/optimization_catalog.md) | 37 个优化方向及其原始编号：F-1…F-15、S-1…S-13、I-1…I-9 |
| [官方 PyPTO plugin 的 AGENTS.md](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/plugins-official/pypto-op-orchestrator/AGENTS.md) | 选取 Stage 7 的 dispatch、验证和 accumulated_context 管理规则 |
| [pypto-op-optimizer.md](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/plugins-official/pypto-op-orchestrator/agents/pypto-op-optimizer.md) | 单阶段执行角色、阶段输入输出、不得自行跨阶段或加载 debug skill |
| [pypto-docs-search/SKILL.md](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/ops/pypto-docs-search/SKILL.md) | 查询 API 和已有实现，获取配置先验 |

上游已有 [Codex optimizer 角色适配文件](https://github.com/hicann/cannbot-skills/blob/94216ba2f0b569cfad22c7794ad99426a07ab7b6/plugins-official/pypto-op-orchestrator/agents/codex/pypto-op-optimizer.toml)，
但一个角色 TOML 并不能替代完整的阶段调度和工具适配。
另有第三方 [McgAtom/codex-cannbot](https://github.com/McgAtom/codex-cannbot/blob/71e71449ed7abaf6ec7264d0298f79c7bca8b306/README.md)，
其 README 也明确把 skills 移植与官方团队工作流/hooks 区分开。
本设计以本次检查的上游性能子系统为依据，不以第三方的旧打包版本为主来源。

## 2. 抽取边界

PTOBench 的任务已经是“在固定契约下优化现有 Candidate”。因此直接选取性能阶段，
不把整个“从需求到新算子”的开发系统作为 baseline 接进来。

| 上游路径，相对 cannbot-skills 仓库 | 处理方式 | 原因 |
| --- | --- | --- |
| `ops/pypto-op-perf-tune/` | 原样冻结技能、脚本、references、cases 和必要模板依赖；保留原相对路径 | 原始性能方法就在此处 |
| `plugins-official/pypto-op-orchestrator/AGENTS.md` 的 Stage 7 | 保留完整上游文件，选用原 Stage 7 路由、验证和交接规则；不改写章节 | 其他 Stage 属于开发流程 |
| `plugins-official/pypto-op-orchestrator/agents/pypto-op-optimizer.md` | 原样使用角色约束和阶段输入输出 | 优化执行者与流程控制者的职责明确 |
| `ops/pypto-docs-search/` | 原样保留检索技能与其依赖；检索资源版本作为实验环境记录 | 原方法主动查 API、查 production kernel 先验 |
| `ops/pypto-api-explore/` | 首版不作为独立必经阶段 | 主要服务开发前 API 可行性分析；优化中 API 问题由 docs-search 承接 |
| `ops/pypto-op-knowledge/` | 首版不把完整经验预检或错误处理流程注入优化执行者 | `experience_preflight.md` 明确属于 Stage 5；性能技能直接依赖的是性能 catalog 和 docs-search |
| Golden/precision、intent、plan/design/develop、op-verify 等技能 | 不接入执行流程 | Case、Oracle、Fixture、Comparator 与最终验证由 PTOBench 提供 |
| general-debug/precision-debug 团队 | 不接入首版 | 原 optimizer 明确不加载 debug 子技能；不额外赋予 baseline 一套新调试方法 |
| OpenCode 状态 hook、全项目安装/构建/换卡流程 | 不整套安装；只移植必要性能流程语义 | 当前用 Codex/Pi，环境、设备、预算已由实验固定 |
| AscendC、TileLang、Triton、其他模型开发/Git 工作流 | 不接入 | 与本次 PyPTO 优化方法无关 |

性能 catalog、原始调优指南中的经验和例子属于 CANNBot 自己的方法资料，应该保留。
不用我们的 PHK 替换它们，也不要求 CANNBot 采用 Ours 的 hypothesis schema。
选择性能入口只决定本次调用哪个上游入口，不删减 catalog、改写角色或
拼装新技能。完整来源文件及其 hash 保留，启用范围单独记录。

## 3. 保留的搜索方法

采用官方 plugin Stage 7 的跨阶段 dispatch 规则；领域技能负责阶段内部分析和迭代。
其中 `tune-orchestrator` 的 ITER、Todo 和退出条件供阶段执行者使用。
性能主技能与 plugin 在角色和路由描述上存在差别，不把它们的整份指令同时混装：
这层关系由上游 Stage 7 和 optimizer 角色本身规定；接线程序按原关系提供工具，
不把两份路由规则重新编写为我们设计的控制策略。

```text
INIT → S1_SETUP → S2_COLLECT → S3_ANALYZE
     → S4_FRONTEND
     → S4_SWIMLANE_1 → S4_INCORE_1
     → S4_SWIMLANE_2 → S4_INCORE_2
     → S4_SWIMLANE_3 → S4_INCORE_3
     → S5_REPORT

共享预算用尽，可在任意位置结束探索，再执行共同的 finish。
阶段正常退出则按原顺序前进，不保证有限预算能走完所有阶段。
```

通俗地说：先检查程序写法和切块；再看真实执行中核是否用满、任务是否均衡、
调度是否产生空档；最后处理核内搬运、缓存和流水等问题。

必须保留：

1. 先分析再生成本阶段调优点清单；选择有分析依据的原 catalog 项。
2. 每次做一个优化点或参数变化，验证，记录；精度失败或性能回退时回滚。
   不把这一原则扩大成“所有场景只能改一个标量”：领域技能中的明确例外，例如权重 cache policy 的批量处理，仍按原文处理。
3. FRONTEND、SWIMLANE、INCORE 的连续无提升阈值分别为 5、8、5；
   退出还要核查清单标记，不能仅到阈值就强行跳阶段。
4. 编译/运行/精度失败属于失败累计，不冒充有性能结果的无提升轮次。
5. SWIMLANE 和 INCORE **每次进入都重新采集当前源码的 profile**，不复用上一阶段的旧图。
6. 阶段摘要保留已采纳/失败优化、约束、当前配置和精简趋势；下一阶段使用独立会话。
7. 配置搜索保留“从已有实现取先验，再在优秀配置附近局部变异”的方式。
   原文的“约 10 个初始候选”受剩余预算约束，不能变成 10 个免费试验。

共同实验输入和外围约束：

- Case、当前源码、缓存初始性能、预算和实验目标作为任务输入提供，和 Basic 使用同一口径。
  原流程对目标、任务参数和运行条件的检查由 CANNBot 自己执行；接入侧不改写原规则，
  不替方法填写任意达标值或增加方法内部预检。
- 原默认 12 小时、最多 3 轮外循环和核利用率/气泡率判定保持原定义。
  共享硬件预算是外部限制，到预算上限由公共层结束考生进程，不修改原阶段阈值。
- 原始初始 correctness 检查和阶段采集保留。Bench 不强制额外 baseline evaluate，
  初始性能仍取 Case 缓存；CANNBot 按原流程主动请求初始验证/采集则照常计费。
  不通过接线偷偷合并 S1b/S2 来节省预算，也不把 Ref 证明冒充 Candidate 的已验证结果。
  原流程的 Stage 6 前置检查及其处理由 CANNBot 自己负责，不成为接入侧的认证任务。
- 设备、环境、Fixture 和 Comparator 由共同实验固定。上游请求换卡、改环境或
  算法级优化时，按共同实验权限给出真实结果或限制，记录请求及结束原因，
  不从技能原文删去这些分支。

小预算可能全部花在初始采集和 FRONTEND。这是原方法在当前预算下的表现；
不能为了让它覆盖三个阶段而暗中缩短 FRONTEND、免计 profile 或重排顺序。

## 4. 统一硬件入口与成绩口径

| 公开操作 | CANNBot 与 Basic 共用的接口 | 预算 |
| --- | --- | --- |
| 初始 baseline 性能 | 读取 `.benchmark_context/reference.json` 对应缓存 | 不扣探索预算 |
| correctness / 正式性能评测 | `ptobench evaluate`，由 CANNBot 自行使用并解释返回值 | 每次接受的 evaluate 请求扣一项 |
| runtime profile | `ptobench profile runtime`，返回原始结果与文件路径 | 每次请求扣一项 |
| 需要 PMU | `ptobench profile pmu` | 每次请求扣一项 |
| 需要编译产物 | `ptobench profile compiler` | 沿用共同 CLI 计费规则 |
| 读源码、查冻结文档、分析已有 profile、更新 Todo/摘要 | 本地操作 | 不扣硬件预算 |
| 最终回归验证 | 共同的 PTOBench finish | 沿用所有方法共同的独立最终验证策略 |

三种 mode 都开放 `runtime / pmu / compiler`；CANNBot 只是按原方法选择如何使用。
诊断产生的运行时间与正式延迟必须区分：原 `analyze_perf.py` 的 `max_work_time`
是核工作时间统计，不是 PTOBench 的 clean evaluation latency。
最后成绩以相同 fixture/聚合口径的正式 evaluate 为依据；候选取舍和回退由原方法决定。
多个 fixture 下不能拿某一个 task 或 fixture 的时间替代 Case 成绩。

所有需要真实执行的测试都走共同入口；不额外运行旧项目测试命令或隐藏的 NPU 查询测试。
Case/预算/CLI 说明作为共同任务输入提供。CANNBot 自己决定如何调用和分析结果，
接入侧不替旧命令执行转换，不修改 debug_options 指导，也不将 profile 转换为旧格式。
公开 CLI 返回原始结果；方法使用过程中的请求、答复和错误原样记录。
失败预占不退款、探索预算与 finish 分开等规则完全使用现有 Session 逻辑。
原始事件仍保留执行阶段信息，后续可按论文定义区分请求预算和真正进入硬件的交互。

## 5. 接入侧不承担方法内部兼容性工作

API 选择、profile 文件读取、原脚本运行和原工具使用均由 CANNBot 自己处理。
接入侧不建立 catalog 兼容表、不预先运行分析脚本验证格式、不替换 API、
不转换或合成旧 profile 文件，也不以这些检查作为 mode 启用条件。

我们的工作是让 CANNBot 能读取原资料、启动 Pi/Codex、访问同一份 CLI，
并记录它实际做了什么、收到什么结果，以及发生的原始错误。
运行中失败照常保存，不把失败改成成功，不增加接入侧的诊断或修复流程。

## 6. 独立方法与 PTOBench 的接口

职责按功能划分，不按代码存放目录划分：

| 部分 | 负责什么 | 不承担什么 |
| --- | --- | --- |
| PTOBench evaluator/Session | 固定 Case、公开契约/context、correctness、clean timing、原始 profiling、预算、源码快照、最终验证 | 技能加载、诊断、优化点选择、阶段推进、回退决策 |
| 通用实验启动/记录层 | 根据显式入口启动一个方法程序，传入 Run/workspace/预算接口，监控进程生命周期，归档原始输出与提交的附件 | 判断 FRONTEND 是否完成、安排下一阶段、解读 CANNBot 清单或报告 |
| `modes/basic` | 普通任务说明，准备 Pi/Codex 配置，执行一次原生优化会话 | 实现独立的硬件评测或预算 |
| `modes/cannbot` | 原样上游资料/脚本、原始流程、Pi/Codex 执行接线、各阶段会话和交接记录 | 为当前 Case 改写方法知识/分析规则、修改 Bench 正确性、隐藏硬件执行、重置预算 |
| `modes/ours` | 自己的 PHK、状态分析、hypothesis search 与 replanning | 要求 evaluator 替自己选择 hypothesis |

方法程序通过同一公开接口读取 Case/预算，调用 evaluate/profile/fork，
读取原始返回值和附件，并提交自己的原始执行档案。
PTOBench 可以保存方法身份和方法提供的 metadata，但不解释 catalog ID、
PHK item 或阶段状态，也不依据它们改变评测规则。
所有正式对比方法使用相同的 evaluate/profile 权限和预算政策，
不由某个方法自己的 mode 文件决定裁判允许它做哪些硬件实验。

以 Pi 为例，执行链为：

```text
通用实验启动层：创建一次 PTOBench Session，启动选定的方法入口
  ├─ basic.runner：准备普通说明 → 启动 Pi → 返回
  └─ cannbot.runner：加载上游原文 → 连接原流程所需工具
       → Pi(S1) → Pi(S2) → Pi(S3) → Pi(FRONTEND)
       → Pi(SWIMLANE) → Pi(INCORE) → ... → 返回

所有 Pi 会话 ──相同 evaluate/profile CLI──> PTOBench
方法入口结束或共享预算耗尽 ──> 通用生命周期处理与共同最终验证
```

这里 Pi 的启动、阶段 prompt、资料挂载和 session 关联由方法侧完成。
Bench 只看到一个考生进程及其硬件请求，不需要知道内部有多少优化阶段。
已有通用 Pi/Codex 进程/原生记录工具可以在执行层复用，
但不把它们变成 evaluator 必须理解的方法实现。

## 7. 方法侧 Codex/Pi 阶段会话设计

原方法不是“一份 prompt 配一个知识库”：它有阶段执行者、Todo、
交接摘要和新的 Task 会话。当前 Launcher 一个 Run 只启动一个方法进程，
不能假设 Codex/Pi 自动拥有原 OpenCode 的 `Task / state_transition / todowrite`。

`modes/cannbot/` 已实现**独立 runner + 阶段会话的宿主工具桥接**：

- 通用启动层创建一份 PTOBench Session、workspace 和全局预算，
  将公开接口信息交给方法入口；CANNBot 接收它们，不另建实验。
- 原编排者按选定的官方 Stage 7 指令请求 dispatch，外围工具启动 Codex/Pi：
  S1、S2、S3、各 S4 PHASE、S5 使用原输入输出及新原生会话。
- controller/桥接只承载原工具调用、进程和返回值；优先复用上游可运行实现。
  原流程的路由、核查、退出判断由原指令/原规则决定，不能把我们的摘要重新写成
  一套强制控制策略。分析、优化点选择、改代码和交接摘要由 Pi/Codex 按原技能完成。
  宿主执行接线在方法目录内，PTOBench 不替 CANNBot 实现阶段业务。
- 阶段 Agent 不能自己创建下一阶段，不能重建 Bench Session 或重置预算。
- Task/Todo 等宿主工具优先复用原实现；Pi 缺少同名工具时，用方法侧扩展连接原调用，
  提供阶段会话启动/答复传递和记录。原流程如何使用工具由 CANNBot 自己决定，
  不增加接入侧的方法兼容性审查，也不修改原指南。
- Stage 1–6 的开发提交门禁和整个原 OpenCode hook 不移植。
  当前上游 plugin 明确 Stage 7 使用自己的产出核查；不能宣称移植了完整原门禁。
- 下一会话只接收原 accumulated_context 所要求的信息和当前程序，
  不把整段上阶段对话作为 prompt；历史原始数据仍完整归档。
- 每个 dispatch 固定相同 model、reasoning、权限与通用配置，
  方法资源按阶段提供；记录实际 prompt 和可见资源。
- 若配置整次实验时间上限，所有 dispatch 共用该限制；
  不给每阶段重新发一份完整 wall-time 限额。

这属于将原 Task 调度连接到 Codex/Pi 的进程调度，需在接线记录中明确披露。
不依赖 backend 自主并行 Agent 功能，也不在阶段间并行硬件试验。
Basic 的独立 runner 启动一次原生会话；不用让所有 mode 强制采用这个控制器。

方法 runner 在阶段 Pi 退出后继续存活并处理交接。
通用 Launcher 不观察阶段退出，不安排下一阶段，
只在整个方法进程结束、共享预算耗尽或中断后执行共同 finalization。
方法 runner 负责正常关闭其子会话；公共生命周期机制仍需能清理归属本 Run 的
整个进程组，包括异常退出遗留的原生 Agent/evaluator。该机制对任意方法相同。
CANNBot 启动阶段 Agent 时继承方法的进程组，不使用 `start_new_session=True`；
当前协议不支持脱离组的 Agent workers。evaluator 的独立执行组继续由 Bench 管理。

方法侧会话收集保存每个 dispatch 的独立日志、原生 session IDs、配置和归档状态；
不能覆盖之前的 `stdout.log / agent/events.jsonl` 或只收集最后一次会话。
方法提交会话附件清单，通用记录层只核对文件关联/完整性并归档原始字节，
不解读阶段业务。缺失某一阶段的原生会话仍应被明确报告为采集不完整。
tokens/turns/cost 后续从所有原生会话计算，不在 Bench 再造统计系统。

只加载原 skills、一个会话跑到底的版本可以用于接线验证，
但改变了原上下文隔离，不能不披露就作为本设计的正式 CANNBot baseline。

### Pi Task 工具的具体接线

本机 `@earendil-works/pi-coding-agent` 提供扩展接口 `registerTool`，以及
`examples/extensions/subagent/index.ts`：后者通过独立 `pi` 进程实现隔离上下文，
读取 JSON 事件并返回阶段结果。该扩展是示例，并非默认已启用的 CANNBot Task。

实现放在 `modes/cannbot/bridge/pi_task.ts`，在 CANNBot 会话通过
`--extension` 显式加载，不全局安装。主会话注册 Task/skill/Todo，阶段会话
仅注册 skill/Todo。扩展提供阶段任务输入和答复返回，
角色引用定位到冻结的上游 `pypto-op-optimizer.md`，不使用示例中的 scout/planner 等角色。

1. 主 Pi 会话按原编排说明生成阶段任务和原 accumulated_context，调用 Task。
2. 扩展保存原 tool-call ID、输入 prompt/角色路径及当前 Run/源码关联，启动新 Pi 进程。
3. 子 Pi 使用相同 model/reasoning/provider 和共同评测权限，直接加载原 optimizer 角色
   和其 mandatory reads；只接收原 dispatch 信息，不恢复主会话或复制其完整历史。
4. 子 Pi 的 cwd 指向同一已注册 Candidate workspace，并继承同一 PTOBench Session
   环境/进程组；独立会话不自动产生新的 workspace 或硬件预算。
5. 收集子进程的原生 JSON 事件、stdout/stderr、原生会话 ID 和最终答复。
   将阶段原始答复或真实错误作为 Task 结果交回主 Pi；主 Pi 按原规则核查并决定后续 dispatch。

示例默认子命令含 `--no-session`，不能直接用于正式记录；本方法的接线要使用
Run 内持久化 `--session-dir`，为每次 dispatch 记录独立原生 session ID 和日志。
命令应复用实际冻结的 provider/model 配置，不只调用一个可能使用其他默认模型的裸 `pi`。
子进程不脱离方法组，中断/预算结束能一并清理。原阶段执行者不启用跨阶段 Task 调度。
桥接不增加自动摘要、优化排序、独立成功判定或隐式重试；交接摘要由原角色生成。

扩展已实现并通过真实 Pi 加载器与受控原生进程验证。Codex 的局部 stdio MCP
暴露相同工具，并通过本机配置解析和 JSON-RPC 调用验证。验证不调用模型。

## 8. 代码和资料组织

实际布局如下，Basic、`_shared`、CANNBot 和通用入口均已实现：

```text
modes/
  PROTOCOL.md                  # 已实现的 preflight/prepare/run/collect 协议
  _shared/                     # 执行侧 native readiness/profile/配置/原生记录工具
  basic/
    mode.yaml                  # 方法身份与入口声明
    runner.py                  # 普通说明 → 一次 Pi/Codex 会话
    instructions.md
  cannbot/
    mode.yaml                  # implemented: true
    runner.py                  # 接收公开任务参数，加载原文并启动原流程
    runtime.py                 # 原生会话、源代码关联与原始答复
    tools.py                   # 局部 MCP / Pi helper；Task、skill、Todo
    resources.py               # 原文字节核对与冻结
    README.md                  # 运行与实际档案布局
    DESIGN.md
    entry.md                   # 原始入口/角色路径声明；共同任务说明与原文分开
    upstream.lock.json         # 实际上游 commit、文件 hash、冻结时间
    upstream/                  # 运行时直接使用原文原脚本；保留层级、LICENSE/声明
    bridge/                    # Pi 阶段会话启动、Task/Todo 调用承载与记录
  ours/                        # 独立 runner、知识、分析和决策实现

launcher/
  ...                          # 通用启动：一个方法入口、生命周期、原始档案收集
evaluator/
  ...                          # 裁判：Case、预算、evaluate/profile/finish、原始附件
```

运行时直接读 `upstream/`，不建立 `adapted/` 优化指南，也不移植新的分析器。
上游文件逐字节核对 hash；共同 Case、预算、工具权限和启动参数放在单独任务输入中。
每个阶段记录实际任务输入、原生答复和 CLI 调用；公共 CLI 通过 PATH/Session 环境直接开放。

当前模式 manifest 使用 schema 2：`id / implemented / description / entrypoint`。
`launcher/methods.py` 只解析声明并调用四个操作，Basic 的 runner 自己组装 prompt，
原 native readiness、profile、配置与录制代码已迁入 `modes/_shared/`。
原 `evaluator/experiment_modes.py` 已移除；evaluator 只保留不透明方法 ID 与共同评测政策。
现有 Basic 的实验 YAML/backend/profile 配置保持兼容。

CANNBot 已通过同一入口接入：
不要在 evaluator 添加 controller 导入或资源检索逻辑，
也不要按 `mode == cannbot` 在公共 Launcher 中写阶段分支。
Basic 迁移只是移走职责，没有改变优化要求或成绩规则。

方法 runner 自行准备并声明**实际暴露的**冻结资料，放在独立的方法 context 中；
通用记录层保存清单、文件和 hash，不替方法选择、加载或解释内容。
Bench 自己提供的 Case/预算 context 与这些方法资料分开。
方法临时报告与 Todo 位于独立 work dir，
不混进 Candidate 的源码快照。不要全局安装 CANNBot skills 影响其他方法。

## 9. 资料冻结和实验边界

原 docs-search 能同步最新 devkit/docs/tests/实现，原技能和检索步骤保持原样。
Run 已记录冻结的 CANNBot 文件版本和共同环境源码版本；动态检索的外部资料
依赖原生会话记录及 Agent 保存的文件，接线程序不代替 docs-search 同步资料。
资料可见范围和在线访问采用共同实验政策。
若原流程要求同步，保留请求和实际结果，不从原文删去该步骤，也不伪造同步成功。
共享政策使该依赖不可用时明确报告，不能用我们另写的检索指南替代原方法。

当前冻结清单包括上游 skills commit、公开原例子、选择依据及逐文件 hash，
接入代码也原样复制到 Run。动态取得的 API 文档不自动成为已冻结语料；
其完整版本/cutoff 需要正式实验的公共资料政策另行定义，不能由接线程序补造。
原指南会搜索 production kernels，不能随意删掉这一能力，也不能让它直接检索整个主机仓库。

要检查公开案例语料与评测 Case 的重合；保留允许的原案例并明确声明重合政策。
排除 sealed Ref/Golden、其他实验生成的优化轨迹、我们的 PHK 和 Ours 决策资料。
若移除原案例，应在 selection manifest 记录，而非只展示一个干净的 corpus hash。

各方法及其共用执行工具冻结实际项目指令、宿主 skills 和扩展发现配置，
通用实验设施验证共同访问规则与文件档案，不给某个方法提供专属知识装配。
资源放在不同目录只是一种组织方式，不能自动证明某方法没有看到其他方法的知识；
现有配置归档也不能代替真实的暴露范围控制。

## 10. 需要新增保留的原始信息

PTOBench 已有源码、raw measurements、profile 附件、错误、事件和原生会话归档。
接入时只补会丢失的方法执行信息，不重复存完整派生曲线和指标：

| 信息 | 谁产生 | 保存方式 |
| --- | --- | --- |
| 本 Run 实际使用的上游原文/脚本、接线版本与 hash | 方法自己的资源加载器 | 方法 context + manifest，由通用设施归档 |
| 每次 dispatch 的阶段/轮次、输入 prompt、可见资源、workspace | 方法 runner/controller | 方法原始记录，裁判不解释阶段 |
| 每阶段全部原生对话和工具事件 | Codex/Pi，由方法侧关联 | 多会话原生档案，不只末尾摘要 |
| 调优清单、原 catalog ID、分析依据、尝试/不适用原因 | 阶段 Agent | 原 Todo/阶段输出，绑定源码和 action IDs |
| 交接摘要、退出原因、恢复/回滚动作 | 阶段 Agent/controller | 原始摘要和路由记录 |
| 实际看到的分析报告及其输入文件 | 方法分析器 | 输入 action/file hash + 报告 + 脚本版本 |
| 原始源码、correctness、latency samples、错误及硬件阶段 | PTOBench | 继续使用共同 action archive |

Todo 与摘要里的性能数字是 Agent 实际看到的信息，因此应保存原样；
正式成绩仍以 evaluate 档案为准，不由 Agent 自报数值决定。
transformation patch、budget curve、interaction-to-best、失败率等可后处理产生，
不要求 CANNBot 再填一份 Ours 风格的结构化 hypothesis。

## 11. 实现顺序和验收

1. **原文资源接入**：加载固定版本的原始角色、技能、脚本和例子，记录逐文件 hash。
   取消独立 adapted 指令集，不给 Pi 注入我们的新优化建议。
2. **独立入口接线**：实现下表四个操作，在 cannbot runner 内承载 Task/Todo 等工具及多原生会话关联；
   公共 Launcher 只管理方法程序生命周期，统一退出后 finish。
3. **公开 CLI 暴露**：向主/子会话传递相同 PATH、workspace 和 Session 环境；
   CANNBot 自己使用 evaluate/profile/fork，接入侧不添加命令或 profile 转换层。
4. **原始记录**：保存每次 dispatch 输入、全部原生会话、原始答复/报告/错误及 action/source 关联。
5. **执行机制验证**：验证资源原样加载、子会话启动/返回、CLI 可见、预算共享、
   中断清理、记录完整性和共同 finalization。验收不含旧 API/profile/原分析器兼容性检查。

| 入口操作 | CANNBot 自己要实现的工作 | 可复用内容 |
| --- | --- | --- |
| `preflight` | 检查资源存在、所选 backend 可启动和记录配置；不检查方法内部依赖/API/profile | `_shared` 的 profile/native readiness/model 固定 |
| `prepare` | 原样挂载上游资料/脚本，记录版本、启用入口与任务参数；接收公开 Case/预算 context | 同一 records/archives 协议，无须扩展 evaluator |
| `run` | Pi/Codex 按原指令执行；承载阶段会话并直接开放公共 CLI，原结果由方法自行使用 | profile 参数渲染；共享 evaluate/profile/fork CLI |
| `collect` | 汇集所有原生会话、原始工具调用及 dispatch/action/source 对应、报告、Todo、交接和退出原因 | 通用档案声明和原生记录工具 |

每个 dispatch 至少持续保存原始输入 prompt、模型/权限配置、原生 session ID、
stdout/stderr、对应源码 hash 和返回摘要，位于方法自己的 `method/dispatches/`。
当前 `_shared/recording.collect` 可以帮助收集同 workspace 的原生文件，
阶段关联由方法侧桥接维护，交接摘要由 Pi/Codex 按原技能生成并原样保存。
整体原生事件可原样转发到方法 stdout，阶段独立事件也保留到方法目录，
避免只留下最后一个阶段。collect 有 60 秒超时，主要工作应已在各阶段落盘。

正式启用至少满足：

- 运行中实际加载的上游技能、角色、catalog、脚本和案例与固定版本逐字节一致。
- 没有 adapted 指导、新优化建议、旧 API 替换或另写的分析器；执行接线的实际差异可追溯。
- Codex/Pi 两条路径执行相同阶段协议，均完整保留全部原生会话。
- evaluate/profile 永远使用共享预算；新阶段不重置预算，诊断也没有免费 NPU 执行。
- 初始性能从 Case cache 获取，合法候选和最终成绩由共同评测判定。
- CLI 原始结果与对应 action/源码保持关联；正式 latency 与诊断时间区分。
- 运行失败原样保存，不以方法兼容性检查替代真实执行。
- Basic 的单会话路径及其现有功能/档案格式保持兼容。
- evaluator 不导入任何方法 controller/分析器，也不理解其知识与优化策略；
  方法侧的资料和工具连接不需要修改 evaluator 或公共 Launcher 的业务逻辑。

论文应注明使用了哪个固定版本和 PyPTO 性能入口，以及由 Pi/Codex 执行的接线方式。
记录原文件版本、实际执行方式和共同实验限制，
不要声称已完整复现上游新算子开发系统。
