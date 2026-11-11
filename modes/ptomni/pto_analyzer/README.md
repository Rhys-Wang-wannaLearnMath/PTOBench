# PTOAnalyzer

> 状态：Stage 3 已完成工具收敛与有界 Cursor 分页。当前对外提供八个只读 Tool（Compute 三个、Dataflow 两个、Schedule 三个），保持角色隔离、全量摘要和来源覆盖。Stage 3 离线回放覆盖 16 份既有 Profile，未新增硬件动作。PTOAnalyzer 是 PTOmni Guidance 的分析模块，不是新的 PTOBench Mode、Profiler 或 LLM Agent。

## 1. 定位

PTOAnalyzer 把一次 PTOBench Profile 的不可变产物整理成可追溯的性能观测、来源覆盖和解释边界，供已启用的 Tuner 参考。Tuner 自行判断是否提出假设并设计下一项实验。Lead 不调用 PTOAnalyzer；它只根据当前共享状态和历史结果制定 RoundPlan。

```text
Lead 读取现有共享状态并发布 RoundPlan
  -> 被启用的 Tuner 检查已有 Profile
  -> 如计划允许，Tuner 经 Runner 请求一次 PTOBench combined profile
  -> Tuner 调用自身专长的 PTOAnalyzer Tools
  -> Tuner 经 Runner 调用 evaluate 验证候选
```

| 模块 | 负责 | 不负责 |
|---|---|---|
| PTOBench | Profiling 采集、归档、硬件动作计数 | 推断优化策略 |
| Runner | 调 PTOBench CLI、设备租约、预算和执行记录 | 解析指标、判断瓶颈 |
| PTOAnalyzer | 离线解析并组织可追溯观测；按专长向 Tuner 提供只读 Tools | 访问 NPU、改代码、评正确性、提出性能结论、记第二套账、向 Lead 暴露额外 Tool |
| PTOmni | Lead 依据共享状态制定计划；Tuner 决策并执行代码实验；Assembler 决定最终交付候选 | 改写 PTOBench 原始事实 |

PTOAnalyzer 的工具围绕具体问题组织证据，分别给出可追溯的观测事实、数据覆盖和解释边界；Tuner 据此自行判断下一步。

## 2. 与当前 PTOBench 的接口

- PTOBench 支持 `combined` 和显式 `split`；默认使用 `combined`，使一次 `ptobench profile [WORKSPACE]` 请求采集 Runtime、PMU 和 Compiler 诊断。需要对照实验时，mode 配置仍可指定 `split`。见 [`profile_modes.py`](../../../evaluator/profile_modes.py) 与 [`AGENT_CLI.md`](../../../evaluator/AGENT_CLI.md)。
- 成功的 `profile` 是一次 PTOBench 硬件动作，combined 仍计一次；它不做 Ref 正确性比较，也不产生 Clean Latency。失败或中断成本仍以 PTOBench 账本为准。`evaluate` 才是候选正确性和正式 Clean Latency 的权威来源。
- PTOAnalyzer 只接收 Runner 提供的确切 Profile 归档引用（至少包含 Session、事件序号和 Candidate 身份），从该事件的受保护归档读取数据。不得扫描 `.benchmark_outputs` 猜“最新结果”，也不得修改或复制覆盖 PTOBench 原始记录。
- PTOmni Host 可将分析结果存入派生状态并引用 Profile 事件和原始文件；PTOAnalyzer 本身不持久化结果，重跑分析也不触发硬件动作。它不维护 Profiling 次数、设备锁或硬件预算。Lead 读取已有状态中的结果，不直接调用 Analyzer 或 Runner。

已有 [`post_baseline_combined_20261106`](../../../manual_runs/post_baseline_combined_20261106/queue.json) 记录了 16 个通过的 Case Profile，每个包含 runtime、PMU、compiler 产物且各计 1 次硬件动作。这是第一批离线样本池，不代表已有优化前后配对，也不能单凭它证明诊断能找到性能根因。

## 3. PTOAnalyzer Tools

Stage 3 前，PTOAnalyzer 曾按每个 Tuner 三个 Tool 的目标实现九项候选诊断。Stage 3 撤下了没有可计算 overlap 观测的 `pipeline_overlap_check`；当前暴露八个 Tool：Compute 三个、Dataflow 两个、Schedule 三个。Tool 对应一个可复核的问题，而不是一个原始文件或指标。一次 `combined` Profile 供所有工具共用；Tuner 不为每个 Tool 分别采集 Profile，也不必读取整份原始产物。工具按问题返回观测、来源和适用边界，不在内部调用 LLM，也不替 Tuner 决定修改方案。

内部共享受保护归档读取与格式解析；对外只向被启用的 Tuner 提供角色专属工具。当前八项工具均为确定性、只读诊断，以观测、来源覆盖和解释边界呈现事实，不输出工具级性能判决。

| Tuner | Tool | 诊断问题 |
|---|---|---|
| **Compute** | `compute_mapping_check` | 可用的 dispatch program、callable name 与 kernel config 映射是什么；Candidate 源码位置关联是否经过验证 |
|  | `compute_unit_activity` | 各 build/rank/dispatch/core 的 PMU 活动计数是什么；单一计数不判定计算瓶颈 |
|  | `compute_task_share` | 哪些 task 的 PMU 计数能与依赖图、callable name 和 AICore span 在同一 dispatch 内可靠关联 |
| **Dataflow** | `tile_access_check` | 编译器报告了哪些 Tile-access 提示 |
|  | `buffer_pressure_check` | 是否有 MemoryReuse 编译提示，以及 Scope 统计中有哪些原始计数 |
| **Schedule** | `dependency_path_check` | 依赖图有多少 task、edge、图深度和端点缺失；该图不含 runtime 权重 |
|  | `dispatch_gap_check` | 在完整 AICore/AICPU 覆盖的同一 rank/dispatch/core 内统计原始周期上的观测间隔，不归因调度开销 |
|  | `task_distribution_check` | 按 rank/dispatch/core 汇总 AICore 物理记录、原始 task token 与区间周期；不推断忙碌时间或负载不均 |

`ToolSession.initial_summary` 会运行当前 Tuner 获准的全部诊断并给出全量摘要，`call(name)` 可重复定向调用并分页读取高基数明细。Dataflow 当前摘要含两个工具；其他两个 Tuner 各含三个。缺失或不适用证据由来源覆盖与限制说明。分析完全离线，不产生新的硬件动作。

## 3.1 已实现接口与边界

```python
from modes.ptomni.pto_analyzer import ProfileRef, open_toolset

profile = ProfileRef.from_event(session_root, event_seq)
session = open_toolset("Compute", profile)
session.tool_specs
session.initial_summary
page = session.call("compute_unit_activity", {"limit": 20})
session.call("compute_unit_activity", {"cursor": page["details"]["next_cursor"], "limit": 20})
```

`ProfileRef.from_event` 绑定调用者指定的事件序号，不搜索最新 Profile。创建 Session 时会校验 session/event 身份、已完成的 combined Profile、Candidate digest、artifact index 和 PTOBench protection manifest；被索引文件在读取时再次核对哈希。越界路径、损坏归档或错误身份会拒绝分析，Tuner 不能通过参数传入文件路径。静态权限表仅开放 Compute 的三个、Dataflow 的两个、Schedule 的三个工具；Lead 无工具。高基数明细使用不透明、版本化的 Cursor，绑定 Tool、Tuner、Profile event 和 Candidate digest；Cursor 只表示稳定排序后的续读位置，不选择归档、文件路径或权限。每页默认 20 条、最多 50 条。结果包含 Profile 身份、全量观测、单位、来源覆盖、来源引用和解释限制；不使用混合数据采集状态与性能结论的单一状态字段。

分页只把明细候选缓存限制在当前 `limit`，不限制 Parser 对输入文件的临时工作集。Stage 3 对最大单卡依赖图（453,375 条明细）的三页连续读取各耗时约 29 秒，`tracemalloc` 峰值约 0.87–0.95 GiB；双卡与 8 卡代表样本的依赖图页约 1.1 秒和 2.6 秒。该成本来自完整依赖图解析与汇总，不是 50 条候选缓存；是否需要 Session 内部缓存留待确认真实 Agent 按需查阅受阻后再评估。

## 3.2 既有 Profile 回放结果

Stage 1 的历史回放报告保留在 [`reports/offline_replay_20261106.json`](reports/offline_replay_20261106.json)。Stage 2 的当前回放结果见 [`reports/offline_replay_stage2_20261108.json`](reports/offline_replay_stage2_20261108.json)：覆盖 16 个 combined Profile，并独立抽查单卡、双卡、8 卡样本。报告记录索引归档 SHA-256 前后核对、原始计数复算和每个工具的来源覆盖；不新增硬件动作。

Stage 2 回放确认 PMU 解析覆盖数按来源只计一次；`compute_task_share` 的关联成功/失败记录另行累计，PMU 行数不会因为后续 join 再加一次。PMU counters 按 build/rank/dispatch/core type 分组；关联任务的 AICore 区间周期与 PMU counters 保持不同单位。

Stage 3 最新离线回放重新核验了 16 份归档、128 个工具摘要及分页调用；索引文件共 5,492 个，回放前后 SHA-256 一致，新增硬件动作为 0。摘要与第一页的 observations/source coverage、Stage 2 的来源覆盖和明细数均通过核验。`buffer_pressure_check` 的 observation 数相对 Stage 2 在 16 份 Profile 中均较多；差值随设备覆盖从 9 到 72 变化。Stage 2 报告只保留 observation 数量和来源覆盖，没有保存 observation 名称和值，因此无法逐项证明历史值完全一致；当前差异和限制记录在 Stage 3 报告中。

`dispatch_gap_check` 只对 Level 4 AICore 区间周期排序，并要求同一文件内 AICore 与 AICPU 记录按 core/register-task identity 一一覆盖；它输出同核相邻区间的正间隔，不解释原因。Stage 2 的九工具回放中，`pipeline_overlap_check` 没有 overlap 观测；Stage 3 检查了运行时轨迹、合并 trace、PMU 与编译器契约后将它从当前接口撤下。PMU 仅有 MTE busy-cycle 计数，Runtime/merged 轨迹没有可配对的 MTE 起止区间，编译 pipeline profile 记录的是编译 pass 时间。若未来需要恢复该 Tool，先由 PTOBench 提供同 Candidate、设备与采集窗口下、公共时间基准且可关联 task/block 的 Compute/MTE 起止区间。`compute_mapping_check` 仍不把 callable 名称映射到 Candidate 源码位置。Stage 3 验证与字段缺口记录见 [`reports/offline_replay_stage3_20261108.json`](reports/offline_replay_stage3_20261108.json)。

回放结果只用于检查解析覆盖和证据可用性。Profile 不提供 correctness 结果或 Clean Latency；编译提示、PMU 计数、依赖结构和时间跨度都不能单独证明性能瓶颈。PTOmni 的性能判断仍需 `evaluate`，Analyzer 是否改善优化效率需另做受控效用小试。

工具权限按 Tuner 专长预先配置。Lead 只输出 RoundPlan，不持有 PTOAnalyzer、Runner 或 PTOBench Tool；RoundPlan 可以限定 Tuner 可请求的 Profile/evaluate 类型和预算，但不启动动作。被启用的 Tuner 检查已有 Profile；若证据不足且计划允许，可经 Runner 请求一次 `combined` Profile，再调用自己的 Analyzer Tools。跨机制发现写入 Tuner 的 ExperimentResult，由 Lead 在后续计划中决定是否启用相关 Tuner。Assembler 消费 Tuner 的 ExperimentResult 和 PTOBench 验证结果，不负责分发 Analyzer 工具。

每个 Tool 采用同一事实契约：来源覆盖、可复算观测、单位、范围、聚合口径、样本数、来源引用和限制。缺失证据不会填成性能零值；缺失、空文件、不可读文件和格式/记录问题分别在来源覆盖中表达。未经校准不输出严重度、综合分数或预期加速比。Candidate 源码位置只有在当前快照关联经过核实时才显示。

Tuner 专长不是原始数据类别：同一条证据可由多个小工具以不同相关视角提供。例如，Tile 改动可能同时影响 Dataflow 与 Schedule；PMU task 计数也只有在 task identity 和采集范围确认一致时，才能与 Runtime task 关联。一次 combined Profile 共享 Candidate 和 PTOBench 事件身份，不代表不同类型的计数天然来自完全相同的时间窗口；单位、采集范围或版本不兼容时只并列呈现，不做跨源运算。

第一批工具应优先覆盖真实样本中稳定、可追溯的字段。单一 PMU 利用率不能证明计算瓶颈；单项 MTE 计数或静态内存占用不能证明数据供给瓶颈；task 多、短任务多或存在依赖边也不能单独证明调度瓶颈。缺失、不支持、空文件和解析失败必须分别说明，不能解释成零或“没有问题”。

## 4. 验证安排

按“先看真实数据、再定规则”的顺序推进。前三阶段不需要新的 NPU 运行。

| 阶段 | 工作 | 通过条件 |
|---|---|---|
| P0：样本盘点 | 从现有 16 个 combined Profile 中选 6–8 个代表 Case；检查 Candidate、Shape、设备和软件版本身份，并确认原始归档完整性。若已有可比的优化前后版本则纳入；没有就不补跑。 | 形成字段清单、来源关系和可比性记录 |
| P1：数据审计与 Fixtures | 检查 Compiler、Runtime、PMU 的真实格式、字段单位、Task/依赖关联及缺失情形；将少量必要文件做只读 Fixtures，并另设缺失/部分采集样本。 | Parser 输入来自真实产物；Fixture 不改写原始归档 |
| P2：离线工具 | Stage 2 实现九项候选诊断；Stage 3 根据实际证据暂缓无 overlap 观测的 Dataflow Tool，当前接口为八项。 | 当前权限表、摘要和文档一致；明细可完整分页，结果有来源且缺失语义正确 |
| P3：人工复核 | 用 6–8 个 Case 检查每个工具结果与原始记录是否一致，验证它返回的信息确实对应该 Tuner 的职责。 | 无无来源数值、无把相关性写成确定根因的结论 |
| P4：效用小试 | 从 Formal/Held-out 清单中选择少量代表性 Case，对比“Agent 直接查看原始 Profile”与“Agent 使用按角色开放的 PTOAnalyzer Tools”；固定模型、起始 Candidate、PTOBench 预算和其他 Guidance，只改变 Profiling 信息接口。两组复用相同 Profile 配置与采集次数。 | 按 Case 配对检查性能、有效优化所需硬件动作、无效/重复尝试、Token 与墙钟时间；确认小工具是否减少无效尝试 |

P4 是 PTOAnalyzer 的工程效用验证，不自动扩展论文主实验或新增消融。若未来要把 Analyzer 的独立效果作为论文主张，再根据小试结果另行设计有统计把握的实验。

实际调用 `profile` 和 `evaluate` 都按 PTOBench 账本计硬件动作；离线解析、Fixture 测试和人工报告复核不计硬件动作。P4 的正式性能仍用 `evaluate` 测量，Profile 指标只作诊断证据。

## 5. 首版范围

- 当前 8 个离线只读诊断 Tools，按 Compute / Dataflow / Schedule 专长分配（3/2/3）；Lead 不持有 Analyzer Tools。共享归档读取和解析，不做重复的底层实现。
- Parser 只规范化首批工具需要的字段；大型 IR、Timeline 默认按需读取相关片段，不整包塞给 Tuner。
- 初期不做 Profile 自动选择器、动态规则引擎、学习型阈值、数据库、自动 Case Memory 写入或新的硬件采集器。
- PyPTO 的 `critical_path`、`sched_overhead_analysis` 等既有工具，只有在确认当前环境版本、输入要求和离线可用性后才考虑复用。

### 初步模块架构

PTOAnalyzer 的对外 seam 是一个绑定 Tuner 身份和 Profile 身份的 `ToolSession`。它隐藏归档定位、字段解析、角色授权和诊断规则；调用方不需要了解 Compiler、Runtime、PMU 文件格式，也不能传入任意文件路径。

```python
session = open_toolset(tuner, profile_ref)

session.tool_specs       # 仅该 Tuner 可见的工具定义
session.initial_summary # 该 Tuner 获准工具的全量摘要与来源覆盖
session.call(name, args) # 定向读取，返回结构化观测、覆盖情况与限制
```

`profile_ref` 由 Runner/执行状态层传入，绑定 PTOBench 事件、Candidate 和受保护归档；它不是 Agent 可编辑的路径参数。创建 session 时验证事件与 Candidate 身份，身份或 manifest 无法核实时拒绝建立 session。各产物按需独立读取；单个产物缺失、为空或解析失败只影响依赖它的诊断，不丢弃其他来源的证据，并在 `source_coverage` 中给出具体文件和记录问题。观测为空不能被解释为性能为零或根因结论。

```text
Runner 提供 ProfileRef
  -> open_toolset(Tuner, ProfileRef)
  -> 固定权限表选择该 Tuner 的 ToolSpec
  -> 对该 Tuner 的获准工具生成 initial_summary
  -> Tuner 按需调用专属工具
  -> 只读归档按需解析 -> 确定性分析 -> 观测、覆盖情况、限制和来源引用
```

PTOAnalyzer 不连接 Runner，也不执行 PTOBench。未来的 PTOmni host 将 `tool_specs` 和 `call` 接到 Pi/Codex 的原生工具传输；Analyzer 自身不实现第二套 MCP server。现有 CANNBot 的工具传输仅作为宿主接线模式参考，不作为 PTOAnalyzer 的运行依赖。

建议的最小源码布局如下，所有 Analyzer 实现与测试都放在本目录：

```text
ptomni/pto_analyzer/
├── README.md
├── __init__.py          # 导出 open_toolset
├── toolset.py           # ToolSession、静态权限表、工具定义与调用分发
├── profile.py           # ProfileRef 校验、受限归档读取、按需解析
├── diagnostics/
│   ├── compute.py       # Compute 专属纯诊断
│   ├── dataflow.py      # Dataflow 专属纯诊断
│   └── schedule.py      # Schedule 专属纯诊断
└── tests/
    ├── fixtures/        # 从已归档真实 Profile 精简出的 CPU-only 样本
    └── test_toolset.py
```

内部诊断通过受保护的 `ProfileArchive` seam 按需读取各自需要的产物；`common.py` 提供共享解析器、证据引用和结果契约，单项诊断只提取其实际使用的字段。解析出的事实保留原始来源和单位。不同采集窗口、单位或版本无法确认兼容时，只并列提供证据，不跨源计算。首版只用 Python 标准库及 PTOBench 已有依赖，不增加框架、数据库或通用插件注册层。

`ToolSession` 的测试面就是外部 Interface：用固定 Fixture 验证角色只能看到获准工具、输出可追溯到源文件、缺失与不支持证据不会坍缩为零、相同输入重复分析结果稳定、原始归档不变。测试在 CPU 上完成，不启动 NPU 或 PTOBench CLI。

当前八项工具均为只读实现。Stage 2 的九工具历史回放见 [`reports/offline_replay_stage2_20261108.json`](reports/offline_replay_stage2_20261108.json)；Stage 3 的 Cursor、工具收敛、测试和离线回放见 [`reports/offline_replay_stage3_20261108.json`](reports/offline_replay_stage3_20261108.json) 与 [`planning/stage3/TASK_CHECKLIST.md`](planning/stage3/TASK_CHECKLIST.md)。

当前架构依据：[PTOmni 架构设计（2026-11-08）](../../PTOmni/design_docs/PTOmni_架构设计_2026-11-03.md)。Lead 只消费已有共享状态并制定 RoundPlan；被启用的 Tuner 在计划许可范围内经 Runner 请求实验，PTOAnalyzer 只向 Tuner 提供专长诊断 Tools；Assembler 综合候选。PTOAnalyzer 不改变这些角色权限。
