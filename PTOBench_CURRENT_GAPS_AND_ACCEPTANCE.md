# PTOBench 记录缺口与验收标准

PTOBench 的记录目标是能还原一次实验：**哪个 Agent 在哪个 Case 上，提交了什么代码，调用了什么评测，看到什么结果，最后选了哪个版本，并消耗了多少资源。**

只要求保存原始事实和它们之间的关联。无需数据库，也不要求 PTOBench 在线分析 Agent 行为。以下现状依据现有 Case、Launcher、Evaluator 和 Run 记录；所列补齐项是论文实验的验收要求，不代表当前实现已经具备。

| 信息 | 当前基础 | 主要缺口 |
|---|---|---|
| Case | Case 配置、输入、正确性契约和 baseline 已有 | 多层优化标签等论文元数据未统一登记 |
| Run | Session、配置、环境、预算和 Agent 记录已保存 | 信息分散，部分资源字段未统一汇总 |
| Attempt | 每次评测的源码、结果和附件有归档 | 结果阶段、代码/编译配置身份及 Diff 未统一成易分析记录 |
| Trace | Pi/Codex 原始记录及 CANNBot 子任务记录可保存 | 不同方法的记录关联方式不同，未处处精确连到评测事件 |
| Final Result | 最佳 Snapshot 和独立最终验证已记录 | 总设备时间和 Token 等成本未统一汇总 |

## 1. Case：固定任务信息

每个 Case 应能直接读取：

- `case_id`、`family`；需要按来源去重或分组时再加 `source_id`。
- 输入 `shape`、`dtype`；多组输入时引用固定的 Fixture 清单。
- `initial_source` 的路径，指出 Agent 从哪份初始代码开始。
- `editable_scope`：允许修改的源码和编译控制。
- `levels`：该 Case 实际开放给 Agent 的 Tensor、Tile、Block 优化层次，可多选；这是人工维护的 Case 标签，不是每次修改的分类。
- `eval_config`：统一引用正确性标准、测试输入和性能测量配置。
- `baseline_latency`：原始正确实现的参考性能及其测量配置。

**当前缺口：**现有 `case.json` 已定义初始代码目录、代码入口、配置、Comparator、环境、Fixture 和正确性要求；baseline 也有独立记录。每个 Run 会记录 Case 资产 hash，每次评测会记录候选源码 hash，因此无需再增加一份初始代码 hash。论文需要的 `family`、输入描述、可编辑范围、`levels` 等元数据尚未形成统一字段；评测标准和 baseline 也应能通过稳定引用找到。

## 2. Run：一次 Agent 实验

每个 Run 应记录 `run_id`、`case_id`、`agent`、`backend`、`model`、`config`、`environment`、`budget`、起止时间和 `stop_reason`。其中 `agent` 取运行模式的 `mode.id`（如 `basic`、`cannbot`），`backend` 取实际 Agent 后端（如 `pi`、`codex`）；“Basic Pi”就是 `agent=basic`、`backend=pi`。当前流程没有 Run 级 Agent 随机种子，因此不要求 `seed`；Case Fixture 的 `seed` 只用于生成固定测试输入。配置可引用文件路径，覆盖 Prompt、工具权限、知识库、Profiling 和团队策略；环境需标明 NPU、软件栈及关键版本。

**当前基础与缺口：**现有 Session 和实验档案已记录 Case、Agent/backend、配置、环境、预算、时间和结束原因，Launcher 也保存原始 Run 文件；但缺少一个便于读取的统一索引。Model、Token usage 和 Check/Bench/Profile 次数部分依赖各 Agent 原生记录或离线聚合，应统一汇总；工具未提供的值标成 `null`/`unavailable`，不要推测。

## 3. Attempt：候选版本与每次评测调用

Attempt 保存 Agent 提交的候选版本；同一候选可以被多次 Check、Bench 或 Profile。因此候选版本和工具调用都要分别留档，后一次结果不能覆盖前一次。

候选版本至少要能还原源码快照或 Diff、父版本、编译选项及版本 hash。候选身份必须区分**相同源码但不同编译选项**。Diff 可以离线从相邻快照生成，只要结果能准确重建。

每次 Check、Bench 或 Profile 调用都应记录：

- `run_id`、`attempt_id`、事件 ID/顺序、时间、调用类型和候选版本身份；
- `compile_status`、`correctness_status`、`failure_stage` 和原始错误/诊断日志路径；未执行的阶段标成 `not_run`；
- Bench 的原始 `latency_samples`、统计延迟、相对初始实现的 `speedup` 和测量配置；
- Profiling 的原始结果及附件路径；Profiling 数据不当作正式延迟成绩；
- 实际硬件调用、设备执行时间等可观测消耗。

**规则：**正确性未通过或未 Bench 时，正式 `latency`、`speedup` 为 `null`，样本为空。只有正确性通过且 Bench 完成的候选才有正式性能结果。当前每次操作已有单独事件、源码 hash、结果和原始附件，也保留失败候选；但编译/失败阶段、Diff、有效编译配置身份和资源消耗尚未统一为稳定字段。

## 4. Trace：Agent 原始交互

保存 Agent 工具已有的原始 Session、Tool Calls 和 Tool Outputs，并将它们关联到 Run、评测事件和候选版本。还应保留 Profiling 原始结果、知识库/专家调用、Token usage，以及 Check、Bench、Profile 的调用次数和设备执行时间。

PTOmni 的记录还需能还原每轮专家选择和团队切换，以支持 FixedTeam 消融。只保存运行时可导出的内容；**不要求记录隐藏推理，也不应把事后猜测的动机当作记录事实。**工具不提供的关联 ID 或资源值应明确标记不可用。

**当前基础与缺口：**Pi/Codex 原生 Session 和 CANNBot 的 dispatch、工具输入输出已有保存机制；评测事件和源码版本也有各自记录。但不同方法的关联格式不一致，未必能从一条评测事件直接定位对应的原生 Tool Call。正式实验需要验证 Session、调用 ID、事件 ID 和候选 hash 的关联链完整；PTOmni 专属的专家/团队记录也必须纳入其 Run 档案。

## 5. Final Result：Run 的最终结果

每个 Run 保存 `best_valid_source`、`final_source`、独立复测的正确性、延迟和 speedup，以及 `total_cost`：硬件调用次数、设备执行时间、Token 和 wall-clock 时间。区分：

- `best_observed`：优化过程中测得的最佳正确候选；
- `final_verified`：结束后在干净条件下独立复测的最终版本。

论文最终性能以 `final_verified` 为准。Best-so-far 曲线、有效尝试率等可从事件记录离线计算，不必在运行时重复保存。

**当前基础与缺口：**现有 `final_result.json` 记录选中的 Snapshot/hash、最佳观察值、最终独立验证及部分预算和墙钟信息；源码可从 Snapshot/最终工作区还原。Token 与设备执行时间尚未统一进入 Run 的成本汇总。

## 实验比较与分析

| 分析 | 依赖记录 |
|---|---|
| RQ1：探索哪些优化层次 | Case `levels`、候选 Diff、Session |
| RQ2：哪些修改有效 | Diff、正确性、Bench 样本和版本关系 |
| RQ3：为何失败、重复或停滞 | 完整 Attempt、Session、错误日志和 Profiling |
| 主实验与模型比较 | 相同 Case/eval 配置下的性能、时间和硬件消耗 |
| NoGuidance 消融 | Run 配置版本、工具/知识调用、最终性能 |
| FixedTeam 消融 | 专家与团队切换记录、Tokens、性能 |

对比实验应固定任务集、正确性标准、性能测量方式、硬件和预算口径；开发集与独立测试集应预先固定，不能依据测试集结果反复调整方法。

## 最小验收清单

- [ ] 每个 Case 的初始代码路径、输入、`editable_scope`、`levels`、`eval_config` 和 baseline 可追溯。
- [ ] 每个 Run 的 Agent/model/config、环境、预算、时间、停止原因和 Session 可追溯。
- [ ] 每个候选可由快照或 Diff 重建；相同源码但不同编译选项可区分。
- [ ] 每次 Check/Bench/Profile 都有独立事件，保留结果、原始日志/样本和候选版本关联；失败阶段可区分。
- [ ] 原始 Session、工具调用、Profiling、专家/团队记录（适用时）与评测事件可关联；资源消耗可统计，缺失值明确标记。
- [ ] Run 保存最佳正确版本、最终版本和独立复测结果；总成本可从 Run 记录与事件中核对。

不要求 PTOBench 自动分类 Tensor/Tile/Block 修改、分析瓶颈、解释失败或生成优化建议。这些分析应在实验后基于 Diff、Session、Profiling 和评测结果完成。JSON/JSONL、源码快照与原始日志足以承载记录，无需复杂数据库。

现有记录格式和归档位置见 [`RUN_RECORDS.md`](RUN_RECORDS.md)；本清单定义的是在此基础上的论文实验验收要求。

## 02–05 具体修缮规划

目标是让实验结束后能从现有记录回答：哪个 Agent 在哪个 Case 上做了哪些尝试、每次看到什么结果、最后选了什么版本、实际花了多少资源。修缮以补字段和打通关联为主，沿用现有 Run、事件、Session 和最终结果文件，不另建数据库或 Agent 记录系统。本规划不处理 hash、防篡改或安全审计要求。

### 02. Run：让每次实验有一张清楚的“封面”

**要补什么**

- 在现有 Run 主记录中统一放入：`run_id`、`case_id`、`agent`、`backend`、`model`、配置引用、运行环境、预算、起止时间和 `stop_reason`。其中 `agent` 取 `session.json` 的 `mode.id`（当前如 `basic`、`cannbot`）；`backend` 单独取 `optimizer.backend`（如 `pi`、`codex`）。例如“Basic Pi”由 `agent=basic` 和 `backend=pi` 表达。不要用 `optimizer.id` 代替 `agent`，它是实验配置里的自定义标识。模型名记录实际生效值，优先引用 Agent/backend 记录。
- 当前 Run 配置和启动流程没有使用 Agent 随机种子，因此不增加 Run 级 `seed` 字段。Case Fixture 中的 seed 用于生成固定测试输入，属于 Case 输入记录，不是 Run seed。
- Run 结束时汇总可获得的 Attempt 数量、Check/Bench/Profile 次数、硬件调用次数、设备执行时间、Token 用量和 wall-clock 时间。Agent 或工具没有提供的数值标记为 `unavailable`，不估算。
- 配置和环境可以引用现有文件或记录，避免把 Prompt、知识库和软件环境全文复制进 Run 文件。

**通俗地说：**Run 记录就像一场实验的封面和账单：说明谁、用什么配置、在什么机器上跑了多久，并把能统计的资源花费列出来。

**完成标准：**只看一份 Run 摘要，就能找到实验身份、配置、环境、Session 和事件记录，并读到可获得的总用量。

### 03. Attempt：每次正式操作都留一张记录

**统一口径**

- 每次 Agent 调用 PTOBench 做候选验证或 Profiling，都记为一条 Attempt；不再把候选提交、evaluate 和 profile 设计成不同层级的记录。Attempt 中保留实际操作类型，说明这次做了什么。
- 同一版本被重复检查、测量或 Profiling 时，每次调用仍各有一条记录，不能用新结果覆盖旧结果。
- `fork` 不是 Attempt，但要作为 Event 写入同一时间线，记录发生顺序、时间和可获得的父子 Run 关系。

**每条 Attempt 要记录什么**

- 所属 Run、Attempt 顺序、时间、操作类型，以及当时使用的源码快照位置；另记该版本从哪个先前版本演变而来（可获得时）。
- 本次实际编译选项、编译结果、正确性结果、失败阶段和原始诊断日志位置。没有执行的阶段写 `not_run`。
- 若执行正式 Bench，保存原始延迟样本、统计延迟、相对 baseline 的 speedup 和测量配置；如果执行 Profiling，保存原始结果及附件位置。
- 记录本次实际发生的硬件调用次数和可获得的设备执行时间。

**结果规则与通俗解释**

- 只有正确性通过并完成正式 Bench，才填写正式 latency、samples 和 speedup。编译失败、正确性失败、未 Bench 或仅做 Profiling 时，这些正式性能结果留空。
- Profiling 是一次 Attempt，但它的结果是诊断数据，不等于正式性能成绩。这样既能保留 Agent 做过的操作，也不会把 Profile 数据误算成跑分。
- Diff 可以之后从相邻源码快照生成；验收重点是能找回每次操作对应的代码和编译选项。

**完成标准：**按时间排列 Attempt 和 fork Event，可以还原每次操作用了哪个版本、做了什么、结果是什么；重复调用互不覆盖，失败也有记录。

### 04. Trace：把 Agent 看到的反馈接到对应 Attempt

**要补什么**

- 继续保存 Agent 工具原生导出的 Session、Tool Calls 和 Tool Outputs；PTOBench 不另开发对话记录器，也不记录隐藏推理。
- 尽可能使用已有调用 ID，把 Agent 的工具调用与对应 Attempt、源码版本和评测输出关联起来。若某种 Agent 不提供调用 ID，就保存实际可用的事件顺序、时间和操作信息；不能可靠关联的值明确标为不可用。
- 保留 Agent 实际看到的 Check/Bench/Profile 输出，以及 Profiling 附件。适用时同时保留知识库/专家调用、PTOmni 的专家选择和团队切换记录。
- Token 用量优先从 Agent 原生记录读取，并汇总到 Run；没有记录就标为不可用。

**通俗地说：**Attempt 告诉我们“代码做了什么、测试结果如何”；Session 告诉我们“Agent 当时调用了什么工具、收到了什么反馈”。两边对得上，才能事后研究 Agent 是否根据反馈改变了尝试方向。没有保存下来的动机不做猜测。

**完成标准：**从一条 Attempt 出发，能找到相应的工具调用和 Agent 可见输出；若工具本身没有提供可关联信息，记录中明确说明缺失，而不是伪造精确对应关系。

### 05. Final Result：明确最后交付和最佳成绩

**要补什么**

- 在现有 `final_result.json` 中分别指明：Agent 最后提交的版本、优化过程中测得的最佳正确版本 `best_observed`，以及结束后独立复测的版本和结果 `final_verified`。
- 为这些版本保留源码位置或对应 Attempt，并记录最终复测的正确性、延迟和 speedup。最终复测应作为单独评测操作留档，避免把实验过程中的一次测量当作最终确认。
- 汇总 Run 中可获得的硬件调用次数、设备执行时间、Token 和 wall-clock 时间；缺失项标记为不可用，不填估算值。

**通俗地说：**Agent 最后停下来的代码不一定是它测过的最好版本，也不一定是最后重新验过的版本。分开记这三者，论文报告时就不会把“最后提交”误当成“最佳成绩”。最终性能以 `final_verified` 为准。

**完成标准：**读者能明确找到 Agent 的最后提交、过程最佳正确版本和独立复测结果，并能核对最终结果对应的源码和评测记录。

### 实施顺序与范围

1. 先对照当前记录格式，列出上述字段已有、缺失或无法获取的部分。
2. 在现有 Run 摘要、Attempt/Event 记录、Session 归档和 `final_result.json` 中补缺失字段与关联，不新建平行记录体系。
3. 用成功验证、编译失败、正确性失败、重复操作、Profiling、fork 和最终独立复测等记录走查，确认时间线、结果和成本能够还原。

这项修缮只负责保存和关联实验事实，不让 PTOBench 自动分类优化层次、诊断失败、分析瓶颈或替 Agent 选择优化方向；这些留给实验后的离线分析。
