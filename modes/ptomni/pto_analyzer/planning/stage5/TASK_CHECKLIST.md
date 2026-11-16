# PTOAnalyzer Stage 5 任务清单

## 范围与固定约束

- [ ] 只实现 `modes/basic_wiki-tool` 和必要的 Mode 适配；不修改 `modes/basic/` 的 Basic Pi 对照实现。
- [ ] `modes/ptomni/pto_analyzer/` 是唯一维护的 PTOAnalyzer 本体。接入 Basic 时复制当前本体到 `modes/basic_wiki-tool/pto_analyzer/`；适配层可以变化，但不得反向修改本体来迎合 Basic。
- [ ] 处理组只增加 `pto_wiki/opt_wiki` 和 PTOAnalyzer；不加入 PTOmni 的 Lead、Tuner、Assembler、Challenger、Curator 或其它工作流。
- [ ] Basic Pi 不按三个 Tuner 分配工具；九个工具全部注册给 Basic Pi。每次工具调用都记录，包含成功、失败和取消。
- [ ] PTOAnalyzer、Wiki 和工具调用不启动硬件动作。`evaluate`、`profile`、设备锁和预算继续由 PTOBench Runner/Event Ledger 负责。
- [ ] 不重跑已经完成的 75 个 Case Profile 覆盖；只在配对试验发现输入身份确实缺失时，按 Runner/PTOBench 正常流程补采一个明确目标。
- [ ] 不新增自动 diff、baseline 选择器或第二套硬件账本；Agent 可分别读取它有权访问的 Profile，自行比较。
- [ ] PTOBench 公共代码需先提供已验收的通用 Pi extension/socket 接口。该公共改动由另一任务负责人实施；Stage 5 不修改 `evaluator/isolation.py`、公共 Runner 或其测试。

公共接口的设计参考为[独立接入计划](../../../../../升级/1110/Plan%2005后轻量Pi工具接口与遗留代码清理.md)；是否完成以公共任务的实现和验收证据为准。

## Task 0：冻结输入与工作树

- [ ] 核对 `modes/ptomni/pto_analyzer/` 的当前版本、九个工具名、CPU 回归结果和 Stage 4/75 Case 归档索引；记录本次接入使用的版本身份。
- [ ] 核对 `pto_wiki/opt_wiki/INDEX.md`、`README.md` 和文件清单，生成只读知识库快照身份（例如提交/目录摘要或完整 SHA-256 清单）。
- [ ] 确认 `modes/basic/` 当前协议和 Basic Pi 的 `preflight/prepare/run/collect` 行为；不把任何处理组文件写回 Basic。
- [ ] 检查工作树并保留无关改动，尤其不覆盖 `evaluator/` 和已有 PTOAnalyzer 本体改动。
- [ ] 检查 PTOBench 通用接口已落地并通过其自身验收：成对可选 `pi_tool_extension`/`pi_tool_socket`、Session-local Broker-owned extension 与 Unix socket、固定沙箱映射和 `PTOBENCH_PI_TOOL_SOCKET`，且现有 Pi extensions 可并存、CANNBot 兼容。若未完成或无验收证据，停止所有依赖 Pi 接线的 Stage 5 任务并报告；不在本 Stage 实现该接口。

**验收：** 形成输入版本记录；明确复制源、快照身份、通用接口验收依据和不应改动的路径。外部接口未就绪时，Stage 5 只可继续不依赖它的文档/输入准备工作。

## Task 1：创建 `basic_wiki-tool` Mode 骨架

**Files:**

- Create: `modes/basic_wiki-tool/__init__.py`
- Create: `modes/basic_wiki-tool/mode.yaml`
- Create: `modes/basic_wiki-tool/runner.py`
- Create/modify: `modes/basic_wiki-tool/prompt.py`
- Create/modify: `modes/basic_wiki-tool/instructions.md`
- Create: `modes/basic_wiki-tool/README.md`

- [ ] 从 Basic Mode 复制生命周期和配置结构，再用独立 `id`（例如 `basic_wiki-tool`）登记；保持同一 Pi backend、`combined` profile strategy、Case workspace 和 PTOBench CLI 入口。
- [ ] Prompt 只增加两项公开能力说明：可按需查阅 `pto_wiki/opt_wiki`，可按需调用九个 PTOAnalyzer 工具；保留 Basic 的编辑范围、正确性要求、预算和停止条件。
- [ ] 不把 Wiki 全量文本预注入 Prompt；不自动调用所有工具；Agent 自行决定何时查询 Wiki 或工具。
- [ ] 记录 Mode 配置、Prompt 和源文件哈希，保证 Basic 对照组与处理组可审计。

**验收：** Mode 能完成 CPU-only `preflight` 和 `prepare`，生成与 Basic 等价的基础 Agent 配置；Mode 文件不改变 `modes/basic/`。

## Task 2：复制并接入 PTOAnalyzer 本体

**Files:**

- Create: `modes/basic_wiki-tool/pto_analyzer/`（复制 `modes/ptomni/pto_analyzer/` 的运行时代码与所需静态资源）
- Create: `modes/basic_wiki-tool/analyzer_adapter.py`（或同职责的 Mode-local 适配模块）
- Create: `modes/basic_wiki-tool/tools.py` and `modes/basic_wiki-tool/bridge/pi_analyzer.ts`（复用 Pi 原生 extension + 本地 Unix socket 的传输模式；只借用传输机制，不复制 CANNBot 的 Task/Skill 工作流）
- Modify: `modes/basic_wiki-tool/runner.py`

**依赖：** 使用已实现并验收的 `evaluator/isolation.py::agent_command(...)` 通用 Pi extension/socket 接口。Stage 5 不实现或修改该公共接口。

- [ ] 在实现接入前复制最新 PTOAnalyzer；记录源目录版本/文件清单哈希和复制体哈希。
- [ ] 适配层为每个当前 Case 的授权 `ProfileRef` 建立/复用一组复制体 `ToolSession`（Compute、Dataflow、Schedule 各一个）；每个 `ToolSession` 仍绑定单一 Profile，Broker 可按已授权 Profile ID 选择对应的一组会话，并向 Agent 暴露九个工具。
- [ ] 从三组 `tool_specs` 合并出统一注册表，使用工具名路由到对应 `ToolSession`；Pi 注册时把现有 `input_schema` 映射为 Pi `parameters`，并在扩展层为全部工具增加可选 `profile` ID 字段，分页工具同时保留 `cursor`/`limit`。
- [ ] 由 Runner 提供当前 Case 的授权 Profile 索引（Profile ID 到已验证 `ProfileRef`/身份的映射）；PTOBench 接受新的 Profile Event 后，Runner 更新该索引，Broker 只从索引解析 Profile，不自行扫描任意目录或接受路径。工具参数可带可选 `profile` ID，省略时使用当前默认 Profile。拒绝未知 ID、跨 Case Profile、未授权 Event 和 Candidate 身份不符的 Profile；没有可用 Profile 时返回可记录的明确错误。
- [ ] 每个复制体 `ToolSession` 仍只读且绑定单一 Profile，保留摘要和 Cursor 分页并拒绝未授权路径/事件；适配不改变九个工具的名称、结果字段或解释边界。
- [ ] 在 `prepare` 阶段把 Pi extension 作为 Broker-owned 只读文件放入本次 Session 的 `agent/pi-tools/`。Extension 使用 Pi 原生 `registerTool` 注册九个具名工具；将各 Tool Spec 的 `input_schema` 映射为 Pi `parameters`，并为 Profile 选择增加可选 `profile` ID 参数，分页工具仍保留 `cursor`/`limit`。
- [ ] 在 `run` 阶段由 Python Mode 启动本次 Agent Session 独有的 Unix-socket Tool Broker，再启动 Pi。使用单行 JSON socket 消息：`call` 携带 `tool_call_id`、`name`、`arguments`，`cancel` 携带 `tool_call_id`。Broker 校验 Profile ID，按 Profile 选择/缓存一组 Compute、Dataflow、Schedule `ToolSession`，从参数中移除 `profile` 后调用对应工具。Broker 返回完整工具结果或结构化错误给 Extension，Extension 原样返回 Pi。Pi 已发起但在分发前被取消的调用仍需提交一条可归档的取消请求，不允许在 extension 内静默丢弃。
- [ ] 通过 PTOBench 通用接口挂载 Broker-owned extension 和本次 Agent Session 的 socket（接口固定映射到 `/ptobench/pi-tool.sock`，并提供 `PTOBENCH_PI_TOOL_SOCKET`）。Agent 正常结束、启动失败或取消时都关闭 Broker、清理 socket；未完成的调用写入取消/终止状态。
- [ ] 每次调用在本次 PTOBench Session 的 Mode-owned 归档中保存完整 request/result（例如 `method/tool_calls/<tool_call_id>/request.json` 和 `result.json`），包含 `tool_call_id`、时间、工具名、原始参数、选用的 Profile identity、最终状态和完整结果/错误。若为避免重复存大结果而采用不可变结果文件引用，必须保存完整结果文件及哈希，不得截断或只留摘要。
- [ ] Pi extension 只负责注册工具、参数转发、取消和结果返回；Profile 授权、工具执行和调用归档由 Python Broker 负责。该接线不增加 Agent 子角色、额外 PTOBench CLI 权限、硬件动作或写 Candidate 的能力。
- [ ] 调用 `isolation.agent_command(...)` 时只使用通用 `pi_tool_extension`、`pi_tool_socket` 参数，不依赖 CANNBot 专用参数；将 PTOAnalyzer extension 与 File Guard、Provider extension 及既有 Pi extensions 一并加载，不覆盖它们；不在 Basic Mode 中伪装成 CANNBot。
- [ ] 复用 PTOBench 原有 Profile/Event 身份和硬件账本；不在复制体或适配层执行 CLI，不把工具调用计为硬件动作。

**验收：** 复制体与本体在同一 fixture/profile 上输出一致；本体文件哈希不变；九个工具都由 Pi extension 注册并可经 Broker 调用；工具调用记录可与 Pi 原生 tool-call ID 关联；Broker 在 Agent 正常退出、启动失败和取消时均关闭。通用隔离接口及 CANNBot 兼容由公共接口任务验收，Stage 5 不承担其实现或回归责任。

## Task 3：接入 `pto_wiki/opt_wiki`

**Files:**

- Modify: `modes/basic_wiki-tool/runner.py`
- Modify: `modes/basic_wiki-tool/instructions.md` or `prompt.py`
- Create: `modes/basic_wiki-tool/knowledge/` manifest/snapshot metadata as needed

- [ ] 在 Prepare 阶段把 `pto_wiki/opt_wiki` 冻结复制到本次 Run 的 Broker-owned `agent/pi-tools/opt_wiki`，利用现有 Pi 隔离只读挂载暴露为 `/ptobench/pi-tools/opt_wiki`；不把源 Wiki 目录直接交给 Agent 写入。
- [ ] 向 Agent 提供 `/ptobench/pi-tools/opt_wiki/INDEX.md` 和 `README.md` 的入口及按主题查阅方式；允许使用 `rg`/普通文件读取按需检索，不强制预读全部文档。
- [ ] 冻结并记录 Wiki 快照身份、可见路径和文件数量/哈希清单；不允许 Agent 写回知识库。
- [ ] 明确 Wiki 内容是参考知识，不能替代 Profile 原始证据、PTOBench `evaluate` 结果或工具解释边界。

**验收：** Agent workspace 中可读到固定 Wiki 快照且不可写；快照前后哈希一致；Basic 对照组看不到新增 Wiki 文件。

## Task 4：九个工具的发现、调用与记录测试

**Files:**

- Create/modify: `modes/basic_wiki-tool/tests/test_mode.py`
- Create/modify: `modes/basic_wiki-tool/tests/test_analyzer_adapter.py`

- [ ] 测试工具列表恰好包含九个工具：`compute_mapping_check`、`compute_unit_activity`、`compute_task_share`、`tile_access_check`、`buffer_pressure_check`、`tile_memory_allocation_summary`、`dependency_path_check`、`dispatch_gap_check`、`task_distribution_check`。
- [ ] 用 Pi `registerTool` 测试桩加载原生 extension，确认 Pi 收到的九个工具名与上列完全一致；再调用已注册工具，经 Session socket 到达 Broker，并确认完整结果/错误返回对应 tool call。使用 CPU Profile fixtures；不启动真实 LLM、PTOBench Case 或 NPU。该测试证明 PTOAnalyzer extension 按 Pi API 注册且可调用，不替代公共隔离接口任务的验收。
- [ ] 测试 Basic 处理组不按 Tuner 隐藏工具；三组工具都可由同一 Agent 会话调用。
- [ ] 对每个工具各执行至少一次成功调用并核对记录；另覆盖一个失败或非法参数调用，确认失败也有记录且不会伪装成空成功。
- [ ] 验证省略 `profile` 使用默认 Profile，显式 Profile ID 能选到同 Case 的另一份授权 Profile；未知 ID、跨 Case ID、任意路径及不匹配 Cursor 均明确失败并记录。
- [ ] 对至少一个分页工具验证 Cursor 页调用、Profile identity 和调用记录的连续性；高基数结果保存完整内容或可验证的不可变文件引用。
- [ ] 验证 Pi tool-call ID 在 Pi 注册工具 → Extension → Broker → 归档记录 → Pi 返回值链路中保持一致；测试取消信号确实到达 Broker，并且 Agent 退出后 socket 已关闭。
- [ ] 测试工具调用不修改 Profile 原始归档，不增加 `evaluate/profile` Event 或 PTOBench 硬件动作。
- [ ] 测试复制体与 `modes/ptomni/pto_analyzer` 在现有 fixtures 上的结果一致；测试本体源文件哈希在 Mode 接入前后不变。

**验收：** CPU 测试证明 Pi 收到并能调用全部九个工具，且覆盖成功/失败/分页/取消记录和只读边界；每次调用都有唯一可追溯记录。Stage 5 验收 PTOAnalyzer 的 Pi 注册、传输、Broker 路由和返回；通用隔离 seam 本身由公共接口任务负责验收。

## Task 5：Basic 对照与真实 Agent 配对试验

**Case：**

| Case | 重点 |
|---|---|
| `deepseek_v4.lm_head_tp2_flash` | 双卡 MatMul/通信与 Compute 证据 |
| `deepseek_v4.prefill_layer_ep2_csa_demo` | 多阶段 Dataflow、Tile/Buffer/静态分配 |
| `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | CP8 多 rank、依赖和高基数 Schedule |

- [ ] 从已完成的 Formal/Held-out Profile 归档中核对这三个 Case 的 ProfileRef、Candidate、设备和配置身份；缺失时只补采本试验需要的明确 Candidate，不重跑 75 Case。
- [ ] 每个 Case 运行一组 Basic Pi 和一组 `basic_wiki-tool`，共六个独立 Session；固定模型/backend、初始 Candidate、PTOBench 配置、evaluate/profile 权限、预算和软件环境。
- [ ] 使用隔离 workspace；配对的运行顺序交错或随机，避免固定顺序造成系统性偏差。
- [ ] 处理组由 Agent 自主决定是否查 Wiki、调用哪些工具和调用次数；不强制每个工具恰好调用一次，但记录要求覆盖每一个实际调用。九工具可发现/可调用性由 Task 4 CPU 集成测试证明，不要求为此额外启动 Agent smoke run，也不把 Task 4 测试计入性能结果。
- [ ] 每个 Session 保存 Case、模型、Mode/版本、初始与最终 Candidate hash、代码 Diff、工具调用记录、Wiki 快照身份、Profile/evaluate Event、正确性、Clean Latency、硬件动作、Token、wall-clock 和结束原因。
- [ ] 统计工具调用次数、工具覆盖、调用后是否出现相关代码/配置修改、无效尝试、evaluate 结果和性能变化；不把“调用工具”直接解释成“使用证据”或“获得收益”。

**验收：** 六个 Session 均可从 Session → Agent call → Profile/evaluate → Candidate Diff 追溯；失败、中断、无 Candidate 变化和零工具调用均保留。

## Task 6：离线分析与报告

**Files:**

- Create: `modes/ptomni/pto_analyzer/reports/stage5_basic_wiki_tool_*.json`（或项目约定的报告路径）
- Create: `modes/basic_wiki-tool/reports/README.md`（如需要）

- [ ] 分开报告 Basic 与 `basic_wiki-tool` 的工具使用、Wiki 使用、Candidate 变化、硬件动作、正确性、Clean Latency、Token 和 wall-clock。
- [ ] 将工具调用和工具输出按 Profile identity 与 Agent call ID 关联；区分“工具被调用”“结果被读取/引用”“之后发生了相关修改”三个事实。
- [ ] 对同一 Case 的结果只报告可比指标；Profile 观测由 Agent 自行比较或离线做简单汇总时，注明两份 Profile 身份、单位和采集口径，不创建自动 baseline 结论。
- [ ] 报告工具或 Wiki 接入错误、Profile 缺失、权限拒绝、调用失败和硬件异常；不把缺失值当作零。
- [ ] 明确这是 Wiki + PTOAnalyzer 的组合消融；不得从六个 Session 宣称单独的 Wiki 或 PTOAnalyzer 因果效果。

**验收：** 生成可审计的 Stage 5 报告和原始记录索引；所有硬件数字来自 PTOBench/Event Ledger，所有工具调用来自 Mode 记录。

## Task 7：收尾检查

- [ ] 运行 Basic 现有 CPU 测试，确认对照实现未被改变。
- [ ] 运行 `basic_wiki-tool` 的 Mode/Analyzer CPU 测试和 PTOAnalyzer 原有测试。
- [ ] 复核 `modes/ptomni/pto_analyzer/` 与 `modes/basic_wiki-tool/pto_analyzer/` 的版本关系；若修复复制体发现本体缺陷，先在本体中单独修复、验收并重新复制，不只改复制体。
- [ ] 检查 Git 跟踪清单：Stage 5 文档、Mode、复制体、Wiki 快照元数据、测试和报告索引均被跟踪；不提交运行时凭证、NPU 临时文件或可写归档。

**验收：** Basic 对照可复现，处理组可运行，九个工具调用可审计，原 PTOAnalyzer 本体保留且未被接入适配污染。
