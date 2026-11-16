# PTOAnalyzer Stage 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use the repository's `executing-plans` workflow to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 按本计划的最小验收集合执行；已通过且代码未变化的检查不重复运行。

**Goal:** 实现独立的 `basic_wiki-tool` Mode，在 Basic Pi 上接入 `pto_wiki/pypto_wiki` 和 PTOAnalyzer 九个工具，完成三组真实配对试验并留下可复核记录。

**Architecture:** 保留 Basic 的单 Agent 和 `preflight → prepare → run → collect` 流程。Mode 持有 PTOAnalyzer 的固定复制体，通过 Pi 原生 extension、本地 Unix socket 和 Python Broker 暴露九个工具；Wiki 作为 Workspace 内独立只读资料提供。PTOBench 继续管理 Evaluate、Profile、预算和设备。

**Tech Stack:** Python 标准库、现有 Pytest、Pi TypeScript extension、Node 22+、PTOBench 通用 Pi Tool 接口；不新增服务框架或依赖。

**Spec:** [Stage 5 目标](../README.md)、[方法入口协议](../../../../../PROTOCOL.md)、[Analyzer 接口](../../../README.md)。旧 Stage 5 文档保留背景；实施步骤、接口状态、路径与验收范围以本文为准。

## Global Constraints

- Mode 目录、声明和实验配置统一使用 `basic_wiki-tool`；`schema_version: 2`，`profile_strategy: combined`，首版仅支持 Pi。
- 本体 `modes/ptomni/pto_analyzer/` 保留；只复制运行时代码，接入适配放在复制体之外。本阶段不改变九个工具的计算、输出、分页和证据语义。
- Basic 对照不改；处理组只增加 Wiki 和 Analyzer，不加入 PTOmni 角色、子 Agent、强制工具调用或自动诊断流程。
- 不修改 `evaluator/`、`launcher/`、`modes/basic/`、`modes/cannbot/`、`modes/_shared/`、Case、采集器或公共测试。公共缺口交给原负责人，Stage 5 可继续不依赖缺口的工作。
- 九工具全部对同一个 Pi 可见，每次实际调用均记录；不增加 diff、baseline 选择器、第十个 Agent 工具或第二套硬件账本。
- Analyzer 只读取已完成的 combined Profile。读 Wiki、调用工具均无 NPU 动作；采集与验证仍由 Agent 调用 PTOBench CLI。
- Formal 60 + Held-out 15 的采集按既定约定作为前置输入；不在此计划中重跑 75 Case。真实试验仅为下文三个 Formal Case、六个 Session。
- 不输出、归档或提交凭据；不覆盖现有无关改动。每个实现任务完成时只暂存自己的文件，按仓库提交约定保存，禁止 `git add .`。

## 当前代码基线与前置关系

以下为 **2026-11-15 静态检查**，不是本轮运行验收的声明：

| 项目 | 当前实际情况 | 本计划处理 |
|---|---|---|
| 通用 Pi Tool 接口 | `evaluator/isolation.py::agent_command` 已有成对的 `pi_tool_extension` / `pi_tool_socket`；已有两项对应隔离测试 | 直接消费，复用公共任务验收证据 |
| 新 Mode 源码保护 | `create_session/_fixed_roots` 仍使用固定 Mode 列表，Launcher 尚未传入当前 Mode 源目录 | [通用 Mode 接入计划](../../../../../后续Mode统一接入PTOBench计划.md) Task 1 是真实 Session 的外部前置；不追加静态清单 |
| 新 Mode | `modes/basic_wiki-tool/` 目前为空目录 | 创建 Mode 自有实现 |
| Analyzer | 九工具已存在；`open_toolset("Compute" / "Dataflow" / "Schedule", ProfileRef)`，每个 ToolSession 绑定一个 Profile | 原样复制、外部适配 |
| 离线证据 | Stage 4 报告记录 16 份 Profile 和九工具；本轮未定位到能独立核实全部 75 Case 的完成索引 | 执行时定位已有完整索引，不能把 16 份报告写成 75 份验收 |
| Wiki 可见性 | `pi-tools` 对 Pi 文件工具与 Shell 隐藏；`.benchmark_context` 在 Mode prepare 前已保护 | 使用初始 Workspace 的独立只读子目录，不改已有 Context |
| 当前预算 | 每个 accepted Evaluate/Profile 共用并消耗一次 `max_hardware_actions` | 报告合并统计两类硬件动作，不重新计账 |

Task 1–3 的 Mode 代码和 CPU 测试可独立推进。**Task 4 真实 Session** 必须等公共接口验收、新 Mode 源码保护和输入证据就绪；缺少外部条件时留下精确依赖与已完成结果，不宣称整个 Stage 5 完成。

## Review Focus

1. 首个 Profile 尚未产生时，九个工具仍能注册；调用返回明确缺少 Profile 的错误，不暗中采集。→ Task 1/2。
2. 新 Profile 到达或使用 fork 后，旧 Cursor 不得切换 Candidate；允许明确选择旧 Profile。→ Task 1。
3. 事件和保护 manifest 正在发布时，不能把短暂读取不一致当成损坏；非法 Profile 路径仍须拒绝。→ Task 1。
4. 工具取消、断连、Agent 启动失败或退出时，不能丢失调用记录，也不能留下可继续接收请求的 socket。→ Task 2/3。
5. Wiki 必须实际可读、不可改，且不会污染 Candidate、Context 和既有 Session 完整性。→ Task 3。

## 文件与接口分工

下表中 `M = modes/basic_wiki-tool/`，均为新建文件；不在公共代码增加 Mode 分支。

| 文件 | 职责 |
|---|---|
| `M/mode.yaml`、`runner.py`、`prompt.py`、`instructions.md` | 声明和四步生命周期；沿用 Basic，仅补充 Wiki/工具能力 |
| `M/pto_analyzer/` | 本体运行时代码的逐文件原样复制 |
| `M/analyzer_manifest.json` | 复制源/副本 SHA-256，以及从当前本体生成的九工具规格 |
| `M/analyzer_adapter.py` | 当前 Session 的 Profile 选择、工具路由；不做诊断计算 |
| `M/tools.py`、`bridge/pi_analyzer.ts` | socket、Pi 注册、取消、调用记录 |
| `M/resources.py` | Wiki 冻结、只读可见副本与哈希清单 |
| `M/.env.example`、`.gitignore`、`README.md` | 独立 Provider 使用说明与运行说明；`.env` 只在本地使用 |
| `M/tests/test_analyzer_adapter.py` | 复制与规格、Profile 生命周期和授权 |
| `M/tests/test_pi_bridge.py` | 注册→socket→九工具调用及错误/取消记录 |
| `M/tests/test_mode.py` | Mode 特有接线、Wiki 和进程收尾 |
| `M/experiments/stage5/*.yaml` | 六份正式执行配置，Task 4 前冻结 |

执行记录统一放在 `modes/basic_wiki-tool/reports/stage5/`：`inputs.json`、`progress.md`、`experiment_manifest.json`、`summary.json`、`README.md`。原始 Agent、Profile 和工具调用归档留在各自 Run 内；报告只索引它们，不复制大产物。

## Task 0：确认输入与外部依赖

**Files:** 创建 `M/reports/stage5/inputs.json`、`progress.md`；只读检查以上当前基线和已有验收记录。

- [ ] 记录工作树起点、Basic/Analyzer 源文件身份、当前 Provider 非敏感有效配置、Wiki 文件身份；不得用单一 Git HEAD 代替含未提交修改的文件哈希。
- [ ] 定位已有 75 Case 完成索引，记录 Case ID、Session/event、Candidate hash、完成状态和原始索引路径；用已有清单核对覆盖，不重新解析所有大 Profile。找不到时记 `not_located`，保持为真实试验前的待办，不重跑全量补证据。
- [ ] 核对通用 Pi Tool 接口及公共任务验收证据。已有对应版本的通过记录就复用；没有记录时仅运行 `python -m pytest evaluator/tests/test_isolation.py -k 'pi_tool_' -q` 一次，由公共负责人修复失败。
- [ ] 检查通用 Mode 计划 Task 1 的实现/验收是否完成，重点看 Launcher 是否传入选中 Mode、Session manifest 是否只绑定该 Mode。未完成时继续 Task 1–3，不启动 Task 4，不自行改公共代码。

**完成标准：** 输入明确分为已核实、依赖未就绪、未定位；`progress.md` 记录每项测试命令、结果、代码身份及下一步。此任务不启动模型或硬件。

## Task 1：固定复制体，完成 Profile 与工具适配

**Files:** `M/pto_analyzer/`、`analyzer_manifest.json`、`analyzer_adapter.py`、`tests/test_analyzer_adapter.py`。

**Interfaces:**

```python
class AnalyzerAdapter:
    def __init__(self, session_root: Path): ...
    def specifications(self) -> list[dict]: ...
    def call(self, name: str, arguments: dict) -> dict: ...
```

`specifications()` 返回 Pi 的 `name/label/description/parameters`，不打开 Profile、不执行分析。`call()` 返回本体 `ToolSession.call()` 的原始结果；选择/参数错误抛出异常，由 Task 2 统一记录并转换为 Pi 错误。

- [ ] 写 `test_copy_and_specs_match_source`：核对复制体逐文件一致，规格与本体三组 `tool_specs` 一致；九工具名如下，前三、居中、后三分别路由 Compute、Dataflow、Schedule。使用现有 `tests/test_toolset.py::fixture_session` 生成 CPU 临时档案，不复制本体测试套件。

  ```text
  compute_mapping_check       compute_unit_activity       compute_task_share
  tile_access_check           buffer_pressure_check       tile_memory_allocation_summary
  dependency_path_check       dispatch_gap_check           task_distribution_check
  ```

- [ ] 写 `test_profile_selection_lifecycle`：无 Profile 时规格可读且调用失败；发布首个 Profile 后可调用；发布第二个后默认选择新事件；指定旧 ID 仍返回旧 Candidate；分页显式绑定旧 ID；未知 ID、跨 Session/Case、路径输入、`cursor` 未指定 `profile` 均被拒绝。模拟 fork 的同 Case 事件仍可选，不把“最新”解释为当前源码或最好版本。
- [ ] 写 `test_profile_open_uses_consistent_session_snapshot`：用受控发布过程确认 adapter 通过现有 `session._session_lock(root)` 在同一读取窗口选择事件、建立 ProfileRef 和打开 ToolSession；分析阶段已释放 Session 锁。无需重测公共锁实现。
- [ ] 运行 `python -m pytest modes/basic_wiki-tool/tests/test_analyzer_adapter.py -q`，确认新增行为尚未实现而失败。
- [ ] 复制本体的 `__init__.py`、`profile.py`、`toolset.py`、`diagnostics/*.py`；不复制 planning、reports、fixtures、tests、缓存。记录相对路径和源/副本哈希。使用临时 CPU fixture 的三个 `tool_specs` 生成 manifest 中的静态规格，不调用 `initial_summary`，不制造新的 Profile 采集。
- [ ] 实现 adapter：只接受 Runner 固定的 `session_root`。从该 Session 的 `events.jsonl` 读取完整的 completed combined 事件，交给复制体 `ProfileRef.from_event` / `open_toolset` 验证。Agent 输入 `profile` 只能是该 Session 内的 `session_id:event_seq`，不得是路径；不访问其他 Run 或全量 Profile 仓库。
- [ ] 所有工具在原 schema 上添加可选 `profile: string`；其余参数原样传给本体。无 `profile` 且无 Cursor 时，选择调用开始时最新的合法已完成 Profile；有 Cursor 必须显式带 `profile`。结果中的 Profile identity 原样返回，Prompt 说明下一页沿用这个 ID。
- [ ] 三组 ToolSession 按需建立；仅缓存最近使用的一个 Profile，切换时释放旧组，避免长期 Session 累积大档案。Cursor 密钥仍由同一 Python 进程的复制体维护，不重载模块；明确 ID 的旧分页可重新打开。
- [ ] 再运行本任务测试，全部通过后记录结果。首次没有 Profile 的错误说明应指向正常 `ptobench profile .`，工具本身不得代跑。

**完成标准：** 九工具先于 Profile 可发现；能处理优化过程中后来发布的 Profile；复制体字节不变，适配完全在外部。由于目录名含 `-`，Runner 使用 Mode-local 导入路径加载 `pto_analyzer`，不写非法的 Python 点式 import，也不误导入本体。

## Task 2：Pi extension、socket 与逐次调用记录

**Files:** `M/tools.py`、`bridge/pi_analyzer.ts`、`tests/test_pi_bridge.py`。

**Interfaces:** `ToolServer(adapter: AnalyzerAdapter, socket_path: Path, log_root: Path)` 启动服务，`close() -> None` 停止并清理。复用 CANNBot `bridge/pi_task.ts` 和 `tools.py` 的传输形式，业务仅调用 Task 1 adapter。

```json
{"action":"call","tool_call_id":"pi-call-1","name":"tile_access_check","arguments":{"profile":"session-id:3","limit":20}}
{"action":"cancel","tool_call_id":"pi-call-1"}
```

- [ ] 写 `test_registered_tools_roundtrip_and_audit`：Node 加载真实 `pi_analyzer.ts`，用 `registerTool` 桩获得恰好九个工具，并触发下述 session_start 可见性记录，然后经过真实 Unix socket 各调用一次 CPU fixture。逐项核对工具名、Pi call ID、Profile identity、完整返回值及一份终态记录；再通过同一路径调用一次下一页和一次非法参数。九工具只在这里做一次完整调用验收。
- [ ] 写 `test_cancel_disconnect_and_close_are_recorded`：受控慢调用覆盖调用前取消、执行中取消和客户端断开；最后关闭服务，socket 消失且不再接收请求。相同 call ID 的重传不可覆盖旧记录或重复执行；异常信息保留错误类型，不伪装为空成功。
- [ ] 运行 `python -m pytest modes/basic_wiki-tool/tests/test_pi_bridge.py -q` 确认失败；实现下列接线后再次运行并记录通过结果。
- [ ] `prepare` 时由规格 manifest 生成 `agent/pi-tools/pto-analyzer-specs.json`，与扩展同目录。扩展启动直接读取规格并注册全部工具，不依赖首次 Profile 或启动时 socket 往返。只用 `PTOBENCH_PI_TOOL_SOCKET`；`execute` 转发 `toolCallId`、参数、AbortSignal，返回 Pi `content/isError`，不截断本体摘要、不添加瓶颈结论。进入 execute 时已取消的请求仍发送 `call`，附 `cancelled: true`，服务只记取消而不执行，避免在扩展提前返回造成漏记。
- [ ] 使用当前 Pi 已提供的 `session_start`、`getAllTools()` 和 `getActiveTools()`，记录九工具是否已注册并进入模型的 active tool set。经内部 `{"action":"registered","names":[...],"active_names":[...]}` 消息写入 `method/analyzer_registration.json`；它不是 Agent 工具，也不计工具调用。缺少任何一个工具明确报接入错误，不自动改变 Basic 其他工具的启用状态。Task 4 直接使用这份真实 Pi 记录证明可见性。
- [ ] 服务使用标准库 Unix socket 和有界请求读取，沿用 CANNBot 的 1 MiB 请求上限；分析调用串行处理，取消通知可以并行接收。不引入 MCP、HTTP、进程池或每调用一个 Python 子进程。
- [ ] 在 Run 的 `method/analyzer_calls/<broker-generated-id>/` 先保存 `request.json`，终结时保存 `result.json`；目录名不使用外部传入的 call ID。记录工具名、全部参数、选中 Profile、开始/结束时间、成功/错误/取消/中断、完整输出或异常。`tool_call_id` 保留原值，失败在 Profile 解析之前也有记录。
- [ ] 取消语义明确为：尚未开始的分析不执行；同步本体已在执行时记取消请求，结束后丢弃成功响应并记取消，不宣称能立即中断解析。不改本体加取消框架。连接关闭不丢弃归档；强制结束留下的未终结请求由 collect 标为 interrupted，不伪造结果。
- [ ] 在 roundtrip 测试中一并核对 fixture 原始文件、Session/Event 字节未变化；Analyzer 路由无 CLI/NPU 调用。复用这一检查，不再另跑大档案哈希回放。

**完成标准：** 真实扩展代码经 Pi API 桩可注册并调用全部工具；这证明接入协议，不声称桩等于真实 Pi/LLM。真实 Pi 加载和行为在 Task 4 同一批 Session 验收。

## Task 3：Basic Mode、Wiki 与生命周期接线

**Files:** `M/mode.yaml`、`runner.py`、`prompt.py`、`instructions.md`、`resources.py`、`.env.example`、`.gitignore`、`README.md`、`tests/test_mode.py`。

**Interfaces:** Runner 保持现有 `preflight(request) / prepare(request) / run(request) / collect(request)`；`resources.freeze_wiki(source: Path, session_root: Path, workspace: Path) -> dict` 返回源/快照哈希、可见路径、文件清单，供 Prompt 和归档使用。

- [ ] 写 `test_mode_wires_tools_and_preserves_basic_contract`：Fake Agent 执行四步流程，断言 Basic 指令/默认优化要求保留，Pi backend、combined、通用接口成对参数正确；工具在 Agent 启动前就绪，启动异常和正常退出均关闭服务。用 SIGTERM 子进程场景验证可捕获退出的清理；collect 归档 runtime 调用记录，prepare 不提前冻结日志目录。
- [ ] 写 `test_wiki_visible_readonly_without_context_changes`：实际 Session Workspace 下生成 Wiki；以 `ptobench-agent` 身份在现有 Agent 视图读取 `INDEX.md`，写入、删除、替换失败。保持 Candidate、Context 原内容/哈希及既有保护 manifest 有效；从 fork workspace 通过 Prompt 中同一绝对路径可读。该定向 CPU 测试复用已有隔离测试设施，不新增挂载，不启动 LLM/NPU。
- [ ] 运行 `python -m pytest modes/basic_wiki-tool/tests/test_mode.py -q` 确认失败，再实施以下步骤。
- [ ] 从 Basic 复制四步入口、基础 Prompt 和 `instructions.md`，沿用 `_shared` Provider/recording/provenance 和 `agent_launch.run_process`。声明 Mode v2；preflight 拒绝非 Pi，不读取或执行其他 Mode 业务工具。独立 `.env` 沿用当前 Basic 有效配置，密钥只从既有本地配置/环境取，`.gitignore` 排除 `.env` 和缓存。
- [ ] Wiki 的归档快照放 `<session>/method/resources/pypto_wiki/`，Agent 可见副本放 `<initial_workspace>/.mode_resources/pypto_wiki/`，两者是独立普通文件。可见目录为 Broker-owned、`ptobench-share` 组可读，目录 `0750`、文件 `0640`；不使用链接、不改 Workspace 根权限。初始 Workspace 的绝对可见路径放进 Prompt，供所有 fork 使用，不为每个 fork 再复制。
- [ ] `freeze_wiki` 拒绝链接/特殊文件，保存 source/snapshot/visible-copy 的相对路径与哈希清单。prepare 只通过既有 `archives` 声明冻结 `method/resources`；可见副本由 DAC 只读，并在 collect 对照清单确认未变。不要将 `workspaces/...` 当作 `archives` 路径——现协议只允许 `method/`、`agent/` 附件。
- [ ] prepare 放好扩展与规格；run 先建立 `AnalyzerAdapter` 和 `ToolServer`，再调用以下现有接口；正常/异常/可捕获中断通过同一清理路径关闭服务。socket 路径使用当前 Session 内短路径，不能移到 Session 外规避 Unix 路径长度约束。

  ```python
  isolation.agent_command(
      argv, cwd=workspace, environment=environment,
      workspace=workspace, session_root=run_root, backend="pi",
      pi_tool_extension=run_root / "agent/pi-tools/pi_analyzer.ts",
      pi_tool_socket=run_root / "agent/analyzer.sock",
  )
  ```

- [ ] Prompt 仅新增：Wiki 的 `INDEX.md` 路径；九工具可按需调用；先由正常 CLI 产生 Profile；省略 `profile` 读最近已完成事件，比较旧版本及翻页需明确 ID。Agent 可自主分别读取两份结果比较；不自动 profile、不自动 initial_summary、不强制九工具调用。
- [ ] collect 保留 Basic recording，补充 Wiki/Analyzer 身份与调用归档；将未终结调用记为 interrupted，静态附件不重复注册。再次运行本任务测试，记录通过结果。

**完成标准：** Mode 在真实 Workspace 权限下完成 CPU 接入验收；Basic 与 Analyzer 本体哈希未被本任务改变；新 Mode 运行时不依赖原本体路径。已有公共隔离、Basic、CANNBot 和 Analyzer 本体测试不在此重复执行。

## Task 4：三组真实配对试验，同时验收真实 Pi 接入

**Files:** `M/experiments/stage5/` 六份 YAML；`M/reports/stage5/experiment_manifest.json`。

**依赖：** Task 0 的外部条件就绪，Task 1–3 测试通过。只有相关实现后来改变，才重跑受影响测试。不要为了进入实验再完整重跑三套测试。

| 顺序 | Case | 实际设备要求按 Case/env 核对 | 组内顺序 |
|---|---|---|---|
| 1 | `deepseek_v4.lm_head_tp2_flash` | 2 卡 | Basic → basic_wiki-tool |
| 2 | `deepseek_v4.prefill_layer_ep2_csa_demo` | 2 卡 | basic_wiki-tool → Basic |
| 3 | `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | 8 卡 | Basic → basic_wiki-tool |

- [ ] 冻结三对配置：各对相同 Case seed/Candidate、模型、reasoning、Provider 配置、Pi 版本、Profile 权限与设备。使用现有 Pi `default` profile，运行前解析并记录真正生效的模型；处理组和 Basic 相同，不能只比较 profile 名称。
- [ ] 本次是接入/行为试点，统一使用现有默认 `budget: {max_hardware_actions: 20, finish_actions: 1}`，不将它当作论文最终 B。Evaluate/Profile 共用硬件动作预算，Profile 按需；不引入新的 Profile quota、wall-time 参数或预算实现。若已有同条件正式 Basic Session 可复用，先核实完整配置/初始 hash/账本，复用并记录，而非无条件重跑。
- [ ] 六份 YAML 按 `01_lm_head_basic.yaml`、`02_lm_head_wiki_tool.yaml`、`03_prefill_layer_wiki_tool.yaml`、`04_prefill_layer_basic.yaml`、`05_cp8_basic.yaml`、`06_cp8_wiki_tool.yaml` 命名。共同字段为 `optimizer.backend: pi`、`optimizer.profile: default`；`mode`、`optimizer.id`、输出目录按组独立。保存相对项目根可解析的配置，凭据不写 YAML。
- [ ] 按上述顺序串行运行 `python pto-launcher run modes/basic_wiki-tool/experiments/stage5/<对应文件>.yaml`。使用当前 PTOBench 设备锁，CP8 按完整设备组等待；不实现额外调度器，不同时启动冲突设备的 Session。
- [ ] 每个真实 Session 的 Profile 从自己的 CLI/预算流程产生；75 Case 历史归档用于前置覆盖和开发，不只给处理组额外注入历史优化答案或免费 Profile。历史 Profile 缺少元数据不应触发全量重采。
- [ ] 第一个处理组同时检查真实 Pi 是否加载并向模型启用九工具、Wiki 路径是否正确、自然调用能否返回并归档。保留 `analyzer_registration.json` 和 Pi 原生调用记录。其余处理组继续按自主优化流程执行；不要求九工具逐一调用。
- [ ] 若全部处理组零调用，保留这一合法行为结果，不追加强制调用会话。报告只声称“真实 Pi 已注册且模型可见、CPU 往返通过”，明确“未观察到真实模型调用”；不能把注册记录当成实际使用或收益。这不阻塞本次组合试点的交付。
- [ ] 保留正常、失败、中断、零修改、零工具调用的所有记录；单个算法/正确性失败是试验结果。基础设施错误先修 Mode 自有问题，只重跑受影响 Session 并保留原记录；公共缺口交给负责人，不扩展 Case 队列。

**完成标准：** 三个匹配对均有原始记录或符合条件的复用记录；真实 Pi 可见性已验证，实际调用按观察结果报告。六 Session 不用于宣称普遍性能优势；不另安排 smoke run、重复种子或九工具的真实 NPU 强制调用。

## Task 5：归档报告与收尾

**Files:** `M/reports/stage5/summary.json`、`README.md`、`progress.md`，更新 `M/README.md`。

- [ ] 汇总每个 Session 的初始/最终 Candidate hash 与 Diff、正确性、Clean Latency、共享 Evaluate/Profile 预算使用、Token、wall-clock、结束原因。数字来自原账本与 recording；不可用项写明原因，不填零。
- [ ] 汇总九工具调用次数、成功/错误/取消/中断、Profile identity、原始记录路径；将“扩展已注册”“工具实际被调用”“结果被后续决策引用”“最终收益”分别报告。Wiki 读取证据只来自实际可观察的 Agent 记录。
- [ ] 对比三组 Basic 和处理组；仅评价 Wiki + Analyzer 组合及接入可行性，不分别归因、不把 Profile 时间当 Clean Latency，不自动推导性能瓶颈。
- [ ] 复核本体、复制体和 Wiki 清单；检查工作差异没有触及禁止路径、凭据或大运行产物。运行 `git diff --check -- modes/basic_wiki-tool modes/ptomni/pto_analyzer/planning/stage5/plans`，核对新文件被正确跟踪且 `.env` 未纳入。
- [ ] 使用 Task 1–4 已记录的验收证据，不重复全套测试、75 Case 回放、全部归档哈希检查或额外 Agent 试验。仅新增修复时运行对应回归；将完成项、外部依赖和未验收项写进最终报告。

**完成标准：** 新 Mode 可用，九工具接入可审计，本体保留，三组试验和必要验收有证据。未解决公共前置或缺失真实 Pi 注册/可见性证据时不能把整个目标记为完成。

## 最小验证集合

| 验证 | 唯一负责的位置 | 不重复的内容 |
|---|---|---|
| 复制/规格/动态 Profile/分页归属 | Task 1 adapter 测试 | 不重新测九工具内部统计与大文件算法 |
| 九工具注册、往返、错误、取消、只读 | Task 2 bridge 测试，CPU fixtures | 不再建第二套 Python-only 九工具调用测试 |
| Wiki 实际权限、Mode 参数与生命周期 | Task 3 mode 测试 | 不重测公共 mount、Provider、CLI、设备锁实现 |
| 真实 Pi、Wiki/工具行为和优化结果 | Task 4 同一批配对 Session | 不固定追加 smoke、全 Case 或重复种子 |
| 完整性与文档交付 | Task 5 清单/差异/报告 | 不以“收尾”为由再执行已通过且代码未变化的测试 |

执行中将命令、退出码、证据路径和适用代码身份写入 `progress.md`；goal 继续运行或上下文恢复时先读该文件，避免重做已经完成的任务。
