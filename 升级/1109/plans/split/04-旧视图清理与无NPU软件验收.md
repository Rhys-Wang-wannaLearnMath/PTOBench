已被 1110 Dedicated Agent User 计划取代，仅供历史参考、不得执行。

# 旧 Workspace View 清理与无 NPU 软件验收 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task.

**Goal:** 在真实 Workspace 路径验证通过后删除失去调用者的导出/同步实现，并完成 75 个 Workspace 的无 NPU 软件验收。

**Architecture:** 沿用 Plan 03 已验证的真实目录、挂载和反馈实现；按实际调用关系清理旧代码，保留非 Workspace 历史归档及已有 CLI/评测合同。本计划只执行原计划 Task 6–7。

**Tech Stack:** Python 3、Bubblewrap、Linux user namespace、POSIX UID/GID 与目录权限、现有 unittest 测试。

**Spec:** [`升级/1109/PTOBench真实目录Workspace与沙箱简化修缮计划.md`](../../PTOBench真实目录Workspace与沙箱简化修缮计划.md)

**Source plan:** [`升级/1109/plans/PTOBench真实目录Workspace与沙箱简化Implementation Plan.md`](../PTOBench真实目录Workspace与沙箱简化Implementation Plan.md)

**Execution order:** This is Plan 04 of 05. Plan 01–03（原计划 Task 0–5）全部完成并通过；Task 6 清理前须确认新路径已工作，Task 7 仅在清理回归通过后运行。

**Plan boundary:** Execute only original Task 6–7; do not implement tasks assigned to another split plan. Task 6–7 的旧视图搜索、隔离/CLI/Combined-split/Finalize 回归、75 个临时 Workspace 扫描及 lightweight_e2e 独立性检查通过后停止；任何失败或输入资产 hash 改变时不得进入真实 Pi 或硬件验收。

## Global Constraints

- 只保留真实目录 Workspace：workspaces/worker_NNN/{candidate,.benchmark_context,.benchmark_outputs}，不以链接、副本或同步目录呈现给 Agent。
- Agent 和 Broker 必须是不同的宿主 UID；继续使用现有 `evaluator/isolation.py::agent_command` / `View.argv` Bubblewrap `--unshare-user` 路径，并依据当前宿主配置验证 subordinate UID/GID 映射和 Broker 主 GID（`os.getgid()`）。只为 Pi Agent 子进程清除未授权附加组；Broker 和受信任 NPU Worker 保持原有 UID/GID/groups 与硬件权限。优先使用通过验证的无特权启动方式；必要时才使用 Plan 01 中固定、无参数、一次性配置的最小权限启动器，调用者 UID/GID 动态检测。不得让 Agent 获得 host sudo/root，也不得以 root 执行用户可修改代码。sudoers 不使用通配参数或通用 root Shell。
- 每个 Agent Sandbox 只对当前 Session 的 workspaces/ 做一个可写 bind mount；Candidate、Context、Outputs 的权限由宿主文件系统 DAC 强制执行。
- 保留 ptobench evaluate/profile/fork 的语法、Combined 默认、显式 split、预算计数、事件记录、正确性检查、正式延迟、Snapshot、Finalize 和 Candidate Worker 执行路径。
- Basic 与 CANNBot 继续使用各自的 `.env`，配置为 DeepSeek Flash 和 High thinking；PTOmni 尚未作为本轮 Agent 后端验收对象。
- 保留 Pi 私有文件隔离、Gateway、Broker Unix Socket、模型 API 通道、Agent 网络隔离及 NPU/HCCL 受信任执行路径。
- 不修改 Case、Fixture、Seed、Reference、Baseline、CV 或 case_selection 数据；验收结果另存到新的 reports 目录。
- lightweight_e2e 不得成为正式 PTOBench 运行、导入、测试或部署依赖。
- Candidate Worker 内部评测文件的可见范围仍是独立问题，本轮不扩展处理范围，也不声称已解决。

## Review Focus

1. 普通登录带附加组时，Plan 01 选择的启动方式是否只清理 Agent 子进程；Broker/NPU Worker 凭据是否保持，Agent user-namespace 映射是否匹配当前宿主配置，真实 DAC canary 是否拒绝未授权访问。由 Task 1 在生产 Bubblewrap 启动路径验证。
2. Agent 新建 0600 文件或不可遍历目录后，Broker 是否明确拒绝 Candidate 快照而不是生成残缺 Snapshot；由 Task 2 和 Task 7 验证。
3. Fork 在 staging、pending、发布和恢复之间中断时，是否出现可被 CLI 使用的半成品或重复 Worker ID；由 Task 3 的故障注入测试验证。
4. Pi Shell 子 Sandbox 是否仍能写 Candidate、不能改 Context/Outputs；必要 Unix Socket 是否允许映射 UID 连接而不开放目录枚举；由 Task 4 的生产 Bubblewrap 集成测试验证。
5. 诊断文件是否仍可读、只读，且路径指向 Agent 可见的 Workspace；由 Task 5 的发布测试和 Combined Profile 抽查验证。

## 原计划任务（正文原样迁移）

### Task 6: 删除已失去调用者的 Workspace View 代码

**目标及实现方式：** 只在 Task 4/5 的真实目录路径通过后清理旧导出 Workspace 机制。依据真实调用关系删除，不按名字批量删除；`run_legacy` 必须先改到相同真实目录路径，不能因它仍调用共享 Broker 而留下第二套 view。

**可安全删除的旧代码（确认全仓无 Workspace 调用者后）：**

- sandbox_supervisor.py::Broker.export、Broker.sync_source、Broker.feedback，以及 Broker.exports、Broker.candidates、Broker.public 这些 Workspace 视图字段。`Broker.rewrite` 只有在 Task 5 证明即时 CLI 响应无需路径转换，或已迁为窄用途 formatter 后才删除。
- isolation.py 中从 PTOBENCH_AGENT_WORKSPACE_ROOT 推导 Candidate 池、分别挂载 public/Candidate 的代码，以及对应旧环境变量。
- pi_shell_wrapper.py 中 /ptobench/candidates 的单独挂载和相关配置。
- test_isolation.py 中要求 Workspace 入口为符号链接、要求导出视图/同步回写的测试；用真实目录、宿主 UID/GID 与直接反馈测试替代。

**明确保留：**

- sandbox_supervisor.py 顶层 copy_tree：run_legacy 仍用它归档 method/agent 历史；删除前先确认并替代这些非 Workspace 调用。
- workspace.copy_candidate_tree、copy_tree、source_manifest：仍用于 Case Seed、Fork、Action freeze、Snapshot 和 Candidate Worker 输入。
- file_tree 的 no-follow、稳定扫描、原子写入；Session/Context 完整性和 Candidate 特殊文件拒绝；Pi File Guard、Shell 隔离、Gateway/Socket、Profile Combined/split 和预算逻辑。
- 不删正式 PTOBench 代码或测试以 lightweight_e2e 的行为替代；不处理 Candidate Worker 评测源码边界。

**文件：** evaluator/sandbox_supervisor.py、evaluator/isolation.py、evaluator/pi_shell_wrapper.py、evaluator/tests/test_isolation.py、必要的 Agent-facing README/帮助文档。

- [ ] 在删除前用全仓 rg 确认旧 Workspace 环境变量、方法和链接路径的调用者；逐项清理只与旧 Workspace View 有关的用法。
- [ ] 删除后再 rg 导出目录、旧环境变量和 `/ptobench/candidates`；运行时不得残留旧 Workspace View。另查所有 `rewrite` 调用，确保只保留经 Task 5 验证仍需要的用户可见 CLI 路径格式化。文档不再要求 Agent 检查链接、导出副本或 Broker 同步。
- [ ] 保留 non-Pi 后端原有支持入口，但其 Workspace 也使用唯一真实目录实现；本轮真实 Agent 端到端验收范围仍仅 Pi。

**验证命令与预期：**

    rg -n 'PTOBENCH_AGENT_WORKSPACE_ROOT|PTOBENCH_PUBLIC_DIR|/ptobench/candidates|Broker\.(export|sync_source|feedback)' evaluator launcher modes
    python -m unittest discover -s evaluator/tests -p 'test_isolation.py' -v

预期：第一条搜索无旧 Workspace View 运行路径；任何保留的响应路径格式化不再读取/复制 Candidate 或 Outputs；非 Workspace 历史归档无回归。

**完成标准及失败处理：** 若旧代码仍被实际调用，先迁移该调用再删除；不能通过保留两种 Workspace 后端或全量重构绕过调用关系检查。

---

---

### Task 7: 无 NPU 软件回归与 75 个 Workspace 扫描

**目标及实现方式：** 扩展现有 maintainer acceptance 工具以支持按已选 Case/action 运行；补齐 Combined artifact 验证。默认无参数行为保持当前“全部 registered Cases + clean/runtime/pmu/compiler split”合同。

**文件：**

- Modify evaluator/accept_all_cases.py：`AcceptanceRunner.__init__`、`run_case`、`blocked_case`、`summarize`、`run`、`main`；新增 `select_cases(rows, selection_csv)` 和 `action_items(actions, profile_strategy)`。
- Modify evaluator/acceptance_artifacts.py::inspect_artifacts：增加 `mode == 'combined'`，在同一 artifact root 上运行 Runtime、PMU、Compiler 三类现有 validator，全部通过才通过。
- Create evaluator/tests/test_workspace_acceptance.py：对 Formal/Held-out CSV 的 75 个 Case 调用 `preflight_case` 和 `workspace.init_workspace`/`apply_permissions`，临时目录中检查三个普通目录及 mode，不创建 Session Action 或启动 NPU。
- Test evaluator/tests/test_accept_all_cases.py、evaluator/tests/test_workspace_acceptance.py、evaluator/tests/test_agent_cli.py、evaluator/tests/test_isolation.py。

- [ ] 先写 selection/action 测试：空 CSV、缺 `case_id`、重复/未知 ID 在设备检查和 Session 创建前拒绝；保留注册 Case 原顺序。`evaluate` 只生成一项；combined profile 只生成一项；split profile 仍三项；默认仍四项。
- [ ] 让 AcceptanceRunner 按选中的 action 数计算 Session budget、Fork worker 数、摘要行数和 Finalize：只跑 evaluate 或 Combined Profile 时各用一个 Worker，不创建未使用的 Fork；默认四项仍保持四个独立 Worker 和原有预算/Finalize 行为。`--selection-csv`、`--actions all|evaluate|profile`、`--profile-strategy split|combined` 缺省时不改变当前行为。
- [ ] 写 Combined artifacts 测试：Runtime、PMU、Compiler 子验证全部通过才 PASS；分别缺失任一类、空文件或 schema 错误都 FAIL；split 既有报告保持不变。
- [ ] 对 75 个 CSV Case 做临时目录扫描：`lstat` 三个 Workspace 入口均为目录且非链接，mode 符合 Task 2；断言不存在 `mode.json`、`profiling_modes.json`、`fixture_catalog.json`。不创建 `runs/`、Snapshot 或 hardware Action，不写回 Case/Seed/Fixture/Reference。
- [ ] 增加正式代码不依赖 `lightweight_e2e` 的静态检查；仅当正式 import、调用或运行路径引用该目录时失败。
- [ ] `accept_all_cases.asset_roots` 纳入两个选择 CSV 所在的 `case_selection/`，让工具批次前后校验 Case、sealed assets、environment 和选择文件内容不变。

**验证命令与预期：**

    python -m unittest discover -s evaluator/tests -p 'test_accept_all_cases.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_workspace_acceptance.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_agent_cli.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_isolation.py' -v
    python -m unittest discover -s evaluator/tests -p 'test_*.py' -v
    rg -n 'lightweight_e2e' evaluator launcher ptobench pto-launcher

预期：测试通过；默认仍为每 Case 四项；75 个临时 Workspace 检查不创建 Session/硬件记录；正式代码扫描无 import/call 依赖。`rg` 无命中时退出码 1 是预期结果，有命中则失败。若有测试失败或输入资产 hash 变化，停止，不进入 Pi 或硬件验收。

**完成标准及失败处理：** Workspace 权限、CLI、Combined/split、Snapshot、Finalize、隔离和轻量 E2E 独立性检查全部通过后才开始 Task 8。任何 `BLOCKED` 不算 PASS。

---
