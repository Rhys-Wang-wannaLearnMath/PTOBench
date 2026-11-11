已被 1110 Dedicated Agent User 计划取代，仅供历史参考、不得执行。

# 真实 Pi Smoke 与 Formal/Held-out 最终验收 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task.

**Goal:** 用 Basic/CANNBot 真实 Pi 及已选 Formal/Held-out Case 验收真实 Workspace 边界、Evaluate 和 Combined Profile。

**Architecture:** 只在无 NPU 软件回归通过后复用正式 Pi、Broker、Gateway 和 Candidate Worker 路径；先运行不占 NPU 的 Smoke，再严格串行执行硬件验收。本计划只执行原计划 Task 8–9。

**Tech Stack:** Python 3、Bubblewrap、Linux user namespace、POSIX UID/GID 与目录权限、现有 unittest 测试。

**Spec:** [`升级/1109/PTOBench真实目录Workspace与沙箱简化修缮计划.md`](../../PTOBench真实目录Workspace与沙箱简化修缮计划.md)

**Source plan:** [`升级/1109/plans/PTOBench真实目录Workspace与沙箱简化Implementation Plan.md`](../PTOBench真实目录Workspace与沙箱简化Implementation Plan.md)

**Execution order:** This is Plan 05 of 05. Plan 01–04（原计划 Task 0–7）全部完成并通过；Task 9 必须等待 Task 8 的四个 Basic/CANNBot Formal/Held-out Pi Smoke 通过。

**Plan boundary:** Execute only original Task 8–9; do not implement tasks assigned to another split plan. Task 8–9 与最终发布门槛全部通过；任何 `BLOCKED`/`UNSUPPORTED`、correctness、Snapshot、integrity、Combined artifact、HCCL 或资产 hash 问题均按原计划停止，不宣称可投入使用。

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

### Task 8: Basic/CANNBot 真实 Pi Smoke（不占 NPU）

**目标及实现方式：** 增加 maintainer-only Smoke 命令，在临时 Session 中复用 Basic/CANNBot 的既有 `preflight`、`prepare`、`run_private_pi`、真实 `agent_command`、Broker 和 Gateway。Smoke harness 不调用 Launcher 自动 Finalize，避免无 NPU Smoke 意外触发 final clean run。只给 Smoke Session 写短 prompt，要求 Pi 创建 helper 并执行 `ptobench fork .`；Broker 在此场景只允许 `fork`/help，Agent sandbox 不挂 NPU device，因此不会启动 Candidate Worker 或硬件 Action。此测试不改正式 Case、prompt、mode instructions 或 Agent-facing CLI。

**文件：** Create evaluator/pi_sandbox_smoke.py；Modify evaluator/sandbox_supervisor.py::Broker.__init__、Broker.serve、run_private_pi（增加仅供 maintainer harness 调用的 `allowed_operations` 参数，生产默认集合不变）；Modify modes/basic/runner.py::prepare、modes/cannbot/runner.py::prepare、modes/cannbot/runtime.py::run_dispatch（只在受信任 harness request 中替换任务 prompt，并关闭 CANNBot Task 子 Agent 工具；不接受 Agent/experiment YAML 传入此标记）；Test evaluator/tests/test_pi_sandbox_smoke.py。复用 evaluator/session.py::create_session、launcher/methods.py::call/retain_method_reply、evaluator/isolation.py::agent_command。

- [ ] Smoke CLI 接受 `--mode basic|cannbot`、`--case-id`、`--operation fork|evaluate-profile`、`--output-dir`；`fork` 为默认操作。Case 必须来自 formal_60 或 heldout_15 CSV。Profile 从对应 mode 的 `.env` 解析，启动前断言模型为 DeepSeek Flash、thinking 为 High。临时 Session/Workspace 建在 `TemporaryDirectory`，报告和日志仅写新的 `reports/agent_upgrade/direct_workspace/pi_smoke/`。
- [ ] Smoke harness 在可丢弃 Session 中，通过对应 mode 的既有 `preflight` 和 `prepare` 传入受信任内部标记，让 Smoke prompt 在 `retain_method_reply` 和完整性保护前生成；之后复用既有 request/provenance/冻结路径并直接调用 `run_private_pi`，不进入 Launcher 自动 Finalize。Basic prompt 由 `modes/basic/runner.py::prepare` 生成，CANNBot prompt 由 `modes/cannbot/runner.py::prepare` 写入 `method/config.json` 和 `prompt.txt`。禁止准备结束后改写 `prompt.txt`，不得覆盖任何正式 Session 或正式 prompt。
- [ ] `fork` Smoke prompt 只要求读取 Workspace README、在 `candidate/` 创建 helper、执行 `ptobench fork .`、读取子 Worker 的 Candidate 并退出。Broker allowlist 只允许 fork/help 并拒绝 evaluate/profile；CANNBot ToolServer 不开放 Task 子 Agent 工具；确认 Agent sandbox 未挂载 NPU device、Session 中没有 hardware action 或 Snapshot。
- [ ] `evaluate-profile` Smoke prompt 只要求对原 Candidate 依次执行 `ptobench evaluate .` 和 `ptobench profile .`，不编辑代码或 tuning。Broker allowlist 只允许 evaluate/profile/help，Session 预算设为两个 hardware action；Candidate Worker 由受信任 Broker 启动，不给 Agent sandbox 挂 NPU device。此操作用于 Task 9 硬件验收。
- [ ] Basic/CANNBot 各覆盖一个 Formal 和一个 Held-out，共四次真实 Pi 启动；检查加载到的是各自 `.env` 的 DeepSeek Flash / High。CLI/Gateway/CANNBot ToolServer（CANNBot 路径）连接成功，Agent 进程没有上游 key。
- [ ] 权限拒绝项由 Task 4 的确定性 Bubblewrap 探针验收，不依赖模型是否照做；真实 Pi Smoke 只验证 Pi 启动、正常 Candidate 写入、Broker fork 可见、父子 Candidate 独立和会话归档。
- [ ] 每次 `fork` 运行后检查 Session 没有 hardware Action/Snapshot/Candidate Worker；同一 Pi 进程内可见 fork，Agent 修改子 Candidate 后 Broker 可读且父 Candidate 不变。Task 8 只运行 `fork` Smoke。

**验证命令与预期：**

    python -m unittest discover -s evaluator/tests -p 'test_pi_sandbox_smoke.py' -v
    python evaluator/pi_sandbox_smoke.py --mode basic --case-id deepseek_v4_flash_mtp.sample_pro --operation fork --output-dir reports/agent_upgrade/direct_workspace/pi_smoke/basic_formal
    python evaluator/pi_sandbox_smoke.py --mode basic --case-id qwen3_14b.topk_select_k1 --operation fork --output-dir reports/agent_upgrade/direct_workspace/pi_smoke/basic_heldout
    python evaluator/pi_sandbox_smoke.py --mode cannbot --case-id deepseek_v4.prefill_layer_ep2_csa_demo --operation fork --output-dir reports/agent_upgrade/direct_workspace/pi_smoke/cannbot_formal
    python evaluator/pi_sandbox_smoke.py --mode cannbot --case-id deepseek_v4_flash_dspark.prefill_indexer_flash --operation fork --output-dir reports/agent_upgrade/direct_workspace/pi_smoke/cannbot_heldout

预期：四次真实 Pi smoke 完成；`hardware_actions_used == 0`、没有 Snapshot/Evaluator/NPU 调用；模型 API 经 Gateway 工作，三类必要 Unix Socket 均可用。

**完成标准及失败处理：** 任一 Pi 无法启动、Socket/Gateway 失败、权限合同失败、fork 不可见、Agent 能调用硬件 Action 或出现 NPU 访问时，停止 Task 9；修复后用新的临时 Session 和 output-dir 重跑四项。

---

---

### Task 9: Formal/Held-out Evaluate 与 Combined Profile 硬件验收

**目标及实现方式：** 新实现投入前，使用 Task 7 的同一 acceptance 工具对已选 60 个 Formal 和 15 个 Held-out Case 做正确性 Evaluate；再以 Combined profile 覆盖正式 Case 的三种 Granularity、1/2/4/8 卡及 Held-out 可用的 Granularity/1/2/4 卡。Profile coverage 只存入新 reports；不更新 Baseline/CV，不改选择 CSV。

**文件：** 无新增运行代码；使用 evaluator/accept_all_cases.py 及新验收输出目录。可在 `reports/agent_upgrade/direct_workspace/combined_profile_cases.csv` 保存经审核的 Profile coverage 输入清单。

**当前 selection CSV 已核实：** Formal 60 分布为 operator 12、subgraph 43、model 5；卡数为 1 卡 11、2 卡 28、4 卡 11、8 卡 10。Held-out 15 为 operator 3、subgraph 10、model 2；卡数为 1 卡 10、2 卡 3、4 卡 2、无 8 卡。可用的覆盖样例包括：Formal `deepseek_v4_flash_mtp.sample_pro`（operator/1 卡）、`deepseek_v4.synthetic_token_loop_prefill_flash`（model/2 卡）、`deepseek_v4.prefill_layer_ep4_hca_flash`（subgraph/4 卡）、`deepseek_v4.prefill_layer_ep8_csa_demo`（subgraph/8 卡）；Held-out `qwen3_14b.topk_select_k1`（operator/1 卡）、`deepseek_v4_flash_dspark.dspark_prefill_pro`（model/2 卡）、`deepseek_v4.moe_ep_ep4_flash`（subgraph/4 卡）。

- [ ] 先从正式 selection CSV 和环境 JSON 审核覆盖列表；验证 granularity 与 device_ids 和 CSV 一致，并判断是否有明确跨 Host 配置。8 卡只代表 8 个设备，不代表多机。若存在跨 Host Case，加入覆盖列表；若没有，报告“本轮无可验证的多机样本”，不声称已覆盖多机。
- [ ] 分别运行 Formal 60、Held-out 15 的 `--actions evaluate`；要求每个 Case 的所有 declared Fixture correctness、Snapshot、integrity 都 PASS。Acceptance 结果必须恰好每 Case 一条 clean/evaluate 记录。
- [ ] 用经审核的 coverage CSV 运行 `--actions profile --profile-strategy combined`；每个 Case 必须由 `ptobench profile .` 产生同次 Runtime、PMU、Compiler 三类有效 artifact，缺一项即失败。Combined Profile 不替代 Evaluate 或 clean latency 检查。
- [ ] 用真实 Basic Pi + Formal `deepseek_v4_flash_mtp.sample_pro`、CANNBot Pi + Held-out `qwen3_14b.topk_select_k1` 各运行一次 `--operation evaluate-profile`；确认同一真实 Candidate 经 Broker RPC 完成 Evaluate 和 Combined Profile、结果由 Agent 可读且只读、Action/Snapshot/Integrity 记录通过。每个 Session 正好两项 hardware action，不触发 Launcher 自动 Finish，不改 Candidate/tuning；这是验证 Agent 到 Broker 到 Candidate Worker 的路径，不代替 75-case Acceptance。
- [ ] 硬件任务严格串行；每批前按功耗/温度和任务协调确认可运行，不以 AICore 利用率判断设备忙闲。多卡 Case 检查 HCCL 和完整 device_ids；真实跨 Host Case（若存在）单独串行。
- [ ] 批次前后保留 acceptance 工具的 Case、sealed、environment、selection CSV hash 对比；输出必须在新 `reports/agent_upgrade/direct_workspace/` 目录。确认 Candidate/Fixture/Seed/Reference/selection CSV/正式 Baseline/CV 未修改。
- [ ] 若实现改动触及 Evaluate/Profile 的 Runtime 调用、状态复位、编译、数据采集或计时路径，扩大 Profile/Evaluate 覆盖到所有受影响 Case；不能假定 Workspace 改动天然不影响测量。

**验证命令与预期：**

    batch=$(date -u +%Y%m%dT%H%M%SZ)
    python evaluator/accept_all_cases.py --out "reports/agent_upgrade/direct_workspace/formal_60/$batch" --selection-csv case_selection/formal_60/cases.csv --actions evaluate
    python evaluator/accept_all_cases.py --out "reports/agent_upgrade/direct_workspace/heldout_15/$batch" --selection-csv case_selection/heldout_15/cases.csv --actions evaluate
    python evaluator/accept_all_cases.py --out "reports/agent_upgrade/direct_workspace/combined_profile/$batch" --selection-csv reports/agent_upgrade/direct_workspace/combined_profile_cases.csv --actions profile --profile-strategy combined
    python evaluator/pi_sandbox_smoke.py --mode basic --case-id deepseek_v4_flash_mtp.sample_pro --operation evaluate-profile --output-dir "reports/agent_upgrade/direct_workspace/pi_hardware/basic_formal/$batch"
    python evaluator/pi_sandbox_smoke.py --mode cannbot --case-id qwen3_14b.topk_select_k1 --operation evaluate-profile --output-dir "reports/agent_upgrade/direct_workspace/pi_hardware/cannbot_heldout/$batch"

预期：Formal 60 与 Held-out 15 的 75 个 Case 全部 correctness PASS；覆盖清单中的 Combined Profile 都产出三类有效诊断；两个真实 Pi Broker 测试中的 Evaluate/Combined Profile 都 PASS 且反馈只读；对 Case/环境/选择 CSV 的 before/after hash 一致。若环境清单没有跨 Host 表达，报告该项未覆盖，不将 8 卡冒充多机测试。

**完成标准及失败处理：** 任一 correctness 失败、Snapshot/integrity 失败、Combined 三类证据缺失、多卡/HCCL 执行失败或正式输入 hash 变化，均阻止投入使用。记录失败 Case 和原因；不重建 Baseline、不更新 CV、不将 `UNSUPPORTED`/`BLOCKED` 当作 PASS。确认停止与清理完毕后才运行下一项。
