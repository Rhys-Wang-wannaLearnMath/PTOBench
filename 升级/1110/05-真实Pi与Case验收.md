# 真实 Pi 与 Case 验收 Implementation Plan

**目标：** 验证 Basic/CANNBot 的生产 Pi、安全 Workspace 和正式 Case 流程，并在全部 Plans 结束后，只对确实受代码改动影响的性能 baseline/Profile Case 安排串行 tmux 重测。

**已完成且可复用的证据：** Plan 03 ledger 记录了一次 Basic 生产 Pi Session，使用真实 Bubblewrap、Pi Shell、CLI transport 与 Broker，`evaluate` 和 Combined `profile` 均 PASS；49 项定向非 workload 测试及 CANNBot Pi 集成测试也已通过。开始本 Plan 时先核对这些证据与当前代码、环境锁、Case/资产 hash；符合下文复用条件就直接引用，不重复运行。

**架构：** 复用 Plan 03 已验证的 Basic Pi 与通用隔离边界，只补齐尚未覆盖的 CANNBot 生产 Pi 路径；之后对 Formal 60、Held-out 15 做 correctness Evaluate，并对已存在的 `(granularity, device_count)` 类别做 Combined Profile。PTOmni 本轮不实现、不验收。

**依赖：** Plan 01–04 完成；管理员已完成专用账号、共享组、sudoers 与 ACL 一次性配置。

**验收边界：** Case selection 只读。不得修改 Case、Fixture、Seed、Reference、Baseline、CV、`case_selection/` 或 `sealed_assets/`。旧 Session 不迁移、不恢复。Candidate Worker 可见评测实现暂不处理；其网络访问不作要求、不加限制、不测试，也不作为验收门槛。联网边界只针对 Pi Agent。所有 NPU Action 串行执行；失败或结果不确定时停止，不自动换例或重试。

## 必要验证

- 75 个 Formal/Held-out Case 各做一次 correctness Evaluate；这是逐 Case 的正确性覆盖，不用少量 Smoke 代替。
- 对两个 selection 中实际存在的每个 `(granularity, device_count)` 类别，各做一个 Combined Profile 代表 Case；不逐一 Profile 全部 75 个 Case。
- Pi/Workspace/Gateway 通用边界、Pi Agent 网络边界和 Basic `evaluate` + Combined `profile` 已由 Plan 03 实际验证。只有当 Plan 04 的改动触及相应生产调用路径、或原始 PASS 证据无法匹配当前代码/配置/资产 hash 时，才重跑受影响的单个 canary；Candidate Worker 网络不在验证范围。
- Basic 不再重复跑两次 `help`/`fork` Smoke。CANNBot 只补一次受 Broker 限权的真实 Pi `evaluate` canary，以验证其 mode/Skill/dispatch 接线；不再分别为 Formal 和 Held-out 各做一次相同 Smoke。
- 复用的 Evaluate/Profile 结果必须来自 PASS 的 Session/Event/Artifact，Case、Fixture、Environment、Evaluator 与运行路径证据均匹配当前验收输入。旧结果不能满足这些条件时，只补缺失的那项，不重跑整套 Plan 03 测试。

## Review Focus

- selection CSV 分别为 60/15 行且无重复 Case ID；运行前后 selection hash 不变。
- Profile 实际设备数必须与 selection 的 `device_count` 一致。Combined 产物使用 `profile_modes.validate_artifacts(build_dir, "combined")` 校验；`build_dir` 来自该 Profile Action 的权威 artifact 引用，不通过 glob 猜路径。
- 至少覆盖一个实际多设备样本。只有 Launcher 本次使用的已解析 host list/hostfile 能证明多个 host 时才可报告跨 host；不得从设备数推断多机。本 Plan 不新增 topology parser。
- Smoke 的操作白名单由 Broker 强制执行；真实 Pi、Shell、Bubblewrap、CLI transport、Gateway 与 Socket 使用生产路径，不泄露上游 API Key。网络检查只针对 Pi Agent 工具配置及其 Shell/Python 出口；不检查或限制 Candidate Worker 的网络访问。

## 修改文件

- 新建 `evaluator/selected_case_acceptance.py` 与 `evaluator/tests/test_selected_case_acceptance.py`。脚本只实现本 Plan 需要的 selection 验证、单次受限 Agent canary、Evaluate、Profile 和结果记录；不另建通用 manifest 框架。用现有 Session/CLI API 创建独立 Session；输出 `results.jsonl`、`summary.json` 和固定的 8 类 Profile 选择清单。成功动作仅在 Session/Event/Snapshot/Artifact 与输入 hash 一致时可复用；失败或状态不明禁止自动重试。
- 如当前 Broker 尚无单次 canary 限制，仅增加本次验收 harness 可设置的 Broker-side action allowlist/计数；不接受 Agent 或 Case YAML 中的开关，不改变正式 Session 的 Action 合同。
- 验收记录写入新建的 `reports/agent_upgrade/dedicated_agent_user/` 子目录，不写入 Case 或 `case_selection/`。

## 步骤

- [ ] 选择已锁定 Python，确认 Formal/Held-out CSV 行数和唯一性；保存两份 CSV 与 `environments/LOCK.json` 的 SHA-256。确认 `git diff` 和未跟踪文件检查显示 `cases/`、`sealed_assets/`、`case_selection/` 无本轮改动；如发现已有改动，先记录基线并确保结束时没有新增或改变。
- [ ] 核对 Plan 03 的 Basic 真实 Pi PASS 记录：从 Session/Event 读取 mode、Case、动作、Candidate/Fixture/Environment/Evaluator 与诊断 artifact 引用，并与当前代码及输入 hash 比较。若 Case 位于 selection 且各 hash 匹配，可复用它满足对应的 Evaluate；若它也是所选 Profile 类别的代表且 Combined artifact validator 通过，可复用该 Profile。否则只运行缺失动作。Plan 03 的定向测试和隔离检查不在此步骤重跑。
- [ ] 只有 CANNBot 生产 Agent canary 尚无等价且 hash 匹配的 PASS 证据时，选择 selection 中一个 Case，以新 Session 和 `smoke_*` optimizer 启动 CANNBot Pi；Broker 只允许一次 `evaluate`，拒绝其他或额外 Action，完成后停止 Agent。该 canary 不是 correctness 覆盖的替代品。
- [ ] 对 Formal 60 与 Held-out 15 执行 correctness Evaluate。每个 Case 使用独立 Session 和注册 Workspace；仅当前一步确认可复用的 Plan 03 PASS 记录允许跳过。逐项核验 CLI 状态、Event/Action、冻结 Snapshot、Fixture correctness 与 Session integrity。FAIL、BLOCKED、TIMEOUT、输入 hash 不匹配或基础设施错误时停止后续硬件队列并调查。
- [ ] 根据两个 selection 生成实际存在的 Profile 类别清单。每类先按 Case ID 排序，按现有 `session.preflight_case()`、`tuning.read()`、`common.environment()` 和 `device_preflight()` 检查一个候选；selection 设备数与运行配置不符是配置错误并立即停止，只有明确的 unsupported 才可在硬件开始前尝试同类下一个候选。所有 8 类确定后固定 `profile_selection.json`。无候选则报告覆盖缺口，不开始 Profile。
- [ ] 对清单中尚无可复用 PASS artifact 的代表 Case 串行执行 Combined Profile。以 Action/Event 的 artifact 引用定位 `build_dir` 并验证 Runtime、Hardware/PMU 与 Compile/Compiler 产物；Profile 结果只证明诊断采集，不替代 correctness，也不直接更新 latency baseline/CV。设备 health/preflight、温度或功耗异常时不启动硬件工作。
- [ ] 验收完成后重新核对 selection CSV、`LOCK.json` 和受保护目录状态；输出每个动作的 Case、mode、Session、状态、输入 hash、Snapshot/Event/Artifact 引用及复用/新运行理由。没有可验证的 host topology 时明确记为未覆盖，不声称多机通过。

## 验证

```bash
PYTHONPATH=evaluator "$PTOBENCH_LOCKED_PYTHON" -m pytest -q evaluator/tests/test_selected_case_acceptance.py
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py preflight --out reports/agent_upgrade/dedicated_agent_user
# 仅当 Plan 03 没有可复用的 CANNBot PASS 证据时执行
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py agent-canary --mode cannbot --operation evaluate --one-action --out reports/agent_upgrade/dedicated_agent_user/cannbot-canary
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py evaluate --out reports/agent_upgrade/dedicated_agent_user/evaluate
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py profile --prepare-only --out reports/agent_upgrade/dedicated_agent_user/profile
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py profile --out reports/agent_upgrade/dedicated_agent_user/profile
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py verify --out reports/agent_upgrade/dedicated_agent_user
git diff --exit-code -- cases sealed_assets case_selection environments/LOCK.json
git status --short -- cases sealed_assets case_selection environments/LOCK.json
git diff --check
```

从 PTOBench 根目录运行，`PTOBENCH_LOCKED_PYTHON` 必须来自 `environment_lock.locked_python_executable()`；不得固定主机绝对路径。命令按顺序执行，所有硬件动作串行。只有缺少可复用 CANNBot PASS 证据时才运行 canary。可复用动作不重复执行；只接受证据完整且 hash 一致的 PASS。遇到失败或状态不明立即停止，不自动重试。运行前后对比 `cases/`、`sealed_assets/`、`case_selection/` 和 `environments/LOCK.json` 的 Git 状态及 tracked diff hash；两次记录必须一致。

## 全部 Plans 完成后的重测队列

此阶段在 Plan 01–05 的实现和验收全部通过后单独执行；不属于当前 Case 验收，也不得提前启动。

- 审阅最终代码差异，将改动映射到实际运行路径。若只改 Agent 身份、Pi 文件视图、Gateway/Socket、README 或 Session 管理，且未触及 Candidate 执行、计时或 Profile 采集路径，则性能 baseline/Profile 无需重测，记录理由即可；Pi 功能性 Combined Profile 验收不自动进入性能重测队列。
- 若改动触及 Candidate 编译/执行、设备配置、计时窗口、baseline 读写或测量合同，选择所有实际依赖该路径的 Case 做 baseline 重测；共享路径影响全量 Case 时才扩展到全量，不默认重测 311 个 Case。
- 若改动触及 Combined Profile 调度、采集或产物格式，只重测受影响的 Profile 类别；若通用采集/产物合同改变，则按其真实影响范围扩展。Profile-only 改动不自动触发 correctness 或 baseline 重测。
- 写出只读 queue manifest，逐项记录 Case、任务类型（baseline 或性能 Profile）、代码影响路径、选择理由、正式命令、前置 hash 和预期产物。只有 manifest 中有必要重测项时，才通过目标主机已使用的串行 tmux 流程排队；baseline 与 Profile 任务均逐项列明，一次只运行一个硬件任务。没有必要项时记录“无需重测”及依据，不创建空队列。仓库未提供可直接复用的 tmux queue launcher；实施者须核对并记录实际使用的主机命令，不能猜测或绕过正式 CLI。
- 队列失败、超时或状态不明时停止队列，不自动换 Case 或重试。新结果按 PTOBench 现有 baseline/CV 发布规则处理，不覆盖旧记录；Profile artifact 保留为新的运行结果。

## 完成标准与停止条件

Basic/CANNBot 的必要 Pi Agent 边界、75 个 Formal/Held-out correctness 结果、8 类 Combined Profile、输入未改写证据及可声明的设备/host 覆盖均通过后，签署单机部署验收。若没有 host topology 证据，明确记为未覆盖；多机部署不得签署通过。任何身份、私有文件、Socket、Gateway、Workspace、Snapshot、正确性、Profile 或输入检查失败，立即停止并调查；Candidate Worker 网络访问不属于检查项。Plan 05 验收结束后，等待 Plan 01–05 全部完成，再按上节将必要的 baseline/Profile 重测加入串行 tmux 队列；在该后续阶段完成前，不宣称性能重测已完成。
