# 真实 Pi 与 Case 验收 Implementation Plan

**目标：** 通过 Basic/CANNBot 真实 Pi 以及 Formal/Held-out Case 证明专用 Agent 用户和真实 Workspace 可以投入使用。

**架构：** 先完成 Basic no-NPU Pi Smoke 与所有不占 NPU 的隔离检查；随后串行运行 CANNBot 固定 Smoke Evaluate、正确性 Evaluate 和 Combined Profile。Basic/CANNBot 使用共同的 Workspace/Agent 合同；PTOmni 本轮不实现、不验收。

**验收边界：** 所有 Smoke 均创建新 Session 并走唯一的专用 Agent 用户路径；不通过旧 Session Workspace、旧身份映射或 legacy runner 验收。Formal/Held-out 清单只作为只读、数据驱动的验收输入；脚本和计划不嵌入 Case ID，不修改 Case、Fixture、Seed、Reference、Baseline、CV 或选择数据。旧历史文件可留存，但不迁移、不恢复，也不要求新实现兼容旧隔离状态。

**依赖：** Plan 01–04 通过；管理员完成一次性账号、共享组、sudoers 与 ACL 配置。

**Tech Stack：** PTOBench CLI、Pi、Broker/Gateway、pytest 与现有 Case selection；Profile/Evaluate 使用既有设备执行路径。

**Spec：** [`00-overview.md`](00-overview.md)，尤其是全局不变量、实验边界与最终验收条件。

## 全局约束

- 先完成所有不占 NPU 的检查，再串行执行 CANNBot Smoke、Formal/Held-out Evaluate 与 Combined Profile。
- Smoke 必须由 Broker 侧操作白名单限权，不能只依靠 Agent prompt。
- 不修改 Case、Fixture、Seed、Reference、Baseline、CV 或 selection 数据；不以 8 卡冒充多机覆盖。

## Review Focus

- Basic no-NPU Smoke 与 CANNBot 单次 clean Evaluate 的操作边界必须被 Broker 强制执行。
- 75 个正式 Case 正确性 Evaluate 与 8 个 `(granularity, device_count)` 类别代表性的 Combined Profile 使用 selection 中的数据驱动 Case ID；不复制或改写 Case 内容。`session.preflight_case(case_id)` 返回 `(case_dir, sealed, case, environment_profile)`；对将要运行的已注册 Candidate 使用 `tuning.read(workspace, case)` 读取当前 `candidate/tuning.json`，在临时设置 `PYPTO_BENCH_PROFILE=combined` 后调用 `env_updates, runtime = common.environment(case, environment_profile["platform"], tuning)`。实际设备列表取自 `runtime.distributed_config.device_ids`，若无则为 `[runtime.device_id]`。selection CSV 的 `device_count` 必须等于实际设备列表长度；不一致即判输入/配置错误并停止，不能按其中一个值继续。`accept_all_cases.required_devices()` 只接受 mapping，因此仅对规范化后的 `{"runtime":{"device_ids": effective_devices}}` 调用；`device_preflight()` 使用 Case environment profile 的 Python/setup/env 字段，并将其 runtime device ids 替换为上述有效设备列表。此 preflight 不执行 Candidate 或启动 NPU workload，但会按现有 helper 对选定设备发出 `npu-smi` health 查询。Combined 产物检查复用 `profile_modes.validate_artifacts(build_dir, "combined")`；不要把它传给只支持 `runtime`/`pmu`/`compiler` 分体模式的 `acceptance_artifacts.inspect_artifacts()`。`build_dir` 必须来自该 Profile Action 的权威 artifact 引用，不通过猜测路径或任意 glob 定位。当前 `common.environment`/Launcher 代码只暴露设备列表，未提供 host topology 解析器；不为本计划新增解析框架。只有实际 Launcher 本次使用的已解析 host list/hostfile 等明确拓扑输入及其 hash 能证明多个 host 时，才记录跨 host 覆盖；没有该输入就标记 `unverifiable/not covered`，不能由 `device_count` 推断。基础设施错误不自动换例或重试。
- 真实 Pi/Shell 使用生产 mount、Socket 和 Gateway 路径；验收 harness 不绕过 Broker 或泄露 API Key。

## 修改文件

- Create `evaluator/pi_sandbox_smoke.py` 及 `evaluator/tests/test_pi_sandbox_smoke.py`：CLI 接受 `--mode`、`--case-id`、`--operation`、必需的 `--manifest` 和 CANNBot 专用的 `--optimizer-id`；启动 Pi 前核验输入 manifest 摘要，不匹配即在创建 Session/Case Action 前退出。Basic 使用验收脚本提供的短期 smoke prompt（不改 mode prompt），依次请求 `ptobench --help`、创建 Candidate helper、Fork 一次；`--operation help` 映射为现有 CLI 的 `ptobench --help`（没有 `ptobench help` 子命令）。CANNBot 使用当前 `smoke_*` prompt 且只允许一次 clean `evaluate`。由 maintainer harness 给 Broker 设一次性操作白名单和计数；越权或超次请求立即拒绝并停止 Smoke Agent，不依赖 prompt 自律。复用已有 Basic/CANNBot preflight、prepare、`run_private_pi` 和 Broker；仅在新建的临时验收 Session 中启动。
- Create `evaluator/selected_case_acceptance.py` 及 `evaluator/tests/test_selected_case_acceptance.py`：提供 `manifest`、`evaluate`、`agent-action`、`profile` 四个显式阶段；`manifest --create` 在任何 Case preflight/Smoke 前生成不可变输入清单，`manifest --verify` 在验收结束时重算并逐项比较；其余阶段记录并校验同一清单摘要。`profile --prepare-only` 只做选择与不执行 Case/NPU workload 的 preflight（设备 health 查询单独记录）；只读读取 Formal/Held-out selection CSV，创建独立 Session，使用 `ptobench` 的实际 Workspace CLI 参数完成 75 个正确性 Evaluate、生产 Agent Bubblewrap 通道 canary、8 个类别的 Combined Profile 和 selection 内可验证的多卡/跨 host 拓扑覆盖；不改 Case 或 selection。输出 `results.jsonl`、`matrix.csv`、`summary.json`、固定选择清单 `profile_selection.json` 和输入清单；每条记录包含 selection/Case ID、phase/action、有效设备列表、host topology source/hash/可验证的 distinct host 数（没有权威输入时记为 `unverifiable`）、Session/Worker、Snapshot/Event、状态、artifact 引用及输入 manifest hash。已有成功动作仅在 Session/Event/Artifact 与输入 hash 一致时可跳过；失败或结果不确定的硬件动作禁止自动重试，必须停止并人工调查。
- Modify `evaluator/sandbox_supervisor.py::Broker.serve`、`run_private_pi` 与 Basic/CANNBot prepare 的测试接口：只接受 maintainer harness 的临时 Smoke 行为，不接受 Agent/YAML 开关，不改正式 Case 和 prompt。Broker 对 Basic Smoke 只接受 `--help` 与一次 `fork`，对 CANNBot Smoke 只接受一次 `evaluate`；第二次或其他 Action 触发拒绝并终止该 Smoke Agent。
- The `agent-action` phase uses a maintainer-only harness to launch the configured Pi Shell wrapper through production `isolation.agent_command` and invoke the mounted `ptobench` transport client over the live Broker Socket. It accepts one fixed Case/Session/Worker/action per invocation, does not start an LLM optimization loop, call host Broker API, or expose a maintenance-only switch to the Agent.
- Create 验收记录于新的 `reports/agent_upgrade/dedicated_agent_user/`；不写入 `case_selection/` 或 Case 数据。

## 步骤

- [ ] Smoke 只接受 Basic/CANNBot 和 Formal/Held-out CSV 中存在的 Case ID。Basic 的 no-NPU Pi Smoke 使用只在临时验收 Session 生效的 smoke prompt（不改正式 mode prompt），依次请求 `ptobench --help`、在 Candidate 创建 helper 并 Fork 一个 Worker；Broker 最多接受一次 `--help`、只接受一次 `fork`，完成后停止 Agent，不触发 Evaluate/Profile/Finalize。CANNBot 现有 `smoke_*` optimizer prompt 明确要求恰好运行一次 Evaluate，因此 CANNBot Smoke 不得再设置 fork-only 预期：Formal 和 Held-out 各选一个 Case，分别用 `optimizer_id` 以 `smoke_` 开头的 Session 完成一次 clean Evaluate 后停止；Broker 只接受一次 `evaluate`，拒绝 Fork/Profile/重复 Evaluate。这两次 CANNBot Smoke 是硬件 Action，排在所有 no-NPU 检查之后并串行运行。
- [ ] Basic 的 Formal/Held-out no-NPU Smoke 各运行一次，并在受限 Broker/CLI harness 中只允许 `ptobench --help`/一次 `fork`，拒绝 Evaluate/Profile/Finalize；完成预期操作后终止 Smoke Agent。记录运行时实际宿主 UID/GID/组、`CapEff`、`NoNewPrivs`。Agent 仅有共享组、没有 sudo、不能直接打开 NPU 设备节点；Broker 与受信任 NPU Worker 仍有原身份和硬件权限。
- [ ] 检查 Pi 设置、session、锁文件可正常读写并可由 Broker 归档；Pi 原生文件工具和 Shell/Python 均不能读取私有设置/Session；Context/Outputs 不可写；Candidate/helper/tuning 可写。用 Pi 原生文件工具及 Shell/Python 分别新建和原子替换 Candidate `.py` 与 `tuning.json`，再由 Broker 读取并完成 Snapshot；Snapshot 对单独注入的 `0600` Candidate 文件仍应明确拒绝。
- [ ] 在同一个存活的 Pi 和嵌套 Shell 进程中验证 Broker 后台发布 Worker 后无需重启/重挂载即可见；同 Session 每个 Worker 的 Candidate 都可读写、Context/Outputs 都可读但不可改；父子 Candidate 独立，Broker 更新一个 Worker Outputs 不覆盖其他结果。
- [ ] 验证 `/tmp` 临时写入与宿主隔离、Runtime 可读/可执行但不可写、其他 Session/Case 与 PTOBench Broker/Evaluator/Launcher 源码对 Agent Shell/Python 不可见。Candidate Worker 进程可能看到更多评测文件，这一项明确不在本次验收范围。
- [ ] 检查 `.env` 分属各 mode、DeepSeek Flash/High 配置生效；通过 Basic 两次 Pi Smoke 和之后的 CANNBot Smoke 验证模型/Gateway 请求正常。验证真实 Key 不在 Agent Shell/Python、Request、Pi/Session 配置或日志中；CLI、Gateway、CANNBot Socket 正常；Web Search/Web Fetch 不可用、Shell/Python 公网连接失败。占位值不是认证凭据；Socket DAC 与 Bubblewrap 可见性是访问边界，Candidate Worker 进程状态不属于本次保证。
- [ ] 所有 no-NPU、私有文件、Runtime、Workspace、Socket 和网络检查完成后，才运行 CANNBot Formal/Held-out `smoke_*` Session。每个 Session 必须使用当前 `modes/cannbot/runner.py` 的固定 Smoke prompt，恰好执行一次 clean Evaluate 后停止；不 Fork、不 Profile、不继续优化。任一 Evaluate 非 PASS 即停止 Case 验收。
- [ ] Smoke 全部通过后，`selected_case_acceptance.py` 对 Formal 60 与 Held-out 15 各运行一次正确性 Evaluate，共 75 个新 Session；每个 Case 都通过 Session API 创建独立 Session、设置 `PTOBENCH_SESSION_ROOT` 并调用其注册 Workspace。只验收既有 Candidate 正确性，不启动优化 Agent，不改 Case。运行前后记录并比较 Formal/Held-out selection CSV 的 hash；selection 行数/Case ID 重复、hash 改变、Case FAIL/BLOCKED/TIMEOUT 或输入资产 hash 不符时停止批次并调查，不继续后续 Case。
- [ ] 对每次 Evaluate 验证 CLI JSON、事件与 Candidate Snapshot 一致、所有 Fixture correctness PASS、Action/Session integrity PASS；任何 Case FAIL/BLOCKED/TIMEOUT 都单列停止调查，不把 Profile 结果代替正确性。
- [ ] 75 个 correctness Evaluate 完成后，先执行 `profile --prepare-only`；它对两个 selection CSV 合并后的每个已存在 `(granularity, device_count)` 类别列出按 Case ID 排序的候选，并完成不占 NPU 的 Session/资产/环境 preflight。对每个候选按 Review Focus 中的真实调用链重建有效 `RunConfig`，并检查 CSV `device_count` 与有效 `device_ids` 数量一致；不匹配是配置错误，不可作为普通 unsupported 候选跳过。每类选字典序最小且 preflight PASS 的 Case；仅当 preflight 明确报告该 Case 不受支持时，才可在开始硬件 Action 前尝试同类下一个候选并记录原因。资产 hash mismatch、运行环境不符或任何基础设施错误都停止全批次，不换 Case。任何类别没有可运行候选时，在 Profile 阶段开始前停止并报告覆盖缺口。8 类均确定后，将清单写入 `profile_selection.json` 并固定；对 selection 中可运行的多卡样本至少覆盖一个实际多设备 Case。只有本次 Launcher 使用的既有已解析 host list/hostfile 等权威输入明确列出多个 host 时才选择跨 host Case；当前代码没有通用 host topology parser，不得为此临时新增。不存在或无法读取该权威输入时，将跨 host 标成未覆盖，不能宣称多机通过。
- [ ] Profile 候选选定后，先运行两项 `agent-action --operation evaluate` canary（Basic、CANNBot 各一个正式 Case）。全部请求都由生产 Shell wrapper、Bubblewrap、Workspace mount、transport shim 和 Broker Socket 执行；CLI 响应、Event、Snapshot 与 Session 记录须一致。
- [ ] 所有 Profile 候选的 Session/资产/环境 preflight 完成且 Evaluate canary 通过后，运行两项 `agent-action --operation profile --profile-strategy combined` canary（Basic、CANNBot 各一个正式 Case），再运行 8 类 `profile` 阶段。每个 Combined Action 从其 Event/Action 记录取得权威 `build_dir`，调用 `profile_modes.validate_artifacts(build_dir, "combined")`，确认 Runtime、Hardware/PMU、Compile/Compiler 三类产物齐全且 validator 返回通过；保存 validator 报告和 artifact 引用。不要调用 `acceptance_artifacts.inspect_artifacts(..., "combined")`，该 helper 不支持 Combined。任何硬件 Profile 失败、超时或结果不确定都停止，不自动换 Case 或重试。Profile 只验收诊断采集，不判定输出正确性或正式 latency。
- [ ] 硬件任务串行运行；只在温度/功耗正常且设备调度空闲时开始。不得仅凭 AICore 利用率决定设备是否空闲。
- [ ] 在本 Plan 第一个 Smoke/Case preflight 前运行 `selected_case_acceptance.py manifest --create`，将 `input_manifest.before.json` 写入 `reports/agent_upgrade/dedicated_agent_user/`。清单用 UTF-8 JSON 保存按仓库相对 POSIX 路径排序的每个目录、普通文件和符号链接条目；普通文件记录 SHA-256 与大小，符号链接记录 link target，目录条目用于检测新增/删除路径。输入范围为整个 `cases/`、`sealed_assets/`、`case_selection/`，`environments/LOCK.json`，以及本轮选中 Case 的 environment profile 所引用的锁定环境配置文件。Cases/sealed assets 下的 Fixture、Seed、Reference、Baseline、CV 和其索引/manifest 由目录递归覆盖；不要把生成的 `runs/`、`reports/`、Pi Session、Profile artifact 或 Candidate 输出纳入输入范围。把清单自身的 SHA-256 摘要记录在 `summary.json`、`results.jsonl` 和 `matrix.csv`；Smoke 记录也保存该摘要。每个 Smoke 通过 `pi_sandbox_smoke.py --manifest`、每个 Evaluate/Agent Action/Profile 阶段通过 `selected_case_acceptance.py` 在开始前核验当前输入树仍与 `input_manifest.before.json` 相符；任何 Session/报告只能写入批准的各自 `runs_root` 或验收报告目录，不允许写入被哈希的输入根。全部 Smoke、75 个 Evaluate、Agent 通道 canary、Profile 和报告生成完成后运行 `manifest --verify`，重算为 `input_manifest.after.json` 并逐条比较路径集合、类型、link target、文件大小和 SHA-256。任何新增/删除/改写或不能读取的输入路径都停止验收并调查，不能重建 before manifest 来覆盖变化。manifest 创建/校验的单元测试覆盖排序稳定、路径新增/删除、普通文件内容变化、符号链接目标变化、权限不足读取失败；CI 测试使用临时目录，不访问或修改 Case。若本次代码改动触及采集、Profile 状态复位、Runtime 编译/执行或测量路径，按影响扩大正确性和性能复验，不假设 Profile 不受影响。

## 验证

```bash
PYTHONPATH=evaluator "$PTOBENCH_LOCKED_PYTHON" -m pytest -q evaluator/tests/test_pi_sandbox_smoke.py
PYTHONPATH=evaluator "$PTOBENCH_LOCKED_PYTHON" -m pytest -q evaluator/tests/test_selected_case_acceptance.py
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py manifest --create --out reports/agent_upgrade/dedicated_agent_user
"$PTOBENCH_LOCKED_PYTHON" evaluator/pi_sandbox_smoke.py --mode basic --case-id <formal-case-id-from-csv> --operation fork --manifest reports/agent_upgrade/dedicated_agent_user/input_manifest.before.json
"$PTOBENCH_LOCKED_PYTHON" evaluator/pi_sandbox_smoke.py --mode basic --case-id <heldout-case-id-from-csv> --operation fork --manifest reports/agent_upgrade/dedicated_agent_user/input_manifest.before.json
"$PTOBENCH_LOCKED_PYTHON" evaluator/pi_sandbox_smoke.py --mode cannbot --case-id <formal-case-id-from-csv> --operation evaluate --optimizer-id smoke_formal --manifest reports/agent_upgrade/dedicated_agent_user/input_manifest.before.json
"$PTOBENCH_LOCKED_PYTHON" evaluator/pi_sandbox_smoke.py --mode cannbot --case-id <heldout-case-id-from-csv> --operation evaluate --optimizer-id smoke_heldout --manifest reports/agent_upgrade/dedicated_agent_user/input_manifest.before.json
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py evaluate --out reports/agent_upgrade/dedicated_agent_user/evaluate
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py profile --prepare-only --out reports/agent_upgrade/dedicated_agent_user/profile
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py agent-action --via-agent-sandbox --mode basic --case-id <formal-case-id-from-csv> --operation evaluate --out reports/agent_upgrade/dedicated_agent_user/agent-action/basic-evaluate/<case-id>
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py agent-action --via-agent-sandbox --mode cannbot --case-id <formal-case-id-from-csv> --operation evaluate --out reports/agent_upgrade/dedicated_agent_user/agent-action/cannbot-evaluate/<case-id>
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py agent-action --via-agent-sandbox --mode basic --case-id <formal-case-id-from-csv> --operation profile --profile-strategy combined --out reports/agent_upgrade/dedicated_agent_user/agent-action/basic-combined-profile/<case-id>
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py agent-action --via-agent-sandbox --mode cannbot --case-id <formal-case-id-from-csv> --operation profile --profile-strategy combined --out reports/agent_upgrade/dedicated_agent_user/agent-action/cannbot-combined-profile/<case-id>
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py profile --out reports/agent_upgrade/dedicated_agent_user/profile
"$PTOBENCH_LOCKED_PYTHON" evaluator/selected_case_acceptance.py manifest --verify --out reports/agent_upgrade/dedicated_agent_user
```

从 PTOBench 根目录运行前，将 `PTOBENCH_LOCKED_PYTHON` 设为该主机 `environment_lock.locked_python_executable()` 返回的绝对路径；不能使用固定 `.venv-official` 路径。先执行 `manifest --create`，之后按顺序运行 Basic smoke、CANNBot smoke、75 个 correctness Evaluate、`profile --prepare-only`、两项 Agent `evaluate` canary、两项 Combined Profile canary、8 类 Combined Profile，最后执行 `manifest --verify`。每个 Smoke/Case 阶段开始前重新核对 manifest；任何输入变化即停止。四条 `agent-action` 命令各有独立 `--out` 目录，避免报告覆盖。验收脚本读取 `case_selection/formal_60/cases.csv` 与 `case_selection/heldout_15/cases.csv`，验证分别正好 60/15 行且 Case ID 无重复；Smoke Case ID 必须来自对应 CSV。Basic Formal/Held-out 各一次 no-NPU Smoke，并只开放 `help`/`fork`；CANNBot Formal/Held-out 各一次 `smoke_*` Evaluate，不能将这两项误报成 fork-only 或 no-NPU。`evaluate` 阶段对 75 个数据驱动的 Case ID 各通过 `session.create_session(...)` 创建独立 Session，通过 `subprocess` 运行 `$PTOBENCH_ROOT/ptobench evaluate <registered-workspace>`，不把 Case ID 传给 CLI。`agent-action` 每次只调用一个由维护者固定的 action，经生产 Agent Bubblewrap、Pi Shell wrapper、CLI transport shim 与 Broker Socket；不读 API Key、不调用 host Broker API、不启动自由优化循环。`profile` 阶段只执行预先固定的 8 个类别代表样本。各阶段写 `results.jsonl`、`matrix.csv` 和 `summary.json`，每条记录保存 selection/Case ID、命令、Session/Worker 路径、CLI JSON、Event、Snapshot/诊断 artifact 引用、有效设备/host 证据、状态与输入 manifest hash。断点只能跳过证据完整且 hash 一致的 PASS；失败或状态不明的硬件 Action 不自动重试，须停止调查。预期 2 个 Basic Pi Smoke、2 个 CANNBot Smoke Evaluate、75 个正式 Evaluate 全部 PASS、4 个 Agent 通道 Action 成功、8 个类别各有 Combined 结果且三类产物齐全。通过有效运行配置证实的多卡样本至少一个；没有现成且可验证的跨-host 拓扑输入时，只记录未覆盖，整体验收不可报告多机通过。所有硬件 Action 串行；不更新性能 baseline/CV。验收开始与结束的逐条输入 manifest 必须一致。

## 完成标准与停止条件

所有 Smoke、正确性、Combined Profile、已声明的多卡覆盖与完整输入 hash 检查通过后，可签署 Basic/CANNBot 的单机部署验收并投入该范围使用；只有跨 host 样本也由现有运行配置明确证明并通过时，才能另行签署多机部署验收。若选择集合或运行配置无法提供跨 host 样本，只报告未覆盖，不将多卡替代多机，也不把多机描述为已验收。若实际部署目标包含多机，则多机验证是上线门槛，须先提供可验证的 host topology 与可运行 Case；不在本计划中新增 parser 或修改 Case。未来 PTOmni Pi 可复用本计划定义的合同，但其接入仍需独立验收。任意身份、ACL、私有文件、Socket、Gateway、Fork、Correctness、Snapshot、Profile 或 Case 输入资产检查失败，立即停队列并调查；不修改 Case/Fixture/Seed/Baseline/CV，不以 Candidate Worker 边界代替本轮验收。
