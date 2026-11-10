# 真实 Pi 与 Case 验收 Implementation Plan

**目标：** 通过 Basic/CANNBot 真实 Pi 以及 Formal/Held-out Case 证明专用 Agent 用户和真实 Workspace 可以投入使用。

**架构：** 先用不占 NPU 的真实 Pi Smoke 验证账号、挂载、工具、Socket、Gateway 和网络边界，再串行运行正确性与 Combined Profile 验收。PTOmni 不在本轮范围。

**验收边界：** 所有 Smoke 均创建新 Session 并走唯一的专用 Agent 用户路径；不通过旧 Session Workspace、旧身份映射或 legacy runner 验收。旧历史记录保留，不自动迁移或恢复旧运行目录。

**依赖：** Plan 01–04 通过；管理员完成一次性账号、共享组、sudoers 与 ACL 配置。

## 修改文件

- Create `evaluator/pi_sandbox_smoke.py` 及 `evaluator/tests/test_pi_sandbox_smoke.py`：复用已有 Basic/CANNBot preflight、prepare、`run_private_pi` 和 Broker；仅在临时 Session 中启动。
- Modify `evaluator/sandbox_supervisor.py::Broker.serve`、`run_private_pi` 与 Basic/CANNBot prepare 的测试接口：只接受 maintainer harness 的临时 Smoke 行为，不接受 Agent/YAML 开关，不改正式 Case 和 prompt。
- Create 验收记录于新的 `reports/agent_upgrade/dedicated_agent_user/`；不写入 `case_selection/` 或 Case 数据。

## 步骤

- [ ] Smoke 只接受 Basic/CANNBot 和 Formal/Held-out CSV 中存在的 Case ID；默认只允许 `fork/help`，不挂 NPU 设备。Pi 在临时 Session 创建 helper、Fork 一个 Worker、读取子 Worker 后退出；不得触发自动 Finalize 或 Case Evaluate。
- [ ] 分别对 Basic、CANNBot 运行一个 Formal 和一个 Held-out Smoke。核实运行时实际宿主 UID/GID、组、`CapEff`、`NoNewPrivs`；Agent 不属于硬件组、没有 sudo 权限；Broker 与 NPU Worker 仍具有原凭据。
- [ ] 检查 Pi 设置、session、锁文件可正常读写并可由 Broker 归档；Pi 原生文件工具和 Shell/Python 均不能读取私有设置/Session；Context/Outputs 不可写；Candidate/helper/tuning 可写。
- [ ] 检查动态 Fork 立即可见、父子 Candidate 独立、Broker 更新 Outputs 后 Agent 可读但不可改；确认 current Session 外的 Session/Case 和 PTOBench 内部源码不可见。
- [ ] 检查 `.env` 分属各 mode、DeepSeek Flash/High 配置生效，真实 Key 只在 Gateway；CLI、Gateway、CANNBot Socket 正常；Web Search/Web Fetch 不可用、Shell/Python 公网连接失败。
- [ ] Smoke 全部通过后，对 Formal 60 和 Held-out 15 执行正确性 Evaluate；结果只写新验收目录，不更新 Seed/Reference/Baseline/CV。
- [ ] 依据已选 Case 的 Granularity、设备数和单机/多机模式做 Combined Profile 覆盖。明确检查至少两个 Granularity、单卡、多卡、8 卡和真实跨 host 多机；缺少跨 host Case 时报告缺口，不把 8 卡称为多机。
- [ ] 硬件任务串行运行；只在温度/功耗正常且设备调度空闲时开始。不得仅凭 AICore 利用率决定设备是否空闲。
- [ ] 检查 Case/Fixture/Seed/Reference/Baseline/CV 与测试前记录的 hash 一致。若本次代码改动触及采集、Profile 状态复位、Runtime 编译/执行或测量路径，按影响扩大正确性与性能复验，不假设 Profile 不受影响。

## 验证

```bash
python -m unittest discover -s evaluator/tests -p 'test_pi_sandbox_smoke.py' -v
python evaluator/pi_sandbox_smoke.py --mode basic --case-id <formal-or-heldout-id> --operation fork
python evaluator/pi_sandbox_smoke.py --mode cannbot --case-id <formal-or-heldout-id> --operation fork
ptobench evaluate <selected-case-or-session>
ptobench profile <selected-case-or-session>
```

执行时使用 CLI 实际接受的单个 `cannbot` mode 名；Formal/Held-out 的 evaluate/profile 命令按既有 PTOBench CLI 选择接口生成，不直接改 Case。预期每个 Smoke 成功，75 个 Evaluate 全部 PASS，Combined Profile 与覆盖矩阵结果记录清楚；硬件不覆盖的组合单列，不标 PASS。

## 完成标准与停止条件

所有 Smoke、正确性、Combined Profile、跨设备覆盖与资产 hash 检查通过后，才宣布可投入 Basic/CANNBot 使用。任意身份、ACL、私有文件、Socket、Gateway、Fork、Correctness、Snapshot、Profile 或 Case 输入资产检查失败，立即停队列并修复；不修改 Case/Fixture/Seed/Baseline/CV，不以 PTOmni 或 Candidate Worker 边界代替本轮验收。
