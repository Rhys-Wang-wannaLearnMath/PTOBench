# 真实 Workspace 与 DAC Implementation Plan

**目标：** 让每个新 Session 的 Workspace 直接使用真实目录，并由 Broker 和共享组权限实现 Candidate 可写、Context/Outputs 只读。

**架构：** Broker 创建并管理真实目录 `workspaces/worker_NNN/{candidate,.benchmark_context,.benchmark_outputs}`。同一 Session 的 Agent 可读写每个 Worker 的 Candidate、读取每个 Worker 的 Context/Outputs；Fork 复制父 Candidate 后仍是独立目录，不自动同步。Candidate 由 Broker 所有并通过共享组/setgid 允许双方读写；Context/Outputs 由 Broker 所有且 Agent 只读。Outputs 在真实 Workspace 中由 Broker 更新；不保留 private/public 副本、链接或同步目录。Snapshot 拒绝 Broker 不可读文件，执行仍使用 Action freeze 的独立拷贝。

**历史数据处理：** 新 Session 只使用真实目录 Workspace。旧 Session registry、导出 Workspace 和运行状态不作迁移、解析兼容或恢复；历史文件可原样留存，但不能启动 Agent Action。仍需优化的任务用新 Session 重启。

**依赖：** Plan 01 已部署专用账号和共享组；Broker 进程已在重新登录后获得共享组权限。

**Tech Stack：** Python 标准库、Linux POSIX DAC、文件锁与 pytest。

**Spec：** [`00-overview.md`](00-overview.md)，尤其是全局不变量与最终验收条件。

## 全局约束

- 唯一实现是专用 Agent 用户、真实 Workspace、共享组 DAC 与单次 Bubblewrap Workspace bind；不保留旧导出视图或 Session 迁移。
- Agent 只能改 `candidate/`；Case、Fixture、Seed、Baseline、CV 与选择数据不变。
- 不可读 Candidate 文件必须在 Snapshot 前明确失败；不能跳过文件或生成部分 Snapshot。

## Review Focus

- Worker 根目录只能让共享组读/进入，不能给组写权限；Candidate 目录单独允许组写。
- Fork 任一崩溃点都不能向 Agent 开放半成品或注册不完整 Worker。
- 旧 Session 拒绝必须发生在自动修复之前，且拒绝过程不修改旧 Session。

## 修改文件

- Modify `evaluator/workspace.py::_readonly`、`_ensure_tuning`、`write_context`、`init_workspace`、`fork_workspace`：创建真实目录并规范化 owner/group/mode。
- Modify `evaluator/session.py::_validate_record`、`_load_locked`、`read_session`、`create_session`、`register_workspace`、`fork_workspace`、`fork_workspace_locked`、`resolve_workspace`、`repair_assets`，并新增 `_reject_retired_workspace_protocol` 与 `recover_pending_forks`：保持 `runs_root` 现有配置合同；在启动 Agent 前验证该路径链已允许 Agent 遍历，并在权限缺失时给出需管理员预配置该根目录的明确错误；Broker-owned 的 `runs_root/optimizer/case/session_root` 目录设共享组可遍历、不可列举的权限，以便 Agent 到达挂载的 `workspaces/`。`session_root/agent/launcher` 的 Request 目录权限由 Plan 01 负责。新 Session 使用 schema v2 且要求 `pending_forks` 映射；在任何完整性自动修复前，根据只读保护备份和 manifest 标记识别旧 schema/export Session，返回 `invalid_request` 且不恢复或改写旧记录。旧 Session 不迁移。ready `workspaces` registry 仍只包含可公开 Worker。
- Modify `evaluator/file_tree.py::atomic_bytes` 调用点，不改变安全默认 `0600`；Agent 可见文件显式指定共享组读写模式。
- Modify `evaluator/session_actions.py::_publish_diagnostics` 与对应文件写入调用：只发布到真实 Worker 的 `.benchmark_outputs/`，新建及原子替换文件显式使用 Broker 所有、共享组只读的权限。
- Modify `evaluator/snapshots.py::create_snapshot` 及其 Workspace 文件遍历调用：不可读 Candidate 文件以明确错误拒绝。
- Replace old-link/candidate-pool assertions in `evaluator/tests/test_workspace.py`、`test_session.py`、`test_snapshots.py`、`test_agent_cli.py` with real-directory/DAC tests; add `evaluator/tests/test_workspace_dac.py` for actual Agent/Broker identity permissions, live publication, and Action freeze.

## 步骤

- [ ] 写测试：多个同 Session Worker 的三个入口均 `lstat` 为普通目录；Agent 账号能列出/进入全部 Worker，能读取和修改每个 `candidate/`、新建 helper、修改 `tuning.json`、原子替换和删除文件。用实际 Pi 文件工具及 Shell/Python 覆盖新建与原子替换，验证正常 Candidate 文件及 `tuning.json` 替换后 Broker 仍可读取并能完成 Snapshot；另单独创建 `0600` 文件，确认 Snapshot 明确拒绝并指出相对路径。若实际 Pi 写入路径生成 Broker 不可读的普通 Candidate 文件，先修正该写入路径的最小权限行为，再继续。Agent 对 Workspace 根、Worker 根、Context/Outputs 目录及其入口尝试写入、chmod、删除、重命名、替换和 symlink substitution 全部失败。验证 Fork 起始 Candidate 是父 Candidate 的副本，随后修改一个 Worker 的 Candidate 不改变其他 Worker；两个 Worker 的 Outputs 可同时读取/比较，Broker 更新一个不会覆盖另一个。
- [ ] 新增 Workspace 权限规范化函数，按 `getgrnam('ptobench-share')` 解析当前共享 GID，不使用数值常量。Workspace 根和 Worker 根为 Broker:`ptobench-share` 的 `0750`/`2750`（Agent 仅可读、进入，不可创建、删除或替换其下入口）；Context/Outputs 目录为 `0750`；Candidate 目录为 `2770`；Broker 创建的 Candidate 文件和 tuning 文件为 `0660`；Context/Outputs 文件为 `0640`。
- [ ] 新建 Session 时，将 Broker-owned 的 `runs_root/optimizer/case/session_root` 路径链设置为 `Broker:ptobench-share`、目录 mode `2710`（Broker 可管理，Agent 组仅能遍历且不能列举）；`session_root/agent/launcher` 及 Request 文件权限由 Plan 01 管理。保留现有 `runs_root` 配置合同，但在 Agent Session 创建前验证 Agent 可逐级遍历路径；当前部署会使用的所有 roots 由用户在开工前盘点并配置。未预先授权的自定义 root 立即以明确错误失败并提示管理员配置，不在任务运行中等待、不扩大为可列举权限。`2710` 的 Session 父目录只阻止列举，不构成对已知 Session 路径的访问隔离；真正的跨 Session 隔离由 Plan 03 的 Bubblewrap 只挂当前 Session `workspaces/` 实现。
- [ ] 由 Plan 01 唯一负责在 `evaluator/agent_dispatch.py::main` exec Bubblewrap/Pi 前设置 `umask(0o007)`。本 Plan 只验证其 DAC 效果：Pi/Shell 创建的新 Candidate 文件由 Broker 可读、子目录继承 setgid；Agent 创建 `0600` 文件或不可遍历目录时，Snapshot 报具体相对路径并拒绝，不跳过文件、不生成部分清单。不要在本 Plan 重复修改 dispatcher。
- [ ] 初始 Worker 与每个 Fork 都创建真实的 `candidate/`、`.benchmark_context/` 和 `.benchmark_outputs/`；`.benchmark_outputs/` 在注册前存在。Broker 发布或更新 Outputs 时，目录统一为 Broker 所有、共享组只读/可遍历 `0750`，普通文件为 `0640`；原子写入显式设置 mode，不能沿用 `atomic_bytes` 的 `0600` 默认或从归档恢复可写权限。Agent 可读取最新反馈但不能改写反馈内容或入口。
- [ ] 创建 Worker 时先在不挂载给 Agent 的 `session_root/staging/forks/`（Broker-only、`0700`）生成 Candidate、Context、Outputs；验证 `st_dev` 确认 staging 与 `workspaces/` 同一文件系统。三个子目录与文件先设置最终权限，Worker 根保持 `0700`；原子 rename 到 `workspaces/worker_NNN/` 后仍不可进入。随后记录 Context integrity hash、注册 Workspace，再以一次目录 `fchmod` 将 Worker 根从 `0700` 改为 `2750` 作为唯一发布点；共享组只有读/进入权限，没有目录写权限，因此 Agent 能访问 `candidate/` 内容但不能替换、删除或重命名 Candidate、Context、Outputs 入口。这样崩溃只会留下完整关闭或完整开放的树，不会留下逐文件 chmod 的半开放状态。修改 `fork_workspace_locked()` 当前把临时目录放在 `target.parent`（即会被挂载的 `workspaces/`）的问题。
- [ ] Fork 状态按可恢复的 reservation → staged → renamed-locked → registered-locked → published 顺序持久化：新 schema v2 的 Session record 中，`pending_forks` 必须存在（初始为空），每条保存已消耗的 Worker ID、父 Worker ID 和 staging 相对路径；`_validate_record` 校验该 schema，ready `workspaces` registry 仍使用 `{path,parent}` 项。先提交 ID reservation/pending，再构建并校验 staging；将 Worker 原子 rename 到最终路径，保持根目录 `0700`；在 Broker Session lock 内注册 Context integrity root 与 ready Workspace registry，打开前确认该根仍为 `0700` 且全部子树权限符合最终合同；以单次根目录 `fchmod(2750)` 发布，最后提交移除 pending。Broker 在 lock/请求处理完成前不把中间态返回给 Agent；Agent 即使看到 Worker 名称，也不能进入未发布目录。恢复由 `session.recover_pending_forks()` 在 Session lock 内执行：只有 reservation 且无目录时移除 pending；不完整 staging 删除暂存并移除 pending；完整 staging 继续 rename；renamed-locked 继续注册；Context root 已注册但 Workspace 尚未注册时，验证其路径、hash 并继续完成注册；registered-locked 执行完整子树校验后单次打开根目录；registered-open 只验证权限/hash 并清除 pending。Worker ID 始终消耗。未知、重复、路径越界、hash/父项不一致、同时存在 staging 与 target、缺失两者或部分开放的子树都停止并报告，绝不猜测或打开权限。覆盖 reservation、partial/complete staging、renamed-locked、context-registered、registered-locked、registered-open 各崩溃边界。Plan 03 在读取 Session 或启动 Pi 前先调用恢复函数。已有 Pi/Shell 的单一 `workspaces/` bind mount 在根权限打开后立即反映 ready Worker。
- [ ] 加 Session 格式前置测试：新建记录为 schema v2 且初始 `pending_forks == {}`；从完整性保护备份读出格式时验证备份 hash。schema v1、`public_bench`/旧 Candidate pool/export 记录必须在 `check_and_restore` 前返回 `invalid_request`，且 Session 全树 hash 不变；当前 v2 的可恢复损坏记录仍通过原完整性修复。
- [ ] 加真实身份集成测试 `evaluator/tests/test_workspace_dac.py`：经固定 dispatcher 和生产 Bubblewrap 运行权限探针，不调用 `sudo -u ... sh/python`，也不扩大 sudoers 授权。确认 Agent 能修改当前 Worker Candidate，不能修改 Context/Outputs 或替换其入口；确认 Broker 能读 Agent 普通新文件，`0600` 文件以相对路径明确指出并在 Snapshot 前失败，不能生成部分 Snapshot。DAC 测试只验证本地文件权限；不把“已知路径下的其他 Session 不可读”作为 DAC 断言，因为各 Session 共用 `ptobench-share`，跨 Session 隔离由 Plan 03 的 Bubblewrap mount 边界验收。
- [ ] 在 Agent UID + DAC 条件下复验 Action Freeze：提交 Action 后同时修改 live Candidate，执行输入及最终 Snapshot 仍绑定提交时冻结副本；覆盖稳定扫描重试、并发修改失败路径和 0600 拒绝路径。
- [ ] 反馈仅写真实 Workspace 各 Worker 的 `.benchmark_outputs/`；正式记录仍是 Session/Action/Event/Snapshot。新 README 只说明 `candidate/`、只读 Context/Outputs 和 evaluate/profile/fork，不含 `mode.json`、`profiling_modes.json`、`fixture_catalog.json` 或“Do not inspect hidden test/reference assets”。

## 验证

```bash
PYTHONPATH=evaluator "$PTOBENCH_LOCKED_PYTHON" -m pytest -q evaluator/tests/test_workspace.py evaluator/tests/test_session.py evaluator/tests/test_snapshots.py evaluator/tests/test_agent_cli.py
PYTHONPATH=evaluator "$PTOBENCH_LOCKED_PYTHON" -m pytest -q evaluator/tests/test_workspace_dac.py
```

运行前将 `PTOBENCH_LOCKED_PYTHON` 设为该主机 `environment_lock.locked_python_executable()` 返回的绝对路径。预期：本地 Workspace/Session/Snapshot/CLI 单元测试通过；需要 Agent/Broker 两个宿主身份的 DAC 集成测试在已预配主机通过。Fork 故障注入不会使未 ready Worker 被 CLI 接受；运行中的 Bubblewrap 进程能立即看到 ready Worker；`0600` Candidate 文件不会造成静默不完整 Snapshot；Action freeze 与 Snapshot 不受 freeze 后对 live Candidate 的修改影响。

## 完成标准与停止条件

新建 Workspace 与 Fork 全部符合真实目录和 DAC 合同、旧 Session 不会被运行时恢复、真实 Workspace Outputs 只读反馈通过、Action freeze 保持稳定后进入 Plan 03。旧 Session 的 Agent Action 请求必须明确失败并要求创建新 Session；不能自动恢复、迁移、覆盖旧 registry 或落回导出视图。任何入口可被 Agent 替换、Context/Outputs 可写、Broker 读不到正常 Candidate、恢复会暴露半成品或权限测试只能依赖放宽目录权限时停止；不保留旧格式兼容或导出 Workspace 兜底。
