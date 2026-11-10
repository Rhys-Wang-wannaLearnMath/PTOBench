# 真实 Workspace 与 DAC Implementation Plan

**目标：** 让每个新 Session 的 Workspace 直接使用真实目录，并由 Broker 和共享组权限实现 Candidate 可写、Context/Outputs 只读。

**架构：** Broker 创建并管理真实目录 `workspaces/worker_NNN/{candidate,.benchmark_context,.benchmark_outputs}`。Agent 在同一 Session 内可读写所有 Worker 的 `candidate/`，只读所有 Worker 的 Context/Outputs；Fork 时复制父 Candidate，之后仍为互相独立的目录。Candidate 使用 Broker 所有、共享组可读写和 setgid；其余入口由 Broker 所有且 Agent 组只读、可进入。Snapshot 明确拒绝 Broker 不可读文件。

**历史数据处理：** 新 Session 只使用真实目录 Workspace。旧 Session registry、导出 Workspace 和运行状态不作迁移、解析兼容或恢复；历史文件可原样留存，但不能启动 Agent Action。仍需优化的任务用新 Session 重启。

**依赖：** Plan 01 已部署专用账号和共享组；Broker 进程已在重新登录后获得共享组权限。

## 修改文件

- Modify `evaluator/workspace.py::_readonly`、`_ensure_tuning`、`write_context`、`init_workspace`、`fork_workspace`：创建真实目录并规范化 owner/group/mode。
- Modify `evaluator/session.py::create_session`、`register_workspace`、`fork_workspace`、`fork_workspace_locked`、`resolve_workspace`、`repair_assets`：保证仅新格式且 ready 的 Worker 可用；拒绝恢复旧 Session registry / 导出视图，不实现旧格式迁移或执行兼容。
- Modify `evaluator/file_tree.py::atomic_bytes` 调用点，不改变安全默认 `0600`；Agent 可见文件显式指定共享组读写模式。
- Modify `evaluator/snapshots.py::create_snapshot` 及其 Workspace 文件遍历调用：不可读 Candidate 文件以明确错误拒绝。
- Test `evaluator/tests/test_workspace.py`、`test_session.py`、`test_snapshots.py`、`test_agent_cli.py`。

## 步骤

- [ ] 写测试：多个同 Session Worker 的三个入口均 `lstat` 为普通目录；Agent 账号能列出/进入全部 Worker，能读取和修改每个 `candidate/`、新建 helper、修改 `tuning.json`、原子替换和删除文件；不能写 Workspace/Worker/Context/Outputs。验证 Fork 起始 Candidate 是父 Candidate 的副本，随后修改一个 Worker 的 Candidate 不改变其他 Worker。
- [ ] 新增 Workspace 权限规范化函数，按 `getgrnam('ptobench-share')` 解析当前共享 GID，不使用数值常量。Workspace 与只读目录为 Broker:`ptobench-share` 的 `0750`；Candidate 为 `2770`；Broker 创建的 Candidate 文件和 tuning 文件为 `0660`；Context/Outputs 文件为 `0640`。
- [ ] 设定新 Agent 文件 umask 为 `0007`；确认新文件继承共享组，子目录继承 setgid。Agent 创建 `0600` 文件或不可遍历目录时，Snapshot 报具体相对路径并拒绝，不跳过文件、不生成部分清单。
- [ ] 创建 Worker 时先在 Agent 不可遍历的 Broker staging 下生成 Candidate、Context、Outputs，执行权限和完整性检查后再发布。Context integrity hash 在 Agent 可读前记录。
- [ ] Fork 采用 pending → ready：Worker ID 先持久化消耗，staging 完整，原子 rename 后提交 ready registry，再设置共享组可访问权限。Agent 只可 fork ready Worker；staging 不在 Agent bind mount 下。启动恢复覆盖每个持久化边界的中断，不能重复 ID 或暴露半成品。
- [ ] 加真实身份集成测试：在管理员部署后用 `sudo -u ptobench-agent` 直接检查共享组 DAC，而不是仅检查 `stat` 输出。确认 Broker 能读 Agent 普通新文件；0600 文件导致 Snapshot 明确失败。
- [ ] 保持反馈目录是只读副本、正式记录仍是 Session/Action/Event/Snapshot；新 README 只说明 `candidate/`、只读 Context/Outputs 和 evaluate/profile/fork，不含 `mode.json`、`profiling_modes.json`、`fixture_catalog.json` 或“Do not inspect hidden test/reference assets”。

## 验证

```bash
python -m unittest discover -s evaluator/tests -p 'test_workspace.py' -v
python -m unittest discover -s evaluator/tests -p 'test_session.py' -v
python -m unittest discover -s evaluator/tests -p 'test_snapshots.py' -v
python -m unittest discover -s evaluator/tests -p 'test_agent_cli.py' -v
```

预期：本地权限单元测试通过；需要 Agent/Broker 两个宿主身份的集成测试在安装后的 Linux 主机通过。Fork 故障注入不会使未 ready Worker 被 CLI 接受；0600 Candidate 文件不会造成静默不完整 Snapshot。

## 完成标准与停止条件

新建 Workspace 与 Fork 全部符合真实目录和 DAC 合同、旧 Session 不会被运行时恢复、反馈只读后进入 Plan 03。任何入口是链接、Context/Outputs 可写、Broker 读不到正常 Candidate、恢复会暴露半成品或权限测试只能依赖放宽目录权限时停止；不保留旧格式兼容或导出 Workspace 兜底。
