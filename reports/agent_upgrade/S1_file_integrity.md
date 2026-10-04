# S1：文件保护、恢复和源码复制

阶段：S1。日期：2026-10-03（Asia/Shanghai）。结论：**PASS，满足进入 S2 的条件。**

依据：已核对 [S0 报告](S0_baseline.md)，读取 `01_interfaces_and_layout.md`、`02_file_integrity.md`、`08_execution_steps.md`。用户指定的 `Benchmark升级/plan/10_stage_prompts.md` 当前不存在；搜索找到了实际文件 `docs/10_stage_prompts.md`，已读取其共同约束及 S1 prompt，并修正 `08_execution_steps.md` 指向它的失效链接。没有改变阶段范围或执行顺序。

本次仅实现 S1 公共工具、测试与说明，没有实现 Session/预算生命周期、Agent CLI、Launcher、Optimizer 或硬件执行。S0 已记录的 Case 资产缺项继续保留。

## 修改内容与接口

| 文件 | 实际行为 |
| --- | --- |
| [workspace.py](../../evaluator/workspace.py) | 抽出 `copy_candidate_tree`、`source_manifest`、`write_context`；init/fork 复用公共入口；多文件及 helper 递归复制，空目录保留，稳定副本发布，context 仍为 0444 |
| [file_tree.py](../../evaluator/file_tree.py) | 共享的目录描述符操作、无链接扫描、固定文件描述符 SHA-256、独立流式复制、无链接删除、同目录原子替换；复用 `common.sha256_file` |
| [integrity.py](../../evaluator/integrity.py) | 清单、独立备份、分组检查/恢复、原子记录提交及新根登记；不重新建立已存在保护基线 |
| [test_integrity.py](../../evaluator/tests/test_integrity.py) | 16 项临时资产行为测试，含真实子进程提交中断和多进程并发 |
| [test_workspace.py](../../evaluator/tests/test_workspace.py) | 扩充至 14 项，覆盖多文件、排除项、稳定 hash、并发修改、类型/链接拒绝、旧 init/fork 及 context 恢复 |
| [Evaluator README](../../evaluator/README.md) | 说明公共 API、清单/备份格式、错误码、hash 序列化、恢复和提交语义，以及后续 Session 调用约定 |

实现的固定 API：

```python
capture_protection(protection_root, roots) -> dict
check_only(protection_root, groups=None) -> IntegrityReport
check_and_restore(protection_root, groups=None) -> IntegrityReport
commit_records(protection_root, replacements) -> None
```

`roots` 使用 `{logical_name: ProtectionRoot(path, group, readonly=False)}`。清单保存冻结的绝对路径、根节点类型、全部普通文件/目录和空目录、mode、文件 SHA-256 及 `backups/<sha256>` 位置。不存在的最终记录按 expected absence 登记；保护清单只排除 `__pycache__`、`.pyc/.pyo`，不排除 `.pt` 或 Ref 输出。

`register_roots` 只用于新增 context/Snapshot 等保护根：拒绝覆盖已有逻辑根、路径重叠及 dirty 的已有保护内容。它不生成 Session 或候选，也不替换已有基线。

## 恢复、路径和记录语义

- `check_only` 保留原始变化证据、不修复工作文件；`check_and_restore` 返回相同 `changed_paths` 和已恢复的 `restored_paths`。恢复成功不抹去原始失败证据，后续 `check_only` 才能证明恢复后的状态一致。
- 恢复前验证所有本次所需备份。备份缺失、hash 损坏或变成链接时明确抛出 `InfrastructureError`，附带 unrecoverable 路径，停止恢复；不扫描当前资产重建基线。
- 根、内层目录或普通文件变成链接时只删除链接本身并重建，所有目录组件通过 `O_NOFOLLOW` 和目录描述符访问；外部哨兵内容保持不变。冻结根之外的祖先发生链接/类型破坏时明确失败，不向不受保护的祖先写入。
- 文件恢复采用同目录临时文件、hash 核对、`fsync` 和 `os.replace`。多余文件/目录/链接会清理，空目录、文件/目录类型与权限会恢复。
- context 从原始保护副本恢复，starting_point 保留创建时版本；当前 Candidate/helper 的编辑保持不变。测试还改变临时 canonical config，证明 context 恢复仍使用旧副本，没有重新生成 context。
- `commit_records` 仅接受注册的 `records` 组和确切 bytes。当前记录 dirty 时拒绝提交，要求先恢复并重新构造合法更新；固定资产及未登记路径不能通过此 API 更新。
- 新记录的全部独立备份先保存并核对；**一次原子 manifest 更新**后再替换工作文件。manifest 是恢复依据，多文件工作副本中断后可从新备份恢复一致的新预算/事件，不回退到旧 generation。
- 模块通过 `.integrity.lock` 序列化 manifest 访问。S2 必须在 Session 锁内执行修复、读取、计算与提交，并在全局执行锁内恢复共享固定资产；S1 的并发记录测试不等价于已实现预算上限或 Session 状态机。

## Candidate 复制与源码 hash

`copy_candidate_tree` 先扫描源、复制到目标的临时同级目录、再次扫描源、核对目标，最后 rename 发布。普通编辑允许初次尝试加最多三次重试；持续变化返回 `InvalidWorkspaceError(error_code="invalid_workspace")`，不发布半成品、不覆盖非空目标。源码或目标的根、祖先和纳入复制的节点拒绝链接及特殊文件；扫描后被替换为链接也被拒绝。

统一递归排除 `.benchmark_context`、`.benchmark_outputs`、`.git`、`.pytest_cache`、`__pycache__` 和 `.pyc/.pyo`；被排除的目录不遍历，但排除入口本身是链接或特殊节点时仍拒绝。其余普通文件、二进制资源和目录全部保留。

源码 hash 来自按相对路径排序的 `{path, sha256}` 列表，使用 UTF-8、`ensure_ascii=False`、`sort_keys=True`、`separators=(',', ':')` 的固定 JSON。文件路径与内容同时参与 hash；目录记录单独保留用于复制校验，mode、输出和缓存不进入内容 hash。非 UTF-8 文件名在发布前拒绝。后续 staging/Snapshot 可以直接复用相同工具与规则。

## 实际测试与证据

所有命令从 `/data1/home/ziruiwang/pypto-lib` 执行，正式解释器为 `.venv-official/bin/python`；日志保存使用 `set -o pipefail`，未掩盖失败退出码。

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_integrity.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S1_evidence/integrity_final.log

PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_workspace.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S1_evidence/workspace_final.log

PYTHONDONTWRITEBYTECODE=1 .venv-official/bin/python -m unittest discover \
  -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p 'test_*.py' -v \
  2>&1 | tee benchmarks/pypto_lib_bench-v0.2/reports/agent_upgrade/S1_evidence/unittest_final.log
```

| 原始日志 | 退出码 | 实际结果 |
| --- | ---: | --- |
| [unittest_before.log](S1_evidence/unittest_before.log) | 0 | S1 修改前 14 tests / 2.061s，OK；S0 前置基线重复验证 |
| [integrity_first.log](S1_evidence/integrity_first.log) | 1 | 首轮 15 tests，1 个失败：测试假设默认权限为 0644，实际 umask 下原始权限为 0664；已改为核对捕获时的原始 mode，保留精确权限恢复标准 |
| [workspace_first.log](S1_evidence/workspace_first.log) | 0 | 首轮 12 tests，OK |
| [unittest_first_full.log](S1_evidence/unittest_first_full.log) | 0 | 修正上述测试构造后 39 tests，OK |
| [integrity_final.log](S1_evidence/integrity_final.log) | **0** | 最终 16 tests / 0.248s，OK |
| [workspace_final.log](S1_evidence/workspace_final.log) | **0** | 最终 14 tests / 0.097s，OK |
| [unittest_final.log](S1_evidence/unittest_final.log) | **0** | 最终完整 CPU 回归 **41 tests / 2.458s，OK** |

没有 skip/expectedFailure。新增测试数据、哨兵、FIFO、线程编辑区和子进程记录均在 TemporaryDirectory 内；不调用 NPU、模型 API、prepare-ref 或正式 Fixture builder。完整回归保留原 S0 comparator 覆盖：192 个清单项、156 ready、183 个现存 evaluator；9 个缺 evaluator 的 pending 项继续明确 NOT_RUN。

六个真实进程中断边界的实际结果保存于最终 integrity/完整回归日志，子进程均使用 `os._exit(23)`，不是只 mock 返回值：

| 中断边界 | 恢复前 (budget, event) | 恢复后 |
| --- | --- | --- |
| 保存新备份前 | (0, 0) | (0, 0) |
| 保存首个新备份后 | (0, 0) | (0, 0) |
| manifest 更新前 | (0, 0) | (0, 0) |
| manifest 更新后、替换工作文件前 | (0, 0) | **(1, 1)** |
| 首个工作记录替换后 | **(1, 0)** | **(1, 1)** |
| 最后一个工作记录替换后 | (1, 1) | (1, 1) |

双进程提交由同步 Event 同时放行，结果为某一完整 transaction，而非混合预算/事件。Candidate 并发编辑使用同步 Event 和真实线程；稳定后返回一致副本，持续修改四次则失败并清理 staging，未依赖 sleep 推测时序。

## 保护与改动保存

Git revision：`9c5dfec2dfa18f056098000123e10de06de32426`；实际对象为保留已有本地修改的当前工作区。操作前后状态、index、修改前文件、实现差异、源文件 SHA 和日志 SHA 保存于 [S1_evidence](S1_evidence/validation_summary.json)。

[preservation.json](S1_evidence/preservation.json) 实测结果：

- 297 个操作前已存在的 tracked 文件 SHA-256 未变化，Git index 未变化；未 reset、clean、暂存或提交。
- canonical Case/sealed/environment 共 6701 个非缓存文件，无新增、删除或变化。
- 3709 个非 `.pt` 文件内容 SHA-256 一致，涵盖 Case/config/comparator/Seed/Ref/Oracle/Case evaluator/认证与环境配置。
- 2992 个 `.pt` 约 216.98 GiB 的 size、mtime_ns、mode 一致；未重新读取这些 tensor 的全量字节 hash。所有本阶段写操作仅涉及公共代码、测试、文档、证据和临时工作区。

没有改变真实 Case 资产、golden、comparator 或测量语义；没有新增 Docker、Supervisor、socket 服务、独立项目或包发布流程。

## 进入 S2 的条件

S1 的公共复制/context、清单/备份、路径和目录恢复、原始 starting_point、损坏恢复源失败、原子记录提交及稳定源码复制均有实现和行为测试；最终阶段测试与完整 CPU 回归通过。**S1 无未解决公共实现阻塞，可以进入 S2。**

S2 仍需实现实际 Session 创建、资产 preflight、registry、状态、全局/Session 锁、预算预占及 active_action，并按上述调用约定接入 integrity。S0 的 6 个 ready Case 缺 Seed 和 36 个 pending Case 的资产缺项继续保持 BLOCKED，未修补真实资产，也未将它们算作 Session/硬件验收通过。本次没有执行 S2 或之后阶段。
