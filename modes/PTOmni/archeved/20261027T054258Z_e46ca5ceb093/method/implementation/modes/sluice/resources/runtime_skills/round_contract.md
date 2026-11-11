# SLUICE 本轮合同

- 先读 `sluice_progress`，再按需搜索和完整读取发布知识；按当前 Decision 行动。
- 修改 Candidate 或获准的 tuning 前先登记一个 Hypothesis；修改后必须 `sluice_evaluate`。
- profile 只回答登记问题；unavailable 不等于零。每个问题独立保留其 EvidenceBundle。
- `diagnose` Decision 必须引用机会和 catalog 注册问题；无 active Hypothesis 时用
  `sluice_profile_baseline` 与对应 detail，随后读 progress 并再次提交 Decision。`sluice_profile`
  仅用于 evaluate 后已登记的问题。
- Stage 01 smoke 只把有有效、integrity-passed correctness PASS evaluate 结果的 cycle 计入两个
  optimize cycle；编译失败或 `action_failed` 不计数。完成首个 raw 文件读取后，必须在后续 profile
  action 后以完全相同的 `profile_ref` 和 `file_ref` 再读并比对完整内容。剩余预算仍允许时，失败后按
  诊断修正并继续；门槛未达成就不能报告通过。全部门槛达成后调用 `sluice_finish`，不必耗尽预算。
- 遵守 workspace ABI、tuning 白名单、设备分配和预算。硬件工作只能经 SLUICE 公共工具。
- Tensor/Tile 层次不同；slice 描述 Tensor 区域，load/store 搬运 Tensor 与 Tile。保持 valid tails、
  输出写入和 reduction identity 正确。
- 改 tile 尺寸前查阅相应 alignment/内存知识并核算实际布局、dtype 下的连续字节宽度；违反编译器
  对齐条件的修改需要按错误诊断调整，不计入完成 cycle。
- `pl.range` 顺序，`pl.parallel` 仅独立迭代，`pl.unroll` 常量展开；核对当前安装 API。
- 依赖、scope 和 pipeline stage 会影响正确性及资源生命周期。不要擅自删除依赖或使用未授权开关。
- evaluate 决定 correctness 与 clean latency；profile 是诊断。缺失不是零，相关性不是因果。
- Controller 管理状态、预算、失败、回退和 best；重新读取 ProgressView/Decision 后再继续。
