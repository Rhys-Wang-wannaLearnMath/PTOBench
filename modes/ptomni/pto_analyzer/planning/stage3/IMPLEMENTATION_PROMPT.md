# PTOAnalyzer Stage 3 实施 Prompt

请按 [Stage 3 任务清单](TASK_CHECKLIST.md) 实施 PTOAnalyzer 的工具收敛和高基数分页修缮。先完成 Task 0 的基线核验，再按任务顺序推进；每项都要满足自己的验收条件。这个 Prompt 只授权 PTOAnalyzer 范围内的代码、测试和文档修缮，不授权自行启动任何 NPU、PTOBench、Basic Pi 或 Agent 实验。

## 目标

1. 当前 16 份归档中的 `pipeline_overlap_check` 没有 overlap 观测。默认从 Dataflow 工具集合移除，不加空工具替代；只有确认存在同 Candidate/设备/窗口、公共时间基准和可关联 task/block 的 Compute/MTE 起止区间时，才实现真实 overlap 统计。
2. 将高基数明细从 `offset` 分页改为不透明 `cursor`/`next_cursor`。默认页 20 条、最大 50 条，内部明细候选缓存上限为当前 `limit`，与页序号无关。
3. 保持 `initial_summary` 的全量覆盖和统计、稳定排序、只读归档、角色隔离、来源引用与异常输入安全。
4. 运行 CPU-only 测试和离线归档回放，报告新增 Case/Agent 实验的门槛和可选方案；只有明确缺口且采集器能产生目标字段时，才提交最小 Profile 计划。

## 允许修改的范围

- `modes/ptomni/pto_analyzer/toolset.py`
- `modes/ptomni/pto_analyzer/diagnostics/`
- `modes/ptomni/pto_analyzer/tests/`
- `modes/ptomni/pto_analyzer/README.md`
- `modes/ptomni/pto_analyzer/planning/README.md`
- `modes/ptomni/pto_analyzer/reports/` 中的 Stage 3 验证记录

不改 PTOBench 原始 Profile、PTOBench 采集器、硬件锁、Runner、其它 Mode 或全局配置。若审计证明采集器缺少 overlap 字段，只写一份最小采集需求，不顺手扩展 PTOBench。

## 设计约束

- 外部 seam 继续是 `ToolSession.tool_specs`、`initial_summary` 和 `call(name, args)`；调用方不需要知道解析器或排序实现。
- Cursor 必须绑定 Profile event/Candidate、Tool 和 Tuner，且不能被当作任意路径、权限或资源选择参数。
- 使用同一套内部分页实现，不在九个诊断中复制 Cursor 编解码或验证逻辑。先实现 CPU-only 测试，再写实现。
- 一页的候选明细缓存不得超过 `limit`；所有页与初始摘要的全量 observations/source coverage 相同。记录数仍由完整数据扫描计算。
- 不加数据库、磁盘索引、第二个 Tool Server、全量明细缓存、诊断状态、根因推断、性能阈值或优化处方。
- 测试锁定结构化行为，不用脆弱的墙钟断言；对真实大 Profile 记录实际扫描时间和内存，若性能不足再单独评估内部 Session 缓存。
- 把 `pipeline_overlap_check` 从权限表和 `initial_summary` 撤下时，检查所有 PTOAnalyzer 文档中“9 个工具”“每位 Tuner 三个工具”的陈述并同步当前状态；不要把两工具的 Dataflow 摘要填充成占位项。

## 验证

从 `benchmarks/PTOBench/` 运行：

```bash
python -m unittest modes.ptomni.pto_analyzer.tests.test_toolset -v
```

此外按任务清单运行：

- 100,000 以上明细的 CPU-only 分页选择器压力测试，检查保留候选不超过 limit。
- Cursor 错误输入、跨 Profile/Tool、重复 sort key、末页/空页和全量无重无漏遍历测试。
- 16 份当前可用 combined Profile 的离线回放，以及单卡最大明细、双卡和 8 卡人工抽查。
- 回放前后核对受保护原始文件 SHA-256。纯离线工作记录新增硬件动作数为 0。

Task 6 的 Basic Pi 轨迹与 Task 7 的配对效用小试是可选后续任务，不在本次代码实施中自动启动。它们只有在清单所列条件成立且通过 Runner/PTOBench 正常路径、设备锁及预算约束时才执行。

## 交付

- Stage 3 代码、回归测试和更新后的 PTOAnalyzer 文档。
- Stage 3 离线回放记录：归档数、设备覆盖、工具清单、全量摘要差异、分页压力指标、归档哈希核对及硬件动作计数。
- 若需要新增采集字段，给出字段定义、身份/单位/时钟要求、目标 Case 理由与独立验收步骤；没有可采集的数据时明确停止。
- 若启动可选真实 Agent 试验，单独报告方法配对、预算、Case、Candidate/Profile/evaluate 关联与结果限制；不得把单次试验写成因果或论文结论。
