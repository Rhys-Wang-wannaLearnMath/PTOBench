# PTOAnalyzer Stage 4 实施 Prompt

请按本目录的 [Stage 4 任务清单](TASK_CHECKLIST.md) 实施。本轮目标是降低 PTOAnalyzer 在大 Profile 上的解析/分页成本、限制坏记录的诊断内存、同步 Stage 3 的真实完成状态，并为 Dataflow Tuner 增加只读的 `tile_memory_allocation_summary`。

## 开始前

1. 阅读本目录 `README.md`、`TASK_CHECKLIST.md`，PTOAnalyzer `README.md`、Stage 3 清单和回放报告。
2. 从 `benchmarks/PTOBench/` 运行 `python -m unittest modes.ptomni.pto_analyzer.tests.test_toolset -v`，确认当前基线；当前已记录 43 项通过。
3. 检查 `modes/ptomni/pto_analyzer/` 当前工作区改动。工作树可能有用户改动，不得覆盖、清理或回滚无关变更。
4. 只做计划中的 Stage 4 工作；本 Prompt 不授权任何 NPU、PTOBench `profile`/`evaluate`、Basic Pi 或 Agent 实验。

## 核心要求

- 保留现有八个工具、角色隔离、完整摘要、完整可分页明细、稳定排序、Profile/Candidate 绑定和只读归档；只新增一个 Dataflow 专属 `tile_memory_allocation_summary`，完成后共九个工具。`pipeline_overlap_check` 继续停用，不修改其他八个工具的职责或权限。
- 错误明细示例应有上限，并显式返回精确错误总数和未展示条数；不能把坏数据当作零或静默丢失。
- PMU CSV 一次遍历且不保留全部行；`events.jsonl` 逐行读取；共享来源查找避免重复线性扫描。
- 依赖图的多个 Cursor 页不得重复解析并构建完整 task/edge 图。先量化峰值来源，再使用最小的标准库方案；不得用删掉 graph observation、Top-K、截断或改变排序的方式降成本。
- 新工具先审计现有 16 份 Profile 中的 `memory_after_AllocateMemoryAddr` / `*_after_AllocateMemoryAddr.py` 原始格式，再按产物、函数（若来源提供）和 memory space 汇总编译器明确记录的 `allocation_count` 与 `requested_bytes_sum`，保留来源引用；无函数边界时只做产物级汇总。返回完整分组摘要，不增加逐记录详情或分页。`buffer_pressure_check` 仍独立整理 MemoryReuse 提示和 Scope 原始统计。
- 分配字节总和仅表示对应分组内记录的静态分配请求大小；不得跨函数合并为峰值，也不得推断运行时占用、容量压力、生命周期重叠、spill、瓶颈或优化处方。缺少或不支持的输入按来源状态表达，不猜测补值。
- 如果现有归档无法提供经核实的 AllocateMemoryAddr 样本，记录阻塞并停止新工具实现；本 Prompt 不授权新增 Profile、Collector 改动或其他硬件动作。
- 分页测试从用户可观察的调用结果和资源压力验证上界，不读 `_heap` 等私有字段，也不固定 Cursor 的 Base64 表示形式。
- 只有当仓库已经有正式 PTOmni Host 时才做薄接线；只修改它现有的原生 Tool 入口。找不到 Host 时记录依赖，不创造新的工具服务器或实验 Harness。
- 不实现 `pipeline_overlap_check`。当前没有同 Candidate/Profile、rank/device、dispatch、task/block 和采集窗口匹配的 Compute/MTE 区间；不从聚合 PMU 计数估算 overlap。

## 验证与交付

- 运行完整 PTOAnalyzer CPU-only 测试。
- 使用 Stage 3 的 16 份 Profile 离线重放九个工具，覆盖单卡最大依赖图、2 卡和 8 卡样本；新工具对缺失或不支持的归档输入如实报告来源状态。
- 对代表性 AllocateMemoryAddr 产物独立复算按函数/memory space 分组的记录数和请求字节数，并与 Tool 输出逐项核对。
- 对照 Stage 3 比较摘要事实、来源覆盖和明细遍历；记录旧 Stage 2 结果无法逐值比较的限制。
- 核对 5,492 个归档产物回放前后哈希一致；确认 PTOBench CLI 未调用、NPU 未访问、硬件动作新增为 0。
- 写入 `reports/offline_replay_stage4_<date>.json`，更新 PTOAnalyzer README、planning README，并修正 Stage 3 README 和清单状态。
- 若任何目标必须新增 PTOBench 采集字段或破坏既有观测合同，停止该分支并提交独立、可审查的需求说明；不要将问题隐去为“验收通过”。

Task 7 Host 接线和 Task 8 Agent 试验分别受清单中的前置条件控制；可在前置条件未满足时明确记录为未启动，不阻塞 Stage 4 离线修缮完成。
