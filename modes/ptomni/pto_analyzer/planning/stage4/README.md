# PTOAnalyzer Stage 4：大文件处理与 Dataflow 分配摘要

> 状态：规划已更新，尚未实施。Stage 3 的 PTOAnalyzer 实现和离线验收已完成；本次只修订 Stage 4 计划，不启动 Profile、evaluate、Basic Pi 或 Agent 实验。

## 目标

Stage 3 已证明现有八个只读工具能在 16 份归档上重复生成摘要和分页结果，且归档未改变。Stage 4 收敛大文件解析、重复扫描、图分页和错误记录的内存成本；另从编译器 AllocateMemoryAddr 产物汇总静态 tile 分配信息，补上 Dataflow Tuner 的一项独立证据。宿主接线和真实 Agent 效用验证仍是有前置条件的后续任务。

## 当前证据

| 事项 | 当前事实 | Stage 4 处理 |
|---|---|---|
| 离线功能 | 8 个工具、16 份 Profile、128 次摘要/分页调用；43 项 CPU 测试通过 | 保留当前工具权限、完整摘要、Cursor 身份绑定与只读归档行为 |
| 归档安全 | 5,492 个索引产物回放前后 SHA-256 相同；新增硬件动作 0 | 每轮离线回归继续核对哈希，不能写回 Profile |
| 依赖图分页 | 最大单卡图有 453,375 条明细；一页约 29 秒，临时解析峰值约 0.87–0.95 GiB | 优先解决重复全量解析及大对象展开成本，保留完整明细和稳定顺序 |
| 大文件与错误记录 | PMU CSV 会重复扫描；事件账本整文件读入并拆行；JSONL/CSV 问题文本会随坏记录数增长 | 逐行读取事件账本、单遍解析 PMU、限制错误示例并保留精确拒绝数；分别测量剩余输入内存成本 |
| Scope 来源查找 | `buffer_pressure_check` 对 Scope 来源反复线性搜索 | 建立路径索引，保持原有来源引用和聚合结果 |
| Dataflow 分配证据 | 编译 Profile 要求归档 `memory_after_AllocateMemoryAddr` 报告或 `*_after_AllocateMemoryAddr.py`；打印的 `tile.alloc` 记录带 memory space 和字节数 | 规划 `tile_memory_allocation_summary`，按产物、函数（若有）和内存空间汇总分配数与请求字节数 |
| Host | 仓库未发现 PTOmni 宿主调用 `ToolSession` | 先定位正式 Host；存在时接到其原生工具入口，否则记录依赖并停止接线实现 |
| Pipeline overlap | 16 份 Profile 均无可配对的运行时 Compute/MTE 区间 | `pipeline_overlap_check` 继续停用；Stage 4 用独立的编译分配摘要填补 Dataflow 工具空位，不推算 overlap |

这里的“坏记录”指解析时发现的具体格式或内容问题，例如 JSONL 某行不是合法 JSON、末行被截断，或 CSV 行的列数与表头不一致；大文件本身不代表文件损坏。Stage 3 没有报告归档哈希变化，但哈希一致只证明回放期间内容未被改动，不能证明文件内容在生成时语义正确。Stage 4 限制的是保留在内存中的错误示例数量；原始文件读取、哈希校验和解析仍会消耗与输入规模相关的资源，验收时要分别测量，不能宣称总内存恒定。

## 工作顺序

1. **校正文档状态**：把 Stage 3 总览和清单标为“实现及离线验收完成”；Stage 3 Task 6/7 明确标注为可选且未启动。Stage 2 observation 值不可回溯比较的限制继续保留。
2. **限制错误报告内存**：对每个工具/来源最多保留有限条错误示例，并同时报告精确错误总数及未展示示例数。合法记录计数、拒绝计数和来源路径不变；不得静默丢弃错误数量。
3. **降低大文件重复处理**：流式读取 `events.jsonl`；把 PMU CSV 解析改为一次遍历且不缓存全量行；Scope 来源用路径索引；合并 `compute_task_share` 重复的未关联记录处理分支。只做能维持现有观测语义的改动。
4. **修复依赖图分页成本**：先用最大单卡、双卡和 8 卡真实归档采样定位时间与内存来源。再采用能保持所有当前摘要字段和明细排序的最小方案，使同一 `ToolSession` 连续翻页不重复构建完整依赖图，也不长期保存展开后的整张 task/edge 对象图。优先使用标准库和当前归档，不引入数据库、持久索引或新服务。
5. **新增 `tile_memory_allocation_summary`**：先审计现有归档中 AllocateMemoryAddr 输出的真实格式和字段，再实现只读汇总。保留文件、函数、内存空间和记录来源；只报告编译器记录的分配数及分配请求字节数，不推断运行时峰值、容量压力或瓶颈。若现有归档不足以验证格式，记录阻塞，不自动重跑 Profile。
6. **弱化测试对实现细节的依赖**：保留大输入分页内存上界、无重无漏、末页和跨 Profile Cursor 检查；分页器测试不直接读 `_heap`，Cursor 测试不锁定 Base64 编码细节。
7. **完成离线验收**：运行 PTOAnalyzer 全部 CPU 测试，重放 16 份现有 Profile 与最终九个工具，对最大单卡依赖图及 2/8 卡样本记录耗时和峰值内存，并核对 5,492 个归档产物哈希。离线阶段不得增加硬件动作。
8. **Host 接线与 Agent 试验门槛**：若存在正式 PTOmni Host，把 `ToolSession.tool_specs` / `call()` 通过它现有的原生工具传输提供给被启用的 Tuner；Lead 仍无 Analyzer Tool。若 Host 尚不存在，只交付接线契约和依赖项，不创建第二个 Tool Server。Basic Pi 轨迹和“原始 Profile vs PTOAnalyzer”配对试验均为可选后续任务，需独立固定 Case、预算和设备时段后再启动。

## 工具替换与边界

`pipeline_overlap_check` 已从当前接口移除。现有 Profile 有聚合的 MTE PMU 计数、普通 Runtime 时间线和编译 Pass 时间，但没有同一运行时事件中的 Compute 与 MTE 起止时间，不能据此推算重叠。Stage 4 不恢复它，也不使用 `MTE busy cycles` 与 AICore span 的比值冒充 overlap。

Stage 4 规划用 `tile_memory_allocation_summary` 填补 Dataflow 的工具空位。它读取编译器 AllocateMemoryAddr 后的分配产物，并按产物、函数（若来源提供）和 memory space 汇总记录数与请求字节数；`buffer_pressure_check` 仍汇总 MemoryReuse 提示和 Scope 原始计数，两个工具来源和问题不同。

该工具只描述编译器记录的静态分配请求，不表示运行时存活量、峰值占用、可用容量、溢出或性能瓶颈。不同产物或函数的分配字节数不得合并并称作峰值；每项汇总必须保留来源产物和解析范围。只解析真实格式可支持的字段，不根据文件名或缺失字段猜测数值。

## 不在本阶段做

- 除 `tile_memory_allocation_summary` 外，不新增诊断 Tool、阈值、瓶颈结论、优化处方或第二套硬件账本。
- 不修改 PTOBench Collector；若现有归档没有可验证的 AllocateMemoryAddr 产物，另列采集需求和独立评审，不在 Stage 4 擅自扩展采集。
- 不在 PTOAnalyzer 内启动 `profile`、`evaluate`、Runner、NPU 或 Agent。
- 不将 Basic Pi / Agent 可选试验当作 Stage 4 离线实现的完成条件。

## 文件

- [任务清单](TASK_CHECKLIST.md)：执行顺序、涉及文件和可检查的验收条件。
- [实施 Prompt](IMPLEMENTATION_PROMPT.md)：可直接交给后续实施任务。
- [Stage 3 回放报告](../../reports/offline_replay_stage3_20261108.json)：现有八个工具、分页、归档和硬件动作证据。
- [PTOAnalyzer 当前接口](../../README.md)：Stage 3 当前的八工具权限与数据边界；Stage 4 的九工具目标以本规划为准。
