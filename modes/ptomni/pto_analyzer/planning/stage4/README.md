# PTOAnalyzer Stage 4：大文件处理与 Dataflow 分配摘要

> 状态：Stage 4 实现与 CPU-only 离线验收已完成（2026-11-12）。Stage 3 的八个工具已修缮，并新增一个有真实归档依据的静态分配摘要；未启动 Profile、evaluate、Basic Pi 或 Agent 实验。

## 目标

Stage 3 已证明现有八个只读工具能在 16 份归档上重复生成摘要和分页结果，且归档未改变。Stage 4 收敛大文件解析、重复扫描、图分页和错误记录的内存成本；另从编译器 AllocateMemoryAddr 产物汇总静态 tile 分配信息，补上 Dataflow Tuner 的一项独立证据。宿主接线和真实 Agent 效用验证仍是有前置条件的后续任务。

## 当前证据

| 事项 | 当前事实 | Stage 4 处理 |
|---|---|---|
| 离线功能 | 9 个工具、16 份 Profile、144 次摘要/首屏校验；51 项 CPU 测试通过 | 保留当前工具权限、完整摘要、Cursor 身份绑定与只读归档行为 |
| 归档安全 | 5,492 个索引产物回放前后 SHA-256 相同；新增硬件动作 0 | 每轮离线回归继续核对哈希，不能写回 Profile |
| 依赖图分页 | 最大单卡图有 453,375 条明细；Stage 4 首屏约 20 秒，`tracemalloc` 峰值约 509 MiB，后续页不再重复解析 | 保留完整明细和稳定顺序；实测资源对比记录在回放报告中 |
| 大文件与错误记录 | JSONL 与 CSV 的错误示例每个来源最多保留 20 条；原始文件读取、哈希和解析仍随输入大小消耗资源 | `events.jsonl` 逐行读取，PMU CSV 单遍解析；错误总数准确，路径和行/记录号保留 |
| Scope 来源查找 | `buffer_pressure_check` 使用 Scope 路径索引匹配来源 | 保持原有来源引用和聚合结果，避免逐字段线性搜索 |
| Dataflow 分配证据 | 16 份 Profile 均有可验证的 `*_after_AllocateMemoryAddr.py`；独立 AST 复算得到 4,166 个 `pl.tile.alloc`、70 个 `pld.tensor.alloc_window_buffer`、1,228 个分组 | `tile_memory_allocation_summary` 已按产物、函数、memory space 和 allocation op 汇总显式请求字节数 |
| Host | 仓库未发现 PTOmni 宿主调用 `ToolSession` | 接线留待后续阶段；未创建模拟 Host 或第二个 Tool Server |
| Pipeline overlap | 16 份 Profile 均无同一时钟且身份可关联的运行时 Compute/MTE 起止区间 | `pipeline_overlap_check` 继续停用；新增静态分配摘要不推算 overlap |

这里的“坏记录”指解析时发现的具体格式或内容问题，例如 JSONL 某行不是合法 JSON、末行被截断，或 CSV 行的列数与表头不一致；大文件本身不代表文件损坏。Stage 4 限制的是保留在内存中的错误示例数量，错误总数仍准确；原始文件读取、哈希校验和解析仍会消耗与输入规模相关的资源，不能据此宣称总内存恒定。归档哈希一致只证明回放期间内容未被改动，不能证明文件内容在生成时语义正确。

## 实施与验收记录

1. **校正文档状态**：已将 Stage 3 总览和清单标为实现及离线验收完成；Basic Pi 与 Agent 配对试验明确为可选且未启动。Stage 2 observation 值不可回溯比较的限制继续保留。
2. **限制错误报告内存**：已对每个工具/来源最多保留 20 条错误示例，同时报告精确错误总数和省略数；合法记录计数、拒绝计数和来源路径保持不变。
3. **降低大文件重复处理**：已逐行读取 `events.jsonl`，单遍解析 PMU CSV，按路径索引 Scope 来源，并合并 `compute_task_share` 的未关联记录构造。
4. **修复依赖图分页成本**：已采用受保护原文的轻量偏移索引；同一 `ToolSession` 连续翻页不重复构建完整依赖图，也不长期保存展开后的 task/edge 对象图。
5. **新增 `tile_memory_allocation_summary`**：已按真实 AllocateMemoryAddr Python 产物实现只读汇总，保留文件、函数、内存空间、操作和来源行；只报告显式静态请求，不推断运行时峰值、容量压力或瓶颈。
6. **弱化测试对实现细节的依赖**：已保留 100,000 条记录压力覆盖、无重无漏、末页和跨 Profile Cursor 检查；分页上界测试不读取 `_heap`，Cursor 测试不锁定具体 Base64 拼写。
7. **完成离线验收**：已通过 51 项 CPU 测试，重放 16 份现有 Profile 的九个工具，完成最大单卡、2 卡和 8 卡依赖图的分页资源测量，并核对 5,492 个归档产物哈希；没有新增硬件动作。
8. **Host 接线与 Agent 试验门槛**：已确认仓库没有 PTOmni Host 调用 `ToolSession` 的入口，因此 Host 接线、Basic Pi 和 Agent 配对试验留待后续阶段；没有创建模拟 Host 或第二个 Tool Server。

## 工具替换与边界

`pipeline_overlap_check` 已从当前接口移除。现有 Profile 有聚合的 MTE PMU 计数、普通 Runtime 时间线和编译 Pass 时间，但没有同一运行时事件中的 Compute 与 MTE 起止时间，不能据此推算重叠。Stage 4 不恢复它，也不使用 `MTE busy cycles` 与 AICore span 的比值冒充 overlap。

Stage 4 已用 `tile_memory_allocation_summary` 填补 Dataflow 的工具空位。它读取编译器 AllocateMemoryAddr 后的分配产物，并按产物、函数、memory space 和 allocation op 汇总记录数与请求字节数；`buffer_pressure_check` 仍汇总 MemoryReuse 提示和 Scope 原始计数，两个工具来源和问题不同。

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
- [PTOAnalyzer 当前接口](../../README.md)：Stage 4 当前的九工具权限、数据边界与分页契约。
