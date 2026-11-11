# PTOAnalyzer 九个 Tool 的参考项目分析

## 结论

这些项目值得借鉴的是**如何解析和组织可追溯证据**，不是直接搬用它们的性能评级、阈值或优化结论。PTOAnalyzer 面向 PTOBench 的归档数据，继续保持只读、确定性；单项计数或编译器提示只能作为事实或待验证线索，不能直接升级成性能根因。

最有用的实现路径有两条：PyPTO-Agent 把运行症状逐步关联到任务、源码和编译器行为；PyPTO-Gym 的 CANNBot Skills 提供可直接查阅的 PyPTO swimlane、核心使用和依赖链解析脚本。它们都假定各自的采集字段及硬件语义成立，不能默认套用到 PTOBench。

## 逐 Tool 建议

| Tuner / Tool | 值得借鉴的实现模式 | 对 PTOAnalyzer 的具体建议与边界 |
|---|---|---|
| **Compute — `compute_mapping_check`** | PyPTO-Agent 的 L2→L3 流程以及 Gym 的 `leafhash_to_code.py`，先从事件取得 leaf/hash，再关联编译程序和源码位置。 | 借鉴分阶段关联，并且仅在 Profile 的 Candidate 身份、程序映射和源码快照都能核验时输出源码位置。当前 name map 只是 callable 名称，不等于已验证的 Candidate 源码映射；未关联项保留数量、来源和具体缺口。参考：`projects/pypto-agent/engine/diagnosis/l3_attribute.py`、`projects/pypto-gym/cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/scripts/leafhash_to_code.py`。 |
| **Compute — `compute_unit_activity`** | CANN `analyze_perf.py` 和 PyPTO-Agent L1 将 AIC/AIV 分开，字段带单位，并显式处理零分母。 | 可借鉴具名、带单位的 per-core 汇总和稳定排序；继续把 PTOBench PMU 值作为原始计数。不要移植它们的利用率、bubble 公式或评级阈值，除非 PTOBench 对应事件的分母、时间窗口和单位已校准。PMU 计数和不能直接叫“利用率”。参考：`projects/pypto-gym/cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py`、`projects/pypto-agent/engine/diagnosis/l1_symptom.py`。 |
| **Compute — `compute_task_share`** | PyPTO-Agent L2 按 leaf/task 聚合 count、total、average、max，并在报告中优先展示贡献较大的条目。 | 借鉴已验证 task 关联后的分组统计；Tuner 初始摘要须汇总全部类别，不能用 Top-K 代替总体。task 明细可分页。同一 dispatch 内 PMU、deps、name map、swimlane 的严格 join 应保留。PMU counter sum 与 AICore 时间跨度不同量纲，不能相除或合成“任务耗时份额”；未匹配记录应单独计数并说明原因。参考：`projects/pypto-agent/engine/diagnosis/l2_localize.py`。 |
| **Dataflow — `tile_access_check`** | CANN Skill 将 swimlane/编译信息整理成有适用条件的检查项，并建议下一步观察；`analyze_perf.py` 保留逐核指标。 | 保持 PH001 的原始提示、来源和行号；可以补充提示类别的结构化摘要，并要求 Tuner 回到对应 Candidate 检查。提示是编译器观察，不证明运行时瓶颈，也不能直接附带“应增大/减小 Tile”的结论。参考：`projects/pypto-gym/cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/SKILL.md`、`projects/pypto-gym/cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/SKILL.md`。 |
| **Dataflow — `buffer_pressure_check`** | PyPTO-Agent L4 从 `program.json` 提取 spill、DDR relay、CV 通路等明确编译事实，并和 L2 运行事件分开呈现。 | 可参考“编译事实 + 原始 Scope 计数 + 来源”的分层输出；保留每项原始单位，未知单位就明确标未知。当前 PTOBench 缺少经过验证的容量/分配语义时，不要把 Scope 数字换算成压力百分比，也不要仅凭某类 hint 断言 buffer 溢出。参考：`projects/pypto-agent/engine/diagnosis/l4_compiler.py`。 |
| **Dataflow — `pipeline_overlap_check`** | CANN timeline 脚本与 PyPTO-Agent L2 都先归一化事件，再按核和事件身份汇总区间。 | 只有同一 Profile、设备、时间窗口中存在可辨认的 Compute/MTE 区间，且时间单位、事件身份和覆盖完整性通过校验后，才可计算 overlap。若没有，应报告实际来源覆盖和缺少的区间字段，不用诊断标签概括。编译 pass 时间、PMU 总计数和普通 task 起止时间都不能替代这些运行区间。参考：`projects/pypto-agent/engine/diagnosis/l2_localize.py`、`projects/pypto-gym/cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/scripts/analyze_swimlane.py`。 |
| **Schedule — `dependency_path_check`** | CANN `analyze_aiv_dep_chains.py` 和 PyPTO-Agent 的图分析会归一化节点/边、去重并按层展示依赖链。 | 可借鉴稳定的图规范化、重复边处理和链条证据引用。当前无 runtime 权重时，深度、扇入/扇出只描述图结构；不要称为运行时关键路径或延迟瓶颈。参考：`projects/pypto-gym/cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/scripts/analyze_aiv_dep_chains.py`、`projects/pypto-agent/engine/diagnosis/l2_localize.py`。 |
| **Schedule — `dispatch_gap_check`** | PyPTO-Agent `compactness.py` 根据同核任务区间生成观测 gap，并只在独立 L1 wait 证据可用时尝试归因；其报告在证据不足时写出“类型未知”。 | 可借鉴“先计算可观察区间，再独立检查归因证据”的分层。实施前仍须验证 Level-4 覆盖、原始与合并 swimlane 的记录对应、同核时间语义和时钟单位。不能把观测空档直接命名为调度开销；没有独立支持证据时，只报告空档统计以及归因证据缺少的事实。参考：`projects/pypto-agent/engine/report/compactness.py`。 |
| **Schedule — `task_distribution_check`** | CANN `analyze_core_usage.py` 与 PyPTO-Agent L2 将 task 按 leaf/core 聚合；相关工具会在缺少硬件基线时明确表达该基线不可用。 | 可借鉴每 rank/core 的计数和持续时间分布，并明确写出基线字段是否存在。当前 swimlane tuple 行数不必然等于唯一 task 执行次数，记录跨度也不等于忙碌时间；需要验证 tuple schema、捕获范围和理论核数后，才能计算 occupancy 或 imbalance。参考：`projects/pypto-gym/cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/scripts/analyze_core_usage.py`、`projects/pypto-agent/engine/diagnosis/l2_localize.py`。 |

## 对 PTOAnalyzer 的统一落地原则

1. 先沿用归档和 PTOBench 的身份/哈希边界，再解析；不扫描“最新目录”，不让工具自行重跑 Profile。
2. 对外返回完整聚合摘要和来源引用；大规模 task/core 明细按需分页。PyPTO-Agent 的 Top-K 可用于定位高贡献明细，但不能替代 Tuner 初始摘要中的全量统计。
3. 对外只呈现原始观测、可复算的统计、来源覆盖和解释边界。已有参考工具中的“类型未知”、评级、根因和优化建议体现的是其自身工作流；PTOAnalyzer 不复制为 Tool 级性能结论或状态标签。
4. 外部项目的 AIC/AIV 公式、阈值、hardware core count 和优化建议依赖各自数据契约；先核实 PTOBench 的字段、单位、身份和覆盖，再决定是否借鉴解析与聚合实现。

## 当前阶段的优先级

- **直接参考实现细节**：`compute_task_share` 的可追溯分组、`dependency_path_check` 的图规范化、`task_distribution_check` 对硬件基线覆盖的明确表达、`dispatch_gap_check` 的 gap 计算与归因分层。
- **先补数据契约再设计**：`compute_mapping_check` 的 Candidate 源码位置映射、`compute_unit_activity` 的利用率计算、`buffer_pressure_check` 的归一化压力、`pipeline_overlap_check` 的运行时阶段重叠。
- **维持保守输出**：`tile_access_check` 保留编译器提示作为待检查线索，不把它扩写成确定性优化处方。

本分析依据本目录保留的三个固定版本参考材料，并对照当前 PTOAnalyzer 的 `diagnostics/compute.py`、`diagnostics/dataflow.py`、`diagnostics/schedule.py` 与 `toolset.py`。仓库描述的 NPU 实验结果是作者报告，不是本项目独立复现结果。当前保存的参考材料不含 KernelPro 的源码或输出契约，因此关于 KernelPro 仅沿用“围绕具体证据问题组织 Micro-tool”的设计思路，不归纳其状态字段或报告格式。参考工具脚本未在本次分析中运行，也未修改 PTOAnalyzer 实现。
