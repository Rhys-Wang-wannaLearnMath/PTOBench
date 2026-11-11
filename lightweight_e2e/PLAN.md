# Lightweight e2e：两条优化路线与单 token 验收规划

更新：2026-10-18。本文按最新需求替代之前的连续生成规划；规划和任务清单已完成，66 项优化实验已启动；测试 harness 和集成验证仍待实施。

2026-10-19 范围核查补充：按“大子图优先、去掉被覆盖的小 case”形成的 [24＋4 精简方案](REDUCED_TASK_SET.md) 已应用到串行队列。2026-10-19 18:29 +08:00 起按文件隔离条件全新启动全部 28 项，复用数为 0，每项 100 次探索＋1 次收尾验证。下文 62＋4 为原实验范围；当前来源及启动要求以补充方案和 [最新实验回执](experiments/basic_pi_deepseek_flash_max_no_wiki/launch_receipt.json) 为准。

本轮 [Basic Pi 实验配置](experiments/basic_pi_deepseek_flash_max_no_wiki/README.md) 已覆盖 62＋4：当前 A004 及后续任务每项 100 次探索预算（另保留 1 次收尾验证）、无 wiki，并通过 `model_source: profile` 选择 DeepSeek Flash Max。Basic 的动态开关及配置检查已完成，实验已修复 max 映射，并通过 tmux 串行队列重启；下文的单 token harness 和优化集成仍是后续工作。

## 1. 本轮目标

同时推进两条路线，分别观察独立 case 的收益和完整轻量 forward 的收益：

- **路线 A：独立 case 优化 → 工程适配 → 集成至对应的完整 case。** 主范围为 Flash 35 个、Pro 27 个已有独立 benchmark。
- **路线 B：直接优化四个完整 case。** 面向整个 forward 做算子优化、融合、调度和通信优化。

两条路线最终都执行一次 prefill 或 decode，并比较一个指定预测位置的完整词表 raw logits。保留当前固定 shape、7 层配置和 EP2/TP2；“一个 token”指一个输出预测位置，不要求 prompt 长度为 1，也不将现有 kernel 的 token 维度改成 1。

本轮不要求持续生成、外层 token loop、跨多步 cache 延续、tokenizer 或文本质量评估。Decode 仍需有效初始 cache；每次独立测试及性能采样前恢复同一份初始状态。Sampling 沿用现有实现，数值回归检查到 sampling 之前的 logits。

## 2. 四个完整 case

| 模型 | Prefill | Decode | 固定轻量配置 |
| --- | --- | --- | --- |
| Flash | [synthetic_token_loop_prefill_flash](../cases/deepseek_v4/synthetic_token_loop_prefill_flash/case.json) | [synthetic_token_loop_decode_flash](../cases/deepseek_v4/synthetic_token_loop_decode_flash/case.json) | 7 层、EP2/TP2、hidden=4096、MoE intermediate=2048、64 个有效路由专家；SWA/CSA/HCA |
| Pro | [synthetic_token_loop_prefill_pro](../cases/deepseek_v4/synthetic_token_loop_prefill_pro/case.json) | [synthetic_token_loop_decode_pro](../cases/deepseek_v4/synthetic_token_loop_decode_pro/case.json) | 7 层、EP2/TP2、hidden=7168、MoE intermediate=1024、192 个有效路由专家；HCA/CSA |

四个 case 当前都输出 FP32 `logits`，形状为 `[2, 8, 129280]`，词表分片已汇集，并另行输出 `sampled_ids`。验收从中选一个有明确请求/位置映射的有效行 `[129280]`，不比较 padding 行。

本机为 8×Ascend 910B4，每卡 32 GiB HBM。仅轻量 INT8 路由专家矩阵约为 Flash 5.25 GiB/卡、Pro 13.78 GiB/卡；完整占用还包括其他权重、cache、通信和 scratch，首轮需实际检查。这些测试反映缩减配置的收益。

## 3. 路线 A：独立优化后集成

[CASE_MAP.md](CASE_MAP.md) 列出全部主范围 case、对应调用位置、计算包含关系和适配要求。[optimization_tasks.csv](optimization_tasks.csv) 为 62 项路线 A 任务逐项记录源 case、目标完整 case、接入位置、优先级及适配说明。

任务清单的 `source_case_path` 相对 PTOBench 根目录，`target_case_ids` 用分号分隔。路线 A 的 P1–P6 依次为 RMSNorm、QKV/压缩器/indexer、HC/gate/专家、sparse attention/LM head、完整 attention/MoE、完整 layer；这是建议推进顺序，路线 B 标为 `independent`。这些优先级尚未依据设备 profile 排序。

“集成到四个 case”表示覆盖这四个载体中的对应路径：Flash 优化迁入 Flash，Pro 优化迁入 Pro，prefill/decode 按实际调用匹配；共用 helper 可在适配 token extent 后迁入两阶段。

每项任务执行：

1. 在独立 Candidate workspace 中优化源 case，通过其既有正式评测，记录局部延迟和 speedup。
2. 将通过验收的优化迁入完整 case 的独立 workspace/adapter。保持目标数学语义、精度、权重、形状和初始状态，适配源 case 的 ABI、布局、cache 及调度依赖。
3. 对受影响的完整 case 做单 token logits 检查及单步计时，记录局部收益如何反映到完整 forward。
4. 先分别替换，随后组合已通过的优化；每个组合重新与冻结 baseline 比较。

Leaf、attention、layer 是包含关系，不能重复执行其中的 norm/输出投影，也不能将其局部耗时或 speedup 相加。选择完整子图实现时，明确其包含哪些 leaf 优化。

其他实现版本、局部 dispatch/combine、EP4/EP8 为辅助优化来源，详见 CASE_MAP 第 5 节；不自动扩大本轮 62 项主范围。Pro 有 4 项明显配置差异，已在任务清单标记，适配时沿用目标 e2e 的配置。

## 4. 路线 B：直接优化完整 case

为四个载体分别建立独立优化 workspace，入口、固定配置和正式评测契约保持原样。直接优化其依赖文件及整图，包括算子实现、融合、内存复用、执行顺序、通信/计算重叠；输出 raw logits 的语义与精度要求保持一致。

路线 B 从同一个原始 baseline 出发，先单独验证和计时。它无需以路线 A 的成功为前提。以后若组合 A 与 B，另存一个组合版本并重新验收，不将组合结果计作纯路线 B。

| 版本标识 | 含义 | 比较对象 |
| --- | --- | --- |
| `baseline` | 开始优化前冻结的完整实现及依赖 | 基准 |
| `integrated_<task_id>` | 只接入一项路线 A 优化 | `baseline` |
| `integrated_all` | 兼容且通过验收的路线 A 优化组合 | `baseline` |
| `direct` | 路线 B 独立优化完整 case | `baseline` |
| `combined`（可选） | A 与 B 的兼容组合 | `baseline`，并列展示 A/B 结果 |

每个载体有自己的 baseline；Flash/Pro、prefill/decode 之间不互作数值 reference。

## 5. 最简单的正确性测试

详细步骤和伪代码见 [SINGLE_TOKEN_TEST.md](SINGLE_TOKEN_TEST.md)。两条路线共用这一方法：

1. 每个完整 case 固定一份非零权重测试样本、输入、位置 metadata、初始状态和目标 logits 行。
2. 用原始 baseline 运行一次，保存这一行的 FP32 raw logits。
3. 优化后恢复同一份初始状态，运行一次，取同一行的完整词表 raw logits。
4. 检查 shape、有限值，并执行 `torch.testing.assert_close`。不要求逐 bit 一致；第一版容差建议起点为 `rtol=1e-2, atol=1e-2`，在优化前准备数值基线时确认并冻结，未经实测不视为已校准的通用容差。

Prefill 选固定 prompt 最后一个有效位置；decode 选固定输入 token 对应的一个有效预测位置，并复用预先准备好的初始 cache 快照。无需在线 prefill→decode 串联，也无需采样下一步 token。

现有零权重输入用于冒烟；有意义的数值回归使用同一份非零权重/量化参数和匹配状态。两边分别从初始快照恢复，不能把 baseline 执行后修改过的状态直接交给 Optimized。

这项 raw logits 回归与既有 PTOBench 正式验收分别记录。正式 `case.json` 和 `public/comparator.json` 保持原规则；四个完整 case 的正式评测还检查其他输出，当前 logits comparator 是 `exact`，不能把本规划的附加容差写入其中。

## 6. 性能对比

只对正确性通过的版本测量。固定设备/工具链、权重、输入、有效行布局、初始状态和全部运行参数，覆盖原始完整 case 单次入口的全部执行工作；既有 sampling 在所有版本中保持一致并计入执行耗时。旧计划中的状态驻留和计时策略已废止；本文件保留为轻量实验的历史目标记录。

附加单步测试默认 **5 次预热、10 次正式测量，取中位数**。每轮按现有 PTOBench/runtime 边界处理输入/权重上传、初始 cache/压缩器状态、scratch 必要初值和通信信号恢复；全部必要设备工作完成后结束执行计时，再回读 logits/状态供正确性比较。输入上传、初始状态恢复、最终验证回读和 CPU 比较不计入；正常 host dispatch、执行必需的跨卡通信及依赖等待继续计入。Ref/各 Candidate 采用相同边界并一起重测。性能统计只用原始逐轮 `effective_us`，不另采集 host lifecycle 时间戳，也不做启动差后处理或扣时。

记录 `speedup = baseline_median_ms / optimized_median_ms`。Prefill 指处理整个固定 prompt 的延迟，decode 指现有固定 shape 下的一次完整调用延迟；只验收一个 logits 行不意味着设备只计算一个 token。两者分别报告，不相加为连续生成耗时。正式 PTOBench 结果与非零样本上的附加结果分开保存。

## 7. 实施顺序、产物与状态

1. 准备：冻结四个 baseline、公开配置/源码版本及测试输入标识；验证工具链、入口和实际内存占用。
2. 单步基线：准备非零样本和 decode 初始状态快照，确认目标行有效且 logits 有限、非恒零；确认并冻结容差。
3. Flash：路线 A 按 RMSNorm → QKV/compressor/indexer → HC/专家 → attention/MoE/layer/LM head 推进；路线 B 可独立开展。
4. Pro：复用相同方法，完成配置适配，分别推进 A/B。
5. 汇总：四个载体分别展示 baseline、各单项集成、集成组合及 direct 的正确性、耗时和 speedup。

每次结果最少保存：case ID、路线/版本/源任务 ID、源码与依赖标识、设备和工具链、输入/权重/状态标识、目标行对应请求及位置、rtol/atol、PASS/FAIL、最大绝对误差、延迟原始样本及中位数。失败时再定位中间张量或 cache。

`optimization_tasks.csv` 中全部任务当前为 `planned`。完成条件为：路线 A 有源 case 正式验收、适配记录和对应完整 case 的数值/性能结果；路线 B 有完整 case 正式验收及同样的附加数值/性能结果。未通过项和无收益项也保留原始记录。

单 token 回归只覆盖指定输入和预测位置；多步 cache 更新、其他有效行、长序列数值与采样器优化不由这项附加检查证明，后续需要时再扩展。
