# DeepSeek V4 Flash：PTOBench case 与 e2e 推理的对应关系

核查日期：2026-10-15 至 2026-10-16。范围是当前工作树的全部 319 个 canonical `cases/*/*/case.json`、公开配置与 Candidate 源码结构，重点跟踪 Flash 的 attention、MoE、layer、head、prefill/decode 和 speculative 路径。没有执行 NPU 测试，也没有据此认定任何迁移中的 case 已经 ready。逐项清单见 [case_inventory](deepseek_v4_flash_e2e_case_inventory_20261016.csv)。没有改动 Case、Fixture、Ref、Comparator 或运行代码。

外部参考固定为 [hw-native-sys/pypto-serving @ 9e16e45098cc324e9aa48f904906ca5fcc58b810](https://github.com/hw-native-sys/pypto-serving/tree/9e16e45098cc324e9aa48f904906ca5fcc58b810)。该版本绑定的 pypto-lib submodule 是 `a595ab3f9496ad4f9c201a6967c3f97b56b4da17`，与本地 HEAD `9c5dfec2dfa18f056098000123e10de06de32426` 不同；实际接入必须固定并核验 kernel ABI，不能把最新版 serving 当作与所有历史 case 兼容。

**1. 结论与全部 case 的范围**

PTOBench 已覆盖 Flash 主干的大多数计算组件，可以把优化后的实现接入同一个完整推理框架，对比优化前后的 e2e。但当前公开 case 不是一个可直接串联的真实 Flash 全模型、多步推理基准：319 个 case 的 correctness mode 全部是 `single_step`，固定配置与独立 fixture 不能自动构成同一个模型实例。

**2026-10-16 本机容量核查补充：本机是 8×910B4、每卡 32 GiB。完整 Flash W8A8 的路由专家矩阵本身就需要 258 GiB，EP8 下每卡 32.25 GiB，超过单卡总 HBM；因此前文的 serving 集成建议是计算路径方案，不能视作本机可直接运行完整 Flash 的承诺。Qwen3-14B 是本机更可行的完整模型目标，仍需匹配软件栈和 checkpoint 后实测。**

此判断限定于所核查 pypto-serving 版本的 W8A8 全权重常驻路径，没有执行完整模型启动或观察到 OOM。它不能证明允许低比特执行、CPU offload 或逐层流式权重后，完整 Flash 仍然无法在本机运行。

| 规范目录 | case 数 | 对 Flash e2e 的作用 |
| --- | ---: | --- |
| `deepseek_v4` | 202 | 主体：不同 preset 的算子、attention、MoE、prefill layer、HC/LM head、MTP projection、缩减 synthetic forward |
| `deepseek_v4_flash_dspark` | 49 | DSpark attention、prefill/decode layer、MoE、LM head、drafter/Markov 等 |
| `deepseek_v4_flash_mtp` | 38 | CP prefill attention、zigzag cache 交换、sampling 等；目录名不代表完整 MTP 引擎 |
| `deepseek_v3_2` | 8 | V3.2 的 front/back 和 scope 分解，不属于 V4 Flash 主干 |
| `qwen3_14b` | 12 | Qwen 的 embedding/prefill/decode/sampling 等，不属于 V4 Flash 主干 |
| `qwen3_32b` | 5 | Qwen decode 和 scope 分解，不属于 V4 Flash 主干 |
| 合计 | 319 | |

同一组件的 `demo`、`flash`、`pro` 是不同固定契约；leaf、attention、layer 又有包含关系。不能将它们视为依次执行的 319 个节点，也不能同时计入 layer 与其内部 attention/MoE 的耗时。

**2. Flash 完整主干应如何组成**

```text
token IDs → embedding → HC streams 初始化
          → 43 个主干层，每层 attention + MoE
          → HC head → final RMSNorm → LM head / vocab 分片合并
          → raw logits → sampling → 下一步 token IDs
```

按本地 Flash 配置的前 43 个 `compress_ratios`：layer 0、1 是 SWA；layer 2、4、…、42 是 CSA；layer 3、5、…、41 是 HCA。总计 SWA×2、CSA×21、HCA×20。每层有自己的真实权重和 cache；不能重复使用同一层权重来代表 43 层。前 3 层的 hash routing 也要保留真实 `layer_id` 和 `tid2eid`。配置中 main layers 之后的额外 attention ratio 不应被误计为第 44 个主干层。

Prefill 消费 prompt 并建立全部层的 cache；prompt 最后一个有效位置的 logits 预测第一个输出 token。随后 decode 每次消费选定 token，更新同一请求的 cache 并预测下一个 token。Prefill 和 decode 必须共用请求状态，不能每步恢复成独立 fixture。

| 组件 | 可复用的 PTOBench case，省略重复 preset | 接入含义 |
| --- | --- | --- |
| Prefill 整层 | `deepseek_v4.prefill_layer_ep{2,4,8}_{swa,csa,hca}_flash`，以实际存在的目录为准 | 已含 attention 与 MoE；优先从这一粒度验证 prefill 优化 |
| Prefill attention | `deepseek_v4.prefill_attention_{swa,csa,hca}_flash` | 已含 HC pre/post、norm/QKV、对应 attention，以及 CSA/HCA 压缩和 CSA indexer；使用后不要重复执行内部 leaf |
| Decode attention | `deepseek_v4.decode_attention_{swa,csa,hca}_flash` | 与 decode MoE 合成完整一层；当前 CSA case 的声明入口缺失，不能直接运行该公开 seed |
| MoE | `deepseek_v4.moe_ep{2,4,8}_flash`、`prefill_moe_ep{2,4,8}_flash` | 已含 routing、共享/路由专家、dispatch/combine、HC；按对应部署重新集成 |
| 可进一步替换的 leaf | `rmsnorm_*`、`hc_pre/post*`、`*qkv_proj_rope*`、`*compressor*`、`*indexer*`、`*sparse_attn*`、`gate*`、`expert_*`、`dispatch*`、`combine*` | 先独立正确，再替换上层调用，保持相同数学语义、dtype 和通信布局 |
| 最终输出 | `deepseek_v4.hc_head_flash`、`rmsnorm_decode_flash` / `rmsnorm_prefill_flash`、`lm_head_tp2_flash` | HC head 后仍需 final RMSNorm，再做 vocabulary projection；这些固定 case 的维度和 dtype 需要适配 |
| Sampling | `deepseek_v4_flash_mtp.sample_flash` | 验证 sampling 时单独接入；验证模型主干时在 raw logits 处观察 |
| DSpark 的整层替代路径 | `deepseek_v4_flash_dspark.prefill_layer_{swa,csa,hca}_flash`、`decode_layer_{swa,csa,hca}_flash`、`lm_head_flash` | 适用于 DSpark 部署，不能直接与主路径 block-128 fixture 混用 |
| 可选 CP | `deepseek_v4_flash_mtp.prefill_cp_{swa,csa,hca}_cp{2,4,8}_flash`、`prefill_cp_zigzag_cp{2,4,8}_flash` | 长 prompt 的 attention 并行与 cache 交换组件；不是额外的 Transformer 层，CP 本身也不等于 MTP |
| 可选 speculative | `deepseek_v4.mtp_projection_flash`；DSpark 的 `dspark_proj/prefill/drafter/markov_flash` | 还要接完整 draft、target verify、accept/reject、位置与 cache 提交/回滚，不能只把 drafter 输出接到 sample |

Embedding、tokenizer/聊天模板、真实 checkpoint 加载、layer weight stacking、metadata/page allocator 和持续运行的 host/runtime 调度需要由集成框架提供。它们有源码可参考，但不都是独立 PTOBench case。Embedding 可见 [lookup_embedding.py](../../../models/deepseek_v4_flash_mtp/lookup_embedding.py)；完整编排可见 [prefill_fwd.py](../../../models/deepseek_v4_flash_mtp/prefill_fwd.py) 和 [decode_fwd.py](../../../models/deepseek_v4_flash_mtp/decode_fwd.py)。这些 model 源码的存在不意味着 PTOBench 已有相同契约的全模型 case。

**3. 优先采用的部署路径与当前限制**

推荐第一阶段采用普通 Flash 的 autoregressive 路径，先关闭 MTP/DSpark speculation 与 prefix cache，建立 `prefill → 连续 decode → logits` 的数值与性能基线。pypto-serving 可提供 checkpoint 加载、调度和 resident cache；之后将 case 中的优化逐项迁入同一版本的 kernel 实现，并在完整 forward 中重新验证。

| serving 路径 | 外部部署约束 | 与 PTOBench 的关系 |
| --- | --- | --- |
| 普通 Flash，AR 或 MTP | 8 devices，attention DP8 / MoE EP8，外部 `--tp 1`，raw block 128，W8A8 checkpoint | 主参考；EP2 case 常缩减专家数，需恢复全 256 专家与真实权重。LM head 的内部 TP 参数独立于外部 attention TP |
| DSpark target-only 或 K=7 | 16 devices，TP4 / DP4 / EP16，raw block 32，同类 W8A8 checkpoint | DSpark case 适合贡献组件优化；本地 EP2/TP2 固定契约不代表可直接跑完整 DSpark checkpoint |

依据见 serving 的 [Flash 文档](https://github.com/hw-native-sys/pypto-serving/blob/9e16e45098cc324e9aa48f904906ca5fcc58b810/docs/user-guide/deepseek-v4.md)、[DSpark 文档](https://github.com/hw-native-sys/pypto-serving/blob/9e16e45098cc324e9aa48f904906ca5fcc58b810/docs/developer-guide/deepseek-v4-dspark.md) 和 [runtime 状态说明](https://github.com/hw-native-sys/pypto-serving/blob/9e16e45098cc324e9aa48f904906ca5fcc58b810/docs/developer-guide/deepseek-v4-runtime.md)。这些约束针对所引用版本，并非对所有未来部署的硬件下限判断。

两个容易误读的具体例子：

- `synthetic_token_loop_prefill_flash` / `decode_flash` 的公开配置实际为 7 主干层、EP2、零权重 synthetic；`N_EXPERTS_GLOBAL = 256 // 8 * 2 = 64`。它们虽输出 logits 和 cache，但只适合连通性与缩减链路性能，不能报告成真实 43 层 Flash 的数值或 e2e 性能。
- DSpark 的 `decode_layer_csa_flash` 冻结 `DECODE_BATCH=64`、`DECODE_SEQ=8`、TP2/EP2、block 32，并以 `256 // 16 * 2 = 32` 个全局专家做局部部署见证；LM head case 又冻结不同的 decode seq。名称同为 Flash 不保证布局可直接衔接。

公开入口存在性检查发现 4 个 case 的 `case.json` 声明模块不存在：`deepseek_v4.decode_attention_csa_{demo,flash,pro}`、`deepseek_v4_flash_mtp.prefill_cp_hca_cp4_pro`。这是公开资产可用性发现，不是 NPU 运行审计；其余入口存在也不等于已认证 ready。

**4. 正确性：sample 前 logits 是主检查点，但必须固定前缀**

建议保留两种 reference：优化前同量化 checkpoint 的全模型实现用于 A/B 回归；另用可信独立实现验证模型语义，避免优化前后共享同一个错误。W8A8 A/B 应使用相同已转换权重和量化参数；原始 FP8/MXFP4 与 W8A8 的差异应单独评估，不能全部归因于 kernel 优化。

主测试采用 teacher forcing：

```python
# pseudocode: ref/candidate each owns its own weights view and mutable cache
ref_state, ref_logits = ref.prefill(prompt_ids)
opt_state, opt_logits = opt.prefill(prompt_ids)
compare(valid_logits(ref_logits), valid_logits(opt_logits))

for token in fixed_continuation_ids:  # dataset tokens or a frozen reference rollout
    ref_state, ref_logits = ref.decode(token, ref_state)
    opt_state, opt_logits = opt.decode(token, opt_state)
    compare(valid_logits(ref_logits), valid_logits(opt_logits))
```

两边输入的 token、绝对位置、mask 和请求身份完全相同。每边从自己的 prefill 结果继续演进 state；不要每步用 Ref cache 覆盖 Candidate cache，否则会掩盖累计状态错误。若对齐逻辑布局的独立实现采用不同 paging，不必强求物理 block ID 相同。必须按 request/position 对齐有效行并组装正确词表顺序，排除 padding、inactive rows、未选中的 logits 和通信 scratch。

Prefill 至少比较每个请求最后一个有效位置的 logits；诊断配置最好支持全部有效 token 的逐位置 logits。长序列的全词表矩阵可分块比较或流式累积指标，避免把额外内存和 readback 成本带入性能测试。

| 检查对象 | 建议记录/判定 |
| --- | --- |
| Raw logits，LM head/vocab assembly 后、temperature/top-k/mask/Gumbel 前 | FP32 中比较；NaN/Inf 检查、最大绝对误差、RMSE/相对 L2、按 token 的误差分布。Raw logits 也能发现 softmax 不敏感的整体偏移 |
| 输出分布 | `softmax(logits)` 的 KL 或 JS、top-k 概率质量；cosine 和 top-k overlap 用作补充，不能单独作为正确性证明 |
| Greedy 稳定性 | top-1 一致率，以及 reference top1/top2 margin。若该行最大 logits 误差为 ε 且 margin > 2ε，top-1 应保持；低 margin 的差异要单独定位 |
| 中间节点 | 每层输出、Q/KV、压缩 cache/indexer、MoE gate/dispatch/combine、HC head、final norm；按需打开，用于定位首次偏离 |
| 持续状态 | 有效 raw KV、compressed KV、压缩器部分块状态、index cache/scale；比较逻辑有效内容。元数据、token IDs、位置和离散路由协议采用各自明确的规则，不能用一般浮点容差掩盖错误 |

容差应在优化前冻结，来源于所选 dtype、量化实现、独立 reference 偏差与重复运行稳定性。`lm_head_tp2_flash` 的 `rtol=atol=1e-3` 是固定 LM-head case 的规则，不应直接推广到 43 层累计误差。每行报告最坏位置和误差分位数，避免全局平均值淹没少数请求错误；token ID 应按离散值比较。

Sampling 独立测试 greedy、temperature、top-k 和 RNG。相同 seed 只有在 RNG 算法、counter/key、消费顺序和 tie-breaking 一致时才意味着相同随机数；微小 logits 差异也可能改变采样结果。两个实现各自自由生成，一旦 token 分叉，后续 logits 的条件就不同，因此不能拿这样的后续 logits 做逐步数值比较。自由生成仍应保留为生成质量/接口回归检查。

最小覆盖包括：短/长 prompt、ragged batch、不同连续 decode 长度；压缩 ratio=4 与 128 的边界；block 32/128 的边界和 SWA 滚动；indexer top-k 从未填满到填满；chunked 与未分块 prefill 的等价性；prefill+decode 与同前缀重算的对照；请求插入/退出、page 复用和重复运行。Prefix cache 单独比较冷/热路径的 logits。MTP/DSpark 再加入 target 对每个 draft prefix 的 logits、accept/reject、已提交位置和 cache 回滚/提交；greedy speculation 应验证与 target-only 已提交序列一致，随机 speculation 需验证其接受规则与目标分布。

所引用 serving 的 `tests/test_deepseek_v4_accuracy.py` 主要检查非空生成、finish reason、token 计数与 prefix-cache hit，明确不固定 continuation；DSpark 测试还覆盖冷/热 continuation 一致性和 speculative 路径标记。它们是有用的接口与状态回归，但没有形成独立 reference 的逐位置 raw-logits 数值验收。

serving 中已有诊断入口可用：`run_prefill()` 返回 logits；AR decode 当 `allow_device_greedy_sampling=False` 时返回 logits。性能路径可只返回 device-sampled IDs；正确性路径应显式导出所需 logits。融合 sampling 的 kernel 也可以在随机取样前增加只用于诊断的观察点，不要求为了测试永久拆分生产计算。

**5. 如何衡量优化前后的 e2e 效果**

用同一 harness 加载 baseline/optimized 两组 kernels；固定模型 checkpoint/量化、全部 43 层与专家数、拓扑、shape 范围、prompt token IDs、输出长度、并发、cache 策略、speculation 设置和工具链。只对测试前定义的实现变化做 A/B。独立 case 的 fixture cache 重置仍按原规则执行；全模型 e2e 在一次请求内部持续保留 state，只在两次独立重复实验之间恢复相同初态。

- Model e2e：prefill 到 logits、每步 decode 到 logits，以及含 sampling 的完整生成分别测量。使用设备完成的同步时间或 runtime completion，不能只测异步 launch。
- Serving e2e：报告 TTFT、TPOT/inter-token latency、完成请求的总耗时、generated tokens/s 和 p50/p95 等分位数；以实际提交的 token 数计算 speculative 吞吐。
- 固定 token 回放用于比较相同工作负载；自由生成补充真实负载表现。输出分叉可能改变 MoE routing、长度和 speculative acceptance，不能把这种工作量差异误记为纯 kernel 加速。
- 编译/权重转换/加载作为 startup 单独记录；稳定运行先预热，多次测量。完整计时应保留通信、调度、cache 管理和必要同步。logits 拷回、逐层 dump 和 comparator 仅在正确性运行启用。
- 最终 speedup 使用相同边界下的完整耗时比 `T_baseline / T_optimized`。单算子 speedup 或各 case 耗时相加只能辅助分析；融合、通信重叠和 host 调度都会改变其对 e2e 的贡献。

建议交付顺序是：现有 case 的局部正确性 → 相同配置的完整单层 → 真权重 43 层 teacher-forced logits 与连续 cache 验证 → AR e2e 性能 → prefix cache / MTP / DSpark。缩减 synthetic 链路可保留为快速 smoke gate，但须单独命名和报告。

**6. 已核实的替换调用点**

以下是源码级可接入证据，尚未完成 adapter 编译或真实权重运行认证。优化实现应进入 serving 实际编译的 kernel，保持上层请求与状态协议；Case 的测试 wrapper、固定 fixture 和 evaluator 不进入生产调用链。

| Case | serving 实际调用位置 | 需要的工程适配 |
| --- | --- | --- |
| `deepseek_v4.rmsnorm_decode_flash` | `DeepSeekV4PyptoExecutor` 编译 `decode_fwd.l3_decode_fwd`；其 CSA 和最终 norm 调用 `rmsnorm.rms_norm` | 两侧参数均为 `x[T,D] BF16`、`norm_w[D] BF16`、`x_normed[T,D] BF16`。Case 返回 Tensor，serving 绑定 kernel 返回 producer TaskId，CSA 还通过 `task_dummy(deps=[rms_tid])` 建立依赖；替换必须保留这个调度协议 |
| `deepseek_v4.prefill_attention_{swa,csa,hca}_flash` 及其压缩/indexer/attention leaf | `prefill_fwd.l3_prefill_fwd` 的对应 attention 阶段；decode 同理进入对应 `decode_{swa,csa,hca}` | 适配动态有效 token、cache/state 布局、RoPE 数据、量化与完成信号；完整 attention/subgraph 的改造量大于 RMSNorm |
| `qwen3_14b.topk_select_k32` | `Qwen314BPyptoExecutor._compile_model()` 导入 `topk_select`，编译 `qwen3_topk_select_host`，并由 runner 处理 prefill/decode 的候选集 | 同一 `topk_select_fwd(logits, sampling_control, topk_values, topk_indices)` 语义；固定 `BATCH=16` 与部署 `BATCH_PAD`、有效 batch、真实词表 mask、padding 和 tie-breaking 需要对齐 |
| `qwen3_14b.token_embed` / `greedy_sample` | serving 绑定的 Qwen `decode_fwd` 中 `_token_embed_inline` / `_greedy_sample_inline` | 替换融合 forward 内部实现；保留输入 token 语义、采样输出和动态 batch 约束 |
| `qwen3_14b.qwen3_14b_decode` / `prefill_fwd` 等子图 | Qwen 的完整 `decode_fwd` 内部 `_decode_layer` 或 prefill 层循环 | case 入口不是现成的完整 serving HOST ABI；需要迁移计算体或包装成完整层，并保留 paged KV 与 residual/normalization 的语义 |

按 serving 绑定的 pypto-lib commit 读取源码，并做 AST 对照：Qwen top-k 的 `_topk_group_pairs` 在将 `BATCH_PAD` 统一命名为 `BATCH` 后完全一致；主入口存在临时 Tensor 构造等实现差异。Flash RMSNorm 的三个 Tensor 参数注解一致，返回值确实不同。这些证据确认了具体替换位置，不能替代编译、数值和性能验收。

Raw logits 对 Transformer 主干、final norm、LM head 的数值回归很合适，但 top-k/greedy/RNG 位于 logits 之后：这些 case 的优化还必须比较候选值/索引、token IDs、tie-breaking 或随机分布。只比较 raw logits 无法检出 sampler 错误。

诊断模式建议直接调用 runner 或增加专用 replay harness。Flash AR 设置 `allow_device_greedy_sampling=False` 可返回 logits；Qwen decode 应同时关闭 device greedy 和 device top-k sampling，才会走全词表 logits readback。两边按同一 token 前缀连续演进各自的 cache。性能模式重新关闭 readback 与 comparator。

**7. 支持模型与本机是否能运行**

本次 `npu-smi info` 实测：8 张 Ascend 910B4，设备 0–7，每卡 HBM 32768 MiB，驱动 25.5.0；主机内存约 1.5 TiB。本节只评估当前软件路径与容量，没有启动 NPU 作业。

| 模型 | 所核查 serving 的 NPU 支持 | 本机判断 |
| --- | --- | --- |
| Qwen3-14B | 有完整 prefill/decode/LM head/sampling；shape 验证固定 hidden=5120、intermediate=17408、Q heads=40、KV heads=8、head_dim=128，并核对 40 层与词表 | BF16 主矩阵估计约 27.5 GiB；32 GiB 卡有短上下文、受控 scratch/cache 的可行空间，但不是已实测通过。优先跑小请求、短上下文基线，再看 memory admission 日志 |
| DeepSeek V4 Flash W8A8，AR/MTP | 有完整路径；固定 43 层 Flash shape，8 devices，DP8/EP8，外部 TP1 | 当前 resident W8A8 路径装不下；调小 batch/context 不能消除权重容量问题 |
| DeepSeek V4 Flash DSpark | 有 target-only/K=7 路径；固定 16 devices，TP4/DP4/EP16 | 本机只有 8 卡，不满足当前拓扑约束 |
| Qwen3-32B、V3.2、V4.1、V4 Pro | PTOBench 中有相关 case，但当前 serving NPU 集成没有这些完整模型路径；Qwen/Flash executor 还会拒绝不匹配 shape | 需要新增模型适配与独立容量评估，不能从 case 存在推导为已支持 |

Flash 的确定性容量下界来自实际权重规则：每个主干层有 256 个 routed experts，每个 expert 有三块 INT8 矩阵，尺寸为 `2048×4096`、`2048×4096`、`4096×2048`；权重按 EP 分片并常驻设备。

```text
43 × 256 × 3 × 4096 × 2048 × 1 byte = 258 GiB
258 GiB / EP8 = 32.25 GiB per device > 32 GiB
```

可复核的源码链：本地 [Flash 配置](../../../models/deepseek_v4_flash_mtp/config.py) 给出 43 层、256 个路由专家、hidden=4096、expert intermediate=2048；serving 的 [weight_spec.py](https://github.com/hw-native-sys/pypto-serving/blob/9e16e45098cc324e9aa48f904906ca5fcc58b810/pypto_serving/model/deepseek/weight_spec.py#L258) 将 routed w1/w3/w2 明确声明为 `torch.int8`，并通过同文件的 `deepseek_v4_stack_groups()` 将所有主干层纳入堆叠权重；[npu_runner.py](https://github.com/hw-native-sys/pypto-serving/blob/9e16e45098cc324e9aa48f904906ca5fcc58b810/pypto_serving/model/deepseek/npu_runner.py#L4373) 的 `_upload_weight_group()` 对每个堆叠权重调用 `worker.alloc_stacked_tensor()`，由 `_materialize_resident_weights()` 一次性上传主干权重。这是权重容量下界与当前执行路径的证据，不是 OOM 实测记录。

原始 Flash checkpoint 中的 MXFP4 专家不能按这个 INT8 容量直接判断：[serving 文档](https://github.com/hw-native-sys/pypto-serving/blob/9e16e45098cc324e9aa48f904906ca5fcc58b810/docs/user-guide/deepseek-v4.md#L15) 说明当前路径先将 FP8/MXFP4 转换为 W8A8。若专家矩阵保留 packed 4-bit，相同矩阵的原始数据量是 129 GiB（未计 scale 和其他分配）；这说明更低比特路径需要单独评估，不能从 W8A8 下界推出完整 Flash 在本机绝对不可运行。

这里尚未计入 scale、attention、shared experts、embedding/LM head、通信、runtime scratch、KV cache 或 MTP。实际 top-k 只激活部分专家，不等于当前 resident 实现只加载被激活专家。主机内存充足也不能让现有 device-resident 路径自动变成 CPU offload。8×64 GiB 级别设备是更合理的普通 Flash 起点，仍需完整内存准入与实测；CPU offload/更低比特执行或改造多机拓扑属于额外系统工程。

当前 `.venv-official` 中 Torch/PyPTO/simpler 可定位，但 `pypto_serving`、`transformers`、`safetensors` 未安装；在仓库模型目录与本人常规 Hugging Face cache 路径未找到 checkpoint，不能据此排除其他目录或用户持有权重。当前 `PYPTO_LIB_ROOT` 的 Qwen 在 `models/qwen3/14b`，所核查 serving 查找 `models/qwen3_14b`；因此要使用匹配 submodule checkout 或显式做路径/API 适配。建议为完整 serving 建独立、匹配的环境，不直接将现有 PTOBench 环境当作已验证的 serving 环境。
