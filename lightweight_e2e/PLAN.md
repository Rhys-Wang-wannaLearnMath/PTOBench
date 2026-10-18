# Lightweight e2e execution：优化集成与正确性检查初步规划

日期：2026-10-16。本文记录源码核查后的实施规划；尚未实现或实测下述多步 e2e harness。

## 1. 目标与起点

建立可持续输出 token 的轻量推理链路，把 PTOBench case 中的优化迁入其中，检查优化前后的数值正确性和整体耗时。优先完成 DeepSeek V4 Flash，再复用框架支持缩减的 Pro。

本机为 8×Ascend 910B4，每卡 32 GiB HBM。当前 pypto-serving 的完整 Flash W8A8 常驻路径，仅路由专家矩阵就需每卡 32.25 GiB，超出容量。本规划采用缩减配置；其结果代表该轻量链路，不直接代表完整 Flash/Pro 的速度或生成质量。

| 方案 | Prefill case | Decode case | 已有固定配置 |
| --- | --- | --- | --- |
| Flash，第一阶段 | [synthetic_token_loop_prefill_flash](../cases/deepseek_v4/synthetic_token_loop_prefill_flash/case.json) | [synthetic_token_loop_decode_flash](../cases/deepseek_v4/synthetic_token_loop_decode_flash/case.json) | 7 层、EP2/TP2、hidden=4096、64 个有效路由专家、MoE intermediate=2048 |
| Pro，第二阶段 | [synthetic_token_loop_prefill_pro](../cases/deepseek_v4/synthetic_token_loop_prefill_pro/case.json) | [synthetic_token_loop_decode_pro](../cases/deepseek_v4/synthetic_token_loop_decode_pro/case.json) | 7 层、EP2/TP2、hidden=7168、192 个有效路由专家、MoE intermediate=1024 |

上述配置以各 case 的 `public/config.py` 为准。Flash 保留 SWA/CSA/HCA，Pro 保留 HCA/CSA。仅 INT8 路由专家矩阵分别约为 5.25 GiB/卡和 13.78 GiB/卡；总内存还包含其余权重、cache、通信和 scratch，需实际检查。

四个 case 都输出 `logits` 和 `sampled_ids`。当前 `logits` 为 FP32，形状均为 `[2, 8, 129280]`，词表分片已汇集；有效输出行由 `logit_row_indices` 选择，并非全部 prompt 位置。

## 2. 轻量 session 如何运行

执行链路为：

```text
prompt token IDs → embedding → 7 层 prefill → LM head → logits → greedy token
                                                                     ↓
                同一组权重、持续保留的 cache ← 连续 decode ← 上一步 token
```

Prefill 最后一个有效位置的 logits 预测第一个输出 token；decode 消费刚选出的 token，更新 cache，再预测下一个 token。模型先对全部候选 token 打分，这组分数就是 raw logits；sampling 再根据分数选 token。

这些 case 的 LM head 先生成 logits，再执行 greedy sampling。Sampling 只读 logits，运行完成后仍能获取采样前的数值；示例见 [lm_head_with_sampling](../cases/deepseek_v4/synthetic_token_loop_decode_flash/public/candidate_seed/lm_head.py)。

实施时增加独立 session harness：

1. 编译并加载同一配置的 prefill/decode，校验共享权重和 cache 的 shape/dtype/布局。
2. 权重上传一次，cache 初始化一次。起步使用一个有效请求，正确屏蔽其余 padding 行。
3. Prefill 后读取有效 token ID；每步更新输入 token、绝对位置、有效行和分页 metadata，复用同一请求的 cache。
4. 正确处理重复 dispatch 的通信信号清理或 epoch 递增；与模型 cache 的生命周期分别管理。
5. 保存 token 序列，可用匹配 tokenizer 解码；达到长度限制或 EOS 后结束。

参考 [synthetic_token_loop.py](../../../models/deepseek_v4_pro/synthetic_token_loop.py) 的 resident 权重/cache 管理和循环逻辑。该脚本当前入口限定 A5，Flash 采用完整层数，不能直接视为这组 7 层 case 在 910B 上的现成入口。需要适配 A2/A3 工具链、编译/runtime ABI 和缩减配置，并检查实际调用兼容性。

[PTOBench 当前规范](../README.md) 是每个 case 执行单步，且上述 case 会在每次 launch 前恢复初始状态。连续生成应由独立 harness 编排，不能通过反复调用 `ptobench evaluate` 实现。集成使用 case 实现的独立副本/adapter，保留 canonical case 的固定配置、Fixture 和评测语义。

## 3. 优化如何接入

Baseline 与 Optimized 使用同一 harness、权重、拓扑和工作负载，仅切换计算实现。先在原 case 内验证优化，再迁入轻量 forward，保持数学语义、dtype、布局、融合边界和调度依赖。

| 接入顺序 | 对应优化点 |
| --- | --- |
| 1 | RMSNorm、HC pre/post、HC head，验证替换和回归流程 |
| 2 | QKV/RoPE、压缩器、indexer、SWA/CSA/HCA attention |
| 3 | Gate、共享/路由专家、dispatch/combine、完整 MoE |
| 4 | 完整 attention/layer、LM head 及跨组件优化 |

先逐项替换定位影响，再组合已通过的优化。Layer 和其内部 leaf 是包含关系，不能把它们当作独立串联节点，也不能将各 case 的耗时相加当作 e2e 耗时。

第一版数值检查覆盖到 raw logits，sampling 保持既有实现。后续若优化 greedy/top-k/RNG，需补充对应 token ID、索引或分布检查，因为仅比较 logits 无法检出采样器错误。

## 4. 最简正确性检查：固定前缀，比较逐步 raw logits

第一版采用数值回归：Baseline 为优化前实现，Optimized 为集成优化后的实现。比较相同输入条件下的 logits；完整逐层 dump 和 cache 数值对比作为失败后的定位手段。

### 输入与权重

- 固定模型配置、权重、量化参数、prompt token IDs，以及一段保存的续写 token IDs，例如 32 个。
- 现有零权重保留为链路冒烟检查。数值验收另外使用固定种子生成且满足量化/结构约束的非零权重，或一致裁剪的真实权重；两边和 prefill/decode 必须共用相同权重内容。
- 两边各自从空 cache 开始，在各自的连续执行中保留和更新 cache。

### 比较流程

```python
# 伪代码：每个 state 包含该实现自己的独立 cache。
ref_state, ref_logits = baseline.prefill(prompt_ids)
opt_state, opt_logits = optimized.prefill(prompt_ids)
compare(valid_logits(ref_logits), valid_logits(opt_logits))

for token in fixed_continuation_ids:
    ref_state, ref_logits = baseline.decode(token, ref_state)
    opt_state, opt_logits = optimized.decode(token, opt_state)
    compare(valid_logits(ref_logits), valid_logits(opt_logits))
```

Prefill 比较每个请求最后一个有效位置的完整词表 logits；decode 比较每步对应位置的完整词表 logits。按请求和位置对齐有效行，排除 padding。续写 token 可以提前由 Baseline 生成并保存；测试时两边都回放这段 token，避免各自采样后前缀分叉。

### 判定规则

Logits 不要求逐 bit 完全一致。允许由 BF16、量化和计算顺序产生的小数值误差，每个有效元素满足：

```text
|optimized − baseline| ≤ atol + rtol × |baseline|
```

```python
def compare(ref_logits, opt_logits, *, rtol, atol):
    ref = ref_logits.float()
    opt = opt_logits.float()
    assert ref.shape == opt.shape
    assert torch.isfinite(ref).all()
    assert torch.isfinite(opt).all()
    torch.testing.assert_close(opt, ref, rtol=rtol, atol=atol)
```

`rtol/atol` 作为测试配置，在优化前按当前精度与允许的数值误差确定并冻结。初步规划暂不指定统一数值，不能直接把单算子容差当作 7 层 e2e 容差，也不能在某次优化失败后为其放宽门槛。

初始测试集使用单个有效请求、prompt 长度 `3、4、127、128`，每个 prompt 后连续回放 32 个 decode token，覆盖压缩和分页边界。每次记录 pass/fail；失败时记录 stage、decode step、请求/位置、最大绝对误差及对应词表索引。短测试通过后再扩展连续 decode 长度。

## 5. 性能对比与最小交付

正确性通过后，用相同固定 token 回放比较性能。每轮从相同空 cache 开始，轮内持续保留 cache；固定权重、有效请求数、prompt、decode 步数和拓扑。预热后重复测量，记录 prefill 延迟、平均 decode 步延迟、`prefill + N 次 decode` 的总耗时及其中位数，speedup 为 `T_baseline / T_optimized`。

计时包含实际 dispatch、通信及执行边界内的 metadata/cache 管理，并等待设备完成。权重加载、编译、正确性专用 logits 读回和 CPU 比较单独处理。基于相同边界对比，避免不同实现生成不同 token 后造成工作量变化。

实施顺序与交付：

1. Flash 冒烟：在 910B 上完成 prefill → 连续 decode → token ID，记录实际内存和运行环境。
2. Flash 数值基线：加入非零权重、固定续写 token 和冻结的 logits 容差，形成可重复的通过/失败报告。
3. Flash 优化 A/B：接入一项已通过局部测试的 case 优化，完成 logits 回归和整体耗时对比，再扩展组件。
4. Pro 扩展：复用 session 和比较器，按其独立配置重新检查容量、数值和性能。

每次运行保存：源码/工具链版本、Baseline/Optimized 标识、模型配置、设备拓扑、权重标识/种子、prompt/续写 token、`rtol/atol`、逐步比较结果及性能原始样本。

后续可增加 Qwen3-14B 的真实模型验证：PTOBench 有 12 个相关 case，pypto-serving 已有完整推理路径；其 BF16 主权重约 27.5 GiB，32 GiB 单卡的剩余空间有限，实际启动能力仍需验证。该方向可复用固定前缀的 logits 检查方法。
