# 原 S6 canonical hash 的 42 条变化路径

本清单来自原 S6 验收前后的完整字节 SHA-256 比较，描述那个时间段的差异；不是重新扫描当前仓库。原始证据：`../S6_evidence/canonical_comparison.json`。

共 41 个已有文件发生字节变化，新增 1 个环境文件；38 个为同一 Case 的 Fixture 输入 `.pt`。这份 canonical 检查仅覆盖 cases、sealed_assets、environments，不用于证明公共 evaluator/launcher 在升级过程中未修改。

| 类型 | 路径 |
| --- | --- |
| 修改 | `cases/deepseek_v4/prefill_layer_ep2_csa_pro/case.json` |
| 修改 | `cases/deepseek_v4/prefill_layer_ep2_csa_pro/public/config.py` |
| 新增 | `environments/a2a3_prefill_layer_ep2_csa_pro_repair_idle67.json` |
| 修改 | `sealed_assets/deepseek_v4/prefill_layer_ep2_csa_pro/fixture_abi.json` |

38 个输入文件均位于：

```text
sealed_assets/deepseek_v4/prefill_layer_ep2_csa_pro/fixtures/fixture_000/in/
```

```text
csa_cmp_ape.pt
csa_cmp_norm_w.pt
csa_cmp_wgate.pt
csa_cmp_wkv.pt
csa_idx_wq_b.pt
csa_idx_wq_b_scale.pt
csa_inner_ape.pt
csa_inner_norm_w.pt
csa_inner_wgate.pt
csa_inner_wkv.pt
csa_weights_proj.pt
gamma_cq.pt
gate_w.pt
hc_attn_fn.pt
hc_ffn_fn.pt
hca_cmp_ape.pt
hca_cmp_norm_w.pt
hca_cmp_wgate.pt
hca_cmp_wkv.pt
routed_w1.pt
routed_w1_scale.pt
routed_w2.pt
routed_w2_scale.pt
routed_w3.pt
routed_w3_scale.pt
shared_w1.pt
shared_w1_scale.pt
shared_w2.pt
shared_w2_scale.pt
shared_w3.pt
shared_w3_scale.pt
wkv.pt
wo_a.pt
wo_b.pt
wq_a.pt
wq_b.pt
wq_b_scale.pt
x_hc.pt
```

其中大小发生变化的输入文件：

| 文件 | 验收前字节数 | 验收后字节数 |
| --- | --- | --- |
| `csa_idx_wq_b.pt` | 25,167,029 | 16,778,421 |
| `gamma_cq.pt` | 7,329 | 5,281 |
| `wq_a.pt` | 25,166,925 | 16,778,317 |
| `wq_b.pt` | 201,327,693 | 134,218,829 |

其余输入文件大小相同，但字节 SHA-256 不同。仅凭此检查不能判断所有 tensor 数值是否变化；`.pt` 序列化字节变化也会改变 hash。

上述 `public/config.py` 是该 Case 自身的配置。S6 公共层的计划内修改另见 `../S6_evidence/public_source_changes.json` 和 S6 报告“公共实现与回归修复”；pi/OpenCode 接入修改见本目录的 provider 报告。
