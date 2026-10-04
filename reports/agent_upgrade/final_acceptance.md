# 最终验收结果

日期：2026-10-04。S6 公共层与集成验收已通过；S7 验收已完整执行并记录。**全部 v0.2 Case 四项通过：False。**

| 范围 | 当前结果 |
| --- | --- |
| 公共功能、CPU 回归、并发/故障和假进程闭环 | PASS：146 项 benchmark 回归，另 3 项迁移 bootstrap 校验 |
| RMSNorm / Compressor / 多文件 Attention 代表验收 | PASS：规定 clean/profile/独立 finish；两项五组测量等价性 PASS |
| 双设备 prefill_layer_ep2_hca_flash | BLOCKED：基线即缺 canonical Candidate Seed，三项未运行 |
| Codex / ClaudeCode 真实接入 | PASS：相同 Case/default profile、2/1 预算、实际调用、退出、自动复核与清理 |
| pi / OpenCode 真实接入 | PASS：复用指定 profile provider，三个指定模型分别通过真实工具测试和 NPU evaluate/finish |
| S6 指定 Case、环境与 helper 内容 | PASS：369 条目、4,662,156,790 字节内容 hash 相同 |
| 全库 canonical hash 相等 | FAIL：其他 Case 并发修缮产生 42 个路径差异，保留原始证据，未回退 |
| 原 tracked diff 与历史报告 | PASS：原样保留 |
| 旧接口移除及当前说明/skills/维护工具 | PASS |
| 全部 v0.2 Case 的 clean/runtime/PMU/compiler | 验收已完整执行并记录；全部通过=False；792/792 项 |


S7 冻结 198 个 Case，四项均 PASS 的 Case 为 157；统计如下。

| 项目 | PASS | FAIL | BLOCKED | UNSUPPORTED | TIMEOUT |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 157 | 5 | 36 | 0 | 0 |
| runtime | 157 | 5 | 36 | 0 | 0 |
| pmu | 157 | 5 | 36 | 0 | 0 |
| compiler | 157 | 5 | 36 | 0 | 0 |

记录覆盖完整=True，独立审计通过=True，canonical 内容一致=True，公共代码一致=True。

S6 历史全库 hash 相等的 FAIL 仍保留：当时其他 Case 的并发修缮产生 42 条差异，不是本次 hash 结论。缺 Seed Case、pending 和运行缺陷均按 S7 原始记录保存，未自动修复资产。

[S7 完整报告](S7_full_v02_acceptance.md)、[批次矩阵](full_v02/20261004_S7_01/matrix.csv)、[逐项结果](full_v02/20261004_S7_01/results.jsonl)、[未解决问题](S7_evidence/unresolved_issues.json)、[S6 历史报告](S6_integration_acceptance.md)、[Provider 补充验收](S6_provider_followup/provider_acceptance.md)。
