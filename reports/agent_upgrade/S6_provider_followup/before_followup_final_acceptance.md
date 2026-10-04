# 最终验收初稿

日期：2026-10-03。仅完成 S6 本阶段可执行工作；**S7 全量 Case 验收尚未执行**。不能声明全部 v0.2 正确性和三种 profile 验收通过。

| 范围 | 当前结果 |
| --- | --- |
| 公共功能、CPU 回归、并发/故障和假进程闭环 | PASS：146 项 benchmark 回归，另 3 项迁移 bootstrap 校验 |
| RMSNorm / Compressor / 多文件 Attention 代表验收 | PASS：规定 clean/profile/独立 finish；两项五组测量等价性 PASS |
| 双设备 prefill_layer_ep2_hca_flash | BLOCKED：基线即缺 canonical Candidate Seed，三项未运行 |
| Codex / ClaudeCode 真实接入 | PASS：相同 Case/default profile、2/1 预算、实际调用、退出、自动复核与清理 |
| pi / OpenCode 真实接入 | BLOCKED：所选模型 API key 缺失 / 认证列表为空 |
| S6 指定 Case、环境与 helper 内容 | PASS：369 条目、4,662,156,790 字节内容 hash 相同 |
| 全库 canonical hash 相等 | FAIL：其他 Case 并发修缮产生 42 个路径差异，保留原始证据，未回退 |
| 原 tracked diff 与历史报告 | PASS：原样保留 |
| 旧接口移除及当前说明/skills/维护工具 | PASS |
| 全部 v0.2 Case 的 clean/runtime/PMU/compiler | NOT_RUN：属于 S7 |

S6 总体为 BLOCKED，公共层通过、后端认证、Case 发布缺项及全库并发 hash 差异分别记录。没有为了通过修改 Case、回退其他任务资产或调宽性能阈值。

详见 [S6 完整报告](S6_integration_acceptance.md)、[硬件矩阵](S6_evidence/hardware_matrix.json)、[内容 hash 比较](S6_evidence/canonical_comparison.json) 与各真实 Session 原始证据。后续 S7 依据 `Benchmark升级/plan/09_full_v02_acceptance.md` 执行实际全量盘点和逐 Case 验收，并更新本初稿。
