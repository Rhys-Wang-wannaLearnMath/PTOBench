# S7：全部 v0.2 Case 验收

日期：2026-10-04。仅执行 S7。

**记录结论：验收已完整执行并记录。全部项目通过：False。**

冻结清单 198 个 Case；记录 792/792；四项全部 PASS 的 Case：157。FAIL、BLOCKED、UNSUPPORTED、TIMEOUT 均不算通过。

## 实现与前置

S6 公共层与真实硬件入口已通过。新增 `evaluator/accept_all_cases.py`、`acceptance_artifacts.py` 及 15 项 CPU 控制测试；只增加维护者工具，不增加 Agent CLI，不改 Case、Seed、Fixture、Ref、Oracle、comparator、Case-local evaluator 或 RunConfig，不自动 prepare-ref。不使用旧 bench、Docker 或独立仓库。

完整公共 CPU 回归：159 项 / 418.046s / OK（含当时 13 项 S7 用例）；最终 S7 工具：15 项 / 102.774s / OK。S6 的九组真实 native profile 产物全部通过检查器复核。

[完整回归日志](S7_evidence/full_regression.log)、[最终工具测试](S7_evidence/control_tests_release.log)、[native 产物复核](S7_evidence/S6_native_artifact_recheck.json)、[实现与 hash](S7_evidence/implementation_manifest.json)。

## 实际批次

批次 `20261004_S7_01` 冻结全部 `cases/*/*/case.json`，包括 pending 和 ID 异常路径。读取所有资产实际内容做 SHA-256，包括大型 .pt；冻结公共代码 hash 和环境版本。就绪 Case 创建四个相同原始 Seed 的独立 Workspace，通过真实 ptobench 依次执行 clean evaluate、runtime、PMU、compiler。每项完成即保存日志、JSON、Session/event ID、正确性、完整性和产物检查。单命令超时 1800 秒，清理未确认时阻止后续执行。

基线读取错误：0；canonical 内容一致：True；公共代码内容一致：True；停止原因：None。

| 项目 | PASS | FAIL | BLOCKED | UNSUPPORTED | TIMEOUT |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 157 | 5 | 36 | 0 | 0 |
| runtime | 157 | 5 | 36 | 0 | 0 |
| pmu | 157 | 5 | 36 | 0 | 0 |
| compiler | 157 | 5 | 36 | 0 | 0 |

独立 finish 记录 162 份：{"completed": 157, "failed": 5}。它们不替代四项探索记录；全失败且无有效 Snapshot 时无法产生最终成绩。

## 数据与审计

- [完整清单与版本](full_v02/20261004_S7_01/inventory.json)
- [逐项详细结果](full_v02/20261004_S7_01/results.jsonl)
- [全部 Case 矩阵](full_v02/20261004_S7_01/matrix.csv)
- [机器汇总](full_v02/20261004_S7_01/summary.json)
- [批次报告与未通过项](full_v02/20261004_S7_01/report.md)
- [独立 finish](full_v02/20261004_S7_01/finalizations.json)
- [canonical hash 对比](full_v02/20261004_S7_01/canonical_comparison.json)
- [公共代码 hash 对比](full_v02/20261004_S7_01/code_comparison.json)
- [独立覆盖及证据审计](S7_evidence/batch_audit.json)
- [历史报告与原源码保留检查](S7_evidence/preservation_final.json)
- [原始命令和完整批次日志](S7_evidence/commands.txt)

独立审计通过：True；既有公共源码及 S0～S6 报告一致：True；原 tracked diff 一致：True。

## 未解决问题与后续处理

[逐项问题、原始错误和排查归属](S7_evidence/unresolved_issues.json) 保留每个非 PASS 项。资产/认证/ID 问题归 Case 迁移维护；设备/模式关闭归环境配置；运行和产物错误需依据原始日志排查公共加载协议、Case 适配及 runtime/toolchain，不能仅凭 FAIL 判为 Case kernel 错误。

`combine_flash/pro` 已观察到 Case-local evaluator 的 build_tensor_specs 延迟 `from config import ...` 失败；公共加载器的 config alias 为临时作用域。后续专项应核对两者的加载约定，决定公共作用域或 Case 适配修复。本阶段只保留错误，没有修改它们。

修复后用新的完整批次复测，保留本次失败证据；不能将不同版本的成功行拼成“全量通过”。

只有每个 Case 的四项都为 PASS 且内容 hash/覆盖审计通过，才声明全部 v0.2 验收通过。
