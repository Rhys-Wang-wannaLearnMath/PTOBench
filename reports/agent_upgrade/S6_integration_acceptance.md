# S6：公共层回归与真实集成验收

日期：2026-10-03。范围仅 S6，依据 `docs/10_stage_prompts.md` 的共同约束，以及 `Benchmark升级/plan/10_cli_cutover.md`、`07_tests_and_acceptance.md`、`08_execution_steps.md`。S0～S5 报告及已有本地修改保留。**S6 阶段：PASS（公共层升级与集成验收结项）**；公共层和三个代表 Case 已通过，双设备 Case 缺少已发布 Seed；pi/OpenCode 的认证阻塞已通过本次 provider 补充验收解决。另有其他 Case 的并发修缮导致全库 hash 相等检查 FAIL，S6 指定 Case 和环境内容均相同。没有执行 S7，也没有声明全部 v0.2 Case 通过。

## 验收结论

| 范围 | 结论 | 证据 |
| --- | --- | --- |
| 完整 benchmark CPU 回归 | PASS：146 项 / 335.603s | [最终原始日志](S6_evidence/full_regression_final.log) |
| 并发、故障与假进程闭环 | PASS | 完整回归含预算竞争、跨 Session 锁、遗留进程、坏恢复源、冻结源码、Snapshot、全部 null、全失败、耗尽及中断；[维护执行测试](S6_evidence/maintenance_tests.log) |
| 迁移只读 bootstrap 的新 CLI 禁用规则 | PASS：3 项 | [原始日志](S6_evidence/bootstrap_cutover_tests.log) |
| 三个单设备代表 Case | PASS | [运行矩阵](S6_evidence/hardware_matrix.json)、[诊断与权威结果校验](S6_evidence/hardware_checks.json) |
| 指定双设备 Case | BLOCKED | 认证 ready，但 canonical `public/candidate_seed` 从基线起即缺失 |
| 五组交错配对等价性 | PASS：两个指定 Case | 固定公式与每轮原始 samples 已保存 |
| Codex / ClaudeCode 真实后端 | PASS | 实际 Optimizer CLI、ptobench、NPU evaluate、自动 finish 和清理 |
| pi / OpenCode 真实后端 | PASS：三个指定模型各通过 | [Provider 补充验收](S6_provider_followup/provider_acceptance.md)；六组真实 NPU evaluate/finish |
| S6 指定 Case、环境及 family helper 内容 | PASS | 369 条目、4,662,156,790 字节完整 SHA-256 相同 |
| 全库 canonical hash 相等 | FAIL：并发修缮变化 | 11609 → 11610 条目，42 路径变化；均属其他 Case 及其新环境 |
| 原 tracked diff 与历史报告保护 | PASS | 与原基线相同；未恢复其他任务的 canonical 变化 |
| 旧入口、旧语法、无 Session 回退与当前文档 | PASS | [清理审计](S6_evidence/legacy_audit.json) |
| S7 全量 Case 验收 | NOT_RUN | 本任务不包含 S7；代表 Case 结果不能替代全量验收 |

## 公共实现与回归修复

`evaluator/session_actions.py` 增加内部 `run_maintenance_process(argv, environment, action_dir)`，要求调用者已持全局 execution lock，使用全新 `runs/.../workspace`。直接对照复用同一个 `_run_process`，没有第二套测量 Runner、旧命令转发或 Agent 维护 CLI。维护执行不扣 Session 预算、不发布正式分数。它共享 Evaluator 启动 gate、双路日志、PID 出生身份、持久全局 marker、TERM/KILL/reap；父进程被强杀后，下一次管理 action 先清理旧 group。无归属 PID 或无法确认清理时仍阻止后续硬件运行。

首轮完整回归出现一次 `/proc/<pid>/stat` 读取竞争：进程在扫描时退出，内核返回 ESRCH，旧代码将其作为 infrastructure_error。[失败原始日志](S6_evidence/full_regression.log) 保留。`session.process_identity` 现在将 ProcessLookupError 与 ENOENT 一样视为进程已不存在；权限等其他读取错误仍失败。新增[针对性测试日志](S6_evidence/process_identity_tests.log)，最终完整回归全部通过。公共修复在三个代表 Session 完成后应用，避免活动 Session 的固定文件恢复与升级冲突；随后两个真实后端以修复后的公共版本完成 NPU 闭环。没有改变 RunConfig、测量区间、warmup 或 rounds。

`test_launcher.py` 新增真实子进程顺序 `evaluate → fork → 修改 helper → profile → evaluate → exit`，验证事件次序、不同 helper hash、冻结 Snapshot、并列选更早、独立 finish 的新数值；另覆盖全 null profile 和全失败/保护误改无候选。现有真实子进程用例继续覆盖四个 Adapter、提前退出、耗尽等待最后 action、中断、残留 worker 和坏备份。`test_maintenance_execution.py` 新增 4 项锁/中断/强杀恢复测试，`test_process_identity.py` 新增 2 项进程竞争测试。

Comparator 的公共测试适配在 S6 前已完成，保留其变化；本阶段没有修改 comparator 或 Case。进入阶段前 [137 项回归](S6_evidence/readiness_137_tests.log) 已通过；最终完整回归中的 comparator 两项也通过。既有三个非 ready `deepseek_v4_flash_mtp.prefill_cp_zigzag_cp2_{demo,flash,pro}` 缺少 Case-local evaluator，仍按原测试明确记录 NOT_RUN，没有为此生成 canonical 文件或隐藏真实缺项。历史 S5 失败报告没有改写。

## 实际 NPU 结果

| Case | 初始 clean | 规定 profile | 独立 clean finish | 预算与限制 |
| --- | --- | --- | --- | --- |
| `deepseek_v4.rmsnorm_decode_flash` | PASS | PASS：runtime / pmu / compiler | PASS：17.5900 µs | 探索 8，finish 1 |
| `deepseek_v4.compressor_ratio4_flash` | PASS | PASS：runtime / pmu / compiler | PASS：543.5600 µs | 探索 8，finish 1 |
| `deepseek_v4.decode_attention_hca_flash` | PASS | PASS：runtime / pmu / compiler | PASS：1125.7505 µs | 探索 4，finish 1 |
| `deepseek_v4.prefill_layer_ep2_hca_flash` | BLOCKED | BLOCKED：compiler | BLOCKED | canonical Seed 缺失；0 次硬件 action |

前三个 Case 全部使用官方 `fixture_000`、canonical comparator、环境 `a2a3_single_card`、device 0、warmup 5、rounds 10。直接和管理调用源码 hash 相同。每次管理调用执行真实 `ptobench`；direct 由已持锁共享工具调用现有 `evaluate.py`。三者的全部 action correctness/integrity PASS，最终结果对应 `final/workspace` 的权威 clean 结果，保留 10 个正有效微秒 samples，finish 恰为 1，无残留 active action。Attention 的完整 6 文件 helper 闭包通过冻结源码和 final Snapshot hash 校验。

诊断检查解析实际 native 产物：runtime 的 chip swimlane、merged trace、deps 和 scope JSONL；PMU 的非空 `pmu.csv` 和列结构；compiler 的非空 pass dumps 与 pipeline profile JSON。三者各模式直接/管理 schema 一致，允许 PID、时间戳、目录及动态数值变化。每种 profile 固定 correctness phase；没有补跑性能或将 null latency 改成数字。详细路径和 schema 在 [hardware_checks.json](S6_evidence/hardware_checks.json)。

`prefill_layer_ep2_hca_flash` 的 certification/ref_output_cache 均 ready，Fixture/Ref 存在，但 Case 指向的 `public/candidate_seed` 不存在。[baseline](S6_evidence/canonical_before.json) 已记录这一缺失，属于 Case 迁移 Stage 4 发布缺项，不属于 S6 公共实现。Session preflight 拒绝，没有扣硬件预算；clean、compiler 和 finish 三项分别 BLOCKED。环境的 device_ids 原样为 `[0,1]`，两张设备健康检查 OK；没有降为单设备或改用其他 Case，也没有从 Ref 临时拼装 Seed。

## 五组配对与固定阈值

验收前保存 [equivalence_formula.json](S6_evidence/equivalence_formula.json)：每个 Case 按 direct clean → managed clean 交错五组。D/M 是五个各轮 median_effective_us 的中位数；MAD_D 是直接值对 D 的绝对偏差中位数；allowed = max(0.10, 3 × MAD_D / D)；abs(M−D)/D ≤ allowed 才 PASS。没有事后调阈值、删除轮次或增加挑选样本。

| Case | D (µs) | M (µs) | MAD_D (µs) | allowed | 实际差异 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| `deepseek_v4.rmsnorm_decode_flash` | 17.8295 | 17.3495 | 0.1605 | 10.00% | 2.69% | PASS |
| `deepseek_v4.compressor_ratio4_flash` | 543.8890 | 542.2795 | 3.0495 | 10.00% | 0.30% | PASS |

全部 20 份配对 clean 结果保留原始 `samples_effective_us`，各 10 个样本；矩阵记录每组 median、源码 hash、权威 result 和 artifact 路径。

## 真实 Optimizer

两个已认证后端使用同一 `rmsnorm_decode_flash`、同名 default profile、探索上限 2 和 finish 上限 1。S6 harness 仅将 Launcher 的任务文字改为“实际 evaluate 一次后退出”；其余是真实公共 runner、真实 Adapter argv、原 default profile、外部 Optimizer CLI、ptobench 和 NPU Evaluator，没有 mock、工具替换或 provider 替换。实际 argv/cwd/profile hash/prompt 位于各 Session 的保护记录。profile 未指定 model 时如实记录 unknown。

| Backend | 实际版本 | 结论 | 行为 / 阻塞 | Session |
| --- | --- | --- | --- | --- |
| codex | codex-cli 0.159.2 | PASS | 2/1 上限，实际 1/1；`evaluate → finish`；进程组存活=false | `/data1/home/ziruiwang/pypto-lib/benchmarks/pypto_lib_bench-v0.2/runs/s6_real_codex/deepseek_v4.rmsnorm_decode_flash/20261003T065920Z_4be81944da69` |
| claudecode | 2.1.288 (Claude Code) | PASS | 2/1 上限，实际 1/1；`evaluate → finish`；进程组存活=false | `/data1/home/ziruiwang/pypto-lib/benchmarks/pypto_lib_bench-v0.2/runs/s6_real_claudecode/deepseek_v4.rmsnorm_decode_flash/20261003T070102Z_75a7c15206ea` |
| pi | 1.0.0 | PASS（补充验收） | 三个指定模型各完成 evaluate/finish，实际 1/1，正常退出和清理 | [六组矩阵](S6_provider_followup/acceptance_matrix.json) |
| opencode | 1.18.34 | PASS（补充验收） | 三个指定模型各完成 evaluate/finish，实际 1/1，正常退出和清理 | [六组矩阵](S6_provider_followup/acceptance_matrix.json) |

原始输出：[Codex](S6_evidence/real_codex.log)、[ClaudeCode](S6_evidence/real_claudecode.log)。机器结果：[Codex](S6_evidence/real_codex_result.json)、[ClaudeCode](S6_evidence/real_claudecode_result.json)。[认证就绪检查](S6_evidence/backend_readiness.json) 只保存退出状态与脱敏原因，没有认证原文、凭据或代理值。这是接入前的历史认证检查。补充验收现已复用指定 Codex profile 的 provider 与现有认证，pi/OpenCode 的三个指定模型全部通过真实调用和 NPU 闭环；详见 [Provider 补充验收](S6_provider_followup/provider_acceptance.md)。

## 内容保护与旧调用清理

在硬件执行前及全部硬件/后端执行后，对 `cases/`、`sealed_assets/`、`environments/` 全部文件完整读取做 SHA-256，包括大型 `.pt`；没有使用 size/mtime 代替内容 hash。两次都没有观察到文件在单次读取期间变化。**全库不相同**：11609 → 11610 条目，296,508,675,884 → 296,424,788,313 字节，共 42 个路径变化，集中在其他任务的 `deepseek_v4.prefill_layer_ep2_csa_pro` Case/config/Fixture/ABI，以及新 `a2a3_prefill_layer_ep2_csa_pro_repair_idle67.json`。检查期间观察到另一个 Python 进程正在处理该 Case；该进程随后退出，保留 PID/程序/Case 标识，不保存完整 argv 或环境。[作用范围与并发证据](S6_evidence/canonical_scope_and_concurrency.json) 和[全库差异](S6_evidence/canonical_comparison.json) 如实保留，不能把全库相等写成 PASS，也没有回退其他任务的新资产。

S6 的四个指定 Case、两份所用环境和 family helper 共 369 条目、4,662,156,790 字节，前后内容完全相同，差异为零。[前日志](S6_evidence/hash_before.log)、[后日志](S6_evidence/hash_after.log) 保留。现有 tracked 用户 diff 与初始快照相同；S0～S5 报告 hash 未变，见 [preservation_check.json](S6_evidence/preservation_check.json)。S6 控制程序只执行固定四个 Case，CPU 用例只在临时资产上做误改/恢复；本次动作没有改写 canonical Case、Seed、Fixture、Ref、Oracle、Case-local evaluator、comparator 或环境。

清理范围如下；逐文件列表见 [legacy_cleanup_files.json](S6_evidence/legacy_cleanup_files.json)，扫描及实际拒绝结果见 [legacy_audit.json](S6_evidence/legacy_audit.json)。

| 当前有效文件 | 变更 |
| --- | --- |
| benchmark/evaluator/Launcher README，workspace 模板 | 新 CLI、自动 finish、诊断副本、共享持锁 maintenance control 及报告链接 |
| workflow 的 Ref cache 错误提示 | 使用真实 `prepare_ref.py --case` 维护指引 |
| publish / verify / implement / repair / migrate 五个 v0.2 skill | 创建 fresh Session、ptobench public modes、独立 finalize；certification 明确维护 Python 参数 |
| mig_workflow README、contract、Stage 3/Repair/4/5 说明 | 删除旧可执行调用，区分 public mode 与 internal artifact name；保护结果位置与 fresh final workspace |
| 新 `mig_workflow/SESSION_ACCEPTANCE.md` | 可复现 fresh Seed Session recipe，4/1 预算，权威结果与 diagnostics 检查，以及实际 certification 参数 |
| `validate_bootstrap.py` | 禁止只读 bootstrap 调用新 evaluate/profile/fork、Launcher run 与维护硬件脚本；保留对历史旧命令的拒绝检测 |
| docs stage prompts，Benchmark升级 README 与 plan README/steps/cutover | 同步实施状态、新入口与 S6 验收结果；S0/S1 历史任务和证据保留 |

retired `bench` 与 `evaluator/bench.py` 的文件/链接都不存在，Agent parser 只有 evaluate/profile/fork，不增加旧 parser、转发或无 Session 路径。CPU 套件验证旧 modes/参数、未注册 workspace、缺 Session 拒绝且不执行、不扣预算、不发布 Snapshot；另有 6 个当前真实 CLI 拒绝探针。历史日志、阶段报告、COMPARATOR_AUDIT 和周报保留原样，不作为当前可用命令。

## 可复现证据与后续

关键命令见 [commands.txt](S6_evidence/commands.txt)。S6 控制程序为 [accept_s6.py](S6_evidence/accept_s6.py)，只包含固定四个代表 Case；直接命令保存于 `runs/s6_controls/<case>/<action>/command.json`，双路日志/result/identity 相邻；managed 的命令和输出在 `S6_evidence/managed_*`，冻结结果、Snapshots 和 final 仍在各新 Session 中。公共源文件基线与补丁在 [source change index](S6_evidence/public_source_changes.json) 和 source zip 中。没有执行 S7 工具或全量硬件 Case 遍历。

结项口径：按用户确认，缺 Seed 的双设备 Case 保留 BLOCKED 记录，不要求修复该 Case 后才能结束公共层升级。42 条 canonical 差异全部属于其他 Case 资产及新增环境，原全库 hash 相等检查仍为 FAIL；S6 所选四个 Case 与所用环境内容相同。公共回归、可运行代表 Case 和四个真实 backend 均已通过，因此 S6 公共层升级与集成验收结项为 PASS。Case 级缺项和原始差异不被改写为通过，见 [结项决定](S6_provider_followup/stage_decision.json)。[final_acceptance.md](final_acceptance.md) 是初稿，S7 全量 clean/runtime/PMU/compiler 尚未执行。
