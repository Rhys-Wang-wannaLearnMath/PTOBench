# S5 — Launcher、YAML 与四种 Optimizer Adapter

日期：2026-10-03；仓库：`/data1/home/ziruiwang/pypto-lib`。仅实施 S5，未实施 S6。

公共实现和假工具闭环已完成；Codex、ClaudeCode 的真实 NPU Session 均 PASS。
pi、OpenCode 已安装至 npm latest，但默认宿主认证未就绪，真实运行 BLOCKED。
全库回归 **FAIL**：一个既有 Comparator inventory 测试出现 21 个子测试失败，
涉及同时发生变化的 Case；S5 Launcher 测试没有失败。不能将本次全库回归写成 PASS。

## 依据与改动

已核对 `docs/10_stage_prompts.md` 共同约束、plan README、
`06_launcher_and_adapters.md`、`08_execution_steps.md`、CLI 切换决定和 S0～S4
报告/实际 API。复用 S1 文件保护、S2 Session、S3 action/process recovery、
S4 Snapshot 选择与 finalize；没有重写测量、workspace 或 Comparator 公共实现。

新增文件：`pto-launcher`、`launcher/{config,preflight,prompt,adapters,runner,main}.py`、
四个 `launcher/profiles/<backend>/default.yaml`、`requirements.txt`、
`examples/experiment.yaml`、`LAUNCHER.md`、`evaluator/tests/test_launcher.py`。
benchmark README 增加使用说明链接；根项目包名、发布结构与 Git index 不变。

- `run EXPERIMENT.yaml` / `status SESSION_ROOT` 是唯一实验入口。旧 `bench` 和
  `evaluator/bench.py` 不存在，preflight 也拒绝它们重新出现。
- YAML 使用 `yaml.safe_load`，严格检查 mapping、重复键、未知字段、ID、预算、
  命名 profile、managed 路径和目录重叠。默认 profile/output/budget 为
  default、YAML 相对 runs/snapshots、20/1；错误配置在创建 Session 前失败。
- benchmark 的局部依赖仅 `PyYAML>=6,<7`；已在 `.venv-official` 安装 **6.0.3**，
  Python **3.10.20**。测试保留 venv 解释器路径，避免 resolve symlink 后丢失依赖。
- preflight 只检查已准备 Case 资产、Python/CANN/PTOAS、目标设备健康、工具版本/
  参数及已有认证；不生成 Fixture、不准备 Ref、不执行测量。宿主 CANN 安装中的
  版本 symlink 先解析至实际脚本；canonical/输出路径继续使用 S1 no-follow 策略。
- 四个薄 Adapter 使用 argv 数组、`shell=False` 和独立 Optimizer 进程组。
  profile 可选择现有工具参数/模型，支持三个固定 placeholder，prompt 单独传参。
  profile 不接受凭据和 managed PATH/Session 变量覆盖；不保存宿主环境或认证原文。
- 自动创建 Session/worker_000、设置 cwd、PATH 和五个 PTOBENCH 身份变量。
  prompt 提供 evaluate、profile runtime/pmu/compiler、fork 五种操作，规定
  Candidate/helper 可编辑、context 不改、sealed 不读、共享预算、诊断用途及自动复核。
- protected 记录保存实际 argv、版本、profile 副本/hash、prompt、进程出生身份与
  Launcher 收尾结果；未指定模型时写明 `unknown`。stdout/stderr 双路保存/转发，
  `process.log` 合并记录；自然语言输出不参与正式成绩。
- 每秒检查 Session/active_action；最后一个预算 action 完成后才停止 Optimizer。
  正常/非零退出、中断会关闭探索、停止残留 worker/evaluator、确认停止/回收、
  恢复中断 action（不退预算）并调用 S4。TERM 宽限最多 10 秒，再 KILL。
  没有探索墙钟或无操作超时。权威记录失败阻止最终硬件复核，保留不可恢复证据。
- return 0/1/5/130 分别对应最终成功、普通失败、基础设施失败、用户中断；
  无效请求返回 2。最终复核期间的用户中断同样返回 130。status 在短 Session
  锁内读取一致的记录，仅校验/恢复身份、状态、额度和结果路径，不测量或 finalize。
  终端消费阻塞时仍保存完整日志并退出，JSON 输出复用 S3 的有界发送函数。

## 本机安装与参数核验

用户授权使用 `open_proxy` 安装工具，并要求检查三个新增工具都是最新版本。
执行 `open_proxy >/dev/null` 后使用 npm；没有输出代理或认证值。

```bash
npm install --global @anthropic-ai/claude-code @mariozechner/pi-coding-agent opencode-ai
npm view @anthropic-ai/claude-code version
npm view @mariozechner/pi-coding-agent version
npm view @earendil-works/pi-coding-agent version
npm view opencode-ai version
npm uninstall --global @mariozechner/pi-coding-agent
npm install --global @anthropic-ai/claude-code@latest @earendil-works/pi-coding-agent@latest opencode-ai@latest
```

旧 pi 包输出迁移弃用提示，故使用正在维护的 `@earendil-works/pi-coding-agent`。
安装后逐一重新查询 npm latest，并与实际 CLI 版本比较，三项均相同。
核验时间 UTC `2026-10-03T04:56:13.666870+00:00`。

| backend | 实际版本 | 默认非交互参数/配置 | 真实状态 |
| --- | --- | --- | --- |
| codex | codex-cli 0.159.2（原已安装） | exec、skip-git-repo-check、dangerously-bypass-approvals-and-sandbox | PASS |
| claudecode | **2.1.288 = npm latest** | -p、dangerously-skip-permissions | PASS |
| pi | **1.0.0 = npm latest** | --print；默认工具能力 | BLOCKED：所选模型没有 API key |
| opencode | **1.18.34 = npm latest** | run；OPENCODE_CONFIG_CONTENT permission=allow | BLOCKED：默认请求“未提供令牌” |

本机 `--help` 核对上述默认参数。OpenCode 官方配置 schema 的 PermissionActionConfig
包含 allow；安装版本实际接受默认配置，随后进入模型请求并因无 token 退出。
这证明配置可被该版本解析，不证明真实 shell/ptobench 执行已验收。
Codex login status、Claude auth status 的输出只保留认证布尔值/方法摘要。
OpenCode auth list 只保留 credential count=0；没有复制其认证输出或配置文件。

Codex 官方网页请求得到 HTTP 403，证据保留；参数判断使用本机 0.159.2 help。
未静默更换 backend/provider，未将其他工具的凭据迁入 pi 或 OpenCode。

证据：[latest_versions.json](S5_evidence/latest_versions.json)、各工具 version/help、
[readiness_summary.json](S5_evidence/readiness_summary.json)、auth_summary 文件、
[opencode_permission_schema.json](S5_evidence/opencode_permission_schema.json)。
readiness 请求的 45 秒上限只用于独立安装/认证检查，不是 Launcher 探索超时。

## 假工具与公共回归

四种默认 profile 通过真实 `pto-launcher` → 假 Optimizer → PATH 中 `ptobench`
→ 假 CPU Evaluator → Snapshot → S4 finish 的进程闭环验收。测试只创建临时 Case。
每个 backend 都直接执行 evaluate、fork、runtime、pmu、compiler，并核对 argv、
prompt、cwd、五个身份变量、profile/version、双路日志和最终额度。

S5 共 **15 项测试**：另外覆盖 YAML 默认/相对路径/拒绝配置、敏感 profile 拒绝及
宿主 secret 不写入记录、提前退出无 Snapshot、非零退出复核已有历史、预算等待
最后 action、预算结束、TERM→KILL 残留 worker、探索/active evaluator/finish 中断、
权威备份损坏后的停止与禁止 finalize、status 修复且不测量、阻塞终端完整留日志。
最终单独重跑 **15 项 / 123.592 秒，OK**，见
[launcher_tests_verified.txt](S5_evidence/launcher_tests_verified.txt)；机器摘要见
[validation_summary.json](S5_evidence/validation_summary.json)。

```bash
.venv-official/bin/python -m unittest discover -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -p test_launcher.py -v
.venv-official/bin/python -m unittest discover -s benchmarks/pypto_lib_bench-v0.2/evaluator/tests -v
```

完整回归实际执行 **136 项 / 295.068 秒，FAIL（21 个 subtest failure）**。
该次 discovery 包含 S5 的 14 项；之后新增的阻塞终端测试另跑 **1 项 / 9.260 秒，OK**。
全部 S5 项和其他公共 Session/Snapshot/finalize/CLI 回归没有失败。

21 个失败都属于 `test_every_case_owns_and_executes_its_comparator`，该次 inventory
为 198 Case、156 ready、195 个 evaluator 构造成功、3 个 non-ready 无 evaluator：

- MoE ep2/ep4 demo/flash/pro：6 个子测试，现有 inventory 测试没有
  `token_partition_ratio_reldiff` 示例。
- prefill_attention_hca 和 prefill_layer_ep2_csa demo/flash/pro：12 个子测试，
  现有 inventory 测试没有 `mapped_pool_compare` 示例。
- prefill_layer_ep2_csa demo/flash/pro：3 个“unselected chunk rows”示例预期 PASS，
  当前 Case callback 返回 `padded output row changed`。

这些 Case 的 comparator/config/evaluator 在 S5 期间有独立变动，S0 inventory
测试原文件 hash 未变。前 18 项是新比较器覆盖缺口；后 3 项需要 Case 所有者核对
比较规则和旧示例的适配，不能直接据此判定模型实现错误。
归属：**Case/comparator 迁移与 inventory-test 交接，超出 S5**。
依共同约束未修 canonical Case、未改 S0/S1 Comparator 测试、未跳过失败或降低标准。
原始 FAIL 和逐项归属见 [full_regression_final.txt](S5_evidence/full_regression_final.txt)、
[regression_defects.json](S5_evidence/regression_defects.json)。早期故障注入测试的
只读备份写入错误已修复，单独故障测试 PASS，原失败日志保留在本阶段证据内。

## 真实 NPU smoke

使用已有 host auth、默认 profile、`deepseek_v4.rmsnorm_decode_demo`、a2a3 device 0，
每个实验探索预算 1 / finish 1，新 Session 顺序运行；没有修改 Seed。
Agent 先调用公开 `ptobench evaluate`，Launcher 在额度结束后自动做独立 finish。

| backend | Session ID | exit | usage 探索/finish | 最终 correctness/integrity | latency us | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| codex | 20261003T050120Z_f64a04d58138 | 0 | 1/1 | passed / true | 16.9705 | 1.044754132170531 |
| claudecode | 20261003T050230Z_175039dcc9b3 | 0 | 1/1 | passed / true | 17.5795 | 1.0085611081088768 |

两项 close_reason 为 budget_exhausted，事件均为 evaluate PASS + finish PASS，
active_action=null；结束后的 Optimizer 进程组检查均已停止。
新数据位于 `runs/s5_real_<backend>/deepseek_v4.rmsnorm_decode_demo/<Session ID>`
及对应 snapshots 树。status 实际执行成功，未增加事件/预算。
证据：[real_smoke_summary.json](S5_evidence/real_smoke_summary.json)、
real_<backend>.yaml、stdout/stderr/exit/status，以及 Session 内 protected 记录和 final_result。
上述值仅报告本次 smoke，设备同时有其他工作负载；没有宣称优化收益、测量等价性
或真实四种 profile 的 NPU 支持。这些更广的验证不属于 S5。

## 保留与阶段结论

S5 没有写入任何 canonical Case 路径；真实 smoke Case 的资产清单没有变化。
基线包含 7,245 个 canonical 文件：3,954 个非 tensor SHA256，3,291 个 tensor
仅核对 size/mtime/mode，未对大 tensor 做完整字节 hash。
tracked 文件内容/index 未变，S0～S4 历史报告/证据和已有公共源码未变；
已有 benchmark README 的变化仅为本阶段说明链接。

整个共享工作区的 canonical 清单存在独立差异，完整路径/数量见
[preservation.json](S5_evidence/preservation.json) 和
[preservation_summary.json](S5_evidence/preservation_summary.json)。S5 保留这些现状，
没有将其他 Case 恢复至本阶段基线，不能声称全局 canonical 零差异。

公共 S5 与两项真实 Session：**PASS**；pi/OpenCode 真实验收：**BLOCKED**；
共享工作区完整 Comparator 回归：**FAIL，需要上述归属方处理**。
S6 未启动。进入完整后续验收前需解决 Comparator 回归；pi/OpenCode 在已有模型认证
配置就绪后才能分别运行真实 Session，假工具 PASS 不能替代这两项真实验收。
