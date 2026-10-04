# S6 provider 接入补充验收

日期：2026-10-03。pi/OpenCode 的 provider 与认证阻塞已解决；六个 backend/model 组合均通过原生读取工具测试及真实 NPU `evaluate → finish` 验收。按用户确认的验收口径，S6 公共层升级与集成验收通过结项；缺 Seed 的 Case 仍为 BLOCKED，原全库 hash 不一致记录仍为 FAIL。未执行 S7。

Provider 来自 `benchmarks/mig_workflow/profiles_codex/glm-5.3-flash-max`，端点 `https://slb-v1.api.fan/v1`，使用 Responses 协议及其现有 `auth.json`。原始源配置和认证文件只读；key 在启动时读入内存，通过进程环境交给客户端，没有复制到新配置、YAML、命令参数或证据中。

| 模型 | pi 原生工具测试 | OpenCode 原生工具测试 | pi NPU 闭环 | OpenCode NPU 闭环 |
| --- | --- | --- | --- | --- |
| `glm-5.3-flash` | PASS | PASS | PASS | PASS |
| `deepseek-flash` | PASS | PASS | PASS | PASS |
| `gpt-6.1-sol` | PASS | PASS | PASS | PASS |

原生工具测试要求模型读取临时文件并返回只有文件内才有的随机内容，验证工具实际执行及响应正确；不是只查询模型列表。pi JSON 事件确认返回 model；OpenCode 的脱敏 session export 确认 assistant 的 providerID/modelID 与指定值一致。见 [原生工具测试](agent_probes.json)、[OpenCode 模型选择](opencode_model_selection.json) 及相邻脱敏日志。

初始简化 API 探针的 GLM/DeepSeek 返回 HTTP 200，gpt 返回 HTTP 403、提示客户端请求异常；该结果没有推定成模型不支持。随后未经客户端伪装、模型或 provider 替换，使用正常 pi/OpenCode 的 Responses SDK 请求，三个模型均实际成功。证据：[API 探针](provider_probe.json)、[gpt 流式简化探针](gpt_stream_probe.json)。该地址通过 open_proxy 超时，但直连可用；启动脚本只将该主机加入 NO_PROXY，保留其他路由设置。

配置 provider ID 为 `profile-gateway`，两者 provider 列表都只有这三个模型；默认 `glm-5.3-flash`，OpenCode 的小模型任务也固定在此 provider。benchmark 的 pi/OpenCode `default` 已接入，并为三个模型各增加一个同名 profile。宿主便利入口为 `pi-slb` 和 `opencode-slb`。见 [配置与使用说明](../../../../mig_workflow/agent_provider/README.md)、[配置 hash](config_hashes.json)。本地 context=128000、output=8192 为保守设置，短请求测试不验证服务最大容量。

首轮 OpenCode 已连接三个模型，但因旧 PWD 使用仓库根目录，未能读取临时文件；原始失败记录保留于 `agent_probes_initial.json` / `initial_opencode_*.log`。启动脚本现同步 PWD，并对 `opencode run` 显式传入启动目录；复测三个模型均通过，真实 Session 也在注册 workspace 内执行。

六组真实验收均为同一个 `deepseek_v4.rmsnorm_decode_flash`，官方 fixture_000、device 0、warmup 5、rounds 10，探索上限 2 / finish 上限 1，实际各使用 1 / 1。GLM 使用接入后的 default profile，其余使用明确模型名的 profile。harness 仅注入执行一次 evaluate 的任务文字，其余使用真实 Launcher、Adapter、原生 coding agent、ptobench、NPU Evaluator 及自动独立 finish。每组检查事件恰为 evaluate/finish、全部通过、源码 hash 不变、final 使用新 workspace、10 个测量样本、无活动 action、Optimizer 进程组已停止。

| backend | model | profile | final (µs) | 结果 | Session |
| --- | --- | --- | --- | --- | --- |
| pi | glm-5.3-flash | default | 17.4295 | [PASS](pi_glm-5.3-flash_acceptance.json) | `/data1/home/ziruiwang/pypto-lib/benchmarks/pypto_lib_bench-v0.2/runs/s6_provider_pi_glm-5.3-flash/deepseek_v4.rmsnorm_decode_flash/20261003T074612Z_0547832c7318` |
| pi | deepseek-flash | deepseek-flash | 17.3095 | [PASS](pi_deepseek-flash_acceptance.json) | `/data1/home/ziruiwang/pypto-lib/benchmarks/pypto_lib_bench-v0.2/runs/s6_provider_pi_deepseek-flash/deepseek_v4.rmsnorm_decode_flash/20261003T074700Z_7951001d2940` |
| pi | gpt-6.1-sol | gpt-6.1-sol | 17.0190 | [PASS](pi_gpt-6.1-sol_acceptance.json) | `/data1/home/ziruiwang/pypto-lib/benchmarks/pypto_lib_bench-v0.2/runs/s6_provider_pi_gpt-6.1-sol/deepseek_v4.rmsnorm_decode_flash/20261003T074735Z_8b79e5045a7c` |
| opencode | glm-5.3-flash | default | 17.1395 | [PASS](opencode_glm-5.3-flash_acceptance.json) | `/data1/home/ziruiwang/pypto-lib/benchmarks/pypto_lib_bench-v0.2/runs/s6_provider_opencode_glm-5.3-flash/deepseek_v4.rmsnorm_decode_flash/20261003T074836Z_12fe8e9f3757` |
| opencode | deepseek-flash | deepseek-flash | 17.4395 | [PASS](opencode_deepseek-flash_acceptance.json) | `/data1/home/ziruiwang/pypto-lib/benchmarks/pypto_lib_bench-v0.2/runs/s6_provider_opencode_deepseek-flash/deepseek_v4.rmsnorm_decode_flash/20261003T074932Z_57663374620d` |
| opencode | gpt-6.1-sol | gpt-6.1-sol | 17.5190 | [PASS](opencode_gpt-6.1-sol_acceptance.json) | `/data1/home/ziruiwang/pypto-lib/benchmarks/pypto_lib_bench-v0.2/runs/s6_provider_opencode_gpt-6.1-sol/deepseek_v4.rmsnorm_decode_flash/20261003T075017Z_ec0b20af7c4b` |

[完整机器结果](acceptance_matrix.json)、[执行日志](acceptance.log)、[验收程序](accept_agents.py)。接入后 Launcher 公共回归 18 项通过（145.190s），见 [日志](launcher_tests.log)；原完整 146 项 S6 回归证据保留，公共 Evaluator/Launcher 实现未修改。

本补充验收逐字节 SHA-256 核对所用 RMSNorm Case、sealed assets、环境及 family helper：25 个文件、193,643 字节，前后相同。见 [内容保护检查](canonical_followup_check.json)。这只是本次范围检查，原 S6 全库 42 路径差异的 FAIL 仍保留，未覆盖原基线。双设备 `prefill_layer_ep2_hca_flash/public/candidate_seed` 仍缺失。

凭据检查覆盖新配置、证据、pi 宿主配置及六个 Session 的 2007 个文件，没有找到源 key 副本，见 [检查结果](credential_scan.json)。最后复核六个 Session 均无活动 action、Optimizer 进程组均已退出，公共 active marker 不存在，见 [清理结果](cleanup_check.json)。

补充验收之前的报告原文保留于 `before_followup_S6_integration_acceptance.md` 与 `before_followup_final_acceptance.md`；S0～S5 及原 S6 raw evidence 未改写。
