"""Describe observed provider support and real agent acceptance, preserving history."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
matrix = json.loads((HERE / 'acceptance_matrix.json').read_text())
probes = json.loads((HERE / 'agent_probes.json').read_text())
scope = json.loads((HERE / 'canonical_followup_check.json').read_text())
assert len(matrix) == len(probes) == 6
assert all(row['status'] == 'PASS' for row in matrix + probes)
assert scope['status'] == 'PASS'

lines = ['# S6 provider 接入补充验收', '', '日期：2026-10-03。pi/OpenCode 的 provider 与认证阻塞已解决；六个 backend/model 组合均通过原生读取工具测试及真实 NPU `evaluate → finish` 验收。S6 总体仍为 BLOCKED：双设备 Case 缺 Candidate Seed，原全库 canonical hash 差异未消除。未执行 S7。', '',
    'Provider 来自 `benchmarks/mig_workflow/profiles_codex/glm-5.3-flash-max`，端点 `https://slb-v1.api.fan/v1`，使用 Responses 协议及其现有 `auth.json`。原始源配置和认证文件只读；key 在启动时读入内存，通过进程环境交给客户端，没有复制到新配置、YAML、命令参数或证据中。', '',
    '| 模型 | pi 原生工具测试 | OpenCode 原生工具测试 | pi NPU 闭环 | OpenCode NPU 闭环 |',
    '| --- | --- | --- | --- | --- |']
for model in ('glm-5.3-flash', 'deepseek-flash', 'gpt-6.1-sol'):
    lines.append(f'| `{model}` | PASS | PASS | PASS | PASS |')
lines += ['', '原生工具测试要求模型读取临时文件并返回只有文件内才有的随机内容，验证工具实际执行及响应正确；不是只查询模型列表。pi JSON 事件确认返回 model；OpenCode 的脱敏 session export 确认 assistant 的 providerID/modelID 与指定值一致。见 [原生工具测试](agent_probes.json)、[OpenCode 模型选择](opencode_model_selection.json) 及相邻脱敏日志。', '',
    '初始简化 API 探针的 GLM/DeepSeek 返回 HTTP 200，gpt 返回 HTTP 403、提示客户端请求异常；该结果没有推定成模型不支持。随后未经客户端伪装、模型或 provider 替换，使用正常 pi/OpenCode 的 Responses SDK 请求，三个模型均实际成功。证据：[API 探针](provider_probe.json)、[gpt 流式简化探针](gpt_stream_probe.json)。该地址通过 open_proxy 超时，但直连可用；启动脚本只将该主机加入 NO_PROXY，保留其他路由设置。', '',
    '配置 provider ID 为 `profile-gateway`，两者 provider 列表都只有这三个模型；默认 `glm-5.3-flash`，OpenCode 的小模型任务也固定在此 provider。benchmark 的 pi/OpenCode `default` 已接入，并为三个模型各增加一个同名 profile。宿主便利入口为 `pi-slb` 和 `opencode-slb`。见 [配置与使用说明](../../../../mig_workflow/agent_provider/README.md)、[配置 hash](config_hashes.json)。本地 context=128000、output=8192 为保守设置，短请求测试不验证服务最大容量。', '',
    '首轮 OpenCode 已连接三个模型，但因旧 PWD 使用仓库根目录，未能读取临时文件；原始失败记录保留于 `agent_probes_initial.json` / `initial_opencode_*.log`。启动脚本现同步 PWD，并对 `opencode run` 显式传入启动目录；复测三个模型均通过，真实 Session 也在注册 workspace 内执行。', '',
    '六组真实验收均为同一个 `deepseek_v4.rmsnorm_decode_flash`，官方 fixture_000、device 0、warmup 5、rounds 10，探索上限 2 / finish 上限 1，实际各使用 1 / 1。GLM 使用接入后的 default profile，其余使用明确模型名的 profile。harness 仅注入执行一次 evaluate 的任务文字，其余使用真实 Launcher、Adapter、原生 coding agent、ptobench、NPU Evaluator 及自动独立 finish。每组检查事件恰为 evaluate/finish、全部通过、源码 hash 不变、final 使用新 workspace、10 个测量样本、无活动 action、Optimizer 进程组已停止。', '',
    '| backend | model | profile | final (µs) | 结果 | Session |', '| --- | --- | --- | --- | --- | --- |']
for row in matrix:
    root = row['launcher_result']['session_root']
    name = row['backend'] + '_' + row['model'] + '_acceptance.json'
    lines.append(f"| {row['backend']} | {row['model']} | {row['profile']} | {row['final']['final_verified_latency_us']:.4f} | [PASS]({name}) | `{root}` |")
lines += ['', '[完整机器结果](acceptance_matrix.json)、[执行日志](acceptance.log)、[验收程序](accept_agents.py)。接入后 Launcher 公共回归 18 项通过（145.190s），见 [日志](launcher_tests.log)；原完整 146 项 S6 回归证据保留，公共 Evaluator/Launcher 实现未修改。', '',
    f"本补充验收逐字节 SHA-256 核对所用 RMSNorm Case、sealed assets、环境及 family helper：{scope['files']} 个文件、{scope['bytes']:,} 字节，前后相同。见 [内容保护检查](canonical_followup_check.json)。这只是本次范围检查，原 S6 全库 42 路径差异的 FAIL 仍保留，未覆盖原基线。双设备 `prefill_layer_ep2_hca_flash/public/candidate_seed` 仍缺失。", '',
    '凭据检查覆盖新配置、证据、pi 宿主配置及六个 Session 的 2007 个文件，没有找到源 key 副本，见 [检查结果](credential_scan.json)。最后复核六个 Session 均无活动 action、Optimizer 进程组均已退出，公共 active marker 不存在，见 [清理结果](cleanup_check.json)。', '',
    '补充验收之前的报告原文保留于 `before_followup_S6_integration_acceptance.md` 与 `before_followup_final_acceptance.md`；S0～S5 及原 S6 raw evidence 未改写。']
decision_path = HERE / 'stage_decision.json'
stage_pass = decision_path.exists() and json.loads(decision_path.read_text()).get('status') == 'PASS'
if stage_pass:
    lines = [line.replace(
        'S6 总体仍为 BLOCKED：双设备 Case 缺 Candidate Seed，原全库 canonical hash 差异未消除。',
        '按用户确认的验收口径，S6 公共层升级与集成验收通过结项；缺 Seed 的 Case 仍为 BLOCKED，原全库 hash 不一致记录仍为 FAIL。') for line in lines]
(HERE / 'provider_acceptance.md').write_text('\n'.join(lines) + '\n')

s6 = REPORTS / 'S6_integration_acceptance.md'
text = s6.read_text()
text = text.replace('双设备 Case 缺少已发布 Seed，pi/OpenCode 缺认证。', '双设备 Case 缺少已发布 Seed；pi/OpenCode 的认证阻塞已通过本次 provider 补充验收解决。')
text = text.replace('| pi / OpenCode 真实后端 | BLOCKED | 缺所选模型 API key / 认证列表为零 |', '| pi / OpenCode 真实后端 | PASS：三个指定模型各通过 | [Provider 补充验收](S6_provider_followup/provider_acceptance.md)；六组真实 NPU evaluate/finish |')
text = text.replace('| pi | 1.0.0 | BLOCKED | No API key found for selected model；未创建真实验收 Session | — |', '| pi | 1.0.0 | PASS（补充验收） | 三个指定模型各完成 evaluate/finish，实际 1/1，正常退出和清理 | [六组矩阵](S6_provider_followup/acceptance_matrix.json) |')
text = text.replace('| opencode | 1.18.34 | BLOCKED | host auth list has zero credentials；未创建真实验收 Session | — |', '| opencode | 1.18.34 | PASS（补充验收） | 三个指定模型各完成 evaluate/finish，实际 1/1，正常退出和清理 | [六组矩阵](S6_provider_followup/acceptance_matrix.json) |')
text = text.replace('不能据此声称四个 backend 全部就绪。', '这是接入前的历史认证检查。补充验收现已复用指定 Codex profile 的 provider 与现有认证，pi/OpenCode 的三个指定模型全部通过真实调用和 NPU 闭环；详见 [Provider 补充验收](S6_provider_followup/provider_acceptance.md)。')
text = text.replace('pi/OpenCode 需要目标模型可用的现有宿主认证；', 'pi/OpenCode 的宿主认证阻塞已解决；')
if stage_pass:
    text = text.replace('**S6 总体：BLOCKED**', '**S6 阶段：PASS（公共层升级与集成验收结项）**')
    old = '剩余阻塞归属：双设备 Case 需要其迁移发布阶段补齐 canonical Seed；pi/OpenCode 的宿主认证阻塞已解决；全库 hash 相等 gate 需要在其他 canonical 修缮停止且基线明确后独立复核，当前 42 个差异不能抹去或重新定义成 PASS。代表硬件通过不会自动解除这些阻塞。'
    new = '结项口径：按用户确认，缺 Seed 的双设备 Case 保留 BLOCKED 记录，不要求修复该 Case 后才能结束公共层升级。42 条 canonical 差异全部属于其他 Case 资产及新增环境，原全库 hash 相等检查仍为 FAIL；S6 所选四个 Case 与所用环境内容相同。公共回归、可运行代表 Case 和四个真实 backend 均已通过，因此 S6 公共层升级与集成验收结项为 PASS。Case 级缺项和原始差异不被改写为通过，见 [结项决定](S6_provider_followup/stage_decision.json)。'
    text = text.replace(old, new)
    text = text.replace('同步实施状态、新入口与 S6 阻塞结果', '同步实施状态、新入口与 S6 验收结果')
s6.write_text(text)

final = REPORTS / 'final_acceptance.md'
text = final.read_text()
text = text.replace('| pi / OpenCode 真实接入 | BLOCKED：所选模型 API key 缺失 / 认证列表为空 |', '| pi / OpenCode 真实接入 | PASS：复用指定 profile provider，三个指定模型分别通过真实工具测试和 NPU evaluate/finish |')
text = text.replace('公共层通过、后端认证、Case 发布缺项及全库并发 hash 差异分别记录。', '公共层和四个真实后端已通过；剩余 Case 发布缺项及全库并发 hash 差异分别记录。')
note = 'pi/OpenCode 最新接入结果及六组原始证据见 [Provider 补充验收](S6_provider_followup/provider_acceptance.md)。'
if note not in text:
    text += '\n' + note + '\n'
if stage_pass:
    text = text.replace('S6 总体为 BLOCKED，公共层和四个真实后端已通过；剩余 Case 发布缺项及全库并发 hash 差异分别记录。',
        '按用户确认的验收口径，S6 公共层升级和真实集成验收通过结项。缺 Seed 的 Case 仍记录 BLOCKED；全库 hash 的原始 FAIL 及其外部 Case 变化归属继续保留。这些 Case 资产问题不阻止公共层阶段结项，也不代表全部 Case 已通过。')
final.write_text(text)
