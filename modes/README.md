# Experiment modes

`mode` 是实验方法，`optimizer.backend` 是执行工具。同一 Basic 方法可以使用
Pi 或 Codex，CANNBot/Ours 也按同一独立入口接入。

PTOmni 当前是设计中的新方法，尚无可执行 mode。旧 SLUICE 实现仅保存在
[`PTOmni/`](PTOmni/ARTIFACT.md) 的只读 artifact 中，不会被 launcher 加载或复制到 runtime。
旧的单 Pi 设计与分阶段计划已移至 `PTOmni/design_docs/legacy_single_session/`；当前架构以
[`PTOmni 架构设计`](PTOmni/design_docs/PTOmni_架构设计_2026-11-03.md) 为准。
PTOmni 的 Profiling 诊断工具规划见 [`PTOAnalyzer`](ptomni/pto_analyzer/README.md)；它是 PTOmni 的只读组件，不是单独的可执行实验 mode。

代码命名约定：展示名为 `PTOmni`；未来可执行 mode 的目录名、`mode.yaml` ID 和实验配置值统一使用小写 `ptomni`。当前大写 `PTOmni/` 目录仅存放文档与历史资料，不作为代码包导入。

## 每个 mode 的 Pi provider 配置

Basic、CANNBot 和未来 PTOmni 分别读取自己的 `.env`，并保持三份配置一致；
同目录的 `.env.example` 是无密钥模板。Pi 使用 `LLM_BASE_URL`、`LLM_API_KEY`、
`LLM_MODEL`、`LLM_API` 和 `LLM_REASONING` 配置 provider、模型和 reasoning。
当前目标配置为 `deepseek-flash` / `high`，并覆盖 Pi profile 中的模型选择。
`LLM_PROVIDER=provider-default` 是 Pi 中这组服务配置的标识，用于关联
`models.json` 的 provider 条目和 `--provider` 参数；留空也默认使用此标识。
它可以自定义，不代表具体厂商；实际服务由 `LLM_BASE_URL`、`LLM_API` 和
`LLM_MODEL` 决定，认证使用 `LLM_API_KEY`。
`.env` 支持注释、引号和 `export KEY=value`，值不执行 shell 命令、不展开变量。

`LLM_API_KEY` 留空时使用进程环境的 `PYPTO_PROVIDER_API_KEY`，然后尝试已有的
provider 认证。`.env` 被 Git 忽略，不写入 Run 记录。

隔离 Pi Run 在 `agent/pi-config/` 生成只读 `models.json` 和 `settings.json`，
通过 `PI_CODING_AGENT_DIR` 交给 Pi。原生会话与锁文件保存在可归档的
`agent/native/`；受信任工具和 CANNBot 扩展保存在只读 `agent/pi-tools/`。
配置另行脱敏归档到 `agent/config/`。CANNBot 所有 Task 阶段继承同一配置与认证。
provider、地址、协议、模型及已有的模型限制/compatibility 在 preflight 时确定，
key 在启动时读取并仅通过进程环境传递。`.env` 被 Git 和配置保护备份排除，
Run 只归档不包含 key 的有效配置。修改 `.env` 对随后启动的实验生效。

未指定 profile token 限制且没有 `.env`（或文件为空）时沿用原来的 profile/主机配置；
指定限制时要求有效的 mode `.env`，否则明确报错。其他 backend 使用自己的原生配置。

Pi YAML 可直接设置 `context_window: 1048576` 和 `max_output_tokens: unlimited`，
也可将后者设为正整数。它们覆盖宿主模型的旧上限，与模型选择优先级独立；
`unlimited` 表示请求中不发送客户端输出上限，服务端及模型仍有自身限制；
省略参数不保证默认取最大值。当前五个 Pi profiles 因此显式设置官方最大输出，
DeepSeek 为 393216、GLM 为 131072。
Responses 使用原生 compat 开关；Completions 使用 Run 内小型请求扩展删除输出上限字段，
不修改 Pi 安装。现有 Pi profiles 的规格及来源见 [说明](_shared/profiles/pi/README.md)。

```text
modes/
├── basic/
│   ├── mode.yaml          # 方法身份与入口
│   ├── runner.py          # Basic 的准备、执行、原生会话收集
│   ├── prompt.py          # Basic 自己组装任务说明
│   └── instructions.md
├── _shared/
│   ├── adapters.py        # 方法侧共用的 native readiness
│   ├── config.py          # 方法侧 profile 解析与参数生成
│   ├── profiles/          # Pi/Codex 等默认与命名 profile
│   ├── provenance.py      # Agent 配置/instructions 冻结
│   ├── knowledge.py       # 保留的只读知识库工具
│   └── recording.py       # 原生参数与完整会话收集
├── cannbot/
│   ├── mode.yaml          # 已实现：Pi / Codex
│   ├── runner.py          # 独立入口与归档
│   ├── runtime.py         # 新阶段会话、答复和源码关联
│   ├── tools.py           # Task / skill / Todo 与局部 MCP
│   ├── bridge/pi_task.ts  # Pi 的局部扩展
│   ├── upstream/          # 固定版本原文
│   ├── upstream.lock.json # 来源、范围与内容 hash
│   └── README.md
├── PTOmni/                # 当前设计与只读历史 artifact，不是可执行 mode
│   ├── ARTIFACT.md
│   └── design_docs/
│       ├── PTOmni_架构设计_2026-11-03.md
│       ├── legacy_single_session/
│       └── references/    # 只读设计参考
├── ptomni/                # 未来的小写 mode ID；当前仅含设计文档
│   └── pto_analyzer/
│       └── README.md
├── ours/
│   └── mode.yaml          # 临时名称，未实现
└── PROTOCOL.md            # 所有方法共用的入口与归档协议
```

Basic 已接入独立方法入口。它自行准备普通任务说明、固定模型/reasoning、
启动一次 Pi/Codex 等原生会话，并收集工作记录；Agent 自行选择优化方向，
不要求 hypothesis、PHK 或 replanning 表格。

Basic 和 CANNBot 不校验、复制或注入 `raw_wiki`。公共 Case/CLI/结果说明由
`.benchmark_context/README.md` 提供，正常可见的项目资料仍在工作区中。
PTOmni 使用其独立的完整知识库。Pi Agent 在网络隔离环境运行，只通过
受限代理访问当前模型 API；所有 Agent backend 均应用相同限制。PTOBench CLI broker
与 Candidate Worker 保持原有执行路径，以保留 NPU Runtime/HCCL 通信行为。

PTOBench evaluator 只负责固定 Case、correctness、clean timing、原始 profile、
Evaluate 额度、源码档案和最终验证，不导入方法代码，不读取方法 instructions。
`session.json.mode` 和 workspace 的 `mode.json` 只包含方法 ID。
方法身份不会改变评测/预算规则，所有方法使用相同的公开 profile 权限。
Basic 默认开放 evaluate、fork 和一次 combined profile（runtime、pmu、compiler）；
只有接受的 `ptobench evaluate` 消耗 Evaluate 额度，失败或中断不退款；Profile 额度成本为 0
但仍记录为硬件操作，Fork 额度成本为 0 但仍记录为 Session 操作。额度耗尽后 CLI 拒绝新
Evaluate，Agent 仍可 Profile/Fork，直到自然退出或用户中断。需要分次采集时，method 可在
`mode.yaml` 显式设置 `profile_strategy: split`。

通用 Launcher 只调用方法声明的 `preflight / prepare / run / collect`，
传入公开 Run context，管理整个考生进程，保存原始输出和方法声明的附件。
它不直接组装 Basic 的优化 prompt，也不解释 CANNBot 阶段或 Ours 决策。
代码同在 PTOBench 仓库不代表考生职责由裁判实现。

实验 YAML 仍写 `mode: basic`，省略也默认 Basic；backend/profile 配置无须修改。
默认 profile 移到 `modes/_shared/profiles/`；已有旧位置的自定义 profile
仍由方法侧兼容读取。`mode.yaml` 的当前格式为：

```yaml
schema_version: 2
id: basic
implemented: true
description: General coding agent with clean evaluation and profiling feedback.
entrypoint: runner.py
profile_strategy: combined
```

新方法实现自己的 runner 和业务，遵守 [入口协议](PROTOCOL.md) 后声明 entrypoint。
不在 evaluator 或公共 Launcher 添加按方法名称区分的业务分支。
CANNBot 在自己的目录原样加载固定版本的上游技能、角色、catalog、分析脚本和例子，
方法入口承载 Pi/Codex 阶段会话，直接开放与 Basic 相同的 PTOBench CLI，保存完整原始档案。
API、profile 格式、原脚本与工具的使用由 CANNBot 自己处理，不属于接入预检或适配工作。
验证范围为资源加载、会话执行、CLI 可见性、共享预算、清理和记录；详见 [接入说明](cannbot/README.md) 和 [设计](cannbot/DESIGN.md)。
运行时设置 `mode: cannbot`，Pi/Codex 使用相同的 backend/profile 配置。
Ours 的 PHK、状态分析、hypothesis/replanning 同样由它自己实现。

Basic 保留既有 Run 的 prompt/profile/backend/config/native 档案布局。
额外的 `method.json / method_request.json / method_collection.json` 记录通用
入口和关联信息；完整方法说明在 Basic 提供的 Run `mode.json` 中冻结。
fork、测试与最终验证 workspace 继承共同评测政策和不透明的方法 ID。
初始性能仍来自 Case 缓存，Agent 主动 Evaluate 初始源码照常消耗 Evaluate 额度。
