# Experiment modes

`mode` 是实验方法，`optimizer.backend` 是执行工具。同一 Basic 方法可以使用
Pi 或 Codex，CANNBot/Ours 也按同一独立入口接入。

## 每个 mode 的 Pi provider 配置

Basic 和 CANNBot 的 Pi 实验分别读取 `basic/.env`、`cannbot/.env`，可以从
同目录的 `.env.example` 复制。修改 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`
即可切换网关和模型；`LLM_API` 为 Pi 协议名，例如 `openai-responses` 或
`openai-completions`，`LLM_REASONING` 控制 thinking level。
这些值优先于实验 `optimizer.profile` 中的 provider/model/reasoning。
`LLM_PROVIDER=provider-default` 是 Pi 中这组服务配置的标识，用于关联
`models.json` 的 provider 条目和 `--provider` 参数；留空也默认使用此标识。
它可以自定义，不代表具体厂商；实际服务由 `LLM_BASE_URL`、`LLM_API` 和
`LLM_MODEL` 决定，认证使用 `LLM_API_KEY`。
`.env` 支持注释、引号和 `export KEY=value`，值不执行 shell 命令、不展开变量。

`LLM_API_KEY` 留空时先使用进程环境的 `PYPTO_PROVIDER_API_KEY`，再使用原有
`benchmarks/mig_workflow/profiles_codex/glm-5.3-flash-max/auth.json` 的认证。
独立部署或切换到其他认证时，在 `.env` 中填写对应 key 即可。

每个 Run 在 `agent/pi/` 生成自己的 `models.json` 和 `settings.json`，
通过 `PI_CODING_AGENT_DIR` 交给 Pi；该目录允许 Pi 写入原生缓存，
配置另行脱敏归档到 `agent/config/`。CANNBot 所有 Task 阶段继承同一目录与认证。
provider、地址、协议、模型及已有的模型限制/compatibility 在 preflight 时确定，
key 在启动时读取并仅通过进程环境传递。`.env` 被 Git 和配置保护备份排除，
Run 只归档不包含 key 的有效配置。修改 `.env` 对随后启动的实验生效。

没有 `.env`（或文件为空）时沿用原来的 profile/主机配置；其他 backend 继续
使用自己的原生配置，不读取这份 Pi provider `.env`。

```text
modes/
├── basic/
│   ├── mode.yaml          # 方法身份与入口
│   ├── runner.py          # Basic 的准备、执行、原生会话收集
│   ├── prompt.py          # Basic 自己组装任务说明
│   ├── knowledge/         # 固定原始资料与版本 pin
│   └── instructions.md
├── _shared/
│   ├── adapters.py        # 方法侧共用的 native readiness
│   ├── config.py          # 方法侧 profile 解析与参数生成
│   ├── profiles/          # Pi/Codex 等默认与命名 profile
│   ├── provenance.py      # Agent 配置/instructions 冻结
│   ├── knowledge.py       # 方法侧知识库校验和只读复制
│   └── recording.py       # 原生参数与完整会话收集
├── cannbot/
│   ├── mode.yaml          # 已实现：Pi / Codex
│   ├── runner.py          # 独立入口与归档
│   ├── runtime.py         # 新阶段会话、答复和源码关联
│   ├── tools.py           # Task / skill / Todo 与局部 MCP
│   ├── bridge/pi_task.ts  # Pi 的局部扩展
│   ├── upstream/          # 固定版本原文
│   ├── upstream.lock.json # 来源、范围与内容 hash
│   ├── knowledge/         # 与 Basic 相同的固定原始资料
│   └── README.md
├── ours/
│   └── mode.yaml          # 临时名称，未实现
└── PROTOCOL.md            # 所有方法共用的入口与归档协议
```

Basic 已接入独立方法入口。它自行准备普通任务说明、固定模型/reasoning、
启动一次 Pi/Codex 等原生会话，并收集工作记录；Agent 自行选择优化方向，
不要求 hypothesis、PHK 或 replanning 表格。

Basic 和 CANNBot 各自保存 `knowledge/raw_wiki/` 与版本 pin，内容完全一致。
两种方法复用 `modes/_shared/knowledge.py` 自行校验、独立复制并设置只读权限，
在 session 中提供 `method/resources/raw_wiki/` 和简短的 `method/README.md`。
PTOBench 仅按既有通用附件协议保护这些资料，不导入知识库实现或解释内容。
公共 Case/CLI/结果说明由 `.benchmark_context/README.md` 提供；方法提示指向
这份导览，CANNBot 每个阶段还独立获得知识库入口。

PTOBench evaluator 只负责固定 Case、correctness、clean timing、原始 profile、
预算、源码档案和最终验证，不导入方法代码，不读取方法 instructions。
`session.json.mode` 和 workspace 的 `mode.json` 只包含方法 ID。
方法身份不会改变评测/预算规则，所有方法使用相同的公开 profile 权限。
Basic 开放 evaluate、fork 和全部三个 profile：runtime、pmu、compiler。

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
初始性能仍来自 Case 缓存，Agent 主动 evaluate 初始源码照常扣预算。
