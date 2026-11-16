# PTOBench 后续 Mode 接入计划

> 更新：2026-11-15。本文依据当前 PTOBench checkout、`modes/PROTOCOL.md`、现有 Basic/CANNBot 实现和 `basic_wiki-tool` 工作树整理。本文只规定新 Mode 与 PTOBench 的接缝，不规定 Mode 的内部设计。

Plan 05 的最终验收记录为正确性 75/75 PASS、Combined Profile 8/8 PASS。Plan 05 后的通用 Pi Tool 接口已在当前 checkout 实现；本计划不要求重跑 Plan 05 硬件验收。

## 目标与原则

后续 Mode 应优先在 `modes/<id>/` 内完成实现，直接使用 PTOBench 已有协议和服务。每个 Mode 可以按自己的目标选择 Agent、Prompt、知识资料、工具、编排方式、持久化格式、优化策略和测试结构；不要求复制 Basic/CANNBot 的实现，不要求采用共同的 Agent 工作流，也不要求把实现放入 `modes/_shared/`。

接入新 Mode 时不修改 PTOBench 公共层：不改 `launcher/`、`evaluator/`、根级 CLI、Session/Broker、通用沙箱、Case 运行或预算账本。若现有接口确实不能满足一个可复用的需求，应先把缺口、受影响的 Mode、最小通用接口和安全边界单独记录为公共层提案；在公共接口获准并完成之前，该 Mode 不得用私有补丁、临时挂载或复制一份公共机制绕过缺口。

这个原则限定的是公共层的责任，不限制 Mode 自己的设计。Mode 内部只要遵守下面的公开协议和安全边界，就可以使用不同的 Agent 结构、工具服务、资料组织、模型参数、阶段流程和记录格式。

## 当前接入能力

- Launcher 根据 `mode.yaml` 动态加载 `modes/<id>/runner.py`，不需要按 Mode ID 注册分支。声明使用 schema v2，当前公共加载器识别 `schema_version`、`id`、`implemented`、`description`、`entrypoint` 和可选的 `profile_strategy`。
- Mode 通过 `preflight / prepare / run / collect` 四步协议接入。协议、请求字段、Run 记录与 archive 行为以 [`PROTOCOL.md`](PROTOCOL.md) 为准。
- 实验配置可提供 `method_options` mapping；公共配置层将其传给 Mode 的 experiment 配置，不解释其含义。Mode 自行校验、记录并决定是否使用。
- `ptobench` CLI 提供共同的 Evaluate、Profile、Fork 和 Session 账本。Mode 可自由选择内部优化流程，但正式评测动作必须走该 CLI，不能另建 Case 执行器、预算计数器或结果账本。
- Pi Agent 使用现有 Bubblewrap、Gateway、File Guard、Workspace 和 Session 隔离。当前 checkout 的 `evaluator.isolation.agent_command(...)` 已提供通用可选 `pi_tool_extension` / `pi_tool_socket` 接口；Mode 可在自己的目录实现扩展和 Session 工具服务，并传入受管理的扩展及 Unix socket。Pi Tool 接线采用固定沙箱路径，CANNBot 原接口继续兼容。新 Mode 不自行增加挂载或放宽网络。
- `profile_strategy` 默认使用 `combined`；只有 Mode 的诊断流程确实需要分开采集时才显式声明 `split`。具体可用动作和 allowance 以当前 CLI / Session 实现为准，Mode 不自行推算或复制账本规则。
- Basic、CANNBot 与 `basic_wiki-tool` 展示了不同的 Mode 业务组织方式；它们是实现参考，不构成所有新 Mode 的必选架构。`basic_wiki-tool` 当前工作树包含自有分析工具、Pi 扩展和工具服务，可参考其公共接口调用方式，不把其进行中的业务实现视为通用要求。

## 已知公共边界与未承诺能力

当前 `evaluator.common.MODE_SOURCE_DIRS` 仍是固定目录列表（`basic`、`_shared`、`cannbot`），并被 Session 固定资产处理和 Candidate Worker Runtime 收集使用。新建 `modes/<id>/` 不会自动加入这些清单。遵守本计划时不得为了单个 Mode 修改该全局列表、复制 Mode 源码到 Worker，或声称新 Mode 源码已获得逐 Session 的固定资产保护。

如果一个 Mode 的正式使用需要公共层尚未提供的 Session 源码保护或 Runtime 依赖，先暂停该项接入并提交独立、通用的公共层设计；明确它如何同时保持 Session 隔离和 Candidate Worker 最小化。不能仅因某 Mode 使用了某个 Python 包或子目录，就扩大所有 Session 或 Worker 的文件集合。是否实现公共机制由独立变更处理，不属于新 Mode 接入任务。

目前的通用 Pi Tool 接口只解决受控扩展文件和 Session Unix socket 的沙箱接线，不提供通用工具框架、工具 schema、权限模型或执行服务。各 Mode 对这些内容自行负责；公共接口缺口按上一段处理。

## Mode 自主管理范围

Mode 可以自行决定并维护以下内容，按实际需要取舍：

- Agent/backend 选择、模型和 reasoning 参数、Prompt 与任务拆解方式。
- 领域知识、索引、分析器、Skills、工具扩展、Session-local 服务及其协议。
- 单 Agent 或多阶段编排、阶段间交接、计划和状态文件、重试策略。
- `method_options` 的结构和默认值、Mode 自有配置校验及脱敏归档。
- `method/`、`agent/` 下的运行记录、产物命名和 Mode 专属 archive。
- 单元测试、Fake Agent/CLI、工具权限测试和适合该 Mode 的验证方式。

Mode 代码、工具及测试放在该 Mode 自己的目录。Mode 的 Runner 负责内部编排和会话收集；Launcher 只管理协议生命周期，Evaluator 只维护公共评测、Session 和预算规则。`modes/_shared/` 是可复用的便利代码，不是新 Mode 的依赖条件，也不是所有 Mode 的共同实现层。

## 必须遵守的公共合同

- 目录名、`mode.yaml:id` 和实验 YAML 中的 `mode` 使用同一个合法小写 ID；入口使用现有 schema v2 和相对路径。
- 实现四步入口并遵守请求/响应、超时、进程组、退出和归档规则。没有独有业务逻辑时，不为四步协议重复造测试框架。
- Candidate 工作只在当前注册 Workspace 中进行；Context/Outputs 遵守其只读约定。Prompt 和工具说明必须与实际权限一致。
- 正式 Evaluate、Profile 和 Fork 通过公共 CLI 完成；Mode 不覆盖 Session/Event、正式结果或预算记录。运行资料保存在当前 Run 的 `method/` 或 `agent/` 下，静态 archive 与运行期记录按协议分别冻结。
- 凭据通过所选 Agent/backend 支持的安全渠道传递；Pi 使用现有 Provider/Gateway 接线（适用时）。凭据不得进入 Prompt、Agent 可见文件、日志或 archive。Mode 不挂载其他 Mode、Evaluator/Launcher 源码或其他 Session。
- 不修改 Case、Fixture、Seed、Baseline、CV、Selection 或 LOCK 来迁就 Mode。Candidate Worker Runtime 只有在存在独立公共层批准的必要依赖时才改变。
- Mode 可拒绝不支持的 backend、Profile 或选项，并在 `preflight` 给出安全、明确的错误；不要求所有 Mode 支持相同的 Agent 或硬件流程。

## 接入步骤

### 1. 定义 Mode 自己的范围

确定 Mode ID、使用者、支持的 Agent/backend、Mode 专属功能和最小可验证行为。只声明该 Mode 真正实现的能力。将自由选择留在 `modes/<id>/`，先确认需求是否已由公开协议、CLI 和隔离接口覆盖；若没有，记录独立公共层提案，不在本接入中改公共代码。

### 2. 实现 Mode 目录

最小入口通常包含：

```text
modes/<id>/
├── mode.yaml
├── runner.py
├── README.md
├── ... Mode 自己需要的模块、资源、扩展、配置模板与测试
└── tests/                  # 有需要时添加
```

按 Mode 的实际工作流实现 `preflight / prepare / run / collect`。以上只是建议目录，不要求固定文件名、包结构、资料类别或 Agent 流程；只有 `mode.yaml` 声明的入口和四步协议是公共接缝。公共层传入的 `method_options` 可作为 Mode 专属配置入口，配置细节由 Mode 自己定义。

### 3. 使用现有公共服务

通过 Mode 的 Runner 使用公开 Session context 和 `ptobench` CLI。Pi Mode 可复用当前 `agent_command(...)` 提供的隔离和可选 Pi Tool 接线；非 Pi 或工具较少的 Mode 不需要引入 Pi Tool 接口。工具实现、服务生命周期、权限决策和业务错误处理留在 Mode 内部。

如果模式要求无法由现有入口表达的公共能力，停止相关集成步骤，先提出与 Mode 无关的最小接口需求。不得通过调用私有 Evaluator 函数、绕开 Launcher、另行启动 Candidate、创建主机挂载或改写共享预算来实现。

### 4. 编写与需求匹配的验证

至少验证 Mode 自己的配置/流程和对公共合同的正确调用。优先使用临时目录、Fake Agent/CLI、假 Unix socket 等非硬件测试；只覆盖该 Mode 独有行为，不复制已有隔离、预算或 Launcher 回归。公共合同的行为由公共层现有测试覆盖。

先用 Fake 或其他非硬件方式验证该 Mode 的协议生命周期和独有功能。是否追加真实 Agent Session，由该 Mode 的发布/验收范围决定；只有需要验证硬件 Evaluate 时才使用 Case，并事先明确单次任务、额度和停止条件。无需为了新增一个 Mode 重跑全 Case、全 Profile、所有 Mode 或性能队列；相关测试若没有改动则复用已有结果。

### 5. 记录能力与限制

Mode 的 README/设计记录说明支持的配置、Agent、工具、业务流程、运行产物和已验证范围。区分 Fake 测试、真实 Agent Session 和硬件 Case 结果；不把一种 Mode 的证据推广为其他 Mode 的保证。

## 文件边界

| 文件范围 | 责任 |
|---|---|
| `modes/<id>/**` | Mode 的完整业务实现、配置模板、资源、工具、服务、测试及文档 |
| `modes/PROTOCOL.md` | 四步入口、请求/响应及记录/archive 公共合同 |
| `launcher/`、`evaluator/`、根级 CLI | 本计划中不修改；由 PTOBench 维护其既有公共职责 |
| Case 与验收资产 | 新 Mode 接入期间不修改 |

## Review 清单

- Mode 是否仅通过动态声明和四步协议接入，且没有新增公共层改动？
- Mode 的 Agent、工具、知识、配置和编排是否可由 `modes/<id>/` 自己维护？
- 正式评测动作是否走公共 CLI，是否避免复制 Session、Event、预算和结果账本？
- Agent 实际可见/可改的路径是否与 Prompt、README 和工具权限说明一致？
- 是否保护密钥和其他 Session，且没有增加主机目录挂载、网络访问或 Candidate Worker 文件？
- 测试是否只覆盖 Mode 独有逻辑，并按实际改动范围执行？
- 如有公共能力缺口，是否将它明确分离为独立提案，而未在 Mode 接入中夹带公共层修改？

## 可省略的工作

- 不复制 Basic/CANNBot/`basic_wiki-tool` 的 Prompt、知识结构、阶段流程、工具和归档布局。
- 不强制所有 Mode 使用相同模型、backend、Profile 策略、Agent 数量、任务拆分或内部状态格式。
- 不把 Mode 代码或依赖默认加入 `MODE_SOURCE_DIRS`、Candidate Worker Runtime 或其他 Session 的固定资产清单。
- 不为每个 Mode 重写通用 Sandbox、Gateway、CLI Broker、Session、Snapshot、预算或隔离测试。
- 不因接入一个 Mode 而修改公共层、其他 Mode、Case 资产或硬件验收范围。
