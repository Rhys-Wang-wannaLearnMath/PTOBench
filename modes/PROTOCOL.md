# 独立方法入口协议

PTOBench 负责固定 Case、评测、原始 profiling、预算和最终验证。
各方法负责优化任务说明、知识/分析器、优化流程、原生 Agent 执行和会话记录。
通用 Launcher 管理方法程序的生命周期，保存原始输出和方法声明的记录；
它不读取方法 instructions，也不解释阶段、catalog、PHK 或 hypothesis。

Basic 和 CANNBot 已使用本协议；Ours 尚未实现。

## 方法声明

`modes/<id>/mode.yaml`：

```yaml
schema_version: 2
id: basic
implemented: true
description: General coding agent with clean evaluation and profiling feedback.
entrypoint: runner.py
```

`entrypoint` 是方法目录内的相对 Python 文件路径。Launcher 用自身的 Python
启动该入口；不需要修改 evaluator 或添加方法名分支。
占位方法在创建 Run 之前被拒绝。方法声明不指定 profile 权限；
所有方法使用 evaluator 的公共 profile 政策，Session 冻结这份共同政策。

实验 YAML 保持 `mode`、`optimizer.id/backend/profile`、Case、输出和预算配置。
通用层只验证 optimizer 字段的结构与标识格式；支持哪些 backend、如何解释
profile，由方法的 preflight 决定。已有 Basic YAML 无须修改。

## 四个操作

入口命令统一为：

```text
python /path/to/modes/<id>/runner.py <operation> /path/to/request.json
```

`preflight / prepare / collect` 每次返回一个 JSON 对象，必须有
`schema_version: 1`；不得往 stdout 混写终端日志。这些元数据操作有 60 秒超时，
不能执行硬件实验。`run` 是实际方法程序，stdout/stderr 原样保存与转发，
由共享预算、进程退出或用户中断结束；当前没有探索 wall-time 上限。

失败返回非零退出码及 `error_code: invalid_request | infrastructure_error`、
不含凭据或原始认证输出的 `message`。

### preflight

输入：`schema_version` 和完整 `experiment` 配置。
此时还没有创建 Run。方法检查自身资料、backend 可用性、版本和参数。
可以返回：

```json
{
  "schema_version": 1,
  "optimizer_metadata": {
    "command": ["pi", "--print", "{prompt}"],
    "tool_version": "..."
  }
}
```

除通用 `optimizer_metadata.command/tool_version` 外，回复中的其他字段由方法自己定义；
Launcher 不解释它们，而是原样作为后续请求的 `preflight` 传回方法。
Basic 用它传递已经固定的 native profile 和工具 metadata。

### prepare

Run 创建后，输入包括：

- `experiment`：相同实验配置。
- `preflight`：方法之前返回的完整对象。
- `context`：`session_root`、`session_id`、`workspace`、`budget`、`profiling_modes`，
  以及公开的 `case.case_id/module/entry`。

请求不包含 sealed 路径/实现或完整内部 Session 状态。
Case 契约、环境摘要和缓存基准可从 workspace 的 `.benchmark_context` 读取。
请求保存到受保护的 `method_request.json`；入口声明保存到 `method.json`。

方法自行准备 prompt、冻结资料和配置，并返回需要保存的原始记录及已有附件：

```json
{
  "schema_version": 1,
  "records": {
    "prompt.txt": "...",
    "method/resource_manifest.json": {"files": []}
  },
  "archives": {
    "resources": "method/resources"
  }
}
```

records 的字符串原样保存；其他 JSON 值保存为 JSON。
archives 是方法提交的只读附件。返回声明之前，方法应去掉附件文件的写权限，
使实际权限与只读保护清单一致。运行中会继续写入的日志/Todo/dispatch 目录
在 collect 时提交，不能在 prepare 时作为冻结附件提交。
新方法记录放在 `method/` 或 `agent/`；为兼容既有 Basic 档案，还允许
`prompt.txt / profile.json / mode.json / backend.json`。
不能覆盖 Session、预算、Bench events、基准或正式结果。
archives 声明 Run 内已准备好的文件/目录，通用层按同一规则建立内容 hash/备份保护。
附件不与 records 重叠，archive ID 只首次注册，collect 不重复声明 prepare 中的档案。
工作中还要修改的 Todo/报告不要在 prepare 注册为只读档案，
而是在 collect 阶段冻结。方法业务不由通用归档层实现。

### run

输入为相同的受保护请求文件，在注册 workspace 中运行，获得 PATH 和
`PTOBENCH_SESSION_ROOT/SESSION_ID/CASE_ID/OPTIMIZER_ID/WORKSPACE`。
保留兼容的 `PTOBENCH_AGENT_BACKEND` 标识，但公共层不解释其值。
方法通过公开 `ptobench evaluate/profile/fork` CLI 使用同一 Session。

Basic 将进程 `exec` 为一次原生 Pi/Codex 会话，保留同一个 PID、进程组和退出码。
CANNBot 在 run 内启动原始编排角色，由其通过局部 Task 工具启动多个阶段会话。
阶段退出和新阶段启动属于方法内部逻辑，Launcher 不参与。
CANNBot 当前设计要求原样使用固定上游内容，方法入口只承载会话执行、交接和记录。
公共 CLI 直接向主/子会话开放；API、profile 和原工具的使用问题由方法自行处理。

**子进程必须留在方法拥有的进程组内**，以便通用生命周期机制在预算耗尽、
中断或异常退出时完整清理。当前协议不支持方法自行启动脱离该组的 Agent worker。
evaluator 的独立执行组仍由原有 Bench 机制管理。

一个方法进程对应一个 Run；阶段间不创建新 Session、不重置预算。
方法整体退出可以是非零，正式成绩仍由历史合法 Snapshot 和共同 finish 判定。

### collect

探索关闭且拥有的运行进程清理后调用。输入额外带 `started` 和 `close_reason`。
通用层先按共同的保护规则检查/恢复方法源码和已冻结配置，再运行收集入口，
避免执行 Agent 在最后一次评测之后改写的控制代码。
若进程清理或源码恢复失败，不调用收集入口，按 infrastructure error 结束并保留已落盘记录。
方法收集全部原生会话、实际 prompt/配置、知识暴露、阶段摘要和方法日志；
输出与 prepare 相同的 records/archives，另可返回：

```json
{"schema_version": 1, "recording": {"status": "recorded"}}
```

`recording.status: missing` 表示原始记录不完整；Launcher 保存已经获得的
独立最终成绩，同时报告 `recording_incomplete`。Basic 保留既有
`agent/session.json` 和录制状态；原生工具未提供的信息不能补造。
通用 Launcher 保存 `method_collection.json`，status 不必了解某个 backend 的会话格式。

当前 collect 每次有 60 秒超时。CANNBot 在每阶段持续落盘原生档案和关联，
collect 只做收尾，不能依赖最后一次调用才生成此前全部决策记录。

## 共用工具的位置

`modes/_shared/` 提供 native backend readiness、profile 解析、模型/reasoning
固定、配置冻结和原生会话收集。它是方法执行侧的工具库，evaluator 和通用
Launcher 不导入它。方法可以复用，也可以提供自己的实现。
`knowledge.py` 同样属于方法侧：Basic/CANNBot 各自提供固定原始文档，
prepare 时复制到 session，设置只读文件/目录权限，返回版本记录和附件声明。
方法自行向主会话和新阶段提供知识库入口；Bench 的公共 workspace 导览
只说明 Case、可编辑/可读文件、CLI、预算和反馈，不解释方法资源。
所有生成给 Agent 的任务提示、入口导览和 CLI 帮助必须使用简明、清晰的英文。
原始知识库和上游角色/技能保留原文，不为此翻译或改写。
Agent 导览不介绍测量轮数、自动选择历史源码、最终验证策略或内部控制目录。
默认 profile 在 `modes/_shared/profiles/`；Basic 仍能读取旧位置
`launcher/profiles/<backend>/<name>.yaml` 中已有的自定义 profile，优先使用它，
该兼容路径由方法侧处理。

`session.json.mode` 和 workspace 的 `mode.json` 只保留不透明方法 ID。
Basic 完整说明及其 hash 位于方法声明的 Run `mode.json`，由 Basic 提供。
这与共同的 profile 权限、正确性规则和正式计分无关。
