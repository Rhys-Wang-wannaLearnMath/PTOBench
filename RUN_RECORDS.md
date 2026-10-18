# 原始实验记录

Run 同时保存 PTOBench 的实验事实和 Codex/Pi 的原生工作记录。后续脚本可从
这些文件计算曲线、计数、统计量、源码 diff 和失败分类，本次不新增分析报表。

```text
SESSION_ROOT/
├── experiment.json          # case、mode、backend、预算、输出配置
├── method.json              # 通用方法入口与声明 hash
├── method_request.json      # 交给考生的公开 Run context
├── method_collection.json   # 原始档案路径与方法报告的录制状态
├── mode.json                # 方法入口、instructions/原文版本与内容 hash
├── baseline.json            # Case 已认证的 ref_performance.json
├── profile.json             # 启动配置，含实际追加的原生记录参数
├── backend.json             # 工具版本、实际 argv、cwd
├── prompt.txt               # 实际传给 Agent 的任务
├── environment.json         # Case Python/软件包/源码/硬件/工具链信息
├── agent/
│   ├── configuration.json   # 启动前配置、instructions 来源和文件 hash
│   ├── config/              # 冻结配置与 instructions，认证字段已移除
│   ├── events.jsonl         # 原样保存的原生 stdout 事件流
│   ├── native/              # 原生会话文件：消息、工具结果、usage
│   └── session.json         # 本 Run 与原生会话 ID、文件的绑定
├── events.jsonl             # Bench 操作、起止时间、源码 hash、操作退出码
├── staging/action_000001/
│   ├── workspace/.benchmark_context/starting_point/  # 当次完整源码
│   ├── authority_result.json # 原始 evaluator JSON，失败时也保留
│   └── artifacts/           # 原始结果、stdout/stderr、执行记录、编译/诊断文件
│       └── index.json       # 文件位置、hash、对应事件和源码路径
└── final/                   # 独立最终验证及其原始附件
```

配置、原生会话和每次操作附件都纳入现有的内容 hash/备份保护。运行中的
终端流先写入 `stdout.log` / `stderr.log`，退出、失败或中断后归档。
workspace 的 `.benchmark_outputs/.../latest_result.json` 仅表示最近一次操作，
不能代替每次操作的历史档案。

## Pi 与 Codex

以下 Agent 专属准备与会话收集由 `modes/basic/runner.py`、`modes/cannbot/runner.py` 和方法侧
`modes/_shared/` 实现，通用 Launcher 只归档返回的原始记录与附件。
PTOBench 不解释原生会话格式；新方法按同一入口协议自行提交完整档案。

Basic 的 Pi profile 自动使用 `--mode json --session-dir SESSION_ROOT/agent/native`，
原生会话直接属于本 Run。所有 Codex profile 自动使用 `--json`；收集器读取
本次 `thread.started.thread_id`，只复制 ID 和 workspace 都匹配的原生会话。
它不会使用主机上“最新的会话”。实验启动禁止 ephemeral/no-session 和
继续、恢复旧会话。usage 仅保留原生记录实际提供的值。

`agent/session.json.status = recorded` 表示找到了对应原生会话。若启动失败
或中断太早，可能只有事件流；状态明确写成 `not_started` 或 `missing`。
`missing` 表示采集不完整，Launcher 报告 `recording_incomplete` 并返回失败；
已经获得的独立最终性能验证仍会保留。
ClaudeCode/OpenCode 保留原有终端记录行为；本次原生接入覆盖 Codex/Pi。

CANNBot 为主会话和每次 Task 分别建立 `method/dispatches/<id>/`，每个目录
保存独立的 prompt/system、argv/profile、原始输入/答复/错误、源码指纹和
`agent/events.jsonl`、`agent/native/`。Pi 的 session-dir 属于当前 dispatch；
Codex 按该 dispatch 实际输出的 thread ID 和 cwd 收集。
根目录 `agent/session.json` 汇集所有会话与父 Task 关联；一个阶段历史缺失
即整体 `missing`。完整路径和额外原始工具/报告档案见 [CANNBot 记录](modes/cannbot/README.md)。

实际 model、reasoning、工具权限和上下文配置来自保存的 argv、启动前配置和
原生会话。Codex 将当前选择的 model/reasoning 追加为显式参数；Pi 显式指定
thinking，并显式固定 profile 或主机设置里的 provider/model。profile 可通过 `model`、
`reasoning` 设置这些值。工具未提供的种子、温度或内部信息不会被补造。

## 性能与执行事实

初始性能来自 `baseline.json`，不额外测 baseline。Agent 主动 evaluate
原始代码仍是一项普通探索操作，照常消耗预算。每个 fixture 保存：

- `raw_samples_effective_us`：原始顺序和所有 timing slot；0、负值和缺失值
  保留，非有限值使用 `nan` / `inf` / `-inf` 字符串。
- `samples_effective_us`：旧有的有效正数样本，用于当前正式成绩。
- `measurement_config`：指标/单位、请求的 warmup/rounds、runtime 返回的
  实际 warmup/rounds、输入恢复设置。
- 原有 correctness/error、latency、speedup。

探索 `ptobench evaluate` 固定使用 2 次预热、3 次正式测量；session 最终验证
使用 Case 的正式配置，目前为 5 次预热、10 次正式测量。两者都保留全部正式
样本，以及请求和实际执行次数。

clean PASS 必须包含原始样本和测量配置；缺少它们不会成为正式成绩。
历史 baseline cache 的完整性取决于认证时保存的内容，本次不重新认证。

`execution.json` 保存 evaluator 起止时间、平台/设备，以及 Golden runtime
dispatch 的进入、返回或原始异常。进入 dispatch 可能仍在设备准备阶段，
不能等同于内核成功执行；成功返回和已测出的样本提供进一步证据。
观察器不在 measured rounds 中加入逐轮记录。
`process_execution.json` 另外保留原始子进程退出码、收到的信号与起止时间；
它和 CLI 返回的 130 等状态码分开，避免把 CLI 状态误当成进程的退出原因。

预算延续现有规则：evaluate/profile 接受后先预占一项，编译失败和中断不
退还；finish 使用独立保留预算。原始事件与执行阶段保留，以便后续按论文
定义区分预算预占和实际进入硬件阶段。事件 seq 不是 interaction 序号，
finish 也不是探索预算的一部分。

没有合法 Snapshot 的失败尝试也保留原始源码、结果和错误输出。附件索引
保留 runtime 生成的链接/特殊节点描述，但不读取其指向的外部内容；普通
文件完整归档。源码 Snapshot 的 PASS 条件保持原有规则。
