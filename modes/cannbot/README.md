# CANNBot mode

CANNBot 与 Basic 是同层次的独立方法，使用同一 PTOBench CLI、Case、预算和
最终验证。Basic 启动普通编码会话；CANNBot 加载原始 PyPTO 性能流程，
由主会话按上游 Stage 7 指令安排新的 optimizer/verifier 会话。

```bash
python pto-launcher run examples/cannbot_pi.yaml
# 或
python pto-launcher run examples/cannbot_codex.yaml
```

实验 YAML 只需要设置 `mode: cannbot`，backend/profile 与 Basic 的用法相同。
首版支持 Pi 和 Codex，使用已有主机认证与相同的 profile/model/reasoning。
实际 NPU 运行需要该 Case 的已认证资产和环境；示例不自动安装依赖。

## 方法与执行接线

`upstream/` 原样保留 `hicann/cannbot-skills` 的固定快照
`94216ba2f0b569cfad22c7794ad99426a07ab7b6` 中选定的 56 个文件，包括完整
PyPTO 性能技能目录、catalog、分析脚本、例子、依赖技能、原始角色及许可证。
[upstream.lock.json](upstream.lock.json) 记录来源、选择范围与逐文件 SHA256。
每个 Run 都复制相同的原始字节；不在原文中改 API、profile 格式或调优规则。

`entry.md` 单独说明本次任务的 Case、性能入口、CLI 和共同实验约束。
它不是改写后的上游调优技能。`runner.py` 实现通用方法协议；
`runtime.py` 承载原生会话；`tools.py` 与 `bridge/pi_task.ts` 连接工具调用。
evaluator 和公共 Launcher 无须理解 CANNBot 的阶段与策略。

主会话获得 `Task / skill / todowrite / todoread`：

- `Task` 接收原始角色名与完整 prompt，启动一个新原生会话，返回原答复或错误。
  多个 Task 串行执行，共享 Candidate 与 Run；可显式传入公开 fork 的 workspace。
- `skill` 返回原始 SKILL.md 内容及文件路径，引用文件由 Agent 自己读取。
- Todo 原样落盘，接线程序不检查它是否符合某种优化策略。

Pi 通过每次启动的 `--extension` 加载局部扩展；Codex 通过每次启动的 `-c`
配置局部 stdio MCP server。没有全局插件安装或主机配置写入。
阶段会话加载原始角色；optimizer 同时加载上游规定自动加载的性能主技能。
阶段选择、交接摘要、目标与退出判断由 CANNBot 自己完成，运行器不代做。
子会话不获得 Task 工具，也不继承父会话历史。

`knowledge/raw_wiki/` 是与 Basic 相同的原始参考文档快照，版本和指纹固定在
`knowledge/knowledge.lock.json`，与 `upstream/` 分开保存。CANNBot 自行校验并
复制到 `RUN/method/resources/raw_wiki/`；文件设为 0444、目录设为 0555，再由
现有附件协议登记内容保护和恢复备份。这是权限与完整性保护，不是 OS 隔离。
主会话、全部新阶段和 fork workspace 共享同一份资料；每个阶段自动获得
`.benchmark_context/README.md` 与 `RUN/method/README.md` 的入口，无需父会话转述。
知识库读取使用普通文件工具，不占硬件预算；上游角色和技能原文保持不变。

所有会话直接使用 `ptobench evaluate/profile/fork`。公开 CLI 的原始返回值
不经过格式转换；旧 API、原脚本及其他上游宿主工具的使用由方法自行处理。
本接入不是完整的新算子开发插件，没有模拟 Stage 6 通过状态或补造成功结果。
初始性能使用 Case 缓存；方法主动请求的初始验证/profile 正常消耗预算。

## 保留的原始记录

```text
RUN/method/
├── config.json              # 公开运行参数、角色/技能路径
├── upstream.json            # 来源与内容 hash
├── README.md                # Agent 的知识库入口说明
├── knowledge.json           # 实际暴露的知识库路径、版本和指纹
├── resources/
│   ├── upstream/            # 冻结的原始内容
│   ├── integration/         # 本接入代码与入口的确切版本
│   └── raw_wiki/            # CANNBot 自行复制的只读原始知识库
├── dispatches/<id>/
│   ├── dispatch.json        # 父调用、角色、argv、源码前后指纹、进程与时间
│   ├── config.json          # 此会话配置与 workspace
│   ├── task.txt             # Task 原始输入
│   ├── prompt.txt           # 实际用户消息
│   ├── system.md            # 实际加载的角色/技能原文
│   ├── stdout.log / stderr.log
│   ├── result.json          # 原始末尾答复、退出码、原始错误
│   ├── recording.json       # 此会话的原生档案关联
│   └── agent/               # 原生 events.jsonl 与 native 会话
├── tools/<id>/              # 原始工具输入、结果或未返回的中断标记
└── work/                    # Agent 自己的 Todo、报告、memory
```

`RUN/agent/session.json` 汇集全部主/子会话。任何已启动阶段缺少对应原生历史，
整体状态都报告 `missing`，不能用主会话的历史掩盖。预算耗尽和中断由公共
生命周期终止整个方法进程组，收集器保留未完成的 dispatch 和 Task 输入。
Pi 的 tool-call ID 是原生调用 ID；Codex 接线能取得的是 MCP request ID，
记录明确标明来源，Codex 原生调用 ID 仍保留在原生会话中。
PTOBench 每次硬件操作的源码、测量、correctness、profile 和原始错误仍在
共同 action 档案中；diff、预算曲线等后续由这些记录计算。

测试覆盖真实工具传输与 CPU 评测替身，未执行真实 LLM 优化或 NPU 调优。
详细抽取边界与原流程依据见 [DESIGN.md](DESIGN.md)。
