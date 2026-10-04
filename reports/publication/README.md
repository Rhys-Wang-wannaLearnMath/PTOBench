# PTOBench 命名与发布

目标仓库：https://github.com/Rhys-Wang-wannaLearnMath/PTOBench.git

正式项目和目录使用 `PTOBench`，当前 pypto-lib 中的路径为 `benchmarks/PTOBench`。
`ptobench`、`pto-launcher` 命令保留，以继续使用既有 CLI 参数和 Agent prompt。
原 `benchmarks/pypto_lib_bench-v0.2` 目录保留；其中历史 Session 绑定绝对路径，
不能通过简单搬动目录迁移。本次保留其源码、资产与运行记录，避免改变历史运行入口。
新目录使用外置资产配置，不额外复制数百 GiB 的 Fixture 或运行产物。

## 修改与行为

此前独立运行修缮包括：根目录自动识别与显式覆盖、外置 sealed assets、29 个环境
profile 的路径占位符、虚拟环境 Python 与 CANN 安装链接处理、外部 `golden` 导入、
Pi/OpenCode 的 provider launcher 发现与系统 CLI fallback、Git 忽略规则及独立部署测试。

此次命名与发布步骤只修改 4 份当前使用文档：项目标题/仓库 URL、clone 和启动说明、
嵌套目录路径及测试命令；另外增加本发布报告与文件对比证据。
运行 Python 代码、CLI 入口、环境 JSON、1,006 个公开 Case 文件与已验收版本逐文件 SHA-256 相同。
见 [business_logic_preservation.json](business_logic_preservation.json) 和 [source_manifest.json](source_manifest.json)。

因此本次发布不改变 Case 实现、输入/输出契约、comparator 容差、Fixture/Ref/Oracle、
RunConfig、计时与成绩计算、预算扣除、Snapshot 选择、完整性保护和自动收尾逻辑。
路径/依赖配置和 Agent 启动方式属于部署适配：独立环境没有原 provider launcher 时，
Pi/OpenCode 使用主机工具及其认证；默认模型记录为 unknown，命名 profile 需要对应 provider。
这些部署差异不能解释为不同机器或不同 Agent 模型会给出完全相同的优化结果或计时数值。

## 验证

已验收版本完整回归：独立 Git clone 165/165 通过；嵌套模式 164 项通过。
详见 [双模式验收报告](../portability_recheck/README.md)。
正式 `benchmarks/PTOBench` 的新检查：从 `/` 启动两个 CLI，移除继承的 PYTHONPATH；
4 项 portability 回归通过，见 [portability_tests.log](portability_tests.log)。
正式目录的真实 NPU evaluate 与独立 finish 检查记录见 [hardware.json](hardware.json)。

Git 只发布控制源码、公开 Case、配置和文档/轻量验收记录。受保护资产以及 runs、
snapshots、build_output 和大型验收产物由 `.gitignore` 排除。
独立 NPU 使用需要外部 PyPTO-Lib、Torch/PyPTO 依赖、CANN/PTOAS 和匹配的 sealed asset bundle。
本次没有重跑 198 个 Case 的整批硬件验收，不能声明全部硬件 Case 均通过。
