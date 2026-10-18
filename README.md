# PTOBench

实验入口为 `pto-launcher run EXPERIMENT.yaml` / `pto-launcher status SESSION_ROOT`。
配置、工具接入与自动收尾说明见 [LAUNCHER.md](LAUNCHER.md)。

实验方法通过 YAML 顶层 `mode` 选择，默认 `basic`，与 Codex/Pi backend 独立。
Basic 已使用独立方法入口，自行准备任务说明、执行 Agent 并收集原生会话；
PTOBench 只提供共同评测、预算和原始实验记录。
Basic 与 CANNBot 均开放 evaluate、fork 和全部三种 profile。
CANNBot 使用固定上游 PyPTO 性能入口，支持 Pi/Codex；[运行说明](modes/cannbot/README.md)。
Ours 目录已预留。方法组织和接入方式见 [modes/README.md](modes/README.md)，必须保留的
原始会话、配置、测量和失败附件见 [RUN_RECORDS.md](RUN_RECORDS.md)。初始
性能直接使用 Case 的已认证缓存，不要求 Agent 额外 evaluate 原始代码。

本目录既可以作为 `pypto-lib/benchmarks/PTOBench` 嵌套运行，
也可以从 [PTOBench GitHub 仓库](https://github.com/Rhys-Wang-wannaLearnMath/PTOBench) 独立克隆。独立运行时，设置
`PTOBENCH_REPO_ROOT` 指向要评测的 PyPTO-Lib 源码 checkout；未设置时，嵌套在
PyPTO-Lib 的 `benchmarks` 目录下且父源码树包含 `golden/` 时会自动使用源码根目录；
独立 checkout 默认使用自身目录。这样 benchmark 自身的
Case、Evaluator、Session 和 Launcher 不依赖当前工作目录或固定的仓库名称。
如果受保护的 Fixture、Ref 和 Oracle 资产不随 PTOBench 仓库分发，可用
`PTOBENCH_SEALED_ROOT` 将它们挂载到外部资产目录；其目录结构必须保持与
`sealed_assets/` 相同。这样大型二进制资产可以放在 Release、对象存储或内部挂载点，
而控制代码仍可作为普通 GitHub 仓库发布。

独立仓库包含控制代码和公开 Case。实际 NPU 评测还需要安装好的 PyPTO-Lib
源码/运行时、CANN、PTOAS 和匹配的 sealed asset bundle；环境路径设置见
[environments/README.md](environments/README.md)。Launcher 依赖安装、Agent 工具和
认证设置见 [LAUNCHER.md](LAUNCHER.md)。

当前主机的软件版本、依赖来源、工具链路径和已验证范围见
[当前软件栈与运行环境](ENVIRONMENT.md)（2026-10-05 核实），完整版本记录见
[环境快照](environments/inventory/20261005.json)。

## 独立运行

使用已安装 Torch/PyPTO 依赖的 Python 3.10+ 解释器，配置与 Case 匹配的工具链和资产：

```bash
git clone https://github.com/Rhys-Wang-wannaLearnMath/PTOBench.git
cd PTOBench
export PTOBENCH_ROOT="$PWD"
export PTOBENCH_REPO_ROOT=/path/to/pypto-lib
export PTOBENCH_SEALED_ROOT=/path/to/sealed_assets
export PTOBENCH_PTOAS_ROOT=/path/to/ptoas
export PTOBENCH_CANN_ROOT=/path/to/cann
# 使用 Miniforge 的环境配置时再设置此变量
# export PTOBENCH_MINIFORGE_ROOT=/path/to/miniforge3
python -m pip install -r requirements.txt
python pto-launcher run examples/experiment.yaml
```

示例默认使用 Codex；所选 Agent 工具须已安装并完成主机认证。公开源码和 Case
随 Git 仓库分发，受保护资产单独配置。`PTOBench` 是项目/仓库名称，命令仍为
`ptobench` 和 `pto-launcher`，例如 `ptobench evaluate`、`ptobench profile runtime`。

v0.2 将一个 Case 分成公开资产和评测专用资产：

- `cases/`：Agent 可见的 Case 描述、只读契约、公开比较规则和 Candidate workspace。只有 workspace 可写，其内部文件自由。
- `sealed_assets/`：评测器专用的输入、初始状态、NPU Ref、CPU Oracle、缓存输出和 Case-local evaluator。Agent 不应读取此目录。

Case 资产按模型组织为 `cases/<model>/<case>/` 和
`sealed_assets/<model>/<case>/`。历史 `models_0912.*`、`models_1001.*`
Case ID 仍可使用，Evaluator 会将它们映射到新的模型目录；旧的时间前缀目录不再作为规范布局。

每个公开 Case 使用三个简单文件划分职责：

- `case.json`：Case 路由、环境和评测模式；
- `public/config.py`：本 Case 唯一的固定配置，Ref 和 Candidate 共用；
- `public/comparator.json`：实际生效的输出比较方法与容差。

Agent 可以读取并 import 后两者，Bench 会校验并恢复受保护文件。Evaluator
直接读取 canonical Case 中的这两份文件；workspace 中的副本只是给 Agent 阅读的
只读快照，不参与正式评测。

`public/candidate_seed/` 可以包含任意数量的实现文件和子目录。Session 创建时会
递归复制整个目录，Agent 可以修改、增加或删除这些实现文件；只需保证
`case.json` 指定的入口仍可加载。Sealed `ref/` 独立保存初始 NPU Ref 运行所需的
完整冻结依赖，不从 Candidate workspace 导入代码。Candidate 和 Ref 共同读取
唯一的 public `config.py`，但不共用可编辑实现文件。

Comparator 的上游对齐原则在集成到 pypto-lib 时见 `../mig_workflow/COMPARATOR_POLICY.md`；
当前全部 Case 的逐项复核结论和证据分组见 `COMPARATOR_AUDIT.md`。

所有 Case 只执行一个 step，并只使用一个固定的正式 Fixture
`fixture_000`。Ref 在正式优化前生成一次输出和性能基线，Candidate 后续直接与
缓存的 Ref 结果比较。

Agent CLI 是 `ptobench`，提供 evaluate、profile runtime/pmu/compiler 和 fork。
`evaluate` 始终使用 clean 模式；启用的三种 profile 只运行 Candidate，
生成诊断信息，不检查输出正确性、不产生正式性能成绩。Agent profile 成功
表示诊断执行成功，结果不含 `correctness` 字段；正确性以 `evaluate` 和最终
fresh clean 为准。profile 不生成可供最终选择的 Snapshot。
迁移验收使用的 maintenance Session 仍保留原有 profiling 正确性检查。
探索 `evaluate` 使用 2 次预热、3 次正式测量并取中位数；session 最终验证
使用 Case 的正式配置，目前为 5 次预热、10 次正式测量。两者都保留原始样本。
中央 `evaluator/evaluate.py` 只提供统一结果格式。每个 Case 的
`sealed_assets/<family>/<case>/evaluate.py` 保留从原始实现提取的 TensorSpec、运行
方式，并完成该 Case 的具体执行。Comparator 的规则公开存放，由 Case-local
evaluator 读取并应用。

每个 Case 还提供 `sealed_assets/<family>/<case>/fixture_builder.py`。统一生成入口
只负责调用它；具体的官方 `build_tensor_specs()` 编排由 Case 自己保存。

Session 创建和 fork 会在 Candidate workspace 中生成给 Agent 阅读的 `.benchmark_context/`
目录：

```text
.benchmark_context/
├── README.md
├── workspace.json
├── case.json
├── config.py
├── comparator.json
├── reference.json
├── environment.json
├── starting_point/       # read-only initial Candidate snapshot
└── profiling_modes.json   # 仅在至少开放一种 profiling mode 时存在
```

这些是公开快照；`starting_point/` 保存本轮优化开始时的 Candidate，供对比和恢复使用，
不参与评测。正式评测始终从 canonical `cases/` 和 `sealed_assets/` 读取权威内容。

评测产生的结果和编译、诊断文件写入 Workspace 自己的
`.benchmark_outputs/<mode>/`，不会使用调用目录下的公共输出目录。

在已创建的 Session 中，设置 `PTOBENCH_SESSION_ROOT`，将 benchmark 目录加入 PATH，
以登记的 workspace 为当前目录。Agent 的完整操作集合是：

```bash
ptobench evaluate [WORKSPACE]
ptobench profile runtime [WORKSPACE]
ptobench profile pmu [WORKSPACE]
ptobench profile compiler [WORKSPACE]
ptobench fork [PARENT_WORKSPACE]
```

方括号表示可选位置参数，默认 `.`。workspace 必须是该 Session 登记的根目录。
缺少 Session 环境、未登记路径、旧 profile 名称或旧参数会明确失败；不会自动查找、
创建 Session 或回退评测。旧 `bench` 和 `evaluator/bench.py` 已删除。

`ptobench fork` 从父 Candidate 创建独立 workspace，返回 Bench 分配的绝对路径。
它复制完整实现及 helper，重新生成 context，不复制输出或缓存，不消耗探索额度。
evaluate 和每种 profile 各预占一次探索额度；预占后的失败或中断不退还。

## 启动 Agent 优化

Session 创建使用 [内部 Python API](evaluator/SESSION.md)，要求已有 ready 认证、
Fixture 和 Ref cache，不自动 prepare-ref。探索 CLI、事件、Snapshot 和 S4 内部最终
选择/复核已实现；[Launcher](LAUNCHER.md) 已提供 YAML 启动和自动复核。
参见 [最终复核 API](evaluator/FINALIZE.md)；集成到 pypto-lib 时还可参考
`../mig_workflow/SESSION_ACCEPTANCE.md`。

将内部 API 返回的 Session 和初始 workspace 路径传给 Agent：

```bash
# 从仓库根目录将完整 benchmark 目录加入 PATH
export PTOBENCH_ROOT=/path/to/PTOBench  # 嵌套使用时填 pypto-lib/benchmarks/PTOBench
export PTOBENCH_REPO_ROOT=/path/to/pypto-lib
export PATH="$PTOBENCH_ROOT:$PATH"

export PTOBENCH_SESSION_ROOT=/absolute/path/to/session
cd /absolute/path/to/session/workspaces/worker_000
```

Agent 只修改 workspace 中的 Candidate 实现文件，并通过 CLI 获取评测反馈：

```bash
# clean 正确性与性能，固定一次 action
ptobench evaluate .

# 需要时运行已开放的诊断模式；这些结果不是正式成绩
ptobench profile runtime .
ptobench profile pmu .
ptobench profile compiler .
ptobench fork .
```

Case、预算和开放模式由受保护 Session registry 决定。`ptobench` 必须使用完整的
benchmark 目录，不能只把脚本单独复制到 workspace。若不使用
`PATH`，也可以直接调用该脚本的绝对路径。

CLI 保留 Evaluator 的 stdout/stderr 并保存两份日志，最后打印一行 Session JSON。
evaluate 成功且 correctness/integrity PASS 时，从同一份冻结源码保存只读 Snapshot。
Agent profile 仅保存诊断和执行记录，仍检查执行错误及完整性，但不作正确性认证。
公开 `.benchmark_outputs/<internal_mode>/latest_result.json` 和
`latest_session_result.json` 是诊断副本；修改它们不影响权威事件或成绩。
pmu/compiler 仍分别使用内部 hardware/compile 产物目录。profile 无有效 latency
时记录 null，不补测性能。详细行为、错误码和日志背压说明见 [CLI 使用说明](evaluator/AGENT_CLI.md)。

Agent 将 CLI 当作外部评测工具，只读取当前 workspace 及其公开结果。
PATH、hash 和只读权限不构成操作系统隔离。直接绕过管理路径调用内部脚本违反实验规则。

Profiling 模式和新 workspace 的开放列表由 `evaluator/profile_modes.py` 配置。
新建 context 只向 Agent 公开当前开放的模式；没有开放的模式不会出现在
`.benchmark_context/README.md` 或 `.benchmark_context/profiling_modes.json` 中。Agent 不能通过 CLI
修改固定的 RunConfig 参数。Profiling 下的耗时仅用于诊断；最终性能比较必须
来自 clean 的 `ptobench evaluate`。Profiling 和 clean `evaluate` 使用同一个
固定 Fixture；区别只在于 profiling 生成诊断产物，而 clean `evaluate` 生成正式
性能成绩。

维护者在无相关活动 Session 时，继续使用 `evaluator/prepare_ref.py` 和官方
Fixture builder；普通 prepare-ref 复用完整 Fixture，只有输入契约、builder 或 seed
policy 变化时才使用 `--regenerate-fixtures`。内部直接对照使用 `evaluator/evaluate.py`。
这些维护能力不属于 Agent 操作，具体接口见 [Evaluator 说明](evaluator/README.md)。
