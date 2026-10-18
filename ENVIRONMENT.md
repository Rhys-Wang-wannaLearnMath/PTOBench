# 当前软件栈与运行环境

核实日期：**2026-10-05，Asia/Shanghai**。本文件记录当前主机和工作区的实测状态；机器路径、软件版本和验证范围随环境变化需要更新。完整版本与安装位置见 [机器可读快照](environments/inventory/20261005.json)。

PTOBench 当前 Case 的环境 profile 使用 **`.venv-official/bin/python`、CANN 9.0.0、PTOAS 0.60**。这套环境的最小 A2/A3 与 A5 功能模拟器运行、Golden 保存/回放及 Benchmark 已通过；完整 PyPTO-Lib Golden 测试仍有一项 Torch dtype 兼容性失败。正式 Case 的认证结果应以对应 Case 和 Session 记录为准。

## 主机与硬件

| 项目 | 当前状态 |
| --- | --- |
| 主机架构 | `aarch64` |
| 操作系统 | Huawei Cloud EulerOS 2.0 |
| Linux 内核 | `5.10.0-136.12.0.86.r1526_92.hce2.aarch64` |
| NPU | 8 张 Ascend **910B4**，设备 ID 0–7 |
| HBM | 每卡 32 GiB |
| NPU 驱动 / npu-smi | `25.5.0` |
| 本机实机目标 | `a2a3`；没有 Ascend 950 / A5 实机 |
| 功能模拟器 | 最小内核在 `a2a3sim` 和 `a5sim` 均通过 |

设备利用率和占用是动态状态。profile 中的 device ID 仅表示配置，不代表设备已经分配给当前任务；运行前使用项目的设备分配方式确认设备集。

## 源码与核心依赖

| 组件 | 角色 | 当前主要环境 / 来源 |
| --- | --- | --- |
| PyPTO-Lib | 模型、示例和 `golden/` 验证接口 | `/data1/home/ziruiwang/pypto-lib` |
| PTOBench | Case、Evaluator、Session、Launcher | `/data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench` |
| Python | 应用解释器 | `.venv-official/bin/python`，`3.10.20` |
| PyPTO | DSL、IR、编译和代码生成 | 包版本 `0.1.0`；来源 checkout `pypto-official-747e1e4b` |
| simpler | Ascend / 模拟器运行时 | 包版本 `0.1.0`；来自同一 PyPTO checkout 的 `runtime/` 子模块 |
| Torch | CPU golden、tensor 和 dtype | `2.6.0+cpu` |
| PTOAS | PTO / MLIR 到内核 C++ 的汇编与优化 | `0.60`；`/data1/home/ziruiwang/ptoas-v0.60` |
| PTO ISA | 编译与 runtime 使用的 tile headers | 由已安装 simpler 的 `pto_isa.pin` 管理 |
| CANN | AICore/AICPU 编译器、ACL/HCCL、设备工具 | `9.0.0` |
| PyYAML | PTOBench Launcher 读取实验 YAML | `6.0.3`；声明依赖为 `PyYAML>=6,<7` |

当前源码定位：

| 源码 / 子模块 | Git revision |
| --- | --- |
| PyPTO-Lib HEAD | `9c5dfec2dfa18f056098000123e10de06de32426` |
| PTOBench HEAD | `7a31910487ca1eb8867efad937bc7211e95ca457` |
| 官方 PyPTO 来源 HEAD | `747e1e4ba99f13dea1a609d893ac7512e85a5897` |
| 官方 simpler 子模块 | `4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0` |
| 官方 PTO ISA pin / checkout / runtime 构建元数据 | `a8040450238f162985d8b596fbebeb54bfba2bf5` |

PyPTO-Lib 和 PTOBench 工作区都有未提交或未跟踪内容，Git HEAD 不能完整表示当前工作区。PyPTO/simpler 两个包都显示 `0.1.0`，因此辨认环境还需要安装来源和 commit。

官方 PyPTO 来源树的 `jit/decorator.py` 有未提交修改，但已安装版本与该树的 HEAD 文件一致，并未包含当前源码修改。来源目录不是 editable 安装，不应假定修改源码后已安装包随之改变。

## 本机几个 Python 环境的区别

所有以下路径均相对于 PyPTO-Lib 根目录，PTOAS 环境除外。

| 环境 | PyPTO 来源 | Torch | 当前用途与限制 |
| --- | --- | --- | --- |
| `.venv-official` | `747e1e4b…`；simpler `4e4d3a4a…` | `2.6.0+cpu` | PTOBench profiles 选择的主要环境；缺 E8M0 dtype |
| `.venv` | `672871bd9628aac43241f7e5e0e552efeb891f71`；simpler `dccb8379080b43173744b9981de2542b3d025e19` | `2.13.0+cu130` | 旧 API 环境；不能替代当前 PTOBench 环境 |
| `.venv-official-a5` | 未安装 PyPTO/simpler | 未安装 Torch | 空应用环境，不代表已具备 A5 软件栈 |
| `/data1/home/ziruiwang/ptoas-v0.60` | 未安装 PyPTO/simpler | 未安装 Torch | PTOAS 的独立 Python 环境 |

当前 shell 未激活 venv，默认 `python` / `python3` 指向 PTOAS 环境，直接运行模型会缺少 PyPTO/Torch。启动 Launcher、读取 Case 元数据或执行 PyPTO-Lib 代码时，明确使用 `.venv-official/bin/python`。

PTOBench Evaluator 会依据 Case profile source CANN、设置环境并 re-exec 到指定 Python，因此其执行环境与调用者的默认 shell 不是同一个概念。Launcher 初始进程仍需要具备 PyYAML、Torch/PyPTO 等 preflight 依赖。

## 编译器、包管理和辅助工具

| 工具 | 当前 shell / 主机 | `.venv-official` 内安装版本 |
| --- | --- | --- |
| GCC / G++ | `10.3.1`，`/usr/bin` | 使用外部编译器 |
| `g++-15` | `15.2.0`，`/data1/home/ziruiwang/.local/bin/g++-15` | 使用外部编译器 |
| `gcc-15` | PATH 中未找到 | 新建/重建模拟器环境的前提尚未完全满足 |
| CMake | `/usr/bin/cmake`，`3.22.0` | `4.4.3` |
| Ninja | `/usr/bin/ninja`，`1.8.2` | `1.13.2` |
| Git | `2.33.0` | 使用外部 Git |
| uv | `0.11.8` | 当前包管理工具 |
| scikit-build-core | — | `1.0.3` |
| nanobind | — | `2.15.0` |
| pytest | — | `9.1.1` |
| NumPy | — | `2.2.6` |
| SymPy | — | `1.13.1` |

`.venv-official` 没有 `pip` 模块；维护现有环境使用 `uv pip ... --python .venv-official/bin/python`。venv 中的 CMake/Ninja 包版本与未激活 shell 实际解析到的命令不同，构建记录需要保留所使用的可执行文件路径。

已读取 CANN runtime、bisheng-compiler、HCCL、simulator、pyACL 的版本文件，均为 **9.0.0**。MindStudio operator-profiler 的组件版本文件是 **26.0.0**；`msprof --version` 不支持该查询方式，不能用其退出码判定工具未安装。

当前 shell 的 `PYTHONPATH` 还包含 CANN Python 包目录。版本快照区分 venv 自身包与从 CANN 路径可见的包；可见的发行包元数据不等于每个模块均已导入或验证。快照内的软件包列表是盘点结果，不是可直接安装的锁文件。

## PTOBench 环境路径与参数来源

| 变量 / 路径 | 当前嵌套运行值或默认值 | 用途 |
| --- | --- | --- |
| `PTOBENCH_ROOT` | `/data1/home/ziruiwang/pypto-lib/benchmarks/PTOBench` | Bench 命令和资产根目录 |
| `PTOBENCH_REPO_ROOT` | `/data1/home/ziruiwang/pypto-lib` | 外部模型源码；嵌套布局可自动推导 |
| `PTOBENCH_PTOAS_ROOT` | `/data1/home/ziruiwang/ptoas-v0.60` | profile 的 `{ptoas_root}` |
| `PTOBENCH_CANN_ROOT` | `/usr/local/Ascend/cann-9.0.0/cann` | profile 的 `{cann_root}`，这里包含 `set_env.sh` |
| `PTOBENCH_MINIFORGE_ROOT` | `/data1/home/ziruiwang/miniforge3` | 某些 profile 用其 `lib/` 提供 libstdc++ |
| `PTOBENCH_SEALED_ROOT` | 默认 `PTOBench/sealed_assets`；也可外部挂载 | Case 的受保护资产位置 |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/cann-9.0.0/cann-9.0.0` | source CANN 后的实际工具链根 |
| `CANN_SET_ENV_FILE` | `/usr/local/Ascend/cann-9.0.0/cann/set_env.sh` | 当前 shell 的 CANN setup 脚本 |
| `PTOAS_ROOT` | `/data1/home/ziruiwang/ptoas-v0.60` | PyPTO 编译时查找 PTOAS |

这些 PTOBench 路径来自 `evaluator/common.py` 的嵌套默认值，可用 `PTOBENCH_*` 变量显式覆盖。独立 checkout 应显式配置源码、工具链和资产根目录。上表的默认值不表示对应变量都已经 export。

`PTO_ISA_ROOT` 当前导出到官方 venv 的 `simpler_setup/_assets/build/pto-isa`。新版 PyPTO/simpler 根据自己的 pin 解析托管 ISA，不使用 ambient `PTO_ISA_ROOT` 替换它；切换 Python 后应重新确认实际 pin 与构建元数据。

实际评测的 platform/device、Python、setup scripts、环境导出由 `case.json.environment` 选择的 profile 决定；Case 额外 runtime 参数在 `case.json.runtime`。Benchmark 默认 **5 warmup / 10 measured rounds**，Case 可以覆盖。它与 PyPTO-Lib Golden 的通用 **5 / 100** 默认不同。Evaluator 会设置 `PYPTO_BENCH`、轮次和 `PYPTO_BENCH_RESET_INPUTS=each_launch` 等执行参数。

## Agent 工具

PTOBench Launcher 支持以下 backend，本次仅核对本机命令和版本。backend 的认证、provider、模型选择及实际任务成功率由主机配置和 Session 记录确定。

| backend | 可执行命令 | 本机版本 |
| --- | --- | --- |
| `codex` | `codex` | `codex-cli 0.159.2` |
| `claudecode` | `claude` | `2.1.288` |
| `pi` | `pi` | `1.0.0` |
| `opencode` | `opencode` | `1.18.34` |

四个命令当前位于 `/data1/home/ziruiwang/.local/npm/bin/`。具体 argv 和 profile 以方法侧的 `modes/_shared/profiles/` 为准。

## 已验证的行为与已知问题

| 项目 | 2026-10-05 检查结果 |
| --- | --- |
| 官方环境 PyPTO/simpler/Torch/Golden 导入 | 通过 |
| 官方环境 `uv pip check` | 通过，检查 venv 内 26 个包 |
| 官方环境 PyPTO-Lib `tests/golden` | **308 passed / 1 failed** |
| 失败原因 | Torch 2.6.0 缺少 `torch.float8_e8m0fnu`，影响当前 MX/E8M0 路径 |
| 旧环境 `tests/golden` | **241 passed / 68 failed**，主要是新旧 API 漂移 |
| 旧环境包依赖检查 | `nvidia-cusparselt-cu13` 平台不兼容 |
| 使用当前接口构造的最小内核 | `a2a3sim`、`a5sim` 均 compile/runtime/validate PASS |
| Golden 保存与 `runtime_dir` / `golden_data` 回放 | PASS |
| 回放后的 Benchmark | PASS；实测 3 measured rounds + 1 warmup，先 validate 后 benchmark |
| V4.1 C1A reindex 的小型 CPU golden | PASS |
| PTOBench Launcher / Agent CLI | 使用官方解释器运行 `--help` 通过 |
| 完整模型、HCCL 多卡、真实权重、A5 实机 | 本次没有重新认证 |

另有三项需要维护时留意：

1. 旧 `.venv` 的来源树要求 PTOAS **0.54** 和 ISA `83d01313d9bfc247c4b7c8bcf969d1019f0d106f`，与当前 shell 的官方工具链不同。PyPTO-Lib 本地 CI 文件仍固定 PTOAS **0.43** / ISA `8bd3ac8f30bd237f9eaf12c142002a5cc0edb143`，不能作为当前官方环境的版本证明。
2. 仓库入门 hello-world 和现存 10 个目录示例仍含旧 Golden 调用，hello-world 实测前端失败；它们不能作为新环境安装成功的验收入口。应使用已经按当前接口适配的 Case 或最小内核。
3. In-Core 指令级 profiling 与 PyPTO 功能模拟器不同。`clean_sim_trace`、依赖和 scope-stats 分析模块可以定位，但本次尚未闭合 `msopprof` worker、注入库和 PTOAS testcase generator 的采集前提；不能从功能模拟器 PASS 推导指令级采集可用。

## 当前主机的调用方式

以下命令从 PyPTO-Lib 根目录执行，路径对应本机当前安装：

```bash
export PTOBENCH_ROOT="$PWD/benchmarks/PTOBench"
export PTOBENCH_REPO_ROOT="$PWD"
export PTOBENCH_PTOAS_ROOT=/data1/home/ziruiwang/ptoas-v0.60
export PTOBENCH_CANN_ROOT=/usr/local/Ascend/cann-9.0.0/cann
export PTOBENCH_MINIFORGE_ROOT=/data1/home/ziruiwang/miniforge3

# 初始进程明确使用具有 Torch/PyPTO/PyYAML 的解释器。
.venv-official/bin/python "$PTOBENCH_ROOT/pto-launcher" --help
.venv-official/bin/python "$PTOBENCH_ROOT/ptobench" --help

# 按所选实验的 Case profile 和已分配设备启动。
.venv-official/bin/python "$PTOBENCH_ROOT/pto-launcher" run \
  "$PTOBENCH_ROOT/examples/experiment.yaml"
```

快照保存当前安装状态；迁移主机或重建环境时，先固定 PyPTO revision、对应 simpler 子模块、CPU Torch 的所需 dtype 能力，再读取 `toolchain/versions.env` 和 `runtime/pto_isa.pin` 获取匹配 PTOAS/ISA。不要只凭 `pypto==0.1.0`、目录名称或一个共享 PATH 重建整个环境。

原始核查证据保存在 PyPTO-Lib 的 `test_results/official_docs_audit_20261005/`，包括两套测试日志、模拟器运行日志、依赖与来源比对。本目录的版本快照保留环境盘点和验证摘要，原始测试日志仍在上述位置。
