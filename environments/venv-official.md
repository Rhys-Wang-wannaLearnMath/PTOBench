# PTOBench `.venv-official` 环境与复现

核查日期：2026-11-02（Asia/Shanghai）。这是当前 PTOBench 实际使用的应用环境，
不是当前 shell 默认的 PTOAS Python，也不是旧 `.venv`。

记录依据为 Case 环境配置、实际解释器的包元数据和安装文件、已安装 API 的签名与 CPU
解析检查，以及只读 `npu-smi` 查询。随后补齐已有 GCC 15 的用户命令入口，并验证普通
CPU C/C++ 编译、链接和 Python 加载；没有重新安装应用环境、运行模型或占用 NPU。

## 当前实际配置

以下相对路径均以 PyPTO-Lib 根目录为基准。

| 项目 | 当前值 |
| --- | --- |
| 应用 Python | `/data1/home/ziruiwang/pypto-lib/.venv-official/bin/python` |
| Python | CPython **3.10.20**，Linux aarch64；uv 0.11.8 创建的 venv |
| 基础解释器目录 | `/data1/home/ziruiwang/.local/share/uv/python/cpython-3.10-linux-aarch64-gnu/bin` |
| venv 隔离 | `include-system-site-packages=false`；没有 pip 模块，使用 `uv pip --python ...` |
| PyPTO | **0.1.0**，普通安装，非 editable；接口为 `pypto.language` |
| PyPTO 来源 | `747e1e4ba99f13dea1a609d893ac7512e85a5897` |
| simpler | **0.1.0**；同一 PyPTO checkout 的 `runtime/` 子模块 |
| simpler 来源 | `4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0` |
| PTO ISA pin | `a8040450238f162985d8b596fbebeb54bfba2bf5`，由 simpler 管理 |
| Torch | **2.6.0+cpu**；没有 `torch_npu`，NPU 执行走 PyPTO/simpler |
| NumPy / PyYAML | **2.2.6 / 6.0.3** |
| CANN | **9.0.0**；profile source `/usr/local/Ascend/cann-9.0.0/cann/set_env.sh` |
| 实际 CANN 根 | `/usr/local/Ascend/cann-9.0.0/cann-9.0.0` |
| PTOAS | **0.60**；`/data1/home/ziruiwang/ptoas-v0.60/bin/ptoas` |
| C++ 动态库 | `/data1/home/ziruiwang/miniforge3/lib/libstdc++.so.6.0.34` |
| 构建工具（venv 内） | CMake **4.4.3**、Ninja **1.13.2**、nanobind **2.15.0**、scikit-build-core **1.0.3** |
| 主机 C/C++ 编译器 | `/usr/bin/gcc` / `g++` **10.3.1**；`~/.local/bin/gcc-15` / `g++-15` 均为 **15.2.0**，指向既有 `miniforge3/envs/gcc15` |
| 硬件 | **8 × Ascend 910B4**，每卡 **32 GiB HBM**，驱动 **25.5.0** |
| 实机平台 | `a2a3`；本机没有 A5/Ascend950 实机 |

## 2026-11-03 固定与校验

完整锁定清单见 [`LOCK.json`](LOCK.json)。它记录 Python、PyPTO、simpler、Torch、
PyYAML、PTOAS、CANN、PTO ISA、runtime 源码工作树补丁、已安装 simpler runtime 库和
所有 Case profile 的内容哈希。`evaluator/environment_lock.py` 在 Launcher preflight
和 evaluator re-exec 后检查这些值；任何错配都会在 Agent 或 NPU 工作开始前失败。

PTOBench 不添加 runtime host lifecycle trace，也不对设备 `effective_us` 做启动错位扣除。
正式评分直接使用同一 Evaluator 生成的原始逐轮 `effective_us` 样本；整项 Agent Session
的耗时另记录在 `final_result.json`，与 NPU 性能分开。

2026-11-03 已从同一固定源码重建并重装 `simpler==0.1.0`，没有升级依赖或切换环境；
`LOCK.json` 已同步 runtime 源码、安装二进制和当前 PyPTO 来源树的哈希。PyPTO 安装文件及
包版本未变。锁校验与 `DataType` 导入通过，锁定 runtime Python 测试通过 **2120 项**。

后续不得使用旧 `.venv`、PTOAS 默认 Python、其他 PyPTO checkout 或未通过
`LOCK.json` 的 runtime 库进行正式 Case、baseline 或 Agent evaluate。环境更新必须先
显式生成新的锁文件并重新认证，当前流程不会自行升级任何依赖。

全部 **87** 个当前环境 profile 都解析到这一 `.venv-official/bin/python`，平台均为
`a2a3`。其中 80 个使用相对 Python/工具链占位符，7 个仍写本机绝对路径；后者迁移时必须
相应替换。device、设备集合、runtime 和 benchmark 轮次仍由各自 Case/profile 决定。

完整 26 个 Python 发行包、795 个 PyPTO/simpler 安装文件的 SHA-256、来源和 profile hash
记录在 [机器可读快照](reproducibility/venv-official-20261021.json)。
[固定 Python 依赖](reproducibility/venv-official-python.txt) 列出除 PyPTO/simpler/Torch
之外的 23 个包；这三项按下面的固定来源单独安装，避免从 PyPI 误装同名包或 CUDA Torch。

本次对安装的 PyPTO 136 个、simpler 21 个可映射的 Python/元数据文件逐一与来源 commit
比较，全部一致；快照还保存安装的原生库 hash。来源 checkout 当前可能有未提交修改，
重建时必须使用干净的固定 commit。包版本 `0.1.0` 和来源目录名称都不能替代内容校验。

## 参数从哪里来

`case.json.environment` → `environments/<profile>.json` →
`evaluator/common.py` 的 `load_profile_environment()` / `bootstrap_environment()`。

实际过程是：source profile 的 CANN 脚本，应用 profile 的 prepend/env，再 re-exec 到指定
Python。NPU profile 会将现有 Miniforge `lib/` 放在 `LD_LIBRARY_PATH` 首位，供 PTOAS
加载所需的 `GLIBCXX_3.4.29`。因此仅 `source .venv-official/bin/activate` 不足以重现评测。
Launcher 和 Evaluator 共用这套环境加载规则。

| PTOBench 配置 | 本机默认值 | 含义 |
| --- | --- | --- |
| `PTOBENCH_REPO_ROOT` | `/data1/home/ziruiwang/pypto-lib` | 应用源码与相对 Python 路径根 |
| `PTOBENCH_CANN_ROOT` | `/usr/local/Ascend/cann-9.0.0/cann` | profile 的 `{cann_root}` |
| `PTOBENCH_PTOAS_ROOT` | `/data1/home/ziruiwang/ptoas-v0.60` | profile 的 `{ptoas_root}` |
| `PTOBENCH_MINIFORGE_ROOT` | `/data1/home/ziruiwang/miniforge3` | 兼容 C++ 动态库目录 |

source 后的关键有效值：

```text
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0/cann-9.0.0
PTOAS_ROOT=/data1/home/ziruiwang/ptoas-v0.60
PYTHONPATH=/data1/home/ziruiwang/pypto-lib
PATH: PTOAS_ROOT/bin 在 CANN 工具目录之前，随后保留原 PATH
LD_LIBRARY_PATH: miniforge3/lib 在 CANN 和 driver 库目录之前
```

完整有效路径见快照；其中 shell 的临时 Codex PATH 等属于继承状态，无需为复现硬编码。
`PTO_ISA_ROOT` 的 ambient 值不能替换 simpler 安装内的 pin。不要另选一份 ISA 或将旧
`.venv`、PTOAS 独立环境与应用环境混用。继承的显卡可见性变量也不等于设备已分配。

## 在本机使用现有环境

从 PyPTO-Lib 根目录执行：

```bash
export PTOBENCH_REPO_ROOT="$PWD"
export PTOBENCH_ROOT="$PWD/benchmarks/PTOBench"
export PTOBENCH_CANN_ROOT=/usr/local/Ascend/cann-9.0.0/cann
export PTOBENCH_PTOAS_ROOT=/data1/home/ziruiwang/ptoas-v0.60
export PTOBENCH_MINIFORGE_ROOT=/data1/home/ziruiwang/miniforge3

.venv-official/bin/python "$PTOBENCH_ROOT/pto-launcher" --help
.venv-official/bin/python "$PTOBENCH_ROOT/ptobench" --help
```

实际实验通过 Launcher 使用 Case 的 profile。不得用 shell 当前的 `python` 是否能 import
PyPTO 来判断 profile 是否正确；当前默认 Python 指向 PTOAS 环境，它没有 PyPTO/Torch。

Runtime Python 回归测试使用锁定的解释器和 runtime checkout，并通过固定入口清除继承的
`PYTHONPATH`：

```bash
.venv-official/bin/python "$PTOBENCH_ROOT/environments/reproducibility/run_locked_runtime_python_tests.py"
```

不要给这组测试设置 `PYTHONPATH=python`。该目录含 `pypto_core` 的类型声明，不含二进制
扩展；放在导入路径前面会遮住 `.venv-official` 已安装的 `pypto.pypto_core`，造成
`DataType` 导入失败。固定入口从环境 profile 和 `LOCK.json` 读取 Python 与 runtime 路径，
仅对测试子进程移除 `PYTHONPATH`，不影响 PTOBench Case 的环境配置。

## GCC 15 工具链与命令入口

GCC 15 的 C/C++ 编译器原本都已安装在
`/data1/home/ziruiwang/miniforge3/envs/gcc15/bin/`，命名为
`aarch64-conda-linux-gnu-gcc` / `aarch64-conda-linux-gnu-g++`。之前仅创建了
`~/.local/bin/g++-15`，遗漏 `gcc-15` 入口，曾被记录为重建限制；现已补齐：

```text
~/.local/bin/gcc-15 -> /data1/home/ziruiwang/miniforge3/envs/gcc15/bin/aarch64-conda-linux-gnu-gcc
~/.local/bin/g++-15 -> /data1/home/ziruiwang/miniforge3/envs/gcc15/bin/aarch64-conda-linux-gnu-g++
```

修复只新增第一个软链接，没有安装/升级包，也没有替换系统默认 `gcc/g++`。
使用实际 Case 加载环境、已安装 simpler 的 `Gxx15Toolchain.get_cmake_args()`，以及
venv 内 CMake 4.4.3 / Ninja 1.13.2，已验证 C11、C++23 `std::expected`、混合链接、
可执行文件及 `.venv-official` Python 的共享库加载。详见
[编译器验证记录](reproducibility/gcc15-verification-20261021.json)。

换机重建工具链时，可用 [19 个包的精确清单](reproducibility/gcc15-linux-aarch64-explicit.txt)
固定 conda-forge 的包 URL 和 SHA-256。以下在新目录创建工具链，命令入口仅用于该构建：

```bash
export PYPTO_REPRO_TOOLCHAIN="$PWD/gcc15-reproduced"
conda create --prefix "$PYPTO_REPRO_TOOLCHAIN" \
  --file "$PTOBENCH_ROOT/environments/reproducibility/gcc15-linux-aarch64-explicit.txt"
export PYPTO_REPRO_TOOL_BIN="$PYPTO_REPRO_TOOLCHAIN/ptobench-bin"
mkdir -p "$PYPTO_REPRO_TOOL_BIN"
ln -s "$PYPTO_REPRO_TOOLCHAIN/bin/aarch64-conda-linux-gnu-gcc" "$PYPTO_REPRO_TOOL_BIN/gcc-15"
ln -s "$PYPTO_REPRO_TOOLCHAIN/bin/aarch64-conda-linux-gnu-g++" "$PYPTO_REPRO_TOOL_BIN/g++-15"
export PATH="$PYPTO_REPRO_TOOL_BIN:$PATH"
gcc-15 --version
g++-15 --version
```

本机现有入口已可用，无需执行上述重建。该清单适用于 Linux aarch64，生成工具为
conda 26.3.2（支持 SHA-256 explicit 清单）；新环境的完整重建尚未执行。

## 在新目录重建同一软件版本

下面是基于固定来源的重建步骤，本次没有执行安装。新目录不能覆盖正在使用的 venv。
先准备同架构 Linux、CANN 9.0.0、兼容驱动和 C++17 编译器；原生扩展构建还需要 CMake、
Ninja、nanobind。固定 simpler commit 的默认安装会自动构建可检测到的 runtime，包含
模拟器；其模拟内核使用 **g++-15 / C++23**，CMake 默认选择 **gcc-15 / g++-15**。
因此完整默认重建需要两者；本机的命令入口已补齐并通过上面的 CPU 编译检查。
普通 GCC 10 与 GCC 15 应分别记录。完整默认 simpler/PyPTO 源码重建仍未执行。

```bash
# 在放置源码和环境的新父目录执行；PTOBENCH_ROOT 指向现有 PTOBench。
export PYPTO_REPRO_SOURCE="$PWD/pypto-747e1e4b"
export PYPTO_REPRO_VENV="$PWD/venv-official-reproduced"

# PTOBENCH_CANN_ROOT / PTOBENCH_MINIFORGE_ROOT 按上表设置；换机使用新主机对应路径。
source "$PTOBENCH_CANN_ROOT/set_env.sh"
export LD_LIBRARY_PATH="$PTOBENCH_MINIFORGE_ROOT/lib:${LD_LIBRARY_PATH:-}"

git clone --no-checkout https://github.com/hw-native-sys/pypto.git "$PYPTO_REPRO_SOURCE"
git -C "$PYPTO_REPRO_SOURCE" checkout --detach 747e1e4ba99f13dea1a609d893ac7512e85a5897
git -C "$PYPTO_REPRO_SOURCE" submodule update --init --recursive
# 这两项应分别为表中的 simpler commit 和 ISA pin。
git -C "$PYPTO_REPRO_SOURCE/runtime" rev-parse HEAD
cat "$PYPTO_REPRO_SOURCE/runtime/pto_isa.pin"

uv venv --python 3.10.20 "$PYPTO_REPRO_VENV"
uv pip install --python "$PYPTO_REPRO_VENV/bin/python" \
  'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PYPTO_REPRO_VENV/bin/python" \
  -r "$PTOBENCH_ROOT/environments/reproducibility/venv-official-python.txt"

# 使用刚固定的构建工具；完整默认构建前确认 GCC 15 工具链已准备好。
export PATH="$PYPTO_REPRO_VENV/bin:$PATH"
command -v gcc-15 g++-15
gcc-15 --version
g++-15 --version

# 普通安装，使用已固定的构建依赖，避免隔离构建另行升级依赖。
uv pip install --python "$PYPTO_REPRO_VENV/bin/python" --no-build-isolation --no-deps \
  "$PYPTO_REPRO_SOURCE/runtime" "$PYPTO_REPRO_SOURCE"
uv pip check --python "$PYPTO_REPRO_VENV/bin/python"
```

新环境不会自动替换 Case 的解释器选择。若在新的 PyPTO-Lib checkout 直接复用相对路径
profiles，应把新 venv 建在该 checkout 的 `.venv-official`；若保留其他 venv 路径，应在
独立的目标 checkout 新建 profile，明确设置其 `python` 并让目标 Case 引用它。不要只改
shell PATH，也不要为试用新环境覆盖本机正在使用的共享 profile。

PTOAS 保持为独立环境。aarch64/CPython 3.10 的原安装 wheel 名为
`ptoas-0.60-cp310-cp310-manylinux_2_34_aarch64.whl`，来自固定 v0.60 发布。
其官方 SHA-256 由该 PyPTO commit 的 `toolchain/versions.env` 给出：

```text
7ebc8bb45fd5a5a0b55c918227fe2194a7e33a7f3bd4b5ab64005a51717cb08e
```

在新路径安装该 wheel 后，设置 `PTOBENCH_PTOAS_ROOT` 指向其 venv。CANN 与驱动需要使用
匹配架构的厂商安装包单独准备，Python requirements 无法安装它们。使用同版本的兼容
libstdc++，并将 `PTOBENCH_MINIFORGE_ROOT` 指向对应根目录。迁移 profile 中的 7 个绝对路径；
设备和多卡拓扑按新主机实际情况重新分配，不能照搬旧设备号。

这些步骤固定版本和源码语义；本次没有保存原 wheel 的构建命令或完整构建容器，因而不能
保证重新编译的 `.so` 字节与快照一致。需要字节级恢复时应保存原 wheel/环境镜像并核对
快照中的安装文件 hash，不能把一次源码重编译称为二进制完全复现。

## 检查环境是否漂移

环境 profile、工具链和安装 wheel 的校验仍以本文件前面的固定版本说明为准。历史
Sluice 审计脚本随旧实现一并归档，不再作为 PTOBench 启动或验证步骤。

本次 `uv pip check` 的 26 个包检查通过。Torch 2.6.0 缺少部分新 dtype 的历史限制仍不能
靠包可导入来排除；本次没有重跑 Golden、完整模型或实机认证。所有性能和正确性结果仍以
具体 Case 的公开 evaluate 为准。2026-10-05 的测试记录保留在 [旧环境说明](../ENVIRONMENT.md)，
不能解释为本次重新运行的结果。
