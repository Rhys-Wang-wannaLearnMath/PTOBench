# Benchmark Environments

当前本机版本、安装来源、工具链路径和已知兼容性问题见
[当前软件栈与运行环境](../ENVIRONMENT.md)。
[2026-10-05 版本快照](inventory/20261005.json) 是环境盘点记录，不是 Case runtime profile。

Each profile names the Python interpreter, target platform, CANN setup script,
PTOAS toolchain, and runtime options used by a Case. The evaluator loads the
profile before importing the Case evaluator, so `bench` commands use the same
environment regardless of the caller's current shell.

For NPU profiles, the common environment loader also prepends the existing
`{miniforge_root}/lib` when it contains `libstdc++.so.6`. PTOAS v0.60 requires
`GLIBCXX_3.4.29`; the host system library alone is insufficient. This applies to
Ref preparation, Agent evaluation and every profiling mode, without upgrading
any package. The loader runs `ptoas --version` before entering the evaluator,
so a compiler loading failure is reported before Ref certification is changed.
Standalone hosts must configure the compatible installed runtime through
`PTOBENCH_MINIFORGE_ROOT` or their profile's library path.

Case evaluator entry points keep their public config/helper imports scoped for
the whole invocation, including delayed imports during input construction.
Same-named modules are restored on exit so one Case cannot reuse another
Case's configuration. Agent Session creation requires a ready certification
and a baseline explicitly recording `PYPTO` with
`analyze_auto_scopes_for_deps=false`; pending refreshes are unavailable to Agents.

The paths in the checked-in profiles use `{repo_root}`, `{benchmark_root}`,
`{ptoas_root}`, `{cann_root}`, and `{miniforge_root}` placeholders. Nested use
in the pypto-lib checkout keeps the existing local defaults. A standalone
PTOBench checkout must set `PTOBENCH_REPO_ROOT` and, for NPU Cases,
`PTOBENCH_PTOAS_ROOT`, `PTOBENCH_CANN_ROOT`, and (when a profile uses it)
`PTOBENCH_MINIFORGE_ROOT`.

`a2a3_single_card.json` is the local Ascend 910B/910C profile. It follows the
upstream installation contract: Python 3.10, CANN 9.0.0, PyPTO and `simpler`
from the same checkout, and PTOAS v0.60. The PTO ISA revision is owned by the
installed `simpler` runtime pin; it is not selected by a benchmark Case.

The profile's `benchmark.warmup` and `benchmark.rounds` are benchmark policy,
not environment variables that an Agent can change. Case-specific runtime
fields such as ring sizing belong in that Case's `case.json` under `runtime`.

An A5 profile should be added separately when an Ascend 950 host and the
matching PyTorch/PyPTO/CANN toolchain are available. A5-only dtypes and Cases
must not be silently run through this A2/A3 profile.

During migration, reuse an existing profile only when its Python, setup scripts,
platform/device, toolchain exports and defaults match the Case. Otherwise create
a new Case-specific or platform-specific JSON and reference it from
`case.json.environment`; do not change a shared profile just to make one Case
pass. Confirm values against the current upstream CI/test setup.

Supported profile fields are:

```text
name, platform, device, python, setup_scripts, prepend_env, env,
runtime, benchmark.warmup, benchmark.rounds
```

Case-only RunConfig values belong in `case.json.runtime`. Validate them against
the `pypto.runtime.RunConfig` signature from the profile's Python interpreter.
