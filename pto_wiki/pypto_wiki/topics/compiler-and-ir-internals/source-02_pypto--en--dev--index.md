> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# Developer Documentation

How PyPTO is built: the IR, the pass pipeline, code generation, and the
infrastructure around them.

This is documentation for people working *on* the compiler. If you are writing
PyPTO programs, start with the [User Manual](../setup-and-first-kernels/source-02_pypto--en--user--index.md).

## Sub-chapters

| Chapter | What it covers |
| ------- | -------------- |
| [IR](source-02_pypto--en--dev--ir--index.md) | Node hierarchy, type system, operators, builder, parser, serialization, structural comparison |
| [Passes](source-02_pypto--en--dev--passes--index.md) | The pass framework and every pass in the default pipeline, numbered in execution order |
| [Language](source-02_pypto--en--dev--language--index.md) | The Python DSL syntax specification and external C++ kernel integration |
| [Code Generation](source-02_pypto--en--dev--codegen--index.md) | Lowering IR to PTO-ISA dialect MLIR and to orchestration C++ |
| [Backend](source-02_pypto--en--dev--backend--index.md) | Per-architecture dispatch through `BackendHandler` |
| [Debug](../debugging-correctness-and-precision/source-02_pypto--en--dev--debug--index.md) | Lowering IR to an executable PyTorch script for numerical validation |

## Top-level topics

| Page | What it covers |
| ---- | -------------- |
| [PTO Project Ecosystem](source-02_pypto--en--dev--00-ecosystem.md) | The multi-repo toolchain — PyPTO, PTOAS, pto-isa, simpler, pypto-lib — and how they fit together |
| [Compile Profiling](../performance-and-optimization/source-02_pypto--en--dev--01-compile-profiling.md) | Built-in wall-clock timing of the compilation pipeline |
| [Error Handling](../debugging-correctness-and-precision/source-02_pypto--en--dev--02-error-handling.md) | `CHECK` vs `INTERNAL_CHECK`, PyPTO exception types, IR source locations in failures |
| [Logging](../debugging-correctness-and-precision/source-02_pypto--en--dev--03-logging.md) | The two independent logging subsystems and which one a message came from |
| [Runtime DFX Flags](../debugging-correctness-and-precision/source-02_pypto--en--dev--03-runtime-dfx.md) | The five runtime diagnostic sub-features exposed through `RunConfig` |
| [Replaying an Existing `build_output`](../debugging-correctness-and-precision/source-02_pypto--en--dev--03-runtime-replay.md) | Re-run, edit, and re-measure a compiled build directory without recompiling |
| [Simulator Trace Cleaning](../debugging-correctness-and-precision/source-02_pypto--en--dev--04-simulator-trace-cleaning.md) | Converting MindStudio Insight binary dumps into readable traces |
| [Per-Task Ring Sizing](../execution-and-runtime/source-02_pypto--en--dev--05-runtime-ring-sizing.md) | The three ring-size overrides on `RunConfig` and when to tune them |
| [Persistent L3 execution](../execution-and-runtime/source-02_pypto--en--dev--06-persistent-l3.md) | Reusing one worker across prepared distributed programs |
| [Memory Map](../debugging-correctness-and-precision/source-02_pypto--en--dev--07-memory-map.md) | Rendering a pass dump into an interactive HTML map of on-chip memory |
| [Compile and Execution Entry Points](../execution-and-runtime/source-02_pypto--en--dev--08-entry-points.md) | Every compile and execution entry point, the layer it belongs to, and when to reach for it |
| [Distributed Operators](../distributed-programming/source-02_pypto--en--dev--distributed_ops.md) | The N6 distributed op family — typed DSL access to collectives and primitives |
| [PTOAS Op Status Matrix](source-02_pypto--en--dev--ptoas-op-status.md) | Which public and compatibility PTOAS ops the compiler currently emits |

## See Also

- [PTO ISA reference](../npu-architecture-and-isa/source-02_pypto--en--reference--index.md) — the hardware model the backend targets.
- Runtime documentation (`https://hw-native-sys.github.io/simpler/`) — the scheduler that executes compiled programs.
