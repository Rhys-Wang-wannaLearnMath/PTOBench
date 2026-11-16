> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](source-BASELINE.md).

# PyPTO-Lib

PyPTO-Lib holds [PyPTO](source-02_pypto--en--index.md) kernels and end-to-end LLM
models for Ascend NPUs, with the harness that compiles, runs, and validates
them.

Use this documentation to move from a first simulator run to validated model
kernels and systematic precision or performance tuning.

## Choose a path

### Run a kernel

Start with [installation and environment setup](source-01_pypto_lib--get-started--installation.md),
then [run your first kernel](source-01_pypto_lib--get-started--first-kernel.md).

### Write a kernel

Use the [PyPTO Coding](../kernel-authoring-and-tutorials/source-01_pypto_lib--pypto-coding--index.md) chapter for the canonical kernel
style and hand-written CCE extern-kernel conventions.

### Examples

Use the [example catalog](source-01_pypto_lib--examples--index.md) for focused, self-contained
kernels organized by learning level.

### Models

Use the [model pages](../model-implementations/source-01_pypto_lib--models--index.md) for end-to-end and component-level model
implementations.

### Run and Validate

Read the
[compile and runtime workflow](../execution-and-runtime/source-01_pypto_lib--run-and-validate--compile-runtime-workflow.md)
and the
[Golden Harness overview](../execution-and-runtime/source-01_pypto_lib--run-and-validate--golden-harness.md) to understand
how a script compiles, executes, and checks its result.

### Diagnose and optimize

Begin with the [debugging playbook](../debugging-correctness-and-precision/source-01_pypto_lib--debug-and-tune--debugging.md), then choose
the [precision](../debugging-correctness-and-precision/source-01_pypto_lib--debug-and-tune--precision-tuning.md) or
[performance](../performance-and-optimization/source-01_pypto_lib--debug-and-tune--performance-tuning.md) workflow for the problem
at hand.
[DeepSeek V4 decode optimization](../model-implementations/source-01_pypto_lib--models--deepseek_v4_flash_mtp--decode_optimization.md)
follows one model's decode path end to end and records which levers paid.
[Qwen3-14B optimization](../model-implementations/source-01_pypto_lib--models--qwen3_14b--optimization.md) records how a
single-card dense model's kernels were tuned, in the order the work happened.

## Ecosystem

- [PyPTO documentation](source-02_pypto--en--index.md) covers the programming
  model, language semantics, and compiler.
- simpler documentation (`https://www.pypto.ai/simpler/`) covers the runtime.
- PyPTO-Lib on GitHub (`https://github.com/hw-native-sys/pypto-lib`) contains
  the source code and issue tracker.
