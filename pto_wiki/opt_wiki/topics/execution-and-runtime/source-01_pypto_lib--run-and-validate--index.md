> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# Run and Validate

Most runnable PyPTO-Lib scripts use the local Golden Harness to turn a kernel
definition into a correctness result. A few specialized smoke, regeneration,
or external-runtime drivers call PyPTO's compile/runtime APIs directly; inspect
the selected script's `__main__` block and `--help` before assuming that it
performs golden validation.

The common flow is:

```text
PyPTO compile
    ↓
input generation
    ↓
PyTorch golden computation
    ↓
simpler runtime execution
    ↓
output validation
```

Use the following references:

- [Compile and Runtime Workflow](source-01_pypto_lib--run-and-validate--compile-runtime-workflow.md) explains the
  generated artifacts and each stage in detail.
- [Golden Harness](source-01_pypto_lib--run-and-validate--golden-harness.md) introduces `TensorSpec`, `ScalarSpec`,
  `run`, validation, and `RunResult`.
- [Save and Replay Golden Data](source-01_pypto_lib--run-and-validate--save-and-replay.md) freezes inputs and expected
  outputs for repeated performance-oriented runs.

For a first end-to-end command, start with
[Run Your First Kernel](../setup-and-first-kernels/source-01_pypto_lib--get-started--first-kernel.md).
