> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# Tools

What to reach for when the result is not what you expected.

Four questions, four tools. The order matters — each one is cheaper than the next, and the
compiler has usually answered the first before you ask it.

| Question | Tool | Page |
| -------- | ---- | ---- |
| What went wrong, and where? | Error types, log levels, IR dumps | [Debugging](source-02_pypto--en--user--tools--00-debugging.md) |
| Is the **IR** wrong, or the device? | `pypto.debug.torch_codegen` | [Torch codegen](source-02_pypto--en--user--tools--01-torch-codegen.md) |
| What is on chip, and for how long? | `pypto.tools.memory_map` | [Memory map](source-02_pypto--en--user--tools--02-memory-map.md) |
| Where did the time go? | The L2 swimlane | [Performance](../performance-and-optimization/source-02_pypto--en--user--performance--00-swimlane.md) |

## The cheap checks first

Two things cost nothing to read, and the compiler has already produced both:

- **`report/perf_hints.log`**, written on every compile — what the compiler noticed but did
  not refuse: transfers below the hardware granularity, a matmul it could not tile, a
  pipeline depth that did not fit. One summary line also goes to stderr.
- **The error message**, when there is one. PyPTO distinguishes a user error from an
  internal one, and the distinction tells you whether to fix your code or file a bug — see
  [Debugging](source-02_pypto--en--user--tools--00-debugging.md).

## See Also

- [Precision](source-02_pypto--en--user--precision--index.md) — the workflow these tools serve when numbers are wrong.
- [Performance](../performance-and-optimization/source-02_pypto--en--user--performance--index.md) — the same, when the numbers are right but slow.
- [Execution](../execution-and-runtime/source-02_pypto--en--user--execution--index.md) — the compile and dispatch surface being debugged.
