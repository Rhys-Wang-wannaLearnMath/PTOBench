> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# PyPTO Coding

Use this chapter when writing or reviewing PyPTO-Lib kernels. The first six
pages are the canonical coding style, split by what you are writing:

| Page | Covers |
|---|---|
| [L2 Programming](source-01_pypto_lib--pypto-coding--l2-programming.md) | The two authoring forms, `pl.Out` / `pl.InOut` directions, `pl.at` regions and their `optimizations`, mixed cube + vector regions, and dynamic B / S shapes |
| [Operations](../api-and-operations-reference/source-01_pypto_lib--pypto-coding--operations.md) | Vector, cube, MTE and scalar ops — the four families a kernel body is written from |
| [Loops](source-01_pypto_lib--pypto-coding--loops.md) | `pl.range`, `pl.unroll`, `pl.parallel`, `pl.pipeline`, `pl.spmd`, and where each one is legal |
| [L3 Programming](../distributed-programming/source-01_pypto_lib--pypto-coding--l3-programming.md) | Multi-card kernels: the host driver, HCCL window buffers, cross-rank data movement, and the notify / wait protocols that order it |
| [Golden and Run](../execution-and-runtime/source-01_pypto_lib--pypto-coding--golden-and-run.md) | Writing a kernel's validation: specs, the Torch reference, the `run` call, and the conventional CLI flags |
| [Naming and Comments](source-01_pypto_lib--pypto-coding--naming-and-comments.md) | Constant naming, what a comment may say, and where allocations go |

One further page covers work that leaves PyPTO entirely:

- [CCE Extern Kernel](source-01_pypto_lib--pypto-coding--cce-extern-kernel.md) covers hand-written mixed cube and
  vector kernels called through `pl.jit.extern`, including runtime, ABI,
  synchronization, and validation constraints.

Start with L2 Programming for every kernel change. Add L3 Programming when the
kernel spans more than one card. Use the extern-kernel page only when the
implementation crosses from PyPTO into hand-written CCE code.
