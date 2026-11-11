> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# Reference

Hardware and instruction-set material behind the code PyPTO generates.

Read this when you are looking at generated PTO code, tuning a cross-core pipeline,
or reasoning about why a pass lowers something the way it does. For the language you
write, see the [User Manual](../setup-and-first-kernels/source-02_pypto--en--user--index.md); for how the compiler transforms it,
see the [developer documentation](../compiler-and-ir-internals/source-02_pypto--en--dev--index.md).

## PTO ISA

| Page | What it covers |
| ---- | -------------- |
| [Cluster Architecture](source-02_pypto--en--reference--pto-isa--00-cluster_architecture.md) | The 1 Cube + 2 buddy Vector core cluster and its flag-based synchronization |
| [TPUSH/TPOP Instructions](source-02_pypto--en--reference--pto-isa--01-tpush_tpop.md) | Moving tiles between InCore kernels co-scheduled on Cube and Vector cores |
| [Buffer Management](source-02_pypto--en--reference--pto-isa--02-buffer_management.md) | Where the TPUSH/TPOP ring buffer lives per platform — GM on A2/A3, consumer on-chip memory on A5 |

## See Also

- [PTO Project Ecosystem](../compiler-and-ir-internals/source-02_pypto--en--dev--00-ecosystem.md) — how PyPTO, PTOAS, pto-isa, and the runtime fit together.
- [PTO Codegen](../compiler-and-ir-internals/source-02_pypto--en--dev--codegen--00-pto_codegen.md) — how PyPTO IR becomes PTO-ISA dialect MLIR.
- [PTOAS Op Status Matrix](../compiler-and-ir-internals/source-02_pypto--en--dev--ptoas-op-status.md) — which PTOAS ops the compiler currently emits.
