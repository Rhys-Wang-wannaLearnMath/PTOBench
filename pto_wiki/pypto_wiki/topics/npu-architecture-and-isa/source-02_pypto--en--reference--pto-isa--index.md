> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# PTO ISA

The hardware model and instruction semantics that PyPTO's generated code targets.

| Page | What it covers |
| ---- | -------------- |
| [Cluster Architecture](source-02_pypto--en--reference--pto-isa--00-cluster_architecture.md) | The 1 Cube + 2 buddy Vector core cluster and its flag-based synchronization |
| [TPUSH/TPOP Instructions](source-02_pypto--en--reference--pto-isa--01-tpush_tpop.md) | Moving tiles between InCore kernels co-scheduled on Cube and Vector cores |
| [Buffer Management](source-02_pypto--en--reference--pto-isa--02-buffer_management.md) | Where the TPUSH/TPOP ring buffer lives per platform — GM on A2/A3, consumer on-chip memory on A5 |

## See Also

- [SkewCrossCorePipeline Pass](../compiler-and-ir-internals/source-02_pypto--en--dev--passes--29-skew_cross_core_pipeline.md) — the pass that software-pipelines cross-core loops onto this architecture.
- [InjectGMPipeBuffer Pass](../compiler-and-ir-internals/source-02_pypto--en--dev--passes--25-inject_gm_pipe_buffer.md) — the GM-routed cross-core pipe workspace on Ascend910B.
