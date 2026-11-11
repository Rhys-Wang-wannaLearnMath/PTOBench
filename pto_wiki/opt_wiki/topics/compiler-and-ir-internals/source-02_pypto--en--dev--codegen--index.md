> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# Code Generation

Lowering PyPTO IR to the two artifacts a compiled program consists of: device
kernels and the host-side orchestration that launches them.

Both generators follow the same design principle — a **strict 1-to-1 mapping** from
IR to generated code. Anything that needs a decision was already decided by a pass.

| Page | What it covers |
| ---- | -------------- |
| [PTO Codegen](source-02_pypto--en--dev--codegen--00-pto_codegen.md) | Generating PTO-ISA dialect MLIR from PyPTO IR |
| [Orchestration Code Generation](source-02_pypto--en--dev--codegen--01-orchestration_codegen.md) | Generating the host-side C++ that submits tasks to the runtime |

## See Also

- [Passes](source-02_pypto--en--dev--passes--index.md) — everything that must hold before codegen runs.
- [PTO ISA reference](../npu-architecture-and-isa/source-02_pypto--en--reference--index.md) — the instruction semantics the generated code uses.
- [PTOAS Op Status Matrix](source-02_pypto--en--dev--ptoas-op-status.md) — which PTOAS ops are available to emit.
