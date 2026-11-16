> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# Backend

Per-architecture behaviour, kept out of the passes.

Passes never branch on `BackendType`. Everything architecture-specific — codegen
target, runtime API names, hazard workarounds, cross-core layout rules — is answered
by a `BackendHandler` obtained from the active `PassContext`. Adding an architecture
means adding a handler, not editing passes.

| Page | What it covers |
| ---- | -------------- |
| [BackendHandler: principled backend dispatch](source-02_pypto--en--dev--backend--00-backend_handler.md) | The virtual interface, how passes query it, and what adding a new backend requires |

## See Also

- [Pass, PassContext, PassPipeline, and PassManager](source-02_pypto--en--dev--passes--00-pass_manager.md) — where the handler comes from.
- [PTO ISA reference](../npu-architecture-and-isa/source-02_pypto--en--reference--index.md) — the hardware differences the handlers abstract over.
