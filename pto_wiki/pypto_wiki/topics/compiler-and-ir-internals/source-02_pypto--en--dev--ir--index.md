> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# IR

PyPTO's intermediate representation — the contract every other layer depends on.

The IR definition is the source of truth for the whole compiler: passes are
replaceable, IR node definitions are not. Read [Overview](source-02_pypto--en--dev--ir--00-overview.md) first, then
the page covering what you are changing.

| Page | What it covers |
| ---- | -------------- |
| [Overview](source-02_pypto--en--dev--ir--00-overview.md) | The compilation pipeline from Python DSL to generated kernels, and where the IR sits in it |
| [Node Hierarchy](source-02_pypto--en--dev--ir--01-hierarchy.md) | Complete reference of every IR node type, organized by category |
| [Types and Examples](source-02_pypto--en--dev--ir--02-types.md) | The type system with practical usage examples |
| [Structural Comparison](source-02_pypto--en--dev--ir--03-structural_comparison.md) | Comparing IR nodes by structure rather than pointer identity |
| [Serialization](source-02_pypto--en--dev--ir--04-serialization.md) | MessagePack-based `.pto` serialization |
| [Operator System](source-02_pypto--en--dev--ir--05-operators.md) | Type-safe operator definitions with automatic type deduction |
| [IR Builder](source-02_pypto--en--dev--ir--06-builder.md) | Constructing IR incrementally — context managers in Python, Begin/End in C++ |
| [IR Parser](source-02_pypto--en--dev--ir--07-parser.md) | Converting Python DSL to IR via `@pl.function` / `@pl.program`, and the SSA properties it enforces |
| [Parameter Directions](source-02_pypto--en--dev--ir--08-param-directions.md) | How `In`/`Out`/`InOut` is inferred — the registry declaration every stage reads, and the four passes that build on it |
| [Multi-Output Operators](source-02_pypto--en--dev--ir--09-multi_output_ops.md) | Operators that produce several values: `TupleType` results, why destinations never become arguments, and what the registry enforces |

## See Also

- [Passes](source-02_pypto--en--dev--passes--index.md) — the transformations that run over this IR.
- [Python IR Syntax Specification](source-02_pypto--en--dev--language--00-python_syntax.md) — the surface syntax the parser accepts.
- [IR Verifier](source-02_pypto--en--dev--passes--99-verifier.md) — the property verifiers that check IR legality between passes.
