> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# Language

The Python DSL as the compiler sees it.

| Page | What it covers |
| ---- | -------------- |
| [Python IR Syntax Specification](source-02_pypto--en--dev--language--00-python_syntax.md) | Module structure, type system, expressions — the entry point to the spec |
| [Statements and Control Flow](source-02_pypto--en--dev--language--01-statements.md) | Assignment, if/for/while, scope context managers, yield, compile-time directives, SSA phi nodes |
| [Manual Dependency Primitives](../task-graphs-and-scheduling/source-02_pypto--en--dev--language--02-manual_dependencies.md) | `pl.manual_scope`, explicit `deps=` edges, dispatch predicates, array-carry fences |
| [Functions and Program Structure](source-02_pypto--en--dev--language--03-functions.md) | Function types, parameter directions, cross-module reuse, printing IR |
| [Integrating Hand-Written C++ Kernels](source-02_pypto--en--dev--language--04-external-kernels.md) | Calling an existing hand-written C++ InCore kernel from a PyPTO program |
| [GM Cache-Access Policy](source-02_pypto--en--dev--language--05-cache-policy.md) | `pl.set_cache_policy` / `pl.load(cache=...)` — the declared GM cache policy, its coherency contract, and how it reaches codegen |

## See Also

- [IR Parser](source-02_pypto--en--dev--ir--07-parser.md) — how this syntax becomes IR.
- [Language Guide](../kernel-authoring-and-tutorials/source-02_pypto--en--user--language--index.md) — the same language from a user's perspective.
