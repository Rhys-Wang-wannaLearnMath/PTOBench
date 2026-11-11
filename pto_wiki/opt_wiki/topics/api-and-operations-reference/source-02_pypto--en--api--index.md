> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# API Reference

Generated from the source docstrings, so it cannot drift from the code.

| Page | Covers |
| ---- | ------ |
| [`pl`](README.md) | Decorators, types, control flow, and the type-dispatched operator wrappers |
| [`pl.tile`](README.md) | Tile-level operators — inside an InCore function |
| [`pl.tensor`](README.md) | Tensor-level operators — orchestration, whole tensors |
| [`pl.system`](README.md) | Synchronization, cache and cross-core primitives |
| [`pl.array`](README.md) | Fixed-length arrays, mainly for `pl.TASK_ID` fan-in |
| [`pl.prefetch`](README.md) | Asynchronous GM to L2 prefetch |
| [`pl.optimizations`](README.md) | The entries that `pl.at(..., optimizations=[...])` accepts |

## How to use it

**Start from the [catalog](source-02_pypto--en--user--ops--01-catalog.md), not from here.** It groups every operator
by family with a one-line description, and each name links into the page above that carries
its signature. These pages answer "what are the arguments"; the catalog answers "which
operator do I want".

**Names are canonical, not as you spell them.** `pl.create_tensor` is an alias for
`tensor.create`, so it appears here as `create` on the `pl.tensor` page. The catalog links
resolve that for you.

**A missing symbol is a bug in the docstring, not in the page.** These pages render whatever
the source carries; if something reads thin, the fix is in the docstring.

## What the build checks

`mkdocs build --strict` fails on a docstring whose `Args:` names a parameter the signature
does not have, and on a catalog link naming a symbol that is not rendered. Both are real
defects, and both were found the first time these pages were built.

## See Also

- [Catalog](source-02_pypto--en--user--ops--01-catalog.md) — the classified index into these pages.
- [Choosing a namespace](source-02_pypto--en--user--ops--00-dispatch.md) — `pl.` vs `pl.tile.` vs `pl.tensor.`.
- [Language Guide](../kernel-authoring-and-tutorials/source-02_pypto--en--user--language--index.md) — the prose behind the types these signatures use.
