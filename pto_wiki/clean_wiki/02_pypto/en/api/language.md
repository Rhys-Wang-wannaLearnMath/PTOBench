> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../../../BASELINE.md).

# `pl`

The names you reach directly on `pl` — decorators, types, control-flow constructs, and the
type-dispatched operator wrappers. An operator here dispatches on what you hand it: `pl.add`
of two tiles is `pl.tile.add`, of two tensors `pl.tensor.add`. See
[Choosing a namespace](../user/ops/00-dispatch.md).

See the complete installed [signatures and docstrings](../../static_api/pypto.language.md).
