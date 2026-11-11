# `pypto.language.array` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `create`

```text
pypto.language.array.create(extent: int, dtype: pypto.pypto_core.DataType) -> pypto.language.typing.array.Array
```

Allocate an on-core array (C-stack local).

Args:
    extent: Number of elements (positive Python int — compile-time constant).
    dtype: Element data type (must be integer or BOOL).

Returns:
    ``Array`` wrapping the ``array.create`` Call expression.

## `get_element`

```text
pypto.language.array.get_element(array: pypto.language.typing.array.Array, index: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.scalar.Scalar
```

Read an element from an array at the given index.

Args:
    array: Source ``Array``.
    index: Element index (Python int, raw Expr, or Scalar DSL value).

Returns:
    ``Scalar`` wrapping the ``array.get_element`` Call expression.

## `update_element`

```text
pypto.language.array.update_element(array: pypto.language.typing.array.Array, index: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, value: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.array.Array
```

Functional update: return a new Array with the given element replaced.

Args:
    array: Source ``Array``.
    index: Element index (Python int, raw Expr, or Scalar DSL value).
    value: Replacement value (Python int, raw Expr, or Scalar DSL value).

Returns:
    ``Array`` wrapping the ``array.update_element`` Call expression.
