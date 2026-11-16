# `pypto.language.optimizations` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `CrossCoreSlot`

```text
pypto.language.optimizations.CrossCoreSlot(slot_num: 'int') -> None
```

Slot count (ring depth) for the automatic cross-core pipe.

Orthogonal to splitting — it sizes a data channel, it does not partition
work. Sets the ``slot_num`` scope attr, which ``OutlineIncoreScopes``
propagates to the outlined function and ``ExpandMixedKernel`` reads to size
**both** the reserved buffer (``slot_size * slot_num``) and the emitted
``initialize_pipe`` ``slot_num`` attribute, in whichever directions the
scope actually uses (cube→vector, vector→cube, or both).

Omitting the entry keeps the default depth of 2 per live direction — enough
to double-buffer the handoff while leaving on-chip room for the tiles. The
value is ignored when the outlined scope ends up with no cross-core ops.

Args:
    slot_num: Ring depth. Must be a positive integer.

## `Optimization`

```text
pypto.language.optimizations.Optimization()
```

Base class for ``pl.at(..., optimizations=[...])`` entries.

## `Split`

```text
pypto.language.optimizations.Split(mode: 'SplitMode', slot_num: 'int | None' = None) -> None
```

Cross-core data-transfer split hint.

Sets ``ScopeStmt::split_`` on the enclosing ``pl.at`` scope; that metadata
is consumed by the ``ExpandMixedKernel`` pass via the outlined function's
``SplitMode``. ``optimizations=[pl.split(mode)]`` lowers the scope to
``ScopeKind::InCore`` with the split metadata attached.

Args:
    mode: Split mode (``SplitMode.NONE``, ``SplitMode.UP_DOWN``, or
        ``SplitMode.LEFT_RIGHT``).
    slot_num: **Deprecated** — use ``pl.cross_core_slot(slot_num=N)``, which
        carries the same value without naming a split mode. Kept as an alias so existing kernels keep
        working; see [`CrossCoreSlot`][pypto.language.optimizations.CrossCoreSlot].

## `cross_core_slot`

```text
pypto.language.optimizations.cross_core_slot(*, slot_num: 'int') -> 'CrossCoreSlot'
```

Create a ``CrossCoreSlot`` optimization entry.

Args:
    slot_num: Cross-core pipe slot count (ring depth). Must be positive.

Returns:
    ``CrossCoreSlot`` instance for use in ``pl.at(..., optimizations=[...])``.

Raises:
    ValueError: If ``slot_num`` is not a positive integer.

Examples:
    >>> # Deepen the auto-inserted ring so the producing core can run further ahead
    >>> with pl.at(level=pl.Level.CORE_GROUP,
    ...            optimizations=[pl.cross_core_slot(slot_num=4)]):
    ...     ...

## `split`

```text
pypto.language.optimizations.split(mode: 'SplitMode', *, slot_num: 'int | None' = None) -> 'Split'
```

Create a ``Split`` optimization entry.

Args:
    mode: Split mode. May be ``SplitMode.NONE``,
        ``SplitMode.UP_DOWN``, or ``SplitMode.LEFT_RIGHT``.
    slot_num: **Deprecated** — use ``pl.cross_core_slot(slot_num=N)``.
        Must be positive when set. Emits a ``DeprecationWarning``.

Returns:
    ``Split`` instance for use in ``pl.at(..., optimizations=[...])``.

Raises:
    ValueError: If ``slot_num`` is set but not positive.
