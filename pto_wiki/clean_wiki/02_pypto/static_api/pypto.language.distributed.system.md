# `pypto.language.distributed.system` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `defer_wait`

```text
pypto.language.distributed.system.defer_wait(signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], expected: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, *, cmp: pypto.pypto_core.ir.WaitCmp) -> pypto.pypto_core.ir.Call
```

Register a deferred completion condition on a local signal slot.

Unlike :func:`wait`, this operation does not block the device core and does
not resume the kernel when the condition becomes true. The enclosing task
may finish executing, while its TaskId remains incomplete until
``signal[offsets] >= expected``. Continuation work must therefore be placed
in a dependent task.

Args:
    signal: Window-bound INT32 :class:`pld.DistributedTensor` signal matrix.
    offsets: Offsets into the local slice, one per ``signal`` dimension.
    expected: Integer scalar threshold value to compare against.
    cmp: Must be :class:`pld.WaitCmp.Ge` (keyword-only).

## `get_comm_ctx`

```text
pypto.language.distributed.system.get_comm_ctx(dist_tensor: pypto.language.distributed.typing.distributed_tensor.DistributedTensor) -> pypto.language.distributed.typing.comm_ctx.CommCtx
```

Return the :class:`pld.CommCtx` handle of a window-bound DistributedTensor.

The op verifier (C++) enforces that ``dist_tensor`` carries
:class:`ir.DistributedTensorType` — passing a plain :class:`pl.Tensor`
is rejected by precise-ObjectKind match.

Args:
    dist_tensor: A :class:`pld.DistributedTensor` (annotation form or
        value wrapper). Raw :class:`ir.Expr` is also accepted for
        parser-side invocations that have already unwrapped.

Returns:
    A :class:`pld.CommCtx` wrapping an :class:`ir.Call` of type
    :class:`ir.CommCtxType`. Pass to :func:`rank` / :func:`nranks` to
    read scalar fields.

## `notify`

```text
pypto.language.distributed.system.notify(target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, peer: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], value: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, *, op: pypto.pypto_core.ir.NotifyOp) -> pypto.pypto_core.ir.Call
```

Cross-rank notify: deposit ``value`` at the peer rank's slot of ``target``.

Side-effect-only (the returned Call carries ``UnknownType``). Lowers to
inline peer-offset arithmetic + ``addptr`` + ``make_tensor_view`` + ``TNOTIFY`` at
codegen.

``target`` / ``peer`` / ``offsets`` / ``value`` are positional-or-keyword
so the printed IR (which emits them positionally) round-trips through the
parser; ``op`` stays keyword-only because it lowers to an IR attr (printed
as ``op=<int>``), mirroring ``pld.tensor.window``'s ``dtype``.

.. note::

   ``notify`` names this operand ``target``; the companion ``wait`` names
   the same logical operand ``signal``. Both refer to the same
   window-bound signal tensor.

Args:
    target: Window-bound :class:`pld.DistributedTensor` signal matrix.
    peer: Peer rank index.
    offsets: Offsets into the remote slice, one per ``target`` dimension.
    value: Scalar payload to deposit at the peer slot.
    op: :class:`pld.NotifyOp` selecting atomic-add vs set semantics
        (keyword-only).

## `nranks`

```text
pypto.language.distributed.system.nranks(ctx: pypto.language.distributed.typing.comm_ctx.CommCtx) -> pypto.language.typing.scalar.Scalar
```

Return the rank count of the comm group as an ``INT32`` :class:`Scalar`.

Codegen lowers each call site to an ``i32`` load of the runtime
``CommContext::rankNum`` field.

Args:
    ctx: A :class:`pld.CommCtx` handle from :func:`get_comm_ctx`.

Returns:
    :class:`Scalar` wrapping an :class:`ir.Expr` of type
    ``ScalarType(INT32)``.

## `rank`

```text
pypto.language.distributed.system.rank(ctx: pypto.language.distributed.typing.comm_ctx.CommCtx) -> pypto.language.typing.scalar.Scalar
```

Return the local rank as an ``INT32`` :class:`Scalar`.

Codegen lowers each call site to an ``i32`` load of the runtime
``CommContext::rankId`` field.

Args:
    ctx: A :class:`pld.CommCtx` handle from :func:`get_comm_ctx`.

Returns:
    :class:`Scalar` wrapping an :class:`ir.Expr` of type
    ``ScalarType(INT32)``.

## `wait`

```text
pypto.language.distributed.system.wait(signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], expected: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, *, cmp: pypto.pypto_core.ir.WaitCmp) -> pypto.pypto_core.ir.Call
```

Cross-rank wait: block until the local slot of ``signal`` matches ``cmp(expected)``.

Side-effect-only (the returned Call carries ``UnknownType``). Lowers to
TWAIT at codegen.

``signal`` / ``offsets`` / ``expected`` are positional-or-keyword so the
printed IR round-trips through the parser; ``cmp`` stays keyword-only
because it lowers to an IR attr (printed as ``cmp=<int>``).

Args:
    signal: Window-bound :class:`pld.DistributedTensor` signal matrix.
    offsets: Offsets into the local slice, one per ``signal`` dimension.
    expected: Scalar threshold value to compare against.
    cmp: :class:`pld.WaitCmp` selecting equality vs greater-or-equal
        (keyword-only).

## `world_size`

```text
pypto.language.distributed.system.world_size() -> pypto.language.typing.scalar.Scalar
```

Return the distributed world size as an ``INT64`` :class:`Scalar`.

Parser context (host-only, not inside a nested device-side scope) is
validated by the parser before this wrapper is invoked. The DSL-side
return wrapping lets call sites compose naturally with Python operators
(``pld.world_size() * 4``, ``pl.range(pld.world_size())``), which the
parser's ``invoke_dsl`` unwraps back to the underlying Call.

.. warning::

   This function is callable **only** inside HOST-level orchestration
   (``level=pl.Level.HOST, role=pl.Role.Orchestrator``). Calling it
   inside InCore (``type=pl.FunctionType.InCore``) raises a parser error.

.. seealso::

   :func:`rank` and :func:`nranks` — the per-rank equivalents for InCore
   kernels, called on a :class:`CommCtx` obtained via
   :func:`get_comm_ctx`.
