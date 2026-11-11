# `pypto.language.distributed` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `AtomicType`

```text
pypto.language.distributed.AtomicType(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Combine mode for global-memory writes — pld.tensor.put (TPUT) and tile.store (TSTORE)

## `CommCtx`

```text
pypto.language.distributed.CommCtx(*, expr: pypto.pypto_core.ir.Expr | None = None) -> None
```

DSL wrapper for an ``ir.CommCtxType``-valued expression.

Construct without arguments to obtain an annotation-only placeholder
(``ctx: pld.CommCtx``). Construct with ``expr=`` to wrap an IR
``Call`` returned by :func:`pld.get_comm_ctx`.

### `CommCtx.unwrap`

```text
CommCtx.unwrap(self) -> pypto.pypto_core.ir.Expr
```

Return the wrapped :class:`ir.Expr`.

Raises:
    RuntimeError: If the instance was constructed without an
        ``expr`` (annotation-only — never returned by a wrapper).

## `DistributedTensor`

```text
pypto.language.distributed.DistributedTensor(*args: Any, **kwargs: Any) -> ~TensorT
```

Tensor backed by a per-rank slice of a HCCL window buffer carved by a CommDomainScopeStmt.

Same DSL surface as :class:`pl.Tensor` — supports memref / layout /
tensor_view in the third (and fourth) subscript slot. The IR-level type
(:class:`ir.DistributedTensorType`) is a precise-``ObjectKind`` subclass of
``TensorType``: cross-rank op verifiers dispatch on it to reject plain
``Tensor`` arguments. ``pl.load`` / ``pl.store`` etc. operate
transparently on a DistributedTensor (local rank's slice).

## `NotifyOp`

```text
pypto.language.distributed.NotifyOp(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Cross-rank notify semantics for pld.system.notify (TNOTIFY)

## `ReduceOp`

```text
pypto.language.distributed.ReduceOp(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Reduction operator for collective reductions (pld.tensor.allreduce, ...)

## `WaitCmp`

```text
pypto.language.distributed.WaitCmp(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Cross-rank wait predicate for pld.system.wait (TWAIT)

## `alloc_window_buffer`

```text
pypto.language.distributed.alloc_window_buffer(size: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], *, dtype: pypto.pypto_core.DataType | None = None, name: str = '') -> pypto.language.typing.ptr.Ptr
```

Declare a per-rank HCCL window-buffer in a comm-domain scope slot.

Two forms:

* **Canonical byte form:**
  ``alloc_window_buffer(size, *, name=...)`` — ``size`` is the per-rank
  allocation size in **bytes**.  Accepts an ``int`` literal, a DSL
  ``Scalar``, or a raw :class:`ir.Expr`.

* **Shape+dtype convenience overload:**
  ``alloc_window_buffer(shape, *, dtype=..., name=...)`` — ``shape`` is a
  list / tuple of per-rank dimensions.  The byte size is computed
  automatically as ``product(shape) x dtype.get_byte()`` and the call
  normalizes to the canonical byte form.  ``dtype`` is required when
  ``shape`` is a sequence and rejected otherwise.

Both forms return the singleton :class:`ir.PtrType` allocation-identity
token that ``pld.tensor.window`` consumes.  The two-phase
``alloc → window`` design is preserved — a single buffer can back multiple
``window()`` views across loop iterations.

Args:
    size: (Canonical form) Per-rank allocation size in **bytes**.
    shape: (Convenience form) Per-rank shape as a sequence of per-dimension
        sizes — ``int``, DSL ``Scalar``, or raw :class:`ir.Expr`.
    dtype: (Convenience form, required) Element data type. Must be
        ``None`` (default) in the canonical byte form.
    name: Unique buffer identifier. The parser injects this from the LHS
        of the surrounding assignment
        (``buf = pld.tensor.alloc_window_buffer(...)``); users **must not**
        pass it explicitly.

Returns:
    A :class:`pl.Ptr` wrapping the underlying ``ir.Call`` of result type
    :class:`ir.PtrType`. The parser unwraps it back to ``ir.Expr`` and
    binds it to the LHS as a plain :class:`ir.Var`; passing that Var
    through :func:`window` materialises a :class:`DistributedTensor`
    view.

.. note::

   This function is callable only inside HOST-level orchestration
   (``level=pl.Level.HOST, role=pl.Role.Orchestrator``). Calling it
   inside InCore (``type=pl.FunctionType.InCore``) raises a parser error.

.. seealso::

   :func:`window` for creating typed DistributedTensor views over an
   allocated buffer.

Raises:
    ValueError: If ``name`` is empty (the parser must have injected it).
    ValueError: If ``shape`` is a sequence but ``dtype`` is not provided.
    ValueError: If ``size`` is scalar but ``dtype`` is provided.

## `get_comm_ctx`

```text
pypto.language.distributed.get_comm_ctx(dist_tensor: pypto.language.distributed.typing.distributed_tensor.DistributedTensor) -> pypto.language.distributed.typing.comm_ctx.CommCtx
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

## `nranks`

```text
pypto.language.distributed.nranks(ctx: pypto.language.distributed.typing.comm_ctx.CommCtx) -> pypto.language.typing.scalar.Scalar
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
pypto.language.distributed.rank(ctx: pypto.language.distributed.typing.comm_ctx.CommCtx) -> pypto.language.typing.scalar.Scalar
```

Return the local rank as an ``INT32`` :class:`Scalar`.

Codegen lowers each call site to an ``i32`` load of the runtime
``CommContext::rankId`` field.

Args:
    ctx: A :class:`pld.CommCtx` handle from :func:`get_comm_ctx`.

Returns:
    :class:`Scalar` wrapping an :class:`ir.Expr` of type
    ``ScalarType(INT32)``.

## `remote_load`

```text
pypto.language.distributed.remote_load(target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, peer: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None) -> pypto.language.typing.tile.Tile
```

Load a region of ``peer`` rank's slice of a DistributedTensor into a local tile.

Mirrors :func:`pl.tile.load` at the user-visible surface, but the source
is a *remote* slice of a window-bound :class:`pld.DistributedTensor`.
Address translation happens at codegen via inline peer-offset arithmetic + addptr
+ make_tensor_view.

.. code-block:: python

   # Barrier example — after notify/wait, all windows are visible.
   peer_tile = pld.tile.remote_load(
       data, peer=peer, offsets=[0, 0], shape=[1, SIZE]
   )

All arguments are positional-or-keyword (mirroring :func:`pl.tile.load`),
so the printed IR — which emits them positionally — round-trips through
the parser. Callers may still pass them by keyword for readability.

Args:
    target: A window-bound :class:`pld.DistributedTensor` (any rank, any
        dtype). The C++ verifier refuses plain :class:`pl.Tensor` here
        (precise ObjectKind match on :class:`ir.DistributedTensorType`).
    peer: Peer rank index. Accepts an ``int`` literal, a DSL
        ``Scalar``, or a raw ``ir.Expr`` (e.g. ``pld.rank(ctx) + 1``).
    offsets: Offsets into the remote slice, one per ``target`` dimension.
    shape: Per-dimension shape of the tile to load. Determines the output
        :class:`pl.Tile` shape.
    valid_shape: Optional valid extent inside ``shape``. This keeps the
        physical tile fixed-size while a ragged tail reads only real data.
        Every symbolic source or requested valid extent that survives
        inference must be runtime-bound by a kernel scalar, loop variable,
        or physical tensor-shape parameter; a type-metadata-only symbol is
        rejected during PTO codegen.
Returns:
    A local :class:`pl.Tile` of the requested shape, dtype equal to
    ``target.dtype``.

## `remote_store`

```text
pypto.language.distributed.remote_store(src: pypto.language.typing.tile.Tile | pypto.language.typing.tensor.Tensor | pypto.pypto_core.ir.Expr, target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, peer: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_) -> pypto.pypto_core.ir.Call
```

Push a local value into a region of ``peer`` rank's slice of ``target``.

Dispatches on the IR level of ``src``:

* a :class:`pl.Tile` (``@pl.jit.incore`` / ``@pl.program``) routes to
  :func:`pld.tile.remote_store`;
* a tensor-level value (``@pl.jit``) routes to
  :func:`pld.tensor.remote_store`, which ``ConvertTensorToTileOps`` lowers
  1:1 to the tile form.

Either way the value reaches the peer as a single ``pto.tstore``, with no
global-memory round-trip. See the two canonical entry points for the full
argument documentation.

Args:
    src: Local 2-D :class:`pl.Tile` or :class:`pl.Tensor` value (dtype must
        match ``target.dtype``).
    target: Window-bound :class:`pld.DistributedTensor` destination (rank >= 2).
    peer: Peer rank index.
    offsets: Offsets into the remote slice, one per ``target`` dimension.
    atomic: :class:`pld.AtomicType` selecting plain-store (the default) vs
        atomic-add combine semantics on the peer's region (keyword-only).

Returns:
    A side-effect-only :class:`ir.Call` (no SSA result for downstream use).

## `window`

```text
pypto.language.distributed.window(buf: pypto.language.typing.ptr.Ptr, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], *, dtype: pypto.pypto_core.DataType) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
```

Materialise a window-buffer Ptr handle as a DistributedTensor view.

Shape and dtype enter the type system here; the result type
(:class:`ir.DistributedTensorType`) carries an optional back-reference to
the source :class:`ir.WindowBuffer` that the comm-collection pass fills
in later.

Args:
    buf: A :class:`pl.Ptr` produced by :func:`alloc_window_buffer` (or a
        raw :class:`ir.Expr` of type :class:`ir.PtrType`).
    shape: Per-rank shape (list / tuple of ints, DSL ``Scalar``s, or raw
        ``ir.Expr``s — anything :data:`IntLike` accepts).
    dtype: Element data type. Kwarg-only.

Returns:
    A :class:`DistributedTensor` view of the given shape and dtype.

.. seealso::

   :func:`alloc_window_buffer` for the two-phase ``alloc → window`` pattern.

## `world_size`

```text
pypto.language.distributed.world_size() -> pypto.language.typing.scalar.Scalar
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
