# `pypto.language.distributed.tensor` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `all_to_all`

```text
pypto.language.distributed.tensor.all_to_all(input: pypto.language.typing.tensor.Tensor | pypto.language.distributed.typing.distributed_tensor.DistributedTensor, target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
```

All-to-all: symmetric personalized exchange (push-based).

3-arg form: ``pld.tensor.all_to_all(input, target, signal)``.
Every rank pushes its per-destination chunks directly to every peer's
``target`` window via TPUT, then synchronises with a notify/wait
barrier.  Returns ``target`` in-place (window-as-result — same idiom as
``reduce_scatter`` / ``broadcast``).

``input`` must be a DIFFERENT buffer from ``target`` — never pass the same
window for both.  For the HOST-level builtin dispatch, ``input`` is
typically a second window (e.g. a :class:`pld.DistributedTensor` staged
by an earlier InCore step) rather than the InCore composite's plain
:class:`pl.Tensor` — both are accepted.

**Signal shape:** host builtins accept rank-1 ``[world_size]`` or rank-2
``[world_size, 1]``. InCore composites take rank-2 ``[nranks, 1]`` -- the
rank count may be dynamic.

Args:
    input: [NR, SIZE] Tensor or DistributedTensor with per-destination
        chunks, distinct from ``target``.  ``input[dest, :]`` is the
        chunk destined for rank ``dest``.
    target: :class:`pld.DistributedTensor` [NR, SIZE] window that receives
        the result in-place.  After the call,
        ``target[src, :]`` holds the chunk received from rank ``src``.
    signal: :class:`pld.DistributedTensor` [NR, 1] INT32 barrier —
        rank-1 ``[world_size]`` or rank-2 ``[world_size, 1]``.  Reusable
        across calls — see :func:`allreduce` for the shared barrier
        protocol.

Returns:
    The ``target`` :class:`pld.DistributedTensor` (window-as-result).

## `all_to_all_v`

```text
pypto.language.distributed.tensor.all_to_all_v(input: pypto.language.typing.tensor.Tensor | pypto.language.distributed.typing.distributed_tensor.DistributedTensor, target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, send_counts: pypto.language.typing.tensor.Tensor | pypto.language.distributed.typing.distributed_tensor.DistributedTensor, recv_counts: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, *, core_num: int = 1) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
```

All-to-all: variable-size personalized exchange (push-based, window-as-result).

5-arg form: ``pld.tensor.all_to_all_v(input, target, signal, send_counts,
recv_counts)``.

Each rank pushes ``clamp(send_counts[dest], 0, MAX_RECV)`` rows to peer
``dest`` — the counts are read at runtime, so they may be data-dependent
(e.g. MoE tokens per expert), and they **do** set the transfer size, so
padding up to the capacity never crosses the wire.  Mirrors the symmetric
``pld.tensor.all_to_all`` otherwise: rows are pushed into a flat 2D staging
window via ``pld.tile.put``, and the window is returned so the caller can
read back via ``pl.load``.  There is no built-in read-back phase — the user
writes the read-back loop in the InCore function.

``input`` is a flat 2D [NR*MAX_RECV, SIZE] Tensor or DistributedTensor whose
rows ``dest*MAX_RECV .. dest*MAX_RECV + send_counts[dest] - 1`` hold the
chunk for peer ``dest``.  ``target`` is a flat 2D DistributedTensor
[NR*MAX_RECV, SIZE] — the staging window that doubles as the result;
rank ``src``'s rows land at ``src*MAX_RECV ...``.

``MAX_RECV = target.shape[0] // NR`` is the compile-time per-peer
*capacity*: it fixes the row-index arithmetic so a receiver can locate each
sender's block without knowing that sender's count.  It is **not** the
transfer size — the transfer extent is the runtime ``[rows, SIZE]``, where
``rows = clamp(send_counts[dest], 0, MAX_RECV)``.

The clamp is two-sided.  A count above ``MAX_RECV`` is capped so it cannot
push into the next destination's slice; a **negative** count is floored at
``0``, and since the published value is that same clamped count, a negative
``send_counts[dest]`` publishes ``recv_counts = 0`` rather than the negative
number.

Rows beyond a sender's count are **not written at all** — they no longer
cross the wire.  As with ``MPI_Alltoallv``, the receiver must not read past
its published ``recv_counts`` entry.

.. warning::

   Trim to ``recv_counts`` **before** doing arithmetic over the capacity
   block.  The untouched tail is *uninitialised* and may decode as **NaN or
   Inf** — not merely as a wrong-but-finite value.  Code that reduces or
   otherwise computes across the dense ``[NR*MAX_RECV, SIZE]`` block and
   masks by ``recv_counts`` afterwards will propagate NaN into
   otherwise-valid rows.  Mask first, then compute.

During the same push, each rank also publishes
``clamp(send_counts[dest], 0, MAX_RECV)`` into peer ``dest``'s
``recv_counts[my_rank, 0]`` via ``pld.system.notify`` (Set). After the
barrier, ``recv_counts[src, 0]`` tells this rank how many rows ``src``
sent — which is now also exactly how many were transferred — so use that
count to know where to stop reading. This is the MPI_Alltoallv recvcounts
side.

The barrier ``signal`` is self-clearing (restored to zero after each call)
and safe to reuse inside a ``for``/``while`` loop.

Args:
    input: Flat 2D Tensor or DistributedTensor [NR*MAX_RECV, SIZE] with
        per-destination chunks (e.g. a window published by a preceding
        exchange).
    target: :class:`pld.DistributedTensor` [NR*MAX_RECV, SIZE] —
        flat 2D staging window (InOut); returned as the result.
    signal: :class:`pld.DistributedTensor` [NR, 1] INT32 barrier (InOut).
    send_counts: INT32 [NR] or [NR, 1] rows-per-destination counts (Input).
        A plain :class:`pl.Tensor` or a window-bound
        :class:`pld.DistributedTensor` (e.g. counts published by a
        preceding exchange).
    recv_counts: :class:`pld.DistributedTensor` INT32 [NR, 1] — after the
        call, ``recv_counts[src, 0]`` holds how many rows ``src`` actually
        sent here, and how many were transferred — the count is clamped
        to ``[0, MAX_RECV]``, so a negative input publishes ``0`` (InOut).
    core_num: Requested AIV block limit for the managed CHIP/L2 rail.
        Called from a CHIP :class:`pl.FunctionType.Orchestration` function,
        the call lowers to one local builtin AIV task
        (``LowerL2TensorCollectives``); the first version accepts only
        ``core_num=1``.  Called from an InCore function, the composite rail
        expands the exchange into point-to-point primitives and requires
        ``core_num=1``.

Returns:
    The ``target`` :class:`pld.DistributedTensor` with received chunks.

## `alloc_window_buffer`

```text
pypto.language.distributed.tensor.alloc_window_buffer(size: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], *, dtype: pypto.pypto_core.DataType | None = None, name: str = '') -> pypto.language.typing.ptr.Ptr
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

## `allgather`

```text
pypto.language.distributed.tensor.allgather(local_data: pypto.language.typing.tensor.Tensor | pypto.language.distributed.typing.distributed_tensor.DistributedTensor, target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
```

All-gather: gather data from all ranks (push-based).

Unified 3-arg form: ``pld.tensor.allgather(local_data, target, signal)``.
``local_data`` is this rank's single ``[1, SIZE]`` chunk; every rank pushes it
into every peer's ``target`` row ``my_rank`` via TPUT, then synchronises
with a notify/wait barrier.  Returns ``target`` in-place (window-as-result
— same idiom as ``all_to_all`` / ``reduce_scatter`` / ``broadcast``).

``local_data`` must be a DIFFERENT buffer from ``target`` — never pass the same
window for both.  On the HOST path ``local_data`` is a ``[1, SIZE]`` staging
window populated by an earlier publish step; on the InCore path it is a
plain :class:`pl.Tensor` ``[1, SIZE]`` — both are accepted.  HOST vs InCore
is a function-context property resolved by the lowering passes.

**Signal shape:** host builtins accept rank-1 ``[world_size]`` or rank-2
``[world_size, 1]``. InCore composites take rank-2 ``[nranks, 1]`` -- the
rank count may be dynamic.

Args:
    local_data: This rank's single chunk — ``[1, SIZE]`` :class:`pl.Tensor`
        (InCore) or ``[1, SIZE]`` :class:`pld.DistributedTensor` staging
        window (HOST).  Must differ from ``target``.
    target: :class:`pld.DistributedTensor` ``[NR, SIZE]`` result window.
        After the call, ``target[src, :]`` holds the chunk from rank
        ``src``.
    signal: Window-bound INT32 :class:`pld.DistributedTensor` barrier
        tensor — rank-1 ``[world_size]`` or rank-2 ``[world_size, 1]``.
        Reusable across calls — see :func:`allreduce` for the
        shared barrier protocol.

Returns:
    The ``target`` :class:`pld.DistributedTensor` (window-as-result).

## `allreduce`

```text
pypto.language.distributed.tensor.allreduce(target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor | object = <object object at 0xfffdd18cb460>, *, op: pypto.pypto_core.ir.ReduceOp = ReduceOp.Sum, mode: str = 'mesh', core_num: int = 1) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
```

In-place cross-rank allreduce of a window-bound DistributedTensor.

After this call returns, every rank's slice of ``target`` holds the
reduced value. Mirrors :func:`pl.store`'s rebind idiom — users assign the
result back to the same name:

.. code-block:: python

    pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum)
    pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum, mode="ring")

    # Mesh mode on the host orchestrator can omit the signal argument:
    pub = pld.tensor.allreduce(pub, op=pld.ReduceOp.Sum)

**Signal shape:** host builtins accept rank-1 ``[world_size]`` or rank-2
``[world_size, 1]`` (the compiler-synthesized signal is rank-2). InCore
composites take rank-2 ``[nranks, 1]`` for mesh -- the rank count may be
dynamic -- and ``[2*(NR-1), NR]`` for ring, where ``NR`` must be a
compile-time constant.

**Signal reuse:** InCore composites use the self-clearing credit-barrier
protocol (see below), so their signal is reusable across back-to-back
calls — including inside ``for`` / ``while`` / ``if``. The HOST builtin
allreduce is not yet self-clearing, so a HOST allreduce (explicit or
synthesized signal) is still rejected inside ``for`` / ``while`` loops.

.. seealso::

    :func:`alloc_window_buffer` and :func:`window` for buffer allocation
    and view creation.

.. rubric:: Implementation Notes

LowerCompositeOps expands the explicit-signal InCore form into either:
(a) a notify/wait ready barrier followed by UB-sized
remote_load+accumulate/store chunks for ``mode="mesh"`` (default); or
(b) the NCCL-style 2(P-1)-step
chunked reduce-scatter + allgather ring schedule for ``mode="ring"``.
In both modes the kernel sees only the lowered primitives.
Host-orchestrator code can omit ``signal`` outside ``for`` / ``while``
loops; the compiler synthesizes a private INT32 signal window of shape
``[pld.world_size(), 1]`` for that call (mesh mode only — ring mode on the
HOST rail is delivered by a subsequent host builtin).

Mesh signal shape is ``[NR, 1]``. The InCore ring schedule uses
``[2 * (NR − 1), NR]`` (one row per ring round). The host builtin ring
schedule uses ``[2 * (NR − 1) + 1, NR]`` (one extra row for the return
barrier). The mesh and ring shapes address cells differently, so one
buffer must not be shared between the two conventions.

.. note::

    The host-builtin ring schedule currently supports only
    ``ReduceOp.Sum`` with ``dtype=FP32``.  ``ReduceOp.Max``,
    ``ReduceOp.Min``, ``ReduceOp.Prod``, and ``FP16`` are not yet
    available with ``mode="ring"``.

Fully-valid packed mesh targets are viewed as one logical 1D stream and
processed in chunks of at most 16 KiB. For statically known smaller targets,
the physical chunk width shrinks to the smallest 32-byte-aligned width that
covers the target. The final chunk uses ``valid_shape`` so arbitrary element
counts do not read or store past the end.

See ``docs/en/dev/distributed_ops.md`` and
``docs/en/dev/passes/13-lower_composite_ops.md`` for the mesh partial-valid /
symbolic-extent target constraints (unchanged by this PR).

**Barrier protocol (self-clearing credit barrier):** every call's
barriers count a call-local generation ``g`` starting at 1 —
``AtomicAdd(1) → WaitGe(g)`` — and a trailing epilogue subtracts the
call's total credit ``N`` back out of every non-self cell with a single
``AtomicAdd(-N)``. Adds and subtracts commute, so the signal is provably
all-zero again once every rank finishes its epilogue: **the next call on
the same signal also starts at generation 1**, with no cross-call state
to go stale. ``N`` may be a runtime scalar, so a mesh allreduce's
per-chunk credit count does not need to be a compile-time constant.

In mesh mode, the ready barrier is generation 1; every reduced chunk then
barriers on the next generation before storing that chunk (``1 +
chunk_count`` credits total for a fully-valid call; the partial-valid
rectangle path issues exactly 2). In ring mode, every subchunk of every
round barriers on its own call-local ready + read-complete generation
pair; the epilogue subtracts ``2 * chunk_count`` (uniform across rounds)
from every row of the ``[2*(NR-1), NR]`` signal.

Ring mode first views a packed ND target as one linear stream, then
traverses each segment in physical subchunks of at most 16 KiB. FP32 uses
balanced logical segment boundaries. FP16 aligns every non-empty segment
start and remote tail span to 32 bytes while restoring the logical
``valid_shape`` before reduction and store. This also covers ``SIZE < NR``
without changing the packed public layout or dropping elements.

For InCore composites, a signal buffer is safely reusable across
back-to-back allreduce calls — including inside ``for`` / ``while`` /
``if`` — since every call is a stateless cycle starting from all-zero.
A call aborted mid-flight (error or timeout) leaves credits on the signal;
recover with a host-side reset (``reset_persistent_windows``) before the
next dispatch.

Args:
    target: Window-bound :class:`pld.DistributedTensor` holding per-rank
        FP16 or FP32 data. The C++ verifier refuses a plain
        :class:`pl.Tensor` and unsupported dtypes.
    signal: Optional window-bound INT32 :class:`pld.DistributedTensor`.
        In InCore code this remains required. In host-orchestrator code
        outside ``for`` / ``while`` loops, omitting it lets the compiler
        synthesize a private signal of shape
        ``[pld.world_size(), core_num]``.
    op: :class:`pld.ReduceOp` selecting element-wise ``Sum``, ``Max``,
        ``Min``, or ``Prod`` (keyword-only). Defaults to
        :attr:`pld.ReduceOp.Sum`.
    mode: Algorithm selector (keyword-only). ``"mesh"`` (default) for
        direct all-to-all exchange; ``"ring"`` for the NCCL-style
        chunked reduce-scatter + allgather ring schedule. ``"ring"``
        requires an explicit ``signal`` — host signal synthesis is
        mesh-only, so omitting the signal with ``mode="ring"`` is
        rejected.
    core_num: Number of AIV blocks used by a HOST AllReduce builtin
        (keyword-only). Must be a positive compile-time Python integer and
        may not exceed the configured backend's AIV core count. Defaults to
        1. Multicore is ``mode="mesh"`` only — ``mode="ring"`` requires
        ``core_num=1``. InCore calls must keep this value at 1 and use an
        enclosing :func:`pl.spmd` for multi-core execution.

Returns:
    The rebound :class:`pld.DistributedTensor` view of ``target`` —
    identical shape / dtype / window-buffer binding, post-reduce content.

## `barrier`

```text
pypto.language.distributed.tensor.barrier(signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
```

Cross-rank barrier synchronisation.

Blocks until all ranks in the comm group have reached the barrier.
Uses a window-bound INT32 ``signal`` tensor for cross-rank
synchronisation.  LowerCompositeOps expands this
into a notify-all / wait-all sequence.

.. code-block:: python

    sig = pld.tensor.barrier(sig)

**Reusable across calls** — see :func:`allreduce` for the shared
self-clearing credit-barrier protocol. Each call is a stateless cycle
that restarts at generation 1, so ``sig`` may be reused for back-to-back
barriers, including inside ``for`` / ``while`` / ``if``.

Args:
    signal: Window-bound INT32 :class:`pld.DistributedTensor` whose
        shape provides one cell per rank — rank-1 ``[world_size]`` or
        rank-2 ``[world_size, 1]``.

Returns:
    The rebound :class:`pld.DistributedTensor` view of ``signal``.

## `broadcast`

```text
pypto.language.distributed.tensor.broadcast(target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, *, root: int) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
```

Broadcast root rank's data to all ranks.

After this call returns, every rank's slice of ``target`` holds
root's data.  Uses a window-bound INT32 ``signal`` tensor for the
cross-rank barrier.

**Signal shape:** host builtins require rank-1 ``[world_size]``. InCore
composites take rank-2 ``[nranks, 1]`` -- the rank count may be dynamic.

.. code-block:: python

    # Root stages data; non-root skip.
    if my_rank == ROOT_RANK:
        data = pl.store(local, [0, 0], data)
    data = pld.tensor.broadcast(data, sig, root=ROOT_RANK)
    # Every rank now has root's data in data[0, 0:SIZE].

Args:
    target: Window-bound :class:`pld.DistributedTensor` holding per-rank
        data.  Root must stage its data before the call; non-root slots
        are ignored on input.
    signal: Window-bound INT32 :class:`pld.DistributedTensor` for the
        cross-rank barrier — rank-1 ``[world_size]`` or rank-2
        ``[world_size, 1]``.  Reusable across calls — see
        :func:`allreduce` for the shared barrier protocol.
    root: Root rank index (int, keyword-only).  Must be non-negative.
        On the HOST path with an explicit static device subset, ``root``
        must also be a valid rank of that subset (``root <``
        participating device count) — checked once the subset size is
        known during ``LowerHostTensorCollectives``.  Not checked at
        compile time for the fully-dynamic "all device" domain, since no
        device count is known there.

Returns:
    The rebound :class:`pld.DistributedTensor` view of ``target``.

## `get`

```text
pypto.language.distributed.tensor.get(dst: pypto.language.distributed.typing.distributed_tensor.DistributedTensor | pypto.language.typing.tensor.Tensor, peer: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, src: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, dst_offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, src_offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, *, chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> pypto.pypto_core.ir.Call
```

Cross-rank get: read the peer rank's slice of ``src`` into local ``dst``.

Side-effect-only (the returned Call carries ``UnknownType``). Semantically
equivalent to ``remote_load + store`` but represented as one tensor-level
bulk communication op. Lowers to inline peer-offset arithmetic + ``addptr`` +
``make_tensor_view`` + ``partition_view`` + a synthesised VEC staging tile +
``TGET`` at codegen.

With no offsets/shape this reads the full peer ``src`` slice into the full
local ``dst`` slice. Supplying ``dst_offsets``, ``src_offsets``, and
``shape`` narrows the transfer to matching subregions; all three must be
provided together.

Args:
    dst: Local destination — either a window-bound
        :class:`pld.DistributedTensor` or a plain :class:`pl.Tensor`.
        TGET only needs a writable local GM region to receive into;
        window membership is not required on the destination side.
    peer: Peer rank index.
    src: Peer rank's window-bound :class:`pld.DistributedTensor` source.
    dst_offsets: Optional offsets into the local ``dst`` slice.
    src_offsets: Optional offsets into the peer ``src`` slice.
    shape: Optional static transfer shape. Required when either offset
        argument is provided.
    chunk_rows: Optional VEC staging-tile row extent (keyword-only,
        ``0`` = full) sizing the staging tile to a sub-tile of the flattened
        transfer so pto-isa TGET auto-chunks the full transfer through it.
        Oversized values are clamped to the transfer extent.
    chunk_cols: Optional VEC staging-tile column extent (keyword-only,
        ``0`` = full innermost dim). Pairs with ``chunk_rows``.
    pipeline: Enable ping-pong double-buffering (keyword-only). When True,
        ``ConvertTensorToTileOps`` allocates two staging tiles and pto-isa
        TGET overlaps TLOAD/TSTORE across chunks through them. Requires both
        ``chunk_rows`` and ``chunk_cols`` to be set (> 0).

Returns:
    The underlying IR Call.

## `put`

```text
pypto.language.distributed.tensor.put(dst: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, peer: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, src: pypto.language.distributed.typing.distributed_tensor.DistributedTensor | pypto.language.typing.tensor.Tensor, dst_offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, src_offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_, chunk_rows: int = 0, chunk_cols: int = 0, pipeline: bool = False) -> pypto.pypto_core.ir.Call
```

Cross-rank put: write the local slice ``src`` into the peer rank's slice of ``dst``.

Side-effect-only (the returned Call carries ``UnknownType``). Rewritten by
``ConvertTensorToTileOps`` to a ``tile.create``-allocated VEC staging tile plus
a ``pld.tile.put`` call so the staging tile flows through PyPTO's memory
allocator (required at ``--pto-level=level3``); backend codegen then emits
inline peer-offset arithmetic + ``addptr`` + ``make_tensor_view`` +
``partition_view`` + ``TPUT`` against that pre-allocated tile. Both operands are GM/tensor-level
window views (the staging tile is internal), so this is a ``pld.tensor`` op,
paired with the GM-to-GM TGET rather than the tile-producing
``pld.tile.remote_load``.

``dst`` / ``peer`` / ``src`` are positional-or-keyword so the printed IR
(which emits them positionally) round-trips through the parser; ``atomic``
stays keyword-only because it lowers to an IR attr (printed as
``atomic=<int>``), mirroring ``pld.system.notify``'s ``op``.

With no offsets/shape this writes the full local ``src`` slice to the full
peer ``dst`` slice. Supplying ``dst_offsets``, ``src_offsets``, and
``shape`` narrows the transfer to matching subregions; all three must be
provided together.

Args:
    dst: Window-bound :class:`pld.DistributedTensor` destination (the peer
        rank's slice). The C++ verifier refuses a plain :class:`pl.Tensor`.
    peer: Peer rank index.
    src: Local source — either a :class:`pld.DistributedTensor` (window-
        bound) or a plain :class:`pl.Tensor`. Must share element type with
        ``dst``. Window membership is not required on the source side;
        TPUT only needs a readable local GM region.
    dst_offsets: Optional offsets into the peer ``dst`` slice.
    src_offsets: Optional offsets into the local ``src`` slice.
    shape: Optional static transfer shape. Required when either offset
        argument is provided.
    atomic: :class:`pld.AtomicType` selecting plain-store
        (``AtomicType.None_``, the default) vs atomic-add
        (``AtomicType.Add``) combine semantics (keyword-only).
        ``Add`` requires an fp32/bf16/fp16/int32/int16/int8 destination —
        TPUT lands its chunks through the same store pipe as ``pl.store``,
        so the hardware atomic-add dtypes apply. A bf16 destination is
        further Ascend910B (A2/A3) only; on Ascend950 use an fp32 window
        and cast after the reduction.
    chunk_rows: Optional VEC staging-tile row extent (keyword-only,
        ``0`` = full). Sizes the staging tile to a sub-tile of the flattened
        transfer (``rows`` = product of leading dims), so pto-isa TPUT
        auto-chunks the full transfer through it — transfers larger than UB
        no longer need to fit in one staging tile. Oversized values are
        clamped to the transfer extent.
    chunk_cols: Optional VEC staging-tile column extent (keyword-only,
        ``0`` = full innermost dim). Pairs with ``chunk_rows``.
    pipeline: Enable ping-pong double-buffering (keyword-only). When True,
        ``ConvertTensorToTileOps`` allocates two staging tiles and pto-isa
        TPUT overlaps TLOAD/TSTORE across chunks through them. Requires both
        ``chunk_rows`` and ``chunk_cols`` to be set (> 0).

## `remote_store`

```text
pypto.language.distributed.tensor.remote_store(src: pypto.language.typing.tensor.Tensor | pypto.pypto_core.ir.Expr, target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, peer: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_) -> pypto.pypto_core.ir.Call
```

Push a tensor-level value into a region of ``peer`` rank's slice of ``target``.

Tensor-level twin of :func:`pld.tile.remote_store` — same four arguments,
same single ``pto.tstore``, one IR level up. This is the entry point a
tensor-level ``@pl.jit`` kernel uses to push a **computed** value cross-rank:
``ConvertTensorToTileOps`` lowers it 1:1 to ``pld.tile.remote_store``, so the
value goes straight from on-core memory to the peer's window with no global-
memory round-trip in between.

.. code-block:: python

   @pl.jit
   def push(x: pl.Tensor[[16, 256], pl.FP32], win: pld.DistributedTensor[[16, 256], pl.FP32], peer):
       with pl.at(level=pl.Level.CORE_GROUP):
           scaled = pl.mul(x[0:16, 0:256], 2.0)
           pld.tensor.remote_store(scaled, win, peer, [0, 0])
           # ...then pld.system.notify() to release it to the peer.

A ``src`` that is still resident in global memory at lowering time (e.g. a
kernel parameter pushed unchanged) is auto-bridged with a ``tile.load``, so
the op is total on its argument surface. Prefer :func:`pld.tensor.put` for a
**bulk** global-memory transfer: TPUT streams through a staging tile and owns
the ``chunk_rows`` / ``chunk_cols`` / ``pipeline`` knobs, so it is not bounded
by what fits on-core.

Args:
    src: Local 2-D :class:`pl.Tensor` value (dtype must match
        ``target.dtype``). A :class:`pld.DistributedTensor` is refused — a
        window-to-window transfer is :func:`put`'s job.
    target: Window-bound :class:`pld.DistributedTensor` destination
        (rank >= 2). The C++ verifier refuses a plain :class:`pl.Tensor`.
    peer: Peer rank index.
    offsets: Offsets into the remote slice, one per ``target`` dimension.
        The pushed region must fit inside ``target`` at these offsets.
    atomic: :class:`pld.AtomicType` selecting plain-store
        (``AtomicType.None_``, the default) vs atomic-add combine semantics
        on the peer's region (keyword-only).

Returns:
    A side-effect-only :class:`ir.Call` (no SSA result for downstream use).

## `reduce_scatter`

```text
pypto.language.distributed.tensor.reduce_scatter(target: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, signal: pypto.language.distributed.typing.distributed_tensor.DistributedTensor, *, op: pypto.pypto_core.ir.ReduceOp = ReduceOp.Sum) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
```

Reduce-scatter: reduce chunks across ranks, one reduced chunk per rank.

``target`` has shape [NR, SIZE] — one row per chunk.  Each rank must
stage all NR chunks before calling::

    for j in range(nranks):
        data = pl.store(chunk_j, [j, 0], data)
    data = pld.tensor.reduce_scatter(data, sig, op=pld.ReduceOp.Sum)
    # data[my_rank, 0:SIZE] now holds this rank's reduced chunk.

**Signal shape:** host builtins require rank-1 ``[world_size]``. InCore
composites take rank-2 ``[nranks, 1]`` -- the rank count may be dynamic.

Args:
    target: Window-bound :class:`pld.DistributedTensor` of shape
        [NR, SIZE].  Each rank stages all NR chunks, one per row.
    signal: Window-bound INT32 :class:`pld.DistributedTensor` for
        the cross-rank barrier — rank-1 ``[world_size]`` or rank-2
        ``[world_size, 1]``.  Reusable across calls (2 credits per
        call — ready + post-reduce) — see :func:`allreduce` for the
        shared barrier protocol.
    op: :class:`pld.ReduceOp` (keyword-only).  ``Sum``, ``Max``, ``Min``,
        or ``Prod`` on the InCore rail; the HOST builtin rail lowers
        ``Sum`` only.

Returns:
    The rebound :class:`pld.DistributedTensor` — rank r's row
    [r, 0:SIZE] holds the reduced chunk r.

## `window`

```text
pypto.language.distributed.tensor.window(buf: pypto.language.typing.ptr.Ptr, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], *, dtype: pypto.pypto_core.DataType) -> pypto.language.distributed.typing.distributed_tensor.DistributedTensor
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
