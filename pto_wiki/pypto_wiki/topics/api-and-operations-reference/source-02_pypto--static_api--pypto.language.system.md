# `pypto.language.system` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `AUTO`

```text
pypto.language.system.AUTO
```

## `KernelType`

```text
pypto.language.system.KernelType(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Which generated kernel an op belongs to.

A mixed InCore function is expanded into an AIC kernel and an AIV kernel.
This says which of the two a cross-core sync op lands in; ``MIX`` means both
take part, which only a barrier can ask for.

Two neighbouring enums mean different things:

- ``FunctionType.AIC`` / ``.AIV`` classify *a function*, and this classifies
  *an op inside one*. A function already declared ``FunctionType.AIV`` needs
  no ``KernelType`` on its ops -- there is only one kernel to land in.
- ``ir.CoreType`` labels one physical core in the SoC inventory, which is
  what ``Backend::GetCoreCount`` counts. That is hardware; this is not, and
  ``MIX`` has no ``CoreType`` counterpart at all.

Members carry no wire value: each op spells the same kernel differently in
its IR attr (``system.syncall`` writes ``"aic_only"`` where
``system.sync_set`` writes ``"aic"``), so the lowering tables are explicit.

## `SyncAllMode`

```text
pypto.language.system.SyncAllMode(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Barrier implementation selected by ``system.syncall``.

- ``HARD``: FFTS barrier with no operands; requires full-core occupancy.
- ``SOFT``: GM-polling barrier; works at partial occupancy.

## `sync_src`

```text
pypto.language.system.sync_src(*, set_pipe: pypto.pypto_core.ir.PipeType, wait_pipe: pypto.pypto_core.ir.PipeType, event_id: int, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Send a synchronization signal (Set Flag).

Args:
    set_pipe: Pipe that sets the flag
    wait_pipe: Pipe that will wait on the flag
    event_id: Event identifier
    span: Optional source span for debugging (auto-captured if not provided)

Returns:
    Call expression for system.sync_src

## `sync_dst`

```text
pypto.language.system.sync_dst(*, set_pipe: pypto.pypto_core.ir.PipeType, wait_pipe: pypto.pypto_core.ir.PipeType, event_id: int, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Wait for a synchronization signal (Wait Flag).

Args:
    set_pipe: Pipe that sets the flag
    wait_pipe: Pipe that waits on the flag
    event_id: Event identifier
    span: Optional source span for debugging (auto-captured if not provided)

Returns:
    Call expression for system.sync_dst

## `sync_set`

```text
pypto.language.system.sync_set(event_id: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, *, pipe: pypto.pypto_core.ir.PipeType, ffts_mode: int | None = None, core_type: pypto.ir.op.system_ops.KernelType | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Set a Cube/Vector cross-core event using a static or dynamic event id.

Args:
    event_id: Event to signal. An int in the user-available range 0-13, or a
        dynamic ``pl.Scalar[pl.INDEX]``. IDs 14 and 15 are reserved.
    pipe: Pipe the event is raised on. The matching [`sync_wait`][pypto.language.system.sync_wait] must
        name the same pipe -- pairing event ids and pipes is the author's responsibility.
    ffts_mode: Optional FFTS mode, 0, 1 or 2. Accepted by ``sync_set`` only.
    core_type: ``pl.KernelType.AIC`` or ``pl.KernelType.AIV``, to keep the event
        in the intended kernel when a mixed InCore function is expanded. Omit it
        to leave the event in both, which is what an explicitly typed kernel
        wants.
    span: Optional source span

## `sync_wait`

```text
pypto.language.system.sync_wait(event_id: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, *, pipe: pypto.pypto_core.ir.PipeType, core_type: pypto.ir.op.system_ops.KernelType | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Wait for a Cube/Vector cross-core event using a static or dynamic event id.

Args:
    event_id: Event to wait on -- the one a matching [`sync_set`][pypto.language.system.sync_set] raises.
    pipe: Pipe the event is awaited on; must match the ``sync_set``.
    core_type: ``pl.KernelType.AIC`` or ``pl.KernelType.AIV``, to keep the wait
        in the intended kernel when a mixed InCore function is expanded. Omit it
        to leave the wait in both, which is what an explicitly typed kernel
        wants.
    span: Optional source span

## `set_ffts`

```text
pypto.language.system.set_ffts(workspace: pypto.language.typing.tensor.Tensor, *, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Declare the A3 FFTS setup operand for explicit cross-core synchronization.

## `bar_v`

```text
pypto.language.system.bar_v(*, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Vector unit barrier.

## `bar_m`

```text
pypto.language.system.bar_m(*, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Matrix unit barrier.

## `bar_all`

```text
pypto.language.system.bar_all(*, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Global barrier synchronization.

## `fence`

```text
pypto.language.system.fence(*, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Memory barrier over global memory.

Lowers to ``pto.fence.barrier_all #pto.fence_scope<gm>``.

## `cacheinvalid`

```text
pypto.language.system.cacheinvalid(tensor: pypto.language.typing.tensor.Tensor | None = None, shapes: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar] | None = None, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar] | None = None, *, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Invalidate one addressed cache line, or the whole GM address space.

Two forms selected by arity:

- No arguments: invalidate the entire GM address space; lowers to
  ``pto.cmo.cacheinvalid all #pto.address_space<gm>``.
- ``(tensor, shapes, offsets)``: locate a tensor sub-region and invalidate
  only the cache line containing that view's base address; lowers to
  ``pto.partition_view`` +
  ``pto.cmo.cacheinvalid %payload_view single_cache_line : !pto.partition_tensor_view<...>``
  for every region size, a single element included. ``shapes`` does not
  make the operation walk every cache line in the region.

Args:
    tensor: Target tensor whose view base addresses the cache line; omit for whole-GM.
    shapes: Per-dimension region sizes; length must equal the tensor rank.
    offsets: Per-dimension start offsets; length must equal the tensor rank.
    span: Optional source span for debugging (auto-captured if not provided).

## `syncall`

```text
pypto.language.system.syncall(*, core_type: pypto.ir.op.system_ops.KernelType = <KernelType.MIX: 3>, mode: pypto.ir.op.system_ops.SyncAllMode = <SyncAllMode.HARD: 'hard'>, gm_workspace: pypto.language.typing.tensor.Tensor | None = None, used_cores: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Cross-core all-participant barrier (``pto::SYNCALL``).

Two modes:

- ``mode=pl.SyncAllMode.HARD`` (default): FFTS barrier with no operands.
  Requires the enclosing ``pl.spmd`` launch to fill **all** physical cores
  of ``core_type`` (a partial launch deadlocks on device — error 507018).
  The compiler rejects a partial-occupancy hard launch at compile time
  (``HardSyncallOccupancy`` verifier, issue #1935). See
  ``pypto.ir.op.system_ops.syncall``.
- ``mode=pl.SyncAllMode.SOFT``: GM-polling barrier that works at partial
  occupancy. Each participant updates a shared counter in an exclusive
  64-byte GM cache line and polls until all participants arrive. Supported
  for every participant set.

Both modes synchronize arrival only. They do not wait for preceding data
instructions or publish/invalidate business-data cache lines. For a
cross-core GM handoff that may span multiple cache lines, conservatively
call whole-GM ``pl.system.cacheinvalid()`` and ``pl.system.fence()`` before
the barrier, then call ``pl.system.cacheinvalid()`` again on the consumer
before it reads. The tensor-region overload covers only the cache line
containing the view's base address.

Soft-mode arguments:

Args:
    core_type: Participant set, a [`KernelType`][pypto.language.system.KernelType]
        member — ``MIX`` rendezvouses both kernels, and then ``used_cores`` is the
        *total* AIC + AIV participant count.
    mode: A [`SyncAllMode`][pypto.language.system.SyncAllMode] member.
    gm_workspace: Soft mode only. A shared, zero-initialized GM ``INT32``
        tensor with at least 16 elements (64 bytes), visible to every
        participating block. Pass it as a kernel parameter so all SPMD
        blocks share one buffer. The buffer must occupy an exclusive cache
        line and be zero-initialized before its first use.
    used_cores: Soft mode only. Required participant count as a Python int
        in the INT32 range or an ``INT32`` scalar. Pass 0 explicitly to ask
        PTO-ISA to derive the count from the device launch configuration.
        That opt-in is unsafe when the runtime's logical grid differs from
        the device launch registers, including the currently pinned Simpler
        runtime.
    span: Optional source span for debugging (auto-captured if not provided).

Returns:
    Call expression for system.syncall.

## `tpush_to_aiv`

```text
pypto.language.system.tpush_to_aiv(tile: pypto.language.typing.tile.Tile, *, split: int, lane_stride: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Push tile data from AIC to AIV via cross-core pipe.

The Vector side receives it with [`tpop_from_aic`][pypto.language.system.tpop_from_aic] and releases the
slot with [`tfree_to_aic`][pypto.language.system.tfree_to_aic]; ``split`` and ``id`` must match across all
three.

Args:
    tile: Tile to send. Its Cube-side buffer stays live until the consumer frees the slot.
    split: pto-isa split code (0=none, 1/2=up-down/left-right, 3/4=the same
        axes over an odd extent). Selects the axis along which the two AIV
        lanes divide the tile, and how their extents relate; 0 sends it whole.
    lane_stride: Partition stride carried when a ragged boundary was balanced
        across the two AIV lanes; omit for the default box partition.
    id: Optional frontend pipe id. Omit to use PTOAS default id 0.
    span: Optional source span

## `tpush_to_aic`

```text
pypto.language.system.tpush_to_aic(tile: pypto.language.typing.tile.Tile, *, split: int, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Push tile data from AIV to AIC via cross-core pipe.

The Cube side receives it with [`tpop_from_aiv`][pypto.language.system.tpop_from_aiv] and releases the
slot with [`tfree_to_aiv`][pypto.language.system.tfree_to_aiv]; ``split`` and ``id`` must match across all
three.

Args:
    tile: Tile to send. Its Vector-side buffer stays live until the consumer frees the slot.
    split: Split mode (0=none, 1=up-down, 2=left-right). Selects the axis along
        which the two AIV lanes divide the tile; 0 sends it whole.
    id: Optional frontend pipe id. Omit to use PTOAS default id 0.
    span: Optional source span

## `tpop_from_aic`

```text
pypto.language.system.tpop_from_aic(*, shape: list[int] | None = None, dtype: pypto.pypto_core.DataType | None = None, split: int = 0, lane_stride: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.tile.Tile
```

Pop tile data from AIC cross-core pipe into AIV.

Args:
    shape: Shape of the tile to receive
    dtype: Data type of the tile to receive
    split: pto-isa split code (0=none, 1/2=up-down/left-right, 3/4=the same
        axes over an odd extent)
    lane_stride: Partition stride carried when a ragged boundary was
        balanced across the two AIV lanes; omit for the box partition
    id: Optional frontend pipe id. Omit to use PTOAS default id 0.
    span: Optional source span

## `tpop_from_aiv`

```text
pypto.language.system.tpop_from_aiv(*, shape: list[int] | None = None, dtype: pypto.pypto_core.DataType | None = None, split: int = 0, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.tile.Tile
```

Pop tile data from AIV cross-core pipe into AIC.

Args:
    shape: Shape of the tile to receive
    dtype: Data type of the tile to receive
    split: Split mode (0=none, 1=up-down, 2=left-right)
    id: Optional frontend pipe id. Omit to use PTOAS default id 0.
    span: Optional source span

## `aic_initialize_pipe`

```text
pypto.language.system.aic_initialize_pipe(c2v_consumer_buf: pypto.pypto_core.ir.Expr | int | float | pypto.ir.op.system_ops._UnwrapsToExpr = 0, v2c_consumer_buf: pypto.pypto_core.ir.Expr | int | float | pypto.ir.op.system_ops._UnwrapsToExpr = 0, *, dir_mask: int, slot_size: int, slot_num: int | None = None, local_slot_num: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Initialize cross-core pipe on AIC side.

Args:
    c2v_consumer_buf: C2V consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
    v2c_consumer_buf: V2C consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
    dir_mask: Direction mask for pipe
    slot_size: Size of each pipe slot
    slot_num: Optional ring-buffer slot count. Omit to let PTOAS pick its
        default (8 unidirectional, 4 per direction bidirectional).
    local_slot_num: Optional local slot count (a2/a3 only, must be
        ``<= slot_num``). On a3 the reserved/imported buffer is sized
        ``slot_size * local_slot_num``; on a5 it is ``slot_size * slot_num``.
    id: Optional frontend pipe id. Omit to use PTOAS default id 0.
    span: Optional source span

## `aiv_initialize_pipe`

```text
pypto.language.system.aiv_initialize_pipe(c2v_consumer_buf: pypto.pypto_core.ir.Expr | int | float | pypto.ir.op.system_ops._UnwrapsToExpr = 0, v2c_consumer_buf: pypto.pypto_core.ir.Expr | int | float | pypto.ir.op.system_ops._UnwrapsToExpr = 0, *, dir_mask: int, slot_size: int, slot_num: int | None = None, local_slot_num: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
```

Initialize cross-core pipe on AIV side.

Args:
    c2v_consumer_buf: C2V consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
    v2c_consumer_buf: V2C consumer buffer base (Expr, int, or DSL ``Scalar``; default 0)
    dir_mask: Direction mask for pipe
    slot_size: Size of each pipe slot
    slot_num: Optional ring-buffer slot count. Omit to let PTOAS pick its
        default (8 unidirectional, 4 per direction bidirectional).
    local_slot_num: Optional local slot count (a2/a3 only, must be
        ``<= slot_num``). On a3 the reserved/imported buffer is sized
        ``slot_size * local_slot_num``; on a5 it is ``slot_size * slot_num``.
    id: Optional frontend pipe id. Omit to use PTOAS default id 0.
    span: Optional source span

## `reserve_buffer`

```text
pypto.language.system.reserve_buffer(*, name: str, size: int, base: int = -1, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.scalar.Scalar
```

Reserve a named buffer for cross-core communication.

Args:
    name: Buffer name for cross-core reference.
    size: Buffer size in bytes.
    base: Base address in local SRAM. Use AUTO (-1) to let the compiler
          pick a non-conflicting address, or an explicit integer for
          manual kernels.
    span: Optional source span.

Returns:
    ``pl.Scalar[pl.INT32]`` wrapping the ``system.reserve_buffer`` IR call (PTO ``... -> i32``).

## `import_peer_buffer`

```text
pypto.language.system.import_peer_buffer(*, name: str, peer_func: str, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.scalar.Scalar
```

Import a buffer from a peer function in the same group.

Args:
    name: Buffer name to import (must match peer's reserve_buffer name).
    peer_func: Name of the peer function that owns the buffer.
    span: Optional source span.

Returns:
    ``pl.Scalar[pl.INT32]`` wrapping the ``system.import_peer_buffer`` IR call (PTO ``... -> i32``).

## `tfree_to_aic`

```text
pypto.language.system.tfree_to_aic(tile: pypto.language.typing.tile.Tile, span: pypto.pypto_core.ir.Span | None = None, *, split: int | None = None, id: int | None = None) -> pypto.pypto_core.ir.Call
```

Release ring buffer slot back to AIC producer.

Call this once the tile from [`tpop_from_aic`][pypto.language.system.tpop_from_aic] has been consumed.
Until it runs, the slot stays occupied and the producer blocks once the ring fills.

Args:
    tile: The tile returned by the matching [`tpop_from_aic`][pypto.language.system.tpop_from_aic].
    span: Optional source span
    split: Leave ``None``. The ``StampTfreeSplit`` pass always takes this from
        the originating ``tpop``, so a value passed here cannot override it.
    id: Optional frontend pipe id, inherited from the originating ``tpop`` when
        omitted. Supplying one that disagrees with the ``tpop`` is rejected.

## `tfree_to_aiv`

```text
pypto.language.system.tfree_to_aiv(tile: pypto.language.typing.tile.Tile, span: pypto.pypto_core.ir.Span | None = None, *, split: int | None = None, id: int | None = None) -> pypto.pypto_core.ir.Call
```

Release ring buffer slot back to AIV producer.

Call this once the tile from [`tpop_from_aiv`][pypto.language.system.tpop_from_aiv] has been consumed.
Until it runs, the slot stays occupied and the producer blocks once the ring fills.

Args:
    tile: The tile returned by the matching [`tpop_from_aiv`][pypto.language.system.tpop_from_aiv].
    span: Optional source span
    split: Leave ``None``. The ``StampTfreeSplit`` pass always takes this from
        the originating ``tpop``, so a value passed here cannot override it.
    id: Optional frontend pipe id, inherited from the originating ``tpop`` when
        omitted. Supplying one that disagrees with the ``tpop`` is rejected.

## `task_invalid`

```text
pypto.language.system.task_invalid(*, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.scalar.Scalar
```

Sentinel ``pl.Scalar[pl.TASK_ID]`` for the "no producer" TaskId.

DSL surface of the IR-level ``system.task_invalid`` op. The printer emits
``pl.system.task_invalid()`` for auto-scope TaskId placeholders so dumps
are valid Python; user code in ``pl.manual_scope`` normally writes ``None``
and the parser lowers it to this op.

Args:
    span: Optional source span.

Returns:
    ``pl.Scalar[pl.TASK_ID]`` wrapping the ``system.task_invalid`` IR call.

## `task_dummy`

```text
pypto.language.system.task_dummy(*, deps: collections.abc.Sequence[pypto.language.typing.scalar.Scalar | pypto.language.typing.array.Array | None]) -> pypto.language.typing.scalar.Scalar
```

Dependency-only dummy TaskId barrier.

This is a parser construct: ``pl.system.task_dummy(deps=[...])`` is
intercepted syntactically and lowered to ``system.task_dummy`` with manual
dep edges. The body exists so the public DSL name resolves for static
checkers and imports.

Args:
    deps: TaskIds the barrier waits on -- each a ``pl.Scalar[pl.TASK_ID]``
        returned by ``pl.submit``, or a ``pl.array`` of them for fan-in.
        ``None`` entries are skipped, so a conditionally-produced TaskId needs
        no branch at the call site.

Returns:
    A ``pl.Scalar[pl.TASK_ID]`` that becomes ready once every dep has. Depend on
    it instead of listing all of `deps` again at each consumer.

## `available_cluster_count`

```text
pypto.language.system.available_cluster_count(*, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.scalar.Scalar
```

This run's MIX cluster (= AIC) count, as reported by the runtime.

Use it as the ``core_num`` of a mixed (AIC+AIV) or cube-only
``pl.spmd`` launch instead of a literal: the count belongs to the device
the run lands on, so a baked-in literal under- or over-fills the launch on
any device with a different usable cluster count. A hard
``pl.system.syncall`` needs full occupancy to complete, so this is the only
launch width that keeps it deadlock-free across devices.

Codegen lowers it to the orchestration helper ``rt_available_cluster_count()``.

Args:
    span: Optional source span.

Returns:
    ``pl.Scalar[pl.INT32]`` wrapping the ``system.available_cluster_count`` IR call.

Example:
    >>> with pl.spmd(pl.system.available_cluster_count(), sync_start=True):
    ...     out = self.mixed_kernel(a, b, out)

Pass the call directly as ``core_num`` as shown. Binding it to a name first
(``n = pl.system.available_cluster_count()``) also compiles, but the launch
width then reaches the outlined Spmd wrapper as a reference to a variable in
the caller, which the IR printer cannot re-parse.

## `available_aiv_count`

```text
pypto.language.system.available_aiv_count(*, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.scalar.Scalar
```

This run's standalone AIV core count, as reported by the runtime.

The AIV counterpart of [`available_cluster_count`][pypto.language.system.available_cluster_count] — the
``core_num`` of a vector-only ``pl.spmd`` launch. A mixed launch sizes itself on
[`available_cluster_count`][pypto.language.system.available_cluster_count] (one block per cluster), not on
this.

Codegen lowers it to the orchestration helper ``rt_available_aiv_count()``.

Args:
    span: Optional source span.

Returns:
    ``pl.Scalar[pl.INT32]`` wrapping the ``system.available_aiv_count`` IR call.

Example:
    >>> with pl.spmd(pl.system.available_aiv_count(), sync_start=True):
    ...     out = self.aiv_kernel(a, b, out)
