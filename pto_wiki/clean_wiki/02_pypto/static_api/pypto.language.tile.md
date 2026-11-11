# `pypto.language.tile` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `MemRefType`

```text
pypto.language.tile.MemRefType()
```

Opaque sentinel type for ``MemRef``-typed variables in printed IR.

The C++ printer emits ``pl.MemRefType`` as the annotation for a bare
``MemRef`` variable (the SSA-edge type of a ``MemRef`` expression).  This
class exists solely so that the printed Python source is valid Python that
the text-parser can ``exec()``.

Note: this is *not* the type of a ``tile.alloc`` / ``tensor.alloc`` result —
those produce a base ``Ptr`` (``PtrType``); see [`alloc`][pypto.language.tile.alloc].

## `alloc`

```text
pypto.language.tile.alloc(memory_space: pypto.pypto_core.ir.MemorySpace, size: int, *, pinned: bool = False) -> pypto.language.typing.ptr.Ptr
```

Stub for the internal ``tile.alloc`` IR operation.

This function is never called in user-written DSL code.  It is emitted
by the C++ python-printer after the InitMemRef / AllocateMemoryAddr
passes and must be importable so that the printed source is valid Python
that the text-parser can ``exec()``.

The result is a base ``Ptr`` (allocation identity token): the printer
annotates the assignment target as ``pl.Ptr``, matching the IR design
where ``tile.alloc`` / ``tensor.alloc`` Calls carry ``PtrType``.

Args:
    memory_space: Target memory space (e.g. ``pl.Mem.Vec``)
    size: Allocation size in bytes
    pinned: True when the author declared this allocation via a
        one-argument ``pl.MemRef(...)``. PyPTO memory planners then keep
        its membership isolated from other allocations.

Returns:
    Opaque ``Ptr`` sentinel (unused at runtime — the parser intercepts the
    call in the AST and never invokes this stub)

## `create_tile`

```text
pypto.language.tile.create_tile(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, target_memory: pypto.pypto_core.ir.MemorySpace | None = None, transpose: bool | None = None, *, flat_layout: bool | None = None, compact: bool | None = None) -> pypto.language.typing.tile.Tile
```

Create a tile from a shape.

Args:
    shape: Shape of the tile
    dtype: Data type of the tile
    target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right).
        ``None`` (the default) leaves the space unset for the compiler to place.
    transpose: When True, allocate the transposed Mat (ZN) fractal layout for a
        matmul ``b_trans`` B-operand (the layout a DN-source ``gather_row`` fills).
        Default ``None`` keeps the canonical layout and is omitted from the op.
    flat_layout: Keyword-only. When True, allocate a flat (non-fractal,
        slayout=none_box) L1/cbuf tile — a contiguous staging buffer rather
        than the boxed NZ layout Mat tiles normally carry. Requires
        ``target_memory=Mat`` and is mutually exclusive with ``transpose``.
        Default ``None`` keeps the canonical layout.
    compact: Keyword-only. Compiler-internal. Declares that this L0C buffer
        holds a valid-region-packed product -- N-fractal pitch
        ``ceil(validRow/16)*16`` rather than the physical row count, which is
        what ``mad`` writes when the matmul's left operand is row-narrowed.
        Requires ``target_memory=Acc``. Kernels do not set this;
        ``AutoTileMatmulL0`` declares it on the accumulator seed it
        synthesizes for a split K.

Returns:
    Tile wrapping the create operation

## `create`

```text
pypto.language.tile.create(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, target_memory: pypto.pypto_core.ir.MemorySpace | None = None, transpose: bool | None = None, *, flat_layout: bool | None = None, compact: bool | None = None) -> pypto.language.typing.tile.Tile
```

Create a tile from a shape.

Args:
    shape: Shape of the tile
    dtype: Data type of the tile
    target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right).
        ``None`` (the default) leaves the space unset for the compiler to place.
    transpose: When True, allocate the transposed Mat (ZN) fractal layout for a
        matmul ``b_trans`` B-operand (the layout a DN-source ``gather_row`` fills).
        Default ``None`` keeps the canonical layout and is omitted from the op.
    flat_layout: Keyword-only. When True, allocate a flat (non-fractal,
        slayout=none_box) L1/cbuf tile — a contiguous staging buffer rather
        than the boxed NZ layout Mat tiles normally carry. Requires
        ``target_memory=Mat`` and is mutually exclusive with ``transpose``.
        Default ``None`` keeps the canonical layout.
    compact: Keyword-only. Compiler-internal. Declares that this L0C buffer
        holds a valid-region-packed product -- N-fractal pitch
        ``ceil(validRow/16)*16`` rather than the physical row count, which is
        what ``mad`` writes when the matmul's left operand is row-narrowed.
        Requires ``target_memory=Acc``. Kernels do not set this;
        ``AutoTileMatmulL0`` declares it on the accumulator seed it
        synthesizes for a split K.

Returns:
    Tile wrapping the create operation

## `read`

```text
pypto.language.tile.read(tile: pypto.language.typing.tile.Tile, indices: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr]) -> pypto.language.typing.scalar.Scalar
```

Read a scalar value from a tile at given indices.

Args:
    tile: Input tile
    indices: A single index expression (for 1-D flat access) or a list of
        index expressions (one per tile dimension)

Returns:
    Scalar wrapping the read operation

## `write`

```text
pypto.language.tile.write(tile: pypto.language.typing.tile.Tile, indices: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], value: pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.pypto_core.ir.Expr
```

Write a scalar value into a tile at given indices.

Args:
    tile: Destination tile
    indices: A single index expression (for 1-D flat access) or a list of
        index expressions (one per tile dimension)
    value: Scalar value to write (DSL Scalar or raw Expr)

Returns:
    The underlying ``tile.write`` call expression. Direct callers
    typically ignore it; the DSL parser surfaces it as an ``EvalStmt``.

## `load`

```text
pypto.language.tile.load(tensor: pypto.language.typing.tensor.Tensor, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], shapes: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, target_memory: pypto.pypto_core.ir.MemorySpace | None = None, clamp: bool = False, cache: pypto.pypto_core.ir.CachePolicy | None = None) -> pypto.language.typing.tile.Tile
```

Copy data from tensor to unified buffer (tile).

Only the valid extent is read, so the tile may be larger than the region that
exists in the source. The tile's valid region is the source's valid region,
shifted by ``offsets`` and cut to the tile — a load never reports source bytes
that do not exist as real data.

Args:
    tensor: Source tensor
    offsets: Offsets in each dimension. Always in the source tensor's
        coordinate system. Each element must be an integer scalar — a float
        (``n / 2`` rather than ``n // 2``) or a nested sequence is rejected.
    shapes: Shape of the region to load in each dimension. Always in the
        source tensor's coordinate system. Each element must be an integer
        scalar, on the same terms as ``offsets``.
    valid_shape: Valid shape of the tile in each dimension. When provided, sets
        TileView.valid_shape in the output TileType. When omitted, shapes is used
        as valid_shape. Uses the same coordinate convention as shapes. Narrows
        the tile; cannot widen it past what the source has. Each element must
        be an integer scalar — one extent per dimension, not a nested
        ``[start, extent]`` pair.
    target_memory: Target memory space (MemorySpace.Vec or MemorySpace.Mat).
        ``None`` (the default) leaves the space unset for the compiler to place.
        MX-layout tensors require an explicit MemorySpace.Mat.
    clamp: Sanction a read that runs off the end of the source. By default a
        load asserts ``offsets + valid_shape`` stays inside the source and is
        rejected when that provably fails; ``clamp=True`` cuts the request back
        to the source edge instead.
    cache: GM cache-access policy for *this* read. ``None`` (the default)
        states no policy, leaving any scope-level declaration to apply.
        ``CachePolicy.BYPASS`` declares a streaming read — it asserts the
        bytes have no reuse worth caching and that nothing writes them while
        the kernel runs; coherency is the author's contract (see
        [`pl.set_cache_policy`][pypto.language.tensor.set_cache_policy] for
        the full contract). An explicit value here always wins over a
        scope-level ``pl.set_cache_policy`` declaration for the same tensor,
        in both directions: ``cache=CachePolicy.DEFAULT`` opts this one read
        back into the cache inside a bypassing scope. PTOAS has no L2-bypass
        path yet (https://github.com/hw-native-sys/PTOAS/issues/1356), so a
        BYPASS request warns and compiles as an ordinary cached access today.

Returns:
    Tile wrapping the load operation

Example:
    >>> # 2D load
    >>> tile = load(tensor, offsets=[0, 0], shapes=[32, 32])
    >>> # streaming read, no cache reuse expected
    >>> tile = load(tensor, [0, 0], [32, 32], cache=pl.CachePolicy.BYPASS)

## `store`

```text
pypto.language.tile.store(tile: pypto.language.typing.tile.Tile, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], output_tensor: ~_TensorT, shapes: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_, st_phase: pypto.pypto_core.ir.STPhase = STPhase.Unspecified) -> ~_TensorT
```

Copy data from tile back to tensor.

Args:
    tile: Source tile
    offsets: Offsets in each dimension. Each element must be an integer
        scalar — a float or a nested sequence is rejected.
    output_tensor: Output tensor
    shapes: Optional ND partition shape. Injected by FlattenTileNdTo2D for ND tensors.
        Each element must be an integer scalar, on the same terms as ``offsets``.
    atomic: Combine mode for the global-memory write. ``AtomicType.None_``
        (default) overwrites; ``AtomicType.Add`` atomically adds the tile
        into existing GM contents — used for split-K accumulation, where
        several cores accumulate partial products into one output.

        NOTE: atomic-add accumulation order across cores is not fixed, so
        floating-point results are non-deterministic. The destination must
        be zero-initialised before the kernel runs. Supported tile dtypes:
        fp32 / bf16 / fp16 / int32 / int16 / int8. bf16 atomic-add is
        available on the Ascend910B (A2/A3) profile; it is not supported on
        A5, where an fp32 accumulator + cast is required instead.
    st_phase: Consumer-side unit-flag phase. A producer that finishes with
        ``acc_phase=pl.AccPhase.Final`` must be consumed by a store with
        ``st_phase=pl.STPhase.Final`` so the unit flag is cleared.

Returns:
    Tensor wrapping the store operation

Example:
    >>> # 2D store
    >>> result = store(tile, [0, 0], tensor)
    >>> # 3D store
    >>> result = store(tile, [0, 0, 0], tensor)
    >>> # atomic-add store (split-K)
    >>> result = store(partial, [0, 0], out, atomic=pl.AtomicType.Add)
    >>> # clear the unit flag after a final phased accumulation
    >>> result = store(acc, [0, 0], out, st_phase=pl.STPhase.Final)

## `assemble`

```text
pypto.language.tile.assemble(target: pypto.language.typing.tile.Tile, source: pypto.language.typing.tile.Tile, offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr]) -> pypto.language.typing.tile.Tile
```

Write source tile data into target tile at specified offset.

Args:
    target: Target tile to update
    source: Source tile to write
    offset: Offset dimensions for where to write

Returns:
    Tile wrapping the assemble operation

## `gather_row`

```text
pypto.language.tile.gather_row(dst: pypto.language.typing.tile.Tile, src: pypto.language.typing.tensor.Tensor, dst_offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], src_offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], shapes: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], transpose: bool = False, *, valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None) -> pypto.language.typing.tile.Tile
```

Load one GM row directly into a sub-region of an on-chip tile (DPS).

Per-row primitive of the paged-gather lowering: DMAs one GM row window
straight into ``dst`` at ``dst_offset`` (``pto.subview`` of ``dst`` +
``pto.tload``, ``GM -> on-chip``, no ``pto.tmov``). The caller computes the
physical ``src_offset`` (block-table lookup + bias) and the ``dst_offset``
slot itself, so arbitrary gather logic stays in the kernel. Writes ``dst``
in place, so a loop-carried accumulator is filled row by row and feeds
``pl.matmul`` directly — the tile-level counterpart of
``pypto.language.op.tensor_ops.gather_row``.

Args:
    dst: Destination on-chip accumulator tile (Mat/L1 or Vec/UB).
    src: Source pool in GM (a ``Tensor``).
    dst_offset: ``[row, col]`` slot within ``dst`` to write.
    src_offset: ``[row, col]`` physical offset within the GM ``src``.
    shapes: GM row window shape ``[r, c]`` (typically ``[1, size]``).
        Must be compile-time constant.
    valid_shape: How much of that window to actually transfer, defaulting to
        all of it. May hold runtime ``Scalar`` values, so a dynamic row count
        leaves the tile's allocation and layout untouched. Not supported
        together with ``transpose=True``.
    transpose: Place the GM row ``[r, c]`` as an on-chip column ``[c, r]`` —
        fills a matmul ``b_trans`` B-operand without a GM round-trip
        (Mat/L1 only).

Returns:
    Tile aliasing ``dst`` (written in place).

## `extract`

```text
pypto.language.tile.extract(src: pypto.language.typing.tile.Tile, index_row: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, index_col: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], *, target_memory: pypto.pypto_core.ir.MemorySpace) -> pypto.language.typing.tile.Tile
```

Extract a sub-tile from ``src`` at ``(index_row, index_col)`` — ISA TEXTRACT.

Maps to ISA TEXTRACT Variant 1 (Standard Extract). The result tile has the
given static ``shape`` and lives in ``target_memory``.

Args:
    src: Source tile (typically in Mat or Acc memory)
    index_row: Starting row offset
    index_col: Starting col offset
    shape: Static 2D shape of the extracted sub-tile
    target_memory: Destination memory space —
        ``Left`` / ``Right`` for Mat sources, ``Mat`` for Acc sources

Returns:
    Tile of the requested shape in ``target_memory``

## `scatter_update`

```text
pypto.language.tile.scatter_update(input: pypto.language.typing.tile.Tile, *args: Any, **kwargs: Any) -> pypto.language.typing.tile.Tile
```

Update tile rows at positions specified by 2D index tile with values from src.

Supports two rank variants:

- 2D: ``input [rows, d]``, ``src [b*s, d]``, ``index [b, s]``
- 4D: ``input [blockNum, blockSize, 1, d]``, ``src [b, s, 1, d]``, ``index [b, s]``

Accepts the same flexible call shapes as the IR builder
``pypto.ir.op.tile.scatter_update``:

- ``scatter_update(input, dim, index, src)``
- ``scatter_update(input, index, src, dim=-2)``
- ``scatter_update(input, dim, index=..., src=...)``

Tile / Scalar wrappers are unwrapped before forwarding so the IR builder
receives raw ``Expr`` operands.

## `concat`

```text
pypto.language.tile.concat(src0: pypto.language.typing.tile.Tile, src1: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Concatenate two tiles along the column dimension.

Args:
    src0: First source tile
    src1: Second source tile

Returns:
    Tile with concatenated columns

## `move`

```text
pypto.language.tile.move(tile: pypto.language.typing.tile.Tile, target_memory: pypto.pypto_core.ir.MemorySpace, blayout: pypto.pypto_core.ir.TileLayout | None = None, slayout: pypto.pypto_core.ir.TileLayout | None = None) -> pypto.language.typing.tile.Tile
```

Move tile between memory levels.

Args:
    tile: Input tile
    target_memory: Target memory space (MemorySpace.Vec, .Mat, .Left, .Right,
        .LeftScale, .RightScale)
    blayout: Optional block layout for the destination tile
    slayout: Optional scatter layout for the destination tile

Returns:
    Tile wrapping the move operation

## `aiv_shard`

```text
pypto.language.tile.aiv_shard(x: ~_SplitOperandT, span: pypto.pypto_core.ir.Span | None = None) -> ~_SplitOperandT
```

Bring a cube-produced operand onto the AIV lane (AIC -> AIV crossing).

In a data-parallel region (``UP_DOWN`` / ``LEFT_RIGHT``) the crossing also
**halves** the operand along the split axis, so each lane gets one half; in a
task-parallel ``mode=NONE`` region there is no split axis, so it crosses and
**preserves the shape**. Either way, writing it is how a C->V crossing into a
region is named — the ``AivSplitValid`` verifier rejects an unnamed one.

The split mode is **inherited** from the enclosing
``for aiv_id in pl.split_aiv(mode=...)`` scope — it is not passed here.
This wrapper therefore only resolves inside a parsed kernel, where the
parser intercepts the call and fills the inherited mode. Calling it eagerly
(outside a parsed program) raises, since there is no scope to read the mode
from.

The operand may be a ``Tile`` (legacy ``@pl.program`` form -> ``tile.aiv_shard``)
or a high-level ``Tensor`` (``@pl.jit`` / ``pl.spmd`` form -> ``tensor.aiv_shard``,
lowered 1:1 to ``tile.aiv_shard`` at ConvertTensorToTileOps). The Tensor form
is region-only. Distributed tensors are not supported.

Args:
    x: Input operand (2D Tile or Tensor)
    span: Optional source span

Returns:
    Operand of the same kind: the split axis halved in a data-parallel region,
    the shape unchanged in a ``mode=NONE`` one.

## `aic_gather`

```text
pypto.language.tile.aic_gather(x: ~_SplitOperandT, span: pypto.pypto_core.ir.Span | None = None) -> ~_SplitOperandT
```

Hand a vector-produced operand to the cube (AIV -> AIC crossing).

Inverse of [`aiv_shard`][pypto.language.tile.aiv_shard]: in a data-parallel region it **rejoins** the two
lanes' halves along the split axis, and in a task-parallel ``mode=NONE`` region
it crosses and **preserves the shape**. It is how a V->C crossing out of a
region is named; an unnamed one is rejected by the ``AivSplitValid`` verifier.

Out of a ``mode=NONE`` region the two lanes share one destination slot with no
per-lane offset and nothing arbitrates between them: both push, so when they
hold different values the cube receives an **unspecified** one of the two.
Guarding the *production* of the value does not help — lane 1 still reaches
the push and still sends its own tile. Gather only a value both lanes agree
on; if they must contribute different data, use a data-parallel region.

Like [`aiv_shard`][pypto.language.tile.aiv_shard], the split mode is
**inherited** from the enclosing ``for aiv_id in pl.split_aiv(mode=...)``
scope and must not be passed here. Calling it eagerly (outside a parsed
program) raises, since there is no scope to read the mode from.

The operand may be a ``Tile`` (legacy ``@pl.program`` form -> ``tile.aic_gather``)
or a high-level ``Tensor`` (``@pl.jit`` / ``pl.spmd`` form -> ``tensor.aic_gather``,
lowered 1:1 to ``tile.aic_gather`` at ConvertTensorToTileOps). The Tensor form
is region-only. Distributed tensors are not supported.

Args:
    x: Input operand (2D Tile or Tensor)
    span: Optional source span

Returns:
    Operand of the same kind: the split axis doubled in a data-parallel region,
    the shape unchanged in a ``mode=NONE`` one.

## `full`

```text
pypto.language.tile.full(shape: list[int], dtype: pypto.pypto_core.DataType, value: int | float) -> pypto.language.typing.tile.Tile
```

Create a tile from a shape and fill with value in Vec.

Args:
    shape: Shape of the tile
    dtype: Data type of the tile
    value: filling scalar

Returns:
    Tile wrapping the full operation

## `ci`

```text
pypto.language.tile.ci(start: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int], dtype: pypto.pypto_core.DataType = int32, descending: bool = False, *, tmp: pypto.language.typing.tile.Tile | None = None) -> pypto.language.typing.tile.Tile
```

Generate a contiguous integer sequence into a tile.

Equivalent to ``numpy.arange``-style index generation. Maps to ``pto.tci``.
For a column index ``k`` in the first row of the destination, ascending gives
``dst[0, k] = start + k`` and descending gives ``dst[0, k] = start - k``.

Note:
    ``pto.tci`` uses the destination's valid-column count as the sequence
    length and does NOT populate additional rows. Leading dimensions must
    be 1 — prefer shapes of the form ``[1, N]``.

Args:
    start: Starting integer (plain int or a Scalar). Must match ``dtype``.
    shape: Shape of the destination tile (static, innermost dim != 1).
    dtype: Destination dtype. One of {INT16, INT32}. Defaults to INT32.
    descending: If True, generate a descending sequence.
    tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated.

Returns:
    Tile wrapping the ci operation.

## `arange`

```text
pypto.language.tile.arange(start: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int], dtype: pypto.pypto_core.DataType = int32, descending: bool = False, *, tmp: pypto.language.typing.tile.Tile | None = None) -> pypto.language.typing.tile.Tile
```

Generate a contiguous integer sequence into a tile.

Equivalent to ``numpy.arange``-style index generation. Maps to ``pto.tci``.
For a column index ``k`` in the first row of the destination, ascending gives
``dst[0, k] = start + k`` and descending gives ``dst[0, k] = start - k``.

Note:
    ``pto.tci`` uses the destination's valid-column count as the sequence
    length and does NOT populate additional rows. Leading dimensions must
    be 1 — prefer shapes of the form ``[1, N]``.

Args:
    start: Starting integer (plain int or a Scalar). Must match ``dtype``.
    shape: Shape of the destination tile (static, innermost dim != 1).
    dtype: Destination dtype. One of {INT16, INT32}. Defaults to INT32.
    descending: If True, generate a descending sequence.
    tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated.

Returns:
    Tile wrapping the ci operation.

## `tri`

```text
pypto.language.tile.tri(diagonal: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int], valid_shape: collections.abc.Sequence[int] | None = None, dtype: pypto.pypto_core.DataType = int32, upper: bool = False) -> pypto.language.typing.tile.Tile
```

Generate a lower- or upper-triangular mask tile.

``upper=False`` writes one where ``j <= i + diagonal``; ``upper=True``
writes one where ``j >= i + diagonal``. Only the optional valid region is
written.

Args:
    diagonal: Offset of the boundary from the main diagonal, in columns.
        0 includes the diagonal; positive shifts it right, negative left.
        May be a runtime ``Scalar``.
    shape: Shape of the destination tile (static).
    valid_shape: Optional written region (each dim ``<= shape``). Elements
        outside it are not written, so their value is whatever the freshly
        allocated tile holds. Defaults to the full shape.
    dtype: Destination dtype. Defaults to ``INT32``.
    upper: Select the upper triangle instead of the lower.

Returns:
    A tile holding 1 inside the selected triangle and 0 outside it.

## `random`

```text
pypto.language.tile.random(key0: int | pypto.language.typing.scalar.Scalar, key1: int | pypto.language.typing.scalar.Scalar, counter0: int | pypto.language.typing.scalar.Scalar, counter1: int | pypto.language.typing.scalar.Scalar, counter2: int | pypto.language.typing.scalar.Scalar, counter3: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int], valid_shape: collections.abc.Sequence[int] | None = None, dtype: pypto.pypto_core.DataType = uint32, rounds: int = 10) -> pypto.language.typing.tile.Tile
```

Generate counter-based pseudo-random values into a tile.

Implements a counter-based (Philox/ChaCha-style) RNG. Each element is derived
deterministically from the 64-bit key ``(key0, key1)`` and 128-bit counter
``(counter0..counter3)`` plus the element position, so the same seeds always
reproduce the same tile. Maps to ``pto.trandom``.

Args:
    key0: Low INT32 key word (plain int or Scalar).
    key1: High INT32 key word (plain int or Scalar).
    counter0: First INT32 counter word.
    counter1: Second INT32 counter word.
    counter2: Third INT32 counter word.
    counter3: Fourth INT32 counter word.
    shape: Shape of the destination tile (static).
    valid_shape: Optional written region (each dim ``<= shape``); ``pto.trandom``
        only fills the valid rows/cols. Defaults to the full shape.
    dtype: Destination dtype. One of {INT32, UINT32}. Defaults to UINT32.
    rounds: Cipher round count, 7 or 10. Defaults to 10.

Returns:
    Tile wrapping the random operation.

## `fillpad`

```text
pypto.language.tile.fillpad(tile: pypto.language.typing.tile.Tile, pad_value: pypto.pypto_core.ir.PadValue | int | float = PadValue.zero) -> pypto.language.typing.tile.Tile
```

Fill remaining tile elements with specified padding value.

Args:
    tile: Input tile
    pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
        the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
        ``PadValue.zero``. Other values raise — the hardware only supports
        the three padding modes.

Returns:
    Tile wrapping the fillpad operation

## `fillpad_inplace`

```text
pypto.language.tile.fillpad_inplace(tile: pypto.language.typing.tile.Tile, pad_value: pypto.pypto_core.ir.PadValue | int | float = PadValue.zero) -> pypto.language.typing.tile.Tile
```

Fill padding elements of input tile in place.

Unlike fillpad which allocates a new output tile, this operation reuses
the input tile's UB buffer. The result shares the same memory address,
making it equivalent to TFILLPAD_INPLACE on the hardware.

Args:
    tile: Input tile
    pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
        the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
        ``PadValue.zero``. Other values raise — the hardware only supports
        the three padding modes.

Returns:
    Tile with padding filled (shares memory with the input tile).

## `fillpad_expand`

```text
pypto.language.tile.fillpad_expand(tile: pypto.language.typing.tile.Tile, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], pad_value: pypto.pypto_core.ir.PadValue | int | float = PadValue.zero) -> pypto.language.typing.tile.Tile
```

Copy a smaller source tile into a larger destination tile, padding the rest.

Unlike [`fillpad`][pypto.language.tile.fillpad] (which keeps the same physical shape and only fills the
valid-region expansion), this op produces a *larger* output tile: the source's
valid region is copied to the top-left and every other element is filled with
``pad_value``. Equivalent to TFILLPAD_EXPAND on the hardware.

Args:
    tile: Source tile
    shape: Destination shape; each dimension must be >= the source dimension
    pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
        the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
        ``PadValue.zero``. Other values raise — the hardware only supports
        the three padding modes.

Returns:
    Tile wrapping the fillpad_expand operation (a new, larger tile).

## `get_block_idx`

```text
pypto.language.tile.get_block_idx() -> pypto.language.typing.scalar.Scalar
```

Get the current block index.

This operation returns the index of the current compute tile. It is typically
used in tile-level programming to identify which block of data is being processed.

Returns:
    Scalar wrapping the get_block_idx operation (INDEX type)

Example:
    >>> block_idx = pl.tile.get_block_idx()
    >>> if block_idx < 10:
    >>>     # Process first 10 blocks differently
    >>>     ...

## `get_subblock_idx`

```text
pypto.language.tile.get_subblock_idx() -> pypto.language.typing.scalar.Scalar
```

Get the current sub-block (vector core) index.

Returns the index of the current vector core within a split execution.
Core 0 returns 0, core 1 returns 1.

Returns:
    Scalar wrapping the get_subblock_idx operation (INDEX type)

## `get_block_num`

```text
pypto.language.tile.get_block_num() -> pypto.language.typing.scalar.Scalar
```

Get the total number of blocks in the current SPMD task.

This operation returns the total count of blocks dispatched for the current
task. Used with get_block_idx() for SPMD work partitioning.

Returns:
    Scalar wrapping the get_block_num operation (INDEX type)

Example:
    >>> block_idx = pl.tile.get_block_idx()
    >>> block_num = pl.tile.get_block_num()

## `add`

```text
pypto.language.tile.add(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile | int | float | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tile.Tile
```

Element-wise addition of tile and tile or scalar.

Supports broadcasting when both operands are tiles. A scalar ``rhs``
canonicalizes to ``tile.adds``.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile or scalar

Returns:
    Tile wrapping the add operation

## `sub`

```text
pypto.language.tile.sub(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile | int | float | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tile.Tile
```

Element-wise subtraction of tile and tile or scalar.

Supports broadcasting when both operands are tiles. A scalar ``rhs``
canonicalizes to ``tile.subs``.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile or scalar

Returns:
    Tile wrapping the sub operation

## `mul`

```text
pypto.language.tile.mul(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile | int | float | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tile.Tile
```

Element-wise multiplication of tile and tile or scalar.

Supports broadcasting when both operands are tiles. A scalar ``rhs``
canonicalizes to ``tile.muls``.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile or scalar

Returns:
    Tile wrapping the mul operation

## `div`

```text
pypto.language.tile.div(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile | int | float | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, high_precision: bool = False) -> pypto.language.typing.tile.Tile
```

Element-wise division of tile and tile or scalar.

Tile-tile division requires identical physical and valid shapes. A scalar
``rhs`` canonicalizes to ``tile.divs``, which does not expose the ``tdiv``
precision mode — hence ``high_precision`` applies only to the tile-tile form.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile or scalar
    high_precision: Whether to select PTOAS's high-precision division mode.
        Only available when ``rhs`` is a Tile.

Returns:
    Tile wrapping the div operation

## `adds`

```text
pypto.language.tile.adds(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise addition of tile and scalar.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the adds operation

## `subs`

```text
pypto.language.tile.subs(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise subtraction of tile and scalar.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the subs operation

## `muls`

```text
pypto.language.tile.muls(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise multiplication of tile and scalar.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the muls operation

## `divs`

```text
pypto.language.tile.divs(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise division of tile and scalar.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the divs operation

## `neg`

```text
pypto.language.tile.neg(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise negation.

Args:
    tile: Input tile

Returns:
    Tile wrapping the neg operation

## `exp`

```text
pypto.language.tile.exp(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise exponential.

Args:
    tile: Input tile

Returns:
    Tile wrapping the exp operation

## `sin`

```text
pypto.language.tile.sin(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise sine of a tile (radians). FP32 only.

Non-FP32 inputs are rejected rather than promoted — cast explicitly with
``pl.cast(tile, pl.FP32)`` first.

Args:
    tile: Input tile (FP32)

Returns:
    Tile wrapping the sin operation

## `cos`

```text
pypto.language.tile.cos(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise cosine of a tile (radians). FP32 only.

Non-FP32 inputs are rejected rather than promoted — cast explicitly with
``pl.cast(tile, pl.FP32)`` first.

Args:
    tile: Input tile (FP32)

Returns:
    Tile wrapping the cos operation

## `sqrt`

```text
pypto.language.tile.sqrt(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise square root.

Args:
    tile: Input tile

Returns:
    Tile wrapping the sqrt operation

## `rsqrt`

```text
pypto.language.tile.rsqrt(tile: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile | None = None) -> pypto.language.typing.tile.Tile
```

Element-wise reciprocal square root.

Args:
    tile: Input tile
    tmp: Optional scratch tile (same shape/dtype as ``tile``) that activates
        the high-precision PTO lowering.

Returns:
    Tile wrapping the rsqrt operation

## `recip`

```text
pypto.language.tile.recip(tile: pypto.language.typing.tile.Tile, high_precision: bool = False) -> pypto.language.typing.tile.Tile
```

Element-wise reciprocal.

Args:
    tile: Input tile
    high_precision: Whether to select PTOAS's high-precision reciprocal mode (FP16/FP32 only)

Returns:
    Tile wrapping the recip operation

## `log`

```text
pypto.language.tile.log(tile: pypto.language.typing.tile.Tile, high_precision: bool = False) -> pypto.language.typing.tile.Tile
```

Element-wise natural logarithm.

Args:
    tile: Input tile
    high_precision: Whether to select PTOAS's high-precision logarithm mode

Returns:
    Tile wrapping the log operation

## `abs`

```text
pypto.language.tile.abs(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise absolute value.

Args:
    tile: Input tile

Returns:
    Tile wrapping the abs operation

## `relu`

```text
pypto.language.tile.relu(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise ReLU activation (max(0, x)).

Args:
    tile: Input tile

Returns:
    Tile wrapping the relu operation

## `cast`

```text
pypto.language.tile.cast(tile: pypto.language.typing.tile.Tile, target_type: int | pypto.pypto_core.DataType, mode: str | int = 'round', *, tmp: pypto.language.typing.tile.Tile | None = None) -> pypto.language.typing.tile.Tile
```

Cast tile to target data type (element-wise).

Args:
    tile: Input tile (TileType)
    target_type: Target data type (DataType)
    mode: Rounding mode — string name ("none", "rint", "round", "floor",
          "ceil", "trunc", "odd") or int (0–6)
    tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated.

Returns:
    Tile wrapping the cast operation

Example:
    >>> tile_fp32 = pl.tile.cast(tile_bf16, pl.FP32)

## `quant_mx`

```text
pypto.language.tile.quant_mx(src: pypto.language.typing.tile.Tile, *, group_axis: int, dtype: pypto.pypto_core.DataType = fp8e4m3fn) -> tuple[pypto.language.typing.tile.Tile, pypto.language.typing.tile.Tile]
```

MXFP8 block-32 dynamic quantization with PTOAS ``grpAxis`` selection.

``group_axis`` is required and must be ``0`` or ``1`` (same meaning as
PTOAS ``#pto<mx_group_axis axis*>``). Axis 1 is the A-side ``[M, K]`` path;
axis 0 is the B-side ``[N, K]`` path (LowerCompositeOps transposes to
``[K, N]`` before quantization). GM ``TensorLayout.MX_*`` is only for
tensor / ``tensor.view`` annotations, not this tile op.

This release supports **MXFP8 only** (``dtype=FP8E4M3FN``). MXFP4 is out of
scope for this PR.

Args:
    src: Source tile (FP16/FP32/BF16, 2D).
        ``group_axis=1``: ``[M, K]`` with ``M%16==0``, ``K%64==0``.
        ``group_axis=0``: ``[N, K]`` with ``N%32==0``, ``K%64==0``.
    group_axis: PTOAS grouping axis — ``1`` (A-side) or ``0`` (B-side).
    dtype: Must be ``FP8E4M3FN`` (default).

Returns:
    ``group_axis=1``: ``(quant[M,K], scale[M,K/32])`` row/row ZZ scale.
    ``group_axis=0``: ``(quant[K,N], scale[K/32,N])`` — data transposed to
    Cube RHS layout; scale col/col NN.

Note:
    ``quant_mx`` and ``matmul_mx`` cannot currently share one InCore mixed
    task. Stage the quantized data and scale through GM between separate
    AIV and AIC kernels; automatic cross-core data+scale transport is a
    follow-up.

## `tmov_x2zz`

```text
pypto.language.tile.tmov_x2zz(src: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, *, group_axis: int = 1, dst_rows: int | None = None, dst_cols: int | None = None) -> pypto.language.typing.tile.Tile
```

Exponent X-to-ZZ layout conversion (A5-only).

``tmp`` is a write-only workspace. Axis1 requires capacity
``64 + ceil(rows/16) * cols`` bytes (typically 32-byte-aligned by the
caller) and ``dst_rows``/``dst_cols`` for the ZZ ``[M, G]`` shape. Axis0
(TMovDnTo2Zz) still requires a Vec tmp operand; use a minimal 32-byte-aligned
pad.

## `matmul`

```text
pypto.language.tile.matmul(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Matrix multiplication of two tiles.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile

Returns:
    Tile wrapping the matmul operation

## `batch_matmul`

```text
pypto.language.tile.batch_matmul(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Batch matrix multiplication of two tiles.

Broadcasts the batch dims: for inputs shaped ``[...batch_dims, M, K]`` and
``[...batch_dims, K, N]``, the output is ``[...broadcast_batch_dims, M, N]``.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile

Returns:
    Tile wrapping the batch_matmul operation

## `matmul_acc`

```text
pypto.language.tile.matmul_acc(acc: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, init_cond: bool | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None) -> pypto.language.typing.tile.Tile
```

Matrix multiplication with accumulation: acc += lhs @ rhs.

``init_cond`` makes the accumulator's initial value conditional: on the steps
where it holds, ``acc`` is overwritten with ``lhs @ rhs`` rather than
accumulated into. This is the split-K idiom, and it removes the need to zero
the accumulator or to peel the first K step::

    for n0 in pl.range(N // N_TILE):
        for k0 in pl.pipeline(0, K, K_TILE, stage=2):
            acc_n = pl.tile.slice(acc, [M, N_TILE], [0, n0 * N_TILE])
            pl.tile.matmul_acc(acc_n, a, b, init_cond=(k0 == 0))

A literal ``True`` / ``False`` selects one form at compile time; a runtime
predicate lowers to a branch over the two, with no phi on the accumulator.

Note:
    When ``acc`` is a window of a larger ``Mem.Acc`` (L0C) tile, slice it
    along **columns**, as above -- the window must span every row of its
    parent, or be at most 16 columns wide inside a single 16-column block.
    L0C stores 16x16 blocks column-major, so a *row* window of a parent with
    more than one block column is strided, and the MAD writes its result
    compactly from a bare pointer with no destination stride. Slicing
    ``[ROW_TILE, N]`` at ``[t0, 0]`` out of an ``[M, N]`` accumulator is
    therefore rejected by ``CanonicalizeTileSlice`` rather than silently
    miscomputed. Column windows address the same L0C memory in the order the
    hardware writes it, so nothing is lost by preferring them.

Args:
    acc: Accumulator tile
    lhs: Left-hand side tile
    rhs: Right-hand side tile
    init_cond: Optional predicate selecting overwrite over accumulate

Returns:
    Tile wrapping the matmul_acc operation

## `batch_matmul_acc`

```text
pypto.language.tile.batch_matmul_acc(acc: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, init_cond: bool | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None) -> pypto.language.typing.tile.Tile
```

Batch matrix multiplication with accumulation: acc += lhs @ rhs.

Performs the in-place ``acc += lhs @ rhs`` with batch-dim broadcasting between
``lhs`` and ``rhs``. The broadcast batch shape must equal the batch shape of
``acc`` (acc is the in-place accumulation target and is not broadcast).

``init_cond`` behaves exactly as on
[`matmul_acc`][pypto.language.op.tile_ops.matmul_acc]: where it holds, ``acc``
is overwritten with ``lhs @ rhs`` rather than accumulated into.
``FlattenTileNdTo2D`` forwards the predicate to every 2D ``tile.matmul_acc``
it unrolls this op into.

Args:
    acc: Accumulator tile (at least 2D)
    lhs: Left-hand side tile (at least 2D)
    rhs: Right-hand side tile (at least 2D)
    init_cond: Optional predicate selecting overwrite over accumulate

Returns:
    Tile wrapping the batch_matmul_acc operation

## `matmul_bias`

```text
pypto.language.tile.matmul_bias(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Matrix multiplication with bias add: C = lhs @ rhs + bias.

Args:
    lhs: Left-hand side tile [M, K]
    rhs: Right-hand side tile [K, N]
    bias: Bias tile [1, N] with the accumulator dtype (FP32 for
        floating-point matrix operands, INT32 for integer matrix operands)

Returns:
    Tile wrapping the matmul_bias operation

## `matmul_mx`

```text
pypto.language.tile.matmul_mx(lhs: pypto.language.typing.tile.Tile, lhs_scale: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs_scale: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

MX block-scale matrix multiplication.

Scales use logical ``[M, K/32]`` / ``[K/32, N]`` shapes. Both data tiles
passed to this operation must be FP8E4M3FN. For the supported FP4 x FP8
input form, explicitly cast the FP4 lhs to FP8E4M3FN before calling this
operation; native FP4 x FP4 is not supported.

Args:
    lhs: Left-hand side data tile (FP8E4M3FN)
    lhs_scale: Left-hand side scale tile (FP8E8M0)
    rhs: Right-hand side data tile (FP8E4M3FN)
    rhs_scale: Right-hand side scale tile (FP8E8M0)

Returns:
    Tile wrapping the matmul_mx operation

## `matmul_mx_acc`

```text
pypto.language.tile.matmul_mx_acc(acc: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, lhs_scale: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs_scale: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

MX block-scale matmul with accumulation.

Data operands follow [`matmul_mx`][pypto.language.tile.matmul_mx]: an FP4 lhs must first be cast to
FP8E4M3FN, and the operation itself receives two FP8E4M3FN tiles.

Args:
    acc: Accumulator tile
    lhs: Left-hand side data tile (FP8E4M3FN)
    lhs_scale: Left-hand side scale tile (FP8E8M0)
    rhs: Right-hand side data tile (FP8E4M3FN)
    rhs_scale: Right-hand side scale tile (FP8E8M0)

Returns:
    Tile wrapping the matmul_mx_acc operation

## `matmul_mx_bias`

```text
pypto.language.tile.matmul_mx_bias(lhs: pypto.language.typing.tile.Tile, lhs_scale: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs_scale: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

MX block-scale matmul with bias.

Data operands follow [`matmul_mx`][pypto.language.tile.matmul_mx]: an FP4 lhs must first be cast to
FP8E4M3FN, and the operation itself receives two FP8E4M3FN tiles.

Args:
    lhs: Left-hand side data tile (FP8E4M3FN)
    lhs_scale: Left-hand side scale tile (FP8E8M0)
    rhs: Right-hand side data tile (FP8E4M3FN)
    rhs_scale: Right-hand side scale tile (FP8E8M0)
    bias: Bias tile

Returns:
    Tile wrapping the matmul_mx_bias operation

## `gemv`

```text
pypto.language.tile.gemv(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified) -> pypto.language.typing.tile.Tile
```

General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N].

``lhs`` must have exactly one physical and logical row. The rhs logical K
must cover the lhs logical K. Inputs must use the same INT8, FP16, BF16, or FP32
dtype; the output is INT32 for INT8 inputs and FP32 otherwise.

Args:
    lhs: Row vector tile [1, K]
    rhs: Right-hand side tile [K, N]
    acc_phase: Producer-side unit-flag phase. Use ``pl.AccPhase.Partial``
        for intermediate chunks and ``pl.AccPhase.Final`` for the last chunk.

Returns:
    Tile wrapping the gemv operation

## `gemv_acc`

```text
pypto.language.tile.gemv_acc(acc: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified, *, init_cond: bool | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None) -> pypto.language.typing.tile.Tile
```

GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N].

``acc`` must use the GEMV output dtype. The logical K extents and lhs/rhs
dtype requirements are identical to [`gemv`][pypto.language.tile.gemv].

``init_cond`` makes the accumulator's initial value conditional, exactly as in
[`matmul_acc`][pypto.language.tile.matmul_acc] — GEMV is a matmul whose M is
1, run on the same cube MAD, so it carries the same predicate bit. On the
steps where it holds, ``acc`` is overwritten with ``lhs @ rhs`` rather than
accumulated into, which removes the peeled first step from split-K::

    for k0 in pl.pipeline(0, K, K_TILE):
        a = pl.load(vec, [0, k0], [1, K_TILE], target_memory=pl.MemorySpace.Mat)
        b = pl.load(mat, [k0, 0], [K_TILE, N], target_memory=pl.MemorySpace.Mat)
        acc = pl.tile.gemv_acc(acc, a, b, init_cond=(k0 == 0))

A literal ``True`` / ``False`` selects one form at compile time; a runtime
predicate lowers to a branch over the two, with no phi on the accumulator.

``init_cond`` is keyword-only because ``acc_phase`` already owns the fourth
positional slot.

Args:
    acc: Accumulator tile [1, N]
    lhs: Row vector tile [1, K]
    rhs: Right-hand side tile [K, N]
    acc_phase: Producer-side unit-flag phase. Use ``pl.AccPhase.Partial``
        for intermediate chunks and ``pl.AccPhase.Final`` for the last chunk.
    init_cond: Optional predicate selecting overwrite over accumulate

Returns:
    Tile wrapping the gemv_acc operation

## `gemv_bias`

```text
pypto.language.tile.gemv_bias(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified) -> pypto.language.typing.tile.Tile
```

GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N].

``bias`` must use the GEMV output dtype and its valid shape must cover the
logical output shape ``[1, N]``. The logical K extents and lhs/rhs dtype
requirements are identical to [`gemv`][pypto.language.tile.gemv].

Args:
    lhs: Row vector tile [1, K]
    rhs: Right-hand side tile [K, N]
    bias: Bias tile [1, N] with the accumulator dtype (FP32 for
        floating-point matrix operands, INT32 for integer matrix operands)
    acc_phase: Producer-side unit-flag phase. Use ``pl.AccPhase.Partial``
        for intermediate chunks and ``pl.AccPhase.Final`` for the last chunk.

Returns:
    Tile wrapping the gemv_bias operation

## `row_max`

```text
pypto.language.tile.row_max(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise max reduction.

Reduces the last axis with keepdim, producing output shape
``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

Args:
    tile: Input tile
    tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
        every dimension at least as large as the corresponding input dimension

Returns:
    Tile wrapping the row_max operation

## `row_sum`

```text
pypto.language.tile.row_sum(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise sum reduction.

Reduces the last axis with keepdim, producing output shape
``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

Args:
    tile: Input tile
    tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
        every dimension at least as large as the corresponding input dimension

Returns:
    Tile wrapping the row_sum operation

## `row_min`

```text
pypto.language.tile.row_min(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise min reduction.

Reduces the last axis with keepdim, producing output shape
``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

Args:
    tile: Input tile
    tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
        every dimension at least as large as the corresponding input dimension

Returns:
    Tile wrapping the row_min operation

## `row_prod`

```text
pypto.language.tile.row_prod(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise product reduction.

Reduces the last axis with keepdim, producing output shape
``input_shape[:-1] + [1]`` (e.g. ``[rows, 1]`` for a 2D ``[rows, cols]`` input).

Args:
    tile: Input tile
    tmp_tile: Scratch tile with the same dtype and rank as ``tile`` and
        every dimension at least as large as the corresponding input dimension

Returns:
    Tile wrapping the row_prod operation

## `col_sum`

```text
pypto.language.tile.col_sum(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile | None = None) -> pypto.language.typing.tile.Tile
```

Column-wise sum reduction.

Output shape is ``[1, N]`` for an ``[M, N]`` input.

Passing ``tmp_tile`` activates the binary-tree reduction path (O(log M) depth,
better precision); omitting it uses the sequential path.

Args:
    tile: Input tile
    tmp_tile: Optional scratch tile (same shape/dtype as input) that selects
        the binary-tree reduction path. Unlike the arg reductions, this is not
        enforced by type deduction -- pass the input's shape and dtype.

Returns:
    Tile wrapping the col_sum operation

## `col_max`

```text
pypto.language.tile.col_max(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Column-wise max reduction.

Output shape is ``[1, N]`` for an ``[M, N]`` input.

Args:
    tile: Input tile

Returns:
    Tile wrapping the col_max operation

## `col_min`

```text
pypto.language.tile.col_min(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Column-wise min reduction.

Output shape is ``[1, N]`` for an ``[M, N]`` input.

Args:
    tile: Input tile

Returns:
    Tile wrapping the col_min operation

## `col_prod`

```text
pypto.language.tile.col_prod(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Column-wise product reduction.

Output shape is ``[1, N]`` for an ``[M, N]`` input.

Args:
    tile: Input tile

Returns:
    Tile wrapping the col_prod operation

## `row_argmax`

```text
pypto.language.tile.row_argmax(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise argmax (column index of the per-row maximum, int32 output).

Output shape is ``[rows, 1]`` with INT32 index dtype.

Args:
    tile: Input tile
    tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``

Returns:
    Tile wrapping the row_argmax operation

## `row_argmin`

```text
pypto.language.tile.row_argmin(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise argmin (column index of the per-row minimum, int32 output).

Output shape is ``[rows, 1]`` with INT32 index dtype.

Args:
    tile: Input tile
    tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``

Returns:
    Tile wrapping the row_argmin operation

## `col_argmax`

```text
pypto.language.tile.col_argmax(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Column-wise argmax (row index of the per-column maximum, int32 output).

Output shape is ``[1, N]`` with INT32 index dtype. Unlike [`col_max`][pypto.language.tile.col_max], the
column argmax requires a ``tmp_tile`` scratch buffer.

Args:
    tile: Input tile
    tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``

Returns:
    Tile wrapping the col_argmax operation

## `col_argmin`

```text
pypto.language.tile.col_argmin(tile: pypto.language.typing.tile.Tile, tmp_tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Column-wise argmin (row index of the per-column minimum, int32 output).

Output shape is ``[1, N]`` with INT32 index dtype. Unlike [`col_min`][pypto.language.tile.col_min], the
column argmin requires a ``tmp_tile`` scratch buffer.

Args:
    tile: Input tile
    tmp_tile: Scratch tile with exactly the same shape and dtype as ``tile``

Returns:
    Tile wrapping the col_argmin operation

## `maximum`

```text
pypto.language.tile.maximum(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise maximum of two tiles.

Supports broadcasting between the two tiles.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile

Returns:
    Tile wrapping the maximum operation

## `row_expand`

```text
pypto.language.tile.row_expand(target: pypto.language.typing.tile.Tile, row_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand row vector to target shape.

Args:
    target: Target tile defining output shape [M, N]
    row_vec: Row vector to expand [M, 1]

Returns:
    Tile wrapping the row_expand operation

## `row_expand_sub`

```text
pypto.language.tile.row_expand_sub(tile: pypto.language.typing.tile.Tile, row_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise broadcast subtraction.

Subtracts a row vector from each row of the tile:
``tile[i, :] - row_vec[i, 0]`` for all ``i``.

Args:
    tile: Input tile [M, N]
    row_vec: Row vector [M, 1]

Returns:
    Tile wrapping the row_expand_sub operation

## `row_expand_div`

```text
pypto.language.tile.row_expand_div(tile: pypto.language.typing.tile.Tile, row_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise broadcast division.

Divides each row of the tile by the corresponding row vector value:
``tile[i, :] / row_vec[i, 0]`` for all ``i``.

Args:
    tile: Input tile [M, N]
    row_vec: Row vector [M, 1]

Returns:
    Tile wrapping the row_expand_div operation

## `row_expand_mul`

```text
pypto.language.tile.row_expand_mul(tile: pypto.language.typing.tile.Tile, row_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise broadcast multiplication.

Multiplies each row of the tile by the corresponding row vector value:
``tile[i, :] * row_vec[i, 0]`` for all ``i``.

Args:
    tile: Input tile [M, N]
    row_vec: Row vector [M, 1]

Returns:
    Tile wrapping the row_expand_mul operation

## `row_expand_add`

```text
pypto.language.tile.row_expand_add(tile: pypto.language.typing.tile.Tile, row_vec: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile | None = None) -> pypto.language.typing.tile.Tile
```

Row-wise scalar or packed-block expansion addition.

A non-row-major ``[M, 1]`` carrier broadcasts one scalar per row:
``tile[i, :] + row_vec[i, 0]`` for all ``i``. A row-major carrier instead holds
one 32-byte lane block per row and repeats that block across the destination
columns.

Args:
    tile: Input tile [M, N]
    row_vec: DN ``[M, 1]`` scalar carrier or row-major packed 32-byte carrier
    tmp: Optional PTOAS scratch tile

Returns:
    Tile wrapping the row_expand_add operation

## `row_expand_max`

```text
pypto.language.tile.row_expand_max(tile: pypto.language.typing.tile.Tile, row_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise broadcast maximum: max(tile, row_vec broadcasted).

Takes the element-wise maximum of each row and the row vector value:
``max(tile[i, :], row_vec[i, 0])`` for all ``i``.

Args:
    tile: Input tile [M, N]
    row_vec: Row vector [M, 1]

Returns:
    Tile wrapping the row_expand_max operation

## `row_expand_min`

```text
pypto.language.tile.row_expand_min(tile: pypto.language.typing.tile.Tile, row_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise broadcast minimum: min(tile, row_vec broadcasted).

Takes the element-wise minimum of each row and the row vector value:
``min(tile[i, :], row_vec[i, 0])`` for all ``i``.

Args:
    tile: Input tile [M, N]
    row_vec: Row vector [M, 1]

Returns:
    Tile wrapping the row_expand_min operation

## `row_expand_expdif`

```text
pypto.language.tile.row_expand_expdif(tile: pypto.language.typing.tile.Tile, row_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Row-wise exp-diff: exp(tile - row_vec) with per-row scalar.

Computes ``exp(tile[i, :] - row_vec[i, 0])`` for all ``i``.

Args:
    tile: Input tile [M, N]
    row_vec: Row vector providing per-row scalar [M, 1]

Returns:
    Tile wrapping the row_expand_expdif operation

## `col_expand`

```text
pypto.language.tile.col_expand(target: pypto.language.typing.tile.Tile, col_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand column vector to target shape.

Args:
    target: Target tile defining output shape [M, N]
    col_vec: Column vector to expand [1, N]

Returns:
    Tile wrapping the col_expand operation

## `col_expand_mul`

```text
pypto.language.tile.col_expand_mul(tile: pypto.language.typing.tile.Tile, col_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand column vector and multiply with tile.

Multiplies each column of the tile by the corresponding column vector value:
``tile[:, j] * col_vec[0, j]`` for all ``j``.

Args:
    tile: Input tile [M, N]
    col_vec: Column vector [1, N]

Returns:
    Tile wrapping the col_expand_mul operation

## `col_expand_div`

```text
pypto.language.tile.col_expand_div(tile: pypto.language.typing.tile.Tile, col_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand column vector and divide tile by it.

Divides each column of the tile by the corresponding column vector value:
``tile[:, j] / col_vec[0, j]`` for all ``j``.

Args:
    tile: Input tile [M, N]
    col_vec: Column vector [1, N]

Returns:
    Tile wrapping the col_expand_div operation

## `col_expand_sub`

```text
pypto.language.tile.col_expand_sub(tile: pypto.language.typing.tile.Tile, col_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand column vector and subtract from tile.

Subtracts a column vector from each column of the tile:
``tile[:, j] - col_vec[0, j]`` for all ``j``.

Args:
    tile: Input tile [M, N]
    col_vec: Column vector [1, N]

Returns:
    Tile wrapping the col_expand_sub operation

## `col_expand_add`

```text
pypto.language.tile.col_expand_add(tile: pypto.language.typing.tile.Tile, col_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand column vector and add to tile.

Adds a column vector to each column of the tile:
``tile[:, j] + col_vec[0, j]`` for all ``j``.

Args:
    tile: Input tile [M, N]
    col_vec: Column vector [1, N]

Returns:
    Tile wrapping the col_expand_add operation

## `col_expand_max`

```text
pypto.language.tile.col_expand_max(tile: pypto.language.typing.tile.Tile, col_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand column vector and take element-wise maximum with tile.

Computes ``max(tile[:, j], col_vec[0, j])`` for all ``j``.

Args:
    tile: Input tile [M, N]
    col_vec: Column vector [1, N]

Returns:
    Tile wrapping the col_expand_max operation

## `col_expand_min`

```text
pypto.language.tile.col_expand_min(tile: pypto.language.typing.tile.Tile, col_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand column vector and take element-wise minimum with tile.

Computes ``min(tile[:, j], col_vec[0, j])`` for all ``j``.

Args:
    tile: Input tile [M, N]
    col_vec: Column vector [1, N]

Returns:
    Tile wrapping the col_expand_min operation

## `col_expand_expdif`

```text
pypto.language.tile.col_expand_expdif(tile: pypto.language.typing.tile.Tile, col_vec: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Expand column vector and compute exp-diff with per-column scalar.

Computes ``exp(tile[:, j] - col_vec[0, j])`` for all ``j``.

Args:
    tile: Input tile [M, N]
    col_vec: Column vector providing per-column scalar [1, N]

Returns:
    Tile wrapping the col_expand_expdif operation

## `expands`

```text
pypto.language.tile.expands(target: pypto.language.typing.tile.Tile, scalar: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Expand scalar to target tile shape.

Broadcasts a scalar value to match the shape of the target tile.

Args:
    target: Target tile defining output shape
    scalar: Scalar value to expand

Returns:
    Tile wrapping the expands operation

## `minimum`

```text
pypto.language.tile.minimum(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise minimum of two tiles.

Supports broadcasting between the two tiles.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile

Returns:
    Tile wrapping the minimum operation

## `cmp`

```text
pypto.language.tile.cmp(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, cmp_type: int = 0) -> pypto.language.typing.tile.Tile
```

Element-wise comparison of two tiles.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile
    cmp_type: Comparison type (EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5)

Returns:
    Tile wrapping a packed predicate mask. Use tile.sel with an explicit tmp tile to materialize values.

## `cmps`

```text
pypto.language.tile.cmps(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, cmp_type: int = 0) -> pypto.language.typing.tile.Tile
```

Element-wise comparison of tile and scalar.

Args:
    lhs: Tile
    rhs: Scalar value
    cmp_type: Comparison type (EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5)

Returns:
    Tile wrapping a packed predicate mask. Use tile.sel with an explicit tmp tile to materialize values.

## `max`

```text
pypto.language.tile.max(lhs: pypto.language.typing.scalar.Scalar | int | pypto.pypto_core.ir.Expr, rhs: pypto.language.typing.scalar.Scalar | int | pypto.pypto_core.ir.Expr) -> pypto.language.typing.scalar.Scalar
```

Scalar max of two values.

Tile reductions are direction-specific — use [`row_max`][pypto.language.tile.row_max] (collapses the
last axis) or [`col_max`][pypto.language.tile.col_max] (collapses axis 0).

Args:
    lhs: First scalar operand
    rhs: Second scalar operand

Returns:
    Scalar wrapping the max operation

## `min`

```text
pypto.language.tile.min(lhs: pypto.language.typing.scalar.Scalar | int | pypto.pypto_core.ir.Expr, rhs: pypto.language.typing.scalar.Scalar | int | pypto.pypto_core.ir.Expr) -> pypto.language.typing.scalar.Scalar
```

Scalar min of two values.

Tile reductions are direction-specific — use [`row_min`][pypto.language.tile.row_min] (collapses the
last axis) or [`col_min`][pypto.language.tile.col_min] (collapses axis 0).

Args:
    lhs: First scalar operand
    rhs: Second scalar operand

Returns:
    Scalar wrapping the min operation

## `slice`

```text
pypto.language.tile.slice(tile: pypto.language.typing.tile.Tile, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, drop_dims: collections.abc.Sequence[int | pypto.pypto_core.ir.Expr] | None = None, pad_value: pypto.pypto_core.ir.PadValue | int | float | None = None) -> pypto.language.typing.tile.Tile
```

Create a slice of a tile with static shape and optional valid shape.

The slice is never valid where the source tile is not: the source's valid
region, shifted by ``offset`` and cut to the window, bounds the result.

Args:
    tile: Input tile
    shape: Static shape dimensions. Always full-rank — a scalar-indexed axis
        contributes a unit dim here and is listed in ``drop_dims``.
    offset: Offset dimensions for the slice
    valid_shape: Valid shape dimensions. When omitted, the source's validity
        under the window is used. Narrows the result; cannot widen it.
    drop_dims: Optional axes to erase from the result type (numpy-style rank
        reduction). Each listed axis must be a static unit dim of ``shape``
        and must still be fully valid after the intersection above.
        Because tiles are physically 2D, the result is clamped back to 2D
        if reduction would take it below 2D. ``None`` / ``[]`` drops nothing.
    pad_value: Optional padding mode for out-of-valid-shape elements.
        ``None`` means the source's padding mode carries through.
        Accepts ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or
        the literal sugars ``0``, ``math.inf``, ``-math.inf`` (same
        spelling as [`tile.fillpad`][pypto.language.tile.fillpad]). Only meaningful when the
        *effective* valid region is smaller than ``shape`` — which an explicit
        ``valid_shape`` or a partially-valid source tile can each bring about.

Returns:
    Tile wrapping the slice operation

Note:
    Unlike [`tensor.slice`][pypto.language.tensor.slice], there is no ``clamp``
    option: an on-chip window has nothing that could clamp it, so
    ``offset + shape`` must stay inside the source tile.

## `reshape`

```text
pypto.language.tile.reshape(tile: pypto.language.typing.tile.Tile, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr]) -> pypto.language.typing.tile.Tile
```

Reshape tile to new shape.

The valid region is carried through, never widened: the result holds real
data in exactly the cells the input did. See ``pl.reshape`` for the cases
that always map.

Args:
    tile: Input tile
    shape: New shape dimensions. A tile is physically 2D, so a higher-rank
        result is an intermediate that ``FlattenTileNdTo2D`` later collapses.

Returns:
    Tile wrapping the reshape operation

Raises:
    ValueError: If the element count changes, or if the input holds real
        data in only part of its buffer and no origin-anchored region of
        ``shape`` describes those same cells.

## `reinterpret_view`

```text
pypto.language.tile.reinterpret_view(data: pypto.language.typing.tile.Tile, dtype: pypto.pypto_core.DataType, *, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None) -> pypto.language.typing.tile.Tile
```

Reinterpret a tile over the same bytes with a different dtype.

Args:
    data: Input tile.
    dtype: Target element dtype, which must differ from the source dtype.
    shape: Optional byte-equivalent target shape. When omitted, the
        physically contiguous dimension is scaled according to the
        source/target dtype byte ratio.

Returns:
    Tile wrapping the zero-copy reinterpret-view operation.

## `transpose`

```text
pypto.language.tile.transpose(tile: pypto.language.typing.tile.Tile, axis1: int, axis2: int, tmp_tile: pypto.language.typing.tile.Tile | None = None) -> pypto.language.typing.tile.Tile
```

Transpose tile by swapping two axes.

The ``pto.ttrans`` scratch buffer is a codegen detail allocated later by
``FlattenTileNdTo2D``, so user code never supplies ``tmp_tile``. The optional
parameter exists only so the lowered 4-arg form round-trips through the parser.

Args:
    tile: Input tile.
    axis1: First axis to swap (supports negative indexing).
    axis2: Second axis to swap (supports negative indexing).
    tmp_tile: Optional scratch tile — compiler-generated lowered IR only.

Returns:
    Tile wrapping the transpose operation.

## `transpose_view`

```text
pypto.language.tile.transpose_view(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Zero-copy fractal-layout reinterpretation (NZ<->ZN) of a tile.

Swaps the trailing two dims together with the block/scatter layouts, aliasing
the source buffer byte-for-byte: an NZ ``[..., N, K]`` tile and a ZN
``[..., K, N]`` tile over the same L1 bytes are mutual transposes. Emits no
data movement, so one GM->L1 load can feed both a ``b_trans=True`` and a
``b_trans=False`` matmul on a shared operand.

Args:
    tile: Input tile (TileType, >=2D; typically Mat-resident).

Returns:
    Tile wrapping the transposed-layout view.

## `set_validshape`

```text
pypto.language.tile.set_validshape(tile: pypto.language.typing.tile.Tile, valid_rows: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, valid_cols: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tile.Tile
```

Update valid-shape metadata of a tile without data movement.

.. note::
    The operand must not be a view (a ``pl.tile.slice`` or reshape result): a
    view carries its valid extent in its type, so there is nothing to update.
    Narrow at the slice with ``valid_shape=`` instead.

Args:
    tile: Input tile (must be 2D)
    valid_rows: Number of valid rows (int or Scalar[INDEX])
    valid_cols: Number of valid columns (int or Scalar[INDEX])

Returns:
    Tile with updated valid_shape metadata

## `rem`

```text
pypto.language.tile.rem(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, high_precision: bool = False) -> pypto.language.typing.tile.Tile
```

Element-wise remainder (modulo) of two tiles.

Computes lhs % rhs element-wise. Maps to the TREM hardware intrinsic.
On A2/A3, every INT32 input element must be in ``[-2**24, 2**24]``.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile
    tmp: Same-dtype scratch tile whose physical and valid capacity
        provably provides two rows and covers the dividend columns on
        A2/A3. It must not overlap either source there.
    high_precision: Whether to select PTOAS's high-precision TREM mode.
        This mode is defined only for FP32 and is ignored by A2/A3 hardware.

Returns:
    Tile wrapping the rem operation

## `rems`

```text
pypto.language.tile.rems(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise remainder (modulo) of tile and scalar.

Computes lhs % rhs element-wise. Maps to the TREMS hardware intrinsic.
On A2/A3, source valid extents must be provably positive and every INT32
source element and scalar must be in ``[-2**24, 2**24]``.

Args:
    lhs: Tile
    rhs: Scalar value
    tmp: Same-dtype scratch tile whose physical and valid capacity
        provably provides one row and covers the dividend columns on
        A2/A3. It must not overlap the dividend there.

Returns:
    Tile wrapping the rems operation

## `part_add`

```text
pypto.language.tile.part_add(src0: pypto.language.typing.tile.Tile, src1: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Partial element-wise add of two tiles.

Adds over the destination valid region; where only one source is valid the
result copies that source. Maps to the TPARTADD hardware intrinsic.

Args:
    src0: First source tile
    src1: Second source tile

Returns:
    Tile wrapping the part_add operation

## `part_mul`

```text
pypto.language.tile.part_mul(src0: pypto.language.typing.tile.Tile, src1: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Partial element-wise multiply of two tiles.

Multiplies over the destination valid region; where only one source is valid
the result copies that source. Maps to the TPARTMUL hardware intrinsic.

Args:
    src0: First source tile
    src1: Second source tile

Returns:
    Tile wrapping the part_mul operation

## `part_max`

```text
pypto.language.tile.part_max(src0: pypto.language.typing.tile.Tile, src1: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Partial element-wise max of two tiles.

Takes the max over the destination valid region; where only one source is
valid the result copies that source. Maps to the TPARTMAX hardware intrinsic.

Args:
    src0: First source tile
    src1: Second source tile

Returns:
    Tile wrapping the part_max operation

## `part_min`

```text
pypto.language.tile.part_min(src0: pypto.language.typing.tile.Tile, src1: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Partial element-wise min of two tiles.

Takes the min over the destination valid region; where only one source is
valid the result copies that source. Maps to the TPARTMIN hardware intrinsic.

Args:
    src0: First source tile
    src1: Second source tile

Returns:
    Tile wrapping the part_min operation

## `fmod`

```text
pypto.language.tile.fmod(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, high_precision: bool = False) -> pypto.language.typing.tile.Tile
```

Element-wise truncating remainder of two tiles.

Computes the truncating remainder of lhs / rhs element-wise, matching
``torch.fmod`` with the result taking the dividend sign. Maps to TFMOD.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile
    high_precision: Whether to select PTOAS's high-precision TFMOD mode.
        This mode is defined only for FP32.

Returns:
    Tile wrapping the fmod operation

## `fmods`

```text
pypto.language.tile.fmods(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise truncating remainder of tile and scalar.

Computes the truncating remainder of lhs / rhs element-wise, matching
``torch.fmod`` with the result taking the dividend sign. Maps to TFMODS.
A2/A3 requires every source valid extent to be provably positive.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the fmods operation

## `and_`

```text
pypto.language.tile.and_(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise AND of two tiles.

Computes lhs & rhs element-wise. Maps to the TAND hardware intrinsic.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile

Returns:
    Tile wrapping the and operation

## `ands`

```text
pypto.language.tile.ands(lhs: pypto.language.typing.tile.Tile, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise AND of tile and scalar.

Computes lhs & rhs element-wise. Maps to the TANDS hardware intrinsic.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the ands operation

## `or_`

```text
pypto.language.tile.or_(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise OR of two tiles.

Computes lhs | rhs element-wise. Maps to the TOR hardware intrinsic.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile

Returns:
    Tile wrapping the or operation

## `ors`

```text
pypto.language.tile.ors(lhs: pypto.language.typing.tile.Tile, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise OR of tile and scalar.

Computes lhs | rhs element-wise. Maps to the TORS hardware intrinsic.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the ors operation

## `xor`

```text
pypto.language.tile.xor(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise XOR of two tiles.

Computes lhs ^ rhs element-wise. Maps to the TXOR hardware intrinsic.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile
    tmp: Temporary tile required by the hardware

Returns:
    Tile wrapping the xor operation

## `xors`

```text
pypto.language.tile.xors(lhs: pypto.language.typing.tile.Tile, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise XOR of tile and scalar.

Computes lhs ^ rhs element-wise. Maps to the TXORS hardware intrinsic.

Args:
    lhs: Tile
    rhs: Scalar value
    tmp: Temporary tile required by the hardware

Returns:
    Tile wrapping the xors operation

## `shl`

```text
pypto.language.tile.shl(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise left shift of two tiles.

Computes lhs << rhs element-wise. Maps to the TSHL hardware intrinsic.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile

Returns:
    Tile wrapping the shl operation

## `shls`

```text
pypto.language.tile.shls(lhs: pypto.language.typing.tile.Tile, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise left shift of tile and scalar.

Computes lhs << rhs element-wise. Maps to the TSHLS hardware intrinsic.

Note:
    The scalar shift amount must be zero or positive; negative values are
    not supported by the hardware and will be rejected by codegen.

Args:
    lhs: Tile
    rhs: Scalar shift amount; must be >= 0

Returns:
    Tile wrapping the shls operation

## `shr`

```text
pypto.language.tile.shr(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise right shift of two tiles.

Computes lhs >> rhs element-wise. Maps to the TSHR hardware intrinsic.

Args:
    lhs: Left-hand side tile
    rhs: Right-hand side tile

Returns:
    Tile wrapping the shr operation

## `shrs`

```text
pypto.language.tile.shrs(lhs: pypto.language.typing.tile.Tile, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise right shift of tile and scalar.

Computes lhs >> rhs element-wise. Maps to the TSHRS hardware intrinsic.

Note:
    The scalar shift amount must be zero or positive; negative values are
    not supported by the hardware and will be rejected by codegen.

Args:
    lhs: Tile
    rhs: Scalar shift amount; must be >= 0

Returns:
    Tile wrapping the shrs operation

## `maximums`

```text
pypto.language.tile.maximums(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise maximum of tile and scalar.

Computes max(lhs, rhs) element-wise. Maps to the TMAXS hardware intrinsic.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the maximums operation

## `minimums`

```text
pypto.language.tile.minimums(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise minimum of tile and scalar.

Computes min(lhs, rhs) element-wise. Maps to the TMINS hardware intrinsic.

Args:
    lhs: Tile
    rhs: Scalar value

Returns:
    Tile wrapping the minimums operation

## `prelu`

```text
pypto.language.tile.prelu(tile: pypto.language.typing.tile.Tile, slope: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise parametric ReLU of a tile.

Computes prelu(tile, slope) element-wise. Maps to the TPRELU hardware intrinsic.

Args:
    tile: Input tile
    slope: Slope tile used for negative values
    tmp: Temporary tile required by the hardware

Returns:
    Tile wrapping the prelu operation

## `not_`

```text
pypto.language.tile.not_(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise bitwise NOT of a tile.

Computes ~tile element-wise. Maps to the TNOT hardware intrinsic.

Args:
    tile: Input tile

Returns:
    Tile wrapping the not operation

## `addc`

```text
pypto.language.tile.addc(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise carry addition of three tiles.

Computes ``src0 + src1 + carry`` element-wise. Maps to TADDC.

Args:
    lhs: First source tile
    rhs: Second source tile
    rhs2: Per-element carry-in tile, normally containing 0 or 1

Returns:
    Tile wrapping the addc operation

## `subc`

```text
pypto.language.tile.subc(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise carry subtraction of three tiles.

Computes ``src0 - src1 + carry`` element-wise. Maps to TSUBC.

Args:
    lhs: Minuend tile
    rhs: Subtrahend tile
    rhs2: Per-element carry-in tile, normally containing 0 or 1

Returns:
    Tile wrapping the subc operation

## `addsc`

```text
pypto.language.tile.addsc(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise scalar carry addition.

Computes ``src0 + scalar + carry`` element-wise. Maps to TADDSC.

Args:
    lhs: Source tile
    rhs: Scalar addend with the same dtype as ``lhs``
    rhs2: Per-element carry-in tile, normally containing 0 or 1

Returns:
    Tile wrapping the addsc operation

## `subsc`

```text
pypto.language.tile.subsc(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise scalar carry subtraction.

Computes ``src0 - scalar + carry`` element-wise. Maps to TSUBSC.

Args:
    lhs: Minuend tile
    rhs: Scalar subtrahend with the same dtype as ``lhs``
    rhs2: Per-element carry-in tile, normally containing 0 or 1

Returns:
    Tile wrapping the subsc operation

## `lrelu`

```text
pypto.language.tile.lrelu(tile: pypto.language.typing.tile.Tile, slope: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Element-wise leaky ReLU with scalar slope.

Computes max(tile, slope * tile) element-wise. Maps to the TLRELU hardware intrinsic.

Args:
    tile: Input tile
    slope: Scalar slope for negative values

Returns:
    Tile wrapping the lrelu operation

## `sel`

```text
pypto.language.tile.sel(mask: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Per-element selection between two tiles using a predicate mask tile.

For each element (i, j): dst[i,j] = lhs[i,j] if mask[i,j] is true, else rhs[i,j].
Maps to the TSEL hardware intrinsic. The mask encoding is target-defined.

Args:
    mask: Predicate mask tile; encoding is target-defined
    lhs: Source tile 0, selected where mask is true
    rhs: Source tile 1, selected where mask is false
    tmp: Scratch tile required by TSEL (UINT32 [1, 16] on A2/A3; unread ABI placeholder on A5)

Returns:
    Tile wrapping the sel operation

## `sels`

```text
pypto.language.tile.sels(mask: pypto.language.typing.tile.Tile, src: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, scalar: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
```

Per-element selection between a source tile and a scalar.

For each element (i, j): dst[i,j] = src[i,j] if mask[i,j] is true,
else scalar. Maps to the TSELS hardware intrinsic.

Args:
    mask: Predicate mask tile; encoding is target-defined
    src: Source tile, selected where mask is true
    tmp: Scratch tile required by TSELS
    scalar: Scalar value, selected where mask is false

Returns:
    Tile wrapping the sels operation

## `tpush_to_aiv`

```text
pypto.language.tile.tpush_to_aiv(tile: pypto.language.typing.tile.Tile, *, split: int, lane_stride: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
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
pypto.language.tile.tpush_to_aic(tile: pypto.language.typing.tile.Tile, *, split: int, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
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
pypto.language.tile.tpop_from_aic(*, shape: list[int] | None = None, dtype: pypto.pypto_core.DataType | None = None, split: int = 0, lane_stride: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.tile.Tile
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
pypto.language.tile.tpop_from_aiv(*, shape: list[int] | None = None, dtype: pypto.pypto_core.DataType | None = None, split: int = 0, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.tile.Tile
```

Pop tile data from AIV cross-core pipe into AIC.

Args:
    shape: Shape of the tile to receive
    dtype: Data type of the tile to receive
    split: Split mode (0=none, 1=up-down, 2=left-right)
    id: Optional frontend pipe id. Omit to use PTOAS default id 0.
    span: Optional source span

## `sort32`

```text
pypto.language.tile.sort32(src: pypto.language.typing.tile.Tile, idx: pypto.language.typing.tile.Tile, *, tmp: pypto.language.typing.tile.Tile | None = None) -> pypto.language.typing.tile.Tile
```

Sort fixed 32-element blocks with explicit index tile.

Sorts 32-element blocks in src, permuting idx alongside.
Returns an 8-byte value-index-pair tile. Its last dimension is 2x the input
width for FP32 and 4x the input width for FP16.

For FP16 src: initialize idx with [0, 1, 2, ..., 31] per block.
For FP32 src: initialize idx with [0, 2, 4, ..., 62] per block.

Args:
    src: Input value tile (FP16 or FP32)
    idx: Input index tile with sequential offsets
    tmp: Optional A2/A3 PTOAS scratch tile. Normally compiler-generated.

Returns:
    Tile wrapping the dtype-dependent expanded sort32 output

## `gather`

```text
pypto.language.tile.gather(src: pypto.language.typing.tile.Tile, indices: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Gather elements from src tile by per-element indices (index form).

Computes ``dst[i, j] = src[indices[i, j]]``. Maps to PTOAS ``pto.tgather``
index form. For the hardware mask-pattern variant, use [`gather_mask`][pypto.language.tile.gather_mask].

Args:
    src: Source tile (FP16, FP32, INT16, or INT32)
    indices: Index tile (INT32 with any src, or INT16 with a 16-bit src — FP16/INT16);
        selects which elements of ``src`` to gather
    tmp: Temporary workspace tile (any Vec dtype; required as an operand but
        not constrained by the A5 index form — A2/A3 narrows this at PTOAS)

Returns:
    Tile with gathered elements (same dtype as ``src``)

## `gatherb`

```text
pypto.language.tile.gatherb(src: pypto.language.typing.tile.Tile, offset: pypto.language.typing.tile.Tile, *, output_dtype: int | pypto.pypto_core.DataType | None = None) -> pypto.language.typing.tile.Tile
```

Gather 32-byte blocks from ``src`` by UINT32 byte offsets.

Each offset selects one 32-byte source block. One offset column expands to
``32 / sizeof(output_dtype)`` output elements. ``output_dtype`` defaults to
``src.dtype`` and may select another supported byte interpretation.
A sliced source must have a byte address that PyPTO can prove is 32-byte
aligned; dynamic column offsets are rejected conservatively.

Args:
    src: Source tile to gather blocks from.
    offset: UINT32 tile of **byte** offsets into ``src`` -- not element indices.
    output_dtype: Byte interpretation of the result. Defaults to ``src.dtype``.

Returns:
    Tile wrapping the gatherb operation.

## `gather_mask`

```text
pypto.language.tile.gather_mask(src: pypto.language.typing.tile.Tile, mask_pattern: int, *, output_dtype: int | pypto.pypto_core.DataType | None = None) -> pypto.language.typing.tile.Tile
```

Gather elements from src tile by a fixed hardware mask pattern (mask form).

Selects elements according to a stride/mask pattern baked into the hardware.
For the per-element indices variant, use [`gather`][pypto.language.tile.gather].

Args:
    src: Source tile (FP16, FP32, INT16, or INT32)
    mask_pattern: Mask pattern selector (1-7), see [`MaskPattern`][pypto.language.tile.MaskPattern].
        1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111
    output_dtype: Optional output dtype. When provided, the result tile has
        this dtype instead of ``src``'s dtype (bit reinterpretation, no
        conversion). Hardware requires ``sizeof(dst_dtype) == sizeof(src_dtype)``.
        Example: ``output_dtype=pl.UINT32`` to extract sort32 index bits from
        FP32 memory.

Returns:
    Tile with mask-selected elements

Examples:
    # Same dtype
    out = gather_mask(src, mask_pattern=pl.tile.MaskPattern.P0101)

    # Cross-type output (FP32 bits → UINT32)
    out = gather_mask(src, pl.tile.MaskPattern.P1010, output_dtype=pl.UINT32)

## `gather_compare`

```text
pypto.language.tile.gather_compare(src: pypto.language.typing.tile.Tile, kvalue: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, tmp: pypto.language.typing.tile.Tile, *, cmp_mode: str | int = 'eq', offset: int = 0, out_cols: int, count_dtype: int | pypto.pypto_core.DataType | None = None) -> tuple[pypto.language.typing.tile.Tile, pypto.language.typing.tile.Tile]
```

Compare-form gather (tile-level): produce (dst, cdst) — gathered indices
and per-row match counts.

Maps to PTOAS ``pto.tgather`` compare-form. Hardware DPS allocation of
dst/cdst is handled downstream — only the three inputs (src, kvalue, tmp)
appear at this surface.

DSL form (inside ``@pl.function``)::

    dst, cdst = pl.tile.gather_compare(src, kvalue, tmp,
                                        cmp_mode="eq", offset=0,
                                        out_cols=K)

The ``a, b = call(...)`` Python tuple unpack is desugared by the parser
into ``_tuple = call; a = _tuple[0]; b = _tuple[1]``. The parser
consumes the underlying tuple-typed ``ir.Call`` returned by
``pypto.ir.op.tile_ops.gather_compare``; the ``(Tile, Tile)`` split
below only runs in interactive Python contexts.

Args:
    src: Source tile (FP16/FP32/INT16/INT32, 2D).
    kvalue: Scalar threshold (dtype must match ``src``; applied to every row).
    tmp: Workspace tile (UINT8) sized for the codegen kernel.
    cmp_mode: ``"eq"`` / ``"ne"`` / ``"lt"`` / ``"le"`` / ``"gt"`` /
        ``"ge"`` or int ``0..5``. Defaults to ``"eq"``.
    offset: Starting index offset (default 0).
    out_cols: Output column count per row for ``dst`` (positive int, required).
    count_dtype: Per-row count dtype, INT32 or UINT32; defaults to INT32.

Returns:
    ``(dst, cdst)`` where ``dst`` is a Tile ``[rows, out_cols]`` of INT32
    gathered indices and ``cdst`` is a Tile ``[1, rows]`` of ``count_dtype``
    per-row match counts.

## `scatter`

```text
pypto.language.tile.scatter(dst: pypto.language.typing.tile.Tile, src: pypto.language.typing.tile.Tile, indexes: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Scatter elements of ``src`` into ``dst`` at per-element flattened indices.

Computes ``dst.flat[indexes[i, j]] = src[i, j]``, i.e. ``indexes`` carries the
*flattened* destination offset for each ``src`` element and therefore has the
**same [rows, cols] shape as** ``src``. Maps to PTOAS ``pto.tscatter`` index
form. The op is DPS — ``dst`` is the first (in/out) argument, rewritten in
place, and the returned Tile aliases the same buffer. For the hardware
mask-pattern variant, use [`scatter_mask`][pypto.language.tile.scatter_mask].

Args:
    dst: Destination tile (same dtype as ``src``; rewritten in-place).
        Flat-addressed, so its column count is independent of ``src``.
    src: Source tile (FP16/FP32/BF16/INT8/INT16/INT32, 2D)
    indexes: Per-element flattened destination index tile (INT16 or INT32;
        same shape as ``src``). The element width must match ``dst``: 4-byte
        dst → INT32, 2-byte dst → INT16, 1-byte dst → INT16.

Returns:
    Tile aliasing the post-scatter ``dst`` tile.

## `scatter_mask`

```text
pypto.language.tile.scatter_mask(dst: pypto.language.typing.tile.Tile, src: pypto.language.typing.tile.Tile, mask_pattern: int) -> pypto.language.typing.tile.Tile
```

Scatter ``src`` rows into mask-marked columns of ``dst`` (mask form).

For each row, the elements of ``src`` are written into the columns of
``dst`` selected by ``mask_pattern`` (the inverse of [`gather_mask`][pypto.language.tile.gather_mask]).

Unlike [`gather_mask`][pypto.language.tile.gather_mask] (a real ``pto.tgather`` ISA op on A2/A3 and A5),
mask-pattern scatter is not a distinct pto-isa instruction — PyPTO emits it
as a ``pto.tscatter`` mask-form construct for A2/A3 / CPU-sim style lowering
paths.

Args:
    dst: Destination tile (rewritten on positions selected by ``mask_pattern``)
    src: Source tile (compact rows; same dtype as ``dst``)
    mask_pattern: Mask pattern selector (1-7), see [`MaskPattern`][pypto.language.tile.MaskPattern].
        1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111

Returns:
    Tile aliasing the post-scatter ``dst`` tile.

Examples:
    out = scatter_mask(dst, src, mask_pattern=pl.tile.MaskPattern.P0101)

## `mscatter`

```text
pypto.language.tile.mscatter(src: pypto.language.typing.tile.Tile, idx: pypto.language.typing.tile.Tile, output_tensor: ~_TensorT) -> ~_TensorT
```

Scatter-store tile elements into a tensor at per-element indices.

Semantics: ``output_tensor[idx[i, j]] = src[i, j]``

Maps to the PTOAS ``pto.mscatter`` instruction.

Args:
    src: Source tile (FP16, FP32, INT16, or INT32)
    idx: Index tile (INT32, same rank as src)
    output_tensor: Output tensor to scatter into (same dtype as src)

Returns:
    Tensor wrapping the mscatter operation

Example:
    >>> result = pl.tile.mscatter(src_tile, idx_tile, out_tensor)

## `mgather`

```text
pypto.language.tile.mgather(mem: pypto.language.typing.tensor.Tensor, idx: pypto.language.typing.tile.Tile | pypto.language.typing.tensor.Tensor, coalesce: str | int = 'row', *, gather_oob: str | int = 'undefined', target_memory: pypto.pypto_core.ir.MemorySpace = MemorySpace.Vec, scratch: pypto.language.typing.tensor.Tensor | None = None, valid_shape: collections.abc.Sequence[int] | None = None) -> pypto.language.typing.tile.Tile
```

Gather-load rows or elements from a GM tensor into a fresh Vec or Mat tile.

Vec output uses a 2D INT32 index tile. Mat output uses a GM INT32 index tensor
and produces canonical NZ layout; its element mode additionally requires a
same-dtype GM scratch tensor.

Args:
    mem: Source tensor in GM.
    idx: Two-dimensional INT32 index tile for Vec output, or GM tensor for
        Mat output.
    coalesce: ``"row"``/``0`` for row gather or ``"elem"``/``1`` for flat
        element gather. Integer values support printed-IR round trips.
    gather_oob: Out-of-bounds handling: ``"undefined"``, ``"clamp"``,
        ``"wrap"``, ``"zero"``, or the corresponding integer ``0..3``.
    target_memory: ``MemorySpace.Vec`` (default) or ``MemorySpace.Mat``.
        This selects the operator *variant*, not merely a placement: the two
        take a different ``idx`` type and produce a different output shape
        and view, so it cannot be left for the compiler to infer.
    scratch: Same-dtype GM workspace required by Mat element gather and
        forbidden by the other forms.
    valid_shape: Optional two-dimensional written region for Mat output.
        Vec output derives its valid region from the index tile.

## `MaskPattern`

```text
pypto.language.tile.MaskPattern()
```

Hardware mask pattern selectors for tile.gather_mask.

Bit patterns are read right-to-left; lower bits correspond to lower indices.

## `mrgsort`

```text
pypto.language.tile.mrgsort(src0: pypto.language.typing.tile.Tile, src1: pypto.language.typing.tile.Tile | None = None, src2: pypto.language.typing.tile.Tile | None = None, src3: pypto.language.typing.tile.Tile | None = None, tmp: pypto.language.typing.tile.Tile | None = None, exhausted: bool = False, *, block_len: int | pypto.language.typing.scalar.Scalar | None = None) -> pypto.language.typing.tile.Tile
```

Merge sort — format1 (single-list) or format2 (2-4 way merge).

Format1: sorts a tile containing multiple pre-sorted runs of length block_len.
Format2: merges 2, 3, or 4 pre-sorted input tiles into one sorted output.

Format1 usage (keyword block_len):
    out = mrgsort(src, block_len=64)

Format2 2-way usage (keyword tmp):
    out = mrgsort(src0, src1, tmp=tmp_tile)
    out = mrgsort(src0, src1, tmp=tmp_tile, exhausted=True)

Format2 3-way usage:
    out = mrgsort(src0, src1, src2, tmp=tmp_tile)

Format2 4-way usage (5 positional args):
    out = mrgsort(src0, src1, src2, src3, tmp)
    out = mrgsort(src0, src1, src2, src3, tmp, exhausted=True)

Args:
    src0: For format1: input tile with pre-sorted runs (FP16 or FP32).
          For format2: first sorted input tile.
    src1: (format2) Second sorted input tile.
    src2: (format2, optional) Third sorted input tile (3-way or 4-way).
    src3: (format2, optional) Fourth sorted input tile (4-way only).
    tmp: (format2) Temporary workspace tile (same shape as output).
          Pass as keyword arg for 2-way and 3-way.
    exhausted: (format2) If True, marks inputs as exhausted (default: False).
    block_len: (format1, keyword-only) Run length, must be multiple of 64.

Returns:
    Tile with merged sorted elements
