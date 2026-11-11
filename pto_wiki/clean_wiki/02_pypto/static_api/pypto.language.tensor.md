# `pypto.language.tensor` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `create_tensor`

```text
pypto.language.tensor.create_tensor(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, layout: pypto.pypto_core.ir.TensorLayout = TensorLayout.ND, manual_dep: bool = False, init_value: int | float | None = None) -> pypto.language.typing.tensor.Tensor
```

Create a new tensor with specified shape and dtype.

Args:
    shape: List of dimension sizes (int or Expr)
    dtype: Data type of tensor elements
    layout: Tensor layout (default: ND)
    init_value: **Removed.** Passing anything but ``None`` raises
        ``ValueError``. The runtime dropped its create-info fill, so a
        runtime-allocated buffer can no longer be pre-filled from
        orchestration. Seed it with a kernel — ``pl.full`` materializes a
        constant tensor, or write the buffer from your own kernel — and
        order every reader after that kernel with an explicit dependency.
    manual_dep: Opt this tensor out of OverlapMap auto-dep tracking for
        its **entire lifetime**. When True, every task that reads or
        writes this tensor skips OverlapMap lookup and insert, so the
        runtime neither makes the task wait on prior writers nor
        registers it as a producer for later readers. Creator retention
        (the original ``tensor.create``'s owner_task_id) still applies.

        This is the **tensor-lifetime** granularity of opting out of
        auto-dep tracking. The other two granularities, both orthogonal
        to this one, are:

          * ``with pl.manual_scope():`` — scope-wide opt-out.
          * ``pl.no_dep(t)`` at a kernel-call arg position — single-task
            opt-out for one arg.

        All three opt-outs compose with the orthogonal **explicit edges**
        mechanism (``pl.submit(..., deps=[...])`` / ``pl.at(..., deps=)``):
        the final task fanin is *auto-tracked deps* ∪ *explicit deps*,
        so you can use ``manual_dep=True`` to silence auto-tracking on
        a scratch buffer while still pinning its ordering with a
        ``deps=[tid]`` on the consumer.

        Internally also used by ``InjectGMPipeBuffer`` to mark the
        ring-buffer slots it synthesises.

Returns:
    Tensor wrapping the create operation

## `create`

```text
pypto.language.tensor.create(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, layout: pypto.pypto_core.ir.TensorLayout = TensorLayout.ND, manual_dep: bool = False, init_value: int | float | None = None) -> pypto.language.typing.tensor.Tensor
```

Create a new tensor with specified shape and dtype.

Args:
    shape: List of dimension sizes (int or Expr)
    dtype: Data type of tensor elements
    layout: Tensor layout (default: ND)
    init_value: **Removed.** Passing anything but ``None`` raises
        ``ValueError``. The runtime dropped its create-info fill, so a
        runtime-allocated buffer can no longer be pre-filled from
        orchestration. Seed it with a kernel — ``pl.full`` materializes a
        constant tensor, or write the buffer from your own kernel — and
        order every reader after that kernel with an explicit dependency.
    manual_dep: Opt this tensor out of OverlapMap auto-dep tracking for
        its **entire lifetime**. When True, every task that reads or
        writes this tensor skips OverlapMap lookup and insert, so the
        runtime neither makes the task wait on prior writers nor
        registers it as a producer for later readers. Creator retention
        (the original ``tensor.create``'s owner_task_id) still applies.

        This is the **tensor-lifetime** granularity of opting out of
        auto-dep tracking. The other two granularities, both orthogonal
        to this one, are:

          * ``with pl.manual_scope():`` — scope-wide opt-out.
          * ``pl.no_dep(t)`` at a kernel-call arg position — single-task
            opt-out for one arg.

        All three opt-outs compose with the orthogonal **explicit edges**
        mechanism (``pl.submit(..., deps=[...])`` / ``pl.at(..., deps=)``):
        the final task fanin is *auto-tracked deps* ∪ *explicit deps*,
        so you can use ``manual_dep=True`` to silence auto-tracking on
        a scratch buffer while still pinning its ordering with a
        ``deps=[tid]`` on the consumer.

        Internally also used by ``InjectGMPipeBuffer`` to mark the
        ring-buffer slots it synthesises.

Returns:
    Tensor wrapping the create operation

## `no_dep`

```text
pypto.language.tensor.no_dep(tensor: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Mark a kernel-call argument as no-dependency (caller-site override).

This is a parser-recognized marker — at runtime it returns the wrapped
tensor unchanged. The parser detects ``pl.no_dep(t)`` at kernel call
arg positions and threads the per-arg ``ArgDirection.NoDep`` override
into the IR Call's attrs. ``DeriveCallDirections`` then overwrites the
auto-derived direction at that slot to NoDep.

Effect at runtime: the simpler runtime skips both the OverlapMap
dependency lookup *and* the producer insert for this argument. The
marker is legal regardless of whether the callee declares the param
as ``In`` (read) or ``Out`` / ``InOut`` (write): the caller is
asserting out-of-band that there is no RaW / WaW / WaR conflict on
the slot — for example, paged-attention writes whose target offset
is data-dependent (so the compiler cannot prove disjointness) but
are guaranteed disjoint by the runtime allocation protocol.

Only valid as a direct argument to a kernel call::

    # Read-side override: shared_input is read-only at the callee, but
    # auto-dep would otherwise serialise sibling fan-out reads.
    result = self.my_kernel(pl.no_dep(shared_input), output)

    # Write-side override: the callee writes into k_cache via
    # ``pl.assemble`` at a data-dependent offset; the user knows the
    # offsets are disjoint across the parallel fan-out.
    self.rope_kv_cache(q_proj, pl.no_dep(k_cache), pl.no_dep(v_cache))

Outside a kernel call argument list it is a no-op (returns ``tensor``
unchanged); the parser only injects the override at recognized call
sites.

Args:
    tensor: The tensor to pass through. Must be a ``Tensor`` value.

Returns:
    The tensor unchanged. The marker is consumed at parse time.

## `dump_tag`

```text
pypto.language.tensor.dump_tag(tensor: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Mark a tensor for selective dump within the enclosing orchestration.

Declarative per-tensor dump marker (simpler#844 selective tensor dump).
Writing ``pl.dump_tag(q)`` as a standalone statement records ``q`` so that
every *subsequent* kernel dispatch consuming that exact value gets ``q``
merged into the dispatch's ``dump_vars`` — whether that dispatch lowers to
a plain ``ir.Call`` (the typical ``@pl.jit`` / tensor-op path) or an
``ir.Submit``. The runtime then marks those ``Arg`` slots via
``Arg::dump(...)``.

This is the declarative counterpart to the explicit ``dumps=[...]`` kwarg
on ``pl.submit(...)`` / ``pl.at(...)`` — both feed the same ``dump_vars``
set (mirroring how a scope's auto-inferred and explicit ``deps=`` edges
both feed ``manual_dep_edges``). Use ``dumps=`` when you want to list the
targets explicitly at a single task launch; use ``pl.dump_tag`` when one
declaration should stick across every subsequent consumer.

Use this to keep tensor dump viable on large workloads (e.g.
paged-attention 64bat/8192ctx) where full dump (``enable_dump_args=2``)
saturates the host-side dump collector (~42 MB/s drain rate) by dumping
every binding, eventually triggering a STARS op-timeout kill on the AICPU
side. Run partial dump (``enable_dump_args=1``) and tag only the tensors
of interest, so the runtime filters out large bindings (1 GB kv-cache,
output buffers, etc.) from the collector queue.

Semantics:

* Forward-sticky over the orch scope — one ``pl.dump_tag(q)`` statement
  affects all *subsequent* kernel calls in the same orch that consume the
  tagged value. Tracked by **Var identity**, never by name: reassigning
  ``q`` (e.g. ``q = self.foo(q)``) produces a new value that the prior
  tag does **not** cover — re-tag it if needed.
* Only effective under partial dump (``RunConfig.enable_dump_args == 1``)
  — selective dump filters within the partial pipeline. A no-op when dump
  is off (``0``); under full dump (``2``) every binding is captured, so the
  tag has nothing to narrow.
* Consumed at parse time — recorded into the consuming dispatch's
  ``dump_vars`` and emits no IR statement of its own.

Valid as a standalone statement inside an Orchestration function or an
Inline helper (``@pl.jit.inline`` / ``FunctionType.Inline``) that the
orchestration inlines. The parser rejects ``pl.dump_tag`` written in any
other function type (AIV / AIC / Mix kernel bodies) with a clear error::

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, q: pl.Tensor[...], k_cache: pl.Tensor[...], out: pl.Out[...]):
        pl.dump_tag(q)           # mark q for selective dump
        pl.dump_tag(out)         # mark out for selective dump
        s = self.qk_matmul(q, k_cache, scratch)  # q is dumped here
        out = self.pv_matmul(s, k_cache, out)    # out is dumped here

For a single task launch, list the targets explicitly with the
``dumps=[...]`` kwarg: ``out, tid = pl.submit(self.qk_matmul, q, k_cache,
scratch, dumps=[q])`` or ``with pl.at(..., dumps=[q]) as tid:``.

Inside an Inline helper, the recorded ``dump_vars`` ride on the inline
body's kernel calls; the ``InlineFunctions`` pass splices those calls into
the caller and the mutator substitutes the caller's arg for each inline
parameter, so tags on both inline parameters and inline body-local
``pl.create_tensor(...)`` results take effect at the inlined call sites.
No tag migration is needed; multi-level inlining works at the pass's
fixpoint.

Limitations (MVP):

A tag fires only when the tagged Var reaches a kernel dispatch as a
**static, whole-tensor Arg**. Values that never reach such an Arg are
silently not dumped:

* Dynamic-offset reads — a value read only through a data-dependent
  offset (``q_flat[runtime_row : runtime_row + N, ...]``) lowers to a
  gather / dynamic-address load, not a whole-tensor Arg, so the tag does
  not attach. Stage it through a buffer read with static, compile-time
  tiled offsets and tag that buffer.
* Orch-level ``pl.assemble`` buffers — ``y = pl.assemble(y, tile, off)``
  lowers to a pure name-alias and emits no kernel dispatch, so there is
  no Arg to mark. Use a static in-place slice store ``y[off_slice] =
  tile`` and tag ``y``, or dump the producer kernels' output Args.
* Orch-tier scalar reads — a tensor consumed only by orchestration-level
  ``pl.read(...)`` (e.g. block tables read to compute offsets) never
  enters a device kernel as a Tensor Arg; the MVP runtime path covers
  per-task device Args only.
* Distributed L3+ programs: only chip-level orchestration tasks honour
  the tag; HOST-tier Python SubWorker tensors are not covered by the
  runtime's selective dump path.

Args:
    tensor: The tensor to mark. Must be a ``Tensor`` value bound in
        the enclosing orchestration's scope.

Returns:
    The tensor unchanged. The marker is consumed at parse time.

## `set_cache_policy`

```text
pypto.language.tensor.set_cache_policy(tensor: pypto.language.typing.tensor.Tensor, policy: pypto.pypto_core.ir.CachePolicy) -> None
```

Declare the GM cache-access policy for every read of ``tensor`` in this scope.

The scope-level surface of the cache policy, for tensor programming — where
the GM reads are implicit and there is no ``pl.load`` call to annotate. The
per-access counterpart is the ``cache=`` kwarg on
[`pl.load`][pypto.language.tile.load], which names one read instead of all
of them. ``pl.slice`` deliberately takes no ``cache=``: a slice computes an
address descriptor, it moves no data.

Write it as a **standalone statement directly inside** a ``pl.at(...)`` /
``pl.spmd(...)`` scope body. Anywhere else — nested in an ``if`` / ``for``
inside the scope, or outside a scope altogether — the parser rejects it,
because the declaration attaches to the scope, and a conditionally-executed
declaration would be a promise the compiler cannot check.

Semantics:

* **A contract, not a hint.** ``CachePolicy.BYPASS`` asserts two things
  about ``tensor``: that the kernel streams it with no reuse worth caching,
  and that **nothing writes those bytes while the kernel runs**. Mixing a
  cached write and a bypassing read of the same bytes is a coherency bug
  the compiler cannot detect, so coherency is the author's contract. This
  is why the policy is never a default and never inferred. Declaring
  BYPASS on a tensor the scope itself writes is rejected at outlining.
* **Tracked by Var identity, never by name.** The declaration names the
  binding live at the scope, so rebinding the name afterwards
  (``b = self.foo(b)``) yields a new value the declaration does not cover.
* **Consumed at parse time.** It emits no IR statement of its own: the
  parser records it on the enclosing scope, the scope outliner resolves it
  to the outlined kernel's parameters, and ``ConvertTensorToTileOps`` turns
  it into a ``cache`` kwarg on each ``tile.load`` that reads a declared
  parameter.
* **Explicit wins.** An explicit ``pl.load(..., cache=...)`` overrides the
  scope declaration for that one access, in both directions — so
  ``cache=pl.CachePolicy.DEFAULT`` opts a single read back into the cache
  inside a bypassing scope.

Current status: PTOAS has no L2-bypass path yet
(https://github.com/hw-native-sys/PTOAS/issues/1356). The declaration is
carried all the way to codegen, but codegen emits a warning and compiles it
as an ordinary cached access, so generated code is unchanged today. Writing
the declaration now is what makes the kernel pick the bypass up for free
once that lands.

Args:
    tensor: The tensor whose reads the policy applies to. Must be a
        ``Tensor`` value bound outside the scope and read inside it.
    policy: ``CachePolicy.BYPASS`` to declare a streaming, non-cached read;
        ``CachePolicy.DEFAULT`` for an ordinary cached read.

Returns:
    Nothing. The marker is consumed at parse time and produces no value.

Example:
    >>> with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
    ...     pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
    ...     c = pl.matmul(a, b, out_dtype=pl.FP32)
    ...     out = pl.assemble(out, c, [0, 0])

## `read`

```text
pypto.language.tensor.read(tensor: pypto.language.typing.tensor.Tensor, indices: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr]) -> pypto.language.typing.scalar.Scalar
```

Read a scalar value from a tensor at given indices.

Args:
    tensor: Input tensor
    indices: A single index expression (for 1-D flat access) or a list of
        index expressions (one per tensor dimension)

Returns:
    Scalar wrapping the read operation

## `write`

```text
pypto.language.tensor.write(tensor: pypto.language.typing.tensor.Tensor, indices: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], value: pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.pypto_core.ir.Expr
```

Write a scalar value into a tensor at given indices.

Args:
    tensor: Destination tensor
    indices: A single index expression (for 1-D flat access) or a list of
        index expressions (one per tensor dimension)
    value: Scalar value to write (DSL Scalar or raw Expr)

Returns:
    The underlying ``tensor.write`` call expression. Direct callers
    typically ignore it; the DSL parser surfaces it as an ``EvalStmt``.

## `dim`

```text
pypto.language.tensor.dim(tensor: pypto.language.typing.tensor.Tensor, axis: int | pypto.pypto_core.ir.ConstInt) -> pypto.language.typing.scalar.Scalar
```

Extract a shape dimension from a tensor as a scalar value.

Args:
    tensor: Input tensor
    axis: Dimension index (supports negative indexing). Accepts either
        a Python ``int`` or a ``ConstInt`` (parser-shape).

Returns:
    Scalar wrapping the dim operation (INT64)

## `slice`

```text
pypto.language.tensor.slice(tensor: ~_TensorT, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, drop_dims: collections.abc.Sequence[int | pypto.pypto_core.ir.Expr] | None = None, pad_value: pypto.pypto_core.ir.PadValue | int | float | None = None, clamp: bool = False) -> ~_TensorT
```

Create a slice of a tensor with new shape and optional valid shape.

The slice is never valid where the source is not: the source's valid region,
shifted by ``offset`` and cut to the window, bounds the result.

Args:
    tensor: Input tensor
    shape: New shape dimensions. Always full-rank — a scalar-indexed axis
        contributes a unit dim here and is listed in ``drop_dims``.
    offset: Offset dimensions for the slice
    valid_shape: Valid shape dimensions. When omitted, the source's validity
        under the window is used. Narrows the result; cannot widen it.
    drop_dims: Optional axes to erase from the result type (numpy-style rank
        reduction). Each listed axis must be a static unit dim of ``shape``
        and must still be fully valid after the intersection above.
        ``None`` / ``[]`` drops nothing (fully backward compatible).
    pad_value: Optional padding mode for out-of-valid-shape elements.
        ``None`` means the source's padding mode carries through.
        Accepts ``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or
        the literal sugars ``0``, ``math.inf``, ``-math.inf`` (same
        spelling as [`tensor.fillpad`][pypto.language.tensor.fillpad]). Only meaningful when the
        *effective* valid region is smaller than ``shape`` — which an explicit
        ``valid_shape``, a partially-valid source, or ``clamp=True`` can each
        bring about.
    clamp: Sanction a window that runs off the end of the source. By default
        a slice asserts ``offset + shape`` stays inside the source and is
        rejected when that provably fails; ``clamp=True`` lets the window
        overhang and cuts the valid region back to the source edge. Use it
        for a fixed-width tail read whose overhang is never addressed.

Returns:
    Tensor wrapping the slice operation

## `fillpad`

```text
pypto.language.tensor.fillpad(tensor: pypto.language.typing.tensor.Tensor, pad_value: pypto.pypto_core.ir.PadValue | int | float = PadValue.zero) -> pypto.language.typing.tensor.Tensor
```

Fill invalid tensor view elements with the specified padding value.

Args:
    tensor: Input tensor
    pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
        the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
        ``PadValue.zero``. Other values raise — the hardware only supports
        the three padding modes.

Returns:
    Tensor wrapping the fillpad operation

## `fillpad_expand`

```text
pypto.language.tensor.fillpad_expand(tensor: pypto.language.typing.tensor.Tensor, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], pad_value: pypto.pypto_core.ir.PadValue | int | float = PadValue.zero) -> pypto.language.typing.tensor.Tensor
```

Copy a smaller source tensor into a larger destination tensor, padding the rest.

Unlike [`fillpad`][pypto.language.tensor.fillpad] (which keeps the same shape and only fills the invalid
view region), the destination ``shape`` may be larger than the source in
either dimension. The source's valid region is copied into the top-left of
the destination and every other element is filled with ``pad_value``.

Args:
    tensor: Source tensor
    shape: Destination shape; each dimension must be >= the source dimension
    pad_value: ``PadValue`` enum (``zero`` / ``max`` / ``min``), or one of
        the literal sugars ``0``, ``math.inf``, ``-math.inf``. Default is
        ``PadValue.zero``. Other values raise — the hardware only supports
        the three padding modes.

Returns:
    Tensor wrapping the fillpad_expand operation (a new, larger tensor).

## `full`

```text
pypto.language.tensor.full(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, value: int | float) -> pypto.language.typing.tensor.Tensor
```

Create a tensor of specified shape filled with a constant value.

Args:
    shape: Shape of the tensor
    dtype: Data type of tensor elements
    value: Filling scalar value (int or float)

Returns:
    Tensor wrapping the full operation

## `ci`

```text
pypto.language.tensor.ci(start: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType = int32, descending: bool = False) -> pypto.language.typing.tensor.Tensor
```

Generate a contiguous integer sequence into a tensor.

Equivalent to ``numpy.arange`` / ``torch.arange``. Lowers to ``tile.ci`` → ``pto.tci``.

Note:
    ``pto.tci`` only populates the first row. Leading dimensions must be 1 —
    prefer shapes of the form ``[1, N]``.

Args:
    start: Starting integer (plain int or Scalar). Must match ``dtype``.
    shape: Destination tensor shape (innermost dim != 1).
    dtype: Destination dtype. One of {INT16, INT32}. Defaults to INT32.
    descending: If True, generate a descending sequence.

Returns:
    Tensor wrapping the ci operation.

## `arange`

```text
pypto.language.tensor.arange(start: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType = int32, descending: bool = False) -> pypto.language.typing.tensor.Tensor
```

Generate a contiguous integer sequence into a tensor.

Equivalent to ``numpy.arange`` / ``torch.arange``. Lowers to ``tile.ci`` → ``pto.tci``.

Note:
    ``pto.tci`` only populates the first row. Leading dimensions must be 1 —
    prefer shapes of the form ``[1, N]``.

Args:
    start: Starting integer (plain int or Scalar). Must match ``dtype``.
    shape: Destination tensor shape (innermost dim != 1).
    dtype: Destination dtype. One of {INT16, INT32}. Defaults to INT32.
    descending: If True, generate a descending sequence.

Returns:
    Tensor wrapping the ci operation.

## `random`

```text
pypto.language.tensor.random(key0: int | pypto.language.typing.scalar.Scalar, key1: int | pypto.language.typing.scalar.Scalar, counter0: int | pypto.language.typing.scalar.Scalar, counter1: int | pypto.language.typing.scalar.Scalar, counter2: int | pypto.language.typing.scalar.Scalar, counter3: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType = uint32, rounds: int = 10) -> pypto.language.typing.tensor.Tensor
```

Generate counter-based pseudo-random values into a tensor.

Implements a counter-based (Philox/ChaCha-style) RNG. Each element is derived
deterministically from the 64-bit key ``(key0, key1)`` and 128-bit counter
``(counter0..counter3)`` plus the element position, so the same seeds always
reproduce the same tensor. Lowers to ``tile.random`` → ``pto.trandom``.

Args:
    key0: Low INT32 key word (plain int or Scalar).
    key1: High INT32 key word (plain int or Scalar).
    counter0: First INT32 counter word.
    counter1: Second INT32 counter word.
    counter2: Third INT32 counter word.
    counter3: Fourth INT32 counter word.
    shape: Destination tensor shape (static).
    dtype: Destination dtype. One of {INT32, UINT32}. Defaults to UINT32.
    rounds: Cipher round count, 7 or 10. Defaults to 10.

Returns:
    Tensor wrapping the random operation.

## `matmul`

```text
pypto.language.tensor.matmul(lhs: pypto.language.typing.tensor.Tensor, rhs: pypto.language.typing.tensor.Tensor, out_dtype: int | pypto.pypto_core.DataType | None = None, a_trans: bool = False, b_trans: bool = False, c_matrix_nz: bool = False) -> pypto.language.typing.tensor.Tensor
```

Matrix multiplication with optional transpose.

A transpose flag swaps its own operand's two trailing axes, so that operand must
be at least 2D: ``a_trans`` with a 1D ``lhs`` (or ``b_trans`` with a 1D ``rhs``)
raises rather than being ignored. On the mixed mat-vec / vec-mat forms the flag
applies to the matrix side, so a ``lhs`` stored ``[K, M]`` with ``a_trans=True``
against a ``[K]`` ``rhs`` deduces ``[M]``.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor
    out_dtype: Output data type. Optional; inferred from the operands when
        omitted. When given it must be one of the dtypes the Cube writeback
        can produce: FP32/FP16/BF16 for float operands, INT32 for int
        operands (see ``pl.matmul``)
    a_trans: Whether to transpose lhs (requires a 2D+ lhs)
    b_trans: Whether to transpose rhs (requires a 2D+ rhs)
    c_matrix_nz: C matrix non-zero flag

Returns:
    Tensor wrapping the matmul operation

## `matmul_acc`

```text
pypto.language.tensor.matmul_acc(acc: pypto.language.typing.tensor.Tensor, lhs: pypto.language.typing.tensor.Tensor, rhs: pypto.language.typing.tensor.Tensor, a_trans: bool = False, b_trans: bool = False, init_cond: bool | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None) -> pypto.language.typing.tensor.Tensor
```

Matrix multiplication with accumulation: acc += lhs @ rhs.

``init_cond`` makes the accumulator's initial value conditional: on the steps
where it holds, ``acc`` is overwritten with ``lhs @ rhs`` rather than
accumulated into. This is the split-K idiom, and it removes the need to zero
the accumulator or to peel the first K step::

    for k0 in pl.pipeline(0, K, K_TILE, stage=2):
        acc[t0 : t0 + R, :] = pl.matmul_acc(
            acc[t0 : t0 + R, :], x_k, w_k, b_trans=True, init_cond=(k0 == 0)
        )

Only 2D operands support the predicate; loop over the batch dimension
instead of passing higher-rank operands alongside ``init_cond``.

Args:
    acc: Accumulator tensor
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor
    a_trans: Whether to transpose lhs
    b_trans: Whether to transpose rhs
    init_cond: Optional predicate selecting overwrite over accumulate

Returns:
    Tensor wrapping the matmul_acc operation

## `mul`

```text
pypto.language.tensor.mul(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise multiplication of tensor and tensor or scalar.

Automatically selects between tensor.mul (tensor x tensor) and
tensor.muls (tensor x scalar) based on the rhs type.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

Returns:
    Tensor wrapping the mul operation

## `muls`

```text
pypto.language.tensor.muls(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise multiplication of tensor and scalar.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side scalar (int/float/Expr with ScalarType)

Returns:
    Tensor wrapping the muls operation

## `add`

```text
pypto.language.tensor.add(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise addition of tensor and tensor or scalar.

Automatically selects between tensor.add (tensor + tensor) and
tensor.adds (tensor + scalar) based on the rhs type.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

Returns:
    Tensor wrapping the add operation

## `adds`

```text
pypto.language.tensor.adds(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise addition of tensor and scalar.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side scalar (int/float/Expr with ScalarType)

Returns:
    Tensor wrapping the adds operation

## `sub`

```text
pypto.language.tensor.sub(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise subtraction of tensor and tensor or scalar.

Automatically selects between tensor.sub (tensor - tensor) and
tensor.subs (tensor - scalar) based on the rhs type.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

Returns:
    Tensor wrapping the sub operation

## `subs`

```text
pypto.language.tensor.subs(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise subtraction of tensor and scalar.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side scalar (int/float/Expr with ScalarType)

Returns:
    Tensor wrapping the subs operation

## `div`

```text
pypto.language.tensor.div(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, high_precision: bool = False) -> pypto.language.typing.tensor.Tensor
```

Element-wise division of tensor and tensor or scalar.

Automatically selects between tensor.div (tensor / tensor) and
tensor.divs (tensor / scalar) based on the rhs type.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)
    high_precision: Whether to select PTOAS's high-precision division mode.
        Only available when ``rhs`` is a Tensor.

Returns:
    Tensor wrapping the div operation

## `divs`

```text
pypto.language.tensor.divs(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise division of tensor and scalar.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side scalar (int/float/Expr/Scalar)

Returns:
    Tensor wrapping the divs operation

## `part_add`

```text
pypto.language.tensor.part_add(lhs: pypto.language.typing.tensor.Tensor, rhs: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Partial element-wise add of two tensors.

Args:
    lhs: First source tensor
    rhs: Second source tensor

Returns:
    Tensor wrapping the part_add operation

## `part_mul`

```text
pypto.language.tensor.part_mul(lhs: pypto.language.typing.tensor.Tensor, rhs: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Partial element-wise multiply of two tensors.

Args:
    lhs: First source tensor
    rhs: Second source tensor

Returns:
    Tensor wrapping the part_mul operation

## `part_max`

```text
pypto.language.tensor.part_max(lhs: pypto.language.typing.tensor.Tensor, rhs: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Partial element-wise max of two tensors.

Args:
    lhs: First source tensor
    rhs: Second source tensor

Returns:
    Tensor wrapping the part_max operation

## `part_min`

```text
pypto.language.tensor.part_min(lhs: pypto.language.typing.tensor.Tensor, rhs: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Partial element-wise min of two tensors.

Args:
    lhs: First source tensor
    rhs: Second source tensor

Returns:
    Tensor wrapping the part_min operation

## `fmod`

```text
pypto.language.tensor.fmod(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise truncating remainder of tensor and tensor or scalar.

Automatically selects between tensor.fmod (tensor, tensor) and
tensor.fmods (tensor, scalar) based on the rhs type. The result matches
``torch.fmod`` (the remainder takes the sign of the dividend).

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor or scalar (int/float/Tensor/Scalar)

Returns:
    Tensor wrapping the fmod operation

## `fmods`

```text
pypto.language.tensor.fmods(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise truncating remainder of tensor and scalar.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side scalar (int/float/Expr/Scalar)

Returns:
    Tensor wrapping the fmods operation

## `maximum`

```text
pypto.language.tensor.maximum(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise maximum of tensor and tensor or scalar.

The conversion pass handles the tensor-vs-tensor / tensor-vs-scalar
dispatch internally — there is no separate ``maximums`` front-end op.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor or scalar

Returns:
    Tensor wrapping the maximum operation

## `minimum`

```text
pypto.language.tensor.minimum(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise minimum of tensor and tensor or scalar.

The conversion pass handles the tensor-vs-tensor / tensor-vs-scalar
dispatch internally — there is no separate ``minimums`` front-end op.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor or scalar

Returns:
    Tensor wrapping the minimum operation

## `cmp`

```text
pypto.language.tensor.cmp(lhs: pypto.language.typing.tensor.Tensor, rhs: int | float | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, cmp_type: int = 0) -> pypto.language.typing.tensor.Tensor
```

Element-wise comparison of tensor and tensor or scalar (returns 0/1 tensor).

The conversion pass handles the tensor-vs-tensor / tensor-vs-scalar
dispatch internally — there is no separate ``cmps`` front-end op.

Args:
    lhs: Left-hand side tensor
    rhs: Right-hand side tensor or scalar
    cmp_type: Comparison type code (0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge)

Returns:
    Tensor of 0/1 with the same shape and dtype as ``lhs``

## `and_`

```text
pypto.language.tensor.and_(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise AND of tensor and tensor or scalar.

Automatically selects between tensor.and (tensor & tensor) and
tensor.ands (tensor & scalar) based on the rhs type.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Right-hand side tensor or integer scalar

Returns:
    Tensor wrapping the and operation

## `ands`

```text
pypto.language.tensor.ands(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise AND of tensor and scalar.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Right-hand side integer scalar

Returns:
    Tensor wrapping the ands operation

## `or_`

```text
pypto.language.tensor.or_(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise OR of tensor and tensor or scalar.

Automatically selects between tensor.or (tensor | tensor) and
tensor.ors (tensor | scalar) based on the rhs type.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Right-hand side tensor or integer scalar

Returns:
    Tensor wrapping the or operation

## `ors`

```text
pypto.language.tensor.ors(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise OR of tensor and scalar.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Right-hand side integer scalar

Returns:
    Tensor wrapping the ors operation

## `xor`

```text
pypto.language.tensor.xor(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise XOR of tensor and tensor or scalar.

Automatically selects between tensor.xor (tensor ^ tensor) and
tensor.xors (tensor ^ scalar) based on the rhs type.

Unlike ``pl.tile.xor``, there is no ``tmp`` parameter: the scratch buffer
that pto.txor needs is allocated during Tensor-to-Tile lowering.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Right-hand side tensor or integer scalar

Returns:
    Tensor wrapping the xor operation

## `xors`

```text
pypto.language.tensor.xors(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise XOR of tensor and scalar.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Right-hand side integer scalar

Returns:
    Tensor wrapping the xors operation

## `not_`

```text
pypto.language.tensor.not_(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise NOT of a tensor.

Args:
    input: Input tensor; int16 or uint16, matching the TNOT instruction

Returns:
    Tensor wrapping the not operation

## `shl`

```text
pypto.language.tensor.shl(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise left shift of tensor by tensor or scalar.

Automatically selects between tensor.shl (tensor << tensor) and
tensor.shls (tensor << scalar) based on the rhs type.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Shift amount as a tensor or integer scalar

Returns:
    Tensor wrapping the shl operation

## `shls`

```text
pypto.language.tensor.shls(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise left shift of tensor by scalar.

Note:
    The shift amount must be zero or positive. A negative constant is
    rejected when the op is built; a negative value only known at runtime
    is undefined behaviour on the hardware.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Shift amount; must be >= 0

Returns:
    Tensor wrapping the shls operation

## `shr`

```text
pypto.language.tensor.shr(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.language.typing.tensor.Tensor | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise right shift of tensor by tensor or scalar.

Automatically selects between tensor.shr (tensor >> tensor) and
tensor.shrs (tensor >> scalar) based on the rhs type. The shift is
arithmetic for signed dtypes and logical for unsigned ones, matching the
tile ops and the underlying ISA.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Shift amount as a tensor or integer scalar

Returns:
    Tensor wrapping the shr operation

## `shrs`

```text
pypto.language.tensor.shrs(lhs: pypto.language.typing.tensor.Tensor, rhs: int | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Element-wise bitwise right shift of tensor by scalar.

Note:
    The shift amount must be zero or positive. A negative constant is
    rejected when the op is built; a negative value only known at runtime
    is undefined behaviour on the hardware.

Args:
    lhs: Left-hand side tensor (integer dtype)
    rhs: Shift amount; must be >= 0

Returns:
    Tensor wrapping the shrs operation

## `row_max`

```text
pypto.language.tensor.row_max(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise max reduction (reduces along last axis, keeps dim).

Args:
    input: Input tensor

Returns:
    Tensor wrapping the row_max operation

## `row_sum`

```text
pypto.language.tensor.row_sum(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise sum reduction (reduces along last axis, keeps dim).

Args:
    input: Input tensor

Returns:
    Tensor wrapping the row_sum operation

## `row_min`

```text
pypto.language.tensor.row_min(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise min reduction (reduces along last axis, keeps dim).

Args:
    input: Input tensor

Returns:
    Tensor wrapping the row_min operation

## `row_prod`

```text
pypto.language.tensor.row_prod(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise product reduction (reduces along last axis, keeps dim).

Args:
    input: Input tensor

Returns:
    Tensor wrapping the row_prod operation

## `col_sum`

```text
pypto.language.tensor.col_sum(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise sum reduction (reduces along axis=-2, keeps dim).

Output shape is ``[..., 1, N]`` for an input of shape ``[..., M, N]``.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the col_sum operation

## `col_max`

```text
pypto.language.tensor.col_max(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise max reduction (reduces along axis=-2, keeps dim).

Output shape is ``[..., 1, N]`` for an input of shape ``[..., M, N]``.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the col_max operation

## `col_min`

```text
pypto.language.tensor.col_min(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise min reduction (reduces along axis=-2, keeps dim).

Output shape is ``[..., 1, N]`` for an input of shape ``[..., M, N]``.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the col_min operation

## `col_prod`

```text
pypto.language.tensor.col_prod(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise product reduction (reduces along axis=-2, keeps dim).

Output shape is ``[..., 1, N]`` for an input of shape ``[..., M, N]``.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the col_prod operation

## `row_argmax`

```text
pypto.language.tensor.row_argmax(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise argmax: index of the per-row maximum (int32, reduces along last axis).

Output shape is ``[..., M, 1]`` for an input of shape ``[..., M, N]``.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the row_argmax operation

## `row_argmin`

```text
pypto.language.tensor.row_argmin(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise argmin: index of the per-row minimum (int32, reduces along last axis).

Output shape is ``[..., M, 1]`` for an input of shape ``[..., M, N]``.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the row_argmin operation

## `col_argmax`

```text
pypto.language.tensor.col_argmax(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise argmax: index of the per-column maximum (int32, reduces along axis=-2).

Output shape is ``[..., 1, N]`` for an input of shape ``[..., M, N]``.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the col_argmax operation

## `col_argmin`

```text
pypto.language.tensor.col_argmin(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise argmin: index of the per-column minimum (int32, reduces along axis=-2).

Output shape is ``[..., 1, N]`` for an input of shape ``[..., M, N]``.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the col_argmin operation

## `row_expand`

```text
pypto.language.tensor.row_expand(target: pypto.language.typing.tensor.Tensor, row_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise expansion: expand row_vec [M, 1] to target shape [M, N].

Args:
    target: Target tensor defining output shape (TensorType [M, N])
    row_vec: Row vector to expand (TensorType [M, 1])

Returns:
    Tensor wrapping the row_expand operation

## `row_expand_mul`

```text
pypto.language.tensor.row_expand_mul(tensor: pypto.language.typing.tensor.Tensor, row_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise broadcast multiplication: tensor[i,:] * row_vec[i,0].

Multiplies each row of the tensor by the corresponding row vector value,
for all ``i``.

Args:
    tensor: Input tensor (TensorType [M, N])
    row_vec: Row vector (TensorType [M, 1])

Returns:
    Tensor wrapping the row_expand_mul operation

## `row_expand_div`

```text
pypto.language.tensor.row_expand_div(tensor: pypto.language.typing.tensor.Tensor, row_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise broadcast division: tensor[i,:] / row_vec[i,0].

Divides each row of the tensor by the corresponding row vector value,
for all ``i``.

Args:
    tensor: Input tensor (TensorType [M, N])
    row_vec: Row vector (TensorType [M, 1])

Returns:
    Tensor wrapping the row_expand_div operation

## `row_expand_add`

```text
pypto.language.tensor.row_expand_add(tensor: pypto.language.typing.tensor.Tensor, row_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise broadcast addition: tensor[i,:] + row_vec[i,0].

Adds a row vector to each row of the tensor, for all ``i``.

Args:
    tensor: Input tensor (TensorType [M, N])
    row_vec: Row vector (TensorType [M, 1])

Returns:
    Tensor wrapping the row_expand_add operation

## `row_expand_sub`

```text
pypto.language.tensor.row_expand_sub(tensor: pypto.language.typing.tensor.Tensor, row_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise broadcast subtraction: tensor[i,:] - row_vec[i,0].

Subtracts a row vector from each row of the tensor, for all ``i``.

Args:
    tensor: Input tensor (TensorType [M, N])
    row_vec: Row vector (TensorType [M, 1])

Returns:
    Tensor wrapping the row_expand_sub operation

## `row_expand_max`

```text
pypto.language.tensor.row_expand_max(tensor: pypto.language.typing.tensor.Tensor, row_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise broadcast maximum: max(tensor[i,:], row_vec[i,0]).

Takes the element-wise maximum of each row and the row vector value,
for all ``i``.

Args:
    tensor: Input tensor (TensorType [M, N])
    row_vec: Row vector (TensorType [M, 1])

Returns:
    Tensor wrapping the row_expand_max operation

## `row_expand_min`

```text
pypto.language.tensor.row_expand_min(tensor: pypto.language.typing.tensor.Tensor, row_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise broadcast minimum: min(tensor[i,:], row_vec[i,0]).

Takes the element-wise minimum of each row and the row vector value,
for all ``i``.

Args:
    tensor: Input tensor (TensorType [M, N])
    row_vec: Row vector (TensorType [M, 1])

Returns:
    Tensor wrapping the row_expand_min operation

## `row_expand_expdif`

```text
pypto.language.tensor.row_expand_expdif(tensor: pypto.language.typing.tensor.Tensor, row_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Row-wise exp-diff: exp(tensor[i,:] - row_vec[i,0]).

Computes the exponential of the per-row difference, for all ``i``.

Args:
    tensor: Input tensor (TensorType [M, N])
    row_vec: Row vector providing per-row scalar (TensorType [M, 1])

Returns:
    Tensor wrapping the row_expand_expdif operation

## `col_expand_mul`

```text
pypto.language.tensor.col_expand_mul(tensor: pypto.language.typing.tensor.Tensor, col_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise broadcast multiplication: tensor[:,j] * col_vec[0,j].

Multiplies each column of the tensor by the corresponding column vector
value, for all ``j``.

Args:
    tensor: Input tensor (TensorType [M, N])
    col_vec: Column vector (TensorType [1, N])

Returns:
    Tensor wrapping the col_expand_mul operation

## `col_expand`

```text
pypto.language.tensor.col_expand(tensor: pypto.language.typing.tensor.Tensor, col_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise expansion: expand col_vec [1, N] to target shape [M, N].

Args:
    tensor: Target tensor defining output shape (TensorType [M, N])
    col_vec: Column vector to expand (TensorType [1, N])

Returns:
    Tensor wrapping the col_expand operation

## `col_expand_div`

```text
pypto.language.tensor.col_expand_div(tensor: pypto.language.typing.tensor.Tensor, col_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise broadcast division: tensor[:,j] / col_vec[0,j].

Divides each column of the tensor by the corresponding column vector
value, for all ``j``.

Args:
    tensor: Input tensor (TensorType [M, N])
    col_vec: Column vector (TensorType [1, N])

Returns:
    Tensor wrapping the col_expand_div operation

## `col_expand_sub`

```text
pypto.language.tensor.col_expand_sub(tensor: pypto.language.typing.tensor.Tensor, col_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise broadcast subtraction: tensor[:,j] - col_vec[0,j].

Subtracts a column vector from each column of the tensor, for all ``j``.

Args:
    tensor: Input tensor (TensorType [M, N])
    col_vec: Column vector (TensorType [1, N])

Returns:
    Tensor wrapping the col_expand_sub operation

## `col_expand_add`

```text
pypto.language.tensor.col_expand_add(tensor: pypto.language.typing.tensor.Tensor, col_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise broadcast addition: tensor[:,j] + col_vec[0,j].

Adds a column vector to each column of the tensor, for all ``j``.

Args:
    tensor: Input tensor (TensorType [M, N])
    col_vec: Column vector (TensorType [1, N])

Returns:
    Tensor wrapping the col_expand_add operation

## `col_expand_max`

```text
pypto.language.tensor.col_expand_max(tensor: pypto.language.typing.tensor.Tensor, col_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise broadcast maximum: max(tensor[:,j], col_vec[0,j]).

Takes the element-wise maximum of each column and the column vector
value, for all ``j``.

Args:
    tensor: Input tensor (TensorType [M, N])
    col_vec: Column vector (TensorType [1, N])

Returns:
    Tensor wrapping the col_expand_max operation

## `col_expand_min`

```text
pypto.language.tensor.col_expand_min(tensor: pypto.language.typing.tensor.Tensor, col_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise broadcast minimum: min(tensor[:,j], col_vec[0,j]).

Takes the element-wise minimum of each column and the column vector
value, for all ``j``.

Args:
    tensor: Input tensor (TensorType [M, N])
    col_vec: Column vector (TensorType [1, N])

Returns:
    Tensor wrapping the col_expand_min operation

## `col_expand_expdif`

```text
pypto.language.tensor.col_expand_expdif(tensor: pypto.language.typing.tensor.Tensor, col_vec: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Column-wise exp-diff: exp(tensor[:,j] - col_vec[0,j]).

Computes the exponential of the per-column difference, for all ``j``.

Args:
    tensor: Input tensor (TensorType [M, N])
    col_vec: Column vector providing per-column scalar (TensorType [1, N])

Returns:
    Tensor wrapping the col_expand_expdif operation

## `expands`

```text
pypto.language.tensor.expands(target: pypto.language.typing.tensor.Tensor, scalar: int | float | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor
```

Expand scalar to target tensor shape.

Args:
    target: Target tensor defining output shape
    scalar: Scalar value to expand

Returns:
    Tensor wrapping the expands operation

## `expand_clone`

```text
pypto.language.tensor.expand_clone(src: pypto.language.typing.tensor.Tensor, target: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Clone and expand input to target shape.

Args:
    src: Source tensor
    target: Target tensor defining output shape

Returns:
    Tensor wrapping the expand_clone operation

## `exp`

```text
pypto.language.tensor.exp(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Element-wise exponential operation.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the exp operation

## `log`

```text
pypto.language.tensor.log(input: pypto.language.typing.tensor.Tensor, high_precision: bool = False) -> pypto.language.typing.tensor.Tensor
```

Element-wise natural logarithm operation.

Args:
    input: Input tensor
    high_precision: Whether to select PTOAS's high-precision logarithm mode

Returns:
    Tensor wrapping the log operation

## `sin`

```text
pypto.language.tensor.sin(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Element-wise sine operation (input in radians). FP32-only.

Args:
    input: Input tensor (must be FP32; cast explicitly otherwise)

Returns:
    Tensor wrapping the sin operation

## `cos`

```text
pypto.language.tensor.cos(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Element-wise cosine operation (input in radians). FP32-only.

Args:
    input: Input tensor (must be FP32; cast explicitly otherwise)

Returns:
    Tensor wrapping the cos operation

## `neg`

```text
pypto.language.tensor.neg(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Element-wise negation operation.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the neg operation

## `abs`

```text
pypto.language.tensor.abs(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Element-wise absolute value operation.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the abs operation

## `recip`

```text
pypto.language.tensor.recip(input: pypto.language.typing.tensor.Tensor, high_precision: bool = False) -> pypto.language.typing.tensor.Tensor
```

Element-wise reciprocal (1/x) operation.

Args:
    input: Input tensor
    high_precision: Whether to select PTOAS's high-precision reciprocal mode (FP16/FP32 only)

Returns:
    Tensor wrapping the recip operation

## `sqrt`

```text
pypto.language.tensor.sqrt(input: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Element-wise square root operation.

Args:
    input: Input tensor

Returns:
    Tensor wrapping the sqrt operation

## `rsqrt`

```text
pypto.language.tensor.rsqrt(input: pypto.language.typing.tensor.Tensor, high_precision: bool = False) -> pypto.language.typing.tensor.Tensor
```

Element-wise reciprocal square root operation.

Args:
    input: Input tensor
    high_precision: If True, lower to the higher-precision PTO path. The
        compiler allocates a scratch buffer during tensor-to-tile conversion.

Returns:
    Tensor wrapping the rsqrt operation

## `cast`

```text
pypto.language.tensor.cast(input: pypto.language.typing.tensor.Tensor, target_type: int | pypto.pypto_core.DataType, mode: str | int = 'round') -> pypto.language.typing.tensor.Tensor
```

Type casting operation.

Args:
    input: Input tensor
    target_type: Target data type
    mode: Rounding mode — string name ("none", "rint", "round", "floor",
          "ceil", "trunc", "odd") or int (0–6)

Returns:
    Tensor wrapping the cast operation

## `assemble`

```text
pypto.language.tensor.assemble(target: ~_TensorT, source: pypto.language.typing.tensor.Tensor, offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_) -> ~_TensorT
```

Write/update tensor values at specified offset.

Args:
    target: Target tensor to update
    source: Source tensor to write
    offset: Offset dimensions for where to write
    atomic: Combine mode for the write. ``AtomicType.None_`` (default)
        overwrites; ``AtomicType.Add`` atomically adds ``source`` into the
        target at ``offset`` — used for split-K, where several cores
        accumulate partial products into one output. Only valid inside an
        InCore function — typically a ``pl.at(level=pl.Level.CORE_GROUP, ...)``
        scope — where ``source`` lowers to an on-chip tile (a compute result)
        and the write lowers to an atomic-add store into a global-memory
        target (a function output tensor). Every other form is rejected at
        compile time: a tensor-to-tensor assemble has no store to carry the
        combine, an assemble into an on-chip tile has no global-memory
        destination, and at the orchestration level no atomic-combine
        instruction exists at all.

        NOTE: atomic-add accumulation order across cores is not fixed, so
        floating-point results are non-deterministic. The target must be
        zero-initialised before the kernel runs. Supported dtypes:
        fp32 / bf16 / fp16 / int32 / int16 / int8. bf16 atomic-add is
        available on the Ascend910B (A2/A3) profile; it is not supported on
        A5, where an fp32 accumulator + cast is required instead.

Returns:
    Tensor wrapping the assemble operation

## `concat`

```text
pypto.language.tensor.concat(src0: pypto.language.typing.tensor.Tensor, src1: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Concatenate two tensors along the column dimension.

Args:
    src0: First source tensor
    src1: Second source tensor

Returns:
    Tensor with concatenated columns

## `reshape`

```text
pypto.language.tensor.reshape(tensor: pypto.language.typing.tensor.Tensor, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr]) -> pypto.language.typing.tensor.Tensor
```

Reshape tensor to new shape.

The valid region is carried through, never widened: the result holds real
data in exactly the cells the input did. See ``pl.reshape`` for the cases
that always map.

Args:
    tensor: Input tensor
    shape: New shape dimensions

Returns:
    Tensor wrapping the reshape operation

Raises:
    ValueError: If the element count changes, or if the input holds real
        data in only part of its buffer and no origin-anchored region of
        ``shape`` describes those same cells.

## `reinterpret_view`

```text
pypto.language.tensor.reinterpret_view(data: pypto.language.typing.tensor.Tensor, dtype: pypto.pypto_core.DataType, *, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None) -> pypto.language.typing.tensor.Tensor
```

Reinterpret a tensor over the same bytes with a different dtype.

Args:
    data: Input tensor.
    dtype: Target element dtype, which must differ from the source dtype.
    shape: Optional byte-equivalent target shape. When omitted, the
        physically contiguous dimension is scaled according to the
        source/target dtype byte ratio.

Returns:
    Tensor wrapping the zero-copy reinterpret-view operation.

## `transpose`

```text
pypto.language.tensor.transpose(tensor: pypto.language.typing.tensor.Tensor, axis1: int, axis2: int) -> pypto.language.typing.tensor.Tensor
```

Transpose tensor by swapping two axes.

Args:
    tensor: Input tensor
    axis1: First axis to swap (supports negative indexing)
    axis2: Second axis to swap (supports negative indexing)

Returns:
    Tensor wrapping the transpose operation

## `view`

```text
pypto.language.tensor.view(tensor: ~_TensorT, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, *, layout: pypto.pypto_core.ir.TensorLayout | None = None) -> ~_TensorT
```

Reinterpret a tensor over the same physical memory.

At least one of ``shape`` or ``layout`` must be provided: ``shape`` derives
canonical strides for the requested shape, ``layout`` derives the canonical
ND/DN layout view. The result is a zero-copy view over the same physical
memory, and its target shape must have rank at least 1.

Args:
    tensor: Source tensor.
    shape: New shape for the view. Must be product-preserving unless
        symbolic dimensions are present. Rank-zero views are not supported.
    valid_shape: Explicit valid dimensions for a packed ND
        leading-dimension collapse to 2D or contiguous-prefix linear
        collapse to ``[1, product(shape)]``. Required when either supported
        collapse reinterprets a source with partial validity.
    layout: Target ``TensorLayout`` (ND or DN); DN requires rank at least 2.
        Orchestration supports ND shape reinterprets and shaped
        ND/MX_A_ZZ/MX_B_NN backing and consumer views for dynamic FP8E8M0
        MX scales. Other layout-changing shape views are limited to in-core
        lowering.

Returns:
    Tensor wrapping the view operation.

Raises:
    ValueError: If the requested shape/layout is missing, unsupported, or
        inconsistent with the source tensor metadata.

## `scatter_update`

```text
pypto.language.tensor.scatter_update(input: pypto.language.typing.tensor.Tensor, *args: Any, **kwargs: Any) -> pypto.language.typing.tensor.Tensor
```

Update input tensor rows at positions specified by 2D index with values from src.

Supports two rank variants:

- 2D: ``input [rows, d]``, ``src [b*s, d]``, ``index [b, s]``
- 4D: ``input [blockNum, blockSize, 1, d]``, ``src [b, s, 1, d]``, ``index [b, s]``

For each ``(i, j)``, row ``input[index[i*s + j]]`` receives row ``src[i*s + j]``
(linear layout).

Accepts the same flexible call shapes as the IR builder
``pypto.ir.op.tensor.scatter_update``:

- ``scatter_update(input, dim, index, src)``
- ``scatter_update(input, index, src, dim=-2)``
- ``scatter_update(input, dim, index=..., src=...)``

Tensor / Scalar wrappers are unwrapped before forwarding so the IR
builder receives raw ``Expr`` operands.

## `set_validshape`

```text
pypto.language.tensor.set_validshape(tensor: pypto.language.typing.tensor.Tensor, valid_rows: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr, valid_cols: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr) -> pypto.language.typing.tensor.Tensor
```

Update valid-shape metadata of a tensor without data movement.

.. note::
    Prefer expressing the extent at its source where possible —
    ``pl.load(..., valid_shape=...)`` or a slice's ``valid_shape=`` — and use
    this to pin an extent the type deducer cannot infer.

Args:
    tensor: Input tensor (must be 2D)
    valid_rows: Number of valid rows (int or Scalar[INDEX])
    valid_cols: Number of valid columns (int or Scalar[INDEX])

Returns:
    Tensor with updated valid_shape metadata

## `sort32`

```text
pypto.language.tensor.sort32(src: pypto.language.typing.tensor.Tensor, idx: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Sort fixed 32-element blocks with explicit index tensor (tensor-level).

Tensor-level counterpart of ``pl.tile.sort32``. Sorts 32-element blocks in
src, permuting idx alongside. Returns an 8-byte value-index-pair tensor;
its last dimension is 2x the input width for FP32 and 4x for FP16.

For FP16 src: initialize idx with [0, 1, 2, ..., 31] per block.
For FP32 src: initialize idx with [0, 2, 4, ..., 62] per block.

Args:
    src: Input value tensor (FP16 or FP32)
    idx: Input index tensor with sequential offsets

Returns:
    Tensor wrapping the dtype-dependent expanded sort32 output

## `mrgsort`

```text
pypto.language.tensor.mrgsort(src0: pypto.language.typing.tensor.Tensor, src1: pypto.language.typing.tensor.Tensor | None = None, src2: pypto.language.typing.tensor.Tensor | None = None, src3: pypto.language.typing.tensor.Tensor | None = None, *, exhausted: bool = False, block_len: int | pypto.language.typing.scalar.Scalar | None = None) -> pypto.language.typing.tensor.Tensor
```

Merge sort — format1 (single-list) or format2 (2-4 way merge), tensor-level.

Tensor-level counterpart of ``pl.tile.mrgsort``. The scratch ``tmp`` and
``executed`` tiles required by the tile-level op are synthesized
automatically during conversion as local Vec tiles — users do not pass them.

Format1 usage (keyword block_len):
    out = mrgsort(src, block_len=64)

Format2 usage:
    out = mrgsort(src0, src1)                # 2-way
    out = mrgsort(src0, src1, src2)          # 3-way
    out = mrgsort(src0, src1, src2, src3)    # 4-way
    out = mrgsort(src0, src1, exhausted=True)

Args:
    src0: For format1: input tensor with pre-sorted runs (FP16 or FP32).
          For format2: first sorted input tensor.
    src1: (format2) Second sorted input tensor.
    src2: (format2, optional) Third sorted input tensor (3-way or 4-way).
    src3: (format2, optional) Fourth sorted input tensor (4-way only).
    exhausted: (format2) If True, marks inputs as exhausted (default: False).
    block_len: (format1, keyword-only) Run length, must be multiple of 64.

Returns:
    Tensor with merged sorted elements

## `gather`

```text
pypto.language.tensor.gather(input: pypto.language.typing.tensor.Tensor, dim: int | None = None, index: pypto.language.typing.tensor.Tensor | None = None, *, mask_pattern: int | None = None, output_dtype: int | pypto.pypto_core.DataType | None = None, kvalue: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None, cmp_mode: str | int | None = None, out_cols: int | None = None, offset: int = 0, count_dtype: int | pypto.pypto_core.DataType | None = None) -> pypto.language.typing.tensor.Tensor | tuple[pypto.language.typing.tensor.Tensor, pypto.language.typing.tensor.Tensor]
```

Gather elements of ``input`` (tensor-level) — index / mask / compare form.

The tensor layer exposes a single unified ``gather``. Based on the arguments
you pass, it lowers to one of three tile-level ops:

Index form (``dim`` + ``index``) → [`pl.tile.gather`][pypto.language.tile.gather]::

    output[b, k] = input[b, index[b, k]]

    MVP: only rank-2 inputs with ``dim == -1`` (or ``rank - 1``).
    ``index`` must be an INT32 tensor, or INT16 when ``input`` is a 16-bit
    dtype (FP16/INT16); its shape matches ``input`` on every axis except ``dim``.

Mask form (``mask_pattern=<int>``) → [`pl.tile.gather_mask`][pypto.language.tile.gather_mask]:
    Selects columns of each row by a fixed hardware mask pattern. Last-dim
    shrinks by 2 (P0101/P1010) or 4 (P0001..P1000), or stays the same for P1111.

Compare form (``kvalue`` + ``cmp_mode`` + ``out_cols``) →
[`pl.tile.gather_compare`][pypto.language.tile.gather_compare]:
    Scalar threshold compare (applied to every row). Returns ``(dst, cdst)`` —
    gathered indices ``[rows, out_cols] INT32`` and per-row match counts
    ``[1, rows] count_dtype``.

Args:
    input: Source tensor (FP16/FP32/INT16/INT32).
    dim: (index form) Axis to gather along; only ``-1`` / ``rank - 1`` accepted in MVP.
    index: (index form) Index tensor (INT32, or INT16 with a 16-bit input), same rank as input.
    mask_pattern: (mask form, keyword-only) Mask pattern selector (1-7).
        1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111.
    output_dtype: (mask form, keyword-only) Optional output dtype with the same
        bit width as ``input.dtype`` (e.g. FP32 → UINT32 for sort32 index bits).
    kvalue: (compare form, keyword-only) Scalar threshold (dtype must match ``input``).
    cmp_mode: (compare form, keyword-only) ``"eq"``/``"ne"``/``"lt"``/``"le"``/
        ``"gt"``/``"ge"`` or int 0..5.
    out_cols: (compare form, keyword-only) Output column count per row.
    offset: (compare form, keyword-only) Starting index offset (default 0).
    count_dtype: (compare form, keyword-only) Per-row count dtype, INT32 or UINT32
        (defaults to INT32).

Returns:
    Tensor (index/mask form) or ``(dst, cdst)`` tuple (compare form).

Examples:
    out = gather(input, dim=-1, index=idx)
    out = gather(input, mask_pattern=1)
    out = gather(input, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.UINT32)
    dst, cdst = gather(input, kvalue=kv, cmp_mode="eq", out_cols=8)

## `paged_gather`

```text
pypto.language.tensor.paged_gather(src: pypto.language.typing.tensor.Tensor, indices: pypto.language.typing.tensor.Tensor, block_table: pypto.language.typing.tensor.Tensor, block_size: int, size: int, max_indices: int, *, space: pypto.pypto_core.ir.MemorySpace = MemorySpace.Mat, col_off: int = 0, is_trans: bool = False, is_b_matrix: bool = False) -> pypto.language.typing.tensor.Tensor
```

Paged gather directly into an on-chip buffer (L1 by default, or UB).

Gathers scattered rows of a 2D paged KV pool ``src`` selected by ``indices``,
translated through a paged ``block_table``, directly into an L1 (``space=Mat``,
default) or UB (``space=Vec``) tile — so a subsequent matmul reads from L1
without a GM round-trip. The lowering is a fully-scalar per-row ``GM -> on-chip``
load loop on the Cube core (the bulk KV never touches UB).

Physical row per logical index ``idx``::

    phys = block_table[idx // block_size] * block_size + idx % block_size

Args:
    src: Paged KV pool in GM (2D; FP16/BF16/FP32/INT8).
    indices: Logical row indices (INT32; 1D ``[n]`` or 2D ``[1, n]``).
    block_table: Page table mapping logical block -> physical block (INT32).
    block_size: Number of tokens per page block.
    size: Number of elements gathered per row (<= src columns).
    max_indices: Static upper bound on gathered rows; sizes the on-chip tile.
    space: Destination space — ``MemorySpace.Mat`` (L1, default) or ``MemorySpace.Vec`` (UB).
    col_off: Column start offset within each src row (default 0).
    is_trans: Transpose for matmul B-operand layout (requires ``space=Mat``).
    is_b_matrix: Hint that the result feeds matmul as the B matrix.

Returns:
    Tensor of shape ``[max_indices, size]`` (or transposed) in the chosen space.

Examples:
    kv_l1 = pl.paged_gather(kv_pool, topk_idx, block_table, block_size=128,
                            size=head_dim, max_indices=256)
    out = pl.matmul(q, kv_l1)

## `create_l1`

```text
pypto.language.tensor.create_l1(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, transpose: bool = False) -> pypto.language.typing.tensor.Tensor
```

Create an on-chip (L1/Mat) accumulator for a kernel-driven paged gather.

Companion of [`gather_row`][pypto.language.tensor.gather_row]. Returns a tensor-typed value that composes
with ``pl.matmul`` / softmax but lowers to an L1 (``MemorySpace.Mat``) tile,
so a kernel can build a matmul operand directly on-chip — no GM round-trip.

Args:
    shape: Accumulator shape (static dims). For a matmul B-operand use
        ``[rows, size]``; for a ``b_trans`` B-operand filled by transposing
        gathers use the transposed shape ``[size, rows]`` with ``transpose=True``.
    dtype: Element dtype (matches the gathered ``src`` pool).
    transpose: Allocate the transposed Mat (ZN) layout — required when filling
        with ``pl.gather_row(..., transpose=True)`` (a ``DN2ZN`` per-row load)
        and consuming as a ``pl.matmul(..., b_trans=True)`` B-operand.

Returns:
    Tensor of shape ``shape`` backed by L1.

Examples:
    kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
    for r in pl.range(ATTN_K_TILE):
        kv = pl.gather_row(kv, kv_pool, [r, 0], [phys, 0], [1, HEAD_DIM])
    oi = pl.matmul(probs, kv)

## `gather_row`

```text
pypto.language.tensor.gather_row(acc: pypto.language.typing.tensor.Tensor, src: pypto.language.typing.tensor.Tensor, dst_offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], src_offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], shapes: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], transpose: bool = False, *, valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None) -> pypto.language.typing.tensor.Tensor
```

Gather one GM row into a sub-region of an on-chip accumulator (DPS).

Per-row primitive for a kernel-driven paged gather into L1 — the flexible
counterpart to [`paged_gather`][pypto.language.tensor.paged_gather]: the caller computes the physical
``src_offset`` (block-table lookup, multi-source selection, invalid clamping)
and the ``dst_offset`` slot itself, so arbitrary gather logic stays in the
kernel. DMAs ``src`` straight into ``acc`` (``GM -> L1``, no ``tmov``); the
returned tensor feeds ``pl.matmul`` directly.

Args:
    acc: On-chip accumulator from [`create_l1`][pypto.language.tensor.create_l1] (loop-carried).
    src: Source pool in GM.
    dst_offset: ``[row, col]`` slot within ``acc`` to write.
    src_offset: ``[row, col]`` physical offset within the GM ``src``.
    shapes: GM row window ``[r, c]`` (typically ``[1, size]``).
        Must be compile-time constant.
    valid_shape: How much of that window to actually transfer, defaulting to
        all of it. May hold runtime ``Scalar`` values, so a dynamic row count
        leaves the accumulator's allocation and layout untouched. Not
        supported together with ``transpose=True``.
    transpose: Place the GM row ``[r, c]`` as an L1 column ``[c, r]`` — use
        for a matmul B-operand whose consumer would otherwise need ``b_trans``.

Returns:
    Tensor aliasing ``acc`` (written in place).

## `scatter`

```text
pypto.language.tensor.scatter(input: pypto.language.typing.tensor.Tensor, dim: int | None = None, index: pypto.language.typing.tensor.Tensor | None = None, src: pypto.language.typing.tensor.Tensor | None = None, *, mask_pattern: int | None = None, dst: pypto.language.typing.tensor.Tensor | None = None) -> pypto.language.typing.tensor.Tensor
```

Scatter elements of ``src`` into ``input`` (tensor-level) — index or mask form.

The tensor layer exposes a single unified ``scatter``. Based on the arguments
you pass, it lowers to one of two tile-level ops:

Index form (``dim`` + ``index`` + ``src``) → [`pl.tile.scatter`][pypto.language.tile.scatter] — the
column-wise inverse of [`gather`][pypto.language.tensor.gather], so ``index`` has the same shape as
``src`` (just like gather's index matches its output)::

    output = input
    output[b, index[b, k]] = src[b, k]   # for all b, k

    MVP: rank-2 input with ``dim == -1``. ``src``/``index`` are ``[rows, K]``;
    ``input``/output are ``[rows, S]`` with ``K <= S``. ``index`` element
    width must match ``input``: 4-byte input → INT32, 2-byte → INT16,
    1-byte → INT16.

Mask form (``mask_pattern=<int>`` + ``dst``) → [`pl.tile.scatter_mask`][pypto.language.tile.scatter_mask]:
    Writes each row of ``input`` into the columns of ``dst`` selected by the
    hardware mask pattern. ``dst.cols`` equals ``input.cols * stride``
    (stride = 2 for P0101/P1010, 4 for P0001..P1000, 1 for P1111).
    Unlike the gather mask form (a real ``pto.tgather`` ISA op on A2/A3 and
    A5), mask-pattern scatter is not a distinct pto-isa instruction — PyPTO
    emits it as a ``pto.tscatter`` mask-form construct for A2/A3 / CPU-sim
    style lowering paths.

Args:
    input: Base tensor (FP16/FP32/BF16/INT8/INT16/INT32, 2D).
    dim: (index form) Axis along which to scatter. MVP accepts -1.
    index: (index form) Per-element destination column indices, same shape
        as ``src`` (INT16/INT32).
    src: (index form) Source values tensor (same dtype as ``input``).
    mask_pattern: (mask form, keyword-only) Mask pattern selector (1-7).
        1=P0101, 2=P1010, 3=P0001, 4=P0010, 5=P0100, 6=P1000, 7=P1111.
    dst: (mask form, keyword-only) Destination tensor; ``dst.cols ==
        input.cols * stride``.

Returns:
    Tensor representing the post-scatter result.

Examples:
    out = scatter(input, dim=-1, index=idx, src=src_vals)
    out = scatter(input, mask_pattern=pl.tile.MaskPattern.P0101, dst=dst)

## `alloc`

```text
pypto.language.tensor.alloc(memory_space: pypto.pypto_core.ir.MemorySpace, size: int) -> pypto.language.typing.ptr.Ptr
```

Stub for the internal ``tensor.alloc`` IR operation.

This function is never called in user-written DSL code. It is emitted
by the C++ python-printer after the InitMemRef pass and must be
importable so that the printed source is valid Python.

The result is a base ``Ptr`` (allocation identity token): the printer
annotates the assignment target as ``pl.Ptr``, matching the IR design
where ``tensor.alloc`` Calls carry ``PtrType``.

Args:
    memory_space: Space the allocation lives in, as resolved by InferTileMemorySpace.
    size: Allocation size in bytes.

Returns:
    A ``Ptr`` standing for the allocation, carrying no address of its own.

## `get_block_idx`

```text
pypto.language.tensor.get_block_idx() -> pypto.language.typing.scalar.Scalar
```

Get the current block index (tensor-scope alias of ``pl.tile.get_block_idx``).

Lowers to ``tile.get_block_idx`` in ``ConvertTensorToTileOps``.

Returns:
    Scalar wrapping the ``tensor.get_block_idx`` operation (INDEX type)

Example:
    >>> block_idx = pl.tensor.get_block_idx()

## `get_subblock_idx`

```text
pypto.language.tensor.get_subblock_idx() -> pypto.language.typing.scalar.Scalar
```

Get the current sub-block (vector core) index (tensor-scope alias of ``pl.tile.get_subblock_idx``).

Lowers to ``tile.get_subblock_idx`` in ``ConvertTensorToTileOps``.

Returns:
    Scalar wrapping the ``tensor.get_subblock_idx`` operation (INDEX type)

## `get_block_num`

```text
pypto.language.tensor.get_block_num() -> pypto.language.typing.scalar.Scalar
```

Get the total number of blocks in the current SPMD task.

Tensor-scope alias of ``pl.tile.get_block_num``; lowers to
``tile.get_block_num`` in ``ConvertTensorToTileOps``.

Returns:
    Scalar wrapping the ``tensor.get_block_num`` operation (INDEX type)

Example:
    >>> block_idx = pl.tensor.get_block_idx()
    >>> block_num = pl.tensor.get_block_num()
