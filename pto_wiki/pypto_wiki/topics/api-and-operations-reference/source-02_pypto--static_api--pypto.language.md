# `pypto.language` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `jit`

```text
pypto.language.jit(func: 'Any' = None, *, auto_scope: 'bool' = True) -> 'Any'
```

The ``pl.jit`` object.

Supports::

    @pl.jit                               # entry-point (Orchestration)
    @pl.jit.host                          # entry-point (HOST Orchestrator)
    @pl.jit.incore                        # InCore sub-function
    @pl.jit.incore(level=pl.Level.AIC)   # InCore with explicit level
    @pl.jit.inline                        # Inline sub-function (spliced at call site)
    @pl.jit.opaque                        # Opaque sub-function (separate IR function)

``host`` is the L3+ entry variant: it authors the per-rank dispatch loop
(``for r in pl.range(pld.world_size()): chip_orch(..., device=r)``) and
window-buffer allocation that today only ``@pl.function(level=HOST,
role=Orchestrator)`` inside ``@pl.program`` could express. It is keyed
off ``_func_type='host'`` and specializes into
``@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)``.

## `JITFunction`

```text
pypto.language.JITFunction(func: 'Any', func_type: 'str | None' = None, level: 'Any' = None, auto_scope: 'bool' = True, external_core_type: 'str | None' = None, external_aic_source: 'str | None' = None, external_aiv_source: 'str | None' = None, external_dual_aiv_dispatch: 'bool' = False, external_include_dirs: 'tuple[str, ...]' = ()) -> 'None'
```

A JIT-compiled function with shape specialization and caching.

Created by the ``@jit`` or ``@jit.incore`` decorators.

Attributes:
    _func: Original Python function.
    _func_type: 'orchestration' | 'host' | 'incore' | 'inline' | 'opaque'.
        ``'host'`` is the HOST-level orchestrator produced by
        ``@pl.jit.host`` — it owns ``pld.alloc_window_buffer`` /
        ``pld.window`` / ``pld.world_size()`` and the per-rank
        ``device=`` dispatch loop. End-to-end runtime dispatch works when
        the caller supplies ``config=RunConfig(distributed_config=...)``:
        the config is forwarded through ``_compile`` → ``ir.compile()``
        (see ``RunConfig.compile_kwargs``), which yields a
        ``DistributedCompiledProgram`` that ``__call__`` dispatches
        per-rank.
    _level: pl.Level or None.
    _auto_scope: Whether the compiler auto-inserts AUTO runtime scopes
        (SIMPLER_SCOPE) around the body and each for/if body. ``True`` by
        default; set ``False`` via ``@pl.jit(auto_scope=False)`` /
        ``@pl.jit.host(auto_scope=False)`` to place scopes by hand with
        ``with pl.scope()``. Also accepted on
        ``@pl.jit.inline(auto_scope=False)`` — after the ``InlineFunctions``
        pass splices the body, hand-placed scopes land in the caller.
        ``incore`` / ``opaque`` kinds reject it (they outline into
        separate kernels, so scopes never land in the caller).
    _dep_graph_state: Last resolved graph and its validating bindings,
        published together. Each call pins its own validated graph.
    _cache: L1 in-memory cache: CacheKey → CompiledProgram (post-pass ir.Program wrapped).

### `JITFunction.compile`

```text
JITFunction.compile(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```

Specialize + compile for the shape/dtype combination implied by *args*,
and return the underlying ``CompiledProgram``.

Same specialization / cache pipeline as ``__call__``, minus the
on-device dispatch. Use this when you want to drive execution through
the runtime worker API directly:

- ``pypto.runtime.ChipWorker.run`` / ``register``
  for explicit L2 dispatch.
- ``CompiledProgram.chip_callable`` / ``runtime_name`` / ``runtime_config``
  to drive a hand-constructed ``simpler.worker.Worker``.

``config=RunConfig(...)`` is still consumed (and its compile-side
knobs forwarded to ``ir.compile()``) so the returned
``CompiledProgram`` honours the same options as a direct
``kernel(*args, config=...)`` call. Runtime-side fields on the
``RunConfig`` (``device_id``, DFX flags, ...) do not apply here —
they affect dispatch, not the compiled artefact.

Subsequent calls (either ``__call__`` or [`compile`][pypto.language.JITFunction.compile]) with the
same specialization key hit the L1 cache and return the same
``CompiledProgram`` instance. Dump, compile-profiling, explicit output,
and custom pass-diagnostic requests always compile afresh, preserving
ordinary cached entries. Omitting ``config`` uses ``RunConfig`` defaults,
including disabled dumps.

**Compiling without tensors.** When called with **no tensor
arguments** — neither positional nor keyword — the shape/dtype contract
is read directly from the kernel's own parameter annotations, so no
throwaway ``torch.empty(...)`` dummies are needed. (Passing tensors by
keyword, e.g. ``compile(a=sample_a)``, still binds them normally.) This
requires every tensor parameter to carry a full ``pl.Tensor[[...],
dtype]`` annotation (a bare ``pl.Tensor`` has no shape to read and
raises). Dynamic dims (``pl.dynamic`` / ``bind_dynamic``) need no value —
the artifact is extent-independent. Scalar parameters have no value in
the signature, so pass them as keyword args (or via a signature
default); a literal **specializes** that value into the artifact, while
``pl.RUNTIME`` leaves the parameter **unspecialized** — it stays a real
``pl.Scalar`` parameter supplied at dispatch and, like a dynamic dim,
drops out of the cache key. A signature-mode call shares a cache entry
with an equivalent ``compile(*sample_tensors)`` call whenever the two
agree on every specialized scalar; ``pl.RUNTIME`` is its own
specialization and is rejected on the sample-argument path, which always
specializes the scalar value it is handed.

Example::

    M = pl.dynamic("M")

    @pl.jit
    def my_kernel(
        x: pl.Tensor[[M, 4096], pl.BF16],
        w: pl.Tensor[[4096, 4096], pl.BF16],
        out: pl.Out[pl.Tensor[[M, 4096], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        ...

    worker = ChipWorker(config=RunConfig(platform="a2a3"))

    # From sample tensors (shape/dtype read; contents ignored):
    compiled = my_kernel.compile(sample_x, sample_w, sample_out, 128)

    # Or straight from the (fully-annotated) signature — no tensors.
    # num_tokens varies per launch, so keep it out of the artifact: its
    # value is supplied on each dispatch through the compiled artifact
    # (below), not by calling my_kernel(...) directly — an eager call
    # re-specializes and compiles a separate artifact.
    compiled = my_kernel.compile(num_tokens=pl.RUNTIME)

    w_dev = worker.alloc_tensor(real_w.shape, real_w.dtype, init=real_w)
    h = worker.register(compiled)
    for batch in stream:
        h(batch.x, w_dev, batch.out, batch.num_tokens)

Args:
    *args: Positional arguments matching the decorated function's
        params. Tensor values are inspected for shape/dtype only; their
        contents are not read. Omit **all** positional args to compile
        straight from the signature annotations instead.
    **kwargs: Keyword arguments. A ``config`` keyword, if present, is
        a ``RunConfig``. In signature mode,
        scalar parameter values are also passed here (by name) — a
        literal to specialize it, or ``pl.RUNTIME`` to leave it
        unspecialized.

Returns:
    The cached ``CompiledProgram`` for this specialization.

### `JITFunction.specialize`

```text
JITFunction.specialize(self, *args: 'Any', **kwargs: 'Any') -> '_ir.Program'
```

Specialize this JIT function and return its **pre-pass** IR.

The step [`lower`][pypto.language.JITFunction.lower] takes before it
runs the pass pipeline: entry and every transitive dep are specialized
into ``@pl.program`` source and parsed, and the parsed program is
returned untransformed. No passes, no code generation, no ``ptoas``, no
device, and the compiled-program cache is neither read nor written.

Use this to hand a JIT kernel to a consumer that wants to drive the
pass pipeline itself — most notably ``ir.compile(program,
output_dir=...)``, which runs passes *and* code generation and so must
be given the program before any pass has touched it. Passing
``lower()``'s result there would run the pipeline a second time.

Two JIT kernels that specialize to the same program compare equal
*after* passes, not before: the specializer renames SSA-rebound locals
(``out`` becomes ``out_v1``), which canonicalization removes. Compare
``lower()`` output when asserting equivalence against a hand-written
``@pl.program``.

Args:
    *args: Positional sample arguments matching the decorated function.
        Omit tensor samples to specialize from the annotations, which
        then must carry full shapes.
    **kwargs: Keyword sample arguments. Unlike ``lower()`` there is no
        ``config``: the pass pipeline never runs here, so a
        ``RunConfig`` would have nothing to configure.

Returns:
    The parsed ``ir.Program``, before any pass has run.

Raises:
    TypeError: ``config=`` was passed. Accepting it silently would let
        a caller believe a strategy or diagnostics setting shaped the
        returned IR, when no pass ran to read it.

### `JITFunction.lower`

```text
JITFunction.lower(self, *args: 'Any', **kwargs: 'Any') -> '_ir.Program'
```

Specialize this JIT function and return its post-pass IR.

A ``config=RunConfig(...)`` keyword controls pass execution through its
strategy, diagnostics, dependency-analysis, memory-planner, and platform
fields. Runtime and artifact fields are ignored. This method does not
run code generation, invoke ``ptoas``, execute on a device, write
artifacts, or access the compiled-program cache.

Args:
    *args: Positional sample arguments matching the decorated function.
        Omit tensor samples to specialize from fully shaped annotations.
    **kwargs: Keyword sample arguments and an optional ``config``. In
        signature mode a scalar parameter takes a literal (specialized
        into the IR) or ``pl.RUNTIME`` (left unspecialized).

Returns:
    The specialized ``ir.Program`` after configured passes.

## `function`

```text
pypto.language.function(func: collections.abc.Callable[..., typing.Any] | None = None, *, type: pypto.pypto_core.ir.FunctionType = FunctionType.Opaque, level: pypto.pypto_core.ir.Level | None = None, role: pypto.pypto_core.ir.Role | None = None, attrs: dict[str, typing.Any] | None = None, auto_scope: bool = True, strict_ssa: bool = False, external_source: str | pathlib.Path | None = None) -> pypto.pypto_core.ir.Function | collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], pypto.pypto_core.ir.Function]
```

Decorator that parses a DSL function and returns IR Function.

This decorator analyzes the decorated function's AST, parses the DSL
constructs (type annotations, pl.range, pl.yield_, etc.), and builds
an IR Function object.

Args:
    func: Python function decorated with @pl.function
    type: Function type (Opaque, Orchestration, or InCore)
    level: Hierarchy level (e.g. pl.Level.HOST)
    role: Function role (e.g. pl.Role.SubWorker)
    attrs: Function-level attributes dict (e.g. {"split": pl.SplitMode.UP_DOWN})
    auto_scope: If True (default), the compiler inserts AUTO runtime scopes
               (SIMPLER_SCOPE) around the function body and each for/if body.
               Set False to place scopes by hand with ``with pl.scope()``
               (only meaningful for Orchestration functions).
    strict_ssa: If True, enforce SSA (single assignment per variable).
               If False (default), allow variable reassignment (non-SSA mode).
    external_source: Path to a hand-written C++ kernel ``.cpp`` that backs
               this function. Only valid on ``FunctionType.AIC`` /
               ``FunctionType.AIV`` functions, whose body must be a bare
               ``...`` (signature only). The orchestration calls the kernel
               as usual, but the compiler skips PyPTO codegen for it and
               compiles the referenced source as the InCore kernel instead.
               Relative paths resolve against the defining file's directory.

Returns:
    IR Function object (or decorator if used with parameters)

Example:
    >>> @pl.function
    ... def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
    ...     result = pl.create_tensor([64, 128], dtype=pl.FP32)
    ...     return result
    >>> @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
    ... def sub_worker(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...     return x

## `inline`

```text
pypto.language.inline(func: collections.abc.Callable) -> pypto.language.parser.decorator.InlineFunction
```

Decorator that captures a function for inlining at call sites.

Unlike @pl.function which parses to an ir.Function immediately,
@pl.inline defers parsing until the function is called within a
@pl.program. The body is expanded in-place at each call site.

Args:
    func: Python function to capture for inlining

Returns:
    InlineFunction object with captured AST and metadata

Example:
    >>> @pl.inline
    ... def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...     result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    ...     return result

## `program`

```text
pypto.language.program(cls: type | None = None, *, strict_ssa: bool = False) -> pypto.pypto_core.ir.Program | collections.abc.Callable[[type], pypto.pypto_core.ir.Program]
```

Decorator that parses a class with @pl.function methods into a Program.

The class should contain one or more methods decorated with @pl.function.
Each method is parsed as a separate function and added to the program.
Methods must have 'self' as the first parameter (standard Python syntax),
which is automatically stripped from the IR.

Args:
    cls: Class with @pl.function decorated methods
    strict_ssa: If True, enforce SSA (single assignment per variable).
               If False (default), allow variable reassignment (non-SSA mode).

Returns:
    IR Program object (or decorator if used with parameters)

Example:
    >>> @pl.program
    ... class MyProgram:
    ...     @pl.function
    ...     def add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...         result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    ...         return result
    ...
    ...     @pl.function
    ...     def mul(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...         result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    ...         return result
    >>> # MyProgram is now an ir.Program object

## `InlineFunction`

```text
pypto.language.InlineFunction(name: str, func_def: ast.FunctionDef, param_names: list[str], source_file: str, source_lines: list[str], line_offset: int, col_offset: int, closure_vars: dict[str, typing.Any]) -> None
```

Stores AST and metadata for a function to be inlined at call sites.

## `parse`

```text
pypto.language.parse(code: str, filename: str = '<string>', source_map: dict[int, tuple[str, int, int]] | None = None) -> pypto.pypto_core.ir.Function | pypto.pypto_core.ir.Program
```

Parse a DSL function or program from a string.

This function takes Python source code containing a @pl.function decorated
function or @pl.program decorated class and parses it into an IR Function
or Program object. The code is executed dynamically, automatically importing
pypto.language as pl if not already present.

Args:
    code: Python source code containing @pl.function or @pl.program
    filename: Optional filename for error reporting (default: "<string>")
    source_map: Optional ``generated_line → (orig_file, orig_line, orig_col)``
        map. When provided, spans whose emitted line is present are remapped
        to the original source location, so diagnostics point at the user's
        real file rather than this (possibly generated) ``code``. Used by
        ``@pl.jit`` to recover provenance through its specialize→reparse
        round-trip (issue #1612).

Returns:
    Parsed ir.Function or ir.Program object (auto-detected)

Raises:
    ValueError: If the code contains nothing to parse or multiple items
    ParserError: If parsing fails (syntax errors, type errors, etc.)

Warning:
    This function uses `exec()` to execute the provided code string.
    It should only be used with trusted input, as executing untrusted
    code can lead to arbitrary code execution vulnerabilities.

Examples:
    >>> import pypto.language as pl

    >>> # Parse a function
    >>> func_code = '''
    ... @pl.function
    ... def add(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...     result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
    ...     return result
    ... '''
    >>> func = pl.parse(func_code)
    >>> print(func.name)
    add

    >>> # Parse a program
    >>> prog_code = '''
    ... @pl.program
    ... class MyProgram:
    ...     @pl.function
    ...     def add(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...         return pl.add(x, 1.0)
    ... '''
    >>> prog = pl.parse(prog_code)
    >>> print(prog.name)
    MyProgram

## `loads`

```text
pypto.language.loads(filepath: str) -> pypto.pypto_core.ir.Function | pypto.pypto_core.ir.Program
```

Load a DSL function or program from a file.

This function reads a Python file containing a @pl.function decorated
function or @pl.program decorated class and parses it into an IR Function
or Program object (auto-detected).

Args:
    filepath: Path to Python file containing @pl.function or @pl.program

Returns:
    Parsed ir.Function or ir.Program object (auto-detected)

Raises:
    FileNotFoundError: If the file does not exist
    ValueError: If the file contains nothing to parse or multiple items
    ParserError: If parsing fails (syntax errors, type errors, etc.)

Warning:
    This function reads a file and executes its contents. It should only
    be used with trusted files, as executing code from untrusted sources
    can lead to arbitrary code execution vulnerabilities.

Examples:
    >>> import pypto.language as pl

    >>> # Load a function
    >>> func = pl.loads('my_kernel.py')
    >>> print(func.name)

    >>> # Load a program
    >>> prog = pl.loads('my_program.py')
    >>> print(prog.name)

## `parse_program`

```text
pypto.language.parse_program(code: str, filename: str = '<string>') -> pypto.pypto_core.ir.Program
```

Parse a DSL program from a string.

!!! warning "Deprecated"
    Use [`parse`][pypto.language.parse] instead, which auto-detects functions
    and programs.

This is now an alias for [`parse`][pypto.language.parse] that validates the
result is a Program.

Args:
    code: Python source code containing a @pl.program decorated class
    filename: Optional filename for error reporting (default: "<string>")

Returns:
    Parsed ir.Program object

Raises:
    ValueError: If the code contains a function instead of a program
    ParserError: If parsing fails (syntax errors, type errors, etc.)

## `loads_program`

```text
pypto.language.loads_program(filepath: str) -> pypto.pypto_core.ir.Program
```

Load a DSL program from a file.

!!! warning "Deprecated"
    Use [`loads`][pypto.language.loads] instead, which auto-detects functions
    and programs.

This is now an alias for [`loads`][pypto.language.loads] that validates the
result is a Program.

Args:
    filepath: Path to Python file containing @pl.program decorated class

Returns:
    Parsed ir.Program object

Raises:
    FileNotFoundError: If the file does not exist
    ValueError: If the file contains a function instead of a program
    ParserError: If parsing fails (syntax errors, type errors, etc.)

## `Tensor`

```text
pypto.language.Tensor(*args: Any, **kwargs: Any) -> ~TensorT
```

Tensor type for PyPTO Language DSL.

This class serves dual purposes:
1. Type annotation helper for function signatures
2. Runtime wrapper around IR Expr/Call objects

Annotation mode (used in type hints):
    x: pl.Tensor[[64, 128], pl.FP16]
    y: pl.Tensor[[64, 128], pl.FP16, pl.NZ]

Runtime mode (wraps IR expressions):
    tensor = pl.create_tensor([64, 128], dtype=pl.FP32)
    # Returns Tensor wrapping the Call expression

Examples:
    >>> import pypto.language as pl
    >>>
    >>> @pl.function
    ... def my_func(x: pl.Tensor[[64, 128], pl.FP16, pl.NZ]) -> pl.Tensor[[64, 128], pl.FP32]:
    ...     result: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
    ...     return result

### `Tensor.unwrap`

```text
Tensor.unwrap(self) -> pypto.pypto_core.ir.Expr
```

Get underlying IR expression.

Returns:
    The wrapped Expr/Call object

Raises:
    ValueError: If called on an annotation-only Tensor

### `Tensor.bind_dynamic`

```text
Tensor.bind_dynamic(self, dim: int, var: Any) -> None
```

Mark a tensor dimension as runtime-dynamic for @pl.jit specialization.

This is a no-op at runtime.  The @pl.jit specializer reads this call
statically from the AST to determine which dimensions should be
represented as DynVar nodes (ir.Var) in the generated type annotation
rather than as compile-time constants.

Use ``bind_dynamic`` with bare ``pl.Tensor`` parameters, where the
annotation carries no shape to hold the DynVar.  When a parameter is
already annotated with an explicit shape, prefer placing the
``pl.dynamic()`` variable directly in the annotation instead — that
form matches ``@pl.program`` and needs no ``bind_dynamic`` call::

    M = pl.dynamic("M")

    @pl.jit
    def kernel(
        a: pl.Tensor[[M, 128], pl.FP32],          # dim 0 dynamic via annotation
        c: pl.Out[pl.Tensor[[M, 128], pl.FP32]],  # shares the same DynVar
    ):
        K = pl.tensor.dim(a, 1)                    # dim 1 stays constant (128)
        ...

Both forms are honoured and may be combined; the dynamic dimensions
are the union of the two sources.

Args:
    dim: Zero-based dimension index to mark as dynamic.
    var: The DynVar object (created with pl.dynamic()) to bind.

Example::

    @pl.jit
    def kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
        M = pl.dynamic("M")
        a.bind_dynamic(0, M)   # dim 0 of a is runtime-dynamic
        c.bind_dynamic(0, M)   # dim 0 of c shares the same DynVar
        K = a.shape[1]         # dim 1 is compile-time constant
        ...

## `Tile`

```text
pypto.language.Tile(shape=None, dtype=None, expr: pypto.pypto_core.ir.Expr | None = None, memref: 'MemRef | None' = None, memory_space: 'MemorySpace | None' = None, tile_view: 'TileView | None' = None, _annotation_only: bool = False) -> 'Tile'
```

Tile type for PyPTO Language DSL.

Tile represents a tile in unified buffer (UB) memory. It is used for
tile-level programming with operations like load, store, add, mul, etc.

Annotation mode (used in type hints):
    x: pl.Tile[[64, 64], pl.FP32]

Runtime mode (wraps IR expressions):
    tile = pl.load(tensor, [0, 0], [64, 64])
    # Returns Tile wrapping the Call expression

Examples:
    >>> import pypto.language as pl
    >>>
    >>> @pl.function
    ... def my_func(input: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    ...     tile: pl.Tile[[64, 64], pl.FP32] = pl.load(input, [0, 0], [64, 64])
    ...     result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
    ...     return pl.store(result, [0, 0], input)

### `Tile.unwrap`

```text
Tile.unwrap(self) -> pypto.pypto_core.ir.Expr
```

Get underlying IR expression.

Returns:
    The wrapped Expr/Call object

Raises:
    ValueError: If called on an annotation-only Tile

## `Scalar`

```text
pypto.language.Scalar(*args: Any, **kwargs: Any) -> 'Scalar'
```

Scalar type for PyPTO Language DSL.

This class serves dual purposes:
1. Type annotation helper for function signatures
2. Runtime wrapper around IR Expr/Call objects

Annotation mode (used in type hints):
    x: pl.Scalar[pl.FP32]
    count: pl.Scalar[pl.INT32]

Runtime mode (wraps IR expressions):
    scalar_value = pl.scalar.create(3.14, dtype=pl.FP32)
    # Returns Scalar wrapping the Call expression

Examples:
    >>> import pypto.language as pl
    >>>
    >>> @pl.function
    ... def add_scalar(
    ...     x: pl.Tensor[[64], pl.FP32],
    ...     scalar: pl.Scalar[pl.FP32]
    ... ) -> pl.Tensor[[64], pl.FP32]:
    ...     result: pl.Tensor[[64], pl.FP32] = pl.add(x, scalar)
    ...     return result

### `Scalar.unwrap`

```text
Scalar.unwrap(self) -> pypto.pypto_core.ir.Expr
```

Unwrap to get the underlying IR expression.

Returns:
    The wrapped IR expression

Raises:
    RuntimeError: If this is an annotation-only instance

## `Array`

```text
pypto.language.Array(*args: Any, **kwargs: Any) -> 'Array'
```

On-core array wrapper.

Annotation mode (used in type hints — rare; Arrays don't cross function
boundaries in v1, but the annotation is available for clarity and
forward-compat)::

    arr: pl.Array[16, pl.INT32]

Runtime mode (the common path)::

    arr = pl.array.create(16, pl.INT32)
    arr[i] = value       # -> array.update_element (functional, rebinds arr)
    x = arr[i]           # -> array.get_element

### `Array.unwrap`

```text
Array.unwrap(self) -> pypto.pypto_core.ir.Expr
```

## `Tuple`

```text
pypto.language.Tuple(*args: Any, **kwargs: Any) -> 'Tuple'
```

Tuple type for PyPTO Language DSL.

Used exclusively as a type annotation helper in function signatures.
The parser reads the AST (not the runtime value), so this class only
needs to make ``pl.Tuple[T1, T2, ...]`` evaluate without error.

Annotation syntax:
    result: pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.INT32]]

Examples:
    >>> import pypto.language as pl
    >>>
    >>> @pl.function
    ... def multi_return(
    ...     x: pl.Tensor[[64], pl.FP32]
    ... ) -> pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.INT32]]:
    ...     ...

## `DynVar`

```text
pypto.language.DynVar(*args: Any, **kwargs: Any) -> 'Scalar'
```

Dynamic shape variable for use in type annotations.

Creates a symbolic dimension that becomes an ir.Var node in the IR shape.
Inherits from Scalar so that DynVar is accepted wherever Scalar/IntLike
is expected (e.g. shape parameters, TensorView valid_shape).

Example:
    M = pl.dynamic("M")
    N = pl.dynamic("N")

    @pl.function
    def func(a: pl.Tensor[[M, N], pl.FP32]) -> ...:
        ...

### `DynVar.unwrap`

```text
DynVar.unwrap(self) -> pypto.pypto_core.ir.Expr
```

Return the underlying ir.Var, creating it eagerly if needed.

This allows DynVar to participate in _normalize_intlike() and other
Scalar-consuming paths without requiring prior TypeResolver resolution.

## `InOut`

```text
pypto.language.InOut()
```

Wrapper for InOut parameter direction in type annotations.

Usage::

    def kernel(output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]) -> ...:

## `IntLike`

```text
pypto.language.IntLike
```

## `Out`

```text
pypto.language.Out()
```

Wrapper for Out parameter direction in type annotations.

Usage::

    def kernel(result: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> ...:

## `RUNTIME`

```text
pypto.language.RUNTIME
```

## `dynamic`

```text
pypto.language.dynamic(name: str) -> pypto.language.typing.dynamic.DynVar
```

Create a dynamic shape variable for type annotations.

Args:
    name: Variable name for the dynamic dimension

Returns:
    DynVar that can be used in shape annotations

## `const`

```text
pypto.language.const(value: 'int | float', dtype: 'Any') -> 'Scalar'
```

Create a typed constant with an explicit dtype.

Used by the printer to preserve non-default constant dtypes in round-trip.
The parser intercepts pl.const() calls and creates ConstInt/ConstFloat
with the specified dtype.

Statically typed ``-> Scalar`` to mirror its IR semantics: ``pl.const``
builds a ``ConstInt``/``ConstFloat`` expression, so it can be returned
from a ``-> pl.Scalar`` function and combined with other scalars. At
runtime the stub returns the numeric value unchanged (``cast`` is a no-op).

Args:
    value: Numeric value (int or float)
    dtype: DataType for the constant

Returns:
    Parser builds ``ConstInt``/``ConstFloat`` IR; the runtime stub returns
    the numeric value unchanged (type checking sees a ``Scalar``).

## `range`

```text
pypto.language.range(*args: 'RangeArg', init_values: 'tuple[Any, ...] | None' = None) -> 'RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]'
```

Create a range iterator for for loops.

Supports several patterns:
    Simple:        for i in pl.range(10):
    Iter args:     for i, (var1, var2) in pl.range(16, init_values=(init1, init2)):

For software pipelining (body replication for ping-pong buffering), use
``pl.pipeline(N, stage=F)`` instead — it is a sibling loop iterator.

Args can be int literals or Scalar variables:
    for i in pl.range(n):  # n is pl.Scalar[pl.INT64]
    for i in pl.range(0, n, 1):
    for i in pl.range(n * 2 + 1):

Args:
    *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
        Each argument can be an int literal or a pl.Scalar value.
    init_values: Initial values for iteration arguments

Returns:
    If no init_values: RangeIterator yielding loop variable (Scalar)
    If init_values: RangeIterator yielding (loop_var, (iter_args...))

## `parallel`

```text
pypto.language.parallel(*args: 'RangeArg', init_values: 'tuple[Any, ...] | None' = None) -> 'RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]'
```

Create a parallel range iterator for parallel for loops.

Behaves identically to range() at runtime. The distinction is used by the
parser to emit ForKind.Parallel instead of ForKind.Sequential.

Args:
    *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
        Each argument can be an int literal or a pl.Scalar value.
    init_values: Initial values for iteration arguments

Returns:
    If no init_values: RangeIterator yielding loop variable (Scalar)
    If init_values: RangeIterator yielding (loop_var, (iter_args...))

## `unroll`

```text
pypto.language.unroll(*args: 'RangeArg') -> 'RangeIterator[Scalar]'
```

Create an unroll range iterator for compile-time loop unrolling.

Behaves identically to range() at runtime. The distinction is used by the
parser to emit ForKind.Unroll instead of ForKind.Sequential.

Unrolled loops do not support init_values (loop-carried state).

Args:
    *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
        Each argument must be an int literal (compile-time constant).

Returns:
    RangeIterator yielding loop variable (Scalar)

Examples:
    >>> for i in pl.unroll(4):
    ...     x = pl.add(x, 1.0)
    >>> for i in pl.unroll(0, 6, 2):
    ...     x = pl.add(x, i)

## `pipeline`

```text
pypto.language.pipeline(*args: 'RangeArg', stage: 'int', init_values: 'tuple[Any, ...] | None' = None) -> 'RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]'
```

Create a software-pipelined loop iterator.

Replicates the loop body ``stage`` times per outer iteration to enable
ping-pong buffering. The outer loop advances in strides of ``stage * step``;
a tail dispatch covers the remainder when the trip count is not divisible
by ``stage``. Lowered by the ``LowerPipelineLoops`` pass at the tile level.

Positional args match ``pl.range``: (stop) / (start, stop) / (start, stop, step).
The ``stage`` kwarg is required and must be a positive integer.

Args:
    *args: 1-3 positional args — same shape as ``pl.range()``.
    stage: Pipeline depth (positive integer, typically 2-4).
    init_values: Loop-carried state, same semantics as ``pl.range()``.

Examples:
    >>> for i in pl.pipeline(64, stage=4):
    ...     tile = pl.tile.load(...)
    ...     pl.tile.store(...)

## `while_`

```text
pypto.language.while_(*, init_values: 'tuple[ExprType, ...] | None' = None) -> 'WhileIterator[tuple[ExprType, ...]]'
```

Create a while iterator for while loops.

Always requires init_values to specify loop-carried state.
The loop condition must be specified as the first statement in the loop body using pl.cond().

Pattern:
    for (var1, var2) in pl.while_(init_values=(init1, init2)):
        pl.cond(condition)
        # loop body
        var1_out, var2_out = pl.yield_(var1_updated, var2_updated)

Args:
    init_values: Initial values for iteration arguments (required)

Returns:
    WhileIterator yielding tuple of iter_args

Raises:
    ValueError: If init_values is not provided

Examples:
    >>> for (x,) in pl.while_(init_values=(0,)):
    ...     pl.cond(x < 10)
    ...     x = x + 1
    ...     x_out = pl.yield_(x)
    >>>
    >>> for (x, y) in pl.while_(init_values=(0, 1)):
    ...     pl.cond(x < n)
    ...     x_new = x + 1
    ...     y_new = y * 2
    ...     x_out, y_out = pl.yield_(x_new, y_new)

## `yield_`

```text
pypto.language.yield_(*values: 'Any') -> 'Any | tuple[Any, ...]'
```

Yield values from a scope (for, if).

This function is used to explicitly return values from nested scopes
and create SSA phi nodes.

Args:
    *values: Values to yield

Returns:
    The yielded value(s). For single value, returns the value.
    For multiple values, returns tuple.

Examples:
    >>> # Single value yield
    >>> result = pl.yield_(x + 1)
    >>>
    >>> # Multiple value yield
    >>> a, b = pl.yield_(x, y)

## `cond`

```text
pypto.language.cond(condition: 'CondArg') -> 'None'
```

Specify the condition for a pl.while_() loop.

This function must be the first statement in a pl.while_() loop body.
It is purely syntactic - the parser extracts the condition and sets it on the WhileStmt.

Args:
    condition: While loop condition (bool literal or Scalar variable)

Examples:
    >>> for (x,) in pl.while_(init_values=(0,)):
    ...     pl.cond(x < 10)
    ...     x = x + 1
    ...     x_out = pl.yield_(x)

## `static_print`

```text
pypto.language.static_print(*args: 'Any') -> 'None'
```

Print compile-time information about IR objects.

At parse time, prints type/value info to stdout. At runtime, no-op.

Args:
    *args: Values to print (variables, expressions, string labels)

## `static_assert`

```text
pypto.language.static_assert(condition: 'Any', msg: 'str' = '') -> 'None'
```

Assert a condition at compile time (parse time).

At parse time, evaluates ``condition``. If false, raises ``ParserError``.
At runtime, this is a no-op (all semantics are handled by the parser).

Notes:
    * This is a **statement-only** construct. It must be used as a
      standalone statement, not as part of an expression.
    * The ``msg`` argument must be a **string literal** at the call site.
      Passing a variable or expression for ``msg`` will raise
      ``ParserSyntaxError``.
    * The check is evaluated at parse time only; it does not run at
      execution time.

Args:
    condition: Condition to check (must be compile-time evaluable)
    msg: Optional error message as a string literal

## `func_attr`

```text
pypto.language.func_attr(attrs: 'dict[str, Any]') -> 'None'
```

Attach function-level attributes from inside the function body.

Written as the **first statement** of a function body, ``pl.func_attr``
declares metadata about the function as a whole::

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32],
               w: pl.Tensor[[64, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        pl.func_attr({"stationary": w})
        ...

Why the body and not the decorator: a decorator is evaluated *before* the
signature binds any name, so ``@pl.function(attrs={"stationary": w})``
cannot be written — ``w`` does not exist yet. Body position places the
declaration after the parameters are bound, which is what makes an
attribute that references a parameter expressible at all.

Semantics:

* **Prologue only.** Every ``pl.func_attr`` call must precede every other
  statement in the body. An attribute describes the whole function, so it
  must not appear to start applying partway down a body; pinning it to the
  prologue also keeps the printed form deterministic. Consequently only
  parameters — not body-locals — are referenceable.
* **A bare name is always a parameter reference.** ``pl.func_attr({"n": k})``
  records the parameter ``k``, never the value of an enclosing Python
  variable named ``k``. Write Python-level constants as literals.
* **Multiple calls merge.** A key repeated across two calls — or between
  ``pl.func_attr`` and a decorator ``attrs=`` — is a ``ParserSyntaxError``
  naming the key.
* **Consumed at parse time.** The dict lands in ``Function.attrs`` and emits
  no IR statement of its own.

Attributes the parser must read *before* it can parse the body stay on the
decorator as dedicated keywords: ``auto_scope=`` and ``external_source=``.

Args:
    attrs: Attribute dict. Keys must be string literals.

## `at`

```text
pypto.language.at(level: 'ir.Level', role: 'ir.Role | None' = None, *, optimizations: 'list[Optimization] | None' = None, deps: 'list[Any] | None' = None, no_dep_args: 'list[Any] | None' = None, dumps: 'list[Any] | None' = None, allow_early_resolve: 'bool' = False, predicate: 'Any' = None, name_hint: 'str' = '', windowize: 'bool' = False) -> 'AtContext'
```

Mark a region of code for execution at a specific hierarchy level.

With ``level=pl.Level.CORE_GROUP``, the ``optimizations=`` list controls
the resulting scope kind:

- no entries → ``ScopeStmt(InCore)``
- ``pl.split(mode)`` → ``ScopeStmt(InCore, split=mode)``

For all other levels, this creates a Hierarchy scope.

Args:
    level: Target hierarchy level (e.g. pl.Level.HOST, pl.Level.CORE_GROUP).
    role: Function role (Orchestrator or Worker). Default: None.
    optimizations: Optional list literal of optimization entries. Each
        entry must be ``pl.split(mode)`` — written inline at the call
        site, since the DSL parser inspects the AST and does not accept
        dynamically built variables here.
    deps: Optional explicit producer-edge list (TaskId Vars and/or
        ``None`` sentinels). Lowered to the resulting Call's
        ``manual_dep_edges`` attr, which codegen packs into a
        ``set_dependencies(...)`` invocation. Operates at the *TaskId*
        level — orthogonal to ``no_dep_args=``, which operates at the
        *arg-slot* level on captured tensors.
    no_dep_args: Optional list literal of outer-scope tensor names
        captured by the scope body. Each entry must be a bare tensor
        name; the parser resolves it to a Var, the outliner translates
        the Var list into positional indices into the synthesised
        Call's args, and ``DeriveCallDirections`` overwrites those
        slots to ``ArgDirection.NoDep``. Equivalent to wrapping the
        same tensor with ``pl.no_dep(t)`` at an explicit kernel call
        site — use this form when the kernel call is synthesised by
        the ``pl.at`` outliner and there is no syntactic call-arg slot
        to wrap. Legal for both read-only captures and captures that
        the scope body mutates via ``pl.assemble`` / ``pl.store`` (the
        outliner classifies the latter as ``InOut`` on the synthesised
        kernel, and ``NoDep`` overrides ``InOut`` just as it overrides
        ``Input``); the user is asserting that sibling fan-outs touch
        disjoint regions of the tensor and therefore do not need
        OverlapMap dep tracking. Note: ``deps=`` takes TaskIds, while
        ``no_dep_args=`` takes tensors — they describe different things.
    dumps: Optional list literal of outer-scope tensor names to mark for
        selective tensor dump on the synthesised kernel dispatch. The
        scope-level selective-dump surface, symmetric with ``deps=``: the
        outliner translates the captured-tensor entries into the dispatch's
        ``dump_vars`` by Var identity. Equivalent to declaring the same
        tensors with ``pl.dump_tag(t)`` before the scope; use ``dumps=``
        when you want the dump targets listed explicitly at the scope.
    allow_early_resolve: Opt the outlined dispatch in as a speculative
        early-dispatch producer (simpler#1065). Same hint as
        ``pl.submit(..., allow_early_resolve=True)``: the scheduler may
        pre-stage this task's consumers before it completes. Forces the
        scope to lower to an ``ir.Submit`` (even without ``as tid``) so the
        flag can ride to codegen, where it emits
        ``Arg::set_allow_early_resolve(true)``. Pure scheduling hint.
    predicate: Optional **dispatch predicate** — a single comparison of one
        tensor element against an integer literal, e.g.
        ``predicate=(row_count[e] > 0)``. Same surface, validation and
        lowering as ``pl.submit(..., predicate=...)`` / ``pl.spmd(...,
        predicate=...)``: the scheduler evaluates it at the dispatch point
        (after this scope's dependencies are satisfied, so the value is
        current) and retires the task inline — never dispatching it to a
        core — when it is false, while still settling fanin/fanout so
        downstream consumers unlock. The comparison is parsed but **never
        evaluated** in orchestration; reading it there would stall on
        ``wait_for_tensor_ready``, exactly what the predicate avoids. Like
        ``allow_early_resolve`` it forces the scope to lower to an
        ``ir.Submit`` (even without ``as tid``).

        **Only valid with ``level=pl.Level.CORE_GROUP``** and only on a
        scope that is not nested inside ``pl.cluster()`` / ``pl.spmd()`` /
        another ``pl.at(level=pl.Level.CORE_GROUP)``. Every other placement
        either never lowers to a ``Submit`` (Hierarchy scopes are not
        outlined into a task dispatch) or is folded into an enclosing
        wrapper dispatch, so the predicate would be silently dropped; the
        parser rejects those instead.

        **Contract:** the operand tensor's producing task must be one of
        ``deps=`` — otherwise the predicate may read a stale value. Unlike
        ``pl.spmd``, ``deps=`` is available on every ``pl.at`` form. The
        parser makes a best-effort check (see ``pl.submit``); getting
        ``deps=`` right remains the author's responsibility.
    name_hint: Optional name hint for the outlined function (must be a
        valid identifier).
    windowize: Explicitly allow local windowization for the outlined InCore
        kernel. The default is False.

Returns:
    Context manager for the appropriate scope.

Examples:
    >>> # InCore scope:
    >>> with pl.at(level=pl.Level.CORE_GROUP):
    ...     y = pl.ops.add(x, x)

    >>> # InCore scope with split hint:
    >>> with pl.at(level=pl.Level.CORE_GROUP,
    ...            optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
    ...     y = pl.ops.add(x, x)

    >>> # Hierarchy scope (unchanged behavior):
    >>> with pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker):
    ...     y = pl.add(x, x)

    >>> # Conditionally dispatched InCore scope — the scheduler reads
    >>> # rc[0, 0] at the dispatch point and skips the task when it is 0:
    >>> with pl.at(level=pl.Level.CORE_GROUP) as gate_tid:
    ...     rc = pl.store(pl.load(rc, [0, 0], [128, 128]), [0, 0], rc)
    >>> with pl.at(level=pl.Level.CORE_GROUP,
    ...            deps=[gate_tid], predicate=(rc[0, 0] > 0)) as tid:
    ...     out = pl.store(pl.load(x, [0, 0], [128, 128]), [0, 0], out)

## `cluster`

```text
pypto.language.cluster(*, name_hint: 'str' = '') -> 'ClusterContext'
```

Mark a region of code as belonging to a Cluster execution context.

A cluster groups co-scheduled AIC (Cube) and AIV (Vector) kernels that
share the same physical cluster resources. The OutlineClusterScopes pass
extracts Cluster scopes into separate Group-typed functions.

Args:
    name_hint: Optional name hint for the outlined function.

Returns:
    Context manager for Cluster scope

Examples:
    >>> with pl.cluster():
    ...     with pl.at(level=pl.Level.CORE_GROUP):
    ...         y = pl.add(x, x)

## `graph`

```text
pypto.language.graph(name: 'str') -> 'GraphContext'
```

Mark a repeated region of orchestration as one recordable graph.

Under ``runtime="host_build_graph"`` the runtime records the region's task
topology the first time it executes and replays the recording on every later
execution. A decoder stack of N structurally identical layers therefore costs
one graph build instead of N, and occupies N outer task slots instead of
N x (tasks per layer).

``pl.graph`` is the in-place form of ``@pl.jit.graph``: ``OutlineGraphScopes``
extracts the region into a Graph function named after ``name``, so the two
surfaces compile to the same thing. Reach for the scope form when the region
is a slice of a larger orchestration body that you would rather not split into
a separate function; reach for the decorator when the layer is already its own
function.

The recorded topology is fixed after the first execution: only tensor
addresses and boundary scalars are refreshed per replay. ``LegalizeGraphBoundary``
proves that statically and rejects at compile time whatever it cannot prove —
it never silently degrades to a wrong answer.

Args:
    name: Region name. Must be a valid Python identifier; it becomes the
        outlined function's name and hence the runtime's graph key, so keep
        it stable across edits.

Returns:
    Context manager for the Graph scope.

Raises:
    ParserSyntaxError: if written in a device-kernel body (InCore / AIC /
        AIV / Group / Spmd), nested inside ``pl.at`` / ``pl.cluster`` /
        ``pl.spmd``, or nested inside another ``pl.graph``.

Examples:
    >>> for layer in pl.range(40):
    ...     with pl.graph("decoder_layer"):
    ...         h = attention(x, wq, h)
    ...         x = mlp(h, w1, x)

## `spmd`

```text
pypto.language.spmd(core_num: 'RangeArg', *, sync_start: 'bool' = False, name_hint: 'str' = '', optimizations: 'list[Optimization] | None' = None, deps: 'list[Any] | None' = None, allow_early_resolve: 'bool' = False, predicate: 'Any' = None) -> 'SpmdContext'
```

Dispatch a kernel with SPMD (Single Program Multiple Data) multi-block execution.

The first argument is the number of blocks and is positional — mirroring
``range(n)``. Loop start is fixed at 0 and step at 1; each block gets an
index ``i`` in ``[0, core_num)``.

**Three usage forms at a glance:**

| Form | Description |
| --- | --- |
| `with pl.spmd(n):` | Dispatch or inline block (no captured TaskId). |
| `for i in pl.spmd(n):` | Loop-style; `i` = per-block index, body is auto-outlined to InCore. |
| `with pl.spmd(n) as tid:` | Same body as form 1, plus the dispatch TaskId in `tid`. |

``deps=[...]`` works on all three: capturing the TaskId is only needed when a
*later* task must wait on this one.

Usage forms:

1. ``with pl.spmd(n):`` — body is either a *dispatch* body calling a
   pre-defined InCore kernel, or an *inline* block auto-outlined into a
   synthetic InCore kernel (like the loop form, minus the auto-bound index).
   Which one is decided semantically, not by statement count: a body reading
   the per-block index via ``pl.tile.get_block_idx()`` is inline; otherwise it
   is a dispatch body, however many statements it holds — though it may launch
   only one kernel (the lowering stops at the first call). A body that neither
   reads the index nor dispatches a kernel is rejected — every block would run
   identical work. An explicit ``with pl.at(<CORE_GROUP level>, ...):`` as the
   sole body statement *is* the InCore carrier and is not wrapped again; with
   such a body ``optimizations=`` must go on that ``pl.at(...)`` rather than on
   the ``pl.spmd(...)`` line. Captures no producer TaskId (use form 3 for
   that), but still accepts ``deps=``. Can stand alone (implicit cluster) or
   nest inside ``pl.cluster()`` — but only a scope carrying no Submit-only
   metadata may nest: ``deps=`` / ``allow_early_resolve=`` / ``predicate=``
   require the standalone form (a cluster-nested pl.spmd is unwrapped into
   the Group function and never produces the ``Submit`` that carries them).

2. ``for i in pl.spmd(n):`` — loop-style. The iteration variable binds
   the per-block index (equivalent to ``pl.tile.get_block_idx()``); the
   body is auto-outlined into a synthetic InCore function, so inline
   tile/tensor ops work without a separate ``@pl.function(type=InCore)``
   declaration.

3. ``with pl.spmd(n, deps=[...]) as tid:`` — same body shapes as form 1, and
   additionally captures the grid dispatch's producer ``Scalar[TASK_ID]`` in
   ``tid`` (mirroring ``with pl.at(...) as tid:``), usable as a ``deps=`` edge
   on later tasks, stored into a ``pl.array.create(N, pl.TASK_ID)``, or
   crossing into ``pl.manual_scope``. TaskId capture is the only thing this
   adds over form 1 — it is orthogonal to both the inline body and ``deps=``.

Optional ``optimizations=[pl.split(mode)]`` applies to the inner InCore scope
(auto-generated for the for-form and the ``as tid`` form, wrapped around the
call for the plain with-form).

Args:
    core_num: Number of blocks for SPMD dispatch. Positional; accepts a
        Python ``int`` or any ``ir.Expr`` of integer type. Closure-captured
        integer constants and closure arithmetic are folded to ``ConstInt``
        by the parser and ``Simplify``; non-foldable expressions flow
        through to codegen unchanged. Pass
        ``pl.system.available_cluster_count()`` (mixed / cube-only kernel)
        or ``pl.system.available_aiv_count()`` (vector-only kernel) to size
        the launch on the run's own device geometry — the only spelling that
        stays at full occupancy across devices, which a hard
        ``pl.system.syncall`` requires. Pass such a query inline as shown;
        binding it to a name first compiles and lowers correctly, but the
        printed IR of the outlined ``Spmd`` wrapper then references a
        variable defined in the caller and cannot be re-parsed.
    sync_start: If True, all blocks start execution simultaneously (default: False).
    name_hint: Optional name hint for the outlined function.
    optimizations: Optional list literal containing only ``pl.split(mode)``
        entries (the parser inspects the AST).
    deps: Optional explicit producer-edge list (TaskId Vars and/or ``None``
        sentinels), accepted on all three forms. Lowered to the outlined
        Submit's ``manual_dep_edges``; codegen packs it into a
        ``set_dependencies(...)`` invocation (union'd with auto-deps).
        Like ``allow_early_resolve`` it forces the dispatch to lower to an
        ``ir.Submit`` even without ``as tid`` — the Spmd outliner synthesises
        the (unused) producer TaskId Var that shape needs. Rejected on a
        ``pl.cluster()``-nested ``pl.spmd``, which is unwrapped into the Group
        function and never produces a Submit.
    allow_early_resolve: Opt the grid dispatch in as a speculative
        early-dispatch producer (simpler#1065). Same hint as
        ``pl.submit(..., allow_early_resolve=True)`` / ``pl.at(...,
        allow_early_resolve=True)``: the scheduler may pre-stage this task's
        consumers before it completes. Forces the dispatch to lower to an
        ``ir.Submit`` (even without ``as tid``) so the flag rides to codegen,
        where it emits ``Arg::set_allow_early_resolve(true)``. Pure scheduling
        hint. Rejected on a ``pl.cluster()``-nested ``pl.spmd`` (such a scope
        is unwrapped into the Group function and never produces a Submit, so
        the hint would be lost).
    predicate: Optional **dispatch predicate** — a single comparison of one
        tensor element against an integer literal, e.g.
        ``predicate=(row_count[e] > 0)``. Same surface, validation and
        lowering as ``pl.spmd_submit(..., predicate=...)``: the scheduler
        evaluates it at the dispatch point (after this scope's dependencies
        are satisfied, so the value is current) and retires the task inline —
        never dispatching it to a core — when it is false, while still
        settling fanin/fanout so downstream consumers unlock. The comparison
        is parsed but **never evaluated** in orchestration; reading it there
        would stall on ``wait_for_tensor_ready``, exactly what the predicate
        avoids. Like ``allow_early_resolve`` it forces the dispatch to lower
        to an ``ir.Submit`` (even without ``as tid``) and is rejected on a
        ``pl.cluster()``-nested ``pl.spmd``.

        **Contract:** the operand tensor's producing task must be one of
        ``deps=`` — otherwise the predicate may read a stale value. The parser
        makes a best-effort check (see ``pl.spmd_submit``); getting ``deps=``
        right remains the author's responsibility.

Returns:
    Context manager / loop iterator for the SPMD scope.

Examples:
    >>> # Single-kernel context-manager form (direct dispatch)
    >>> with pl.spmd(4):
    ...     out = self.kernel(a, b, out)
    >>>
    >>> # Inline context-manager form — no TaskId; read the block index yourself
    >>> with pl.spmd(4):
    ...     i = pl.tile.get_block_idx()
    ...     offset = i * 128
    ...     tile_a = pl.load(a, [offset, 0], [128, 128])
    ...     tile_b = pl.load(b, [offset, 0], [128, 128])
    ...     out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
    >>>
    >>> # Loop form — body runs per-block with i = tile.get_block_idx()
    >>> for i in pl.spmd(4):
    ...     offset = i * 128
    ...     tile_a = pl.load(a, [offset, 0], [128, 128])
    ...     tile_b = pl.load(b, [offset, 0], [128, 128])
    ...     out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
    >>>
    >>> # Capture-form: inline body + producer TaskId for explicit dep wiring
    >>> with pl.spmd(4, name_hint="stage1") as tid_a:
    ...     i = pl.tile.get_block_idx()
    ...     offset = i * 128
    ...     out = pl.store(pl.add(a[offset], b[offset]), [offset, 0], out)
    >>> with pl.spmd(4, name_hint="stage2", deps=[tid_a]) as tid_b:
    ...     i = pl.tile.get_block_idx()
    ...     out2 = pl.store(pl.relu(out[i * 128]), [i * 128, 0], out2)
    >>>
    >>> # With-form with split hint on the inner InCore wrapper
    >>> with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
    ...     out = self.kernel(a, b, out)
    >>>
    >>> # SPMD inside cluster (mixed kernel)
    >>> with pl.cluster():
    ...     with pl.spmd(4, sync_start=True):
    ...         out = self.kernel(a, b, out)
    >>>
    >>> # Device-sized launch — required for a hard pl.system.syncall
    >>> with pl.spmd(pl.system.available_cluster_count(), sync_start=True):
    ...     out = self.mixed_kernel(a, b, out)
    >>>
    >>> # Dispatch predicate — skip the expert entirely when its row count is 0
    >>> with pl.spmd(1) as gate_tid:
    ...     row_count = self.gate(row_count)
    >>> with pl.spmd(4, deps=[gate_tid], predicate=(row_count[0, 0] > 0)) as tid:
    ...     out = self.expert(x, out)

## `split_aiv`

```text
pypto.language.split_aiv(n: 'int', *, mode: 'ir.SplitMode') -> 'SplitAivContext'
```

Open an explicit AIV-split region as an SPMD-style loop.

Usage::

    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        ...  # body runs per AIV lane; aiv_id = pl.tile.get_subblock_idx()

The loop builds a first-class ``SplitAivScopeStmt``
region carrying the requested ``SplitMode``. Because it is a structural node
(not a whole-InCore-scope flag), it is **nestable**: the region may appear
inside a ``pl.range`` / ``pl.pipeline`` loop or an ``if``, and sibling regions
may carry **different** modes (multi-mode), each lowered independently. The
loop variable binds the AIV lane index (equivalent to
``pl.tile.get_subblock_idx()``).

A top-level ``for aiv_id in pl.split_aiv(...)`` is wrapped in an enclosing
``InCore`` scope so OutlineIncoreScopes can outline it — unless the region is
already in a core context (an open ``pl.at(level=CORE_GROUP)`` scope, or a
function declared ``pl.FunctionType.InCore``), where it is emitted in place.
A region is CORE_GROUP-level, so **authoring** one directly in an
``InCore`` function is rejected by the ``AivSplitValid`` verifier; write it
in a plain ``@pl.function`` / ``@pl.jit`` (Opaque) body instead.

Two dispatch modes:

* **Data-parallel** (``UP_DOWN`` / ``LEFT_RIGHT``): the region's vector
  compute is **halved** on the split axis (rows / cols), so each lane
  processes one half of every tile.
* **Task-parallel** (``NONE``): **no halving** — both lanes run the *full*
  body for disjoint work the author dispatches via ``aiv_id`` (e.g. an
  ``aiv_id``-strided loop). Use this when the tiles cannot be halved
  (unit dims) or a reduction must stay full-width. ``pl.aiv_shard`` /
  ``pl.aic_gather`` are valid here and **preserve the shape**: with no split
  axis they carry the one meaning that still applies — this value crosses
  the AIC/AIV boundary.

**Manual mode.** Opening even ONE region changes the contract for the whole
enclosing function: the regions become authoritative for where vector work
runs, and the ``AivSplitValid`` verifier enforces the division:

==========================  =================  ==========================
op                          inside a region    outside every region
==========================  =================  ==========================
vector compute              AIV                **rejected**
``pl.load`` / ``pl.store``  AIV                allowed (compiler-inserted)
cube compute (``matmul``)   **rejected**       AIC
``pl.aiv_shard`` /
``pl.aic_gather``           the boundary       **rejected**
``pld.system.notify``       pinned to AIV      duplicated onto BOTH lanes
==========================  =================  ==========================

So write **one region per vector phase**, with cube ops and barriers between
them. A phase that must stay full width goes in its own
``for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):`` — whose meaning is
exactly "both lanes run the full body", which is what an un-regioned vector
phase in such a function already did. A function with NO region is
unaffected and keeps its previous behaviour.

**Name every tile that crosses a region edge.** Manual mode hands the AIC/AIV
boundary to the author, so the compiler stops placing it: a cube-produced
value read on the vector lane inside a region must arrive through
``pl.aiv_shard``, and a value defined in a region and read on the cube lane
outside it must leave through ``pl.aic_gather``. Both crossings lower fine
without the op — which is the point: an unnamed boundary is one nobody chose,
so the verifier rejects it and names the value and the reader::

    mm = pl.matmul(q, k)                     # cube, outside every region
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        v = pl.exp(pl.aiv_shard(mm))         # C->V: named
        kv = pl.aic_gather(v)                # V->C: named
    out = pl.matmul(kv, w)                   # cube again, outside

**Every crossing in one function must agree on split-vs-no-split.** All the
``pl.aiv_shard`` / ``pl.aic_gather`` calls in a function ride ONE logical
cross-core pipe -- one ``initialize_pipe`` per side, both directions on the
same pipe -- and pto-isa carries no-split as a parameter of that pipe's TYPE
(``TPipe<..., IsNoSplit, ...>``), selecting a different handshake protocol.
A pipe is therefore split or un-split for its whole lifetime, so a ``NONE``
region that crosses the boundary cannot sit beside a data-parallel one that
also crosses it::

    for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        a = pl.exp(pl.aiv_shard(mm0))        # crossing, no split
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        b = pl.exp(pl.aiv_shard(mm1))        # crossing, split   -> rejected

Two **different** split axes are fine -- the axis is a per-transfer choice,
only split-vs-no-split belongs to the pipe -- so ``UP_DOWN`` beside
``LEFT_RIGHT`` is accepted. A region carrying **no** crossing is free to use
any mode: the ``NONE`` region that only pins a ``pld.system.notify`` to the
vector lane never touches the pipe. When two phases genuinely need different
transports, put them in separate ``pl.at(level=pl.Level.CORE_GROUP)`` scopes
-- each outlines into its own function, and so gets its own pipe. Diagnosed
by ``AivSplitValid`` at OutlineIncoreScopes; without it the program reaches
ptoas, which rejects it at ``pto.initialize_l2g2l_pipe``.

**Gather only a lane-uniform value out of a ``NONE`` region.** The ISA requires
both AIV sub-lanes to take part in a no-split handshake and they share one
destination slot with no per-lane offset, and nothing arbitrates between
them: both lanes push, so if they hold different values the cube receives an
**unspecified** one of the two. Guarding the *production* of the value does
not help — lane 1 still reaches the push and still sends its own tile. So a
gather out of a ``NONE`` region is well-defined only for a lane-uniform
value; if the lanes must contribute different data, use a data-parallel
region instead. **Not diagnosed.**

**Put cross-rank comm ops in a region.** A region also decides placement for
ops that have no lane of their own. ``pld.system.notify`` is core-agnostic
by ISA, so in a kernel that mixes cube and vector work it is emitted on BOTH
the AIC and the AIV lane — and the cube copy can publish the signal before
the vector lane's put has landed the data it releases. Putting the op in a
region pins it to the vector lane::

    for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        pld.tensor.put(dst=win, peer=peer, src=out, ...)
        pld.system.notify(target=sig, peer=peer, offsets=[0, 0], value=1,
                          op=pld.NotifyOp.AtomicAdd)

**A region is not "run once" — shard once-only side effects by aiv_id.**
A ``mode=NONE`` body runs on BOTH AIV sub-lanes, so the snippet above sends
TWO notifies to the same peer. `AtomicAdd` accumulates, so that peer's
counter reads 2 for one arrival, a ``pld.system.wait`` on it is released
early, and the rank races ahead on data that has not landed. Shard the op,
or guard it to one lane::

    # sharded: each lane takes different peers
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        for owner in pl.range(aiv_id, NUM_PEERS, 2):
            pld.system.notify(target=sig, peer=owner, ...)

    # guarded: lane 0 only
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        if aiv_id == 0:
            pld.system.notify(target=sig, peer=peer, ...)

**This is NOT diagnosed.** The compiler guarantees only that a region's comm
ops stay off the cube lane; the sharded and un-sharded forms are the same
single statement in the AIV body, differing only in whether ``aiv_id``
reaches the call's arguments.

**GM traffic is outside all of this.** The crossing rules above govern *tile*
values. A GM tensor belongs to no lane — ``pld.tensor.put`` takes one by
signature — so no boundary op can express a crossing through it, and AIC and
AIV run asynchronously. ``ExpandMixedKernel`` fences one narrow shape by
itself: a cube ``tile.store`` whose GM tensor a vector ``tile.load`` reads
back, when the two share an origin and the load is in or under the store's
body. Everything else — a comm op reading the buffer, a consumer in a
sibling body — needs explicit producer-side cache publication and a GM
fence, a cross-core ``pl.system.syncall``, and consumer-side invalidation.
That sequence remains the author's responsibility.

The region survives parse -> SSA -> ResolveBackendOpLayouts as a structural
node (printer emits ``for aiv_id in pl.split_aiv(...):`` so parse->print->parse
is a fixpoint), then is consumed and erased by LowerAutoVectorSplit (pass 23);
it never reaches ExpandMixedKernel or codegen.

Args:
    n: The AIV sub-core count. Positional; hardware-fixed at 2 (the two AIV
        lanes of one AICore). Any other value is rejected by the parser.
    mode: Required dispatch mode, no silent default. ``pl.SplitMode.NONE``
        (task-parallel, no halving), ``pl.SplitMode.UP_DOWN`` or
        ``pl.SplitMode.LEFT_RIGHT`` (data-parallel halving).

Returns:
    Loop iterator for the explicit AIV-split region.

Examples:
    >>> for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
    ...     offset = aiv_id * 128
    ...     t = pl.load(a, [offset, 0], [128, 128])
    ...     out = pl.store(t, [offset, 0], out)

    A second, full-width vector phase after a cube op needs its own region
    (manual mode) — ``mode=NONE`` keeps it un-halved:

    >>> for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    ...     ...  # phase 1, dispatched per lane via aiv_id
    >>> mm = pl.matmul(q, k)  # cube work stays outside every region
    >>> for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    ...     out = pl.add(mm, bias)  # phase 2, full width on both lanes

## `split`

```text
pypto.language.split(mode: 'SplitMode', *, slot_num: 'int | None' = None) -> 'Split'
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

## `cross_core_slot`

```text
pypto.language.cross_core_slot(*, slot_num: 'int') -> 'CrossCoreSlot'
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

## `add`

```text
pypto.language.add(lhs, rhs)
```

Element-wise addition, dispatched by input type.

## `sub`

```text
pypto.language.sub(lhs, rhs)
```

Element-wise subtraction, dispatched by input type.

## `mul`

```text
pypto.language.mul(lhs, rhs)
```

Element-wise multiplication, dispatched by input type.

## `div`

```text
pypto.language.div(lhs, rhs, high_precision: bool = False)
```

Element-wise division, dispatched by input type.

A scalar ``rhs`` against a Tile dispatches to ``tile.divs``.

Args:
    high_precision: Select PTOAS's high-precision divide. Available for
        Tensor/Tensor and Tile/Tile only -- a scalar divisor has no
        high-precision form, so passing it there raises rather than silently
        falling back.

## `part_add`

```text
pypto.language.part_add(lhs, rhs)
```

Partial element-wise add, dispatched by input type.

## `part_mul`

```text
pypto.language.part_mul(lhs, rhs)
```

Partial element-wise multiply, dispatched by input type.

## `part_max`

```text
pypto.language.part_max(lhs, rhs)
```

Partial element-wise max, dispatched by input type.

## `part_min`

```text
pypto.language.part_min(lhs, rhs)
```

Partial element-wise min, dispatched by input type.

## `maximum`

```text
pypto.language.maximum(lhs, rhs)
```

Element-wise maximum, dispatched by input type.

## `exp`

```text
pypto.language.exp(input: ~T) -> ~T
```

Element-wise exponential, dispatched by input type.

## `log`

```text
pypto.language.log(input: ~T, high_precision: bool = False) -> ~T
```

Element-wise natural logarithm, dispatched by input type.

Args:
    input: Input tensor or tile.
    high_precision: Select PTOAS's high-precision logarithm mode.

## `cast`

```text
pypto.language.cast(input: pypto.language.typing.tensor.Tensor | pypto.language.typing.tile.Tile | pypto.language.typing.scalar.Scalar, target_type: int | pypto.pypto_core.DataType, mode: str | int = 'round') -> pypto.language.typing.tensor.Tensor | pypto.language.typing.tile.Tile | pypto.language.typing.scalar.Scalar
```

Type casting, dispatched by input type.

Args:
    input: Value to convert.
    target_type: Destination dtype.
    mode: Rounding mode, as a name or its int code -- ``"none"`` (0),
        ``"rint"`` (1), ``"round"`` (2, the default), ``"floor"`` (3),
        ``"ceil"`` (4), ``"trunc"`` (5), ``"odd"`` (6). A ``Scalar`` input
        supports the default only and raises for any other mode.

## `concat`

```text
pypto.language.concat(src0: ~T, src1: ~T) -> ~T
```

Column-wise concatenation, dispatched by input type.

## `reshape`

```text
pypto.language.reshape(input: ~T, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr]) -> ~T
```

Reshape operation, dispatched by input type.

A reshape is a zero-copy view, so it never widens the valid region: the
result holds real data in exactly the cells the input did, re-expressed in
``shape``. Because a valid region is an origin-anchored box, not every input
region survives a repartition — reshaping a region that no box of ``shape``
can describe is rejected rather than silently rounded up to fully valid.
Reshapes that only add or drop fully-valid unit axes always work, as does a
region occupying a contiguous prefix of the buffer.

## `reinterpret_view`

```text
pypto.language.reinterpret_view(data: ~T, dtype: pypto.pypto_core.DataType, *, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None) -> ~T
```

Reinterpret the same bytes with a different dtype.

Args:
    data: Input tensor or tile.
    dtype: Target element dtype, which must differ from the source dtype.
    shape: Optional byte-equivalent target shape. When omitted, the
        physically contiguous dimension is scaled according to the
        source/target dtype byte ratio.

Returns:
    A zero-copy view of the same kind as ``data``.

## `transpose`

```text
pypto.language.transpose(input: ~T, axis1: int, axis2: int) -> ~T
```

Transpose operation, dispatched by input type.

Args:
    input: Value to transpose.
    axis1: First axis to exchange. Must be a compile-time constant; negative
        indexing is supported.
    axis2: Second axis to exchange. Must differ from ``axis1`` after negative
        indexing is resolved -- naming the same axis twice is rejected, not
        treated as a no-op.

## `slice`

```text
pypto.language.slice(input: ~T, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, drop_dims: collections.abc.Sequence[int | pypto.pypto_core.ir.Expr] | None = None, pad_value: pypto.pypto_core.ir.PadValue | int | float | None = None, clamp: bool = False) -> ~T
```

Slice operation, dispatched by input type.

The slice is never valid where the source is not: the source's valid region,
shifted by ``offset`` and cut to the window, bounds the result.

``drop_dims`` lists axes to erase from the result type (numpy-style rank
reduction); each must be a static unit dim of ``shape`` that is still fully
valid after that intersection. ``None`` / ``[]`` drops nothing.

``pad_value`` sets the padding mode for elements outside the effective valid
region, on either path. ``None`` carries the source's mode through. Accepts
``PadValue.zero`` / ``PadValue.max`` / ``PadValue.min``, or the literal
sugars ``0``, ``math.inf``, ``-math.inf`` (same spelling as [`fillpad`][pypto.language.fillpad]).
It only bites when the valid region is smaller than ``shape`` — which an
explicit ``valid_shape``, a partially-valid source, or (Tensor-only)
``clamp=True`` can each bring about; passing it otherwise warns.

``clamp`` sanctions a window that runs off the end of the source: by default
the slice asserts ``offset + shape`` stays inside the source and is rejected
when that provably fails, whereas ``clamp=True`` lets the window overhang and
cuts the valid region back to the source edge. It is only available on a
Tensor — an on-chip tile window has nothing that could clamp it.

## `matmul`

```text
pypto.language.matmul(lhs: ~T, rhs: ~T, out_dtype: int | pypto.pypto_core.DataType | None = None, a_trans: bool = False, b_trans: bool = False, c_matrix_nz: bool = False) -> ~T
```

Matrix multiplication, dispatched by input type.

``a_trans`` / ``b_trans`` / ``c_matrix_nz`` are Tensor-only: a tensor value
carries no layout, so a flag is the only place the information can live. At
tile level transposition is a *type* property, so passing any of them with a
Tile operand raises rather than being dropped.

A transpose flag swaps its own operand's two trailing axes, so that operand
must be at least 2D — ``a_trans`` with a 1D ``lhs`` (or ``b_trans`` with a 1D
``rhs``) raises rather than being ignored. On the mixed mat-vec / vec-mat
forms the flag applies to the matrix side: a ``lhs`` stored ``[K, M]`` with
``a_trans=True`` against a ``[K]`` ``rhs`` deduces ``[M]``.

``out_dtype`` is likewise Tensor-only. ``tile.matmul``'s result dtype is
fixed by the Cube accumulator (FP32 for float operands, INT32 for int), so
the Tile path accepts ``out_dtype`` only when it already agrees with that
deduction and raises otherwise.

The Tensor path allows one conversion on top of that deduction, because the
result drains L0C through the FIXPIPE, whose unscaled writeback narrows
``FP32 -> FP16`` / ``FP32 -> BF16``. So float operands accept ``FP32``,
``FP16`` or ``BF16``, and int operands accept only ``INT32``. Every rejected
pair is a *scale-bearing* conversion the FIXPIPE can only do with quantization
parameters this call has nowhere to carry — ``INT32 -> FP32`` is a
dequantization, ``FP32 -> INT8`` a quantization, ``INT32 -> INT8`` a
requantization. Requesting one (notably ``out_dtype=FP32`` on INT8 operands)
raises here rather than lowering to a backend type error. Convert explicitly
with ``pl.cast`` instead.

For Tensor inputs with rank > 2 on either operand, the call is lowered to
``tile.batch_matmul`` (with batch broadcasting) by ``ConvertTensorToTileOps``
and then unrolled to per-batch ``tile.matmul`` by ``FlattenTileNdTo2D``.
Use this entry point (rather than ``pl.batch_matmul``) for tensor-level ND
matmul.

## `batch_matmul`

```text
pypto.language.batch_matmul(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Tile-only batched matrix multiplication.

Tensor batched matmul is handled by ``pl.matmul`` / ``pl.tensor.matmul``:
when any operand has rank > 2, ``ConvertTensorToTileOps`` automatically
dispatches to ``tile.batch_matmul`` (and ``FlattenTileNdTo2D`` later
unrolls it). Use this op only when you are working at the tile level.

## `row_max`

```text
pypto.language.row_max(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Row-wise max reduction, dispatched by input type.

For Tile inputs, ``tmp_tile`` is required and must have the same dtype and
rank as the input, with every dimension at least as large as the input dimension.
Tensor inputs must omit it — the scratch tile is allocated during
Tensor-to-Tile lowering — and passing one raises.

## `row_sum`

```text
pypto.language.row_sum(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Row-wise sum reduction, dispatched by input type.

For Tile inputs, ``tmp_tile`` is required and must have the same dtype and
rank as the input, with every dimension at least as large as the input dimension.
Tensor inputs must omit it — the scratch tile is allocated during
Tensor-to-Tile lowering — and passing one raises.

## `row_min`

```text
pypto.language.row_min(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Row-wise min reduction, dispatched by input type.

For Tile inputs, ``tmp_tile`` is required and must have the same dtype and
rank as the input, with every dimension at least as large as the input dimension.
Tensor inputs must omit it — the scratch tile is allocated during
Tensor-to-Tile lowering — and passing one raises.

## `row_prod`

```text
pypto.language.row_prod(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Row-wise product reduction, dispatched by input type.

For Tile inputs, ``tmp_tile`` is required and must have the same dtype and
rank as the input, with every dimension at least as large as the input dimension.
Tensor inputs must omit it — the scratch tile is allocated during
Tensor-to-Tile lowering — and passing one raises.

## `col_sum`

```text
pypto.language.col_sum(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Column-wise sum reduction, dispatched by input type.

For Tile inputs, passing ``tmp_tile`` activates the binary-tree reduction
path; omitting it uses the sequential path. Tensor inputs must omit it: the
tensor-to-tile conversion always lowers to the sequential path and allocates
its own scratch, so a ``tmp_tile`` there could not select the requested
strategy and raises instead.

## `col_max`

```text
pypto.language.col_max(input: ~T) -> ~T
```

Column-wise max reduction, dispatched by input type.

For Tensor inputs, the tensor-to-tile conversion lowers to ``tile.col_max``.

## `col_min`

```text
pypto.language.col_min(input: ~T) -> ~T
```

Column-wise min reduction, dispatched by input type.

For Tensor inputs, the tensor-to-tile conversion lowers to ``tile.col_min``.

## `col_prod`

```text
pypto.language.col_prod(input: ~T) -> ~T
```

Column-wise product reduction, dispatched by input type.

For Tensor inputs, the tensor-to-tile conversion lowers to ``tile.col_prod``.

## `row_argmax`

```text
pypto.language.row_argmax(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Row-wise argmax (per-row max index, int32), dispatched by input type.

For Tile inputs, tmp_tile is required with exactly the same shape and dtype.
Tensor inputs must omit it — the conversion injects the scratch tile — and
passing one raises.

## `row_argmin`

```text
pypto.language.row_argmin(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Row-wise argmin (per-row min index, int32), dispatched by input type.

For Tile inputs, tmp_tile is required with exactly the same shape and dtype.
Tensor inputs must omit it — the conversion injects the scratch tile — and
passing one raises.

## `col_argmax`

```text
pypto.language.col_argmax(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Column-wise argmax (per-column max index, int32), dispatched by input type.

For Tile inputs, tmp_tile is required (unlike col_max) and must have exactly
the same shape and dtype as the input: the pto-isa kernel reads the column
count from the tmp/src extent, so an oversized scratch walks past the valid
columns. Tensor inputs must omit it — the conversion injects the tmp tile —
and passing one raises.

## `col_argmin`

```text
pypto.language.col_argmin(input, tmp_tile: pypto.language.typing.tile.Tile | None = None)
```

Column-wise argmin (per-column min index, int32), dispatched by input type.

For Tile inputs, tmp_tile is required (unlike col_min) and must have exactly
the same shape and dtype as the input: the pto-isa kernel reads the column
count from the tmp/src extent, so an oversized scratch walks past the valid
columns. Tensor inputs must omit it — the conversion injects the tmp tile —
and passing one raises.

## `row_expand`

```text
pypto.language.row_expand(lhs: ~T, rhs: ~T) -> ~T
```

Row-wise expansion, dispatched by input type.

## `row_expand_add`

```text
pypto.language.row_expand_add(lhs, rhs, tmp: pypto.language.typing.tile.Tile | None = None)
```

Row-wise broadcast addition; ``tmp`` is available only for Tile inputs.

## `row_expand_sub`

```text
pypto.language.row_expand_sub(lhs: ~T, rhs: ~T) -> ~T
```

Row-wise broadcast subtraction, dispatched by input type.

## `row_expand_mul`

```text
pypto.language.row_expand_mul(lhs: ~T, rhs: ~T) -> ~T
```

Row-wise broadcast multiplication, dispatched by input type.

## `row_expand_div`

```text
pypto.language.row_expand_div(lhs: ~T, rhs: ~T) -> ~T
```

Row-wise broadcast division, dispatched by input type.

## `row_expand_max`

```text
pypto.language.row_expand_max(lhs: ~T, rhs: ~T) -> ~T
```

Row-wise broadcast maximum, dispatched by input type.

## `row_expand_min`

```text
pypto.language.row_expand_min(lhs: ~T, rhs: ~T) -> ~T
```

Row-wise broadcast minimum, dispatched by input type.

## `row_expand_expdif`

```text
pypto.language.row_expand_expdif(lhs: ~T, rhs: ~T) -> ~T
```

Row-wise exp-diff (exp(lhs - rhs) with per-row scalar), dispatched by input type.

## `col_expand`

```text
pypto.language.col_expand(lhs: ~T, rhs: ~T) -> ~T
```

Column-wise expansion, dispatched by input type.

## `col_expand_mul`

```text
pypto.language.col_expand_mul(lhs: ~T, rhs: ~T) -> ~T
```

Column-wise broadcast multiplication, dispatched by input type.

## `col_expand_div`

```text
pypto.language.col_expand_div(lhs: ~T, rhs: ~T) -> ~T
```

Column-wise broadcast division, dispatched by input type.

## `col_expand_sub`

```text
pypto.language.col_expand_sub(lhs: ~T, rhs: ~T) -> ~T
```

Column-wise broadcast subtraction, dispatched by input type.

## `col_expand_add`

```text
pypto.language.col_expand_add(lhs: ~T, rhs: ~T) -> ~T
```

Column-wise broadcast addition, dispatched by input type.

## `col_expand_max`

```text
pypto.language.col_expand_max(lhs: ~T, rhs: ~T) -> ~T
```

Column-wise broadcast maximum, dispatched by input type.

## `col_expand_min`

```text
pypto.language.col_expand_min(lhs: ~T, rhs: ~T) -> ~T
```

Column-wise broadcast minimum, dispatched by input type.

## `col_expand_expdif`

```text
pypto.language.col_expand_expdif(lhs: ~T, rhs: ~T) -> ~T
```

Column-wise exp-diff (exp(lhs - rhs) with per-column scalar), dispatched by input type.

## `expand_clone`

```text
pypto.language.expand_clone(src: pypto.language.typing.tensor.Tensor, target: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
```

Clone and expand input to target shape.

Args:
    src: Source tensor
    target: Target tensor defining output shape

Returns:
    Tensor wrapping the expand_clone operation

## `expands`

```text
pypto.language.expands(target: pypto.language.typing.tensor.Tensor | pypto.language.typing.tile.Tile, scalar: int | float | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor | pypto.language.typing.tile.Tile
```

Expand scalar to target shape, dispatched by target type.

Note the argument order: the value being broadcast is the *second* argument.
``target`` supplies the shape and dtype; it is not read.

Args:
    target: Value whose shape the scalar is broadcast to.
    scalar: Value to broadcast into every element.

## `neg`

```text
pypto.language.neg(input: ~T) -> ~T
```

Element-wise negation, dispatched by input type.

## `abs`

```text
pypto.language.abs(input: ~T) -> ~T
```

Element-wise absolute value, dispatched by input type.

## `recip`

```text
pypto.language.recip(input: ~T, high_precision: bool = False) -> ~T
```

Element-wise reciprocal (1/x), dispatched by input type.

Args:
    input: Input tensor or tile
    high_precision: Whether to select PTOAS's high-precision reciprocal mode (FP16/FP32 only)

## `read`

```text
pypto.language.read(src: pypto.language.typing.tensor.Tensor | pypto.language.typing.tile.Tile, offset: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr]) -> pypto.language.typing.scalar.Scalar
```

Read a scalar value at given indices, dispatched by source type.

Args:
    src: Source tensor (global memory) or tile (unified buffer)
    offset: A single index expression (for 1-D flat access) or index list
        (one per dimension) into the source

Returns:
    Scalar wrapping the read value

## `write`

```text
pypto.language.write(dst: pypto.language.typing.tensor.Tensor | pypto.language.typing.tile.Tile, offset: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], value: pypto.language.typing.scalar.Scalar) -> pypto.pypto_core.ir.Expr
```

Write a scalar value to a tensor or tile at given indices.

Args:
    dst: Destination tensor (global memory) or tile (unified buffer)
    offset: A single index expression (for 1-D flat access) or index list
        (one per dimension) into the destination
    value: Scalar value to write

Returns:
    Underlying ``tensor.write`` / ``tile.write`` call expression. Direct
    callers ignore it; the DSL parser surfaces it as an ``EvalStmt``.

## `create_tile`

```text
pypto.language.create_tile(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, target_memory: pypto.pypto_core.ir.MemorySpace | None = None, transpose: bool | None = None, *, flat_layout: bool | None = None, compact: bool | None = None) -> pypto.language.typing.tile.Tile
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

## `fillpad`

```text
pypto.language.fillpad(value: ~T, pad_value: pypto.pypto_core.ir.PadValue | int | float = PadValue.zero) -> ~T
```

Fill invalid elements, dispatched by input type.

``pad_value`` accepts the ``PadValue`` enum or the literal sugars ``0``,
``math.inf``, ``-math.inf``. Other values raise — the hardware only
supports the three padding modes.

## `fillpad_expand`

```text
pypto.language.fillpad_expand(value: ~T, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], pad_value: pypto.pypto_core.ir.PadValue | int | float = PadValue.zero) -> ~T
```

Copy a smaller source into a larger destination, padding the rest.

Dispatched by input type. The destination ``shape`` may be larger than the
source in either dimension; the source's valid region is copied into the
top-left of the destination and every other element is filled with
``pad_value`` (``PadValue`` enum or the literal sugars ``0``, ``math.inf``,
``-math.inf``).

## `load`

```text
pypto.language.load(tensor: pypto.language.typing.tensor.Tensor, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], shapes: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, target_memory: pypto.pypto_core.ir.MemorySpace | None = None, clamp: bool = False, cache: pypto.pypto_core.ir.CachePolicy | None = None) -> pypto.language.typing.tile.Tile
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
pypto.language.store(tile: pypto.language.typing.tile.Tile, offsets: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], output_tensor: ~_TensorT, shapes: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None, *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_, st_phase: pypto.pypto_core.ir.STPhase = STPhase.Unspecified) -> ~_TensorT
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

## `move`

```text
pypto.language.move(tile: pypto.language.typing.tile.Tile, target_memory: pypto.pypto_core.ir.MemorySpace, blayout: pypto.pypto_core.ir.TileLayout | None = None, slayout: pypto.pypto_core.ir.TileLayout | None = None) -> pypto.language.typing.tile.Tile
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

## `gatherb`

```text
pypto.language.gatherb(src: pypto.language.typing.tile.Tile, offset: pypto.language.typing.tile.Tile, *, output_dtype: int | pypto.pypto_core.DataType | None = None) -> pypto.language.typing.tile.Tile
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

## `mgather`

```text
pypto.language.mgather(mem: pypto.language.typing.tensor.Tensor, idx: pypto.language.typing.tile.Tile | pypto.language.typing.tensor.Tensor, coalesce: str | int = 'row', *, gather_oob: str | int = 'undefined', target_memory: pypto.pypto_core.ir.MemorySpace = MemorySpace.Vec, scratch: pypto.language.typing.tensor.Tensor | None = None, valid_shape: collections.abc.Sequence[int] | None = None) -> pypto.language.typing.tile.Tile
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

## `mscatter`

```text
pypto.language.mscatter(src: pypto.language.typing.tile.Tile, idx: pypto.language.typing.tile.Tile, output_tensor: ~_TensorT) -> ~_TensorT
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

## `sqrt`

```text
pypto.language.sqrt(input: ~T) -> ~T
```

Element-wise square root, dispatched by input type.

## `rsqrt`

```text
pypto.language.rsqrt(input, high_precision: bool = False)
```

Element-wise reciprocal square root, dispatched by input type.

``high_precision`` is Tensor-only: the compiler allocates the scratch tile
during Tensor-to-Tile lowering. ``tile.rsqrt`` carries no such attribute —
precision is selected purely by *passing* that scratch tile — so tile
callers use ``pl.tile.rsqrt(tile, tmp)`` directly and passing
``high_precision=True`` with a Tile raises rather than silently yielding the
low-precision path.

## `relu`

```text
pypto.language.relu(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
```

Element-wise ReLU activation (max(0, x)).

Args:
    tile: Input tile

Returns:
    Tile wrapping the relu operation

## `matmul_acc`

```text
pypto.language.matmul_acc(acc: ~T, lhs: ~T, rhs: ~T, a_trans: bool = False, b_trans: bool = False, init_cond: bool | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None) -> ~T
```

Matrix multiplication with accumulation, dispatched by input type.

``a_trans`` / ``b_trans`` are Tensor-only for the same reason as in
[`matmul`][pypto.language.matmul] — at tile level transposition is a type property, not an op
flag — so passing either with Tile operands raises rather than being
dropped.

``init_cond`` makes the accumulator's initial value conditional: on the steps
where it holds, ``acc`` is overwritten with ``lhs @ rhs`` rather than
accumulated into, which is the split-K ``k == 0`` idiom. It applies to 2D
operands only.

For Tensor inputs with rank > 2 on any of acc/lhs/rhs, the call is lowered
to ``tile.batch_matmul_acc`` (with batch broadcasting on lhs/rhs vs the
fixed acc batch) by ``ConvertTensorToTileOps`` and then unrolled to
per-batch ``tile.matmul_acc`` by ``FlattenTileNdTo2D``.

## `matmul_bias`

```text
pypto.language.matmul_bias(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.matmul_mx(lhs: pypto.language.typing.tile.Tile, lhs_scale: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs_scale: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.matmul_mx_acc(acc: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, lhs_scale: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs_scale: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.matmul_mx_bias(lhs: pypto.language.typing.tile.Tile, lhs_scale: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs_scale: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.gemv(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified) -> pypto.language.typing.tile.Tile
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
pypto.language.gemv_acc(acc: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified, *, init_cond: bool | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None) -> pypto.language.typing.tile.Tile
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
pypto.language.gemv_bias(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified) -> pypto.language.typing.tile.Tile
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

## `minimum`

```text
pypto.language.minimum(lhs, rhs)
```

Element-wise minimum, dispatched by input type.

## `min`

```text
pypto.language.min(lhs: pypto.language.typing.scalar.Scalar | int | pypto.pypto_core.ir.Expr, rhs: pypto.language.typing.scalar.Scalar | int | pypto.pypto_core.ir.Expr) -> pypto.language.typing.scalar.Scalar
```

Scalar min of two values.

Tile reductions are direction-specific — use [`row_min`][pypto.language.tile.row_min] (collapses the
last axis) or [`col_min`][pypto.language.tile.col_min] (collapses axis 0).

Args:
    lhs: First scalar operand
    rhs: Second scalar operand

Returns:
    Scalar wrapping the min operation

## `max`

```text
pypto.language.max(lhs: pypto.language.typing.scalar.Scalar | int | pypto.pypto_core.ir.Expr, rhs: pypto.language.typing.scalar.Scalar | int | pypto.pypto_core.ir.Expr) -> pypto.language.typing.scalar.Scalar
```

Scalar max of two values.

Tile reductions are direction-specific — use [`row_max`][pypto.language.tile.row_max] (collapses the
last axis) or [`col_max`][pypto.language.tile.col_max] (collapses axis 0).

Args:
    lhs: First scalar operand
    rhs: Second scalar operand

Returns:
    Scalar wrapping the max operation

## `cmp`

```text
pypto.language.cmp(lhs, rhs, cmp_type: int = 0)
```

Element-wise comparison, dispatched by input type.

Comparison type codes: ``0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge``. For Tile
inputs with a scalar ``rhs``, dispatches to ``tile.cmps`` automatically.

## `cmps`

```text
pypto.language.cmps(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, cmp_type: int = 0) -> pypto.language.typing.tile.Tile
```

Element-wise comparison of tile and scalar.

Args:
    lhs: Tile
    rhs: Scalar value
    cmp_type: Comparison type (EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5)

Returns:
    Tile wrapping a packed predicate mask. Use tile.sel with an explicit tmp tile to materialize values.

## `set_validshape`

```text
pypto.language.set_validshape(input, valid_rows, valid_cols)
```

Update valid-shape metadata without data movement, dispatched by input type.

.. note::
    Prefer expressing the extent at its source where possible —
    ``pl.load(..., valid_shape=...)`` or a slice's ``valid_shape=``. A tile
    view (slice / reshape result) is rejected: it carries its valid extent in
    its type, so there are no runtime operands to update.

## `rem`

```text
pypto.language.rem(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, high_precision: bool = False) -> pypto.language.typing.tile.Tile
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
pypto.language.rems(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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

## `fmod`

```text
pypto.language.fmod(lhs, rhs, high_precision: bool = False)
```

Element-wise truncating remainder, dispatched by input type.

Matches ``torch.fmod`` (the remainder takes the sign of the dividend).
``high_precision`` is available only for the tile-tile form.

## `fmods`

```text
pypto.language.fmods(lhs, rhs)
```

Element-wise truncating remainder with a scalar, dispatched by input type.

## `and_`

```text
pypto.language.and_(lhs, rhs)
```

Element-wise bitwise AND, dispatched by input type.

## `ands`

```text
pypto.language.ands(lhs, rhs)
```

Element-wise bitwise AND with a scalar, dispatched by input type.

## `or_`

```text
pypto.language.or_(lhs, rhs)
```

Element-wise bitwise OR, dispatched by input type.

## `ors`

```text
pypto.language.ors(lhs, rhs)
```

Element-wise bitwise OR with a scalar, dispatched by input type.

## `xor`

```text
pypto.language.xor(lhs, rhs, tmp=None)
```

Element-wise bitwise XOR, dispatched by input type.

``pto.txor`` needs a scratch buffer. Tile buffer lifetimes are user-managed,
so the tile path takes it as ``tmp``; the tensor path omits it because
ConvertTensorToTileOps allocates it — the same asymmetry ``rsqrt`` carries.

## `xors`

```text
pypto.language.xors(lhs, rhs, tmp=None)
```

Element-wise bitwise XOR with a scalar, dispatched by input type.

See [`xor`][pypto.language.xor] for why only the tile path takes ``tmp``.

## `shl`

```text
pypto.language.shl(lhs, rhs)
```

Element-wise bitwise left shift, dispatched by input type.

## `shls`

```text
pypto.language.shls(lhs, rhs)
```

Element-wise bitwise left shift by a scalar, dispatched by input type.

## `shr`

```text
pypto.language.shr(lhs, rhs)
```

Element-wise bitwise right shift, dispatched by input type.

## `shrs`

```text
pypto.language.shrs(lhs, rhs)
```

Element-wise bitwise right shift by a scalar, dispatched by input type.

## `maximums`

```text
pypto.language.maximums(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
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
pypto.language.minimums(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
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
pypto.language.prelu(tile: pypto.language.typing.tile.Tile, slope: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.not_(input: ~T) -> ~T
```

Element-wise bitwise NOT, dispatched by input type (int16/uint16 only).

## `addc`

```text
pypto.language.addc(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.subc(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.addsc(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.subsc(lhs: pypto.language.typing.tile.Tile, rhs: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.lrelu(tile: pypto.language.typing.tile.Tile, slope: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
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
pypto.language.sel(mask: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile
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
pypto.language.sels(mask: pypto.language.typing.tile.Tile, src: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, scalar: int | float | pypto.pypto_core.ir.Expr | pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile
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

## `tri`

```text
pypto.language.tri(diagonal: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int], valid_shape: collections.abc.Sequence[int] | None = None, dtype: pypto.pypto_core.DataType = int32, upper: bool = False) -> pypto.language.typing.tile.Tile
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

## `quant_mx`

```text
pypto.language.quant_mx(src: pypto.language.typing.tile.Tile, *, group_axis: int, dtype: pypto.pypto_core.DataType = fp8e4m3fn) -> tuple[pypto.language.typing.tile.Tile, pypto.language.typing.tile.Tile]
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
pypto.language.tmov_x2zz(src: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, *, group_axis: int = 1, dst_rows: int | None = None, dst_cols: int | None = None) -> pypto.language.typing.tile.Tile
```

Exponent X-to-ZZ layout conversion (A5-only).

``tmp`` is a write-only workspace. Axis1 requires capacity
``64 + ceil(rows/16) * cols`` bytes (typically 32-byte-aligned by the
caller) and ``dst_rows``/``dst_cols`` for the ZZ ``[M, G]`` shape. Axis0
(TMovDnTo2Zz) still requires a Vec tmp operand; use a minimal 32-byte-aligned
pad.

## `AUTO`

```text
pypto.language.AUTO
```

## `tpush_to_aiv`

```text
pypto.language.tpush_to_aiv(tile: pypto.language.typing.tile.Tile, *, split: int, lane_stride: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
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
pypto.language.tpush_to_aic(tile: pypto.language.typing.tile.Tile, *, split: int, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
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

## `aiv_shard`

```text
pypto.language.aiv_shard(x: ~_SplitOperandT, span: pypto.pypto_core.ir.Span | None = None) -> ~_SplitOperandT
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
pypto.language.aic_gather(x: ~_SplitOperandT, span: pypto.pypto_core.ir.Span | None = None) -> ~_SplitOperandT
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

## `tpop_from_aic`

```text
pypto.language.tpop_from_aic(*, shape: list[int] | None = None, dtype: pypto.pypto_core.DataType | None = None, split: int = 0, lane_stride: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.tile.Tile
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
pypto.language.tpop_from_aiv(*, shape: list[int] | None = None, dtype: pypto.pypto_core.DataType | None = None, split: int = 0, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.tile.Tile
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
pypto.language.aic_initialize_pipe(c2v_consumer_buf: pypto.pypto_core.ir.Expr | int | float | pypto.ir.op.system_ops._UnwrapsToExpr = 0, v2c_consumer_buf: pypto.pypto_core.ir.Expr | int | float | pypto.ir.op.system_ops._UnwrapsToExpr = 0, *, dir_mask: int, slot_size: int, slot_num: int | None = None, local_slot_num: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
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
pypto.language.aiv_initialize_pipe(c2v_consumer_buf: pypto.pypto_core.ir.Expr | int | float | pypto.ir.op.system_ops._UnwrapsToExpr = 0, v2c_consumer_buf: pypto.pypto_core.ir.Expr | int | float | pypto.ir.op.system_ops._UnwrapsToExpr = 0, *, dir_mask: int, slot_size: int, slot_num: int | None = None, local_slot_num: int | None = None, id: int | None = None, span: pypto.pypto_core.ir.Span | None = None) -> pypto.pypto_core.ir.Call
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
pypto.language.reserve_buffer(*, name: str, size: int, base: int = -1, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.scalar.Scalar
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
pypto.language.import_peer_buffer(*, name: str, peer_func: str, span: pypto.pypto_core.ir.Span | None = None) -> pypto.language.typing.scalar.Scalar
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
pypto.language.tfree_to_aic(tile: pypto.language.typing.tile.Tile, span: pypto.pypto_core.ir.Span | None = None, *, split: int | None = None, id: int | None = None) -> pypto.pypto_core.ir.Call
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
pypto.language.tfree_to_aiv(tile: pypto.language.typing.tile.Tile, span: pypto.pypto_core.ir.Span | None = None, *, split: int | None = None, id: int | None = None) -> pypto.pypto_core.ir.Call
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

## `assemble`

```text
pypto.language.assemble(target, source, offset, *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_)
```

Write ``source`` into ``target`` at ``offset``, dispatched by target type.

``atomic`` is Tensor-only: the combine lowers to an atomic-add store into
global memory, which a tile-to-tile assemble has no destination for. Passing
the documented default keeps working on both paths; any other value with a
Tile target raises.

## `cos`

```text
pypto.language.cos(input: ~T) -> ~T
```

Element-wise cosine (input in radians), dispatched by input type. FP32 only.

## `gather_row`

```text
pypto.language.gather_row(dst: ~T, src: pypto.language.typing.tensor.Tensor, dst_offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], src_offset: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], shapes: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], transpose: bool = False, *, valid_shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr] | None = None) -> ~T
```

Gather one GM row into a sub-region of an on-chip accumulator (DPS).

Dispatched on ``dst`` — the destination accumulator. ``src`` is a ``Tensor``
on both paths: this op always reads from global memory, so it is the
destination, not the source, that names the level.

## `mrgsort`

```text
pypto.language.mrgsort(src0: ~T, src1: Optional[~T] = None, src2: Optional[~T] = None, src3: Optional[~T] = None, tmp: Optional[~T] = None, *, exhausted: bool = False, block_len: int | pypto.language.typing.scalar.Scalar | None = None) -> ~T
```

Merge sort — format1 (single-list) or format2 (2-4 way), dispatched by input type.

``tmp`` is Tile-only: at tensor level the scratch buffer is synthesized
during Tensor-to-Tile lowering, so passing one is rejected rather than
silently dropped. The tile path's format2 requires it, and ``pl.tile.mrgsort``
raises with the per-format guidance when it is missing.

``exhausted`` is keyword-only here. The tile wrapper also accepts it as a
sixth positional argument; that spelling stays available as
``pl.tile.mrgsort(...)``.

## `scatter_update`

```text
pypto.language.scatter_update(input: ~T, *args: Any, **kwargs: Any) -> ~T
```

Update rows at positions given by a 2D index, dispatched by input type.

Accepts the same flexible call shapes as either level's wrapper — the
positional/keyword forms are identical on both, so the arguments are
forwarded unchanged.

## `sin`

```text
pypto.language.sin(input: ~T) -> ~T
```

Element-wise sine (input in radians), dispatched by input type. FP32 only.

## `sort32`

```text
pypto.language.sort32(src: ~T, idx: ~T) -> ~T
```

Sort fixed 32-element blocks, permuting ``idx`` alongside ``src``.

Dispatched by input type. Returns 8-byte value-index pairs; the last
dimension is 2x the input width for FP32 and 4x for FP16.

## `create_tensor`

```text
pypto.language.create_tensor(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, layout: pypto.pypto_core.ir.TensorLayout = TensorLayout.ND, manual_dep: bool = False, init_value: int | float | None = None) -> pypto.language.typing.tensor.Tensor
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

## `dim`

```text
pypto.language.dim(tensor: pypto.language.typing.tensor.Tensor, axis: int | pypto.pypto_core.ir.ConstInt) -> pypto.language.typing.scalar.Scalar
```

Extract a shape dimension from a tensor as a scalar value.

Args:
    tensor: Input tensor
    axis: Dimension index (supports negative indexing). Accepts either
        a Python ``int`` or a ``ConstInt`` (parser-shape).

Returns:
    Scalar wrapping the dim operation (INT64)

## `full`

```text
pypto.language.full(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, value: int | float) -> pypto.language.typing.tensor.Tensor
```

Create a tensor of specified shape filled with a constant value.

Args:
    shape: Shape of the tensor
    dtype: Data type of tensor elements
    value: Filling scalar value (int or float)

Returns:
    Tensor wrapping the full operation

## `ScopeMode`

```text
pypto.language.ScopeMode(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Dependency-tracking mode of a runtime scope (``SIMPLER_SCOPE``).

- ``AUTO``: OverlapMap auto-dependency tracking is on (``SIMPLER_SCOPE()``).
- ``MANUAL``: auto tracking is off; the user declares every edge via
  ``pl.submit(..., deps=[...])`` (``SIMPLER_SCOPE(ScopeMode::MANUAL)``).

## `scope`

```text
pypto.language.scope(mode: 'ScopeMode' = <ScopeMode.AUTO: 0>)
```

Context manager marking a runtime scope (``SIMPLER_SCOPE``) region.

A runtime scope is a resource-management + dependency-tracking boundary in
the simpler runtime: it bounds OverlapMap auto-dependency tracking and gives
a per-scope HeapRing level (nested scopes reclaim memory independently). The
simpler runtime provides an implicit top-level scope, so writing scopes is
a **tuning / control** mechanism, never a correctness requirement.

By default the compiler inserts AUTO scopes for you (function body + each
``for`` / ``if`` body). To place scopes yourself, set
``@pl.function(auto_scope=False)`` and use this context manager (and the
``pl.range(..., scope=...)`` sugar). See [`ScopeMode`][pypto.language.ScopeMode] for AUTO vs
MANUAL.

Usage::

    with pl.scope():                          # AUTO
        out = self.kernel(a, b, out)

    with pl.scope(mode=pl.ScopeMode.MANUAL):  # MANUAL — user owns deps
        out, tid = pl.submit(self.stage1, x, out)

Rules:
  - Must appear inside an Orchestration function (not InCore).
  - ``mode=AUTO`` is only allowed under ``@pl.function(auto_scope=False)``
    (in the default ``auto_scope=True`` the compiler owns AUTO placement).
  - ``mode=MANUAL`` is allowed in either mode (it is a dependency-semantics
    choice, not ring tuning).
  - AUTO scope may not nest inside a MANUAL scope (runtime forbids).

## `manual_scope`

```text
pypto.language.manual_scope()
```

Alias for ``pl.scope(mode=pl.ScopeMode.MANUAL)``.

Turns OverlapMap auto dep-tracking off for a region.

Inside this block, the simpler runtime skips OverlapMap lookup and insert
for every kernel submit, so the user takes full responsibility for
declaring task-to-task ordering edges. ``manual_scope`` is the
coarsest-grained of the runtime's auto-dep-tracking opt-outs; finer
granularities are available and compose with auto scope:

  - ``with pl.manual_scope():`` — this construct; whole-region opt-out.
  - ``pl.create_tensor(..., manual_dep=True)`` — opt out a single tensor
    for its entire lifetime (any task referencing it skips OverlapMap).
  - ``pl.no_dep(arg)`` at a kernel-call arg position — opt out a single
    tensor for a single task only.

Manual dependency edges (``pl.submit(..., deps=[...])``) are **orthogonal**
to all of the above: the runtime adds them on top of whatever auto-tracked
deps remain (final fanin = auto ∪ explicit), so ``deps=`` works in auto
scope too. Use ``manual_scope`` only when you want full ownership of the
dep graph; otherwise stay in auto scope and use ``pl.submit(..., deps=)``
as a precision tool that patches the edges auto cannot infer.

Usage::

    # Full-manual region.
    with pl.manual_scope():
        scratch, tid = pl.submit(self.stage1, x, scratch)
        out, _       = pl.submit(self.stage2, scratch, out, deps=[tid])

    # Auto scope with explicit-edge patching (no manual_scope needed).
    a, a_tid = pl.submit(self.k1, x)
    b, _     = pl.submit(self.k2, x, deps=[a_tid])

Restrictions:
  - Must appear inside an Orchestration function (not InCore).
  - Cannot be nested inside another ``manual_scope`` (runtime forbids).

## `submit`

```text
pypto.language.submit(*args: Any, **kwargs: Any) -> Any
```

Submit a kernel and capture its producer TaskId.

``pl.submit`` is a **parser construct**, not a runtime function — the
DSL parser intercepts ``result, tid = pl.submit(self.kernel, *args,
deps=[...])`` syntactically and never actually calls this body. It is
defined only so the name resolves (for imports / linters).

Surface form (must be unpacked as a 2-tuple)::

    out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid])
    (a, b), tid = pl.submit(self.multi_out_kernel, x)
    out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid], dumps=[x])
    out, tid    = pl.submit(self.stage1, x, scratch, allow_early_resolve=True)

The kernel-side ``ir.Submit`` natively returns
``Tuple[<kernel return>, TASK_ID]``; element 0 is the tensor result(s),
element 1 is the producer TaskId (``Scalar[TASK_ID]``). The optional
``deps=[...]`` kwarg lists TaskId scalars / arrays this submit must
wait on. The single-LHS form ``res = pl.submit(...)`` is also accepted
and binds the whole flat tuple to one variable.

``pl.submit`` and its ``deps=`` kwarg work in **both auto and manual
scope**: ``Arg::set_dependencies`` is orthogonal to OverlapMap
auto-tracking (final fanin = auto ∪ explicit). In auto scope, use
``deps=[...]`` as a precision tool to patch edges the runtime cannot
infer; in ``pl.manual_scope()``, use it to declare every edge.

In a distributed HOST orchestrator, the returned TaskId is backed by the
L3 runtime's opaque ``TaskHandle``. Each explicit ``deps=[producer]`` entry
lowers to ``TaskArgs.add_dep_wait(producer)`` and therefore adds ordering
without extending the producer's resource lifetime, on top of the
automatically maintained per-rank communication ordering. This HOST form
accepts individual TaskIds and requires every callee argument, including
Out/InOut tensors, because L3 does not allocate output tensors.

The optional ``dumps=[...]`` kwarg is the submit-side selective tensor
dump surface (symmetric with ``deps=``): it lists tensor arguments of
this submit to mark for dump (simpler#844), so an enabled dump pipeline
filters down to just those bindings. Each entry must be a tensor passed
positionally to the submitted kernel. ``dumps=`` is the explicit dump
surface on a submit; the declarative ``pl.dump_tag(t)`` statement feeds the
same ``dump_vars`` set. These marks only take effect under partial dump
(``RunConfig.enable_dump_args == 1``); they are a no-op when dump is off
(``0``) and irrelevant under full dump (``2``), which captures every
binding regardless.

The optional ``allow_early_resolve=True`` kwarg (default ``False``) opts
this task in as a speculative early-dispatch producer (simpler#1065): the
scheduler may pre-stage this task's consumers onto idle cores before it
completes, releasing them with a doorbell the instant it finishes. It is a
producer-side hint — a consumer only pre-stages once *all* of its producers
are flagged (or already complete). It lowers to
``Arg::set_allow_early_resolve(true)`` in orchestration codegen and is a
pure scheduling optimisation (no effect on results). Pays off on critical
paths built from many short tasks; harmless otherwise.

The optional ``predicate=(...)`` kwarg attaches a **dispatch predicate** the
scheduler evaluates at this task's dispatch point — after its dependencies
are satisfied, so the value is current without an orchestration-time wait.
When the comparison is false the task is retired inline (never dispatched to
a core) while its fanin/fanout still settle, so downstream consumers still
unlock. Use it to skip work whose need is only known at runtime (e.g. an MoE
expert with an empty row count)::

    out, tid = pl.spmd_submit(self.expert_ffn, tokens, out, core_num=N,
                              deps=[gather_tid],
                              predicate=(row_count[e] > 0))

The comparison is matched **syntactically, never evaluated**: in this
position ``row_count[e] > 0`` is a declarative spec handed to the scheduler,
not a ``tensor.read`` plus a compare. Only ``tensor[indices] OP
int-literal`` is expressible (one comparison; ``==`` ``!=`` ``>`` ``<``
``>=`` ``<=``), mirroring the runtime's single-comparison predicate — no
chained comparisons, arithmetic, or boolean combination. Reduce anything
richer to a single gate value in a prior kernel and predicate on that.

**Contract:** the operand tensor's producer must be one of this submit's
``deps=`` (the parser enforces this where statically provable), so the
dispatch-point read observes the current value.

The return annotation is ``Any`` (not ``NoReturn``) because the parser
intercepts the call and binds a 2-tuple to the LHS — downstream code
that does ``out, tid = pl.submit(...)`` would not type-check under
``NoReturn``.

## `spmd_submit`

```text
pypto.language.spmd_submit(*args: Any, **kwargs: Any) -> Any
```

Launch a kernel as an SPMD task and capture its producer TaskId.

``pl.spmd_submit`` is a **parser construct**, not a runtime function — the
DSL parser intercepts ``result, tid = pl.spmd_submit(self.kernel, *args,
core_num=N, sync_start=..., deps=[...])`` syntactically and never actually
calls this body. It is defined only so the name resolves (imports / linters).

It is the SPMD sibling of [`submit`][pypto.language.submit]: a single orchestration task that
the runtime fans out across ``core_num`` logical blocks (each kernel reads
its block index via ``pl.tile.get_block_idx()``). Like [`submit`][pypto.language.submit] it
returns one producer TaskId, so the whole dispatch can be named as a
dependency of later tasks.

Surface form (must be unpacked as a 2-tuple)::

    out, tid = pl.spmd_submit(self.incore_kernel, x, y, core_num=8)
    out, tid = pl.spmd_submit(self.kernel, x, core_num=8, sync_start=True,
                              deps=[prev_tid])

``core_num`` is a **required keyword argument** (a positive integer
expression) — the positional slots are the kernel's own arguments.
``sync_start`` (default ``False``) requires all blocks to launch atomically.
``deps=[...]``, ``allow_early_resolve=True``, ``timing_slot=N`` and
``predicate=(...)`` work exactly as on [`submit`][pypto.language.submit]
(note: a ``sync_start`` task cannot itself be
block-by-block pre-staged, but it can still be flagged to let its consumers
pre-stage). The callee may be an InCore / AIC / AIV kernel or a co-scheduled
Group.

Like [`submit`][pypto.language.submit], it works in both auto and manual scope; its primary use
is explicit dependency wiring inside ``pl.manual_scope()``.

## `no_dep`

```text
pypto.language.no_dep(tensor: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
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
pypto.language.dump_tag(tensor: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor
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
pypto.language.set_cache_policy(tensor: pypto.language.typing.tensor.Tensor, policy: pypto.pypto_core.ir.CachePolicy) -> None
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

## `scatter`

```text
pypto.language.scatter(input: pypto.language.typing.tensor.Tensor, dim: int | None = None, index: pypto.language.typing.tensor.Tensor | None = None, src: pypto.language.typing.tensor.Tensor | None = None, *, mask_pattern: int | None = None, dst: pypto.language.typing.tensor.Tensor | None = None) -> pypto.language.typing.tensor.Tensor
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

## `arange`

```text
pypto.language.arange(start: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType = int32, descending: bool = False) -> pypto.language.typing.tensor.Tensor
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

## `gather`

```text
pypto.language.gather(input: pypto.language.typing.tensor.Tensor, dim: int | None = None, index: pypto.language.typing.tensor.Tensor | None = None, *, mask_pattern: int | None = None, output_dtype: int | pypto.pypto_core.DataType | None = None, kvalue: int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr | None = None, cmp_mode: str | int | None = None, out_cols: int | None = None, offset: int = 0, count_dtype: int | pypto.pypto_core.DataType | None = None) -> pypto.language.typing.tensor.Tensor | tuple[pypto.language.typing.tensor.Tensor, pypto.language.typing.tensor.Tensor]
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
pypto.language.paged_gather(src: pypto.language.typing.tensor.Tensor, indices: pypto.language.typing.tensor.Tensor, block_table: pypto.language.typing.tensor.Tensor, block_size: int, size: int, max_indices: int, *, space: pypto.pypto_core.ir.MemorySpace = MemorySpace.Mat, col_off: int = 0, is_trans: bool = False, is_b_matrix: bool = False) -> pypto.language.typing.tensor.Tensor
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

## `random`

```text
pypto.language.random(key0: int | pypto.language.typing.scalar.Scalar, key1: int | pypto.language.typing.scalar.Scalar, counter0: int | pypto.language.typing.scalar.Scalar, counter1: int | pypto.language.typing.scalar.Scalar, counter2: int | pypto.language.typing.scalar.Scalar, counter3: int | pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType = uint32, rounds: int = 10) -> pypto.language.typing.tensor.Tensor
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

## `create_l1`

```text
pypto.language.create_l1(shape: collections.abc.Sequence[int | pypto.language.typing.scalar.Scalar | pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, transpose: bool = False) -> pypto.language.typing.tensor.Tensor
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

## `get_block_idx`

```text
pypto.language.get_block_idx() -> pypto.language.typing.scalar.Scalar
```

Get the current block index (tensor-scope alias of ``pl.tile.get_block_idx``).

Lowers to ``tile.get_block_idx`` in ``ConvertTensorToTileOps``.

Returns:
    Scalar wrapping the ``tensor.get_block_idx`` operation (INDEX type)

Example:
    >>> block_idx = pl.tensor.get_block_idx()

## `get_block_num`

```text
pypto.language.get_block_num() -> pypto.language.typing.scalar.Scalar
```

Get the total number of blocks in the current SPMD task.

Tensor-scope alias of ``pl.tile.get_block_num``; lowers to
``tile.get_block_num`` in ``ConvertTensorToTileOps``.

Returns:
    Scalar wrapping the ``tensor.get_block_num`` operation (INDEX type)

Example:
    >>> block_idx = pl.tensor.get_block_idx()
    >>> block_num = pl.tensor.get_block_num()

## `get_subblock_idx`

```text
pypto.language.get_subblock_idx() -> pypto.language.typing.scalar.Scalar
```

Get the current sub-block (vector core) index (tensor-scope alias of ``pl.tile.get_subblock_idx``).

Lowers to ``tile.get_subblock_idx`` in ``ConvertTensorToTileOps``.

Returns:
    Scalar wrapping the ``tensor.get_subblock_idx`` operation (INDEX type)

## `FunctionType`

```text
pypto.language.FunctionType(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Function type classification

## `ForKind`

```text
pypto.language.ForKind(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

For loop kind classification

## `AccPhase`

```text
pypto.language.AccPhase(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Producer-side unit-flag phase for GEMV accumulator operations

## `AtomicType`

```text
pypto.language.AtomicType(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Combine mode for global-memory writes — pld.tensor.put (TPUT) and tile.store (TSTORE)

## `CachePolicy`

```text
pypto.language.CachePolicy(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

GM cache-access policy declared for a tensor read

## `KernelType`

```text
pypto.language.KernelType(value, names=None, *, module=None, qualname=None, type=None, start=1)
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
pypto.language.SyncAllMode(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Barrier implementation selected by ``system.syncall``.

- ``HARD``: FFTS barrier with no operands; requires full-core occupancy.
- ``SOFT``: GM-polling barrier; works at partial occupancy.

## `Level`

```text
pypto.language.Level(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Hierarchy level in the Linqu machine model

## `MemRef`

```text
pypto.language.MemRef(*args: Any, **kwargs: Any) -> None
```

DSL-level memory reference accepting ``pl.Ptr`` bases and Scalar offsets.

Identical to ``ir.MemRef`` at runtime. The overloads only widen what
pyright accepts so that printed IR — which uses ``pl.Ptr``-annotated
base variables and ``Scalar`` arithmetic byte offsets — type-checks
cleanly when re-loaded as a ``@pl.program``.

Called with **no offset and size**, it declares an allocation of your own
rather than describing an existing one: size and address are left for
``InitMemRef`` to derive, and the compiler's opportunistic reuse never packs
anything else into it. The allocation takes the name of the variable it is
bound to, so the name is written once::

    scratch = pl.MemRef()

    t0: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.load(x, [0, 0], [64, 64])
    t1: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.exp(t0)

Pass ``slots=N`` for N equally-sized slots of one allocation, then pick one by
subscript. The slots are contiguous and identically sized, so rotating through
them is a ping-pong the packer cannot collapse::

    l0c = pl.MemRef(slots=2)

    ping: pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Acc] = pl.tile.matmul(q, b0)
    pong: pl.Tile[[64, 64], pl.FP32, l0c[1], pl.Mem.Acc] = pl.tile.matmul(q, b1)

The subscript is an ordinary index expression, so it may be a **runtime**
value — a rotation needs no unrolling::

    for i in pl.range(N):
        t: pl.Tile[[64, 64], pl.FP32, l0c[i % 2], pl.Mem.Acc] = pl.tile.matmul(q, b)

A constant index folds into a static address; a runtime one becomes the tile's
address at run time. Nothing here is tied to a particular loop form — the
index is just an expression, so `pl.range`, a `pl.pipeline` sub-index, or a
function parameter all work.

Reference it by variable, so a misspelling is a ``NameError`` rather than a
second allocation. Since the variable *is* the name, one declaration may not
be reached through two names (``b = a``) and two declarations may not claim
one name; both are rejected. ``pl.MemRef("other")`` names it explicitly,
overriding the variable — that is the form the IR printer emits, so a dumped
program reparses without a surrounding Python scope.

Co-liveness is checked **per slot**: two tiles on different slots are meant to
be live together (that is the ping-pong), and only two tiles landing on the
*same* slot can corrupt each other. A runtime index has no static slot to
attribute a tile to, so the check is skipped there — the rotation is yours to
get right — while isolation from every other allocation still holds. Tiles
sharing one declared allocation must also agree on memory space.

Declaring an allocation inside a ``pl.pipeline(stage=2)`` body is rejected —
the cloned stages would make a tile co-live with itself. Declaring slots and
asking the compiler to multi-buffer are alternatives, not layers: to
hand-manage a level, drive it with ``pl.range`` and give it its own slots.

Note: ``pl.MemRef(...)`` calls inside a ``@pl.program`` body are resolved
by the parser (``parser/type_resolver.py``), not dispatched through this
``__init__``. A ``Scalar`` byte offset is therefore only ever seen by
pyright; it never reaches the underlying ``ir.MemRef`` constructor.

## `Role`

```text
pypto.language.Role(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Function role at L3-L7 hierarchy levels

## `SplitMode`

```text
pypto.language.SplitMode(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Split mode for cross-core data transfer

## `STPhase`

```text
pypto.language.STPhase(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Consumer-side unit-flag phase for tile.store

## `Mem`

```text
pypto.language.Mem(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Memory space enumeration

## `MemRefType`

```text
pypto.language.MemRefType()
```

Opaque sentinel type for ``MemRef``-typed variables in printed IR.

The C++ printer emits ``pl.MemRefType`` as the annotation for a bare
``MemRef`` variable (the SSA-edge type of a ``MemRef`` expression).  This
class exists solely so that the printed Python source is valid Python that
the text-parser can ``exec()``.

Note: this is *not* the type of a ``tile.alloc`` / ``tensor.alloc`` result —
those produce a base ``Ptr`` (``PtrType``); see [`alloc`][pypto.language.tile.alloc].

## `MemorySpace`

```text
pypto.language.MemorySpace(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Memory space enumeration

## `PipeType`

```text
pypto.language.PipeType(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Pipeline type enumeration

## `Ptr`

```text
pypto.language.Ptr(*, expr: pypto.pypto_core.ir.Expr | None = None) -> None
```

DSL wrapper for an ``ir.PtrType``-valued expression.

Construct without arguments to obtain an annotation-only placeholder
(``buf: pl.Ptr``). Construct with ``expr=`` to wrap an IR ``Call``
returned by an allocation op.

### `Ptr.unwrap`

```text
Ptr.unwrap(self) -> pypto.pypto_core.ir.Expr
```

Return the wrapped ``ir.Expr``.

Raises:
    RuntimeError: If the instance was constructed without an
        ``expr`` (annotation-only — never returned by a wrapper).

## `PrefetchAsyncContext`

```text
pypto.language.PrefetchAsyncContext(*, expr: pypto.pypto_core.ir.Expr | None = None) -> None
```

Handle to an asynchronous GM->L2 prefetch context.

Produced by [`pl.prefetch.make_context`][pypto.language.prefetch.make_context]; consumed by
[`pl.prefetch.async_prefetch`][pypto.language.prefetch.async_prefetch] and
[`pl.prefetch.session`][pypto.language.prefetch.session].

## `AsyncEvent`

```text
pypto.language.AsyncEvent(*, expr: pypto.pypto_core.ir.Expr | None = None) -> None
```

Handle to an in-flight asynchronous prefetch completion event.

Produced by [`pl.prefetch.async_prefetch`][pypto.language.prefetch.async_prefetch]; consumed by
[`pl.prefetch.wait`][pypto.language.prefetch.wait] together with the matching
[`AsyncSession`][pypto.language.AsyncSession].

## `AsyncSession`

```text
pypto.language.AsyncSession(*, expr: pypto.pypto_core.ir.Expr | None = None) -> None
```

Handle to the asynchronous session an [`AsyncEvent`][pypto.language.AsyncEvent] belongs to.

Produced by [`pl.prefetch.session`][pypto.language.prefetch.session]; consumed by
[`pl.prefetch.wait`][pypto.language.prefetch.wait].

## `TensorLayout`

```text
pypto.language.TensorLayout(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Tensor layout enumeration

## `TensorView`

```text
pypto.language.TensorView(stride: collections.abc.Sequence[pypto.pypto_core.ir.Expr | int] | None = None, layout: pypto.pypto_core.ir.TensorLayout | None = None, valid_shape: collections.abc.Sequence[pypto.pypto_core.ir.Expr | int] | None = None, pad: pypto.pypto_core.ir.PadValue = PadValue.null) -> '_TensorViewBase'
```

TensorView factory: accepts Expr or int in stride/valid_shape.

## `TileLayout`

```text
pypto.language.TileLayout(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Tile layout enumeration

## `PadValue`

```text
pypto.language.PadValue(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Pad mode enumeration for tile/tensor views

## `CompactMode`

```text
pypto.language.CompactMode(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Partial-tile compact mode enumeration

## `TileView`

```text
pypto.language.TileView(valid_shape: collections.abc.Sequence[pypto.pypto_core.ir.Expr | int] | None = None, stride: collections.abc.Sequence[pypto.pypto_core.ir.Expr | int] | None = None, start_offset: pypto.pypto_core.ir.Expr | int | None = None, blayout: pypto.pypto_core.ir.TileLayout = TileLayout.row_major, slayout: pypto.pypto_core.ir.TileLayout = TileLayout.none_box, fractal: int = 512, pad: pypto.pypto_core.ir.PadValue = PadValue.null, compact: pypto.pypto_core.ir.CompactMode = CompactMode.null) -> '_TileViewBase'
```

TileView factory: accepts Expr or int in valid_shape/stride.

## `ND`

```text
pypto.language.ND
```

## `DN`

```text
pypto.language.DN
```

## `NZ`

```text
pypto.language.NZ
```

## `MX_A_ZZ`

```text
pypto.language.MX_A_ZZ
```

## `MX_B_NN`

```text
pypto.language.MX_B_NN
```

## `FP4`

```text
pypto.language.FP4
```

## `FP8E4M3FN`

```text
pypto.language.FP8E4M3FN
```

## `FP8E5M2`

```text
pypto.language.FP8E5M2
```

## `FP8E8M0`

```text
pypto.language.FP8E8M0
```

## `FP16`

```text
pypto.language.FP16
```

## `FP32`

```text
pypto.language.FP32
```

## `BF16`

```text
pypto.language.BF16
```

## `HF4`

```text
pypto.language.HF4
```

## `HF8`

```text
pypto.language.HF8
```

## `INT4`

```text
pypto.language.INT4
```

## `INT8`

```text
pypto.language.INT8
```

## `INT16`

```text
pypto.language.INT16
```

## `INT32`

```text
pypto.language.INT32
```

## `INT64`

```text
pypto.language.INT64
```

## `UINT4`

```text
pypto.language.UINT4
```

## `UINT8`

```text
pypto.language.UINT8
```

## `UINT16`

```text
pypto.language.UINT16
```

## `UINT32`

```text
pypto.language.UINT32
```

## `UINT64`

```text
pypto.language.UINT64
```

## `BOOL`

```text
pypto.language.BOOL
```

## `INDEX`

```text
pypto.language.INDEX
```

## `TASK_ID`

```text
pypto.language.TASK_ID
```

## `TaskId`

```text
pypto.language.TaskId
```
