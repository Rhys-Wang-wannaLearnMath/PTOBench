# `pypto.runtime` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `benchmark`

```text
pypto.runtime.benchmark(compiled: Any, args: collections.abc.Sequence[typing.Any], *, rounds: int = 100, warmup: int = 3, platform: str | None = None, device_id: int | None = None, config: pypto.runtime.runner.RunConfig | None = None, persistent: bool = False, reset_persistent_windows: bool | None = None) -> pypto.runtime.bench.BenchmarkStats
```

Register *compiled* once and dispatch *rounds* timed launches.

Dispatches by *compiled* type:

- **L2** (:class:`~pypto.ir.CompiledProgram`): opens a single
  :class:`~pypto.runtime.ChipWorker` with the capabilities recorded in the
  compiled artifact.
- **L3** (:class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`):
  opens a :class:`~pypto.runtime.distributed_runner.DistributedWorker` via
  ``compiled.prepare()``.

Either way it registers *compiled* once, then loops the bound handle so each
launch only re-pays argument coercion + dispatch (not register/load). The
on-NPU ``device_wall_us`` is measured between the orchestrator's
``orch_start`` / ``orch_end`` and is unaffected by the per-launch host-side
arg building.

Timing is read from the runtime's ``[STRACE]`` stderr markers (simpler PR
#1177): this sets the runtime log level to ``timing`` for the worker's
lifetime (restored afterward) and captures ``stderr`` at the file-descriptor
level, so the emitted stderr is diverted into a temp file rather than shown
live. For L2 the capture wraps only the measured loop; for L3 it must wrap
``compiled.prepare()`` as well, because the chip workers are forked there and
inherit fd 2 at fork time — a redirect set up after the fork would miss the
children's markers entirely. (On L3 failure the diverted setup stderr is
echoed back so diagnostics are not lost.)

Args:
    compiled: A single-orchestration
        :class:`~pypto.ir.CompiledProgram` (L2) or a
        :class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`
        (L3) from ``ir.compile``. Multi-orch L2
        programs must pass ``compiled[<name>]``.
    args: Positional dispatch args, same as ``compiled(*args)``. **L3
        requires shared-memory host** ``torch.Tensor`` **args** (allocated
        with ``.share_memory_()`` and reused in place). This helper creates
        its own prepared Worker, so it cannot accept a DeviceTensor owned by
        another Worker; benchmark resident tensors with an explicit
        ``compiled.prepare()`` dispatch loop.
    rounds: Number of measured launches. Must be positive.
    warmup: Number of leading launches discarded before measurement
        (page-in / cache warm). Total launches = ``warmup + rounds``.
    platform: Target platform shorthand, e.g. ``"a2a3"``. Defaults to
        ``compiled.platform``. Mutually exclusive with *config*. **L2 only** —
        not accepted for L3 (device set fixed at compile time).
    device_id: NPU device index. Defaults to ``RunConfig``'s default.
        Mutually exclusive with *config*. **L2 only** — not accepted for L3
        (device set comes from ``distributed_config.device_ids``).
    config: Optional :class:`~pypto.runtime.RunConfig`. L2: full control
        (``aicpu_thread_num``); pass this
        *or* *platform*/*device_id*, not both. L3: forwarded per dispatch for
        ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
        ``ring_dep_pool``); ``None`` reuses the prepared baseline.
    persistent: L3 only. Reuse retained CommDomains across all warmup and
        measured dispatches while fencing each launch with ``Worker.run``.
    reset_persistent_windows: L3 persistent mode only. Restore retained
        windows to zero before reuse. ``None`` (the default) enables reset
        in persistent mode. Set to ``False`` only when the benchmarked
        program manually clears or otherwise manages all reused
        communication-buffer state.

Returns:
    A :class:`BenchmarkStats` with the per-round ``device_wall_us`` /
    ``host_wall_us`` samples and aggregate helpers. For L3 the samples are
    per-round maxima across ranks; summaries are derived on demand via
    :meth:`BenchmarkStats.per_dispatch` (per ``(pid, slot)`` dispatch — nothing
    summed), :meth:`BenchmarkStats.per_rank` (``device`` / ``host`` /
    ``effective``, that rank's dispatches summed per round) and
    :meth:`BenchmarkStats.per_round` (those plus ``union``), and the
    ``round → rank → [dispatch]`` grid is in
    :attr:`BenchmarkStats.rounds_dispatches`.

Raises:
    ValueError: ``rounds <= 0``, ``warmup < 0``, *config* passed together
        with *platform* / *device_id* (L2), or *platform* / *device_id*
        passed for an L3 program.
    RuntimeError: No ``[STRACE]`` markers were captured at all, so no timing
        could be read. The markers are gated by the runtime's compile-time
        ``SIMPLER_HOST_STRACE`` macro; a runtime built without it emits none.

Note:
    On a ``*sim`` platform the host ``<root>`` span is still emitted but the
    device-domain spans are not, so every ``device_wall_us`` sample is ``0``
    — check :attr:`BenchmarkStats.all_zero_device`.

    L3 has no DAG-level device wall (only the forked chip children emit
    markers). Each round's ``device_wall_us`` is the **max across ranks** of
    that round's per-rank **summed** dispatch device walls — a proxy for round
    device time that excludes inter-dispatch idle gaps and cross-rank start
    skew (device clocks are per-invocation, so gaps/skew are unmeasurable).
    A rank that dispatches more than once per round is therefore summed in the
    headline; use :meth:`BenchmarkStats.per_dispatch` (and the per-dispatch
    mean trees) to keep those dispatches apart.
    ``per_round("union")`` complements it with the cross-rank **host-timeline**
    union window (the ``<root>`` host clocks are ``CLOCK_MONOTONIC``,
    cross-process comparable), which *does* capture overlap / start skew but
    includes host-side dispatch overhead. A true pure-device end-to-end DAG
    wall is not recoverable until the runtime emits a device→host clock
    anchor. Per-round alignment assumes a deterministic dispatch shape
    (constant dispatches per round); if that does not hold,
    :attr:`BenchmarkStats.fallback_flattened` is set, the samples become a
    flattened per-dispatch pool, and ``per_rank`` / ``per_round("union")`` are
    empty.

## `execute_compiled`

```text
pypto.runtime.execute_compiled(work_dir: str | pathlib.Path, args: list[torch.Tensor | pypto.runtime.device_tensor.DeviceTensor | _ctypes._SimpleCData], *, platform: str, device_id: int, dfx: pypto.runtime.runner.DfxOptions = DfxOptions(enable_chip_swimlane=0, enable_dump_args=0, enable_pmu=0, enable_dep_gen=False, enable_scope_stats=False), level: int = 2, aicpu_thread_num: int | None = None, analyze_auto_scopes_for_deps: bool = False, config: pypto.runtime.runner.RunConfig | None = None) -> None
```

Deprecated. Dispatch a build directory without recompiling.

:meth:`pypto.ir.CompiledProgram.from_dir` rebuilds the same handle from the
same directory and dispatches it through the same code path. **The two
disagree on precedence, so the migration is not a plain rename.** Here, an
explicit ``platform`` / ``device_id`` / ``dfx`` / ``aicpu_thread_num`` wins
and ``config`` supplies only the ring overrides. On the artifact path a
supplied ``config`` is the sole source of all four — including ``platform``,
which then shadows ``from_dir(platform=...)``. Since ``RunConfig.platform``
defaults to ``"a2a3sim"``, dropping the explicit arguments would silently
move the run to the simulator. Fold them into the config instead::

    # before -- explicit args win; ``cfg`` was read for ring sizing only
    execute_compiled(work_dir, args, platform="a2a3", device_id=0, config=cfg)

    # after -- ``cfg`` carries every execution setting
    cfg = dataclasses.replace(cfg, platform="a2a3", device_id=0)
    ir.CompiledProgram.from_dir(work_dir)(*args, config=cfg)

``dfx`` and ``aicpu_thread_num`` have no separate spelling on that path:
the DFX toggles and ``aicpu_thread_num`` are already ``RunConfig`` fields,
read fresh on every dispatch. ``from_dir(platform=...)`` still decides the
platform for a call made *without* a config.

Emits a :class:`DeprecationWarning` and forwards to the same implementation
the artifact path uses. Behaviour of this function is unchanged; only the
name is going away.

## `execute_distributed_compiled`

```text
pypto.runtime.execute_distributed_compiled(output_dir: 'str | Path', args: 'Sequence[torch.Tensor | DeviceTensor | StackedDeviceTensor | ctypes._SimpleCData]', config: 'RunConfig | None' = None, *, platform: 'str | None' = None, distributed_config: 'DistributedConfig | None' = None) -> 'torch.Tensor | DeviceTensor | StackedDeviceTensor | tuple[torch.Tensor | DeviceTensor | StackedDeviceTensor, ...] | None'
```

Reconstruct a distributed program from ``output_dir`` and run it once.

Deprecated. It reconstructs a
:class:`~pypto.ir.DistributedCompiledProgram` from an already-compiled
build directory (via :meth:`DistributedCompiledProgram.from_dir`) and
dispatches it once — **without** re-running the pypto compile. Both steps
are public, so the replacement is the two lines this function wraps::

    # before
    execute_distributed_compiled(work_dir, args, config=cfg, platform="a2a3")

    # after
    ir.DistributedCompiledProgram.from_dir(work_dir, platform="a2a3")(*args, config=cfg)

Emits a :class:`DeprecationWarning`; behaviour is unchanged.

Args:
    output_dir: A build directory produced by a prior ``ir.compile`` of a
        distributed (L3+) program (must contain ``distributed_meta.json``).
    args: Host ``torch.Tensor`` arguments matching the orchestrator's
        parameter order (in-place, or input-only for a return-style
        program). Resident tensors require a prepared
        :class:`DistributedWorker` and are rejected here.
    config: Optional per-dispatch :class:`RunConfig`, forwarded to
        ``__call__``. Its per-task ring-sizing overrides size this dispatch's
        runtime ring buffers, and its runtime-diagnostic DFX flags
        (``enable_dump_args`` / ``enable_pmu`` / ``enable_dep_gen`` /
        ``enable_scope_stats`` / ``enable_chip_swimlane``) are written per
        dispatch under ``<output_dir>/dfx_outputs/rank{r}/d{k}/``. Other
        compile-side fields are not consumed on the dispatch path.
    platform: Override the persisted platform (e.g. ``a2a3sim`` → ``a2a3``).
    distributed_config: Override the persisted run config (e.g. a different
        set of ``device_ids``).

Returns:
    The call result: allocated output tensor(s) for a return-style program,
    otherwise ``None`` (outputs written in place into the passed arguments).

## `configure_log`

```text
pypto.runtime.configure_log(level: int | str, *, sync_pypto: bool = False) -> None
```

Set the PyPTO runtime log threshold (and optionally PyPTO's C++ logger too).

Args:
    level: Python ``logging`` int (e.g. ``20``) or string (``"debug"``,
        ``"info"``, ``"timing"``, ``"warn"``, ``"error"``, ``"null"``).
        Case-insensitive. See :data:`simpler_setup.log_config.LOG_LEVEL_CHOICES`.
    sync_pypto: When ``True``, also push the closest PyPTO ``LogLevel`` to
        the C++ side so both subsystems display the same band. Defaults to
        ``False`` because the two loggers are intentionally independent.

## `log_level`

```text
pypto.runtime.log_level() -> int
```

Return simpler's effective Python ``logging`` threshold.

## `ensure_pto_isa_root`

```text
pypto.runtime.ensure_pto_isa_root() -> str
```

Resolve the pinned PTO-ISA checkout, cloning it when necessary.

Delegates to :func:`simpler_setup.pto_isa.ensure_pto_isa_root`, for which
``pto_isa.pin`` is the single source of truth: it reuses a managed checkout
only when that checkout is clean and already at the pin, re-clones it
otherwise, serializes concurrent resolutions with a file lock, and verifies
``HEAD`` before returning.

The resolved path is *exported* as ``PTO_ISA_ROOT`` for downstream consumers
that build extern CCE kernels and need the ISA include directory. It is
never *read* back: an ambient value is not the pin.

Returns:
    Absolute path to the pinned PTO-ISA checkout.

Raises:
    RuntimeError: If ``pto_isa.pin`` is missing or malformed.
    OSError: If the pinned checkout cannot be obtained.

## `pto_isa_include_dir`

```text
pypto.runtime.pto_isa_include_dir() -> pathlib.Path
```

Return the include directory of the pinned PTO-ISA checkout.

Use this instead of reading ``PTO_ISA_ROOT`` when a ``pl.jit.extern`` kernel
or an out-of-tree build needs the PTO-ISA headers -- it resolves the pin on
demand rather than trusting whatever the ambient environment happens to say.

Example::

    from pypto.runtime import pto_isa_include_dir

    _INCLUDE = pto_isa_include_dir()

Returns:
    Absolute path to ``<pto-isa>/include``.

Raises:
    RuntimeError: If ``pto_isa.pin`` is missing or malformed.
    OSError: If the pinned checkout cannot be obtained.

## `BenchmarkStats`

```text
pypto.runtime.BenchmarkStats(device_wall_us: list[float] = <factory>, host_wall_us: list[float] = <factory>, rounds: int = 0, warmup: int = 0, invocations: list[pypto.runtime.bench.TraceInvocation] = <factory>, rounds_dispatches: list[dict[int, list[pypto.runtime.bench.TraceInvocation]]] = <factory>, fallback_flattened: bool = False, unstable_dispatch_slots: bool = False) -> None
```

Aggregated per-launch timing from :func:`benchmark`.

**Quick-reference:**

| Accessor | Description |
| --- | --- |
| `stats.device_us_median` | Median device wall (µs). |
| `stats.device_us_min` | Minimum device wall (µs). |
| `stats.device_us_max` | Maximum device wall (µs). |
| `stats.device_us_mean` | Arithmetic mean device wall (µs). |
| `stats.device_us_stdev` | Std-dev of device wall (µs). |
| `stats.all_zero_device` | True if samples exist and every sample is 0 (e.g. sim builds). |
| `stats.samples` | Alias for `device_wall_us` (the raw list). |
| `stats.per_round("device")` | List[float]: device wall per round. |
| `stats.per_rank("device")` | Dict[int, List[float]]: per-rank breakdown (L3). |
| `stats.per_dispatch("device")` | Dict[(pid, slot), List[float]]: per-dispatch (unsummed) view (L3). |
| `stats.dispatch_tasks()` | Dict[(pid, slot), str]: task name (or hash) labelling each dispatch slot. |
| `stats.unstable_dispatch_slots` | True if a rank's dispatch order varies between rounds (L3). |
| `stats.print_tree()` | Render per-dispatch span tree to stdout. |

The min / median / mean / max / stdev helpers operate on
``device_wall_us`` — the on-NPU metric. ``host_wall_us`` samples are kept
for context, but they include per-launch arg coercion + H2D and so are not
the device metric.

The data is organized into three tiers; per-metric summaries are derived on
demand via :meth:`per_round` / :meth:`per_rank` rather than stored as many
parallel fields:

- **Stored headline** (list, length ``rounds``): :attr:`device_wall_us`,
  :attr:`host_wall_us`. Same as ``per_round("device")`` / ``per_round("host")``.
- **Stored raw detail**: :attr:`rounds_dispatches` (the L3 ``round → rank →
  [dispatch]`` grid, the single source for per-dispatch / per-rank /
  effective / union derivations) and :attr:`invocations` (the same dispatches
  flattened, for tree rendering).
- **Derived summaries**: :meth:`per_dispatch` (no fusing — one series per
  ``(pid, slot)`` dispatch), :meth:`per_rank` (that rank's dispatches summed
  per round) and :meth:`per_round`, for the metrics ``device`` / ``host`` /
  ``effective`` (plus ``union`` on ``per_round``).

A rank that dispatches more than once per round is **not** fused by
:meth:`per_dispatch` or by the mean-tree views (:meth:`format_mean_tree`
groups per dispatch by default); :meth:`per_rank` / :meth:`per_round`
deliberately do sum, as a rank's round busy-time.

The min / median / mean / max / stdev helpers operate on
:attr:`device_wall_us` — the on-NPU metric.

Attributes:
    device_wall_us: Per-round on-NPU device wall (µs). L2: one per measured
        launch (the ``<root>.runner_run.device_wall`` span). L3: per-round max
        across ranks of each rank's summed dispatch device walls. Length is
        ``rounds`` (warmup excluded); in the L3 flatten fallback it is instead
        the pooled per-dispatch samples. Equals ``per_round("device")``.
    host_wall_us: Per-round host wall (µs), analogous over the ``<root>``
        span. Equals ``per_round("host")``.
    rounds: Number of measured launches.
    warmup: Number of leading launches discarded before measurement.
    invocations: All measured dispatches' span trees, flattened. L2: one per
        launch; L3: every rank's per-dispatch invocation. Empty when no
        ``[STRACE]`` markers were captured. Render with :meth:`format_tree`.
    rounds_dispatches: L3 only — the navigable ``round → rank → [dispatch]``
        grid: ``rounds_dispatches[k][pid]`` is that rank's
        :class:`TraceInvocation` dispatches in round ``k`` (ordered by
        ``inv``). Each dispatch exposes :attr:`TraceInvocation.task` (the
        ``hid``) and :attr:`TraceInvocation.device_wall_us` / ``host_wall_us``
        / ``effective_us``. It is the single source :meth:`per_dispatch` /
        :meth:`per_rank` / :meth:`per_round` derive from, and its dispatch
        position is the ``slot`` those views key on. Same objects as
        :attr:`invocations`.
        Length is ``rounds``; empty for L2 and the flatten fallback.
    fallback_flattened: L3 only — ``True`` when per-round segmentation was not
        possible (a rank's marker count was not divisible by
        ``warmup + rounds``, i.e. a non-deterministic dispatch shape). Then
        :attr:`device_wall_us` / :attr:`host_wall_us` hold the pooled
        per-dispatch samples (warmup dropped per rank), :attr:`rounds_dispatches`
        is empty, and ``per_rank`` / ``per_round("union")`` return empty.
    unstable_dispatch_slots: L3 only — ``True`` when some ``(pid, slot)``
        carried more than one :attr:`TraceInvocation.task` across the measured
        rounds, i.e. a rank issued a *constant number* of dispatches but not
        always in the same order (round 0 ``A, B`` then round 1 ``B, A``).
        The ordinal slot then does not identify a callable, so grouping by it
        would average distinct kernels under the first round's label — the
        very fusing the per-dispatch views exist to remove. They therefore
        report empty (:meth:`dispatch_groups`, :meth:`per_dispatch`,
        :meth:`dispatch_tasks`) and :meth:`format_mean_tree` explains why.
        Round boundaries are unaffected, so :attr:`rounds_dispatches`,
        :meth:`per_rank` and :meth:`per_round` stay valid and populated.

### `BenchmarkStats.dispatch_groups`

```text
BenchmarkStats.dispatch_groups(self) -> dict[tuple[int, int], list[pypto.runtime.bench.TraceInvocation]]
```

Per-dispatch invocation series: ``{(pid, slot): [round0, round1, ...]}``.

*slot* is the dispatch's position within its rank's round (dispatches are
ordered by ``inv``). Keys are sorted by ``(pid, slot)``.

This is the un-fused view: a rank issuing several dispatches per round
gets one entry per dispatch instead of a single summed number. Derived
from :attr:`rounds_dispatches`, so it is **L3 only** — returns ``{}`` for
L2 and for the flatten fallback, and also when
:attr:`unstable_dispatch_slots` says a slot does not identify one
callable across rounds.

### `BenchmarkStats.dispatch_tasks`

```text
BenchmarkStats.dispatch_tasks(self) -> dict[tuple[int, int], str]
```

``{(pid, slot): name}`` — what each dispatch runs, for labelling slots.

The dispatch's :attr:`TraceInvocation.task_name` (its orchestration
name, or the raw ``hid`` hash when that cannot be resolved),
taken from its first measured round. ``{}`` for L2 / fallback. For the
wire identity itself use ``dispatch_groups()[key][0].task``.

### `BenchmarkStats.per_dispatch`

```text
BenchmarkStats.per_dispatch(self, metric: str = 'device') -> dict[tuple[int, int], list[float]]
```

Per-dispatch, per-round summary (µs): ``{(pid, slot): [round0, ...]}``.

*metric* is one of ``"device"`` / ``"host"`` / ``"effective"``. Unlike
:meth:`per_rank`, nothing is summed — each entry is one dispatch's own
:attr:`TraceInvocation.device_wall_us` / ``host_wall_us`` /
``effective_us``, so a rank's repeated or heterogeneous dispatches stay
separate. Pair with :meth:`dispatch_tasks` to label the slots.

Derived from :attr:`rounds_dispatches`, so it is **L3 only** — returns
``{}`` for L2 and for the flatten fallback.

### `BenchmarkStats.per_rank`

```text
BenchmarkStats.per_rank(self, metric: str = 'device') -> dict[int, list[float]]
```

Per-rank, per-round summary (µs): ``{pid: [round0, round1, ...]}``.

*metric* is one of ``"device"`` / ``"host"`` / ``"effective"`` — each
round entry sums that rank's dispatches'
:attr:`TraceInvocation.device_wall_us` / ``host_wall_us`` / ``effective_us``
(a card runs its dispatches serially), i.e. that rank's busy time for the
round. Use :meth:`per_dispatch` when you need the individual dispatches
rather than their sum. Derived from :attr:`rounds_dispatches`, so it is
**L3 only** — returns ``{}`` for L2 and for the flatten fallback.

### `BenchmarkStats.per_round`

```text
BenchmarkStats.per_round(self, metric: str = 'device') -> list[float]
```

Per-round summary (µs), one entry per measured round.

*metric* is one of:

- ``"device"`` / ``"host"`` — the stored headline (L3: max across ranks).
- ``"effective"`` — L3: per-round max across ranks of each rank's summed
  Effective (orch/sched) window; L2 / fallback: each dispatch's
  :attr:`TraceInvocation.effective_us` in order.
- ``"union"`` — L3 only: per-round cross-rank **host-timeline** union
  window ``max(host-span end) - min(host-span start)`` across all
  ranks' dispatches (host clocks are ``CLOCK_MONOTONIC``, cross-process
  comparable, so this captures overlap / start skew — but includes host
  dispatch overhead). ``[]`` for L2 and the flatten fallback.

### `BenchmarkStats.format_tree`

```text
BenchmarkStats.format_tree(self, launch: int | None = None, *, us: bool = True) -> str
```

Render the captured ``[STRACE]`` span tree(s) as indented text.

Args:
    launch: Measured-launch index to render; ``None`` (default) renders
        every measured launch.
    us: Show durations in microseconds (default) or nanoseconds.

### `BenchmarkStats.print_tree`

```text
BenchmarkStats.print_tree(self, launch: int | None = None, *, us: bool = True, file: Any = None) -> None
```

Print :meth:`format_tree` to *file* (default stdout).

### `BenchmarkStats.mean_invocation`

```text
BenchmarkStats.mean_invocation(self, *, pid: int | None = None, slot: int | None = None) -> 'TraceInvocation | None'
```

A synthetic :class:`TraceInvocation` whose every span's ``dur`` (and
``ts``) is the mean across the measured launches of **one** dispatch
(warmup excluded).

Spans are matched by name; ``depth`` / ``attrs`` (hence
:attr:`TraceSpan.is_device`) come from the first launch that carried the
span. ``inv`` is ``-1`` to mark the aggregate. Returns ``None`` when no
span tree was captured. Useful for rendering one noise-smoothed tree.

Args:
    pid: Restrict to this rank (L3). ``None`` means "any".
    slot: Restrict to this dispatch slot within the round (L3).

Raises:
    ValueError: The selectors do not narrow an L3 run down to a single
        ``(pid, slot)`` dispatch. Averaging distinct dispatches by span
        name would fuse unrelated kernels into one meaningless tree, so
        pass ``pid=`` / ``slot=`` (see :meth:`dispatch_tasks` for the
        available keys) or use :meth:`format_mean_tree`, which renders
        every dispatch's tree.

### `BenchmarkStats.format_mean_tree`

```text
BenchmarkStats.format_mean_tree(self, *, us: bool = True, spread: str = 'stdev', pid: int | None = None, slot: int | None = None) -> str
```

Render a span tree whose every node's duration is the mean across the
measured launches (warmup excluded), annotated with the per-node spread.

On an L3 run one tree is rendered **per dispatch** (``(pid, slot)``, see
:meth:`dispatch_groups`): a rank that dispatches several kernels per round
gets one tree each rather than a single tree averaging them together.

Args:
    us: Show values in microseconds (default) or nanoseconds.
    spread: Spread shown after each node's mean — ``"stdev"`` (``±sd``,
        default), ``"minmax"`` (``[min..max]``), ``"both"``, or
        ``"none"``. Computed across the measured launches.
    pid: Render only this rank's dispatches (L3).
    slot: Render only this dispatch slot within the round (L3).

### `BenchmarkStats.print_mean_tree`

```text
BenchmarkStats.print_mean_tree(self, *, us: bool = True, spread: str = 'stdev', pid: int | None = None, slot: int | None = None, file: Any = None) -> None
```

Print :meth:`format_mean_tree` to *file* (default stdout).

## `TraceInvocation`

```text
pypto.runtime.TraceInvocation(pid: int, inv: int, hid: str, spans: list[pypto.runtime.bench.TraceSpan] = <factory>) -> None
```

Every ``[STRACE]`` span emitted by one measured launch.

One ``(pid, inv)`` group, in emission (scope-exit) order. Use
:meth:`format_tree` to render the nested call tree.

### `TraceInvocation.root`

```text
TraceInvocation.root(self) -> 'TraceSpan | None'
```

The depth-0 span (``chip.run``), or ``None`` if absent.

### `TraceInvocation.by_name`

```text
TraceInvocation.by_name(self) -> dict[str, 'TraceSpan']
```

Map span name → its first-seen span.

### `TraceInvocation.format_tree`

```text
TraceInvocation.format_tree(self, *, us: bool = True, value_fn: 'Callable[[TraceSpan], list[str]] | None' = None) -> str
```

Render this launch's span tree with ``|-`` / `` `- `` branch connectors.

Hierarchy is drawn with ASCII connectors (``|- `` for a non-last child,
`` `- `` for the last, ``|  `` / ``   `` for continuation) rather than by
indentation alone. Nesting is reconstructed from the dotted span names (a
span's parent is the span whose name is its longest proper dotted
prefix), which is robust to the host/device clock-domain split —
device-domain spans (``chip.run.runner_run.device_wall.*``) correctly
nest under their host parent even though they are emitted as a separate
batch. Siblings are ordered by start timestamp; device-domain spans are
tagged ``[dev]``.

Output is column-aligned: the name column (connectors + leaf + tag) is
left-padded to a common width and the value columns are right-aligned, so
the numbers line up regardless of nesting depth. ``value_fn`` returns the
value column(s) per span (default: a single duration column, microseconds
when ``us`` else nanoseconds); :meth:`BenchmarkStats.format_mean_tree`
uses it to add aligned ``±stdev`` / ``[min..max]`` columns.

## `TraceSpan`

```text
pypto.runtime.TraceSpan(pid: int, tid: int, inv: int, hid: str, depth: int, name: str, ts: int, dur: int, attrs: str) -> None
```

One ``[STRACE]`` span — a node in a measured launch's call tree.

A pypto-owned mirror of simpler's ``strace_timing.Span`` so ``benchmark``
callers never depend on simpler types.

Attributes:
    depth: Nesting level; a span at depth ``d`` is a child of the nearest
        enclosing span at depth ``d-1``.
    name: Dotted span path (e.g. ``chip.run.runner_run.device_wall``).
    ts: Start timestamp in nanoseconds (host clock, or device clock when
        :attr:`is_device`).
    dur: Span duration in nanoseconds.
    attrs: Raw trailing attribute string (carries ``clk=dev`` for device
        spans).

## `ChipWorker`

```text
pypto.runtime.ChipWorker(config: 'RunConfig | None' = None, *, level: 'int' = 2, runtime: 'str' = 'tensormap_and_ringbuffer', enable_sdma: 'bool' = False, auto_init: 'bool | None' = None) -> 'None'
```

L2 single-chip execution handle, bound to one ``(platform, device_id, runtime)``.

A ``ChipWorker`` auto-initializes device state in ``__init__`` so that an
immediate ``with chipworker:`` block can dispatch runs without further
setup. Construction without entering a ``with`` block also works — call
:meth:`close` manually when done, or re-enter via ``with`` later.

Inside a ``with`` block, ``CompiledProgram.__call__`` finds this worker via
a ``ContextVar`` and reuses its initialized device context instead of
creating a fresh worker per call.
Reuse only happens when all four binding fields match and the worker has
every capability required by the artifact — otherwise the caller either
falls through to the one-shot path (binding mismatch) or raises (capability
mismatch on the same device).

.. note::
   Distinct from ``simpler.worker.ChipWorker`` (the C++ L2 backend handle
   that this class wraps internally via ``self._impl``). pypto users
   interact only with this class; the simpler C++ name is not re-exported
   through ``pypto.runtime.task_interface``.

Args:
    config: Run configuration providing ``platform`` and ``device_id``.
        Defaults to :class:`RunConfig` defaults.
    level: Hierarchy level. Only ``2`` (single-chip) is currently
        supported on ``ChipWorker``; pass ``level=2`` explicitly or rely
        on the default. L3+ goes through
        :class:`~pypto.runtime.distributed_runner.DistributedWorker`.
    runtime: Runtime implementation name. Must match the runtime the
        program is compiled against; otherwise reuse silently falls
        through to the one-shot path. Defaults to
        ``"tensormap_and_ringbuffer"``.
    enable_sdma: Whether the underlying runtime worker provisions the SDMA
        workspace required by prefetch artifacts. Defaults to ``False``.
    auto_init: If ``True``, call :meth:`init` from ``__init__``. Default
        is ``True``.

### `ChipWorker.init`

```text
ChipWorker.init(self) -> 'None'
```

Initialize device state. Idempotent — a second call is a no-op.

### `ChipWorker.close`

```text
ChipWorker.close(self) -> 'None'
```

Release device state. Idempotent; a later :meth:`init` uses a new backend.

### `ChipWorker.malloc`

```text
ChipWorker.malloc(self, nbytes: 'int', *, worker_id: 'int' = 0) -> 'int'
```

Allocate ``nbytes`` of device memory; returns an opaque pointer.

The returned pointer lives in *worker_id*'s address space.  Pair every
``malloc()`` with a matching :meth:`free` before this ChipWorker is
closed, otherwise the device memory is leaked.

### `ChipWorker.free`

```text
ChipWorker.free(self, ptr: 'int', *, worker_id: 'int' = 0) -> 'None'
```

Release a pointer previously returned by :meth:`malloc`.

### `ChipWorker.copy_to`

```text
ChipWorker.copy_to(self, dst_dev_ptr: 'int', src_host_ptr: 'int', nbytes: 'int', *, worker_id: 'int' = 0) -> 'None'
```

H2D copy: ``nbytes`` bytes from host *src_host_ptr* to device *dst_dev_ptr*.

*src_host_ptr* is typically obtained from ``host_tensor.data_ptr()``;
the caller is responsible for keeping the host tensor alive until
this call returns.

### `ChipWorker.copy_from`

```text
ChipWorker.copy_from(self, dst_host_ptr: 'int', src_dev_ptr: 'int', nbytes: 'int', *, worker_id: 'int' = 0) -> 'None'
```

D2H copy: ``nbytes`` bytes from device *src_dev_ptr* back to host *dst_host_ptr*.

### `ChipWorker.run`

```text
ChipWorker.run(self, compiled: 'CompiledProgram', *args: 'CallArg', config: 'RunConfig | None' = None) -> 'Any'
```

Dispatch *compiled* on this ChipWorker explicitly.

Equivalent to ``compiled(*args, config=config)`` running under
``with chipworker:``, but the binding (platform / runtime_name) is
checked against this ChipWorker up-front rather than relying on
``ContextVar`` discovery. Falls through to the same internal
``_run_chip`` path, so the cid cache is shared with the implicit path.

Returns the same shape as ``compiled(...)``: ``None`` for in-place
calls, a single ``torch.Tensor`` for one-output return-style calls,
or a tuple of tensors otherwise.

Raises:
    ValueError: ``compiled.platform`` != ``self.platform`` or
        ``compiled.runtime_name`` != ``self.runtime``.
    RuntimeError: ChipWorker not initialized, or the compiled program
        requires SDMA but this worker was created without it.

### `ChipWorker.register`

```text
ChipWorker.register(self, compiled: 'CompiledProgram') -> 'RegistrationHandle'
```

Pre-register *compiled* on this ChipWorker. Returns a callable handle.

Eager registration: triggers ``_compile_and_assemble`` on *compiled*
and ``simpler.Worker.register`` immediately, so configuration errors
surface here rather than at first dispatch. The handle reuses the
ChipWorker's existing cid cache (multiple ``register`` calls for the
same *compiled* return aliases of the same cid).

Raises:
    ValueError: Binding mismatch (see :meth:`run`).
    RuntimeError: ChipWorker not initialized, or the compiled program
        requires SDMA but this worker was created without it.

## `DeviceTensor`

```text
pypto.runtime.DeviceTensor(data_ptr: 'int', shape: 'Sequence[int]', dtype: 'torch.dtype', *, buffer: 'Any | None' = None) -> 'None'
```

Handle to a buffer already resident on a Worker.

Attributes:
    data_ptr: Device pointer in the owning Worker's address space.
    shape: Logical tensor shape (all dimensions positive).
    dtype: Element ``torch.dtype``.
    buffer: Optional simpler ``Buffer`` retained for address-free wire
        dispatch. It is populated by :meth:`Worker.alloc_tensor`; a
        manually constructed raw-pointer tensor can be used only with
        low-level direct-chip compatibility helpers. All public
        ``Worker.run`` paths require :meth:`Worker.alloc_tensor` and its
        retained owner ``Buffer``.

## `StackedDeviceTensor`

```text
pypto.runtime.StackedDeviceTensor(shards: 'Sequence[DeviceTensor]', full_shape: 'Sequence[int]', worker_ids: 'Sequence[int]') -> 'None'
```

A leading-dim-stacked ``[B, *tail]`` tensor whose ``B`` shards are each
resident on a (possibly different) worker.

Pass it to a distributed program in place of a host ``torch.Tensor`` for a
parameter that the orchestrator slices along its leading dimension and
dispatches per rank (``for r in range(world_size): child(x[r], device=...)``).
Indexing ``obj[i, ...]`` returns shard ``i``'s :class:`DeviceTensor`, so the
generated ``host_orch`` derives a wire Tensor from its retained Buffer and
skips the per-dispatch H2D upload — the shards are uploaded once (e.g. via
:meth:`~pypto.runtime.distributed_runner.DistributedWorker.alloc_stacked_tensor`)
and reused across dispatches.

Correctness contract: ``worker_ids[i]`` is the worker that holds shard ``i``
and MUST equal the worker the compiled program submits ``x[i]``'s task to
(its ``device=`` expression). The canonical ``device=r`` program uses the
identity ``worker_ids == range(B)``; a permuted or subset placement (e.g.
``device=perm[r]`` / ``device=2*r``) requires the matching ``worker_ids``.

Attributes:
    shards: One :class:`DeviceTensor` per leading-dim index; ``shards[i]`` is
        resident on worker ``worker_ids[i]`` with shape ``full_shape[1:]``.
    full_shape: The logical stacked shape ``(B, *tail)``.
    shape: Alias for ``full_shape``, matching the :class:`DeviceTensor`
        interface.
    worker_ids: ``worker_ids[i]`` is the worker holding ``shards[i]``.

## `DistributedWorker`

```text
pypto.runtime.DistributedWorker(compiled: 'DistributedCompiledProgram | Sequence[DistributedCompiledProgram]', config: 'RunConfig | None' = None, *, persistent: 'bool' = False, reset_persistent_windows: 'bool | None' = None, callbacks: 'dict[str, Callable[..., Any]] | None' = None, sub_worker_overrides: 'dict[str, Callable[..., Any]] | None' = None, inherited_host_tensors: 'Sequence[torch.Tensor | ReadOnlyHostTensor] | None' = None, startup_timeout_s: 'float | None' = None) -> 'None'
```

L3 distributed execution handle: prepare once, dispatch many.

Holds an initialized simpler ``Worker(level=3)`` plus all setup artifacts
(chip callables, host_orch entry, sub-worker fns, comm bootstrap) so the
expensive setup — ``_compile_and_assemble``, generated-module loading, Worker
construction + registration + ``init()`` (fork) — happens exactly once.

Mirrors the L2 ``with ChipWorker(...)`` reuse block: it exposes device-memory
helpers (:meth:`malloc`, :meth:`copy_to`, :meth:`copy_from`, :meth:`free`,
:meth:`alloc_tensor`) so callers can build worker-resident
:class:`~pypto.runtime.DeviceTensor` buffers that survive across dispatches,
then call ``rt(*device_args)`` or ``rt.run(compiled, *device_args)``
repeatedly.

Per-call IO buffers (inputs **and** outputs) are shared-memory host
``torch.Tensor`` objects allocated **before** :meth:`prepare` and reused in
place across dispatches — the forked chip worker reads/writes them through
the inherited shared mapping, and outputs are read straight back from the
tensor (no ``copy_from``). Explicit :meth:`alloc_tensor` /
:meth:`alloc_stacked_tensor` uploads and copy-backs stage through
runtime-owned POSIX-shm Buffers, so their CPU-contiguous host endpoints may
be ordinary tensors created after ``prepare``.

``inherited_host_tensors`` keeps such a host range alive for the worker's
lifetime *and* is a caller guarantee about it: listing a tensor asserts that
its backing is visible across processes — a ``MAP_SHARED`` mapping, whether
torch's own shared memory or an external file mapping — and that the mapping
stays valid for as long as the worker lives. Listed ranges are then named in
place instead of staged, which is what removes the per-copy ``memmove``.
Passing a ``MAP_PRIVATE`` backing is **unsupported**: copy-on-write leaves the
child reading its pre-fork snapshot, so uploads may carry stale or incorrect
data. PyPTO cannot verify the guarantee — ``torch.is_shared()`` returns
``False`` for valid external ``MAP_SHARED`` mappings — so it warns once at
prepare time for the tensors it cannot confirm and proceeds. Listing a tensor
does not make it a valid direct dispatch argument.

The guarantee is made per *allocation*: a listed tensor names its whole storage,
so every view of one storage shares one Buffer and reaches its own bytes by
offset. Wrap an entry in :class:`ReadOnlyHostTensor` when the backing is shared
but not writable — a ``MAP_SHARED`` mapping of a read-only fd. That allocation's
Buffer is then declared ``READ``, and :meth:`copy_from` into it is refused
instead of faulting inside the forked child. Listing the same allocation both
read-only and writable is a ``ValueError``: one allocation has one access mode.

``callbacks`` binds a caller-supplied callable to a SubWorker by name — e.g.
a real sampling closure. Abstract SubWorkers (declared with a ``...`` body)
are runtime-bound callback points and MUST be supplied here; a missing
binding raises ``ValueError`` at prepare time. A callback may also replace a
concrete SubWorker's generated body. Each name must be a sub-worker the
program declares; an unknown name raises ``ValueError``.
(``sub_worker_overrides`` is a deprecated alias for ``callbacks``.)

**Multi-program dispatch.** Pass a sequence of compatible
:class:`DistributedCompiledProgram` objects (or use
``compiled.prepare(extra_compiled=[...])``) to prepare several HOST programs
on one L3 worker. Each program's chip callables, sub-worker functions,
orchestration entry, base tensors, and parameter metadata are registered
independently and selected at dispatch via ``rt.run(compiled, *args)``. This
is what serving needs: prefill and decode are separate JIT HOST programs that
must share one worker lifecycle and one worker-resident
:class:`DeviceTensor` KV cache. Programs must agree on platform, runtime, and
device ids; a mismatch raises ``ValueError``. The ``rt(*args)`` shortcut is
only for single-program workers — in multi-program mode it raises
``TypeError`` since the target program is ambiguous.

Obtain via :meth:`DistributedCompiledProgram.prepare`. Use as a context
manager (recommended) or call :meth:`close` when done::

    host_x = torch.zeros(seq, 4096, dtype=torch.float16).share_memory_()
    host_out = torch.zeros(seq, 4096, dtype=torch.float16).share_memory_()
    host_w = load_weight().share_memory_()      # before prepare()
    with compiled.prepare() as rt:
        weight = rt.alloc_tensor(host_w.shape, host_w.dtype, init=host_w)
        for step in steps:
            host_x.copy_(next_input(step))      # update in place
            rt(host_x, weight, host_out)        # host shm IO + resident weight
            consume(host_out)                   # read directly
        rt.free_tensor(weight)

### `DistributedWorker.malloc`

```text
DistributedWorker.malloc(self, nbytes: 'int', *, worker_id: 'int' = 0) -> 'int'
```

Allocate ``nbytes`` on chip *worker_id*; returns a device pointer.

### `DistributedWorker.free`

```text
DistributedWorker.free(self, ptr: 'int', *, worker_id: 'int' = 0) -> 'None'
```

Release a pointer previously returned by :meth:`malloc`.

### `DistributedWorker.committed_device_memory`

```text
DistributedWorker.committed_device_memory(self, worker_id: 'int' = 0) -> 'int'
```

Total device HBM (bytes) committed by chip *worker_id*'s ``MemoryAllocator``
(tensors + pooled arenas + runtime buffers). Routes through the underlying
simpler ``Worker(level=3)`` facade, which forwards a
``CTRL_COMMITTED_DEVICE_MEMORY`` query to the forked chip child. Sum across
``worker_id``s for a multi-chip total.

### `DistributedWorker.device_memory_info`

```text
DistributedWorker.device_memory_info(self, worker_id: 'int' = 0) -> 'tuple[int, int]'
```

Free and total device HBM (bytes) as the driver sees it, for the device
chip *worker_id* runs on. Routes through the underlying simpler
``Worker(level=3)`` facade, which forwards the query to the forked chip child.

This is a device-wide snapshot and answers a different question from
:meth:`committed_device_memory`, which reports only what this worker's own
``MemoryAllocator`` has committed. Anything else on the card -- another
process, another worker, the driver itself -- moves ``free_bytes`` without
moving that committed total.

Unlike :meth:`committed_device_memory`, a failed query is never softened
into a zero here: a caller sizing a KV cache from a fabricated ``(0, 0)``
would silently under-allocate, so the underlying error propagates instead.
Simulator backends synthesize no device-wide memory and raise
``NotImplementedError``.

Returns:
    ``(free_bytes, total_bytes)``, both as Python ints.

### `DistributedWorker.copy_to`

```text
DistributedWorker.copy_to(self, dst_dev_ptr: 'int', src_host_ptr: 'int', nbytes: 'int', *, dst_offset: 'int' = 0, src_offset: 'int' = 0, worker_id: 'int' = 0) -> 'None'
```

H2D copy: ``nbytes`` from ``src_host_ptr + src_offset`` to ``dst_dev_ptr + dst_offset``.

### `DistributedWorker.copy_from`

```text
DistributedWorker.copy_from(self, dst_host_ptr: 'int', src_dev_ptr: 'int', nbytes: 'int', *, dst_offset: 'int' = 0, src_offset: 'int' = 0, worker_id: 'int' = 0) -> 'None'
```

D2H copy: ``nbytes`` from ``src_dev_ptr + src_offset`` to ``dst_host_ptr + dst_offset``.

### `DistributedWorker.alloc_stacked_tensor`

```text
DistributedWorker.alloc_stacked_tensor(self, host: 'torch.Tensor', *, worker_ids: 'Sequence[int] | None' = None) -> 'StackedDeviceTensor'
```

Upload each leading-dim shard of *host* to a worker once; reuse it.

The leading dimension of *host* is the stack/shard dimension: shard ``i``
(``host[i]``, shape ``host.shape[1:]``) is uploaded to worker
``worker_ids[i]`` and stays resident for the worker's lifetime. Pass the
returned :class:`~pypto.runtime.StackedDeviceTensor` in place of *host*
for a leading-dim-sharded program parameter (a ``[B, *tail]`` tensor the
orchestrator slices per rank: ``for r in range(world_size):
child(x[r], device=...)``). The generated ``host_orch`` indexes ``x[i]``
to shard ``i``'s :class:`~pypto.runtime.DeviceTensor`, so the runtime
skips the per-dispatch H2D upload through its retained Buffer — the stack is
uploaded once here and reused across every ``rt(...)`` dispatch.

Args:
    host: A CPU, contiguous ``[B, *tail]`` tensor. Explicit upload
        stages through a runtime-owned shared Buffer, so it may be an
        ordinary tensor created after :meth:`prepare`.
    worker_ids: ``worker_ids[i]`` is the worker that holds shard ``i``
        and whose task consumes ``x[i]``; it MUST equal the worker the
        program submits ``x[i]``'s dispatch to (its ``device=``
        expression). Entries must be distinct and within
        ``[0, world_size)``. Defaults to ``range(B)`` — the canonical
        ``for r in range(world_size): child(x[r], device=r)`` program. A
        permuted/subset placement (``device=perm[r]`` / ``device=2*r``)
        needs the matching ``worker_ids``.

Returns:
    A :class:`~pypto.runtime.StackedDeviceTensor`; its shards are tracked
    by this worker and auto-freed on :meth:`close` if not released earlier
    via :meth:`free_stacked_tensor`.

### `DistributedWorker.free_stacked_tensor`

```text
DistributedWorker.free_stacked_tensor(self, stacked: 'StackedDeviceTensor') -> 'None'
```

Release every shard of *stacked* against its owning worker. Idempotent.

### `DistributedWorker.release_inherited_host_tensor_refs`

```text
DistributedWorker.release_inherited_host_tensor_refs(self) -> 'None'
```

Release compatibility lifetime references supplied at construction.

Explicit uploads now stage through POSIX shm and remain valid while the
caller itself retains the source tensor. Resident :class:`DeviceTensor`
and :class:`StackedDeviceTensor` allocations are unaffected.

### `DistributedWorker.copy_stacked_from`

```text
DistributedWorker.copy_stacked_from(self, stacked: 'StackedDeviceTensor', host: 'torch.Tensor') -> 'None'
```

Read every shard of *stacked* back to *host* (D2H) — the read-back
symmetric to :meth:`alloc_stacked_tensor`.

Because a :class:`~pypto.runtime.StackedDeviceTensor` skips the
per-dispatch D2H copy, callers that want the shards' current device
contents (e.g. a resident KV cache at the end of an L3 step) must read
them back explicitly. Shard ``i`` is copied from its owning worker
``stacked.worker_ids[i]`` into ``host[i]``.

Args:
    stacked: The resident stacked tensor to read back.
    host: A CPU, contiguous ``[B, *tail]`` tensor whose shape and dtype
        match ``stacked.full_shape`` / ``stacked.dtype``. It may be an
        ordinary tensor allocated after :meth:`prepare`; D2H stages
        through a runtime-owned shared Buffer before copying into it.

### `DistributedWorker.close`

```text
DistributedWorker.close(self) -> 'None'
```

Release runtime resources, retrying incomplete Worker cleanup.

### `DistributedWorker.submit`

```text
DistributedWorker.submit(self, compiled: 'DistributedCompiledProgram', *args: 'Any', config: 'RunConfig | None' = None) -> 'DistributedRunHandle'
```

Submit *compiled* and return before device completion when supported.

The returned handle owns one of two bounded metadata frames plus all
argument and configuration lifetimes. A third submission waits for the
oldest handle before reusing a frame. Diagnostic two-pass swimlane
capture remains a synchronous fallback and returns a completed handle.

Args:
    compiled: A program registered when this worker was constructed.
    *args: In-place program arguments. Mutable arguments must not be
        reused or modified until the returned handle completes.
    config: Optional per-dispatch runtime configuration.

Returns:
    A handle that owns the dispatch lifetime and cached outcome.

### `DistributedWorker.run`

```text
DistributedWorker.run(self, compiled: 'DistributedCompiledProgram', *args: 'Any', config: 'RunConfig | None' = None) -> 'None'
```

Dispatch *compiled* on this DistributedWorker.

Provided for symmetry with :meth:`ChipWorker.run`, so library code can
write ``rt.run(compiled, *args)`` and accept either runtime kind. For a
multi-program worker this selects which prepared program to dispatch.

*compiled* must be one of the :class:`DistributedCompiledProgram` objects
this worker was constructed from; passing an unregistered one raises
``ValueError``.

``config`` is an optional per-dispatch :class:`RunConfig`; its per-task
ring sizing and runtime DFX fields apply without touching the prepared
program's shared config. In a multi-program worker each program can
therefore use its own ring sizes and diagnostics. On onboard L3,
``enable_chip_swimlane`` executes a dep-gen graph pass followed by a
dep-gen-disabled timing pass; mutable arguments are not restored between
them. ``None`` snapshots the program's baseline for this dispatch.

### `DistributedWorker.register`

```text
DistributedWorker.register(self, compiled: 'DistributedCompiledProgram') -> 'RegistrationHandle'
```

Pre-register *compiled* on this DistributedWorker.

Returns a :class:`~pypto.runtime.RegistrationHandle` whose
``__call__`` delegates to :meth:`run`. The cid alias-safety rules
described on :class:`RegistrationHandle` apply.

Unlike L2, the underlying simpler registration already happened
during :meth:`__init__` (it must, for COW propagation to forked
chip children). This method just packages the existing setup as a
callable handle, exposing ``cid=0`` as a placeholder.

Raises:
    RuntimeError: This DistributedWorker has been closed.
    ValueError: *compiled* is not one of the programs this worker was
        constructed from.

## `DistributedRunHandle`

```text
pypto.runtime.DistributedRunHandle(worker: 'DistributedWorker', native_handle: 'Any | None', frame: '_DispatchFrame', dispatch_id: 'int', postprocess: 'Callable[[], None] | None' = None) -> 'None'
```

Completion handle returned by :meth:`DistributedWorker.submit`.

The handle keeps its worker, immutable dispatch configuration, argument
references, generated task arguments, and one bounded metadata frame alive
until terminal completion. Waiting is idempotent and every waiter observes
the same cached outcome.

### `DistributedRunHandle.result`

```text
DistributedRunHandle.result(self, timeout: 'float | None' = None) -> 'None'
```

Wait for completion and raise the cached dispatch error, if any.

Args:
    timeout: Maximum wait in seconds. ``None`` waits without a deadline.

Raises:
    TimeoutError: The dispatch did not complete before ``timeout``.
    ValueError: ``timeout`` is negative or non-finite.
    BaseException: The cached native or result-publication failure.

### `DistributedRunHandle.wait`

```text
DistributedRunHandle.wait(self, timeout: 'float | None' = None) -> 'None'
```

Wait for completion as an alias for :meth:`result`.

Args:
    timeout: Maximum wait in seconds. ``None`` waits without a deadline.

## `ReadOnlyHostTensor`

```text
pypto.runtime.ReadOnlyHostTensor(tensor: 'torch.Tensor') -> None
```

An ``inherited_host_tensors`` entry a worker may read but never write.

Wrap a tensor whose backing is ``MAP_SHARED`` from a read-only fd: it is genuinely
visible across processes — so it is worth naming in place rather than staging — and
still faults on a write. The allocation's Buffer is then declared ``AccessMode.READ``,
which makes the ABI refuse a D2H into it instead of leaving the forked child to fault
on the store.

Writability cannot be inferred, for the same reason visibility cannot: ``torch`` erases
it (``torch.from_numpy`` on a read-only array yields a writable tensor), a write probe
faults rather than raising, and ``/proc/self/maps`` would tie this to Linux. Marking is
therefore a caller guarantee, exactly like listing the tensor in the first place.

The marker governs the **named-copy** descriptor only. Dispatch arguments are named by
the runtime's own path (``Worker.make_tensor_arg``), which this does not reach, so
marking a dispatch IO buffer read-only does not stop the child from writing it.

## `RegistrationHandle`

```text
pypto.runtime.RegistrationHandle(worker: 'Worker', compiled: 'Any', cid: 'Any') -> 'None'
```

Bound dispatcher for one ``(Worker, compiled)`` pair.

Returned by :meth:`Worker.register` (implemented on both
:class:`ChipWorker` and
:class:`~pypto.runtime.distributed_runner.DistributedWorker`).

Three usage styles, all equivalent in steady state:

Direct call (most common)::

    h = worker.register(compiled)
    h(a, b)
    h.unregister()

Context manager (auto-release on scope exit)::

    with worker.register(compiled) as h:
        h(a, b)

Manual control over cid lifetime (rare; cf. benchmarks)::

    h = worker.register(compiled)
    for _ in range(1000):
        h(a, b)
    # h.unregister() optional — Worker.close() releases everything

Calling a handle after :meth:`unregister` or after the parent
``Worker.close()`` raises ``RuntimeError``.

**cid reuse semantics (L2):** Multiple :meth:`ChipWorker.register` calls
for the same ``compiled.chip_callable`` return aliases of the same
underlying cid. :meth:`unregister` only marks the handle closed; it does
NOT call ``simpler.Worker.unregister``. Real cid release happens once,
in :meth:`Worker.close`. ``cid`` is informational only.

**L3 note:** ``DistributedWorker`` doesn't expose a per-callable cid the
way ChipWorker does (its chip / sub registrations are baked at prepare()
time). For L3 handles, ``cid`` is ``0``; the dispatch path still routes
through the orchestrator via ``DistributedWorker.run``.

### `RegistrationHandle.unregister`

```text
RegistrationHandle.unregister(self) -> 'None'
```

Mark this handle closed. Idempotent.

Does NOT call ``simpler.Worker.unregister`` — other handle aliases
for the same cid would silently break. The real reverse-registration
happens once, in :meth:`Worker.close`.

## `CompileOptions`

```text
pypto.runtime.CompileOptions(platform: str = 'a2a3sim', strategy: pypto.ir.pass_manager.OptimizationStrategy = <factory>, dump_passes: bool | pypto.ir.pass_manager.PassDumpLevel = False, dump_ptoas_passes: bool = False, profiling: bool = False, diagnostic_phase: pypto.pypto_core.passes.DiagnosticPhase | None = None, disabled_diagnostics: pypto.pypto_core.passes.DiagnosticCheckSet | None = None, analyze_auto_scopes_for_deps: bool = False, output_dir: str | None = None, memory_planner: pypto.pypto_core.passes.MemoryPlanner | None = None, distributed_config: 'DistributedConfig | None' = None) -> None
```

What compilation reads, in ``ir.compile``'s own vocabulary.

The typed form of :meth:`RunConfig.compile_kwargs`. A caller that only
compiles needs this and not a :class:`RunConfig`::

    from pypto import ir
    from pypto.runtime import CompileOptions

    compiled = ir.compile(program, **CompileOptions(platform="a2a3").as_compile_kwargs())

Field names are ``ir.compile``'s, not ``RunConfig``'s: what ``RunConfig``
spells ``save_kernels_dir`` and ``compile_profiling`` are ``output_dir`` and
``profiling`` here, because this object exists to name the compile side as
the compiler names it.

``platform`` is the only way to name the target. ``ir.compile`` also takes a
``backend_type``, but derives it from ``platform`` whenever one is given, so
carrying both here would offer a pairing that cannot take effect: set them
to disagree and the platform silently wins. ``ir.compile`` keeps its
parameter for callers that pass no platform at all; this object always
passes one.

### `CompileOptions.as_compile_kwargs`

```text
CompileOptions.as_compile_kwargs(self) -> dict[str, typing.Any]
```

Return these options as :func:`pypto.ir.compile` keyword arguments.

## `DfxOptions`

```text
pypto.runtime.DfxOptions(enable_chip_swimlane: int | bool = 0, enable_dump_args: int = 0, enable_pmu: int = 0, enable_dep_gen: bool = False, enable_scope_stats: bool = False) -> None
```

Bundle of runtime DFX toggles passed through the execute pipeline.

Each field maps to a ``CallConfig`` member on the runtime side. The public
``enable_chip_swimlane`` field carries a collection level (see
:func:`_normalize_swimlane_level`). ``any()`` answers whether the runtime
needs an ``output_prefix``.

### `DfxOptions.any`

```text
DfxOptions.any(self) -> bool
```

## `ExecutionMode`

```text
pypto.runtime.ExecutionMode(value, names=None, *, module=None, qualname=None, type=None, start=1)
```

Whether a run reaches real silicon or the simulator.

One of the two axes a ``platform`` string packs. It decides how kernels are
assembled (``.so`` for the simulator, ``.o`` plus a text-section extract for
silicon) and whether the two-pass swimlane capture applies; it never reaches
codegen, which sees only the architecture.

## `RunConfig`

```text
pypto.runtime.RunConfig(*, arch: pypto.pypto_core.backend.BackendType = <factory>, execution_mode: pypto.runtime.runner.ExecutionMode = <ExecutionMode.SIM: 2>, device_id: int = 0, rtol: float = 1e-05, atol: float = 1e-05, strategy: pypto.ir.pass_manager.OptimizationStrategy = <factory>, dump_passes: bool | pypto.ir.pass_manager.PassDumpLevel = False, save_kernels: bool = False, save_kernels_dir: str | None = None, codegen_only: bool = False, enable_chip_swimlane: int | bool = 0, enable_dump_args: int = 0, enable_pmu: int = 0, enable_dep_gen: bool = False, enable_scope_stats: bool = False, compile_profiling: bool = False, diagnostic_phase: pypto.pypto_core.passes.DiagnosticPhase | None = None, disabled_diagnostics: pypto.pypto_core.passes.DiagnosticCheckSet | None = None, golden_data_dir: str | None = None, aicpu_thread_num: int | None = None, ring_task_window: int | list[int] | tuple[int, ...] | None = None, ring_heap: int | list[int] | tuple[int, ...] | None = None, ring_dep_pool: int | list[int] | tuple[int, ...] | None = None, distributed_config: 'DistributedConfig | None' = None, analyze_auto_scopes_for_deps: bool = False, memory_planner: pypto.pypto_core.passes.MemoryPlanner | None = None, dump_ptoas_passes: bool = False, platform: 'str | None' = None) -> None
```

Configuration for compiling and dispatching a program.

Carries both halves: :meth:`compile_kwargs` extracts the compile-side
fields for :func:`pypto.ir.compile`, and the rest is read at dispatch by
:meth:`~pypto.ir.CompiledProgram.__call__` / :meth:`Worker.run`.

When passed to :meth:`pypto.jit.decorator.JITFunction.lower`, only
``platform``, ``strategy``, diagnostics, dependency analysis, and
``memory_planner`` affect pass execution. Runtime and artifact fields such
as ``device_id``, ``dump_passes``, ``dump_ptoas_passes``,
``save_kernels_dir``, and ``compile_profiling`` are ignored; ``lower()``
does not execute or write compilation artifacts.

Attributes:
    arch: Target architecture, as the codegen backend that names it —
        ``BackendType.Ascend910B`` (a2a3) or ``BackendType.Ascend950`` (a5).
    execution_mode: :class:`ExecutionMode.SIM` or ``ONBOARD``.
    platform: **Not a field** — the wire spelling of the two axes above
        (``"a2a3sim"`` / ``"a2a3"`` / ``"a5sim"`` / ``"a5"``), derived on
        read. Accepted as a constructor keyword, where it sets both axes.
    device_id: Hardware device index (ignored for simulator).
    backend_type: **Not a field** — a read-only property derived from
        ``platform``. Accepted as a deprecated constructor keyword, which
        warns and is discarded when it contradicts the platform.
    rtol: Relative tolerance for result comparison.
    atol: Absolute tolerance for result comparison.
    strategy: PyPTO optimisation strategy applied during compilation.
    dump_passes: Per-pass IR dump control. A :class:`~pypto.ir.PassDumpLevel`
        (``NONE`` / ``CONCISE`` / ``EXPLICIT``) or a ``bool``
        (``True`` -> ``CONCISE``, ``False`` -> ``NONE``). ``EXPLICIT`` resolves
        implicit tile layouts and distributed window buffers in the dump.
    dump_ptoas_passes: If ``True``, dump full-module intermediate IR after
        every ptoas pass under
        ``<output_dir>/ptoas_passes/<codegen-unit>/``. Has no effect when
        ptoas is unavailable and compilation stops at raw ``.pto`` output.
    save_kernels: If ``True``, retain generated artefacts after execution.
        When ``False`` (default), a temporary directory is used and cleaned up.
    save_kernels_dir: Directory to save generated artefacts when *save_kernels*
        is ``True``.  If ``None``, a timestamped directory is created under
        ``build_output/<program_name>_<timestamp>``.
    codegen_only: If ``True``, stop after code generation without executing
        on device.  Useful for validating compilation output.
    enable_chip_swimlane: Chip swimlane collection **level** — per-task
        timing records written into
        ``<work_dir>/dfx_outputs/chip_swimlane_records.json``. Mirrors the
        runtime harness's ``--enable-chip-swimlane PERF_LEVEL``; each level
        is a real guard in the runtime collectors, so a lower level never
        stamps the data a higher one does and no post-processing recovers
        it:

        * ``0`` / ``False`` — off.
        * ``1`` — AICore per-task start / end plus the task record buffer.
        * ``2`` — ``1`` plus AICPU-stamped dispatch / finish, which is what
          makes the ``[dispatch, start]`` pickup gap readable.
        * ``3`` — ``2`` plus scheduler main-loop phase records, required by
          ``python -m simpler_setup.tools.sched_overhead_analysis`` and by
          the PyPTO Toolkit plugin's Scheduler View.
        * ``4`` / ``True`` — ``3`` plus orchestrator phase records (the
          Toolkit plugin's AICPU Orchestrator view). ``True`` requests this
          full level, matching the runtime harness's bare
          ``--enable-chip-swimlane``.

        The former spelling ``enable_l2_swimlane`` still works (constructor
        keyword and attribute) but emits a ``DeprecationWarning``.

        On onboard platforms, ``swimlane_converter`` then produces
        ``merged_swimlane_*.json`` alongside the records. Because the converter joins
        the timing against a task graph that only ``deps.json`` carries,
        enabling this on an onboard platform runs the workload **twice**: a
        first dep_gen pass captures ``deps.json``, then a clean swimlane pass
        runs with dep_gen off because collection perturbs timing. L2 runs
        the graph pass in a subprocess so its device/SVM state is fully
        reclaimed before the timing pass (a failed capture is logged, not
        fatal). L3 one-shot uses separate Worker lifecycles; a prepared L3
        worker uses two ``Worker.run()`` fences and keeps resident handles
        alive. Both L3 passes execute the program without restoring mutable
        arguments between them. Simulator platforms (``*sim``) stay single-pass
        and only emit ``chip_swimlane_records.json`` — the merged swimlane file
        is intentionally skipped because the simulator does not yet ship the
        task metadata the converter needs. Mirrors runtime's
        ``CallConfig.enable_chip_swimlane`` field.
    enable_dump_args: Per-task argument dump **level** written into
        ``<work_dir>/dfx_outputs/args_dump/``. Inspect with
        ``python -m simpler_setup.tools.dump_viewer``. Mirrors
        ``--dump-args``:

        * ``0`` / ``False`` — off (no dump).
        * ``1`` / ``True`` — **partial**: only the tensors marked via the
          DSL marker ``pl.dump_tag(t)`` (or ``pl.submit(..., dumps=[...])``).
        * ``2`` — **full**: every task's tensor inputs and outputs.

        Full dump on a large workload can saturate the host-side dump
        collector (~42 MB/s drain rate) and get the AICPU killed by a STARS
        op-execute timeout — prefer partial (level ``1``) plus
        ``pl.dump_tag(t)`` to limit dump to specific tensors
        (simpler#844 selective tensor dump).
    enable_pmu: AICore PMU event type. ``0`` disables collection;
        ``>0`` enables and selects the event (``2`` = PIPE_UTILIZATION,
        ``4`` = MEMORY — see ``runtime/docs/dfx/pmu-profiling.md``).
        Output: ``<work_dir>/dfx_outputs/pmu.csv``. Mirrors
        ``--enable-pmu N``.
    enable_dep_gen: Capture simpler dependency edges into
        ``<work_dir>/dfx_outputs/deps.json``. Render to HTML on demand via
        ``python -m simpler_setup.tools.deps_viewer <deps.json> --format
        html`` (the CLI defaults to text output). Mirrors
        ``--enable-dep-gen``.
    enable_scope_stats: Capture per-scope heap / task_window / tensormap
        ring-fill peaks into
        ``<work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl``. Render to
        HTML on demand via ``runtime/tools/scope_stats_plot.py``. Mirrors
        ``--enable-scope-stats``.
    compile_profiling: If ``True``, enable compile profiling that records
        per-stage wall-clock timings (parse, passes, codegen).
        Results are written to ``report/pipeline_profile.{txt,json}`` in
        the output directory.
    diagnostic_phase: Override the diagnostic phase gate for compilation.
        ``None`` uses the default (``PrePipeline``, or ``PYPTO_WARNING_LEVEL``
        env var). Setting to ``None`` silences warnings AND performance hints;
        finer-grained control uses ``disabled_diagnostics``.
    disabled_diagnostics: Set of diagnostic checks to disable during
        compilation (covers warnings and perf hints). ``None`` uses the
        default (``UnusedControlFlowResult`` disabled, perf hints enabled).
    golden_data_dir: Target directory for ``.pt`` data files.  When set,
        the generated ``golden.py`` always loads tensors from this path.
        If the directory already contains all required ``.pt`` files they
        are reused; otherwise the directory is created and data is generated
        there.  Use a path from a previous run
        (e.g. ``build_output/<name>_<ts>/data``) to reuse existing golden
        data, or specify a new path to persist data to a fixed location.
    aicpu_thread_num: Optional per-invocation override of the AICPU
        thread count. ``None`` (default) defers to the value baked
        into ``kernel_config.py``'s ``RUNTIME_CONFIG`` at compile
        time (which itself may be unset, in which case the simpler
        runtime default applies).
    ring_task_window: Optional per-invocation override of the runtime
        ring's task-slot window (number of in-flight tasks). Forwarded to
        ``CallConfig.runtime_env.ring_task_window``. A scalar (broadcast to
        all scope-depth rings) or a list/tuple of exactly 4 ints sizing
        rings 0..3 independently; each entry must be a power of two ``>= 4`` (a
        ``0`` list-entry leaves that ring at its default). ``None`` (default)
        leaves the field unset so the runtime falls back to its
        compile-time default.
    ring_heap: Optional per-invocation override of the per-ring output-heap
        size in **bytes**. Forwarded to ``CallConfig.runtime_env.ring_heap``.
        A scalar or a list/tuple of 4 ints (per ring 0..3); each entry must
        be a power of two ``>= 1024`` (a ``0`` list-entry leaves that ring at its
        default). ``None`` defers to the runtime's compile-time default.
    ring_dep_pool: Optional per-invocation override of the per-ring
        dependency-edge pool capacity. Forwarded to
        ``CallConfig.runtime_env.ring_dep_pool``. A scalar or a list/tuple
        of 4 ints (per ring 0..3); each entry must be in ``[4, INT32_MAX]`` (a
        ``0`` list-entry leaves that ring at its default). ``None`` defers
        to the runtime's compile-time default.
    distributed_config: Optional L3 distributed-execution config, consumed
        only on the ``@pl.jit`` path. When set, it is forwarded to
        ``ir.compile()`` (via :meth:`RunConfig.compile_kwargs`)
        so a HOST-level ``@pl.jit.host`` kernel compiles to a
        :class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`
        and dispatches per-rank. ``None`` (default) compiles a regular
        single-chip :class:`~pypto.ir.compiled_program.CompiledProgram`.
        :meth:`compile_kwargs` forwards it too, so a ``@pl.program``
        compiled via ``ir.compile(prog, **config.compile_kwargs())`` picks
        up the same distributed target.
    analyze_auto_scopes_for_deps: If ``True``, enable compiler-derived task
        dependency analysis for AUTO runtime scopes during compilation.
        Defaults to ``False`` so existing runs keep using TensorMap fallback
        unless this behavior is explicitly requested.
    memory_planner: Who plans on-chip buffer memory —
        :attr:`~pypto.pypto_core.passes.MemoryPlanner.PYPTO` (PyPTO runs
        ``MemoryReuse`` + ``AllocateMemoryAddr`` and bakes physical
        addresses), ``DSA_RP`` (PyPTO runs its in-tree
        capacity-constrained DSA-RP planner), or ``PTOAS`` (the PyPTO
        allocation passes are skipped and ptoas ``PlanMemory`` owns reuse
        and addressing). ``None`` (default) defers to the active
        ``PassContext``, or to ``PYPTO`` when none is active.
        Forwarded to ``ir.compile()``, which rejects it when a
        ``PassContext`` is already active — set it on that context instead.

### `RunConfig.any_dfx_enabled`

```text
RunConfig.any_dfx_enabled(self) -> bool
```

Return ``True`` when at least one DFX flag is enabled.

DFX (Design For X) covers the five runtime diagnostic sub-features
carried on :class:`~simpler.task_interface.CallConfig`:
chip swimlane, argument dump, PMU, dep_gen and scope_stats. They are
independent toggles that share an output directory.

### `RunConfig.compile_kwargs`

```text
RunConfig.compile_kwargs(self) -> dict[str, typing.Any]
```

Return the compile-side fields as ``ir.compile()`` keyword arguments.

A :class:`RunConfig` carries both compile-time and dispatch-time
settings. This method extracts the compile-time half so a caller can
drive the two phases separately without restating the mapping::

    compiled = ir.compile(program, **config.compile_kwargs())
    compiled(*tensors, config=config)

It is the only mapping onto ``ir.compile``'s parameters: the ``@pl.jit``
path calls it too, so a knob added here reaches both. (``lower()`` keeps
its own narrower mapping — it stops before codegen and targets the pass
pipeline rather than ``ir.compile``.)

Dispatch-only fields (``device_id``, the DFX toggles, the ring-sizing
overrides) are not compile inputs and are consumed by
:meth:`~pypto.ir.CompiledProgram.__call__` instead. ``rtol`` / ``atol``
and ``golden_data_dir`` reach neither phase: only the system-test
harness reads them, to compare a dispatch against its golden.

``output_dir``, ``distributed_config`` and ``memory_planner`` are
forwarded only when set, so an unset value defers to ``ir.compile()``'s
own default. That matters for ``memory_planner`` in particular:
``ir.compile()`` rejects an explicit planner while a ``PassContext`` is
active, and an unset value must defer to that context.

Returns:
    Keyword arguments accepted by :func:`pypto.ir.compile`.

### `RunConfig.compile_options`

```text
RunConfig.compile_options(self) -> 'CompileOptions'
```

Return the compile-side half as a :class:`CompileOptions`.

The field renames are the compiler's own vocabulary:
``save_kernels_dir`` is ``ir.compile``'s ``output_dir`` and
``compile_profiling`` is its ``profiling``.

### `RunConfig.run_options`

```text
RunConfig.run_options(self) -> 'RunOptions'
```

Return the dispatch-side half as a :class:`RunOptions`.

### `RunConfig.dfx_options`

```text
RunConfig.dfx_options(self) -> 'DfxOptions'
```

Return the diagnostic toggles as a :class:`DfxOptions`.

Shorthand for ``run_options().dfx`` — the runtime asks for the
diagnostics alone far more often than for the whole dispatch half.

## `RunResult`

```text
pypto.runtime.RunResult(passed: bool, test_name: str | None = None, error: str | None = None, execution_time: float | None = None, profile: dict[str, typing.Any] | None = None) -> None
```

Result of a program run or harness test execution.

Attributes:
    passed: ``True`` if the program executed and results matched the golden
        reference within the configured tolerances.
    test_name: Optional test case name.  Set by the harness when running
        a named test case; ``None`` outside the harness.
    error: Human-readable error message when ``passed`` is ``False``.
    execution_time: Python wall-clock time in seconds for the full run
        (compile + execute + validate). This mixes host-side compile/golden
        overhead with the actual dispatch, so it cannot isolate device time
        — read per-run device/host timing from the runtime's ``[STRACE]``
        log markers (simpler PR #1177) instead.

## `ScalarSpec`

```text
pypto.runtime.ScalarSpec(name: str, value: int | float, ctype: str = 'int64') -> None
```

Specification for a scalar TaskArg parameter.

Scalar parameters occupy TaskArg slots after all tensor parameters.
The generated golden.py emits ``ctypes`` scalar values so that
Simpler's CodeRunner populates ``orch[i].scalar`` correctly.

Attributes:
    name: Parameter name matching the orchestration function signature.
    value: The scalar value to pass at runtime.
    ctype: ctypes type name — one of ``"int64"``, ``"int32"``,
        ``"uint64"``, ``"uint32"``, ``"float"``, ``"double"``.

## `TensorSpec`

```text
pypto.runtime.TensorSpec(name: str, shape: list[int], dtype: torch.dtype, init_value: int | float | torch.Tensor | collections.abc.Callable | None = None, is_output: bool = False) -> None
```

Specification for a runtime tensor.

Attributes:
    name: Tensor name, used as the parameter name in generated C++ and golden.py.
    shape: Tensor shape as a list of integers.
    dtype: PyTorch dtype (e.g. ``torch.float32``, ``torch.bfloat16``).
    init_value: Initial value strategy.  Can be:

        - ``None`` — zero-initialised.
        - ``int`` or ``float`` — every element set to this constant.
        - ``torch.Tensor`` — use this tensor directly (must have matching shape/dtype).
        - ``Callable`` — a no-argument callable that returns a ``torch.Tensor``, or
          one of the supported ``torch`` factory functions
          (``torch.randn``, ``torch.rand``, ``torch.zeros``, ``torch.ones``)
          that will be called with ``(shape, dtype=dtype)``.
    is_output: If ``True``, the tensor is an output to be validated against the
        golden reference.

Example:
    >>> import torch
    >>> TensorSpec("query", [32, 128], torch.bfloat16, init_value=torch.randn)
    >>> TensorSpec("out", [32, 128], torch.float32, is_output=True)

### `TensorSpec.create_tensor`

```text
TensorSpec.create_tensor(self) -> torch.Tensor
```

Create and return a ``torch.Tensor`` based on this specification.

Returns:
    Initialised tensor with the requested shape and dtype.

## `Worker`

```text
pypto.runtime.Worker() -> 'None'
```

Abstract runtime handle: memory + DeviceTensor + dispatch surface.

Concrete subclasses:

- :class:`~pypto.runtime.ChipWorker` — L2, single-chip execution.
- :class:`~pypto.runtime.distributed_runner.DistributedWorker` — L3+,
  multi-chip / forked execution.

Library code that doesn't care about the execution level should type-hint
against this base. Most users should construct one of the concrete
subclasses directly.

Subclass contract:

- Implement :meth:`malloc`, :meth:`free`, :meth:`copy_to`,
  :meth:`copy_from` with backend-appropriate routing and a readiness guard.
- Implement :meth:`run` and :meth:`register` for the level's dispatch
  semantics. ``register`` should return a :class:`RegistrationHandle`.
- Override :meth:`_require_ready` to match the level's lifecycle (e.g.,
  ``ChipWorker`` requires :meth:`init`; ``DistributedWorker`` requires
  not-yet-closed).
- Override :meth:`_prepare_init` if the host-init upload policy differs
  from L2's defensive CPU copy.
- Call :meth:`_close_owned_tensors` from the subclass's ``close()`` before
  tearing down its backend so any tensor the caller forgot to release is
  reclaimed.

### `Worker.malloc`

```text
Worker.malloc(self, nbytes: 'int', *, worker_id: 'int' = 0) -> 'int'
```

Allocate ``nbytes`` of device memory; return an opaque pointer.

### `Worker.free`

```text
Worker.free(self, ptr: 'int', *, worker_id: 'int' = 0) -> 'None'
```

Release a pointer previously returned by :meth:`malloc`.

### `Worker.copy_to`

```text
Worker.copy_to(self, dst_dev_ptr: 'int', src_host_ptr: 'int', nbytes: 'int', *, worker_id: 'int' = 0) -> 'None'
```

H2D copy: ``nbytes`` bytes from host *src_host_ptr* to device *dst_dev_ptr*.

### `Worker.copy_from`

```text
Worker.copy_from(self, dst_host_ptr: 'int', src_dev_ptr: 'int', nbytes: 'int', *, worker_id: 'int' = 0) -> 'None'
```

D2H copy: ``nbytes`` bytes from device *src_dev_ptr* back to host *dst_host_ptr*.

### `Worker.run`

```text
Worker.run(self, compiled: 'Any', *args: 'Any', config: 'Any' = None) -> 'Any'
```

Dispatch *compiled* on this Worker explicitly.

Subclasses define the accepted *compiled* type:

- :class:`~pypto.runtime.ChipWorker` accepts a
  :class:`~pypto.ir.CompiledProgram`.
- :class:`~pypto.runtime.distributed_runner.DistributedWorker` accepts
  a :class:`~pypto.ir.DistributedCompiledProgram`.

Return value follows the same rules as the implicit
``compiled(...)`` callable surface.

### `Worker.register`

```text
Worker.register(self, compiled: 'Any') -> 'RegistrationHandle'
```

Pre-register *compiled* on this Worker. Returns a callable handle.

Calling the handle dispatches without re-checking the registry cache.
Multiple ``register`` calls for the same *compiled* return aliases of
the same underlying cid. ``RegistrationHandle.unregister()`` only
marks that handle closed; the underlying cid is released once, in
:meth:`close`.

### `Worker.alloc_tensor`

```text
Worker.alloc_tensor(self, shape: 'Sequence[int]', dtype: 'torch.dtype', *, init: 'torch.Tensor | None' = None, worker_id: 'int' = 0) -> 'DeviceTensor'
```

Allocate a device buffer and (optionally) upload host data.

Convenience wrapper around :meth:`malloc` + :meth:`copy_to`. When *init*
is provided its dtype and shape must match exactly; the host buffer
uploaded is :meth:`_prepare_init` applied to *init*. If any step after
:meth:`malloc` raises, the allocation is rolled back via :meth:`free`
before the exception propagates so callers never observe a leaked
pointer.

The returned :class:`DeviceTensor` is tracked by this Worker keyed by
``(worker_id, data_ptr)`` and retained as the allocation identity: if
the caller does not :meth:`free_tensor` it before ``close()``, the
subclass's ``close()`` (via
:meth:`_close_owned_tensors`) reclaims it against the same worker.
Because a :class:`DeviceTensor` does not itself carry its worker scope,
a caller that allocates on a non-default worker MUST pass the same
``worker_id`` to :meth:`free_tensor`.

Returns:
    A :class:`DeviceTensor` referencing the allocated buffer.

### `Worker.free_tensor`

```text
Worker.free_tensor(self, t: 'DeviceTensor', *, worker_id: 'int' = 0) -> 'None'
```

Release a buffer previously returned by :meth:`alloc_tensor`.

Frees the underlying pointer against *worker_id* (which MUST match the
``worker_id`` used to allocate *t*), then untracks *t* only after the
backend reports success. A failed backend admission/free keeps the
ownership key so the caller can retry.
Idempotent for a genuine double-free: if ``t.data_ptr`` is no longer
tracked under *any* worker (e.g. ``_close_owned_tensors`` ran first, or
the caller already freed it), this is a no-op — the underlying ``free``
is NOT called a second time. This protects against a double-free at the
C++ layer, and against a post-close ``RuntimeError`` when explicit
cleanup races with auto-free.

Raises:
    ValueError: if ``t.data_ptr`` is still tracked but under a different
        ``worker_id`` — silently no-oping there would leak the buffer
        until ``close()`` — or if the address is now owned by a newer
        allocation, so a stale handle cannot free the replacement.
