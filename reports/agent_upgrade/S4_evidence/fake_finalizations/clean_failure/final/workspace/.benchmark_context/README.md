# Benchmark Workspace

Case: `test.case`

## Files

- Files outside `.benchmark_context/` and `.benchmark_outputs/`: editable Candidate
  implementation and its local helper dependencies.
- `.benchmark_outputs/`: evaluator-generated results, compiler output, and profiling diagnostics, separated by mode.
- `.benchmark_context/case.json`: Case description and evaluation mode.
- `.benchmark_context/config.py`: fixed configuration for this Case; read-only information.
- `.benchmark_context/comparator.json`: correctness comparison rules; read-only information.
- `.benchmark_context/reference.json`: public summary of certified reference status and performance.
- `.benchmark_context/environment.json`: target platform and device summary.
- `.benchmark_context/workspace.json`: binding between this workspace and the Case.
- `.benchmark_context/profiling_modes.json`: profiling modes available in this workspace.
- `.benchmark_context/starting_point/`: read-only copy of the Candidate at workspace creation.

The files under `.benchmark_context/` are informational snapshots. Official evaluation
reads the canonical Case and sealed assets outside this workspace.
Case identity and budget come from the protected Session
registry. `PTOBENCH_SESSION_ROOT` must be set, and this root must be registered.

## Candidate Contract

The evaluator loads `candidate.py` and calls `entry`. The entry may
import any local files copied into this workspace. Preserve the input/output ABI
described by `case.json` and `config.py`; implementation files, helper layout,
and optimization choices are otherwise free.

## Evaluate

```bash
ptobench evaluate .
```

Formal inputs and reference output tensors are hidden from this workspace.
The default result and compiler output are written under
`.benchmark_outputs/clean/`.
These outputs are diagnostic copies. Bench records authoritative results and
readonly Snapshots from the frozen implementation; edits here cannot change them.
Evaluate reserves one exploration action; failure does not refund it.

## Fork

```bash
ptobench fork .
```

Bench chooses and registers the child workspace and returns its absolute path.
Fork copies implementation files, regenerates context and consumes no hardware action.


## Diagnostics

The following optional diagnostics are available:

- `ptobench profile runtime .`: Runtime scheduling, dependency, and scope diagnostics.
- `ptobench profile pmu .`: Hardware PMU diagnostics.
- `ptobench profile compiler .`: Compiler pass IR and compile-time diagnostics.

Each profiling command performs one correctness dispatch; it does not measure
performance, and reserves one exploration action. Profiling results are diagnostic only. After an optimization, run
`ptobench evaluate .` again for the formal clean performance result.

