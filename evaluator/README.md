# Evaluator Contract

The top-level evaluator resolves the Case, loads its environment and dispatches to
`sealed_assets/<family>/<case>/evaluate.py`. The shared lifecycle in
`workflow.py` owns fixture iteration, Ref certification, output caching, timing,
and Candidate result formatting. The Case-local evaluator supplies only the
TensorSpec/ScalarSpec ABI, official entry, comparator, runner, output capture,
and CPU-oracle callback extracted from that Case's source.

Each Case-local evaluator provides two functions:

```python
prepare_ref(case_dir, sealed, case, platform=None)
evaluate_candidate(case_dir, sealed, case, candidate_workspace,
                   platform=None, phase="all")
```

This is the only Case interface. Do not add Case-name branches or a global
official replay module to the evaluator. Family-specific ABI helpers belong in
that family's sealed directory. Effective comparator settings remain public.

Candidate and Ref implementations may each be one file or multiple local files,
depending on the actual Case. Candidate Seed is an independent copy of the
final Ref implementation tree, with only `ref_npu.py` renamed to `candidate.py`.
Case adapters use `common.load_candidate(...)`, `common.load_ref(...)`, and
`common.load_oracle(...)`: all see canonical public config, while their local
helpers resolve from the Workspace, sealed `ref/`, and sealed `oracle/`
respectively. Oracle loading isolates same-named local modules from Ref's Python
module cache. Session creation recursively copies the complete Candidate seed tree.
Fixture generation runs in a separate process before Ref loading so official
and Ref helpers with the same module name cannot share Python import cache.

The internal `evaluate.py` reads the Case environment profile before importing PyPTO code. It
sources the configured toolchain scripts, sets the Case environment and
re-executes itself with the configured Python interpreter when necessary.
Callers do not need to activate the virtual environment or source CANN.

Prepare a Case once:

```bash
python benchmarks/PTOBench/evaluator/prepare_ref.py \
  --case deepseek_v4.compressor_ratio4_flash --platform a2a3
```

Internal maintenance evaluation (without an active Session):

```bash
python benchmarks/PTOBench/evaluator/evaluate.py \
  --case deepseek_v4.compressor_ratio4_flash --platform a2a3 \
  --candidate /absolute/prepared/workspace \
  --phase all --result latest_result.json
```

`correctness` executes one step. `performance` and `all` additionally run five
warmups and ten measured launches, restoring the fixture state before every
launch. State restoration is outside the measured device interval.

Optional profiling modes are configured in `profile_modes.py`:

```bash
ptobench profile runtime ./workspace
ptobench profile pmu ./workspace
ptobench profile compiler ./workspace
```

Every profiling mode performs exactly one correctness dispatch. `ptobench profile`
does not expose a phase override: profiling checks availability and emitted
diagnostics, not warmups, measured rounds, or performance.

Session creation and fork write only enabled public modes to
`.benchmark_context/profiling_modes.json` and lists only those modes in the
generated README. If no mode is enabled, neither the file nor a profiling
section is generated. Calling a mode absent from the protected Session reports that it
is not enabled.

Every Workspace owns its evaluator output:

```text
.benchmark_outputs/
├── clean/
│   ├── latest_result.json
│   └── build_output/
├── runtime/
├── hardware/
└── compile/
```

The evaluator runs with the Workspace as its current directory and sets the
official PyPTO `PYPTO_PROG_BUILD_DIR` to that mode's `build_output/`. Different
Workspaces and profiling modes therefore cannot overwrite one another's result,
compiler, or diagnostic files. An explicit `--result` path overrides only the
default result location.

The result records the exact JIT output directory in its top-level `build_dir`
field. This value comes from the compiled program rather than from scanning for
the most recently modified directory.

Profiling runs only the Candidate. Every Case has exactly one fixed formal
Fixture, used by Ref preparation, clean evaluation, and profiling. The
profiling timing is diagnostic only and no Ref speedup is reported.
`ptobench evaluate` always sets `profiling_mode` to `clean` and cannot inherit
profiling into the official score.

Agent operations require `PTOBENCH_SESSION_ROOT` and a registered workspace.
The old `bench` entry and parser have been removed. See [Agent CLI](AGENT_CLI.md)
for fixed arguments, budgets, frozen execution, events and Snapshots, and
[Session API](SESSION.md) for creation and accounting. `agent_cli.py` imports no
torch/PyPTO and performs no environment bootstrap. It launches this existing
Evaluator directly, which waits on an internal start pipe until its PID, active
record and logs are bound durably; it then uses the existing environment and
measurement lifecycle. Maintenance invocations without that pipe proceed
normally. The pipe is consumed before environment re-exec.

Direct acceptance controls use `session_actions.run_maintenance_process(argv,
environment, action_dir)` while the caller holds `session.execution_lock()`.
Prepare a fresh `action_dir/workspace` below benchmark `runs/`, call the existing
`evaluate.py` with its actual `--case`, `--candidate`, `--phase`, `--profile-mode`
and `--result` parameters, and use a new directory for each call. This internal
API shares the start gate, durable PID marker, dual logs, TERM/KILL/reap and
abandoned-worker recovery with exploration and finish. It charges no Session
action and publishes no official score. Arbitrary direct host scripts do not
automatically acquire this protection. For fresh Seed qualification, use the
the migration workflow's Session recipe when this benchmark is nested in pypto-lib.

Full v0.2 acceptance is a maintainer operation:

```bash
.venv-official/bin/python benchmarks/PTOBench/evaluator/accept_all_cases.py \
  --out benchmarks/PTOBench/reports/agent_upgrade/full_v02/BATCH_ID
```

The output must be new. The tool freezes every `cases/*/*/case.json` and full
canonical asset/public-code hashes, then checks the original Seed with real
`ptobench evaluate` and all three public profiles in independent Workspaces.
It never prepares a Ref or generates a Fixture. Pending, invalid and blocked
Cases keep four records. Each completed action immediately saves logs, result,
Session/event IDs, correctness, integrity and native diagnostic evidence.
`--timeout-seconds` defaults to 1800 per command; cleanup must be confirmed
before further execution. `inventory.json`, `results.jsonl`, `matrix.csv`,
`summary.json` and `report.md` describe one complete batch/version. Separate
finalization evidence is in `finalizations.json`. Exit 0 means every one of
the four records for every Case is PASS and both final hash checks pass;
otherwise exit 1. A fully recorded batch may still contain failures or blockers.
The tool adds no Agent CLI operation.

S4's [finalization API](FINALIZE.md) selects validated, protected observations
and runs one clean final evaluation. `snapshots.select_best_snapshot` performs
selection without hardware; `snapshots.finalize_session` handles independent
finish accounting, process recovery and atomic terminal publication. Neither is
an Agent CLI command. Official final values come only from `final_result.json`
after this verification, not from an exploration diagnostic or observation.

## Public workspace and integrity tools (S1)

`workspace.copy_candidate_tree(source, destination)` returns a source manifest
after publishing an independent, stable copy. The target must be absent or an
empty real directory. It preserves regular files and empty directories, and
excludes `.benchmark_context`, `.benchmark_outputs`, `.git`, `.pytest_cache`,
`__pycache__`, `.pyc` and `.pyo` at every level. Included source nodes, roots and
ancestors cannot be links, devices, FIFOs or sockets; an excluded entry itself
cannot be a link or special node either. Excluded directories are not traversed.

Copying scans the source, copies to a temporary sibling, rescans the source,
and checks the copy before renaming it into place. Concurrent edits permit an
initial attempt and at most three retries. Failure raises
`workspace.InvalidWorkspaceError` with `error_code="invalid_workspace"` and
does not publish a partial copy or overwrite a nonempty target.

`workspace.source_manifest(source)` uses the same inventory. `files` is the
sorted list of `{path, sha256}` pairs; `content_sha256` hashes that list encoded
as UTF-8 JSON with sorted keys, `ensure_ascii=False`, and separators `(',', ':')`.
File paths and bytes affect the hash; outputs, caches and modes do not. Empty
directories are preserved and checked in `directories`, separately from the
file-content hash. Non-UTF-8 filenames are rejected before publication.

`workspace.write_context(case_dir, sealed, case, destination, ...)` creates the
public context and the original `starting_point`. `init_workspace` and
`fork_workspace` use these public tools. Ordinary integrity recovery restores
captured context bytes; it does not regenerate context from edited Candidate
files. Context files remain mode `0444`, as before.

The Linux `integrity` module provides:

```python
from integrity import (
    ProtectionRoot, capture_protection, check_only, check_and_restore,
    commit_records, register_roots,
)

capture_protection(protection_root, {
    "assets": ProtectionRoot(case_assets, "fixed"),
    "context": ProtectionRoot(workspace / ".benchmark_context", "context", readonly=True),
    "session": ProtectionRoot(session_root / "session.json", "records"),
    "events": ProtectionRoot(session_root / "events.jsonl", "records"),
    "final": ProtectionRoot(session_root / "final_result.json", "records"),
})
report = check_only(protection_root, {"fixed", "context"})
repair = check_and_restore(protection_root, {"fixed", "context"})
commit_records(protection_root, {
    "session": new_session_bytes,
    "events": new_event_log_bytes,
})
```

Root paths are frozen lexical absolute paths; parents must exist and be real
directories. Roots must not overlap one another or the protection control
directory. A missing root (for example `final_result.json`) is recorded as
expected absence. Inventories include root and empty-directory types, file
SHA-256 values and modes. Only bytecode caches are excluded from protected
trees; Fixture `.pt` and Ref outputs are included. Backups are ordinary,
independent copies at `backups/<sha256>`, with same-content deduplication.

`IntegrityReport` lists `changed_paths`, `restored_paths` and
`unrecoverable_paths` as logical names plus relative paths. `passed` means no
change was observed; successful repair retains the original changed paths and
therefore does not turn an observation into a PASS. A subsequent `check_only`
can verify the repaired tree. `groups=None` selects all registered groups.

Recovery validates needed backup hashes, replaces files atomically in their
own directories, removes extra nodes, and repairs missing directories and
file/directory/link substitutions. It uses directory descriptors and
`O_NOFOLLOW`. A root or an inner link is removed itself; its target is never
used. An unsafe or missing ancestor outside a frozen root fails with
`InfrastructureError` (`error_code="infrastructure_error"`), rather than
repairing an unprotected parent. The exception carries the failure report.
Malformed manifests and corrupt/missing/linked recovery sources also fail;
there is no automatic recapture or git reset.

`commit_records` accepts exact bytes for registered `records` roots, or new
`logical_root/relative_file` paths inside a registered record directory. It
first requires clean current records, then saves and verifies all new backups,
atomically publishes one updated manifest, and replaces the working files.
After interruption, the manifest determines whether recovery installs all old
or all new records; budget and events cannot restore to different generations.
Dirty records must be repaired and updates recomputed before committing.

`register_roots` adds new context/Snapshot roots only; it refuses existing root
names, overlap, or dirty existing protection. Publish readonly context/Snapshot
files before capturing a `readonly=True` root. Capture itself does not chmod
working assets. Future Session callers must hold their Session lock across
repair/read/compute/commit and their global execution lock for shared-asset
recovery. These primitives additionally serialize manifest operations with
`protection/.integrity.lock`. Session lifecycle, budgets and Agent CLI routing
are implemented in `session.py` and `session_actions.py`.
