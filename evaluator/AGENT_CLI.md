# Agent CLI and managed observations (S3)

`ptobench` is the repository-local Agent entry. Use the full benchmark directory
on PATH, set `PTOBENCH_SESSION_ROOT` to an existing Session, and work in one of
that Session's registered workspace roots. Session creation is an [internal
Python API](SESSION.md); S3 does not provide a Launcher or finalization command.

```bash
ptobench evaluate [WORKSPACE]
ptobench profile runtime [WORKSPACE]
ptobench profile pmu [WORKSPACE]
ptobench profile compiler [WORKSPACE]
ptobench fork [PARENT_WORKSPACE]
```

Brackets mean an optional positional argument, defaulting to `.`. Absolute and
relative registered roots work. A subdirectory, link, another Session's worker,
or an unregistered root fails. Session identity and permissions come from
protected records, rather than workspace metadata. Missing Session environment
fails without searching or creating a Session. There are no phase, candidate,
result, workspace, RunConfig, budget, fork output/name, or other command flags.
The old entry and parser have been deleted; ordinary workspaces have no fallback.
Sessions captured before the cutover are rejected before fixed-asset recovery,
so their old protection backups cannot reinstall the retired entry/parser or
overwrite the new infrastructure. Keep their historical evidence and create a
fresh Session; their context is not rewritten.
Sessions captured before the diagnostic-only Agent profile policy also remain
readable as historical evidence, but cannot execute or restore their former
Evaluator backups. Create a fresh Session for subsequent operations.
Sessions captured before the shared C++ runtime and delayed Case-import repair
also require a fresh Session. Their old protection snapshots cannot reinstall
the previous environment loader. The Launcher checks compiler loading before
starting an Agent; every Evaluator invocation repeats that small check in its
selected environment before importing the Case.

| Operation | Internal Evaluator phase/mode | Exploration cost |
| --- | --- | ---: |
| evaluate | all / clean | 1 |
| profile runtime | correctness / runtime | 1 |
| profile pmu | correctness / hardware | 1 |
| profile compiler | correctness / compile | 1 |
| fork | source copy and fresh context | 0 |

The public names pmu/compiler map to existing hardware/compile behavior and
artifact directories. Agent profiling runs the Candidate without comparing its
outputs with the cached Ref. The internal `correctness` phase name is retained
for the one-shot execution path; it does not mean output validation was performed.
Profiles add no warmup or performance execution, omit the `correctness` field,
and report `latency_us=null`, `metric_name=null`, `unit=null`.
They retain runtime/compile errors, integrity checks, diagnostics and action costs.
Only evaluate and the final fresh clean establish Agent Candidate correctness.
Maintenance Sessions used for migration qualification retain their existing
profile correctness checks and result/Snapshot policy.
A successful evaluate requires finite positive latency and speedup.

Exploration evaluate uses two warmup launches and three measured launches;
latency is the median of the measured samples. Independent final verification
uses the Case's formal settings (currently five warmups and ten measurements).
Both paths retain raw measured samples and requested/actual measurement counts.
These settings are fixed by Bench and are not public CLI flags.

Before execution, the global lock serializes asset recovery, request validation,
stable Candidate/helper copying and budget reservation. The Evaluator runs from
`staging/action_<seq>/workspace`, with fixed arguments and a fresh result path.
An internal pipe keeps it waiting until the parent has durably bound PID/birth
identity and logs. The existing environment bootstrap then proceeds normally.
Inherited environment/profile markers are removed. The source inventory and
authority result are Bench-protected records.

The two raw streams are drained together into `stdout.log` and `stderr.log` and
forwarded to the terminal. Disk logs continue when a terminal consumer blocks;
forwarding uses those files as its spool and has a bounded final flush. An
unavailable or persistently blocked terminal can omit its tail, including the
final JSON, while disk evidence and completed events remain available. It does
not prevent action completion or keep the execution lock indefinitely.

On SIGINT/SIGTERM, Bench stops the owned Evaluator group, waits up to ten seconds,
then kills survivors and reaps the child. Repeated signals are deferred during
cleanup and evidence publication. Postchecks begin only after confirmed process
stop. If attribution or stop cannot be confirmed, Bench retains active evidence,
closes the Session and blocks execution rather than repairing under a live worker.
The next holder of the global lock first recovers an abandoned global marker,
including another Session's worker. PID birth identity prevents signaling a
reused PID. A reservation left before spawn is recovered once through the S2
lifecycle; a spawn interrupted before PID binding remains explicitly blocked.

Postchecks inspect assets, contexts, records and frozen source before restoring
changes. Record repairs made by concurrent status readers leave a protected
witness in the active action. Restoration cannot turn that observation into a
PASS. The editable original Candidate is retained. Accepted failures and
interruptions consume their reservation; rejected requests do not.

A valid observation requires zero Evaluator exit, PASS verdict and integrity PASS.
Evaluate additionally requires fixture_000 correctness PASS and saves a unique
readonly Snapshot of the same frozen source. An Agent profile saves protected
authority and diagnostics without a `correctness` field, but creates no selectable
Snapshot and cannot supply final-selection metrics. Snapshot metadata
records Session/Case/Optimizer, source workspace, event sequence/hash and metric
provenance. Outputs, context and caches are excluded. The completed event only
references a Snapshot after source/metadata verification and protection
registration; an orphan directory is never a successful candidate.

One protected `events.jsonl` entry records each completed or rejected operation,
its original Evaluator exit, error, integrity/repair evidence, budget usage and
Snapshot ID. Completion and active-action clearing are one S1 record commit.
Infrastructure failures close the Session and never expose an unreliable result
as success. Exploration exhaustion still permits the last accepted action to
finish. Fork assigns the child path and regenerates context without budget cost.

After authoritative completion, the complete diagnostic output set replaces
`.benchmark_outputs/<internal_mode>/` in the original worker. Root and mode
links are removed themselves; publication never writes through them.
`latest_result.json` is the Evaluator copy; `latest_session_result.json` includes
event sequence, status, integrity and Snapshot. They are editable diagnostics,
and changing them cannot alter protected records or scores. Staging remains
available for recorded build/report paths. Diagnostic publication failure does
not invalidate an already committed observation.

Normal terminal output ends with one JSON line containing `status`, `error_code`,
`exit_code`, `event_seq`, `workspace`, `correctness`, `integrity_passed`,
`latency_us`, `snapshot_id`, `remaining_hardware_actions`, and `message`.

| Error | CLI exit |
| --- | ---: |
| success | 0 |
| action_failed | 1 |
| invalid_request / invalid_workspace | 2 |
| budget_exceeded / session_closed | 3 |
| integrity_failed | 4 |
| infrastructure_error | 5 |
| interrupted | 130 |

The raw Evaluator exit is retained in the event, independently of the CLI exit.
Hash/readonly protection is an integrity mechanism, not OS isolation. The trusted
Bench control manifest/backups must not be edited; bypassing managed operations
to run maintenance scripts violates the experiment protocol.

The internal [S4 finalization API](FINALIZE.md) reuses the caller-held execution
helper and postcheck path. It selects protected Snapshot observations and
creates a clean final workspace for one independent finish action. The CLI
continues to expose only the five exploration operations above; finalization
belongs to the experiment manager. Real hardware acceptance remains a later
stage.
