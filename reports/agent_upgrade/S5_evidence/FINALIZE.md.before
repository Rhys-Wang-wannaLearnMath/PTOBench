# Internal finalization (S4)

Use these Python APIs from the benchmark Evaluator import path:

```python
from snapshots import select_best_snapshot, finalize_session

selection = select_best_snapshot(session_root)  # Selection or None; no measurement
result = finalize_session(session_root, "optimizer_exit")  # FinalResult
payload = result.to_dict()
```

These entry points acquire the global execution lock. Do not hold it while
calling them. Their internal implementation uses the S3 caller-held execution
and postcheck helpers and directly launches the existing Evaluator; it never
calls the public CLI. Finalization is absent from the five-operation Agent
parser. Launcher integration and automatic triggering remain S5 work.

The experiment manager first closes exploration and stops/reaps the Optimizer
and its remaining workers. It must allow an already accepted last hardware
action to finish naturally when budget is exhausted. `finalize_session` also
closes exploration using a short Session update before waiting for the global
lock. It recovers attributed abandoned action groups before repairing assets.
A living unmarked owner or an ambiguous spawn remains blocked. No API here
claims to stop an unregistered Optimizer process; the future Launcher owns that
prerequisite.

Selection repairs protected Session/events/authority and Snapshot trees with
S1, then verifies the registered path/readonly binding, source manifest/hash,
metadata identity, successful source event, correctness/integrity, and the
independent authority result. A Snapshot observation must agree with its
Evaluator result and event. Published but unregistered or unreferenced orphan
directories are ignored. Inconsistent committed evidence or failed recovery
rejects selection instead of silently falling back to another source.

The fixed policy is `lowest_observed_latency_else_latest_pass`:

1. Among finite positive `latency_us` observations, choose the lowest value;
   equal values choose the smaller event sequence.
2. When every valid candidate has null latency, choose the latest PASS event.
3. With no valid Snapshot, persist failed/no_valid_snapshot and consume no
   finish action. The current editable Candidate and Seed are never substitutes.

Profile observations retain their public mode and `median_effective_us`/us
provenance and can guide selection, including when their values differ from
clean evaluation. Null observations do not displace a timed candidate. There
is no additional exploration timing, wall-clock estimate or synthetic speedup.

Finalization copies only the selected Snapshot source to a fresh
`final/workspace`, excludes old context/outputs/caches, and regenerates public
context from canonical assets. Snapshot source remains readonly; the copied
workspace can produce its own compiler output. It saves protected
`final/source_manifest.json` and `final/selection.json`, registers context, and
reserves the one independent finish action. The invocation fixes phase=all and
profile-mode=clean, uses `final/result.json` as its independent Evaluator result,
and preserves stdout/stderr in `final/`. Existing RunConfig/measurement behavior
is unchanged.

Final execution uses S3's start pipe, separate process group, dual-log draining,
signal cleanup and finally postcheck. An accepted failure or interruption keeps
finish usage=1, closes the Session as failed and never tries another Snapshot.
A postcheck change invalidates this result even if the Evaluator printed PASS.
Inability to attribute/stop a live group retains active evidence and prevents
asset repair or further hardware execution. Infrastructure failure before
reservation consumes no finish action.

One S1 commit publishes the finish event, Session terminal status/time, cleared
active action, protected complete Evaluator bytes in `final/authority_result.json`,
and `final_result.json`. Failure raw bytes are diagnostic evidence and cannot
supply verified values. No final exploration Snapshot is created. The final
result includes identity/close reason, selected event/hash, policy/reason and
complete observation provenance, both usage counters, finish event sequence,
error and verified metrics. Official comparison reads only completed results
with `final_integrity_passed=true`, finite positive `final_verified_latency_us`
and `final_verified_speedup`; failure verified fields are null.

A committed final result is restored and validated before returning the same
payload on reentry; editable raw/diagnostic files cannot change it. Concurrent
finalizers serialize and return that same result. A consumed finish without a
committed final result produces failed/interrupted (or its already recorded
failure), even if a pre-interruption raw file says PASS. It never repeats the
Evaluator, charge or finish event. A crash after terminal manifest publication
restores the complete committed result; a crash before it retains the consumed
finish and yields failure. Corrupt recovery sources or records may prevent
reliable terminal publication; the API returns infrastructure_error, retains
evidence and performs no substitute measurement.

CPU fault coverage is in `tests/test_finalize.py`, using only temporary fake
Cases and controlled Evaluator processes. Real NPU, measurement equivalence and
Optimizer integration are not certified by these tests.
