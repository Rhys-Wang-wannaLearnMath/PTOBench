# basic_wiki-tool

Basic's single Pi Agent with the same four-step lifecycle, instructions, optimization objective and PTOBench CLI. This Mode adds only the optimization Wiki and nine read-only PTOAnalyzer tools. It does not dispatch subagents, automatically collect Profiles, run initial summaries or force tool use.

The runtime in `pto_analyzer/` is a byte-identical copy pinned in `analyzer_manifest.json`. All selection, transport and auditing live outside that copy. Runtime use does not import `modes/ptomni/pto_analyzer`.

Use an independent local `.env` based on `.env.example`; credentials stay local and are ignored by Git. The default Pi profile is resolved through the existing Provider/recording helpers. Compare the effective model and reasoning, not just the profile name. Other backends are rejected.

Wiki snapshots are archived in `<session>/method/resources/opt_wiki/`. A separate ordinary-file copy at `<initial_workspace>/.mode_resources/opt_wiki/` is readable by `ptobench-agent` through existing Workspace mounts and protected by Broker ownership and DAC. The prompt supplies its absolute `INDEX.md` path, also usable from forks. Collect checks both copies against the frozen manifest.

The extension registers all nine tools from static specifications before any Profile exists. Produce a completed combined Profile with normal `ptobench profile .` first. A call without `profile` uses the newest valid completed Profile in this Session. For older Candidates and pagination, pass the returned `profile.event_id` explicitly; a cursor never chooses a Profile. Analysis reads the archived Profile and uses no hardware.

Pi's native extension talks to a Python Broker over the current Session's `agent/analyzer.sock`, using the existing generic Pi extension/socket interface. Each invocation has a Broker-generated directory under `method/analyzer_calls/` with its original call ID, complete request and terminal result. Errors retain their type and selected Profile when available. Pre-cancelled/queued calls are not executed; cancellation during synchronous analysis discards its eventual response. Disconnects/shutdown are recorded as interrupted. Collect finalizes requests left without a result after forced termination. Duplicate IDs get their own error record and cannot reexecute or overwrite the original.

`method/analyzer_registration.json` separately records the registered and active Pi tool sets. This proves visibility only when emitted by real Pi; it is not evidence of an actual model call or optimization benefit.

Stage 5 is complete: the [final paired-session report](reports/stage5/paired_trials.md) documents three Basic versus Wiki + PTOAnalyzer pairs; the [structured audit](reports/stage5/summary.json) records all six Session/Event roots, Candidate hashes and diffs, correctness, Clean Latency, tool use, Wiki identity, tokens, wall-clock and stop reasons. The [75-Case coverage audit](reports/stage5/evidence/selected_coverage_audit.json) selects one canonical combined Profile for each Formal/Held-out Case. See the [progress ledger](reports/stage5/progress.md) and [acceptance audit](reports/stage5/acceptance_audit.md) for test evidence and remaining limits.

All three real treatment Sessions registered and activated the nine Analyzer tools; five calls succeeded. The records do not establish later decision use of those results. The measured pairs describe Wiki + PTOAnalyzer together and do not isolate either component. The two CP8 arms used a matched 10-Evaluate allowance; Profile events and official finish are separately accounted. No additional experiment is planned.

CPU checks use the existing official Python environment and `PYTHONPATH=.:evaluator:evaluator/tests`. Reuse passed checks while implementation code is unchanged. No dependencies are added.
