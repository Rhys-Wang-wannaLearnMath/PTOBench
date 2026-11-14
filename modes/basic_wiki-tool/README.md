# basic_wiki-tool

Basic's single Pi Agent with the same four-step lifecycle, instructions, optimization objective and PTOBench CLI. This Mode adds only the optimization Wiki and nine read-only PTOAnalyzer tools. It does not dispatch subagents, automatically collect Profiles, run initial summaries or force tool use.

The runtime in `pto_analyzer/` is a byte-identical copy pinned in `analyzer_manifest.json`. All selection, transport and auditing live outside that copy. Runtime use does not import `modes/ptomni/pto_analyzer`.

Use an independent local `.env` based on `.env.example`; credentials stay local and are ignored by Git. The default Pi profile is resolved through the existing Provider/recording helpers. Compare the effective model and reasoning, not just the profile name. Other backends are rejected.

Wiki snapshots are archived in `<session>/method/resources/opt_wiki/`. A separate ordinary-file copy at `<initial_workspace>/.mode_resources/opt_wiki/` is readable by `ptobench-agent` through existing Workspace mounts and protected by Broker ownership and DAC. The prompt supplies its absolute `INDEX.md` path, also usable from forks. Collect checks both copies against the frozen manifest.

The extension registers all nine tools from static specifications before any Profile exists. Produce a completed combined Profile with normal `ptobench profile .` first. A call without `profile` uses the newest valid completed Profile in this Session. For older Candidates and pagination, pass the returned `profile.event_id` explicitly; a cursor never chooses a Profile. Analysis reads the archived Profile and uses no hardware.

Pi's native extension talks to a Python Broker over the current Session's `agent/analyzer.sock`, using the existing generic Pi extension/socket interface. Each invocation has a Broker-generated directory under `method/analyzer_calls/` with its original call ID, complete request and terminal result. Errors retain their type and selected Profile when available. Pre-cancelled/queued calls are not executed; cancellation during synchronous analysis discards its eventual response. Disconnects/shutdown are recorded as interrupted. Collect finalizes requests left without a result after forced termination. Duplicate IDs get their own error record and cannot reexecute or overwrite the original.

`method/analyzer_registration.json` separately records the registered and active Pi tool sets. This proves visibility only when emitted by real Pi; it is not evidence of an actual model call or optimization benefit.

Stage 5 status and CPU evidence: [reports/stage5/progress.md](reports/stage5/progress.md). The latest user decision authorizes missing canonical starting Candidate Profile collection through normal PTOBench CLI, followed by the six planned paired Sessions. Selected-Mode source protection is not a waiting condition. Do not add smoke Sessions or extra Agent trials.

CPU checks use the existing official Python environment and `PYTHONPATH=.:evaluator:evaluator/tests`. Follow the ledger to reuse passed checks whose code has not changed. No dependencies are added.
