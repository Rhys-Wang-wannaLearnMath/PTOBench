# PTOAnalyzer Stage 5 — incomplete, external prerequisites pending

`modes/basic_wiki-tool` implements Basic's single Pi Agent lifecycle plus an optimization Wiki and nine read-only Analyzer tools. Tasks 1–3 have CPU acceptance evidence. Six paired-trial configurations are prepared, but **no real trial Session has been launched**. Stage 5 is not complete.

- [Progress and exact commands](progress.md)
- [Input file identities and external dependency audit](inputs.json)
- [Six configurations, pairing identities, seed manifests and reuse audit](experiment_manifest.json)
- [Structured results with unmeasured fields left null](summary.json)
- [Requirement audit](acceptance_audit.md)
- [Independent review and fixes](evidence/independent_review.md)

CPU evidence proves nine registrations through a Pi API stub and real TS/Node→Unix socket→Analyzer roundtrips, plus pagination, typed errors, cancellation, interruption, duplicate rejection and cleanup. See [bridge/adapter acceptance](evidence/task2_green.log). It does not prove real Pi/model visibility or actual model calls.

Latest affected acceptance: **11 Mode CPU tests passed** in [review_fixes_mode_suite_green.log](evidence/review_fixes_mode_suite_green.log). Two independent-review findings were fixed with failing-then-passing regressions: cached authority is checked on every call without reparsing heavy diagnostics, and nonfinite-result serialization errors have durable typed terminal records with Profile identity. Historical failure/correction logs remain available in the evidence directory.

Mode acceptance covers Basic contract preservation, prepare/run/collect wiring, normal/startup-failure/SIGTERM socket cleanup and interrupted-call finalization. Wiki verification used a CPU fixture Session, a real fork and the existing Agent sandbox under the real `ptobench-agent` UID. Wiki INDEX reading succeeded, mutation/replacement was denied, Candidate/Context hashes and existing protection remained valid. No LLM, API or NPU was invoked.

The generic public Pi extension/socket tests passed once: [public evidence](evidence/public_pi_tool_tests.log). No public code was changed here. The original Analyzer runtime is retained; its copy and SHA-256 manifest are Mode-owned, while adaptation stays outside the copy.

Real Pi registration/active-tool evidence is absent until the three treatment Sessions run. Actual model tool calls, observed Wiki reads, later decisions citing tool results and optimization benefit are all unmeasured. Unlaunched Session metrics are null with an explicit reason, not zero. After real trials, report all outcomes, including failures and zero tool calls, without adding forced-call or smoke Sessions.

Two launch prerequisites remain:

1. Public selected-Mode source protection: `create_session/_fixed_roots` still uses a fixed Mode list and Launcher passes no selected source directory. The generic Mode plan's Task 1 must be implemented and accepted by its owner. Do not add this Mode to a static list or bypass protection.
2. Full Formal60+Heldout15 combined Profile completion index: six existing queue indexes provide verified completed combined events for 26 distinct Cases, including non-Formal fixtures, but no complete 75-Case index was located. Public plan05's PASS summary means 75 correctness Evaluates plus eight representative combined Profiles; it is not full combined-profile coverage. No 75-Case replay/recollection was performed.

The experiment manifest fixes effective deepseek-flash/high, Pi1.5.2, budget20+finish1 and Case seed identity. Case runtime device groups are [0,1], [0,1] and [0..7]. Two historical Basic lm_head Sessions were inspected and rejected for reuse because their budget/reasoning/profile differ. Run configurations in their numbered serial order only after prerequisites and frozen identities are revalidated. Output roots must follow existing Broker/group traversal policy and PTOBench device locks.

Only the Wiki+Analyzer combination can be evaluated by these three pairs. Do not attribute effects separately or claim universal performance improvement. Profile measurements are diagnostic and never Clean Latency.
