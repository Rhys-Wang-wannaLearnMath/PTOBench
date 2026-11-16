# Stage 5 requirement audit

Stage 5 is complete for the stated observational objective. Evidence is scoped to the behavior it directly records. Public PTOBench, Basic and CANNBot code were not modified. The selected-Mode protection dependency remains a public-code concern but was explicitly waived as a prerequisite for this work.

| Requirement | Evidence | Status |
|---|---|---|
| Full Formal60 + Held-out15 combined Profile coverage | [selected coverage audit](evidence/selected_coverage_audit.json): 75/75 canonical Candidate matches, 0 failed, 0 pending; one selected record per Case | complete |
| Coverage duplicate handling and provenance | 94 Session/Event references reviewed; 75 selected records and duplicate references retained; Session/Event/index and device records included | auditable |
| Generic Pi extension/socket interface | [public Pi tool tests](evidence/public_pi_tool_tests.log): 2 passed, 26 deselected, 9 subtests | CPU verified once |
| Mode implementation and runtime bridge | Earlier [Mode suite](evidence/review_fixes_mode_suite_green.log): 11 passed. After the socket-path fix, `test_mode.py` passed 4 tests, the focused socket test passed, and the affected bridge/adapter suite passed 8 tests; runtime copy identity is retained | CPU verified for changed paths |
| Wiki readable and immutable | CPU sandbox/fork verification plus unchanged Wiki hash in each treatment Session archive | verified |
| Six matched real PTOBench Sessions | [summary](summary.json): three cases × Basic/treatment, normal CLI and device locks; exact Session/event paths recorded | complete |
| Full Evaluate/Profile/finish and hardware evidence | Per-event records in summary and original `events.jsonl`, result/authority artifacts; device groups [0,1] and [0..7] | auditable |
| Nine Analyzer tools registered/active | Real treatment Session registration archives show all nine active in all three treatment Sessions | verified |
| Actual tool calls and outcomes | Five successful Analyzer calls (3 + 1 + 1); no Analyzer error/cancel/interruption. Agent harness tool errors and Candidate compile/correctness failures are separated in summary | recorded |
| Non-finite JSON transport edge | All five live Analyzer calls succeeded with no recorded errors; a synthetic Infinity fixture verifies explicit Broker error recording ([targeted test log](evidence/review_fixes_targeted_green.log)) | not observed in real Sessions; regression-tested |
| Later Agent use of Analyzer results | Durable logs do not establish later decision use; invocation alone is not treated as use | unconfirmed |
| Candidate identity and diffs | Final selected Candidate SHA-256, initial Profile Candidate SHA-256 and per-file line diffs against Event 1 snapshot for all six Sessions | recorded |
| Correctness, Clean Latency, tokens, wall-clock, end reason | Final finish result and Launcher recording in each Session archive and summary | recorded |
| Paired outcome attribution | Three descriptive Basic versus Wiki + PTOAnalyzer comparisons; no claim for either individual component | appropriately scoped |
| Recovery failures | Two zero-action prelaunch failures and one interrupted Trial 4 precursor preserved separately; no smoke/seed/full-agent extras | disclosed |
| Selected Mode source protection | Intentionally not required by the Stage 5 design; public PTOBench remains unchanged | out of scope by design |
| Unchanged-test policy | Socket-path changes received focused and affected CPU checks. The public Pi interface suite (2 passed/9 subtests) was reused because its source was unchanged | complied |

No production or public PTOBench code changed during this continuation. Numerical latency differences are not causal evidence for either Wiki or Analyzer individually.
