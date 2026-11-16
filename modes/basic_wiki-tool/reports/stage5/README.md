# PTOAnalyzer Stage 5 — completed

Stage 5 completed with 75/75 canonical Formal/Held-out combined Profiles and six completed PTOBench Sessions forming three matched Basic versus `basic_wiki-tool` pairs. The two CP8 arms each used the confirmed 10-Evaluate allowance; combined Profiles and the official finish action are recorded separately.

| Case | Basic clean latency | Wiki + Analyzer clean latency | Treatment / Basic |
|---|---:|---:|---:|
| `deepseek_v4.lm_head_tp2_flash` | 802.400 µs | 877.580 µs | 1.094× |
| `deepseek_v4.prefill_layer_ep2_csa_demo` | 18,545.710 µs | 9,192.289 µs | 0.496× |
| `deepseek_v4_flash_mtp.prefill_cp_csa_cp8_demo` | 7,204.310 µs | 6,518.110 µs | 0.905× |

These are descriptive matched-session outcomes. The treatment combines the Wiki and PTOAnalyzer; the experiment does not isolate either component. All six final candidates passed official finish correctness. The full Candidate hash/diff, Evaluate/Profile Event ledger, device group, failures, tool registrations/calls/errors, Wiki identity, token and wall-clock records are in [paired Session report](paired_trials.md), [summary.json](summary.json) and the original PTOBench Session archives.

The [coverage audit](evidence/selected_coverage_audit.json) records one selected canonical Profile per each of the 75 Cases (60 Formal + 15 Held-out); 94 profile references were reviewed: 75 selected, 18 usable duplicates not selected by the stated preference, and one unusable original retained as failure evidence. The Mode's copied Analyzer runtime identity and public/Basic/CANNBot boundaries are summarized in [acceptance_audit.md](acceptance_audit.md).

All three treatment Sessions registered all nine Analyzer tools and recorded five successful Analyzer calls in total. Tool invocation is distinct from registration; no durable evidence establishes whether the Agent later used a returned Analyzer result. Wiki snapshots were verified unchanged. The earlier full Mode suite (11 passed) and public Pi interface suite (2 passed/9 subtests) remain at [review_fixes_mode_suite_green.log](evidence/review_fixes_mode_suite_green.log) and [public_pi_tool_tests.log](evidence/public_pi_tool_tests.log). After the Mode-local socket-path fix, `test_mode.py` passed 4 tests, the focused socket test passed, and the affected bridge/adapter suite passed 8 tests. The public suite was reused because public sources did not change.

Two launch attempts failed before Agent startup with zero hardware actions, and one interrupted Trial 4 precursor is retained separately with its nine actions and pending fork record. The completed paired Sessions and all recovery records are listed in [progress.md](progress.md) and [summary.json](summary.json). No extra smoke Session, seed, or full Agent experiment was added.
