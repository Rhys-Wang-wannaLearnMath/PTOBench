# Stage 5 requirement audit

Overall completion is unproven. The latest user decision waives waiting for selected-Mode protection and authorizes missing canonical Profile collection before the paired trials. Evidence is scoped to the requirement it actually exercises.

| Requirement | Current evidence | Status |
|---|---|---|
| Full plan read; sequential Task0–5 execution | progress.md; implementation commits; configuration/report preparation | implementation progressed; full goal incomplete |
| No forbidden public/Basic/CANNBot/Case/collector changes | evidence/identity_audit.json compares 1,118 initial source hashes; own commits touch only this Mode | verified at audit time |
| Preserve unrelated dirty and staged work | evidence/worktree_start.txt; explicit path staging and --only commits | preserved |
| Input identity includes uncommitted source and actual Provider | inputs.json per-file hashes, effective deepseek-flash/high | verified |
| Full Formal60+Heldout15 combined completion index | [selected coverage audit](evidence/selected_coverage_audit.json), rebuilt from raw metadata; missing canonical Candidates collected by normal CLI | in progress |
| Generic Pi extension/socket interface | public_pi_tool_tests.log; version identity in inputs.json | CPU verified once |
| Selected Mode source bound by Launcher/Session protection | remains absent in public code | waiting waived by latest user decision; no public change |
| Frozen runtime only, original retained | analyzer_manifest.json; copy test; identity audit | byte-identical; adaptations outside copy |
| Static nine tool specs, names/schema/role mapping | test_copy_and_specs_match_source; task2_green.log | CPU verified; no initial_summary |
| No-Profile discovery/error, later/new/old/fork Profile selection | test_profile_selection_lifecycle | CPU verified |
| Explicit ID and cursor authorization; cross-session/case/path rejection | same adapter lifecycle test | CPU verified |
| Consistent Session publication snapshot, lock released before analysis | controlled writer test test_profile_open_uses_consistent_session_snapshot | CPU verified |
| Recent Profile cache, lazy role groups, stable cursor process | adapter implementation; lifecycle pagination | CPU verified; review cache-authority fix verified in latest11 |
| Nine native extension registrations and full socket calls | real TS loaded by Node22 with Pi API stub, roundtrip test | protocol verified, not real Pi/LLM |
| session_start registered+active sets audit | extension and broker roundtrip | stub evidence only; real evidence pending |
| Full output/error/call ID/Profile/timestamps and generated archive dirs | roundtrip audit test | CPU verified; review serialization-terminal fix verified in latest11 |
| Pre-start/in-flight cancellation, disconnect, duplicate, close | test_cancel_disconnect_and_close_are_recorded | CPU verified; noninterruptible synchronous analysis documented |
| Forced termination request finalization | Fake collect test | CPU verified |
| Bounded request, serial analysis, no NPU/CLI route | 1MiB ingress/serial lock code; fixture read-only comparison | CPU/code evidence |
| Basic single Agent and normal objective/CLI retained | Fake four-step Mode test, instruction equality, prompt prefix | CPU verified |
| Pi-only Mode v2, combined, existing helpers/interface | mode declaration and wiring test | CPU verified |
| Wiki independent snapshot/visible files; right paths/DAC; no links/special nodes | resources code, fixture Mode test, invalid-node test | CPU verified |
| Wiki readable/immutable under real Agent view and from fork | task3_wiki_green.log uses real UID/dispatcher/Bubblewrap/fork | CPU verified, no model action |
| Candidate/Context/protection manifest unaffected by Wiki | source hashes and integrity checks in same test | CPU verified |
| Prepare static archives only; collect freezes runtime archives | Fake Mode test; protocol records | CPU verified |
| Normal/error/SIGTERM cleanup; socket inside Session | Fake run and subprocess SIGTERM tests | CPU verified |
| Independent Provider secret storage and no credential archive | ignored .env; shared Provider/provenance; identity audit | verified local .env excluded; real archives pending |
| Six exact named configs and serial paired order, equal model/budget/seed/devices | experiment_manifest.json; launcher parser | prepared, unlaunched |
| Reuse only matching Basic records | two historical Session metadata records rejected for budget/reasoning mismatch | no exact reusable record found |
| Three real Basic/treatment pairs through PTOBench | all six Session paths null | pending; follows authorized Profile coverage step |
| Real Pi nine tools active, natural call behavior and Wiki evidence | no real Sessions | unverified; do not conflate with CPU stub |
| Per-Session performance/correctness/diff/actions/tokens/time/end reason | summary.json explicit nulls/reasons | pending raw Sessions |
| Separate registered, actually called, later cited, final benefit | summary claims and report distinctions | accurately separated; actual behavior unmeasured |
| Three combination comparisons without individual attribution/general claims | summary.json pending pair rows and report limits | pending data |
| No 75-Case rerun/extra smoke/forced calls | no real launcher run issued; ledger | complied |
| Final identity/diff/tracking/credential checks | identity_audit.json; git diff --check; final ledger | verified; only Mode report/config edits remain |
| Independent whole-implementation review | evidence/independent_review.md; two regressions RED→GREEN, Mode suite11 passed | completed |

A test passing supports its listed CPU behavior only. Missing real Session metrics and visibility evidence are not zero outcomes and cannot be marked complete.
