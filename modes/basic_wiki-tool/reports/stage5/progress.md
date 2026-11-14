# Stage 5 ledger — plan: modes/ptomni/pto_analyzer/planning/stage5/plans/IMPLEMENTATION_PLAN.md

## 2026-11-15 — start / Task 0

- Read full implementation plan, Stage 5 spec, modes/PROTOCOL.md, Analyzer interfaces, ponytail, executing-plans, TDD and verification skills.
- Current branch `dev`, initial HEAD `45ff9cd`; unrelated dirty/staged paths will be preserved. Only this Mode's files will be staged/committed explicitly.
- Ruling: use the user-requested `reports/stage5/progress.md` as persistent ledger. Skill helper scripts reference absent `.agents/skills/subagent-driven-development`; record RED/GREEN commands here directly. The user's minimum validation and reuse rules override skill-wide full-suite/repeated-test requirements. No additional worktree: current authoritative uncommitted public interfaces are required for this implementation; changing checkout would discard that baseline.
- Pre-flight interfaces: Task 1 static manifest feeds Task 2 extension and Task 3 prepare; adapter binds the current Session, Task 2 owns audit/errors; Task 3 owns socket lifecycle and Wiki DAC; Task 4 needs all three plus public selected-Mode protection and input coverage. No conflicting API found.
- Public blocker verified in evaluator/session.py `_fixed_roots` and `create_session`, launcher/runner.py call: selected Mode source not passed/protected. Task 4 forbidden until public owner completes and validates this dependency.
- Environment finding: default Python lacks PyYAML and evaluator import path. Initial public test attempt failed collection (`ModuleNotFoundError: isolation`), not an interface test failure. Use existing `.venv-official/bin/python`, `PYTHONPATH=.:evaluator:evaluator/tests` thereafter. Log preserved as `evidence/public_pi_tool_initial_environment_error.log`.
- Tasks: 0 in progress; 1–3 pending; 4 external prerequisites pending; 5 pending. No model or hardware action started.

## Task 0 — input audit recorded

- `inputs.json` captures per-file SHA-256 of Basic/Analyzer/Wiki/public dependencies (including uncommitted changes), effective nonsecret Provider configuration and initial worktree status. Actual model `provider-default/deepseek-flash`, reasoning `high`; default profile's nominal model is not used for matching.
- Located six existing manual_runs queue indexes; verified small Session/Event records for 51 completed combined events covering 26 unique Cases. Full Formal60+Heldout15 completion index remains `not_located`; no large Profiles parsed/recollected. Existing Stage4's 16 Profiles are not 75-Case evidence.
- Public interface CPU acceptance: `PYTHONPATH=.:evaluator:evaluator/tests /data1/home/ziruiwang/pypto-lib/.venv-official/bin/python -m pytest evaluator/tests/test_isolation.py -k 'pi_tool_' -q` → exit 0, 2 passed, 26 deselected, 9 subtests; evidence/public_pi_tool_tests.log. Applicable public source hashes are in inputs.json. No repeat unless interface changes.
- External prerequisites remain selected-Mode source protection (verified absent) and full coverage index (not located).

## Task 1 — complete

- Runtime copied byte-for-byte (only __init__, profile, toolset, diagnostics); analyzer_manifest.json records source/copy identity and fixture-generated static specs. No initial_summary called.
- Adapter uses current Session only, explicit event IDs, same-session/case validation, latest legal completed combined event, old-ID pagination, one-Profile cache and lazy role sessions. Selection/open share public Session lock; analysis releases it.
- RED: `...python -m pytest modes/basic_wiki-tool/tests/test_analyzer_adapter.py -q` → 3 failures, missing adapter; evidence/task1_red.log.
- Test fixture correction: compute_mapping_check has a single detail record, so pagination uses tile_access_check (multiple fixture records). Not a production workaround.
- GREEN: same command → exit 0, 3 passed in 0.22s; evidence/task1_green.log. No need to repeat these tests while Task 1 code remains unchanged.

## Task 2 — acceptance correction in progress

- Added bounded (1 MiB) threaded Unix socket ingress, serial Analyzer execution, generated call-directory identities, durable request/result records, duplicate-ID rejection, cancellation/connection interruption and server close cleanup. No NPU/CLI route exists in broker or adapter.
- Extension reads colocated static specs, registers nine tools before Profile availability, forwards Pi call IDs/AbortSignal (including pre-cancelled requests), and records session_start registered+active names through an internal transport action.
- RED: bridge test command → 2 failures (broker absent), evidence/task2_red.log.
- First GREEN: 2 passed. Strengthened assertion exposed missing selected Profile in post-selection parameter errors (evidence/task2_error_identity_red.log). Adapter now annotates the original exception with selected identity; no Analyzer-copy changes.
- Affected regression: `PYTHONPATH=.:evaluator:evaluator/tests /data1/home/ziruiwang/pypto-lib/.venv-official/bin/python -m pytest modes/basic_wiki-tool/tests/test_pi_bridge.py modes/basic_wiki-tool/tests/test_analyzer_adapter.py -q` → 2 failed / 3 passed due to `tools` name collision with repository package; evidence/task2_combined_import_failure.log. The original completion statement was premature and is corrected here. Repeated roundtrip only because error-audit behavior changed. Fixture originals/Session/Event bytes verified unchanged in same test.
- CPU protocol evidence only: real Node 22.23.1 loads real TS; Pi APIs are stubs. No real Pi visibility or LLM call claimed yet. Synchronous ongoing analysis finishes before cancellation discards its response; no immediate parser interruption claimed.
- Task 2 correction complete: import broker by explicit Mode namespace via importlib, avoiding unrelated repository `tools` package already loaded during combined test collection. Latest affected regression command above → exit 0, 5 passed in 0.74s (evidence/task2_green.log). Previous commit saved a failed combined-test state before output was assessed; retained failure log and corrected ledger instead of hiding it.
- Task 1 commit: 32784d1; Task 2 implementation commit: 672dd57, followed by import correction. Runtime Analyzer copy remains byte-identical.

## Task 3 — complete

- Mode v2, Pi-only, combined; copied current Basic instructions and base prompt, adding only Wiki path and optional Analyzer usage. Independent ignored `.env` copied from existing Basic without printing credentials.
- Preserved Basic Provider/provenance/recording and agent_launch.run_process. Prepare writes extension/spec files and freezes only static archives; runtime audit is collected later. Run opens socket before launching and closes it on normal/error/SIGTERM/SIGINT; collect finalizes interrupted requests and verifies Wiki copies.
- Wiki archived and visible copies are independent ordinary files; visible directories 0750/files0640, Broker-owned + ptobench-share. Workspace root permissions unchanged. Snapshot files0440/directories0550. Source symlinks and FIFOs rejected.
- RED: `...python -m pytest modes/basic_wiki-tool/tests/test_mode.py -q` → 4 expected missing-Mode failures; evidence/task3_red.log.
- First implementation run: 3 passed (four-step Fake Agent wiring, startup failure and SIGTERM cleanup, invalid Wiki nodes); Wiki fixture failed before transport copy because SessionTestBase redirects common.BENCHMARK_ROOT. Evidence/task3_green.log retains this intermediate result.
- Wiki test corrections: copy the existing transport under the production source-root patch; use host-resolved Agent UID since the real sandbox masks passwd entries. No production code changes were needed. Affected test only: `...python -m pytest modes/basic_wiki-tool/tests/test_mode.py -k wiki_visible -q` → exit 0, 1 passed/3 deselected, evidence/task3_wiki_green.log.
- Actual CPU Session Workspace + real fork, real agent_command/dispatcher/Bubblewrap and ptobench-agent UID: INDEX read passed; write/unlink/file replacement/directory replacement denied. Candidate/Context hashes and protection manifest before freeze unchanged; integrity.check_only passed before and after. No model, API or NPU work was launched. No additional mounts.

## Task 4 — configuration preparation only; execution blocked

- Six required YAML files written and parsed by existing launcher.config.load_experiment; output roots resolve to project `runs/stage5` and `snapshots/stage5`, outside protected Mode paths. No Run created.
- experiment_manifest.json freezes configs, Case/env/seed file identities and intended device IDs: 2/2/8 cards (Case.runtime.device_ids is authoritative; env device=0 alone is not the device group). Same effective Pi 1.5.2, deepseek-flash/high, Provider metadata, context/max output and budget20+finish1 across both arms.
- Ruling: full Provider profile SHA differs because it includes each Mode's local env_file path; match the actual effective nonsecret model/reasoning/Provider/argv/token limits instead. Do not treat path-specific hashes as different models. Revalidate these identities immediately before real launch because public inputs may change while blocked.
- Existing Basic search: Session-level metadata under runs, pruning large artifacts. Two lm_head records found, both nonmatching (reasoning max; budget unlimited or100; profile deepseek-flash-max). No reusable exact-match record claimed. Paths/reasons in experiment_manifest.json.
- Additional coverage audit: latest public plan05 PASS proves 75 correctness Evaluate records and eight representative combined Profiles, NOT 75 completed combined Profiles. Indexed combined archive audit still covers 26 unique Cases including non-Formal fixtures; full 75-Case combined completion index is not_located. No all-profile replay or recollection performed.
- Real launch blocked by selected-Mode source protection missing from create_session/_fixed_roots/Launcher and missing full input index. No bypass, model, smoke or hardware run started. Six Session roots remain null, real Pi registered/active evidence absent, actual model calls and optimization benefit unmeasured.
- Next: after public owner supplies implementation+acceptance and full coverage index, check code identities; create approved traversable output roots via existing policy, then run only six configs serially in frozen order with PTOBench device locks. No per-step permission needed.

## Coverage continuation — 2026-11-15

- Rechecked the eight `combined_profile_evidence` references in `reports/agent_upgrade/dedicated_agent_user/plan05-continuation-20261114T064555Z-passrefs/summary.json` using only each referenced `session.json` and first `events.jsonl` record. All eight identify a completed `combined` event with PASS runtime/compile metadata; three overlap the previously located queue evidence.
- `evidence/selected_coverage_audit.json` now records the exact Session IDs, event sequence, content hashes, device groups and metadata paths. The union is 17/75 selected Cases with located completed combined-profile evidence (12 queue-index Cases + 8 public references − 3 overlap); 58 selected Cases remain without located evidence in these sources.
- This is a coverage-index audit, not a claim that the remaining Cases lack Profiles. No large profile payload was parsed, no replay/recollection was run, and no Session/model/NPU action was launched. The two launch blockers remain unchanged: selected-Mode source protection is absent in the public Session/Launcher path, and a complete 75-Case combined Profile index is not located.

## Task 5 — partial report, whole-implementation review pending

- Final report will retain null metrics for unlaunched Sessions, separate CPU tool roundtrips from actual model use, and leave overall Stage5 incomplete.
- Executing-plans explicitly requires one fresh whole-branch reviewer; dispatched one read-only reviewer on implementation commits/plan/spec/rulings. No per-task implementation/review delegation. Review is limited to this Mode; public paths are external prerequisites.

## Independent review / one fix pass — complete

- Fresh reviewer: gpt-6-astra/high, read-only whole-implementation review. Two Important findings, no Critical/Minor; full disposition in evidence/independent_review.md. No second review.
- Fixed cached Profile authority bypass: every call now revalidates Session/event/path/index/protection via fresh copied ProfileArchive under existing Session lock. Only small control metadata is read; valid parsed role caches retained. Changed archive authorization clears groups. Runtime copy untouched.
- Fixed strict JSON/nonfinite output completion: serialization runs inside guarded execution, error retains selected Profile, done/monitor cleanup unconditional. No replacement of nonfinite observations and no Analyzer semantic changes; transport inability is a typed error.
- RED: `...python -m pytest modes/basic_wiki-tool/tests/test_analyzer_adapter.py modes/basic_wiki-tool/tests/test_pi_bridge.py -k 'cached_profile_revalidates_authority or nonfinite_output' -q` → two failures (plus leaked monitor warning), evidence/review_fixes_red.log.
- GREEN: same targeted command → 2 passed/5 deselected, evidence/review_fixes_targeted_green.log.
- Affected Mode suite: `PYTHONPATH=.:evaluator:evaluator/tests /data1/home/ziruiwang/pypto-lib/.venv-official/bin/python -m pytest modes/basic_wiki-tool/tests -q` → exit0, 11 passed in2.40s, evidence/review_fixes_mode_suite_green.log. This repeat is necessary because adapter/Broker changes affect downstream Runner; public tests/large archive replay not repeated.
- Final: Ruling: JSON cannot transport Analyzer nonfinite floats as valid standard JSON — preserve the Analyzer copy and surface a typed error with identity, rather than altering numeric evidence. Cost: affected calls fail explicitly instead of returning those nonfinite observations; they remain auditable.
- No deferred Minor findings. Installed Pi1.5.2 APIs/loader statically match the bridge; real visibility remains unverified until planned Sessions.

## Handoff / completion audit

- Requirement-by-requirement audit: acceptance_audit.md. Implementation source and tests: current commit ed7565a; original Basic/Analyzer/Wiki/public source audit confirms all1,118 recorded identities unchanged. Analyzer runtime eight copied files match manifest; no full large-archive hash replay.
- Own commits so far: 32784d1 (fixed copy/adapter), 672dd57 (bridge), 53d09d3 (test import correction), 195d21d (Mode/Wiki), ed7565a (independent-review fixes). Public/input changes are preserved; no public code fix attempted.
- Six configs and partial report are reviewable. Stage5 remains **incomplete**: Task4 real Sessions absent, real Pi registered+active evidence absent, actual model Analyzer calls/Wiki reads/decision citations and optimization benefit unmeasured. summary.json uses null for all unavailable metrics and states reasons. Do not replace missing evidence with zero.
- No 75-Case rerun, extra smoke, LLM/API/NPU invocation or bypass Session launched. Generic public Pi tests reused after initial pass; last own Mode suite11 passed because production review fixes required affected regressions.
- Resume from this ledger. Do not repeat unchanged CPU checks. Revalidate selected-Mode public protection + owner acceptance and locate full combined collection index; if ready, recheck manifest pair identities and run exactly six authorized configs serially. If dependencies remain absent, inspect authoritative state and record the continuing blocker; do not declare Stage5 complete.
- Rulings are recorded above: user ledger/direct existing worktree rather than unavailable skill helpers; effective Provider matching rather than env-path-dependent profile SHA; typed serialization error rather than altering nonfinite Analyzer evidence. Their costs are explicit. No deferred minor findings.
