# Independent implementation review

Reviewer: fresh-context `stage5_review`, model gpt-6-astra, high reasoning, invoked because executing-plans explicitly requires one whole-branch review. Read-only review against the full plan, protocol, implementation commits and progress rulings; no tests run by reviewer. Scope: Mode implementation, not blocked external infrastructure/trials.

Result: two Important findings, no Critical or Minor findings.

1. Warm Profile cache bypassed current path/index/manifest authorization when ProfileRef identity stayed unchanged. Fix: construct fresh copied `ProfileArchive` under the existing Session lock each call; compare artifact root/index/protected entries, invalidate parsed groups when authorization changes; retain valid parsed caches. No heavyweight diagnostic parse added to the authorization check.
2. Nonfinite Analyzer output could make strict JSON encoding fail outside the completion guard, leaving no terminal record and a waiting monitor thread. Fix: encode output inside guarded completion, retain selected Profile in a typed error, always set done and join monitor in cleanup. The source Analyzer intentionally supports nonfinite observations; its copy remains unchanged, and such output is now an explicit transport error rather than a fabricated successful result.

RED: review_fixes_red.log — both new regression tests failed. The serialization test also exposed the waiting monitor's negative-file-descriptor warning.
GREEN: review_fixes_targeted_green.log — both passed.
Affected Mode suite: review_fixes_mode_suite_green.log — 11 passed in 2.40s. Adapter/Broker changes affect the Mode lifecycle, so the Mode's own CPU suite was the regression scope. Public suites, 75-Case data, Stage4 replay and real Agent sessions were not rerun.

Tests: `test_cached_profile_revalidates_authority` warms cache and changes same-event artifact/index paths and protection; `test_nonfinite_output_has_terminal_typed_error_and_profile` checks socket error, durable terminal, selected identity, stable collect finalization and normal shutdown.

The reviewer also statically confirmed installed Pi1.5.2 execute/isError/schema/registration/active APIs and Jiti loader match the extension. This is compatibility evidence, not real Pi visibility or LLM behavior evidence. No re-review was dispatched after the single fix pass; regression tests are the fix gate.
