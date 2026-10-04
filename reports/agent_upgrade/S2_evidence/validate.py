"""Final artifact and evidence audit for S2; no Case execution or mutations."""
import ast
import hashlib
import json
from pathlib import Path
import re

repo = Path(__file__).resolve().parents[5]
bench = repo / 'benchmarks/pypto_lib_bench-v0.2'
evidence = Path(__file__).parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected = {'session_final.log': 16, 'budget_final.log': 17, 'unittest_final.log': 74}
logs = {}
for name, count in expected.items():
    content = (evidence / name).read_text()
    match = re.search(r'Ran (\d+) tests in ([0-9.]+)s\n\nOK', content)
    assert match and int(match[1]) == count, name
    assert 'FAILED (' not in content and 'skipped=' not in content and 'expected failures=' not in content
    logs[name] = {'exit_code': 0, 'tests': count, 'seconds': float(match[2]), 'sha256': digest(evidence / name)}

requirements = [
    ('1', 'Session creation/layout/schema/registry/status and direct init reuse', 'test_session.py',
     ['test_creation_independent_layout_schema_and_original_assets', 'test_creation_record_publication_survives_real_process_crash', 'test_schema_validation_even_for_a_bench_committed_bad_record']),
    ('2', 'Certification, full spec input inventory, Ref outputs/baseline and comparator; no prepare-ref', 'test_session.py',
     ['test_preflight_rejects_every_incomplete_certified_asset_without_preparing']),
    ('3', 'Protected Session identity rather than workspace metadata', 'test_session.py',
     ['test_metadata_cannot_change_identity_budget_module_or_modes', 'test_registered_roots_only_cross_session_subdirectory_and_links']),
    ('4', 'Fixed global lock, short Session lock, order, output independence', 'test_budget.py',
     ['test_lock_order_rejected_and_state_reads_need_no_global_lock', 'test_status_and_close_do_not_wait_for_active_global_measurement', 'test_two_sessions_with_different_outputs_wait_on_one_lock_without_charging']),
    ('5', 'Exploration/finish reservations, active_action and closing', 'test_budget.py',
     ['test_all_public_profiles_charge_once_and_old_modes_never_reserve', 'test_finish_is_independent_requires_closed_exploration_and_cannot_repeat', 'test_close_preserves_active_action_and_finish_waits_for_it']),
    ('6', 'No precheck charge, no accepted refund, no fork charge', 'test_budget.py',
     ['test_invalid_prechecks_do_not_charge_and_global_lock_is_required', 'test_failures_after_reservation_never_refund_and_completion_is_idempotent', 'test_close_during_prechecks_prevents_reservation_without_charge']),
    ('7', 'Concurrency, cross-Session, links, legal updates and residuals', 'test_budget.py',
     ['test_multiprocess_last_allowance_never_exceeds_limit', 'test_dead_owner_and_child_reconciled_once_without_refund', 'test_dead_unstarted_action_records_interrupted_without_refund', 'test_actual_living_orphan_process_blocks_final_measurement', 'test_real_crash_completion_recovers_one_consistent_generation']),
    ('8', 'Last accepted action completes after exhaustion/closure', 'test_budget.py',
     ['test_status_and_close_do_not_wait_for_active_global_measurement', 'test_idempotent_old_completion_preserves_the_next_active_action']),
]
full = (evidence / 'unittest_final.log').read_text()
audit = []
for number, text, source, tests in requirements:
    tree = ast.parse((bench / 'evaluator/tests' / source).read_text())
    present = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert set(tests) <= present
    assert all(re.search(re.escape(name) + r' \(', full) for name in tests)
    audit.append({'requirement': number, 'scope': text, 'status': 'verified', 'test_source': source, 'tests': tests})

source = bench / 'evaluator/session.py'
tree = ast.parse(source.read_text())
functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
assert {'create_session', 'read_session', 'register_workspace', 'reserve_action', 'close_exploration'} <= functions
assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and any(name.name in {'subprocess', 'bench'} for name in node.names) for node in ast.walk(tree))
assert (bench / 'bench').is_file() and (bench / 'evaluator/bench.py').is_file()
audit.append({'requirement': '9', 'scope': 'No old CLI Session routing/search/alias/fallback; old entry retained for S3',
              'status': 'verified', 'evidence': 'Original public and old CLI hashes unchanged; new module imports no bench/subprocess and takes explicit Session roots.'})

preservation = json.loads((evidence / 'preservation.json').read_text())
assert preservation['index_unchanged']
assert not preservation['canonical_changed_paths'] and not preservation['tracked_changed_paths'] and not preservation['previous_public_and_history_changed_paths']
report = bench / 'reports/agent_upgrade/S2_session_and_budget.md'
links = re.findall(r'\]\(([^)]+)\)', report.read_text())
for link in links:
    assert (report.parent / link).exists() or link == 'S2_evidence/validation_summary.json', link
source_files = [source, bench / 'evaluator/tests/test_session.py', bench / 'evaluator/tests/test_budget.py',
                bench / 'evaluator/tests/session_test_support.py', bench / 'evaluator/SESSION.md']
result = {'stage': 'S2', 'status': 'PASS', 'unit_tests': logs,
          'source_sha256': {str(path.relative_to(repo)): digest(path) for path in source_files},
          'requirement_audit': audit, 'preservation': preservation, 'report_sha256': digest(report),
          'report_links_verified': True, 'S2_only': True, 'S3_preconditions_met': True,
          'limits': ['Only fake Cases and processes used for S2 runtime tests.',
                     'Fixture preflight checks metadata/file completeness and existing certification, not fresh tensor correctness.',
                     'No real Optimizer/hardware/Evaluator/Snapshot/finalize implementation or acceptance.',
                     'S0 real Case asset blockers retained; tensor preservation checked by metadata, not byte hashes.']}
(evidence / 'validation_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print('PASS: 74 CPU tests, all nine requirements mapped to current implementation/behavior evidence, preservation and report links verified.')
