"""Agent guidance and measured-default publication; no compilation or NPU use."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import profile_modes
import tuning
import verify_tuning
import workspace


class AgentTuningTests(unittest.TestCase):
    def test_all_cases_keep_measured_identity_and_describe_only_reviewed_options(self):
        ledger = common.read_json(common.BENCHMARK_ROOT / 'reports/tuning/case_defaults_verification.json')
        self.assertEqual(ledger['queue_status'], 'complete')
        rows = {row['case_id']: row for row in ledger['items']}
        paths = common.formal_case_configs()
        self.assertEqual(len(paths), 311)
        self.assertEqual(set(rows), {common.read_json(path)['case_id'] for path in paths})
        for path in paths:
            case = common.read_json(path)
            with self.subTest(case=case['case_id']):
                _, sealed, _ = common.case_paths(case['case_id'])
                baseline = common.read_json(sealed / 'ref_performance.json')
                identity = verify_tuning.identity(path.parent, case)
                row = rows[case['case_id']]
                self.assertEqual(row['status'], 'PASS')
                self.assertTrue(row['baseline_published'])
                if 'tuning_identity' in baseline:
                    self.assertEqual(baseline['tuning_identity'], row['identity'])
                identity_matches = identity == row['identity']
                profile = common._environment_profile(case)
                static = tuning.capabilities(case, profile)
                caps = tuning.agent_capabilities(case, profile, baseline, identity)
                self.assertEqual(caps['baseline'], row['tuning'])
                self.assertEqual(caps['hardware_tested_in_this_change'], identity_matches)
                status = caps['default_configuration_validation']
                expected_status = 'passed' if identity_matches else 'unverified'
                self.assertEqual(status['status'], expected_status)
                self.assertEqual(status['correctness'], expected_status)
                self.assertEqual(status['clean_performance_baseline'], expected_status)
                self.assertFalse(status['other_values_verified'])
                for name, original in static['options'].items():
                    option = caps['options'][name]
                    for field, value in original.items():
                        self.assertEqual(option[field], value)
                    if option['editable']:
                        for field in ('type', 'unit', 'description', 'restrictions'):
                            self.assertTrue(option[field], (name, field))
                        if name in ('ring_task_window', 'aicpu_thread_num'):
                            self.assertEqual(option['stage'], 'dispatch')
                guide = workspace._readme(case, profile_modes.public_enabled_modes(), capabilities=caps)
                self.assertIn(f'**{expected_status}**', guide)
                for name in caps['baseline']:
                    self.assertIn(f'| `{name}` |', guide)
                serialized = json.dumps(caps)
                for private in ('sealed_assets/', 'runs/', 'tuning_evidence', 'tuning_identity'):
                    self.assertNotIn(private, serialized)
                # Agent guidance must not mutate the stable permissions identity.
                self.assertEqual(static, tuning.capabilities(case, profile))

    def test_legacy_metadata_is_optional_but_bad_measurements_cannot_claim_verified_defaults(self):
        case_dir, sealed, case = common.case_paths('deepseek_v4.hc_post_demo')
        baseline = common.read_json(sealed / 'ref_performance.json')
        identity = verify_tuning.identity(case_dir, case)
        fixture = case['fixture_ids'][0]
        changes = {
            'pending': lambda b: b.update(status='pending'),
            'different_settings': lambda b: b['compiler_tuning'].update(memory_planner='PTOAS'),
            'incomplete_defaults': lambda b: b['compiler_tuning'].pop('ring_task_window'),
            'fixture_missing': lambda b: b['fixtures'].clear(),
            'short_samples': lambda b: b['fixtures'][fixture]['samples_effective_us'].pop(),
            'nan_sample': lambda b: b['fixtures'][fixture]['samples_effective_us'].__setitem__(0, float('nan')),
            'zero_sample': lambda b: b['fixtures'][fixture]['samples_effective_us'].__setitem__(0, 0),
        }
        for name, change in changes.items():
            with self.subTest(defect=name):
                changed = copy.deepcopy(baseline)
                change(changed)
                caps = tuning.agent_capabilities(case, {}, changed, identity)
                self.assertFalse(caps['hardware_tested_in_this_change'])
                self.assertEqual(caps['default_configuration_validation']['status'], 'unverified')
        for name, change in {
            'identity_missing': lambda b: b.pop('tuning_identity', None),
            'requested_missing': lambda b: b.pop('resolved_tuning', None),
            'evidence_missing': lambda b: b.pop('tuning_evidence', None),
        }.items():
            with self.subTest(optional_metadata=name):
                changed = copy.deepcopy(baseline)
                change(changed)
                caps = tuning.agent_capabilities(case, {}, changed, identity)
                self.assertTrue(caps['hardware_tested_in_this_change'])
        changed = copy.deepcopy(baseline)
        changed.setdefault('tuning_identity', {}).update(source_sha256='changed')
        caps = tuning.agent_capabilities(case, {}, changed, identity)
        self.assertFalse(caps['hardware_tested_in_this_change'])
        self.assertEqual(caps['default_configuration_validation']['status'], 'unverified')
        self.assertEqual(tuning.agent_capabilities(case, {}, baseline)
                         ['default_configuration_validation']['status'], 'passed')

    def test_initial_workspaces_and_forks_for_every_option_set(self):
        representatives = {}
        for path in common.formal_case_configs():
            case = common.read_json(path)
            representatives.setdefault(tuple(sorted(tuning.defaults(case))), case['case_id'])
        self.assertEqual(len(representatives), 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number, identifier in enumerate(representatives.values()):
                with self.subTest(case=identifier):
                    case_dir, sealed, case = common.case_paths(identifier)
                    parent, child = root / f'parent_{number}', root / f'child_{number}'
                    workspace.init_workspace(case_dir, sealed, case, parent)
                    self.assertEqual(common.read_tuning(parent, case), tuning.defaults(case))
                    chosen = {name: spec['values'][-1] for name, spec in
                              tuning.policy(case)['options'].items() if spec['editable']}
                    (parent / 'candidate' / 'tuning.json').write_text(json.dumps(chosen))
                    workspace.fork_workspace(parent, child)
                    self.assertEqual(common.read_tuning(child, case), chosen)
                    for target in (parent, child):
                        context = target / '.benchmark_context'
                        caps = common.read_json(context / 'tuning_capabilities.json')
                        reference = common.read_json(context / 'reference.json')
                        self.assertEqual(caps['default_configuration_validation'],
                                         reference['default_configuration_validation'])
                        self.assertEqual(caps['default_configuration_validation']['status'], 'passed')
                        self.assertEqual(caps['baseline'], tuning.defaults(case))
                        self.assertEqual((context / 'tuning_capabilities.json').stat().st_mode & 0o222, 0)


if __name__ == '__main__':
    unittest.main()
