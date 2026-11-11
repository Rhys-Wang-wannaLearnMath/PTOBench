from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_modes_test", EVALUATOR_DIR / "profile_modes.py"
)
assert SPEC is not None and SPEC.loader is not None
profile_modes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_modes)


class ProfilePhaseTest(unittest.TestCase):
    def test_combined_is_the_default_and_split_remains_explicit(self):
        self.assertEqual(profile_modes.normalize_strategy(None), 'combined')
        self.assertEqual(list(profile_modes.public_enabled_modes()), ['combined'])
        self.assertEqual(list(profile_modes.public_enabled_modes('split')),
                         ['runtime', 'pmu', 'compiler'])

    def test_only_current_modes_are_enabled(self):
        self.assertEqual(
            profile_modes.ENABLED_PROFILE_MODES,
            ("runtime", "hardware", "compile"),
        )
        self.assertEqual(
            profile_modes.PUBLIC_TO_INTERNAL,
            {"runtime": "runtime", "pmu": "hardware", "compiler": "compile"},
        )

    def test_hardware_defaults_to_one_shot_correctness(self):
        self.assertEqual(profile_modes.phase("hardware"), "correctness")

    def test_all_modes_default_to_one_shot_correctness(self):
        self.assertEqual(profile_modes.phase("runtime"), "correctness")
        self.assertEqual(profile_modes.phase("compile"), "correctness")
        with self.assertRaises(ValueError):
            profile_modes.settings("incore")

    def test_combined_profile_collects_all_three_configurations(self):
        combined = profile_modes.settings("combined")
        for mode in profile_modes.ENABLED_PROFILE_MODES:
            self.assertEqual(
                {key: combined[key] for key in profile_modes.settings(mode)},
                profile_modes.settings(mode),
            )
        self.assertEqual(profile_modes.phase("combined"), "correctness")
        self.assertEqual(
            list(profile_modes.public_enabled_modes("combined")), ["combined"]
        )

    def test_runtime_capture_checks_files_without_interpreting_profile_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / 'rank0/d0'
            capture.mkdir(parents=True)
            (capture / 'chip_swimlane_records.json').write_text('{"aicore_tasks": []}')
            (capture / 'deps.json').write_text('{"tasks": [{"kernel_ids": [-1, -1, -1]}]}')

            result = profile_modes.validate_artifacts(temporary, 'runtime')

        self.assertEqual(result['status'], 'artifacts_present')
        self.assertEqual(len(result['artifacts']), 2)

    def test_hardware_capture_does_not_require_a_task_row_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / 'rank0/d0'
            capture.mkdir(parents=True)
            (capture / 'pmu.csv').write_text('runtime-specific PMU columns')

            result = profile_modes.validate_artifacts(temporary, 'hardware')

        self.assertEqual(result['status'], 'artifacts_present')
        self.assertEqual(result['artifacts'], ['rank0/d0/pmu.csv'])

    def test_missing_requested_artifact_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'pmu.csv missing'):
                profile_modes.validate_artifacts(temporary, 'hardware')

if __name__ == "__main__":
    unittest.main()
