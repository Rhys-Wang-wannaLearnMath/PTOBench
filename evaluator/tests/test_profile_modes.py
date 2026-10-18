from __future__ import annotations

import importlib.util
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

if __name__ == "__main__":
    unittest.main()
