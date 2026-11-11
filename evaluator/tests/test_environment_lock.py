import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCH), str(BENCH / "evaluator")]

import common  # noqa: E402
import environment_lock  # noqa: E402


class EnvironmentLockTest(unittest.TestCase):
    def _locked_python(self):
        resolver = getattr(environment_lock, "locked_python_executable", None)
        self.assertTrue(callable(resolver), "locked Python resolver is missing")
        return resolver()

    def test_locked_python_executable_resolves_absolute_lock_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "python"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            link = root / "python-link"
            link.symlink_to(executable)
            lock = root / "LOCK.json"
            lock.write_text(json.dumps({"python": {"executable": str(link)}}))

            with patch.object(environment_lock, "LOCK_PATH", lock):
                actual = self._locked_python()

            self.assertEqual(actual, executable.resolve())
            self.assertTrue(actual.is_absolute())

    def test_locked_python_executable_rejects_missing_or_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "LOCK.json"
            for python in ({}, {"executable": "relative/python"},
                           {"executable": str(root / "missing")}):
                with self.subTest(python=python):
                    lock.write_text(json.dumps({"python": python}))
                    with patch.object(environment_lock, "LOCK_PATH", lock):
                        with self.assertRaises(environment_lock.EnvironmentMismatch):
                            self._locked_python()

    def test_locked_python_executable_rejects_non_executable_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "python"
            executable.write_text("not executable")
            executable.chmod(0o644)
            lock = Path(temporary) / "LOCK.json"
            lock.write_text(json.dumps({"python": {"executable": str(executable)}}))

            with patch.object(environment_lock, "LOCK_PATH", lock):
                with self.assertRaises(environment_lock.EnvironmentMismatch):
                    self._locked_python()

    def test_current_official_profile_matches_lock(self):
        case = common.read_json(BENCH / "cases/deepseek_v4/combine_flash/case.json")
        profile = common._environment_profile(case)
        exports = common.load_profile_environment(profile)
        environment_lock.validate(profile, exports)

    def test_wrong_python_is_rejected_before_execution(self):
        case = common.read_json(BENCH / "cases/deepseek_v4/combine_flash/case.json")
        profile = common._environment_profile(case)
        exports = common.load_profile_environment(profile)
        with patch.object(environment_lock.sys, "executable", "/usr/bin/python3"):
            with self.assertRaisesRegex(environment_lock.EnvironmentMismatch, "Python mismatch"):
                environment_lock.validate(profile, exports)

    def test_isolated_worker_reuses_trusted_parent_environment_check(self):
        case_id = "deepseek_v4.combine_flash"
        profile = {"name": "fixture", "python": sys.executable}
        marker = f"{case_id}:{profile['name']}:cxx-runtime-v1"
        with patch.dict("os.environ", {"PTOBENCH_WORKER": "1", "PYPTO_BENCH_ENV": marker}):
            with patch.object(common, "_environment_profile", return_value=profile), \
                    patch.object(common, "profile_path", return_value=Path(sys.executable)), \
                    patch.object(common, "validate_toolchain"), \
                    patch.object(environment_lock, "validate", side_effect=AssertionError("revalidated")):
                common.bootstrap_environment({"case_id": case_id})


if __name__ == "__main__":
    unittest.main()
