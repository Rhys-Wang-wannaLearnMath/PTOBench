import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCH), str(BENCH / "evaluator")]

import common  # noqa: E402
import environment_lock  # noqa: E402


class EnvironmentLockTest(unittest.TestCase):
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
