from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATOR_DIR))
SPEC = importlib.util.spec_from_file_location("bench_evaluate_test", EVALUATOR_DIR / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class RunIsolationTest(unittest.TestCase):
    def test_workspaces_and_modes_get_distinct_artifact_roots(self):
        original_cwd = Path.cwd()
        previous_build = os.environ.get("PYPTO_PROG_BUILD_DIR")
        previous_artifacts = os.environ.get("PYPTO_BENCH_ARTIFACT_DIR")
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                first = root / "first"
                second = root / "second"
                first.mkdir()
                second.mkdir()

                first_artifacts, first_result = evaluate.prepare_run(first, "clean", None)
                second_artifacts, second_result = evaluate.prepare_run(second, "runtime", None)

                self.assertEqual(first_artifacts, first / ".benchmark_outputs/clean")
                self.assertEqual(first_result, first_artifacts / "latest_result.json")
                self.assertEqual(second_artifacts, second / ".benchmark_outputs/runtime")
                self.assertEqual(second_result, second_artifacts / "latest_result.json")
                self.assertNotEqual(first_artifacts, second_artifacts)
                self.assertEqual(Path.cwd(), second)
                self.assertEqual(
                    os.environ["PYPTO_PROG_BUILD_DIR"],
                    str(second_artifacts / "build_output"),
                )
                custom_result = root / "custom.json"
                _custom_artifacts, selected_result = evaluate.prepare_run(
                    first, "clean", custom_result,
                )
                self.assertEqual(selected_result, custom_result.resolve())
        finally:
            os.chdir(original_cwd)
            if previous_build is None:
                os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
            else:
                os.environ["PYPTO_PROG_BUILD_DIR"] = previous_build
            if previous_artifacts is None:
                os.environ.pop("PYPTO_BENCH_ARTIFACT_DIR", None)
            else:
                os.environ["PYPTO_BENCH_ARTIFACT_DIR"] = previous_artifacts


if __name__ == "__main__":
    unittest.main()
