from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATOR_DIR))
SPEC = importlib.util.spec_from_file_location("bench_evaluate_test", EVALUATOR_DIR / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class MeasurementPolicyTest(unittest.TestCase):
    def test_exploration_and_formal_settings_reach_runtime_and_raw_records(self):
        import common
        import workflow

        profile = {"platform": "a2a3", "device": 0,
                   "benchmark": {"warmup": 5, "rounds": 10}}
        for mode, case_settings, expected in (
            ("exploration", {"warmup": 9, "rounds": 20}, (2, 3)),
            (None, None, (5, 10)),
            ("formal", {"warmup": 6, "rounds": 12}, (6, 12)),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                candidate = root / "candidate"
                candidate.mkdir()
                sealed = root / "sealed"
                sealed.mkdir()
                case = {"case_id": "test.case", "workspace": "candidate",
                        "fixture_ids": ["fixture_000"]}
                if case_settings is not None:
                    case["benchmark"] = case_settings
                common.write_json(sealed / "certification.json", {
                    "status": "ready", "comparator_sha256": "test-comparator"})
                common.write_json(sealed / "ref_performance.json", {"fixtures": {
                    "fixture_000": {"median_effective_us": 20.0, "mean_effective_us": 20.0}}})

                def run(**_kwargs):
                    warmup = int(os.environ["PYPTO_BENCH_WARMUP"])
                    rounds = int(os.environ["PYPTO_BENCH_ROUNDS"])
                    self.assertEqual((warmup, rounds), expected)
                    self.assertEqual(os.environ["PYPTO_BENCH"], "1")
                    bench = SimpleNamespace(warmup=warmup, rounds=rounds,
                                            per_round=lambda _metric: [10.0 + i for i in range(rounds)])
                    return SimpleNamespace(passed=True, error=None, work_dir=None, bench=bench)

                def adapter(case_dir, assets, runtime_case, workspace, platform, phase):
                    return workflow.evaluate_candidate(
                        case_dir=case_dir, sealed=assets, case=runtime_case,
                        candidate_workspace=workspace, platform=platform, phase=phase,
                        build_specs=lambda _fixture: [], load_candidate=lambda: None,
                        make_program=lambda _module, _case: None,
                        make_compare=lambda: None, run=run,
                    )

                result_path = root / "result.json"
                argv = ["evaluate.py", "--case", "test.case", "--candidate", str(candidate),
                        "--result", str(result_path)]
                if mode is not None:
                    argv.extend(["--measurement-mode", mode])
                original_cwd = Path.cwd()
                try:
                    with patch.dict(os.environ, {"PYPTO_BENCH_WARMUP": "99", "PYPTO_BENCH_ROUNDS": "99"}), \
                            patch.object(sys, "argv", argv), \
                            patch.object(common, "await_action_start"), \
                            patch.object(common, "resolve_case", return_value=(root, sealed, case)), \
                            patch.object(common, "bootstrap_environment"), \
                            patch.object(common, "_environment_profile", return_value=profile), \
                            patch.object(common, "environment", return_value=({}, None)), \
                            patch.object(common, "comparator_sha256", return_value="test-comparator"), \
                            patch.object(common, "load_case_evaluator", return_value=SimpleNamespace(evaluate_candidate=adapter)), \
                            patch.object(evaluate.execution_record, "observe_runtime", return_value=contextlib.nullcontext()):
                        self.assertEqual(evaluate.main(), 0)
                finally:
                    os.chdir(original_cwd)
                recorded = common.read_json(result_path)["runs"][0]
                self.assertEqual(
                    common.read_json(result_path)["effective_tuning"],
                    common.TUNING_DEFAULTS,
                )
                config = recorded["measurement_config"]
                self.assertEqual((config["requested_warmup"], config["requested_rounds"]), expected)
                self.assertEqual((config["actual_warmup"], config["actual_rounds"]), expected)
                self.assertEqual(recorded["raw_samples_effective_us"], [10.0 + i for i in range(expected[1])])
                self.assertEqual(recorded["median_effective_us"], 10.0 + (expected[1] - 1) / 2)
                self.assertEqual(case.get("benchmark"), case_settings)


class RunIsolationTest(unittest.TestCase):
    def test_agent_profile_summary_omits_correctness_and_preserves_execution_failure(self):
        import common
        import golden.runner as runner
        original = runner._compute_golden
        for mode in ('runtime', 'hardware', 'compile'):
            for passed in (True, False):
                with self.subTest(mode=mode, passed=passed), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    candidate = root / 'candidate'
                    candidate.mkdir()
                    case = {'case_id': 'test.case', 'workspace': 'candidate'}
                    result_path = root / 'result.json'

                    def adapter(*_args):
                        self.assertIsNone(runner._compute_golden())
                        return [{'fixture_id': 'fixture_000', 'passed': passed,
                                 'error': '' if passed else 'runtime failed'}]

                    argv = ['evaluate.py', '--case', 'test.case', '--candidate', str(candidate),
                            '--profile-mode', mode, '--phase', 'correctness',
                            '--skip-correctness', '--result', str(result_path)]
                    original_cwd = Path.cwd()
                    try:
                        with patch.dict(os.environ), patch.object(sys, 'argv', argv), \
                                patch.object(common, 'await_action_start'), \
                                patch.object(common, 'resolve_case', return_value=(root, root, case)), \
                                patch.object(common, 'bootstrap_environment'), \
                                patch.object(common, '_environment_profile', return_value={'platform': 'cpu'}), \
                                patch.object(common, 'load_case_evaluator', return_value=SimpleNamespace(evaluate_candidate=adapter)):
                            self.assertEqual(evaluate.main(), 0 if passed else 1)
                    finally:
                        os.chdir(original_cwd)
                    result = common.read_json(result_path)
                    self.assertNotIn('correctness', result)
                    self.assertEqual(result['verdict'], 'PASS' if passed else 'FAIL')
                    self.assertTrue(result['diagnostic_only'])
                    self.assertFalse(result['official_score'])
                    self.assertIs(runner._compute_golden, original)

    def test_skip_correctness_cannot_disable_clean_or_performance_validation(self):
        import io
        for flags in ([], ['--profile-mode', 'runtime', '--phase', 'all'],
                      ['--profile-mode', 'hardware', '--phase', 'performance']):
            with self.subTest(flags=flags), \
                    patch.object(sys, 'argv', ['evaluate.py', '--case', 'test.case',
                                              '--skip-correctness', *flags]), \
                    patch.object(evaluate.common, 'await_action_start'), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                evaluate.main()
            self.assertEqual(error.exception.code, 2)

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
