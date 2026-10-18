"""Compiler loading fails before launching an Agent or querying its NPU."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCH), str(BENCH / "evaluator")]
from launcher import preflight
from launcher.methods import MethodError


class RuntimePreflightTest(unittest.TestCase):
    def test_compiler_loading_failure_precedes_agent_and_hardware_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ptobench").write_text("#!/bin/sh\n")
            (root / "ptobench").chmod(0o755)
            experiment = SimpleNamespace(case="model.case", mode="basic")
            profile = {"platform": "a2a3", "python": sys.executable, "device": 0}
            exports = {"LD_LIBRARY_PATH": "/selected/runtime"}
            with patch.object(preflight.common, "BENCHMARK_ROOT", root), \
                    patch.object(preflight, "load_experiment", return_value=experiment), \
                    patch.object(preflight, "load_method"), \
                    patch.object(preflight.session, "preflight_case", return_value=(None, None, {}, profile)), \
                    patch.object(preflight.common, "load_profile_environment", return_value=exports), \
                    patch.object(preflight.common, "validate_toolchain", side_effect=RuntimeError("GLIBCXX missing")) as check, \
                    patch.object(preflight.subprocess, "run") as hardware_query, \
                    patch.object(preflight, "call") as agent_preflight:
                with self.assertRaisesRegex(MethodError, "compiler environment.*GLIBCXX"):
                    preflight.preflight(root / "experiment.yaml")
                check.assert_called_once_with(profile, exports)
                hardware_query.assert_not_called()
                agent_preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
