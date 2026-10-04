from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bench_common_test", EVALUATOR_DIR / "common.py")
assert SPEC is not None and SPEC.loader is not None
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


class RoleLoadingTest(unittest.TestCase):
    def test_official_fixture_isolates_siblings_through_delayed_imports(self):
        """Both import-time and builder-time helpers must use the selected preset."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {}
            for preset in ("demo", "flash"):
                directory = root / preset
                directory.mkdir()
                (directory / "config.py").write_text(
                    f"PRESET = {preset!r}\n", encoding="utf-8",
                )
                (directory / "fixture_eager.py").write_text(
                    "from config import PRESET\n", encoding="utf-8",
                )
                (directory / "fixture_late.py").write_text(
                    "from config import PRESET\n", encoding="utf-8",
                )
                (directory / "original.py").write_text(
                    "import fixture_eager\n"
                    "def build_tensor_specs():\n"
                    "    import fixture_late\n"
                    "    return [fixture_eager.PRESET, fixture_late.PRESET]\n",
                    encoding="utf-8",
                )
                roots[preset] = directory

            # Preserve other users of the same names rather than deleting them.
            previous = {
                name: types.ModuleType(name)
                for name in ("config", "fixture_eager", "fixture_late")
            }
            for module in previous.values():
                module.PRESET = "outside"
            with patch.dict(sys.modules, previous):
                for order in (("demo", "flash"), ("flash", "demo")):
                    for preset in order:
                        alias = types.ModuleType("config")
                        alias.PRESET = preset
                        for selected_alias in (alias, None):
                            specs = common.load_official_fixture_specs(
                                roots[preset] / "original.py", "official_probe", 0,
                                config_alias=selected_alias,
                            )
                            self.assertEqual(specs, [preset, preset])
                            for name, module in previous.items():
                                self.assertIs(sys.modules[name], module)
                        self.assertFalse((roots[preset] / "__pycache__").exists())

    def test_official_fixture_restores_process_state_after_failure(self):
        import golden

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fixture_helper.py").write_text(
                "from config import PRESET\n", encoding="utf-8",
            )
            old_helper = types.ModuleType("fixture_helper")
            old_config = types.ModuleType("config")
            alias = types.ModuleType("config")
            alias.PRESET = "demo"
            for failure in (
                "import fixture_helper\nraise RuntimeError('import failed')\n",
                "def build_tensor_specs():\n"
                "    import fixture_helper\n"
                "    raise RuntimeError('build failed')\n",
            ):
                (root / "original.py").write_text(failure, encoding="utf-8")
                before_path = sys.path[:]
                before_bytecode = sys.dont_write_bytecode
                before_spec = golden.TensorSpec
                with patch.dict(sys.modules, {
                    "config": old_config, "fixture_helper": old_helper,
                }):
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        common.load_official_fixture_specs(
                            root / "original.py", "failed_probe", 0,
                            config_alias=alias,
                        )
                    self.assertIs(sys.modules["config"], old_config)
                    self.assertIs(sys.modules["fixture_helper"], old_helper)
                    self.assertIs(golden.TensorSpec, before_spec)
                    self.assertEqual(sys.path, before_path)
                    self.assertEqual(sys.dont_write_bytecode, before_bytecode)

    def test_official_fixture_adapter_strips_historical_is_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "official.py").write_text(
                "from golden import TensorSpec\n"
                "def build_tensor_specs():\n"
                "    return [TensorSpec('out', [1], None, is_output=True)]\n",
                encoding="utf-8",
            )
            golden = type(sys)("golden")

            class TensorSpec:
                def __init__(self, name, shape, dtype):
                    self.name, self.shape, self.dtype = name, shape, dtype

            golden.TensorSpec = TensorSpec
            torch = type(sys)("torch")
            torch.manual_seed = lambda seed: None
            previous = sys.modules.get("golden")
            previous_torch = sys.modules.get("torch")
            sys.modules["golden"] = golden
            sys.modules["torch"] = torch
            try:
                specs = common.load_official_fixture_specs(
                    root / "official.py", "test_official", 0
                )
                self.assertEqual(specs[0].name, "out")
                self.assertIs(golden.TensorSpec, TensorSpec)
                self.assertFalse((root / "__pycache__").exists())
            finally:
                if previous is None:
                    sys.modules.pop("golden", None)
                else:
                    sys.modules["golden"] = previous
                if previous_torch is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = previous_torch

    def test_oracle_does_not_reuse_or_replace_ref_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "case"
            sealed = root / "sealed"
            (case_dir / "public").mkdir(parents=True)
            (sealed / "ref").mkdir(parents=True)
            (sealed / "oracle").mkdir(parents=True)

            (sealed / "ref/helpers.py").write_text("VALUE = 'ref'\n", encoding="utf-8")
            (sealed / "ref/ref_npu.py").write_text(
                "import helpers\n\ndef value():\n    return helpers.VALUE\n",
                encoding="utf-8",
            )
            (sealed / "oracle/helpers.py").write_text("VALUE = 'oracle'\n", encoding="utf-8")
            (sealed / "oracle/ref_cpu.py").write_text(
                "import helpers\n\ndef value():\n    return helpers.VALUE\n",
                encoding="utf-8",
            )

            previous = sys.modules.pop("helpers", None)
            try:
                ref = common.load_ref(case_dir, sealed, "test_ref")
                self.assertNotIn("helpers", sys.modules)
                oracle = common.load_oracle(case_dir, sealed, "test_oracle")

                self.assertEqual(ref.value(), "ref")
                self.assertEqual(oracle.value(), "oracle")
                self.assertNotIn("helpers", sys.modules)
                self.assertFalse((sealed / "ref/__pycache__").exists())
                self.assertFalse((sealed / "oracle/__pycache__").exists())
            finally:
                sys.modules.pop("helpers", None)
                if previous is not None:
                    sys.modules["helpers"] = previous


if __name__ == "__main__":
    unittest.main()
