from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATOR_DIR))
import common
import prepare_ref
import workflow


class CertificationBindingTest(unittest.TestCase):
    def test_candidate_rejects_stale_comparator_before_using_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "case"
            sealed = root / "sealed"
            (case_dir / "public").mkdir(parents=True)
            sealed.mkdir()
            (case_dir / "public/comparator.json").write_text(
                '{"schema_version":1,"outputs":{"out":{"method":"exact"}}}\n',
                encoding="utf-8",
            )
            common.write_json(
                sealed / "certification.json",
                {
                    "schema_version": 1,
                    "status": "ready",
                    "case_id": "test.case",
                    "comparator_sha256": "stale",
                },
            )
            case = {
                "case_id": "test.case",
                "comparator": "public/comparator.json",
                "correctness": {"outputs": ["out"]},
            }

            with self.assertRaisesRegex(RuntimeError, "does not match public comparator"):
                workflow.evaluate_candidate(
                    case_dir=case_dir,
                    sealed=sealed,
                    case=case,
                    candidate_workspace=root / "workspace",
                    platform=None,
                    phase="correctness",
                    build_specs=lambda _fixture: self.fail("cache was used"),
                    load_candidate=lambda: self.fail("candidate was loaded"),
                    make_program=lambda _module, _case: self.fail("program was built"),
                    make_compare=lambda: self.fail("comparator was built"),
                    run=lambda **_kwargs: self.fail("candidate was run"),
                )


class FixtureReuseTest(unittest.TestCase):
    def test_complete_fixture_inputs_are_reusable_for_recertification(self):
        with tempfile.TemporaryDirectory() as temporary:
            sealed = Path(temporary)
            fixture = sealed / "fixtures/fixture_000"
            (fixture / "in").mkdir(parents=True)
            (fixture / "manifest.json").write_text("{}\n", encoding="utf-8")

            self.assertTrue(
                prepare_ref.fixtures_ready(
                    sealed, {"fixture_ids": ["fixture_000"]}
                )
            )
            (fixture / "manifest.json").unlink()
            self.assertFalse(
                prepare_ref.fixtures_ready(
                    sealed, {"fixture_ids": ["fixture_000"]}
                )
            )


if __name__ == "__main__":
    unittest.main()
