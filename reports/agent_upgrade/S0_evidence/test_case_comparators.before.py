import json
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import torch


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = EVALUATOR_DIR.parent
sys.path.insert(0, str(EVALUATOR_DIR))
import common


class CaseComparatorTest(unittest.TestCase):
    def test_every_case_owns_and_executes_its_comparator(self):
        self.assertFalse(hasattr(common, "make_comparators"))
        case_paths = sorted((BENCHMARK_ROOT / "cases").glob("*/*/case.json"))
        self.assertTrue(case_paths)

        for index, case_path in enumerate(case_paths):
            case_dir = case_path.parent
            case = json.loads(case_path.read_text(encoding="utf-8"))
            family, name = case["case_id"].split(".", 1)
            sealed = BENCHMARK_ROOT / "sealed_assets" / family / name
            evaluator = common.load_module(
                sealed / "evaluate.py", f"case_comparator_test_{index}"
            )
            source = (sealed / "evaluate.py").read_text(encoding="utf-8")
            self.assertNotIn("common.make_comparators", source)

            comparators = evaluator._compare(case_dir, case)
            self.assertEqual(set(comparators), set(case["correctness"]["outputs"]))
            policy = json.loads(
                (case_dir / case["comparator"]).read_text(encoding="utf-8")
            )
            for name, compare in comparators.items():
                signature = inspect.signature(compare)
                self.assertNotIn("rtol", signature.parameters)
                self.assertNotIn("atol", signature.parameters)
                if policy["outputs"][name]["method"] == "custom":
                    continue
                expected = torch.tensor([1.0, 2.0])
                runner_kwargs = {
                    "actual_outputs": {name: expected},
                    "expected_outputs": {name: expected},
                    "inputs": {},
                    "rtol": 0.0,
                    "atol": 0.0,
                }
                self.assertTrue(
                    compare(expected.clone(), expected, **runner_kwargs)[0]
                )
                self.assertFalse(
                    compare(
                        torch.zeros(2), torch.full((2,), 100.0),
                        **runner_kwargs,
                    )[0]
                )

    def test_declared_methods_and_strict_schema(self):
        qwen_case_path = (
            BENCHMARK_ROOT
            / "cases/qwen3_14b/qwen3_14b_decode/case.json"
        )
        qwen_case = json.loads(qwen_case_path.read_text(encoding="utf-8"))
        qwen_evaluator = common.load_case_evaluator(
            BENCHMARK_ROOT / "sealed_assets/qwen3_14b/qwen3_14b_decode"
        )
        ratio = qwen_evaluator._compare(
            qwen_case_path.parent, qwen_case,
        )["out"]
        expected = torch.zeros(100)
        actual = expected.clone()
        actual[:2] = 1
        self.assertTrue(ratio(actual, expected)[0])
        actual[2] = 1
        self.assertFalse(ratio(actual, expected)[0])
        actual[0] = torch.nan
        self.assertFalse(ratio(actual, expected)[0])

        greedy_case_path = (
            BENCHMARK_ROOT / "cases/qwen3_14b/greedy_sample/case.json"
        )
        greedy_case = json.loads(greedy_case_path.read_text(encoding="utf-8"))
        greedy_evaluator = common.load_case_evaluator(
            BENCHMARK_ROOT / "sealed_assets/qwen3_14b/greedy_sample"
        )
        allclose = greedy_evaluator._compare(
            greedy_case_path.parent, greedy_case,
        )["sampled_ids"]
        self.assertTrue(allclose(torch.tensor([1, 2]), torch.tensor([1, 2]))[0])
        self.assertFalse(allclose(torch.tensor([1, 3]), torch.tensor([1, 2]))[0])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "public"
            public.mkdir()
            invalid_case = {
                **qwen_case,
                "comparator": "public/comparator.json",
                "correctness": {"outputs": ["out"]},
            }
            (public / "comparator.json").write_text(
                '{"schema_version":1,"outputs":{"out":{'
                '"method":"ratio_allclose","rtol":0.0,"atol":0.0}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires exactly"):
                qwen_evaluator._compare(root, invalid_case)

            (public / "comparator.json").write_text(
                '{"schema_version":1,"outputs":{"out":{"method":"unknown"}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported comparator method"):
                qwen_evaluator._compare(root, invalid_case)

            (public / "comparator.json").write_text(
                '{"schema_version":1,"outputs":{"other":{"method":"allclose",'
                '"rtol":0.0,"atol":0.0}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly match"):
                qwen_evaluator._compare(root, invalid_case)


if __name__ == "__main__":
    unittest.main()
