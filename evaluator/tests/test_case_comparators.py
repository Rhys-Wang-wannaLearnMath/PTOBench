import json
import inspect
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


EVALUATOR_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = EVALUATOR_DIR.parent
sys.path.insert(0, str(EVALUATOR_DIR))
import common


def build_case_comparators(evaluator, case_dir, case):
    """Adapt the existing local factories without imposing a runtime API."""
    names = [
        name for name in ("_compare", "_make_compare", "make_compare")
        if callable(getattr(evaluator, name, None))
    ]
    if len(names) != 1:
        raise AssertionError(f"{case['case_id']}: ambiguous/unsupported factories: {names}")
    factory = getattr(evaluator, names[0])
    parameters = inspect.signature(factory).parameters
    if tuple(parameters) not in (
        ("case_dir", "case"), ("case_dir", "case", "config"),
    ) or any(
        parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters.values()
    ):
        raise AssertionError(f"{case['case_id']}: unsupported factory signature: {parameters}")
    config = None
    policy = common.read_json(case_dir / case["comparator"])
    if "config" in parameters or any(
        rule["method"] == "custom" for rule in policy["outputs"].values()
    ):
        # run_path registers its temporary module during execution, as needed
        # by dataclasses with postponed annotations, then restores sys.modules.
        config = SimpleNamespace(**runpy.run_path(
            str(case_dir / case["config"]),
            run_name=f"comparator_config_{case['case_id'].replace('.', '_')}",
        ))
    if "config" in parameters:
        return factory(case_dir, case, config), config
    return factory(case_dir, case), config


def _fixed_output_shape(name, config):
    shape = getattr(config, f"{name.upper()}_SHAPE", None)
    if shape is not None:
        return tuple(shape)
    if name == "x_out" and all(hasattr(config, key) for key in ("T", "HC_MULT", "D")):
        return config.T, config.HC_MULT, config.D
    if name == "x_attn_active" and all(
        hasattr(config, key) for key in ("N_RANKS", "LOCAL_T", "HC_MULT", "D")
    ):
        return config.N_RANKS, config.LOCAL_T, config.HC_MULT, config.D
    if name == "x_moe_next" and all(
        hasattr(config, key) for key in ("N_RANKS", "MOE_TOKENS", "HC_MULT", "D")
    ):
        return config.N_RANKS, config.MOE_TOKENS, config.HC_MULT, config.D
    if name == "x_next":
        if all(hasattr(config, key) for key in ("EP_WORLD_SIZE", "PADDED_TOKENS", "HC_MULT", "D")):
            return config.EP_WORLD_SIZE, config.PADDED_TOKENS, config.HC_MULT, config.D
        if all(hasattr(config, key) for key in ("N_RANKS", "MOE_TOKENS", "HC_MULT", "D")):
            return config.N_RANKS, config.MOE_TOKENS, config.HC_MULT, config.D
    if name == "kv_cache" and all(
        hasattr(config, key) for key in ("PREFILL_ORI_BLOCK_NUM", "BLOCK_SIZE", "HEAD_DIM")
    ):
        return config.PREFILL_ORI_BLOCK_NUM, config.BLOCK_SIZE, 1, config.HEAD_DIM
    return None


def comparator_examples(name, rule, config):
    """Small CPU tensors with the callback's ranks, masks and paired bindings.

    Feature widths unrelated to comparison are kept small. Policy-bound
    capacities, top-k lengths, offsets, dtypes and metadata remain intact.
    These exercise callbacks, not NPU execution or formal Fixture correctness.
    """
    actual_outputs, expected_outputs, inputs = {}, {}, {}
    extras = []
    method = rule["method"]
    if method == "custom" and "name" not in rule:
        alias = rule.get("implementation", rule.get("comparator"))
        if alias is not None:
            rule = {**rule, "name": alias}
    if method in {"exact", "allclose", "ratio_allclose"}:
        shape = _fixed_output_shape(name, config) or (2,)
        expected = torch.ones(shape)
        rejected = torch.full_like(expected, 100)
    elif method != "custom":
        raise AssertionError(f"Unsupported comparator method: {method}")
    elif rule["name"] == "topk_pair_compare" and "paired_output" in rule:
        # Current indexers bind a score tensor and may select only the visible
        # prefix of each active token, padding the rest with -1.
        padded = "padding" in rule
        width = config.INDEXER_SCORE_CAP if padded else config.SCORE_LEN
        shape = (config.T, width) if padded else (config.B, config.S, width)
        scores = torch.arange(width, 0, -1, dtype=torch.float32).expand(shape).clone()
        if not rule["descending"]:
            scores = scores.flip(-1)
        if padded:
            positions = torch.zeros(config.T, dtype=torch.int32)
            positions[:2] = torch.tensor([7, 11], dtype=torch.int32)
            inputs = {"num_tokens": torch.tensor(2, dtype=torch.int32),
                      "position_ids": positions}
            expected = torch.full(shape, rule["padding"], dtype=torch.int32)
            expected[0, :2] = torch.arange(2)
            expected[1, :3] = torch.arange(3)
            bad_tail = expected.clone()
            bad_tail[2, 0] = 0
            duplicate = expected.clone()
            duplicate[0, 1] = 0
            extras.extend([
                ("inactive token must remain padded", bad_tail, False, {}),
                ("duplicate visible index", duplicate, False, {}),
                ("missing token metadata", expected.clone(), False, {"inputs": {}}),
            ])
        else:
            expected = (torch.arange(width, dtype=torch.int32) + rule["offset"]).expand(shape).clone()
        rejected = expected.clone()
        rejected[..., :2] = rejected[..., :2].flip(-1)
        paired = rule["paired_output"]
        actual_outputs[paired] = scores
        expected_outputs[paired] = scores
        extras.extend([
            ("tied permutation", rejected.clone(), True,
             {"actual_outputs": {paired: torch.ones_like(scores)}}),
            ("missing paired score", expected.clone(), False, {"actual_outputs": {}}),
        ])
    elif rule["name"] in {"topk_pair_compare", "indexed_topk_pair"}:
        indexed = rule["name"] == "indexed_topk_pair" or name == "topk_idxs"
        if rule["name"] == "indexed_topk_pair":
            length, offset = config.INDEXER_SCORE_CAP, rule["index_offset"]
            shape = (config.T, length)
        elif indexed:
            length, offset = config.SCORE_LEN, config.OFFSET
            shape = (config.B, config.S, length)
        else:
            if hasattr(config, "TOPK"):
                length = config.TOPK
            elif hasattr(config, "INDEX_TOPK"):
                length = config.INDEX_TOPK
            else:
                length = config.INDICES_SHAPE[-1]
            offset = 0
            shape = (1, length)
        expected = (torch.arange(length, dtype=torch.int32) + offset).expand(shape).clone()
        rejected = expected.clone()
        if length >= 2:
            rejected.reshape(-1, length)[:, :2] = rejected.reshape(-1, length)[:, :2].flip(-1)
        else:
            # The declared K=1 has no adjacent ordering constraint. Exercise
            # shape rejection rather than inventing a two-element output.
            rejected = torch.cat((expected, expected), dim=-1)
        scores = torch.arange(length, 0, -1, dtype=torch.float32).expand(shape).clone()
        if not rule["descending"]:
            scores = scores.flip(-1)
        # Indexed callbacks gather scores using indices. Other callbacks see
        # values already paired with the returned indices.
        actual_outputs[rule["values_output"]] = scores if indexed else scores.flip(-1)
        expected_outputs[rule["values_output"]] = scores
        if length >= 2:
            extras.append(("tied permutation", rejected, True, {
                "actual_outputs": {rule["values_output"]: torch.ones_like(scores)},
            }))
    elif rule["name"] == "valid_rows_allclose":
        shape = (2, rule["recv_max"], 2) if rule["is_3d"] else (2, rule["recv_max"])
        dtype = {"recv_x": torch.int8, "recv_token": torch.int32}.get(name, torch.float32)
        expected = torch.ones(shape, dtype=dtype)
        rejected = torch.zeros_like(expected)
        expected_outputs[rule["count_output"]] = torch.tensor([[1], [0]], dtype=torch.int32)
        ignored = expected.clone()
        ignored[:, 1:] = 100
        ignored[1] = 100
        extras.append(("unused receive rows", ignored, True, {}))
    elif rule["name"] == "active_recv_ratio_reldiff":
        expected = torch.ones((config.N_LOCAL_EXPERTS, config.RECV_MAX, 2), dtype=torch.bfloat16)
        rejected = expected.clone()
        rejected[:, 0] = 100
        inputs[rule["count_input"]] = torch.ones((config.N_LOCAL_EXPERTS, 1), dtype=torch.int32)
        if config.RECV_MAX > 1:
            inactive = expected.clone()
            inactive[:, -1] = 100
            extras.append(("inactive receive row changed", inactive, False, {}))
        extras.append(("missing expert counts", expected.clone(), False, {"inputs": {}}))
    elif rule["name"] == "token_partition_ratio_reldiff":
        rows = rule["valid_rows"]
        if rule["token_axis"] != 1 or rows <= 0:
            raise AssertionError("Unsupported token partition preset")
        # Enough tail rows expose implementations that dilute active errors by
        # pooling both regions into one error budget.
        tail_rows = max(rows, 32)
        expected = torch.ones((2, rows + tail_rows, 1, 2), dtype=torch.bfloat16)
        rejected = expected.clone()
        rejected[:, :rows] = 100
        ratio = rule["base_rule"]["max_error_ratio"]
        for label, region in (("active", slice(0, rows)), ("tail", slice(rows, None))):
            actual = expected.clone()
            values = actual[:, region].clone()
            count = round(ratio * values.numel()) + 1
            values.reshape(-1)[:count] = 100
            actual[:, region] = values
            extras.append((f"{label} errors cannot use the other region's budget", actual, False, {}))
        if all(round(ratio * expected[:, region].numel()) >= 1
               for region in (slice(0, rows), slice(rows, None))):
            allowed = expected.clone()
            allowed[0, 0, 0, 0] = 100
            allowed[0, rows, 0, 0] = 100
            extras.append(("permitted mismatch in each region", allowed, True, {}))
        nonfinite = expected.clone()
        nonfinite[0, 0, 0, 0] = torch.nan
        extras.append(("nonfinite active value", nonfinite, False, {}))
    elif rule["name"] == "mapped_pool_ratio_allclose":
        block_size = rule["block_size"]
        ranked = rule.get("leading_rank_axis", False)
        ranks = rule.get("mapping_shape", [1])[0] if ranked else 1
        shape = (ranks, 2, block_size, 2) if ranked else (2, block_size, 2)
        expected = torch.ones(shape, dtype=torch.float32)
        mapping = torch.full(rule["mapping_shape"], -1, dtype=torch.int32)
        mapping.reshape(-1)[:2] = torch.tensor([0, block_size])
        binding = rule["mapping"] if "mapping" in rule else rule["mapping_name"]
        inputs[binding] = mapping
        rejected = expected.clone()
        rejected[0, 0] = 100
        unmapped = expected.clone()
        unmapped[0, 1] = 100
        invalid_mapping = mapping.clone()
        invalid_mapping.reshape(-1)[0] = 2 * block_size
        duplicate_mapping = mapping.clone()
        duplicate_mapping.reshape(-1)[1] = 0
        extras.extend([
            ("unmapped pool row changed", unmapped, False, {}),
            ("missing mappings", expected.clone(), False, {"inputs": {}}),
            ("mapping outside pool", expected.clone(), False,
             {"inputs": {binding: invalid_mapping}}),
            ("duplicate mapped row", expected.clone(), False,
             {"inputs": {binding: duplicate_mapping}}),
        ])
    elif rule["name"] == "mapped_pool_compare":
        bindings = rule["mapping_inputs"]
        ranks = 2 if rule["ranked"] else 1
        rows = rule["valid_rows"]
        selected = 2 if rows is None else min(2, rows)
        mapping_length = max(2, rows or 0) + 1
        pool_rows = len(bindings) * selected + 1
        dtype = torch.int8 if name == "idx_kv_cache" else torch.float32
        shape = (ranks, pool_rows, 2) if rule["ranked"] else (pool_rows, 2)
        expected = torch.ones(shape, dtype=dtype)
        for index, binding in enumerate(bindings):
            mapping = torch.full((ranks, mapping_length), -1, dtype=torch.int32)
            mapping[:, :selected] = torch.arange(index * selected, (index + 1) * selected)
            inputs[binding] = mapping if rule["ranked"] else mapping[0]
        rejected = expected.clone()
        rejected.reshape(ranks, pool_rows, 2)[:, 0] = 100
        unmapped = expected.clone()
        unmapped.reshape(ranks, pool_rows, 2)[:, -1] = 100
        extras.extend([
            ("unmapped row changed", unmapped, False, {}),
            ("missing mappings", expected.clone(), False, {"inputs": {}}),
        ])
        # Unmapped values must remain exact even when the change would fit the
        # written-row numerical tolerance.
        tolerance = rule["base_rule"]["atol"] + rule["base_rule"]["rtol"]
        if dtype == torch.float32 and tolerance > 0:
            tolerated = expected.clone()
            tolerated.reshape(ranks, pool_rows, 2)[:, -1] += tolerance / 2
            extras.append(("unmapped row cannot use numerical tolerance", tolerated, False, {}))
            if selected:
                written = expected.clone()
                written.reshape(ranks, pool_rows, 2)[:, 0] += tolerance / 2
                extras.append(("written row accepts numerical tolerance", written, True, {}))
        binding = bindings[0]
        invalid = inputs[binding].clone()
        invalid.reshape(ranks, -1)[:, 0] = pool_rows
        extras.append(("mapping outside pool", expected.clone(), False,
                       {"inputs": {**inputs, binding: invalid}}))
        if selected >= 2:
            duplicate = inputs[binding].clone()
            duplicate.reshape(ranks, -1)[:, 1] = duplicate.reshape(ranks, -1)[:, 0]
            extras.append(("duplicate mapped row", expected.clone(), False,
                           {"inputs": {**inputs, binding: duplicate}}))
        if rows is not None:
            invalid_tail = inputs[binding].clone()
            invalid_tail.reshape(ranks, -1)[:, rows] = 0
            extras.append(("mapping beyond valid rows", expected.clone(), False,
                           {"inputs": {**inputs, binding: invalid_tail}}))
    elif rule["name"] == "mapped_active_rows_ratio_allclose":
        dtype = torch.int8 if name == "idx_kv_cache" else torch.float32
        block_count = config.PREFILL_IDX_BLOCK_NUM
        block_size = rule.get("block_size", config.BLOCK_SIZE)
        width = config.IDX_HEAD_DIM if name == "idx_kv_cache" else 1
        expected = torch.ones((block_count, block_size, 1, width), dtype=dtype)
        rejected = expected.clone()
        rejected.reshape(-1, 2)[0] = 100
        mapping_name = rule.get("mapping_input", rule.get("mapping_name"))
        mapping_shape = rule.get("mapping_shape", [2])
        inputs[mapping_name] = torch.full(mapping_shape, -1, dtype=torch.int64)
        inactive = expected.clone()
        inactive.reshape(-1, 2)[-1] = 100
        extras.extend([
            ("inactive mapped row changed", inactive, False, {}),
            ("missing mapping", expected.clone(), False, {"inputs": {}}),
        ])
    elif rule["name"] in {"valid_ratio_reldiff", "packed_valid_ratio_reldiff"} and "chunk_lens_input" in rule:
        shape = _fixed_output_shape(name, config)
        if shape is None:
            shape = (config.N_RANKS, config.T, config.HC_MULT, config.D)
        ranks = shape[0]
        chunks = getattr(config, "USER_BATCH", 2)
        axis = rule.get("valid_axis", 1)
        axis_size = shape[axis]
        chunks = max(1, min(chunks, axis_size))
        expected = torch.ones(shape, dtype=torch.bfloat16)
        rejected = expected.clone()
        rejected.select(axis, 0).fill_(100)
        lengths = torch.ones((ranks, chunks), dtype=torch.int32)
        offsets = torch.arange(chunks, dtype=torch.int32).expand(ranks, -1).clone()
        inputs[rule["chunk_lens_input"]] = lengths
        inputs[rule["chunk_offsets_input"]] = offsets
        padding_changed = expected.clone()
        padding_changed.select(axis, min(chunks, axis_size - 1)).fill_(100)
        padding_policy = rule.get("padding_policy", "ignore")
        if padding_policy not in {"exact", "ignore"}:
            raise AssertionError(f"Unsupported padding policy: {padding_policy}")
        extras.extend([
            ("unselected chunk rows", padding_changed, padding_policy == "ignore", {}),
            ("missing chunk metadata", expected.clone(), False, {"inputs": {}}),
        ])
    elif rule["name"] == "score_valid_region_compare":
        shape = ((config.B, config.S, config.SCORE_LEN)
                 if all(hasattr(config, key) for key in ("B", "S", "SCORE_LEN"))
                 else (1, 1, 4))
        expected = torch.full(shape, rule["invalid_value"], dtype=torch.float32)
        expected[..., :2] = 1
        rejected = expected.clone()
        rejected[..., : min(4, expected.shape[-1])] = 100
        changed_tail = expected.clone()
        changed_tail[..., -1] = 100
        nonfinite = expected.clone()
        nonfinite[..., 0] = torch.nan
        extras.extend([
            ("invalid-value tail is ignored", changed_tail, True, {}),
            ("nonfinite active score", nonfinite, False, {}),
        ])
    elif rule["name"] in {"score_valid_region_ratio_allclose", "score_valid_rows_ratio_allclose"}:
        token_rows = rule["name"] == "score_valid_rows_ratio_allclose"
        if token_rows:
            shape = (config.T, config.INDEXER_SCORE_CAP)
        elif all(hasattr(config, key) for key in ("B", "S", "SCORE_LEN")):
            shape = (config.B, config.S, config.SCORE_LEN)
        else:
            shape = (1, 1, 4)
        expected = torch.full(shape, rule["tail_sentinel"])
        if token_rows:
            inputs["num_tokens"] = torch.tensor(2, dtype=torch.int32)
            expected[:2] = 1
        else:
            expected[..., :2] = 1
        rejected = expected.clone()
        allowance = round(rule["max_error_ratio"] * (2 * config.INDEXER_SCORE_CAP if token_rows else expected.numel()))
        rejected.reshape(-1)[:allowance + 1] = 100
        changed_tail = expected.clone()
        changed_tail.reshape(-1)[-1] = 100
        nonfinite = expected.clone()
        nonfinite.reshape(-1)[0] = torch.nan
        extras.extend([
            ("inactive score tail", changed_tail, not token_rows, {}),
            ("nonfinite active score", nonfinite, False, {}),
        ])
        if token_rows:
            extras.append(("missing token count", expected.clone(), False, {"inputs": {}}))
    elif rule["name"] == "valid_ratio_reldiff" and (
        rule.get("valid_rows_from") == "num_tokens" or rule.get("valid_prefix") == "num_tokens"
        or rule.get("valid_rows") == "num_tokens"
    ):
        axis = rule.get("valid_axis", 0)
        shape = [2, 2, 2]
        shape[axis] = 3
        shape = _fixed_output_shape(name, config) or tuple(shape)
        shape = list(shape)
        rows = min(2, shape[axis])
        expected = torch.zeros(shape, dtype=torch.float32)
        active = [slice(None)] * len(shape)
        active[axis] = slice(0, rows)
        expected[tuple(active)] = 1
        inputs["num_tokens"] = torch.tensor(rows, dtype=torch.int32)
        rejected = expected.clone()
        rejected[0] = 100
        nonzero_tail = expected.clone()
        nonzero_tail[-1] = 1
        extras.extend([
            ("inactive token must remain zero", nonzero_tail, False, {}),
            ("missing token count", expected.clone(), False, {"inputs": {}}),
            ("token count out of bounds", expected.clone(), False,
             {"inputs": {"num_tokens": torch.tensor(shape[axis] + 1, dtype=torch.int32)}}),
        ])
    elif rule["name"] in {"ratio_reldiff", "valid_ratio_reldiff"}:
        rows = rule.get("valid_rows")
        if rule.get("num_tokens") is not None:
            rows = rule["num_tokens"]
            shape = _fixed_output_shape(name, config) or (config.T, 2, 2)
            expected = torch.zeros(shape, dtype=torch.float32)
            expected[:rows] = 1
            rejected = expected.clone()
            rejected[:rows] = 100
            inputs["num_tokens"] = torch.tensor(rows, dtype=torch.int32)
            examples = [("equal", expected.clone(), True, {}),
                        ("different", rejected, False, {})]
            for label, actual, passed, updates in examples:
                yield label, actual, expected, {
                    "actual_outputs": {**actual_outputs, name: actual},
                    "expected_outputs": {**expected_outputs, name: expected},
                    "inputs": inputs, "rtol": 1e10, "atol": 1e10, **updates,
                }, passed
            return
        rank = {"sh": 2, "recv_y": 3, "x_out": 3, "x_next": 4,
                "x_attn_active": 3, "x_moe_next": 3}.get(name, 3)
        shape = list(_fixed_output_shape(name, config) or (2,) * rank)
        if rows is not None:
            axis = rule.get("valid_axis", 0)
            if isinstance(rows, str):
                rows = 2
                inputs["num_tokens"] = torch.tensor(rows, dtype=torch.int32)
            if rule.get("zero_tail") and _fixed_output_shape(name, config) is None:
                shape[axis] = max(shape[axis], rows + 1)
        expected = torch.ones(shape, dtype=torch.bfloat16)
        if rows is not None and rule.get("zero_tail"):
            active = [slice(None)] * rank
            active[axis] = slice(rows, None)
            expected[tuple(active)] = 0
        rejected = torch.full_like(expected, 100)
    elif rule["name"] == "int8_lsb":
        expected = torch.zeros((1, 2), dtype=torch.int8)
        rejected = torch.full_like(expected, rule["max_abs_diff"] + 1)
        extras.append(("allowed LSB difference", torch.full_like(expected, rule["max_abs_diff"]), True, {}))
    else:
        raise AssertionError(f"Unsupported custom comparator: {rule['name']}")

    examples = [
        ("equal", expected.clone(), True, {}),
        ("different", rejected, False, {}),
        *extras,
    ]
    for label, actual, passed, updates in examples:
        kwargs = {
            "actual_outputs": {**actual_outputs, name: actual},
            "expected_outputs": {**expected_outputs, name: expected},
            "inputs": inputs,
            # Call-site tolerances must not replace the public policy.
            "rtol": 1e10,
            "atol": 1e10,
            **updates,
        }
        yield label, actual, expected, kwargs, passed



class CaseComparatorTest(unittest.TestCase):
    def test_every_case_owns_and_executes_its_comparator(self):
        self.assertFalse(hasattr(common, "make_comparators"))
        case_paths = sorted((BENCHMARK_ROOT / "cases").glob("*/*/case.json"))
        self.assertTrue(case_paths)

        constructed = 0
        non_ready_missing = []
        ready = 0
        for index, case_path in enumerate(case_paths):
            case_dir = case_path.parent
            case = json.loads(case_path.read_text(encoding="utf-8"))
            family, name = case["case_id"].split(".", 1)
            sealed = common.sealed_root() / family / name
            certification_path = sealed / "certification.json"
            certification = (
                common.read_json(certification_path) if certification_path.is_file() else {}
            )
            is_ready = certification.get("status") == "ready"
            ready += is_ready
            with self.subTest(case=case["case_id"], certification=certification.get("status")):
                if not (sealed / "evaluate.py").is_file():
                    self.assertFalse(is_ready, "ready Case is missing its evaluator")
                    non_ready_missing.append(case["case_id"])
                    continue
                evaluator = common.load_isolated_module(
                    sealed / "evaluate.py", f"case_comparator_test_{index}"
                )
                source = (sealed / "evaluate.py").read_text(encoding="utf-8")
                self.assertNotIn("common.make_comparators", source)
                comparators, config = build_case_comparators(evaluator, case_dir, case)
                constructed += 1
                self.assertEqual(set(comparators), set(case["correctness"]["outputs"]))
                policy = common.read_json(case_dir / case["comparator"])
                self.assertEqual(set(policy["outputs"]), set(comparators))
                if is_ready:
                    self.assertEqual(
                        certification["comparator_sha256"],
                        common.comparator_sha256(case_dir, case),
                        "ready certification is bound to a stale comparator",
                    )
                for name, compare in comparators.items():
                    with self.subTest(output=name):
                        signature = inspect.signature(compare)
                        self.assertNotIn("rtol", signature.parameters)
                        self.assertNotIn("atol", signature.parameters)
                        for label, actual, expected, kwargs, passed in comparator_examples(
                            name, policy["outputs"][name], config,
                        ):
                            with self.subTest(example=label):
                                result = compare(actual, expected, **kwargs)
                                self.assertIsInstance(result, tuple)
                                self.assertEqual(bool(result[0]), passed, result[1])
        print(
            f"Comparator coverage: inventory={len(case_paths)}, ready={ready}, "
            f"constructed={constructed}, non_ready_missing_evaluator={len(non_ready_missing)}",
            flush=True,
        )
        for case_id in non_ready_missing:
            print(f"NOT_RUN (non-ready, missing evaluator): {case_id}", flush=True)

    def test_declared_methods_and_strict_schema(self):
        ratio_case_path = (
            BENCHMARK_ROOT
            / "cases/deepseek_v4_flash_dspark/dspark_proj_flash/case.json"
        )
        ratio_case = json.loads(ratio_case_path.read_text(encoding="utf-8"))
        ratio_evaluator = common.load_case_evaluator(
            common.sealed_root() / "deepseek_v4_flash_dspark/dspark_proj_flash"
        )
        ratio = ratio_evaluator.make_compare(ratio_case_path.parent, ratio_case)["main_x"]
        rule = common.read_json(ratio_case_path.parent / ratio_case["comparator"])["outputs"]["main_x"]
        expected = torch.zeros(100)
        actual = expected.clone()
        allowed = round(rule["max_error_ratio"] * expected.numel())
        actual[:allowed] = 1
        self.assertTrue(ratio(actual, expected)[0])
        actual[allowed] = 1
        self.assertFalse(ratio(actual, expected)[0])
        actual = expected.clone()
        actual[0] = torch.nan
        self.assertFalse(ratio(actual, expected)[0])

        allclose_case_path = (
            BENCHMARK_ROOT / "cases/deepseek_v4_flash_dspark/dspark_markov_flash/case.json"
        )
        allclose_case = json.loads(allclose_case_path.read_text(encoding="utf-8"))
        allclose_evaluator = common.load_case_evaluator(
            common.sealed_root() / "deepseek_v4_flash_dspark/dspark_markov_flash"
        )
        allclose = allclose_evaluator.make_compare(allclose_case_path.parent, allclose_case)["draft_token_ids"]
        self.assertTrue(allclose(torch.tensor([1, 2]), torch.tensor([1, 2]))[0])
        self.assertFalse(allclose(torch.tensor([1, 3]), torch.tensor([1, 2]))[0])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "public"
            public.mkdir()
            invalid_case = {
                **ratio_case,
                "comparator": "public/comparator.json",
                "correctness": {"outputs": ["out"]},
            }
            (public / "comparator.json").write_text(
                '{"schema_version":1,"outputs":{"out":{'
                '"method":"ratio_allclose","rtol":0.0,"atol":0.0}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid ratio_allclose fields"):
                ratio_evaluator.make_compare(root, invalid_case)

            (public / "comparator.json").write_text(
                '{"schema_version":1,"outputs":{"out":{"method":"unknown"}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported comparator method"):
                ratio_evaluator.make_compare(root, invalid_case)

            (public / "comparator.json").write_text(
                '{"schema_version":1,"outputs":{"other":{"method":"allclose",'
                '"rtol":0.0,"atol":0.0}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output coverage disagrees"):
                ratio_evaluator.make_compare(root, invalid_case)


if __name__ == "__main__":
    unittest.main()
