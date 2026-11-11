from __future__ import annotations

import hashlib
import json
import csv
import io
import re
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import patch

from modes.ptomni.pto_analyzer import ProfileRef, ProfileRefError, open_toolset
from modes.ptomni.pto_analyzer import toolset as toolset_module
from modes.ptomni.pto_analyzer.diagnostics.common import DetailRecords


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
REAL_FIXTURES = FIXTURES / "real_single_card"
ROLE_TOOLS = {
    "Compute": {"compute_mapping_check", "compute_unit_activity", "compute_task_share"},
    "Dataflow": {"tile_access_check", "buffer_pressure_check"},
    "Schedule": {"dependency_path_check", "dispatch_gap_check", "task_distribution_check"},
}


def fixture_session(
    root: Path,
    *,
    fixture_dir: Path = FIXTURES,
    artifact_subdir: str = "",
    include: set[str] | None = None,
    overrides: dict[str, bytes] | None = None,
    session_id: str = "fixture-session",
    candidate_sha256: str = "a" * 64,
) -> ProfileRef:
    include = include or {path.name for path in fixture_dir.iterdir() if path.is_file()}
    overrides = overrides or {}
    archive = root / "staging" / "action_000001" / "artifacts"
    archive.mkdir(parents=True)
    entries = []
    protected = {}
    for name in sorted(include):
        data = overrides[name] if name in overrides else (fixture_dir / name).read_bytes()
        relative = Path(artifact_subdir) / name
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        entries.append({"kind": "file", "path": relative.as_posix(), "sha256": digest})
        protected[relative.as_posix()] = {"kind": "file", "sha256": digest}

    index_data = json.dumps({"event_seq": 1, "files": entries}).encode()
    (archive / "index.json").write_bytes(index_data)
    protected["index.json"] = {"kind": "file", "sha256": hashlib.sha256(index_data).hexdigest()}
    (root / "protection").mkdir()
    event = {
        "seq": 1,
        "action": "profile",
        "mode": "combined",
        "state": "completed",
        "integrity_passed": True,
        "attempt_id": 1,
        "case_id": "fixture.case",
        "content_sha256": candidate_sha256,
        "artifact_index": "staging/action_000001/artifacts/index.json",
        "profile_artifacts_path": "staging/action_000001/artifacts",
    }
    session_data = json.dumps({"session_id": session_id, "case_id": "fixture.case", "session_root": str(root)}).encode()
    event_data = (json.dumps(event) + "\n").encode()
    (root / "session.json").write_bytes(session_data)
    (root / "events.jsonl").write_bytes(event_data)
    manifest = {"roots": {
        "session": {"entries": {"": {"kind": "file", "sha256": hashlib.sha256(session_data).hexdigest()}}},
        "events": {"entries": {"": {"kind": "file", "sha256": hashlib.sha256(event_data).hexdigest()}}},
        "action_artifacts_1": {"entries": protected},
    }}
    (root / "protection" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ProfileRef.from_event(root, 1)


class ToolsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "session"
        self.ref = fixture_session(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_role_permissions_and_repeatability(self) -> None:
        source_hashes = {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.root.rglob("*") if path.is_file()}
        for tuner, names in ROLE_TOOLS.items():
            session = open_toolset(tuner, self.ref)
            self.assertEqual({spec["name"] for spec in session.tool_specs}, names)
            self.assertEqual({item["tool"] for item in session.initial_summary}, names)
            name = next(iter(names))
            self.assertEqual(session.call(name), session.call(name))
            other = next(role for role in ROLE_TOOLS if role != tuner)
            with self.assertRaises(PermissionError):
                session.call(next(iter(ROLE_TOOLS[other])))
        with self.assertRaises(ValueError):
            open_toolset("Lead", self.ref)
        with self.assertRaises(ValueError):
            open_toolset("Compute", self.ref).call("compute_unit_activity", {"path": "/tmp/profile"})
        self.assertEqual(source_hashes, {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.root.rglob("*") if path.is_file()})

    def test_profile_identity_is_returned_as_a_defensive_copy(self) -> None:
        session = open_toolset("Compute", self.ref)
        profile = session.profile
        profile["candidate_sha256"] = "changed"

        self.assertEqual(session.profile["candidate_sha256"], "a" * 64)

    def test_initial_summary_is_complete_and_uses_fact_contract(self) -> None:
        for tuner, names in ROLE_TOOLS.items():
            summary = open_toolset(tuner, self.ref).initial_summary
            self.assertEqual({item["tool"] for item in summary}, names)
            for item in summary:
                self.assertTrue({"tool", "tuner", "profile", "source_coverage", "observations", "limitations"}.issubset(item))
                self.assertNotIn("status", item)
                self.assertNotIn("facts", item)
                self.assertNotIn("evidence", item)
                for source in item["source_coverage"]:
                    self.assertTrue({"source_family", "required", "files_found", "files_read", "files_empty", "files_unreadable", "records_seen", "records_parsed", "records_rejected", "issues"}.issubset(source))
                    for reference in source["source_refs"]:
                        self.assertNotIn("status", reference)
                for observation in item["observations"]:
                    self.assertTrue({"name", "value", "unit", "scope", "aggregation", "sample_count", "source_refs"}.issubset(observation))

    def test_schedule_tool_specs_describe_their_interval_observations(self) -> None:
        descriptions = {
            spec["name"]: spec["description"]
            for spec in open_toolset("Schedule", self.ref).tool_specs
        }

        gap_description = descriptions["dispatch_gap_check"]
        self.assertIn("AICore interval gaps", gap_description)
        self.assertIn("not attributed to scheduling", gap_description)
        self.assertNotIn("remain uninterpreted", gap_description)

        distribution_description = descriptions["task_distribution_check"]
        self.assertIn("AICore interval durations", distribution_description)
        self.assertIn("per core", distribution_description)
        self.assertNotIn("without inferring core use", distribution_description)

    def test_dataflow_exposes_only_tools_with_independent_observations(self) -> None:
        session = open_toolset("Dataflow", self.ref)

        with self.assertRaisesRegex(KeyError, "unknown PTOAnalyzer tool"):
            session.call("pipeline_overlap_check")

    def test_paginated_tool_specs_describe_cursor_and_page_response(self) -> None:
        for tuner in ROLE_TOOLS:
            for spec in open_toolset(tuner, self.ref).tool_specs:
                with self.subTest(tool=spec["name"]):
                    properties = spec["input_schema"]["properties"]
                    self.assertIn("cursor", properties)
                    self.assertIn("limit", properties)
                    self.assertEqual(properties["limit"]["minimum"], 1)
                    self.assertEqual(properties["limit"]["maximum"], 50)
                    self.assertNotIn("offset", properties)
                    self.assertIn("next_cursor", spec["description"])
                    self.assertIn("returned_records", spec["description"])

    def test_initial_summary_matches_full_tool_calls(self) -> None:
        for tuner in ROLE_TOOLS:
            session = open_toolset(tuner, self.ref)
            summaries = {item["tool"]: item for item in session.initial_summary}
            for spec in session.tool_specs:
                name = spec["name"]
                with self.subTest(tuner=tuner, tool=name):
                    summary = summaries[name]
                    if "cursor" in spec["input_schema"]["properties"]:
                        result = session.call(name, {"limit": 1})
                        self.assertEqual(summary["details"]["total_records"], result["details"]["total_records"])
                    else:
                        result = session.call(name)
                    self.assertEqual(summary["profile"], result["profile"])
                    self.assertEqual(summary["source_coverage"], result["source_coverage"])
                    self.assertEqual(summary["observations"], result["observations"])
                    self.assertEqual(summary["limitations"], result["limitations"])
                    self.assertNotIn("_detail_records", summary)

    def test_paginated_details_skip_materializing_rejected_candidates(self) -> None:
        materialized = []
        details = DetailRecords(True, limit=2, key=lambda row: row["key"])

        for key in range(100):
            details.append(lambda key=key: materialized.append(key) or {"key": key}, sort_key=key)

        details.sort()
        self.assertEqual(details.records, [{"key": 0}, {"key": 1}])
        self.assertEqual(materialized, [0, 1])
        self.assertEqual(len(details), 100)
        self.assertEqual(len(details.records), 2)

    def test_paged_selector_bounds_candidates_for_large_and_tail_pages(self) -> None:
        for limit in (20, 50):
            details = DetailRecords(True, limit=limit, key=lambda row: row["key"])
            peak = 0
            for key in range(100_000):
                details.append(lambda key=key: {"key": key}, sort_key=key)
                peak = max(peak, len(details._heap))

            details.sort()
            self.assertEqual(peak, limit)
            self.assertEqual(len(details._heap), 0)
            self.assertEqual(len(details.records), limit)
            self.assertEqual([row["key"] for row in details.records], list(range(limit)))

            tail = DetailRecords(True, limit=limit, after=((99_999,), 100_000), key=lambda row: row["key"])
            for key in range(100_000):
                tail.append(lambda key=key: {"key": key}, sort_key=(key,))
            tail.sort()
            self.assertEqual(tail.records, [])
            self.assertEqual(len(tail._heap), 0)

    def test_repeated_sort_keys_use_stable_unique_tiebreakers(self) -> None:
        rows = [{"source": source, "value": value} for source in ("a", "b") for value in range(4)]
        expected = sorted(rows, key=lambda row: (row["source"], row["value"]))
        pages = []
        cursor = None
        while True:
            page = DetailRecords(
                True,
                limit=3,
                after=cursor,
                key=lambda row: (row["source"], 0),
            )
            for row in rows:
                page.append(row, sort_key=(row["source"], 0))
            page.sort()
            pages.extend(page.records)
            if not page.has_more:
                break
            cursor = page.last_key

        self.assertEqual(pages, expected)

    def test_buffer_pressure_omits_unlabeled_scope_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "empty-scope-field",
                overrides={"scope_stats.jsonl": b'{"": 1}\n'},
            )
            result = open_toolset("Dataflow", ref).call("buffer_pressure_check")

        self.assertNotIn("scope_field:", {item["name"] for item in result["observations"]})

    def test_buffer_pressure_aggregates_numeric_and_mixed_scope_fields(self) -> None:
        scope = b'{"measure": 1, "mixed": 1}\n{"measure": 1e50, "mixed": "one"}\n{"measure": -1e50, "mixed": 1}\n'
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "scope-aggregate",
                overrides={"scope_stats.jsonl": scope},
            )
            result = open_toolset("Dataflow", ref).call("buffer_pressure_check")

        observations = {row["name"]: row for row in result["observations"]}
        self.assertEqual(observations["scope_field:measure"]["value"], {
            "min": -1e50, "max": 1e50, "mean": 1 / 3,
        })
        self.assertEqual(observations["scope_field:mixed"]["value"], {"\"one\"": 1, "1": 2})

    def test_buffer_pressure_preserves_large_integer_scope_values(self) -> None:
        scope = b'{"measure": 9007199254740993}\n{"measure": 9007199254740995}\n'
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "large-integer-scope",
                overrides={"scope_stats.jsonl": scope},
            )
            result = open_toolset("Dataflow", ref).call("buffer_pressure_check")

        measure = next(row for row in result["observations"] if row["name"] == "scope_field:measure")
        self.assertEqual(measure["value"], {
            "min": 9007199254740993,
            "max": 9007199254740995,
            "mean": 9007199254740994.0,
        })

    def test_invalid_pagination_is_rejected_before_diagnostics_run(self) -> None:
        session = open_toolset("Compute", self.ref)
        calls = []

        def diagnostic(archive):
            calls.append(archive)
            return {}

        with patch.dict(toolset_module._TOOLS["Compute"], {"compute_unit_activity": diagnostic}):
            with self.assertRaises(ValueError):
                session.call("compute_unit_activity", {"limit": 0})

        self.assertEqual(calls, [])

    def test_cursor_errors_and_binding_are_checked_before_diagnostics(self) -> None:
        session = open_toolset("Compute", self.ref)
        cursor = session.call("compute_unit_activity", {"limit": 1})["details"]["next_cursor"]
        self.assertIsInstance(cursor, str)
        calls = []

        def diagnostic(archive, **kwargs):
            calls.append(kwargs)
            return {}

        with patch.dict(toolset_module._TOOLS["Compute"], {"compute_unit_activity": diagnostic}):
            for invalid in ("not-a-cursor", None, 123):
                with self.subTest(cursor=invalid), self.assertRaises(ValueError):
                    session.call("compute_unit_activity", {"cursor": invalid})
            with self.assertRaises(ValueError):
                session.call("compute_task_share", {"cursor": cursor})

            for index, character in enumerate(cursor):
                replacement = "A" if character != "A" else "B"
                tampered = cursor[:index] + replacement + cursor[index + 1:]
                with self.subTest(tampered_position=index), self.assertRaises(ValueError):
                    session.call("compute_unit_activity", {"cursor": tampered})

        other_ref = fixture_session(
            Path(self.temp.name) / "other-profile", session_id="other-session",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            open_toolset("Compute", other_ref).call("compute_unit_activity", {"cursor": cursor})

        candidate_ref = fixture_session(
            Path(self.temp.name) / "other-candidate", candidate_sha256="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            open_toolset("Compute", candidate_ref).call("compute_unit_activity", {"cursor": cursor})
        self.assertEqual(calls, [])

    def test_cursor_rejects_noncanonical_signature_base64(self) -> None:
        session = open_toolset("Compute", self.ref)
        cursor = session.call("compute_unit_activity", {"limit": 1})["details"]["next_cursor"]
        body, signature = cursor.rsplit(".", 1)
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        last_index = alphabet.index(signature[-1])
        alternate = next(
            character
            for character in alphabet[(last_index // 16) * 16:(last_index // 16 + 1) * 16]
            if character != signature[-1]
        )
        tampered = f"{body}.{signature[:-1]}{alternate}"

        with self.assertRaisesRegex(ValueError, "invalid cursor"):
            session.call("compute_unit_activity", {"cursor": tampered})

    def test_high_cardinality_details_page_without_changing_full_summary(self) -> None:
        source = (FIXTURES / "pmu.csv").read_text().splitlines()
        base = source[1].split(",")
        rows = []
        for core in range(73):
            row = base.copy()
            row[1] = str(core)
            rows.append(",".join(row))
        pmu = (source[0] + "\n" + "\n".join(rows) + "\n").encode()
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "many-cores", include={"pmu.csv"}, overrides={"pmu.csv": pmu})
            session = open_toolset("Compute", ref)
            summary = session.call("compute_unit_activity")
            initial_summary = next(row for row in session.initial_summary if row["tool"] == "compute_unit_activity")
            wide_page = session.call("compute_unit_activity", {"limit": 50})
            wide_tail_page = session.call("compute_unit_activity", {
                "cursor": wide_page["details"]["next_cursor"], "limit": 50,
            })
            pages = []
            cursor = None
            while True:
                args = {"limit": 17}
                if cursor is not None:
                    args["cursor"] = cursor
                page = session.call("compute_unit_activity", args)
                pages.append(page)
                cursor = page["details"]["next_cursor"]
                if cursor is None:
                    break

        self.assertEqual(summary["observations"], pages[0]["observations"])
        self.assertEqual(initial_summary["observations"], pages[0]["observations"])
        self.assertEqual(initial_summary["source_coverage"], pages[0]["source_coverage"])
        self.assertEqual(summary["source_coverage"], pages[-1]["source_coverage"])
        self.assertTrue(all(page["details"]["total_records"] == 73 for page in pages))
        self.assertTrue(all(page["details"]["returned_records"] == len(page["details"]["records"]) for page in pages))
        self.assertTrue(all(page["details"]["returned_records"] <= 17 for page in pages))
        self.assertEqual(summary["details"]["returned_records"], 20)
        self.assertEqual(len(summary["details"]["records"]), 20)
        self.assertIsNotNone(summary["details"]["next_cursor"])
        self.assertEqual(sum(len(page["details"]["records"]) for page in pages), 73)
        records = [row for page in pages for row in page["details"]["records"]]
        self.assertEqual(records, wide_page["details"]["records"] + wide_tail_page["details"]["records"])
        self.assertEqual(summary["observations"], pages[-1]["observations"])

    def test_all_paginated_tools_traverse_without_gaps_or_duplicates(self) -> None:
        sessions = [
            open_toolset("Compute", fixture_session(
                Path(self.temp.name) / "all-tools-compute", fixture_dir=REAL_FIXTURES,
                artifact_subdir=".benchmark_outputs/combined/build_output/_jit_fixture__001/dfx_outputs/rank0/d1",
            )),
            open_toolset("Dataflow", fixture_session(
                Path(self.temp.name) / "all-tools-dataflow", fixture_dir=REAL_FIXTURES,
            )),
            open_toolset("Schedule", fixture_session(
                Path(self.temp.name) / "all-tools-schedule", fixture_dir=REAL_FIXTURES,
            )),
        ]
        for session in sessions:
            for spec in session.tool_specs:
                if "cursor" not in spec["input_schema"]["properties"]:
                    continue
                name = spec["name"]
                with self.subTest(tool=name):
                    summary = next(row for row in session.initial_summary if row["tool"] == name)

                    def traverse(limit):
                        records = []
                        cursor = None
                        while True:
                            args = {"limit": limit}
                            if cursor is not None:
                                args["cursor"] = cursor
                            page = session.call(name, args)
                            self.assertEqual(page["observations"], summary["observations"])
                            self.assertEqual(page["source_coverage"], summary["source_coverage"])
                            self.assertEqual(page["details"]["returned_records"], len(page["details"]["records"]))
                            records.extend(page["details"]["records"])
                            cursor = page["details"]["next_cursor"]
                            if cursor is None:
                                break
                        return records

                    expected = traverse(50)
                    records = traverse(3)
                    self.assertGreater(len(expected), 3)
                    self.assertEqual(records, expected)
                    self.assertEqual(len(records), summary["details"]["total_records"])

    def test_paged_detail_accumulator_keeps_only_the_requested_sorted_window(self) -> None:
        records = DetailRecords(True, limit=3, key=lambda row: row["key"])
        for key in (9, 1, 7, 3, 8, 0, 5, 4, 6, 2):
            records.append({"key": key}, sort_key=key)
        records.sort()

        self.assertEqual(len(records), 10)
        self.assertEqual(len(records.records), 3)
        self.assertEqual([row["key"] for row in records.records], list(range(3)))

    def test_empty_detail_page_has_no_next_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "empty-page",
                include={"pmu.csv"},
                overrides={"pmu.csv": (FIXTURES / "pmu.csv").read_bytes().splitlines()[0] + b"\n"},
            )
            details = open_toolset("Compute", ref).call("compute_unit_activity")["details"]

        self.assertEqual(details["total_records"], 0)
        self.assertEqual(details["returned_records"], 0)
        self.assertEqual(details["records"], [])
        self.assertIsNone(details["next_cursor"])

    def test_pmu_rejects_bad_rows_without_dropping_valid_rows(self) -> None:
        lines = (FIXTURES / "pmu.csv").read_text().splitlines()
        bad = lines[1].split(",")
        bad[5] = "not-a-counter"
        data = ("\n".join([*lines, ",".join(bad)]) + "\n").encode()
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "partial-pmu", include={"pmu.csv"}, overrides={"pmu.csv": data})
            result = open_toolset("Compute", ref).call("compute_unit_activity")

        coverage = result["source_coverage"][0]
        self.assertEqual((coverage["records_seen"], coverage["records_parsed"], coverage["records_rejected"]), (3, 2, 1))
        self.assertEqual(sum(row["sample_count"] for row in result["observations"] if row["name"] == "pmu_total_cycles_sum"), 2)

    def test_header_only_pmu_with_missing_required_columns_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "header-only-pmu",
                include={"pmu.csv"},
                overrides={"pmu.csv": b"task_id,func_id\n"},
            )
            results = (
                open_toolset("Compute", ref).call("compute_unit_activity"),
                open_toolset("Compute", ref).call("compute_task_share"),
            )

        for result in results:
            with self.subTest(tool=result["tool"]):
                coverage = next(row for row in result["source_coverage"] if row["source_family"] == "pmu")
                self.assertEqual((coverage["records_seen"], coverage["records_parsed"]), (0, 0))
                self.assertTrue(any("missing" in issue["problem"].lower() for issue in coverage["issues"]))

    def test_task_details_include_every_joined_task(self) -> None:
        task_ids = list(range(100, 111))
        graph = {"tasks": [{"task_id": task_id, "kernel_ids": [0]} for task_id in task_ids], "edges": []}
        swimlane = {"chip_swimlane_level": 4, "metadata": {"num_cores": 2}, "aicore_tasks": [[0, task_id, 0, task_id, task_id + 5, 0] for task_id in task_ids]}
        names = {"callable_id_to_name": {"0": "fixture_callable"}}
        pmu_rows = []
        for task_id in task_ids:
            pmu_rows.append({"thread_id": "0", "core_id": "0", "task_id": str(task_id), "func_id": "0", "core_type": "0",
                             "pmu_total_cycles": "10", "vec_busy_cycles": "1", "cube_busy_cycles": "2", "scalar_busy_cycles": "3",
                             "mte1_busy_cycles": "4", "mte2_busy_cycles": "5", "mte3_busy_cycles": "6", "icache_miss": "0", "icache_req": "1", "event_type": "2"})
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(pmu_rows[0]))
        writer.writeheader()
        writer.writerows(pmu_rows)
        overrides = {
            "deps.json": json.dumps(graph).encode(),
            "chip_swimlane_records.json": json.dumps(swimlane).encode(),
            "name_map.json": json.dumps(names).encode(),
            "pmu.csv": stream.getvalue().encode(),
        }
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "many-tasks", fixture_dir=REAL_FIXTURES,
                                  artifact_subdir="build/dfx_outputs/rank0/d0",
                                  include={"deps.json", "chip_swimlane_records.json", "name_map__jit_compressor_test_675uvymw__001.json", "pmu.csv"},
                                  overrides={**overrides, "name_map__jit_compressor_test_675uvymw__001.json": json.dumps(names).encode()})
            session = open_toolset("Compute", ref)
            summary = session.call("compute_task_share")
            page = session.call("compute_task_share", {"limit": 50})["details"]

        self.assertTrue(any(row["name"] == "joined_task_count" and row["value"] == 11 for row in summary["observations"]))
        self.assertEqual(page["total_records"], 11)
        self.assertEqual(len(page["records"]), 11)

    def test_missing_source_is_coverage_not_a_zero_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "missing", include={"deps.json"})
            result = open_toolset("Compute", ref).call("compute_unit_activity")

        pmu = next(row for row in result["source_coverage"] if row["source_family"] == "pmu")
        self.assertEqual(pmu["files_found"], 0)
        self.assertEqual(pmu["records_parsed"], 0)
        self.assertTrue(any("not indexed" in issue["problem"] for issue in pmu["issues"]))
        self.assertEqual(result["observations"], [])

    def test_dispatch_gap_reports_only_verified_same_core_interval_gaps(self) -> None:
        swimlane = json.loads((FIXTURES / "chip_swimlane_records.json").read_text())
        swimlane["aicore_tasks"].extend([
            [0, 4294967299, 3, 140, 170, 1],
            [1, 4294967300, 2, 160, 180, 1],
        ])
        swimlane["aicpu_tasks"] = [[0, 1, 80, 130], [1, 1, 90, 160], [0, 3, 130, 180], [1, 2, 150, 190]]
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "gaps", overrides={"chip_swimlane_records.json": json.dumps(swimlane).encode()})
            result = open_toolset("Schedule", ref).call("dispatch_gap_check")

        gaps = [row for row in result["observations"] if row["name"] == "aicore_observed_gap_cycles"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["value"], {
            "0": {"count": 1, "min": 20, "max": 20, "sum": 20},
            "1": {"count": 1, "min": 10, "max": 10, "sum": 10},
        })
        self.assertEqual(gaps[0]["sample_count"], 2)
        self.assertEqual(gaps[0]["rank"], None)
        self.assertEqual(gaps[0]["dispatch"], None)
        self.assertIn("Observed AICore interval gaps are not attributed", " ".join(result["limitations"]))

    def test_dispatch_gap_counts_merged_trace_events_without_returning_them(self) -> None:
        event_count = 5000
        merged = json.dumps({"traceEvents": [{"name": f"event-{index}"} for index in range(event_count)]}).encode()
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "merged-events",
                include={"merged_swimlane.json"},
                overrides={"merged_swimlane.json": merged},
            )
            result = open_toolset("Schedule", ref).call("dispatch_gap_check")

        count = next(row for row in result["observations"] if row["name"] == "merged_trace_event_count")
        coverage = next(row for row in result["source_coverage"] if row["source_family"] == "merged_swimlane")
        self.assertEqual(count["value"], event_count)
        self.assertEqual(count["sample_count"], event_count)
        self.assertEqual((coverage["records_seen"], coverage["records_parsed"], coverage["records_rejected"]),
                         (event_count, event_count, 0))

    def test_dispatch_gap_reports_invalid_merged_trace_arrays(self) -> None:
        cases = (
            (b'{"traceEvents": {}}', "no traceEvents array"),
            (b'[]', "no traceEvents array"),
            (b'{"traceEvents": [{"name": "event"},]}', "invalid JSON document"),
        )
        for index, (contents, expected_issue) in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temp:
                ref = fixture_session(
                    Path(temp) / str(index),
                    include={"merged_swimlane.json"},
                    overrides={"merged_swimlane.json": contents},
                )
                result = open_toolset("Schedule", ref).call("dispatch_gap_check")

            coverage = next(row for row in result["source_coverage"] if row["source_family"] == "merged_swimlane")
            self.assertTrue(any(expected_issue in issue["problem"] for issue in coverage["issues"]))

    def test_task_distribution_summarizes_all_aicore_rows_by_core(self) -> None:
        swimlane = json.loads((FIXTURES / "chip_swimlane_records.json").read_text())
        swimlane["aicore_tasks"].extend([
            [0, 4294967297, 2, 130, 140, 1],
            [0, 4294967300, 3, 150, 170, 1],
        ])
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "distribution", overrides={"chip_swimlane_records.json": json.dumps(swimlane).encode()})
            result = open_toolset("Schedule", ref).call("task_distribution_check")

        counts = next(row for row in result["observations"] if row["name"] == "aicore_record_count_by_core")
        self.assertEqual(counts["value"], {"0": 3, "1": 1})
        tokens = next(row for row in result["observations"] if row["name"] == "unique_task_token_count_by_core")
        self.assertEqual(tokens["value"], {"0": 2, "1": 1})
        duration = next(row for row in result["observations"] if row["name"] == "aicore_interval_duration_cycles_by_core")
        self.assertEqual(duration["value"]["0"], {"count": 3, "min": 10, "max": 20, "sum": 50})
        self.assertEqual(duration["sample_count"], 4)
        self.assertTrue(any("Tuple row count is distinct from unique raw task-token count" in item for item in result["limitations"]))

    def test_schedule_tools_report_missing_or_invalid_aicore_array(self) -> None:
        tools = ("dispatch_gap_check", "task_distribution_check")
        cases = (
            ("missing", lambda value: value.pop("aicore_tasks"), "aicore_tasks is missing"),
            ("invalid", lambda value: value.__setitem__("aicore_tasks", 5), "aicore_tasks is not an array"),
        )
        for case_name, mutate, expected_issue in cases:
            for tool in tools:
                with self.subTest(case=case_name, tool=tool), tempfile.TemporaryDirectory() as temp:
                    swimlane = json.loads((FIXTURES / "chip_swimlane_records.json").read_text())
                    mutate(swimlane)
                    ref = fixture_session(
                        Path(temp) / "invalid-swimlane",
                        overrides={"chip_swimlane_records.json": json.dumps(swimlane).encode()},
                    )
                    result = open_toolset("Schedule", ref).call(tool)

                source = next(row for row in result["source_coverage"] if row["source_family"] == "runtime_swimlane")
                self.assertTrue(any(expected_issue in issue["problem"] for issue in source["issues"]))

    def test_page_arguments_reject_non_objects(self) -> None:
        session = open_toolset("Compute", self.ref)
        for invalid in ([], "", 0, False):
            with self.subTest(args=invalid), self.assertRaises(ValueError):
                session.call("compute_unit_activity", invalid)
        for invalid in ({"offset": -1}, {"offset": True}, {"cursor": None}, {"cursor": 1}, {"limit": 0}, {"limit": 51}, {"limit": False}, {"path": "pmu.csv"}):
            with self.subTest(args=invalid), self.assertRaises(ValueError):
                session.call("compute_unit_activity", invalid)

    def test_detail_source_references_leave_file_state_in_coverage(self) -> None:
        sessions = {
            "Compute": open_toolset("Compute", self.ref),
            "Dataflow": open_toolset("Dataflow", self.ref),
            "Schedule": open_toolset("Schedule", self.ref),
        }
        for tuner, session in sessions.items():
            for spec in session.tool_specs:
                name = spec["name"]
                with self.subTest(tool=name):
                    result = session.call(name, {"limit": 50})
                    for record in result["details"]["records"]:
                        refs = []
                        if "source_ref" in record:
                            refs.append(record["source_ref"])
                        refs.extend(record.get("source_refs", []))
                        for ref in refs:
                            self.assertNotIn("status", ref)
                            self.assertNotIn("error", ref)

    def test_observations_have_profile_source_and_units(self) -> None:
        compute_session = open_toolset("Compute", self.ref)
        result = compute_session.call("compute_unit_activity")
        self.assertNotIn("status", result)
        self.assertEqual(result["profile"]["event_id"], "fixture-session:1")
        pmu_fact = next(row for row in result["observations"] if row["name"] == "cube_busy_cycles_sum")
        self.assertEqual(pmu_fact["value"], 30)
        self.assertEqual(pmu_fact["unit"], "cycles")
        self.assertEqual(pmu_fact["source_refs"][0]["path"], "pmu.csv")

        schedule_session = open_toolset("Schedule", self.ref)
        graph = schedule_session.call("dependency_path_check")
        self.assertNotIn("status", graph)
        self.assertEqual(next(row["value"] for row in graph["observations"] if row["name"] == "dependency_graph_edge_path_depth"), 1)
        distribution = schedule_session.call("task_distribution_check")
        rank_count = next(row for row in distribution["observations"] if row["name"] == "swimlane_tuple_record_count")
        self.assertEqual(rank_count["value"], 2)
        self.assertTrue(any("AICore start/end cycles" in limitation for limitation in distribution["limitations"]))

    def test_dependency_summary_handles_empty_task_and_edge_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "empty-dependency-graph",
                include={"deps.json"},
                overrides={"deps.json": b'{"tasks": [], "edges": []}'},
            )
            result = open_toolset("Schedule", ref).call("dependency_path_check")

        self.assertEqual(result["details"]["total_records"], 0)
        self.assertEqual(result["details"]["returned_records"], 0)
        self.assertEqual(result["details"]["records"], [])

    def test_real_archived_single_card_artifacts_match_tool_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "real",
                fixture_dir=REAL_FIXTURES,
                artifact_subdir=".benchmark_outputs/combined/build_output/fixture__001/dfx_outputs",
            )
            compute = open_toolset("Compute", ref).call("compute_unit_activity")
            raw_pmu = list(csv.DictReader(io.StringIO((REAL_FIXTURES / "pmu.csv").read_text())))
            first = next(row for row in compute["observations"] if row["name"] == "pmu_total_cycles_sum")
            key = first["core_type"]
            expected_cycles = sum(
                int(row["pmu_total_cycles"]) for row in raw_pmu
                if row["core_type"] == key
            )
            self.assertEqual(first["value"], expected_cycles)
            self.assertEqual(first["unit"], "cycles")
            self.assertEqual(first["source_refs"][0]["sha256"], hashlib.sha256((REAL_FIXTURES / "pmu.csv").read_bytes()).hexdigest())

            graph = open_toolset("Schedule", ref).call("dependency_path_check")
            raw_graph = json.loads((REAL_FIXTURES / "deps.json").read_text())
            self.assertEqual(next(row["value"] for row in graph["observations"] if row["name"] == "dependency_task_metadata_count"), len(raw_graph["tasks"]))
            self.assertEqual(next(row["value"] for row in graph["observations"] if row["name"] == "dependency_edge_record_count"), len(raw_graph["edges"]))

            tile = open_toolset("Dataflow", ref).call("tile_access_check")
            raw_hints = len(re.findall(r"\[perf_hint\s+[^\]]+\]", (REAL_FIXTURES / "perf_hints.log").read_text()))
            counts = next(row["value"] for row in tile["observations"] if row["name"] == "compiler_hint_count_by_category")
            self.assertEqual(sum(counts.values()), raw_hints)

            mapping = open_toolset("Compute", ref).call("compute_mapping_check")
            self.assertNotIn("status", mapping)
            self.assertEqual(next(row["value"] for row in mapping["observations"] if row["name"] == "callable_mapping_count"), 5)

            task_share = open_toolset("Compute", ref).call("compute_task_share")
            self.assertNotIn("status", task_share)
            self.assertEqual(next(row["value"] for row in task_share["observations"] if row["name"] == "joined_task_count"), 5)
            self.assertEqual(task_share["details"]["total_records"], 5)
            self.assertEqual(sum(row["pmu_record_count"] for row in task_share["details"]["records"]), len(raw_pmu))
            pmu_coverage = next(row for row in task_share["source_coverage"] if row["source_family"] == "pmu")
            self.assertEqual((pmu_coverage["records_seen"], pmu_coverage["records_parsed"], pmu_coverage["records_rejected"]),
                             (len(raw_pmu), len(raw_pmu), 0))

    def test_large_pmu_csv_is_processed_without_retaining_all_rows(self) -> None:
        row_count = 100_000
        with (REAL_FIXTURES / "pmu.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(row for _ in range(row_count))
        pmu_data = csv_buffer.getvalue().encode()
        del writer, csv_buffer

        cases = (
            ("compute_unit_activity", {"pmu.csv"}),
            ("compute_task_share", {
                "pmu.csv", "deps.json", "chip_swimlane_records.json",
                "name_map__jit_compressor_test_675uvymw__001.json",
            }),
        )
        for tool, artifacts in cases:
            with self.subTest(tool=tool):
                ref = fixture_session(
                    Path(self.temp.name) / tool,
                    fixture_dir=REAL_FIXTURES,
                    artifact_subdir=".benchmark_outputs/combined/build_output/fixture__001/dfx_outputs/rank0/d1",
                    include=artifacts,
                    overrides={"pmu.csv": pmu_data},
                )
                session = open_toolset("Compute", ref)
                tracemalloc.start()
                try:
                    result = session.call(tool, {"limit": 50})
                    _, peak_bytes = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()

                if tool == "compute_unit_activity":
                    self.assertEqual(result["details"]["total_records"], row_count)
                else:
                    joined_count = next(
                        row["value"] for row in result["observations"] if row["name"] == "joined_pmu_record_count"
                    )
                    self.assertEqual(joined_count, row_count)
                self.assertLess(peak_bytes, 32 * 1024 * 1024)

    def test_large_scope_jsonl_summary_does_not_retain_decoded_file(self) -> None:
        row_count = 100_000
        scope_data = b'{"scope":"same","dropped":1}\n' * row_count
        ref = fixture_session(
            Path(self.temp.name) / "large-scope-jsonl",
            include={"scope_stats.jsonl"},
            overrides={"scope_stats.jsonl": scope_data},
        )
        session = open_toolset("Dataflow", ref)
        tracemalloc.start()
        try:
            result = next(item for item in session.initial_summary if item["tool"] == "buffer_pressure_check")
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        scope = next(row for row in result["source_coverage"] if row["source_family"] == "scope_statistics")
        self.assertEqual((scope["records_seen"], scope["records_parsed"], scope["records_rejected"]),
                         (row_count, row_count, 0))
        self.assertLess(peak_bytes, 5 * 1024 * 1024)

    def test_pmu_activity_keeps_builds_with_matching_rank_and_dispatch_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(
                Path(temp) / "build-scopes",
                artifact_subdir="build_a/dfx_outputs",
                include={"pmu.csv"},
            )
            archive_root = ref.session_root / "staging" / "action_000001" / "artifacts"
            data = (archive_root / "build_a" / "dfx_outputs" / "pmu.csv").read_bytes()
            second_path = "build_b/dfx_outputs/pmu.csv"
            (archive_root / second_path).parent.mkdir(parents=True)
            (archive_root / second_path).write_bytes(data)

            index_path = archive_root / "index.json"
            index = json.loads(index_path.read_text())
            digest = hashlib.sha256(data).hexdigest()
            index["files"].append({"kind": "file", "path": second_path, "sha256": digest})
            index_data = json.dumps(index).encode()
            index_path.write_bytes(index_data)

            manifest_path = ref.session_root / "protection" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            entries = manifest["roots"]["action_artifacts_1"]["entries"]
            entries[second_path] = {"kind": "file", "sha256": digest}
            entries["index.json"]["sha256"] = hashlib.sha256(index_data).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = open_toolset("Compute", ref).call("compute_unit_activity")
            facts = [item for item in result["observations"] if item["name"] == "pmu_record_count"]
            self.assertEqual({item["build_path"] for item in facts}, {"build_a", "build_b"})
            self.assertEqual(len(facts), 4)
            self.assertEqual({item["value"] for item in facts}, {1})

    def test_missing_empty_invalid_and_no_observation_cases_are_distinct(self) -> None:
        zero_pmu = (FIXTURES / "pmu.csv").read_text().splitlines()
        zero_pmu = "\n".join([zero_pmu[0], *(",".join([*row.split(",")[:5], *["0"] * 10]) for row in zero_pmu[1:])]) + "\n"
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "zero", overrides={"pmu.csv": zero_pmu.encode()})
            result = open_toolset("Compute", ref).call("compute_unit_activity")
            self.assertTrue(any(row["name"] == "pmu_total_cycles_sum" and row["value"] == 0 for row in result["observations"]))

        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "empty", overrides={"pmu.csv": b""})
            result = open_toolset("Compute", ref).call("compute_unit_activity")
            self.assertEqual(result["observations"], [])
            self.assertEqual(result["source_coverage"][0]["files_empty"], 1)

        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "empty-hints", overrides={"perf_hints.log": b""})
            result = open_toolset("Dataflow", ref).call("tile_access_check")
            self.assertEqual(result["observations"], [])
            self.assertEqual(result["source_coverage"][0]["files_empty"], 1)

        level_one = json.loads((FIXTURES / "chip_swimlane_records.json").read_text())
        level_one["chip_swimlane_level"] = 1
        with tempfile.TemporaryDirectory() as temp:
            ref = fixture_session(Path(temp) / "level1", overrides={"chip_swimlane_records.json": json.dumps(level_one).encode()})
            gap = open_toolset("Schedule", ref).call("dispatch_gap_check")
            self.assertTrue(any(row.get("capture_level") == 1 for row in gap["observations"]))
            self.assertTrue(any("level-4 swimlane coverage" in issue["problem"] for issue in gap["source_coverage"][0]["issues"]))

        with tempfile.TemporaryDirectory() as temp:
            bad_rows = (FIXTURES / "pmu.csv").read_text().splitlines()
            bad_values = bad_rows[1].split(",")
            bad_values[1] = "not-an-int"
            bad_rows[1] = ",".join(bad_values)
            ref = fixture_session(Path(temp) / "bad", overrides={"pmu.csv": ("\n".join(bad_rows) + "\n").encode()})
            result = open_toolset("Compute", ref).call("compute_unit_activity")
            coverage = result["source_coverage"][0]
            self.assertEqual((coverage["records_seen"], coverage["records_parsed"], coverage["records_rejected"]), (2, 1, 1))

    def test_non_object_json_roots_are_reported_without_crashing(self) -> None:
        name_map = next(path.name for path in REAL_FIXTURES.glob("name_map*.json"))
        cases = [
            ("Compute", "compute_mapping_check", {name_map}, {name_map: b"[]"}, "callable_name_map"),
            ("Schedule", "dependency_path_check", {"deps.json"}, {"deps.json": b"[]"}, "dependency_graph"),
            ("Schedule", "task_distribution_check", {"chip_swimlane_records.json"}, {"chip_swimlane_records.json": b"[]"}, "runtime_swimlane"),
        ]
        for index, (tuner, tool, include, overrides, family) in enumerate(cases):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as temp:
                ref = fixture_session(
                    Path(temp) / str(index),
                    fixture_dir=REAL_FIXTURES,
                    include=include,
                    overrides=overrides,
                )
                result = open_toolset(tuner, ref).call(tool)
                source = next(row for row in result["source_coverage"] if row["source_family"] == family)
                self.assertTrue(source["issues"])

    def test_archive_tampering_is_reported_and_not_modified(self) -> None:
        archive_file = self.root / "staging" / "action_000001" / "artifacts" / "pmu.csv"
        archive_file.write_text(archive_file.read_text() + "tampered\n", encoding="utf-8")
        result = open_toolset("Compute", self.ref).call("compute_unit_activity")
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["source_coverage"][0]["files_unreadable"], 1)

    def test_archive_symlink_cannot_read_outside_artifact_root(self) -> None:
        archive_file = self.root / "staging" / "action_000001" / "artifacts" / "pmu.csv"
        target = self.root / "outside" / "pmu.csv"
        target.parent.mkdir()
        target.write_bytes(archive_file.read_bytes())
        archive_file.unlink()
        archive_file.symlink_to(target)

        result = open_toolset("Compute", self.ref).call("compute_unit_activity")
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["source_coverage"][0]["files_unreadable"], 1)
        self.assertEqual(result["source_coverage"][0]["issues"][0]["problem"], "archived file escapes artifact root")

    def test_malformed_protection_manifest_is_a_profile_error(self) -> None:
        manifest_path = self.root / "protection" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["roots"] = []
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ProfileRefError):
            open_toolset("Compute", self.ref)

    def test_non_object_event_result_uses_session_identity(self) -> None:
        events_path = self.root / "events.jsonl"
        event = json.loads(events_path.read_text().splitlines()[0])
        event.pop("case_id")
        event["result"] = []
        events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

        manifest_path = self.root / "protection" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["roots"]["events"]["entries"][""]["sha256"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        ref = ProfileRef.from_event(self.root, 1)
        self.assertEqual(ref.case_id, "fixture.case")
        self.assertEqual(open_toolset("Compute", ref).profile["event_id"], "fixture-session:1")


if __name__ == "__main__":
    unittest.main()
