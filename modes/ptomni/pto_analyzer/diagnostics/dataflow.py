from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Any

from ..profile import ProfileArchive
from .common import DetailRecords, add_records, family_sources, jsonl_records, make_result, observation, utf8_lines

_HINTS = [("compiler_hints", ["perf_hints.log"], True)]
_BUFFER_FAMILIES = [
    ("compiler_hints", ["perf_hints.log"], True),
    ("scope_statistics", ["scope_stats.jsonl"], True),
]
_HINT_PATTERN = re.compile(r"\[perf_hint\s+([^\]]+)\]")
_LOCATION_PATTERN = re.compile(r"\bat\s+(.+?:\d+(?::\d+)?)\s*$")


def tile_access_check(
    archive: ProfileArchive, *, include_details: bool = True, detail_cursor: tuple[Any, ...] | None = None, detail_limit: int | None = None,
) -> dict[str, Any]:
    sources = family_sources(archive, ["perf_hints.log"])
    stats: dict[str, dict[str, Any]] = {}
    details = DetailRecords(
        include_details, limit=detail_limit, after=detail_cursor,
        key=lambda row: (row["source_ref"]["path"], row["line_number"], row["hint_code"], row["category"]),
    )
    category_counts: Counter[str] = Counter()
    locations = set()
    for source in sources:
        if source.status != "present":
            continue
        try:
            lines = utf8_lines(source.data, newline=None)
        except (AttributeError, UnicodeDecodeError):
            add_records(stats, "compiler_hints", seen=1, rejected=1,
                        issues=["compiler hint file is not UTF-8"], source=source)
            continue
        seen = parsed = rejected = 0
        issues = []
        with lines:
            for line_number, line in enumerate(lines, 1):
                line = line.removesuffix("\n")
                if "perf_hint" not in line:
                    continue
                match = _HINT_PATTERN.search(line)
                if not match:
                    if "[perf_hint " in line:
                        seen += 1
                        rejected += 1
                        issues.append(f"line {line_number}: malformed compiler hint record")
                    continue
                seen += 1
                hint = _parse_hint(match.group(1))
                if hint is None:
                    rejected += 1
                    issues.append(f"line {line_number}: compiler hint has no category")
                    continue
                parsed += 1
                location = _LOCATION_PATTERN.search(line)
                category_counts[hint[1]] += 1
                location = location.group(1) if location else None
                if location:
                    locations.add((source.path, location))
                if include_details:
                    details.append(lambda: {
                        "source_ref": source.source(), "line_number": line_number,
                        "hint_code": hint[0], "category": hint[1], "text": line,
                        "location": location,
                    }, sort_key=(source.path, line_number, hint[0], hint[1]))
                else:
                    details.count_only()
        add_records(stats, "compiler_hints", seen=seen, parsed=parsed, rejected=rejected, issues=issues, source=source)
    location_count = len(locations)
    observations = []
    if details:
        observations.append(observation(
            "compiler_hint_count_by_category", dict(sorted(category_counts.items())), "hints", "all indexed compiler hint files",
            "count every parsed hint record by the category written in its hint text", len(details), sources,
        ))
        observations.append(observation(
            "compiler_hint_location_count", location_count, "locations", "source artifact and recorded location",
            "count distinct source-file/location pairs among parsed hints; locations are not remapped to the Candidate snapshot",
            len(details), sources,
        ))
    details.sort()
    return make_result(
        archive, "Dataflow", "tile_access_check", _HINTS, observations,
        ["Compiler hints are compile-time observations; their text is preserved without treating it as runtime evidence or an optimization instruction."],
        records=stats, detail_records=details,
    )


def buffer_pressure_check(
    archive: ProfileArchive, *, include_details: bool = True, detail_cursor: tuple[Any, ...] | None = None, detail_limit: int | None = None,
) -> dict[str, Any]:
    hint_sources = family_sources(archive, ["perf_hints.log"])
    scope_sources = family_sources(archive, ["scope_stats.jsonl"])
    stats: dict[str, dict[str, Any]] = {}
    details = DetailRecords(
        include_details, limit=detail_limit, after=detail_cursor,
        key=lambda row: (row["source_ref"]["path"], row["line_number"], row["record_type"]),
    )
    hint_counts: Counter[str] = Counter()
    field_values: dict[tuple[str, str], dict[str, Any]] = {}

    for source in hint_sources:
        if source.status != "present":
            continue
        try:
            lines = utf8_lines(source.data, newline=None)
        except (AttributeError, UnicodeDecodeError):
            add_records(stats, "compiler_hints", seen=1, rejected=1,
                        issues=["compiler hint file is not UTF-8"], source=source)
            continue
        seen = parsed = rejected = 0
        issues = []
        with lines:
            for line_number, line in enumerate(lines, 1):
                line = line.removesuffix("\n")
                if "PH-MR-001" not in line and "MemoryReuse" not in line:
                    continue
                seen += 1
                match = _HINT_PATTERN.search(line)
                hint = _parse_hint(match.group(1)) if match else None
                if hint is None:
                    rejected += 1
                    issues.append(f"line {line_number}: malformed MemoryReuse hint")
                    continue
                parsed += 1
                hint_counts[hint[1]] += 1
                if include_details:
                    details.append(
                        lambda: {"record_type": "compiler_hint", "source_ref": source.source(), "line_number": line_number,
                                 "hint_code": hint[0], "category": hint[1], "text": line},
                        sort_key=(source.path, line_number, "compiler_hint"),
                    )
                else:
                    details.count_only()
        add_records(stats, "compiler_hints", seen=seen, parsed=parsed, rejected=rejected, issues=issues, source=source)

    for source in scope_sources:
        if source.status != "present":
            continue

        def collect_scope_record(line_number: int, record: dict[str, Any]) -> None:
            if include_details:
                details.append(
                    lambda: {"record_type": "scope_statistics", "source_ref": source.source(), "line_number": line_number, "record": record},
                    sort_key=(source.path, line_number, "scope_statistics"),
                )
            else:
                details.count_only()
            _collect_field_values(record, "", field_values, source.path)

        seen, parsed, issues = jsonl_records(source, collect_scope_record)
        add_records(stats, "scope_statistics", seen=seen, parsed=parsed, rejected=seen - parsed, issues=issues, source=source)

    observations = []
    if hint_counts:
        observations.append(observation(
            "memory_reuse_hint_count_by_category", dict(sorted(hint_counts.items())), "hints", "all indexed compiler hint files",
            "count every parsed MemoryReuse hint by its recorded category", sum(hint_counts.values()), hint_sources,
        ))
    categorical_fields: dict[str, set[str]] = defaultdict(set)
    for (path, field), values in field_values.items():
        if values["mixed"]:
            categorical_fields[path].add(field)

    category_counts = {}
    for path, fields in categorical_fields.items():
        source = next((item for item in scope_sources if item.path == path), None)
        if source is not None:
            category_counts[path] = _scope_category_counts(source, fields)

    for (path, field), values in sorted(field_values.items()):
        if not field or not values["count"]:
            continue
        source = next((item for item in scope_sources if item.path == path), None)
        if source is None:
            continue
        if values["categories"] is not None or values["mixed"]:
            counts = values["categories"] or category_counts[path][field]
            value = dict(sorted(counts.items()))
            sample_count = values["count"]
            aggregation = "count every parsed categorical value at this exact field/index path"
        else:
            numeric_min = values["min"]
            numeric_max = values["max"]
            if values["all_integral"]:
                numeric_min = int(numeric_min)
                numeric_max = int(numeric_max)
            if values["has_nan"] or values["positive_infinity"] and values["negative_infinity"]:
                numeric_mean = math.nan
            elif values["positive_infinity"]:
                numeric_mean = math.inf
            elif values["negative_infinity"]:
                numeric_mean = -math.inf
            else:
                numeric_mean = float(values["total"] / values["numeric_count"])
            value = {"min": numeric_min, "max": numeric_max, "mean": numeric_mean}
            sample_count = values["numeric_count"]
            aggregation = "min, max, and arithmetic mean over all parsed scalar values at this exact field/index path"
        unit = _scope_unit(field)
        observations.append(observation(
            f"scope_field:{field}", value, unit,
            f"scope artifact {path}; field {field}", aggregation, sample_count, [source],
        ))
    details.sort()
    return make_result(
        archive, "Dataflow", "buffer_pressure_check", _BUFFER_FAMILIES, observations,
        ["MemoryReuse entries remain compiler observations.",
         "Scope fields retain their names, source-file scope, and recorded units; capacity and lifetime semantics are not inferred, so no pressure percentage is calculated."],
        records=stats, detail_records=details,
    )


def _parse_hint(contents: str) -> tuple[str, str] | None:
    parts = contents.strip().split(None, 1)
    if not parts:
        return None
    code = parts[0].rstrip(":")
    category = code
    if len(parts) == 2:
        category_match = re.match(r"([A-Za-z][A-Za-z0-9_]*)", parts[1])
        if category_match:
            category = category_match.group(1)
    return code, category


def _new_scope_summary() -> dict[str, Any]:
    return {
        "count": 0, "numeric_count": 0, "min": None, "max": None,
        "total": Fraction(), "all_integral": True, "has_nan": False,
        "positive_infinity": False, "negative_infinity": False,
        "categories": None, "mixed": False,
    }


def _collect_field_values(
    value: Any,
    path: str,
    output: dict[tuple[str, str], dict[str, Any]],
    source_path: str,
) -> None:
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            _collect_field_values(item, f"{path}.{key}" if path else key, output, source_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_field_values(item, f"{path}[{index}]", output, source_path)
        return

    summary = output.setdefault((source_path, path), _new_scope_summary())
    summary["count"] += 1
    if summary["mixed"]:
        return
    if summary["categories"] is not None:
        summary["categories"][_canonical(value)] += 1
        return
    if not _is_number(value):
        if summary["numeric_count"]:
            summary["mixed"] = True
        else:
            summary["categories"] = Counter({_canonical(value): 1})
        return

    number = value if type(value) is int else float(value)
    if summary["numeric_count"] == 0:
        summary["min"] = number
        summary["max"] = number
    else:
        if number < summary["min"]:
            summary["min"] = number
        if number > summary["max"]:
            summary["max"] = number
    summary["numeric_count"] += 1
    summary["all_integral"] = summary["all_integral"] and (
        type(value) is int or number.is_integer()
    )
    if math.isnan(number):
        summary["has_nan"] = True
    elif math.isinf(number):
        summary["positive_infinity"] |= number > 0
        summary["negative_infinity"] |= number < 0
    else:
        summary["total"] += Fraction(number) if type(value) is int else Fraction.from_float(number)


def _scope_category_counts(source: Any, fields: set[str]) -> dict[str, Counter[str]]:
    counts = {field: Counter() for field in fields}
    prefixes = {""}
    for field in fields:
        prefixes.add(field)
        prefixes.update(field[:index] for index, char in enumerate(field) if char in ".[" )

    def collect(_: int, record: dict[str, Any]) -> None:
        _count_scope_categories(record, "", counts, prefixes)

    jsonl_records(source, collect)
    return counts


def _count_scope_categories(value: Any, path: str, output: dict[str, Counter[str]], prefixes: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            if child in prefixes:
                _count_scope_categories(item, child, output, prefixes)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            if child in prefixes:
                _count_scope_categories(item, child, output, prefixes)
    elif path in output:
        output[path][_canonical(value)] += 1


def _is_number(value: Any) -> bool:
    return type(value) in {int, float}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _scope_unit(field: str) -> str | None:
    if field in {"fatal"}:
        return "boolean"
    if field in {"dropped", "total"}:
        return "records"
    return None
