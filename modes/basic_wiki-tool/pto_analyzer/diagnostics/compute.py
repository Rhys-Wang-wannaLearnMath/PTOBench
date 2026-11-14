from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ..profile import ArtifactRead, ProfileArchive
from .common import DetailRecords, IssueSamples, add_records, csv_records, family_sources, json_value, make_result, observation

PMU_COUNTERS = {
    "pmu_total_cycles": "cycles",
    "vec_busy_cycles": "cycles",
    "cube_busy_cycles": "cycles",
    "scalar_busy_cycles": "cycles",
    "mte1_busy_cycles": "cycles",
    "mte2_busy_cycles": "cycles",
    "mte3_busy_cycles": "cycles",
    "icache_miss": "events",
    "icache_req": "events",
}

_MAPPING_FAMILIES = [
    ("dispatch_program", ["dispatch_program.json"], True),
    ("callable_name_map", ["name_map*.json"], True),
    ("kernel_config", ["kernel_config.py"], True),
]
_PMU_FAMILY = [("pmu", ["pmu.csv"], True)]
_TASK_FAMILIES = [
    ("pmu", ["pmu.csv"], True),
    ("runtime_swimlane", ["chip_swimlane_records.json"], True),
    ("dependency_graph", ["deps.json"], True),
    ("callable_name_map", ["name_map*.json"], True),
]


def compute_mapping_check(
    archive: ProfileArchive, *, include_details: bool = True, detail_cursor: tuple[Any, ...] | None = None, detail_limit: int | None = None,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    programs = []
    mapping_count = 0
    mappings_by_source: dict[str, int] = defaultdict(int)
    details = DetailRecords(include_details, limit=detail_limit, after=detail_cursor,
                            key=lambda row: (row["source_ref"]["path"], row["kind"], str(row.get("callable_id", ""))))
    program_sources = family_sources(archive, ["dispatch_program.json"])
    name_sources = family_sources(archive, ["name_map*.json"])
    config_sources = family_sources(archive, ["kernel_config.py"])

    for source in program_sources:
        if source.status != "present":
            continue
        value, seen, error = json_value(source)
        problem = error
        program = value.get("program") if isinstance(value, dict) else None
        if problem is None and not isinstance(program, str):
            problem = "dispatch program document has no string program field"
        parsed = 0 if problem else 1
        add_records(records, "dispatch_program", seen=seen, parsed=parsed, rejected=seen - parsed,
                    issues=[problem] if problem else (), source=source)
        if problem:
            continue
        programs.append({"source_path": source.path, "program": program})
        if include_details:
            details.append(lambda: {"kind": "dispatch_program", "source_ref": source.source(), "program": program},
                           sort_key=(source.path, "dispatch_program", ""))
        else:
            details.count_only()

    for source in name_sources:
        if source.status != "present":
            continue
        value, seen, error = json_value(source)
        callables = value.get("callable_id_to_name") if isinstance(value, dict) else None
        if error:
            add_records(records, "callable_name_map", seen=seen, rejected=seen,
                        issues=[error], source=source)
            continue
        if not isinstance(callables, dict):
            add_records(records, "callable_name_map", seen=seen, rejected=seen,
                        issues=["name map has no callable_id_to_name object"], source=source)
            continue
        parsed = 0
        for callable_id, name in callables.items():
            try:
                normalized_id = int(callable_id)
                if normalized_id < 0 or not isinstance(name, str):
                    raise ValueError
            except (TypeError, ValueError):
                continue
            parsed += 1
            mapping_count += 1
            mappings_by_source[source.path] += 1
            if include_details:
                details.append(
                    lambda: {"kind": "callable_name", "source_ref": source.source(), "callable_id": normalized_id, "name": name},
                    sort_key=(source.path, "callable_name", str(normalized_id)),
                )
            else:
                details.count_only()
        rejected = len(callables) - parsed
        add_records(records, "callable_name_map", seen=len(callables), parsed=parsed, rejected=rejected,
                    issues=["name map contains invalid callable IDs or names"] if rejected else (), source=source)

    config_count = 0
    for source in config_sources:
        if source.status != "present":
            continue
        try:
            source.data.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            add_records(records, "kernel_config", seen=1, rejected=1,
                        issues=["kernel config is not UTF-8"], source=source)
            continue
        config_count += 1
        add_records(records, "kernel_config", seen=1, parsed=1, source=source)
        if include_details:
            details.append(lambda: {"kind": "kernel_config", "source_ref": source.source()},
                           sort_key=(source.path, "kernel_config", ""))
        else:
            details.count_only()

    observations = []
    if programs:
        observations.append(observation(
            "dispatch_program_mapping", programs, "program labels", "archived dispatch program artifacts",
            "one record for each parsed dispatch_program.json", len(programs), program_sources,
        ))
    if mapping_count:
        observations.extend([
            observation("callable_mapping_count", mapping_count, "mappings", "archived callable name maps",
                        "count all parsed callable_id_to_name entries", mapping_count, name_sources),
            observation("callable_mappings_by_source", dict(sorted(mappings_by_source.items())), "mappings", "source artifact",
                        "count all parsed entries separately for each indexed name map", mapping_count, name_sources),
            observation("candidate_source_location_mapping", {"validated": 0, "unverified": mapping_count}, "mappings",
                        "Candidate source attribution", "all callable entries remain unverified because no Candidate source manifest join is available",
                        mapping_count, name_sources),
        ])
    if config_count:
        observations.append(observation(
            "kernel_config_artifact_count", config_count, "files", "indexed compiler configuration artifacts",
            "count each readable kernel_config.py artifact", config_count, config_sources,
        ))
    details.sort()
    return make_result(
        archive, "Compute", "compute_mapping_check", _MAPPING_FAMILIES, observations,
        ["Compiler program and callable names do not establish source locations in the selected Candidate snapshot."],
        records=records, detail_records=details,
    )


def compute_unit_activity(
    archive: ProfileArchive, *, include_details: bool = True, detail_cursor: tuple[Any, ...] | None = None, detail_limit: int | None = None,
) -> dict[str, Any]:
    sources = family_sources(archive, ["pmu.csv"])
    records: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    detail_records = DetailRecords(
        include_details, limit=detail_limit, after=detail_cursor,
        key=lambda row: (row["source_ref"]["path"], row["record"].get("core_id", ""), row["record"].get("task_id", ""),
                         row["record"].get("func_id", ""), row["record_index"]),
    )
    expected = {"core_id", "task_id", "func_id", "core_type", *PMU_COUNTERS}

    for source in sources:
        if source.status != "present":
            continue
        build, rank, dispatch = _path_scope(source.path)
        source_grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        source_details = detail_records.fork()
        source_issues = IssueSamples()
        semantic_issues = IssueSamples()
        parsed_count = 0

        def consume(index: int, row: dict[str, str]) -> None:
            nonlocal parsed_count
            try:
                core_id = int(row["core_id"])
                core_type = row["core_type"].strip()
                counters = {key: int(row[key]) for key in PMU_COUNTERS}
                if core_id < 0 or not core_type or any(value < 0 for value in counters.values()):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                semantic_issues.add(f"row {index}: invalid core identity or nonnegative integer counter")
                return
            parsed_count += 1
            group_key = (build, rank, dispatch, core_type)
            group = source_grouped.setdefault(group_key, {"records": 0, "sources": {}, "cores": {}})
            group["records"] += 1
            group["sources"][source.path] = source
            core = group["cores"].setdefault(core_id, {"records": 0, "counters": dict.fromkeys(PMU_COUNTERS, 0)})
            core["records"] += 1
            for name, value in counters.items():
                core["counters"][name] += value
            if include_details:
                source_details.append(
                    lambda: {"source_ref": source.source(), "record_index": index, "record": row},
                    sort_key=(source.path, row["core_id"], row["task_id"], row["func_id"], index),
                )
            else:
                source_details.count_only()

        scan = csv_records(source, expected, consume)
        if scan.fatal_error:
            add_records(records, "pmu", rejected=scan.rejected, issues=scan.issues, source=source)
            continue
        source_issues.extend(scan.issues)
        rejected_count = scan.rejected
        if scan.missing_columns:
            source_issues.add("PMU CSV is missing one or more required identity or counter columns")
            rejected_count += scan.parsed
            add_records(records, "pmu", seen=scan.seen, rejected=rejected_count, issues=source_issues, source=source)
            continue
        source_issues.extend(semantic_issues)
        rejected_count += semantic_issues.count
        add_records(records, "pmu", seen=scan.seen, parsed=parsed_count, rejected=rejected_count,
                    issues=source_issues, source=source)
        for group_key, source_group in source_grouped.items():
            group = grouped.setdefault(group_key, {"records": 0, "sources": {}, "cores": {}})
            group["records"] += source_group["records"]
            group["sources"].update(source_group["sources"])
            for core_id, source_core in source_group["cores"].items():
                core = group["cores"].setdefault(core_id, {"records": 0, "counters": dict.fromkeys(PMU_COUNTERS, 0)})
                core["records"] += source_core["records"]
                for metric in PMU_COUNTERS:
                    core["counters"][metric] += source_core["counters"][metric]
        detail_records = source_details

    observations = []
    for (build, rank, dispatch, core_type), group in sorted(grouped.items()):
        core_totals = {
            str(core_id): {"record_count": item["records"], "counters": item["counters"]}
            for core_id, item in sorted(group["cores"].items())
        }
        totals = {
            metric: sum(core["counters"][metric] for core in group["cores"].values())
            for metric in PMU_COUNTERS
        }
        sources_for_group = [group["sources"][path] for path in sorted(group["sources"])]
        scope = "build/rank/dispatch/core_type"
        aggregate = "sum all valid PMU task rows in this build/rank/dispatch/core_type; per-core counts are nested without cross-scope merging"
        for metric, unit in PMU_COUNTERS.items():
            observations.append(observation(
                f"{metric}_sum", totals[metric], unit, scope, aggregate, group["records"], sources_for_group,
                build_path=build, rank=rank, dispatch=dispatch, core_type=core_type,
                core_count=len(core_totals), per_core=core_totals,
            ))
        observations.append(observation(
            "pmu_record_count", group["records"], "records", scope,
            "count all valid PMU rows in this group", group["records"], sources_for_group,
            build_path=build, rank=rank, dispatch=dispatch, core_type=core_type,
            core_count=len(core_totals), per_core={key: value["record_count"] for key, value in core_totals.items()},
        ))
    detail_records.sort()
    return make_result(
        archive, "Compute", "compute_unit_activity", _PMU_FAMILY, observations,
        ["PMU counters are raw event counts; there is no calibrated runtime denominator for utilization.",
         "Build, rank, dispatch, and core-type scopes remain separate; counters with different units are not combined."],
        records=records, detail_records=detail_records,
    )


def compute_task_share(
    archive: ProfileArchive, *, include_details: bool = True, detail_cursor: tuple[Any, ...] | None = None, detail_limit: int | None = None,
) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    details = DetailRecords(
        include_details, limit=detail_limit, after=detail_cursor,
        key=lambda row: (row.get("build_path", ""), row.get("rank") or "", row.get("dispatch") or "",
                         str(row.get("task_id", row.get("record_index", ""))), row["record_type"]),
    )
    observations = []
    coverage_rows = []
    pmu_sources = family_sources(archive, ["pmu.csv"])
    all_deps = family_sources(archive, ["deps.json"])
    all_swimlane = family_sources(archive, ["chip_swimlane_records.json"])
    all_names = family_sources(archive, ["name_map*.json"])
    spans_by_scope = {_path_scope(source.path): source for source in all_swimlane}
    deps_by_scope = {_path_scope(source.path): source for source in all_deps}
    names_by_scope: dict[tuple[str, str, str], list[ArtifactRead]] = defaultdict(list)
    for source in all_names:
        names_by_scope[_path_scope(source.path)].append(source)

    for pmu in pmu_sources:
        if pmu.status != "present":
            continue
        scope = _path_scope(pmu.path)
        deps = deps_by_scope.get(scope)
        swimlane = spans_by_scope.get(scope)
        name_matches = names_by_scope.get(scope, [])
        if deps is None or swimlane is None or len(name_matches) != 1 or not scope[1] or not scope[2]:
            reason = "missing or ambiguous same-build/rank/dispatch companion source"
            if not scope[1] or not scope[2]:
                reason = "rank or dispatch identity is absent from the indexed artifact path"
            if _record_unjoined_pmu_source(pmu, stats, coverage_rows, scope, reason):
                add_records(stats, "runtime_swimlane", seen=0, issues=[reason], source=pmu)
                add_records(stats, "dependency_graph", seen=0, issues=[reason], source=pmu)
                add_records(stats, "callable_name_map", seen=0, issues=[reason], source=pmu)
            continue
        name_map = name_matches[0]
        graph, graph_seen, graph_error = json_value(deps)
        timeline, timeline_seen, timeline_error = json_value(swimlane)
        names, names_seen, names_error = json_value(name_map)
        task_list = graph.get("tasks") if isinstance(graph, dict) else None
        swimlane_rows = timeline.get("aicore_tasks") if isinstance(timeline, dict) else None
        callable_names = names.get("callable_id_to_name") if isinstance(names, dict) else None
        invalid_sources = [error for error in (graph_error, timeline_error, names_error) if error]
        if not isinstance(task_list, list):
            invalid_sources.append("dependency graph has no tasks array")
        if not isinstance(swimlane_rows, list):
            invalid_sources.append("swimlane has no AICore task array")
        if not isinstance(callable_names, dict):
            invalid_sources.append("name map has no callable_id_to_name object")
        for family, seen, error, source in (
            ("dependency_graph", graph_seen, "dependency graph has no tasks array" if not isinstance(task_list, list) else graph_error, deps),
            ("runtime_swimlane", timeline_seen, "swimlane has no AICore task array" if not isinstance(swimlane_rows, list) else timeline_error, swimlane),
            ("callable_name_map", names_seen, "name map has no callable_id_to_name object" if not isinstance(callable_names, dict) else names_error, name_map),
        ):
            add_records(stats, family, seen=seen, parsed=0 if error else seen, rejected=1 if error and seen else 0,
                        issues=[error] if error else (), source=source)
        if invalid_sources:
            reason = "; ".join(invalid_sources)
            _record_unjoined_pmu_source(pmu, stats, coverage_rows, scope, reason)
            continue

        try:
            task_by_id = {}
            for task in task_list:
                if not isinstance(task, dict) or "task_id" not in task:
                    raise ValueError("dependency task has no task_id")
                task_id = _normalize_task_id(task["task_id"])
                if task_id in task_by_id:
                    raise ValueError("duplicate dependency task_id")
                task_by_id[task_id] = task
            spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for row in swimlane_rows:
                if not isinstance(row, list) or len(row) < 6:
                    raise ValueError("swimlane tuple does not match the audited task-join shape")
                task_id = _normalize_task_id(row[1])
                start, end = int(row[3]), int(row[4])
                if end < start:
                    raise ValueError("swimlane task end precedes start")
                spans[task_id].append((start, end))
            normalized_names = {int(key): value for key, value in callable_names.items()}
            if any(key < 0 or not isinstance(value, str) for key, value in normalized_names.items()):
                raise ValueError("callable name map contains an invalid entry")
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc)
            if _record_unjoined_pmu_source(pmu, stats, coverage_rows, scope, reason):
                add_records(stats, "runtime_swimlane", issues=[reason], source=swimlane)
                add_records(stats, "dependency_graph", issues=[reason], source=deps)
                add_records(stats, "callable_name_map", issues=[reason], source=name_map)
            continue

        grouped: dict[int, dict[str, Any]] = {}
        kernel_ids_by_task: dict[int, set[int] | None] = {}
        unjoined = 0
        reasons: dict[str, int] = defaultdict(int)
        valid_rows = 0
        source_details = details.fork()

        def record_unjoined_pmu_row(index: int, row: dict[str, str], reason: str) -> None:
            nonlocal unjoined
            unjoined += 1
            reasons[reason] += 1
            if include_details:
                source_details.append(
                    lambda reason=reason: {
                        "record_type": "unjoined_pmu_row", "source_ref": pmu.source(),
                        "record_index": index, "reason": reason, "record": row,
                    },
                    sort_key=("", "", "", str(index), "unjoined_pmu_row"),
                )
            else:
                source_details.count_only()

        def consume(index: int, row: dict[str, str]) -> None:
            nonlocal unjoined, valid_rows
            try:
                task_id = _normalize_task_id(row["task_id"])
                function_id = int(row["func_id"])
                counters = {name: int(row[name]) for name in PMU_COUNTERS}
                task = task_by_id.get(task_id)
                if task is None:
                    active_ids = set()
                else:
                    if task_id not in kernel_ids_by_task:
                        try:
                            active_ids = set()
                            for value in task.get("kernel_ids", []):
                                kernel_id = int(value)
                                if kernel_id >= 0:
                                    active_ids.add(kernel_id)
                        except (TypeError, ValueError):
                            active_ids = None
                        kernel_ids_by_task[task_id] = active_ids
                    active_ids = kernel_ids_by_task[task_id]
                    if active_ids is None:
                        raise ValueError("invalid task kernel IDs")
                if task is None or task_id not in spans:
                    raise LookupError("PMU task has no dependency and AICore span match")
                if function_id not in active_ids or function_id not in normalized_names:
                    raise LookupError("PMU function ID does not match task kernel IDs and callable map")
                if any(value < 0 for value in counters.values()):
                    raise ValueError("negative PMU counter")
            except LookupError as exc:
                record_unjoined_pmu_row(index, row, str(exc))
                return
            except (KeyError, TypeError, ValueError):
                record_unjoined_pmu_row(index, row, "invalid PMU task identity or counter")
                return
            valid_rows += 1
            task_item = grouped.setdefault(task_id, {"record_count": 0, "function_ids": set(), "counters": dict.fromkeys(PMU_COUNTERS, 0)})
            task_item["record_count"] += 1
            task_item["function_ids"].add(function_id)
            for metric, value in counters.items():
                task_item["counters"][metric] += value

        scan = csv_records(pmu, {"task_id", "func_id", *PMU_COUNTERS}, consume)
        if scan.fatal_error:
            add_records(stats, "pmu", rejected=scan.rejected, issues=scan.issues, source=pmu)
            coverage_rows.append({"source": pmu, "scope": scope, "joined": 0, "unjoined": 0,
                                  "reason_counts": {"invalid CSV": scan.rejected}})
            continue
        if scan.missing_columns:
            issue = "PMU CSV is missing task identity or counter columns"
            scan.issues.add(issue)
            add_records(stats, "pmu", seen=scan.seen, rejected=scan.seen, issues=scan.issues, source=pmu)
            coverage_rows.append({"source": pmu, "scope": scope, "joined": 0, "unjoined": scan.seen, "reason": issue})
            continue
        add_records(stats, "pmu", seen=scan.seen, parsed=scan.parsed, rejected=scan.rejected,
                    issues=scan.issues, source=pmu)
        details = source_details
        coverage_rows.append({"source": pmu, "scope": scope, "joined": valid_rows, "unjoined": unjoined, "reason_counts": dict(sorted(reasons.items()))})
        for task_id, item in sorted(grouped.items()):
            task_spans = spans[task_id]
            span = max(end for _, end in task_spans) - min(start for start, _ in task_spans)
            if include_details:
                details.append(lambda: {
                    "record_type": "joined_task",
                    "source_refs": [pmu.source(), deps.source(), swimlane.source(), name_map.source()],
                    "build_path": scope[0], "rank": scope[1], "dispatch": scope[2], "task_id": str(task_id),
                    "function_ids": sorted(item["function_ids"]),
                    "function_names": [normalized_names[key] for key in sorted(item["function_ids"])],
                    "pmu_record_count": item["record_count"], "pmu_counter_sums": item["counters"],
                    "aicore_record_count": len(task_spans), "aicore_span": span,
                }, sort_key=(scope[0], scope[1], scope[2], str(task_id), "joined_task"))
            else:
                details.count_only()
        if grouped:
            source_set = [pmu, deps, swimlane, name_map]
            observations.append(observation(
                "joined_task_count", len(grouped), "tasks", "build/rank/dispatch",
                "count distinct task IDs with at least one PMU row matched to dependency kernel IDs, callable names, and AICore records",
                len(grouped), source_set, build_path=scope[0], rank=scope[1], dispatch=scope[2],
            ))
            observations.append(observation(
                "joined_pmu_record_count", valid_rows, "records", "build/rank/dispatch",
                "count PMU rows that passed the same-dispatch task/function join", scan.seen, source_set,
                build_path=scope[0], rank=scope[1], dispatch=scope[2], unjoined_record_count=unjoined,
            ))
            for metric, unit in PMU_COUNTERS.items():
                observations.append(observation(
                    f"{metric}_sum", sum(task["counters"][metric] for task in grouped.values()), unit,
                    "build/rank/dispatch", "sum matched PMU rows for all joined tasks in this dispatch",
                    valid_rows, [pmu, deps, swimlane, name_map], build_path=scope[0], rank=scope[1], dispatch=scope[2],
                ))
            spans = [max(end for _, end in spans[task_id]) - min(start for start, _ in spans[task_id]) for task_id in grouped]
            observations.append(observation(
                "aicore_task_span_distribution", {"min": min(spans), "max": max(spans), "sum": sum(spans)},
                "swimlane_timestamp_ticks", "build/rank/dispatch",
                "min, max, and sum of max(end)-min(start) for each joined task; separate from PMU counts",
                len(spans), [pmu, deps, swimlane, name_map], build_path=scope[0], rank=scope[1], dispatch=scope[2],
            ))
    observations.extend(_unjoined_observations(coverage_rows))
    details.sort()
    return make_result(
        archive, "Compute", "compute_task_share", _TASK_FAMILIES, observations,
        ["Task joins are restricted to one indexed build/rank/dispatch scope and require matching PMU task/function IDs, dependency kernel IDs, callable names, and AICore task IDs.",
         "PMU counters and AICore timestamp spans have different units and are reported separately; the span is an envelope that may include gaps."],
        records=stats, detail_records=details,
    )


def _unjoined_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    sources = [row["source"] for row in rows]
    return [observation(
        "unjoined_pmu_record_count", sum(row["unjoined"] for row in rows), "records", "indexed PMU dispatch sources",
        "sum PMU rows without a verified same-dispatch join; reasons are listed by source coverage issues",
        sum(row["unjoined"] for row in rows), sources,
        by_scope=[{"build_path": row["scope"][0], "rank": row["scope"][1], "dispatch": row["scope"][2],
                   "unjoined_records": row["unjoined"], "reason": row.get("reason"), "reason_counts": row.get("reason_counts", {})}
                  for row in rows],
    )]


def _record_unjoined_pmu_source(
    pmu: ArtifactRead,
    stats: dict[str, dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    scope: tuple[str, str, str],
    reason: str,
) -> bool:
    """Record one PMU source that cannot produce a verified task join.

    A valid CSV is still counted as parsed and unjoined; malformed or missing
    identity columns stop before companion-source diagnostics are duplicated.
    """
    scan = csv_records(pmu, {"task_id", "func_id", *PMU_COUNTERS})
    if scan.fatal_error:
        add_records(stats, "pmu", rejected=scan.rejected, issues=scan.issues, source=pmu)
        return False
    if scan.missing_columns:
        issue = "PMU CSV is missing task identity or counter columns"
        scan.issues.add(issue)
        add_records(stats, "pmu", seen=scan.seen, rejected=scan.seen, issues=scan.issues, source=pmu)
        coverage_rows.append({"source": pmu, "scope": scope, "joined": 0, "unjoined": scan.seen, "reason": issue})
        return False
    add_records(stats, "pmu", seen=scan.seen, parsed=scan.parsed, rejected=scan.rejected,
                issues=scan.issues, source=pmu)
    coverage_rows.append({"source": pmu, "scope": scope, "joined": 0, "unjoined": scan.parsed, "reason": reason})
    return True


def _path_scope(path: str) -> tuple[str, str, str]:
    build, _, tail = path.partition("/dfx_outputs/")
    rank = re.search(r"(?:^|/)(rank\d+)(?:/|$)", tail)
    dispatch = re.search(r"(?:^|/)d(\d+)(?:/|$)", tail)
    return build or "unknown", rank.group(1) if rank else "unknown", dispatch.group(1) if dispatch else "unknown"


def _normalize_task_id(value: Any) -> int:
    task_id = int(value, 0) if isinstance(value, str) and value.lower().startswith("0x") else int(value)
    return task_id & ((1 << 64) - 1) if task_id < 0 else task_id
