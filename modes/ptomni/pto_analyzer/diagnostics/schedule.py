from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

from ..profile import ArtifactRead, ProfileArchive
from .common import DetailRecords, add_records, family_sources, json_array_count, json_value, make_result, observation

_GRAPH_FAMILIES = [("dependency_graph", ["deps.json"], True)]
_GAP_FAMILIES = [
    ("runtime_swimlane", ["chip_swimlane_records.json"], True),
    ("merged_swimlane", ["merged_swimlane*.json"], True),
    ("dependency_graph", ["deps.json"], True),
]
_DISTRIBUTION_FAMILIES = [("runtime_swimlane", ["chip_swimlane_records.json"], True)]


def dependency_path_check(
    archive: ProfileArchive, *, include_details: bool = True, detail_cursor: tuple[Any, ...] | None = None, detail_limit: int | None = None,
) -> dict[str, Any]:
    sources = family_sources(archive, ["deps.json"])
    stats: dict[str, dict[str, Any]] = {}
    observations = []
    details = DetailRecords(
        include_details, limit=detail_limit, after=detail_cursor,
        key=lambda row: (row["source_ref"]["path"], row["record_type"], row.get("task_id", ""),
                         row.get("pred", ""), row.get("succ", "")),
    )
    for source in sources:
        if source.status != "present":
            continue
        graph, seen, error = json_value(source)
        if error:
            add_records(stats, "dependency_graph", seen=seen, rejected=seen,
                        issues=[error], source=source)
            continue
        if not isinstance(graph, dict):
            add_records(stats, "dependency_graph", seen=seen, rejected=seen,
                        issues=["dependency graph root is not an object"], source=source)
            continue
        tasks = graph.get("tasks")
        edges = graph.get("edges")
        if not isinstance(tasks, list) or not isinstance(edges, list):
            add_records(stats, "dependency_graph", seen=1, rejected=1,
                        issues=["dependency graph must contain task and edge arrays"], source=source)
            continue
        parsed_tasks: set[str] = set()
        parsed_edges: set[tuple[str, str]] = set()
        edge_record_count = 0
        issues = []
        task_rejected = 0
        edge_rejected = 0
        for index, task in enumerate(tasks):
            if not isinstance(task, dict) or task.get("task_id") is None:
                task_rejected += 1
                issues.append(f"task record {index}: missing task_id")
                tasks[index] = None
                continue
            task_id = str(task["task_id"])
            if task_id in parsed_tasks:
                task_rejected += 1
                issues.append(f"task record {index}: duplicate task_id {task_id}")
                tasks[index] = None
                continue
            parsed_tasks.add(task_id)
            if include_details:
                details.append(
                    lambda: {"record_type": "task", "source_ref": source.source(), "task_id": task_id, "record": task},
                    sort_key=(source.path, "task", task_id, "", ""),
                )
            else:
                details.count_only()
            tasks[index] = None
        if tasks:
            del task
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict) or edge.get("pred") is None or edge.get("succ") is None:
                edge_rejected += 1
                issues.append(f"edge record {index}: missing pred or succ endpoint")
                edges[index] = None
                continue
            pair = (str(edge["pred"]), str(edge["succ"]))
            parsed_edges.add(pair)
            edge_record_count += 1
            if include_details:
                details.append(
                    lambda: {"record_type": "edge", "source_ref": source.source(), "pred": pair[0], "succ": pair[1], "record": edge},
                    sort_key=(source.path, "edge", "", pair[0], pair[1]),
                )
            else:
                details.count_only()
            edges[index] = None
        if edges:
            del edge
        add_records(stats, "dependency_graph", seen=len(tasks) + len(edges),
                    parsed=len(parsed_tasks) + edge_record_count, rejected=task_rejected + edge_rejected,
                    issues=issues, source=source)
        del graph, tasks, edges
        facts = _graph_observations(source, parsed_tasks, parsed_edges, edge_record_count)
        observations.extend(facts)
    details.sort()
    return make_result(
        archive, "Schedule", "dependency_path_check", _GRAPH_FAMILIES, observations,
        ["Graph measurements describe the indexed dependency structure only; no runtime weights or latency are attached.",
         "Duplicate edge records are counted separately from unique task-to-task edges, and endpoints absent from task metadata remain explicit."],
        records=stats, detail_records=details,
    )


def dispatch_gap_check(
    archive: ProfileArchive, *, include_details: bool = True, detail_cursor: tuple[Any, ...] | None = None, detail_limit: int | None = None,
) -> dict[str, Any]:
    raw_sources = family_sources(archive, ["chip_swimlane_records.json"])
    merged_sources = family_sources(archive, ["merged_swimlane*.json"])
    graph_sources = family_sources(archive, ["deps.json"])
    stats: dict[str, dict[str, Any]] = {}
    observations = []
    details = DetailRecords(
        include_details, limit=detail_limit, after=detail_cursor,
        key=lambda row: (row["source_ref"]["path"], row["record_array"], row["record_index"]),
    )
    for source in raw_sources:
        if source.status != "present":
            continue
        value, seen, error = json_value(source)
        if error or not isinstance(value, dict):
            add_records(stats, "runtime_swimlane", seen=seen, rejected=seen,
                        issues=[error or "swimlane root is not an object"], source=source)
            continue
        arrays = _swimlane_arrays(value)
        intervals, rejected, issues, clock_freq_hz = _aicore_intervals(value)
        if not arrays:
            add_records(stats, "runtime_swimlane", seen=1, rejected=1,
                        issues=["swimlane contains no recognized record arrays", *issues], source=source)
            continue
        row_count = sum(len(rows) for _, rows in arrays)
        aicore_rows = value.get("aicore_tasks")
        aicore_count = len(aicore_rows) if isinstance(aicore_rows, list) else 0
        add_records(stats, "runtime_swimlane", seen=row_count, parsed=row_count - aicore_count + len(intervals),
                    rejected=rejected, issues=issues, source=source)
        rank = _rank(source.path)
        observations.append(observation(
            "raw_swimlane_record_count", row_count, "records", f"rank={rank or 'unavailable'}; source={source.path}",
            "count rows in named swimlane arrays without interpreting tuple positions", row_count, [source],
            capture_level=value.get("chip_swimlane_level"),
        ))
        for array_name, rows in arrays:
            for index, row in enumerate(rows):
                if include_details:
                    details.append(
                        lambda: {"source_ref": source.source(), "rank": rank, "record_array": array_name,
                                 "record_index": index, "record": row},
                        sort_key=(source.path, array_name, index),
                    )
                else:
                    details.count_only()
        complete_intervals, completeness_issues = _complete_aicore_intervals(value, intervals, issues)
        if completeness_issues:
            add_records(stats, "runtime_swimlane", issues=completeness_issues, source=source)
        intervals_by_core: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for core, _, _, start, end in complete_intervals:
            intervals_by_core[core].append((start, end))
        per_core_gaps: dict[str, dict[str, int]] = {}
        gap_sample_count = 0
        for core_id, intervals_for_core in sorted(intervals_by_core.items()):
            core_intervals = sorted(intervals_for_core)
            gaps = [right[0] - left[1] for left, right in zip(core_intervals, core_intervals[1:])]
            positive_gaps = [gap for gap in gaps if gap > 0]
            if positive_gaps:
                per_core_gaps[str(core_id)] = {
                    "count": len(positive_gaps), "min": min(positive_gaps), "max": max(positive_gaps), "sum": sum(positive_gaps),
                }
                gap_sample_count += len(positive_gaps)
        if per_core_gaps:
            observations.append(observation(
                "aicore_observed_gap_cycles", per_core_gaps, "cycles", "build/rank/dispatch/core",
                "for each core, sort valid AICore intervals by start cycle and summarize positive next.start - previous.end gaps",
                gap_sample_count, [source], build_path=_build(source.path), rank=rank, dispatch=_dispatch(source.path),
                clock_freq_hz=clock_freq_hz,
            ))
    for source in merged_sources:
        if source.status != "present":
            continue
        count, seen, error = json_array_count(source, "traceEvents")
        if error or count is None:
            add_records(stats, "merged_swimlane", seen=seen, rejected=seen,
                        issues=[error or "merged trace has no traceEvents array"], source=source)
            continue
        add_records(stats, "merged_swimlane", seen=count, parsed=count, source=source)
        observations.append(observation(
            "merged_trace_event_count", count, "events", f"merged artifact {source.path}",
            "count archived traceEvents without assuming they match raw tuple rows", count, [source],
        ))
    for source in graph_sources:
        if source.status != "present":
            continue
        graph, seen, error = json_value(source)
        tasks = graph.get("tasks") if isinstance(graph, dict) else None
        edges = graph.get("edges") if isinstance(graph, dict) else None
        if error or not isinstance(tasks, list) or not isinstance(edges, list):
            add_records(stats, "dependency_graph", seen=seen, rejected=seen,
                        issues=[error or "dependency graph lacks task or edge arrays"], source=source)
            continue
        add_records(stats, "dependency_graph", seen=len(tasks) + len(edges), parsed=len(tasks) + len(edges), source=source)
    details.sort()
    return make_result(
        archive, "Schedule", "dispatch_gap_check", _GAP_FAMILIES, observations,
        ["Observed AICore interval gaps are not attributed to scheduling: producer readiness, dispatch, pickup, core occupancy, and scheduler phase evidence are not joined here.",
         "Raw and merged swimlane coverage are reported separately; their rows are not assumed to have equal or uniquely joined task/block coverage.",
         "Intervals are compared only within one indexed artifact's rank/dispatch/core scope; gaps are in recorded clock cycles."],
        records=stats, detail_records=details,
    )


def task_distribution_check(
    archive: ProfileArchive, *, include_details: bool = True, detail_cursor: tuple[Any, ...] | None = None, detail_limit: int | None = None,
) -> dict[str, Any]:
    sources = family_sources(archive, ["chip_swimlane_records.json"])
    stats: dict[str, dict[str, Any]] = {}
    observations = []
    details = DetailRecords(
        include_details, limit=detail_limit, after=detail_cursor,
        key=lambda row: (row["source_ref"]["path"], row["record_array"], row["record_index"]),
    )
    for source in sources:
        if source.status != "present":
            continue
        value, seen, error = json_value(source)
        if error or not isinstance(value, dict):
            add_records(stats, "runtime_swimlane", seen=seen, rejected=seen,
                        issues=[error or "swimlane root is not an object"], source=source)
            continue
        arrays = _swimlane_arrays(value)
        intervals, rejected, issues, clock_freq_hz = _aicore_intervals(value)
        if not arrays:
            add_records(stats, "runtime_swimlane", seen=1, rejected=1,
                        issues=["swimlane contains no recognized record arrays", *issues], source=source)
            continue
        count = sum(len(rows) for _, rows in arrays)
        aicore_rows = value.get("aicore_tasks")
        aicore_count = len(aicore_rows) if isinstance(aicore_rows, list) else 0
        add_records(stats, "runtime_swimlane", seen=count, parsed=count - aicore_count + len(intervals),
                    rejected=rejected, issues=issues, source=source)
        rank = _rank(source.path)
        observations.append(observation(
            "swimlane_tuple_record_count", count, "records", f"rank={rank or 'unavailable'}; source={source.path}",
            "count every row in the indexed named swimlane arrays; tuple fields are left uninterpreted", count, [source],
            aicore_record_count=len(value.get("aicore_tasks", [])) if isinstance(value.get("aicore_tasks"), list) else None,
            aicpu_record_count=len(value.get("aicpu_tasks", [])) if isinstance(value.get("aicpu_tasks"), list) else None,
            capture_level=value.get("chip_swimlane_level"),
        ))
        for array_name, rows in arrays:
            for index, row in enumerate(rows):
                if include_details:
                    details.append(
                        lambda: {"source_ref": source.source(), "rank": rank, "record_array": array_name,
                                 "record_index": index, "record": row},
                        sort_key=(source.path, array_name, index),
                    )
                else:
                    details.count_only()
        if intervals:
            by_core: dict[int, dict[str, Any]] = {}
            for core_id, task_token, _, start, end in intervals:
                item = by_core.setdefault(core_id, {"record_count": 0, "task_tokens": set(),
                                                    "duration_min": None, "duration_max": None, "duration_sum": 0})
                duration = end - start
                item["record_count"] += 1
                item["task_tokens"].add(task_token)
                item["duration_min"] = duration if item["duration_min"] is None else min(item["duration_min"], duration)
                item["duration_max"] = duration if item["duration_max"] is None else max(item["duration_max"], duration)
                item["duration_sum"] += duration
            scope = "build/rank/dispatch/core"
            identity = {"build_path": _build(source.path), "rank": rank, "dispatch": _dispatch(source.path)}
            observations.extend([
                observation("aicore_record_count_by_core", {str(core): item["record_count"] for core, item in sorted(by_core.items())},
                           "records", scope, "count valid AICore physical tuple rows separately for each core", len(intervals), [source], **identity),
                observation("unique_task_token_count_by_core", {str(core): len(item["task_tokens"]) for core, item in sorted(by_core.items())},
                           "task tokens", scope, "count distinct raw task-token values separately for each core; not a count of logical dependency tasks", len(intervals), [source], **identity),
                observation("aicore_interval_duration_cycles_by_core",
                           {str(core): {"count": item["record_count"], "min": item["duration_min"],
                                        "max": item["duration_max"], "sum": item["duration_sum"]}
                            for core, item in sorted(by_core.items())},
                           "cycles", scope, "summarize end_cycle - start_cycle for every valid AICore tuple, separately by core; sums are not busy time", len(intervals), [source],
                           clock_freq_hz=clock_freq_hz,
                           core_type_by_id={str(core): value["metadata"]["core_types"][core] for core in sorted(by_core)}, **identity),
            ])
    details.sort()
    return make_result(
        archive, "Schedule", "task_distribution_check", _DISTRIBUTION_FAMILIES, observations,
        ["Core ID, raw task token, and AICore start/end cycles use the audited tuple contract and per-file clock metadata; incomplete or invalid rows remain in source coverage issues.",
         "Tuple row count is distinct from unique raw task-token count. Interval-duration sums are not busy time; no occupancy or load-balance statistic is inferred."],
        records=stats, detail_records=details,
    )


def _graph_observations(
    source: ArtifactRead,
    tasks: set[str],
    unique_edges: set[tuple[str, str]],
    edge_record_count: int,
) -> list[dict[str, Any]]:
    task_count = len(tasks)
    nodes = set(tasks)
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    parent: dict[str, str] = {}
    component_sizes: dict[str, int] = {}

    def add_node(node: str) -> None:
        if node not in parent:
            parent[node] = node
            component_sizes[node] = 1

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            ancestor = parent[node]
            parent[node] = root
            node = ancestor
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if component_sizes[left_root] < component_sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_sizes[left_root] += component_sizes.pop(right_root)

    for node in nodes:
        add_node(node)
    for pred, succ in unique_edges:
        nodes.add(pred)
        nodes.add(succ)
        add_node(pred)
        add_node(succ)
        outgoing[pred].append(succ)
        indegree[succ] += 1
        union(pred, succ)

    component_count = len(component_sizes)
    maximum_fan_in = max(indegree.values(), default=0)
    for node in nodes:
        indegree.setdefault(node, 0)
    queue = deque(sorted(node for node in nodes if indegree[node] == 0))
    depth = {node: 0 for node in nodes}
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for succ in outgoing[node]:
            depth[succ] = max(depth[succ], depth[node] + 1)
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)
    prefix = {"build_path": _build(source.path), "rank": _rank(source.path), "dispatch": _dispatch(source.path)}
    result = [
        observation("dependency_task_metadata_count", task_count, "tasks", "build/rank/dispatch graph", "count valid task metadata records", task_count, [source], **prefix),
        observation("dependency_node_count", len(nodes), "nodes", "build/rank/dispatch graph", "union task metadata IDs and valid edge endpoints", len(nodes), [source], **prefix),
        observation("dependency_edge_record_count", edge_record_count, "edge records", "build/rank/dispatch graph", "count valid edge records before duplicate removal", edge_record_count, [source], **prefix),
        observation("dependency_unique_edge_count", len(unique_edges), "edges", "build/rank/dispatch graph", "count unique directed predecessor/successor pairs", len(unique_edges), [source], **prefix),
        observation("dependency_duplicate_edge_record_count", edge_record_count - len(unique_edges), "edge records", "build/rank/dispatch graph", "valid edge records minus unique directed pairs", edge_record_count, [source], **prefix),
        observation("dependency_endpoint_without_task_metadata_count", len(nodes - tasks), "nodes", "build/rank/dispatch graph", "count edge endpoints absent from task metadata", len(nodes), [source], **prefix),
        observation("dependency_weak_component_count", component_count, "components", "build/rank/dispatch graph", "count connected components after treating each directed edge as undirected", len(nodes), [source], **prefix),
        observation("dependency_maximum_fan_in", maximum_fan_in, "predecessors", "build/rank/dispatch graph", "maximum unique incoming neighbors for any graph node", len(nodes), [source], **prefix),
        observation("dependency_maximum_fan_out", max((len(value) for value in outgoing.values()), default=0), "successors", "build/rank/dispatch graph", "maximum unique outgoing neighbors for any graph node", len(nodes), [source], **prefix),
    ]
    if visited == len(nodes):
        result.append(observation("dependency_graph_edge_path_depth", max(depth.values(), default=0), "edges", "build/rank/dispatch graph",
                                  "maximum number of edges in an unweighted directed acyclic path", len(nodes), [source], **prefix))
    else:
        result.append(observation("dependency_graph_cycle_present", True, "boolean", "build/rank/dispatch graph",
                                  "Kahn topological traversal visited fewer nodes than the graph contains", len(nodes), [source], **prefix))
    return result


def _swimlane_arrays(value: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    names = ("aicore_tasks", "aicpu_tasks", "aicpu_scheduler_phases", "aicpu_orchestrator_phases")
    return [(name, value[name]) for name in names if isinstance(value.get(name), list)]


def _aicore_intervals(value: dict[str, Any]) -> tuple[list[tuple[int, str, int, int, int]], int, list[str], int | None]:
    if "aicore_tasks" not in value:
        return [], 0, ["aicore_tasks is missing"], None
    rows = value["aicore_tasks"]
    if not isinstance(rows, list):
        return [], 0, ["aicore_tasks is not an array"], None
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        return [], len(rows), ["AICore tuples lack clock and core metadata"], None
    clock = metadata.get("clock_freq_hz")
    core_count = metadata.get("num_cores")
    core_types = metadata.get("core_types")
    if (type(clock) is not int or clock <= 0 or type(core_count) is not int or core_count <= 0
            or not isinstance(core_types, list) or len(core_types) != core_count):
        return [], len(rows), ["AICore tuples lack valid clock frequency, core count, or core type metadata"], None
    parsed = []
    issues = []
    seen_keys = set()
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) not in {5, 6}:
            issues.append(f"AICore tuple {index}: expected five or six fields")
            continue
        core, token, reg_task_id, start, end = row[0], row[1], row[2], row[3], row[4]
        if (type(core) is not int or not 0 <= core < core_count or token is None
                or type(reg_task_id) is not int or reg_task_id < 0
                or type(start) is not int or type(end) is not int or start <= 0 or end < start):
            issues.append(f"AICore tuple {index}: invalid core, task token, or interval cycles")
            continue
        key = core, reg_task_id
        if key in seen_keys:
            issues.append(f"AICore tuple {index}: duplicate core/register-task identity")
            continue
        seen_keys.add(key)
        parsed.append((core, str(token), reg_task_id, start, end))
    return parsed, len(rows) - len(parsed), issues, clock


def _complete_aicore_intervals(
    value: dict[str, Any],
    intervals: list[tuple[int, str, int, int, int]],
    issues: list[str],
) -> tuple[list[tuple[int, str, int, int, int]], list[str]]:
    if value.get("chip_swimlane_level") != 4:
        return [], ["AICore interval gaps require level-4 swimlane coverage"]
    if issues:
        return [], ["AICore interval coverage contains invalid tuples; no gap distribution is calculated"]
    rows = value.get("aicpu_tasks")
    if not isinstance(rows, list):
        return [], ["AICPU dispatch/finish records are missing"]
    scheduler_keys = set()
    scheduler_issues = []
    metadata = value["metadata"]
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 4:
            scheduler_issues.append(f"AICPU tuple {index}: expected four fields")
            continue
        core, reg_task_id, dispatch, finish = row
        if (type(core) is not int or not 0 <= core < metadata["num_cores"] or type(reg_task_id) is not int
                or reg_task_id < 0 or type(dispatch) is not int or type(finish) is not int
                or dispatch <= 0 or finish < dispatch):
            scheduler_issues.append(f"AICPU tuple {index}: invalid core, task identity, or dispatch/finish cycles")
            continue
        key = core, reg_task_id
        if key in scheduler_keys:
            scheduler_issues.append(f"AICPU tuple {index}: duplicate core/register-task identity")
            continue
        scheduler_keys.add(key)
    aicore_keys = {(core, reg_task_id) for core, _, reg_task_id, _, _ in intervals}
    if scheduler_issues:
        return [], scheduler_issues
    if scheduler_keys != aicore_keys:
        return [], ["AICore and AICPU records do not have complete one-to-one core/register-task coverage"]
    return intervals, []


def _rank(path: str) -> str | None:
    match = re.search(r"(?:^|/)rank(\d+)(?:/|$)", path)
    return match.group(1) if match else None


def _dispatch(path: str) -> str | None:
    match = re.search(r"(?:^|/)d(\d+)(?:/|$)", path)
    return match.group(1) if match else None


def _build(path: str) -> str:
    return path.partition("/dfx_outputs/")[0]
