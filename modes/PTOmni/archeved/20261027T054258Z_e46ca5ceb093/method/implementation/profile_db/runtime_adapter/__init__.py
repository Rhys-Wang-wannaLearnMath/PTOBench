# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Read-only structured adapters for installed runtime analysis functions."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from profile_db.errors import IngestError
from profile_db.runtime_adapter.backend import BackendUnavailable, provenance, require

ANALYSIS_VERSION = "runtime/1"
ADAPTER_VERSION = "pfdb/0.6.0/1"
CP_SIX_COLUMN_HASH = "a6ecb2590456468f3b40ee0245fb5c14122d38db9d7ee6b6cd7232f51f9f0223"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def module(name):
    return require(name)


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, set):
        return sorted(plain(v) for v in value)
    if hasattr(value, "item"):
        return value.item()
    return value


class Calls:
    """Record exact functions used by one analysis family."""

    def __init__(self):
        self.functions = {}
        self.compatibility = []

    def call(self, name, function, *args, **kwargs):
        f = getattr(require(name, function), function)
        self.functions[name + "." + function] = hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
        return f(*args, **kwargs)

    def compatible_call(self, name, function, *args, **kwargs):
        """Omit unsupported empty options; reject loss of meaningful inputs."""
        f = getattr(require(name, function), function)
        signature = inspect.signature(f)
        if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
            for key in list(kwargs):
                if key not in signature.parameters:
                    value = kwargs[key]
                    if value is not None and value != [] and value != {}:
                        raise BackendUnavailable(f"Simpler {name}.{function} does not support {key}")
                    del kwargs[key]
                    self.compatibility.append(f"omit-empty:{name}.{function}.{key}")
        return self.call(name, function, *args, **kwargs)


def snapshot(family, paths, analyze, parameters=None):
    calls = Calls()
    inputs = [{"kind": kind, "sha256": digest(path)} for kind, path in sorted(paths.items())]
    backend = provenance()
    result = dict(
        family=family,
        analysis_version=ANALYSIS_VERSION,
        adapter_version=ADAPTER_VERSION,
        inputs=inputs,
        parameters=parameters or {},
        backend_revision=backend["revision"],
        backend=backend,
        functions=calls.functions,
        compatibility=calls.compatibility,
    )
    try:
        payload = plain(analyze(calls))
        json.dumps(payload)
        result.update(status="ok", payload=payload)
    except (
        ImportError,
        AttributeError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ArithmeticError,
        RuntimeError,
        OSError,
        IngestError,
    ) as exc:
        result.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}", payload=None)
    return result


def critical_path(records, deps, names, calls=None):
    calls = calls or Calls()
    raw = json.loads(Path(records).read_text())
    rows = raw["aicore_tasks"]
    if not rows:
        return dict(static=[], observed=[], cpm_us=None, t0_ticks=None, t1_ticks=None)
    if any(len(r) not in (5, 6, 7) for r in rows):
        raise ValueError("unsupported AICore row width")
    epochs = {r[6] for r in rows if len(r) == 7}
    if len(epochs) > 1 or (epochs and any(len(r) != 7 for r in rows)):
        raise ValueError("critical path requires a single, unmixed execution epoch")
    f = module("critical_path").build_graph
    signature = hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
    needs_patch = any(len(r) != 6 for r in rows)
    # Only the audited six-column parser receives this format workaround.
    if needs_patch and signature == CP_SIX_COLUMN_HASH:
        raw["aicore_tasks"] = [(r + [0])[:6] for r in rows]
        calls.compatibility.append("CP-ROW-001")
    with TemporaryDirectory(prefix="pfdb-cpm-") as work:
        target = Path(work)
        (target / "chip_swimlane_records.json").write_text(json.dumps(raw))
        (target / "deps.json").write_bytes(Path(deps).read_bytes())
        from profile_db.runtime_adapter.kernel_names import document
        (target / "name_map.json").write_text(json.dumps(document(names)))
        graph = calls.call("critical_path", "build_graph", target, target, 2)
        graph.label = "capture"
        result = calls.call("critical_path", "analyze_rank", graph, 2)
    scale = 1e6 / graph.freq
    return dict(
        static=[dict(task_id=t, busy_us=graph.dur[t] * scale) for t in result.cpm_path],
        observed=[
            dict(
                task_id=s.task,
                kind=s.kind,
                compute_us=s.compute * scale,
                stall_us=s.stall * scale,
                start_ticks=graph.start[s.task],
                end_ticks=graph.end[s.task],
            )
            for s in result.segments
        ],
        cpm_us=result.cpm_len * scale,
        t0_ticks=graph.t0,
        t1_ticks=graph.t1,
        freq_hz=graph.freq,
        raw=plain(result),
    )


def scheduler(records, deps, calls):
    from profile_db.runtime_adapter.capture import read_capture

    data = read_capture(records, calls=calls)
    tasks = data["tasks"]
    valid, reason = calls.call("sched_overhead_analysis", "validate_perf_tasks_for_overhead_analysis", tasks)
    if not valid:
        raise ValueError(reason)
    w0, w1 = min(t["start_time_us"] for t in tasks), max(t["end_time_us"] for t in tasks)
    dependency = json.loads(Path(deps).read_text())
    overhead = calls.call("sched_overhead_analysis", "compute_overhead", tasks, dependency, w0, w1)
    grouped = {}
    for task in tasks:
        grouped.setdefault(task["core_id"], []).append(task)
    ordered_tasks = [t for group in grouped.values() for t in sorted(group, key=lambda t: t["start_time_us"])]
    heads, tails = calls.call("sched_overhead_analysis", "compute_head_tail", ordered_tasks)
    per_core, switches, split = calls.call(
        "sched_overhead_analysis", "aicore_switch_stats", tasks, overhead["ready_steps"], w0, w1
    )
    producers = {s.get("producer") for s in data.get("scheduler_streams", []) if s.get("producer")}
    if data.get("scheduler_task_producer"):
        producers.add(data["scheduler_task_producer"])
    if len(producers) > 1:
        raise ValueError("mixed AICPU/AICore scheduler producers are unsupported")
    producer = next(iter(producers), "aicpu")
    threads = (
        calls.call("sched_overhead_analysis", "parse_scheduler_from_json_phases", data)
        if producer == "aicpu"
        else {}
    )
    calls.call("sched_overhead_analysis", "compute_dag_stats_from_deps", dependency, data, threads)
    loops = (
        calls.call("sched_overhead_analysis", "_summarize_scheduler_loops", threads)
        if producer == "aicpu"
        else None
    )
    phase_records = data.get("scheduler_records") or []
    phase_order = (
        calls.call("sched_overhead_analysis", "_scheduler_phases_for_report", threads)
        if producer == "aicpu"
        else []
    )
    phase_totals = {}
    if producer == "aicore":
        for stream in phase_records:
            for record in stream:
                kind = calls.call(
                    "sched_overhead_analysis", "canonical_sched_phase", record.get("phase", "unknown")
                )
                phase_totals[kind] = phase_totals.get(kind, 0) + max(
                    0, record["end_time_us"] - record["start_time_us"]
                )
    edges = calls.call("swimlane_converter", "load_deps_json", deps)
    counters = calls.call("swimlane_converter", "build_overhead_counter_events", tasks, edges)
    return dict(
        window_us=[w0, w1],
        overhead=overhead,
        threads=threads,
        loops=loops,
        producer=producer,
        phase_records=phase_records,
        phase_totals_us=phase_totals,
        phase_order=phase_order,
        capture=[s.get("capture") for s in data.get("scheduler_streams", [])],
        per_core=per_core,
        switches=switches,
        switch_split=split,
        counters=counters,
        rows=[
            dict(
                task_id=str(t["task_id"]), core=t.get("core_id", t.get("core_index")), head_us=h, tail_us=tail
            )
            for t, h, tail in zip(ordered_tasks, heads, tails)
        ],
    )


def dependencies(path, calls):
    edges, nodes, annotations, tensors, tasks = calls.call("deps_viewer", "_load_deps_edges", path)
    meta = calls.call("deps_viewer", "_load_task_meta", path)
    kept, removed, dag = calls.call("deps_viewer", "_transitive_reduction", edges, nodes, annotations)
    dataflow = calls.call(
        "deps_viewer", "_transitive_reduction_dataflow", edges, nodes, annotations, tensors, tasks
    )
    markers = calls.call("deps_viewer", "_task_markers", nodes, edges, meta, tasks)
    return dict(
        edges=edges,
        reduced=kept,
        removed=removed,
        is_dag=dag,
        dataflow=dataflow[0],
        eligibility={str(t): "⭐" in markers.get(t, "") for t in nodes},
    )


def scope(path, calls):
    meta, records = calls.call("scope_stats_plot", "_load", path)
    pairs = calls.call("scope_stats_plot", "_pair_by_ring", records)
    risks = calls.call("scope_stats_plot", "_risk_entries", meta, pairs)
    for risk in risks:
        risk["metric"] = {k: v for k, v in risk["metric"]._asdict().items() if k != "fn"}
    return dict(meta=meta, resources=risks, records=records, paired_scopes=sum(map(len, pairs.values())))
