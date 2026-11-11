# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compact question views over persisted runtime analyses."""

from __future__ import annotations

from profile_db.errors import QueryError
from profile_db.facts import Evidence, Fact
from profile_db.runtime_adapter import Calls, module
from profile_db.runtime_adapter.modalities import events
from profile_db.runtime_adapter.storage import read_snapshot, resolve
from profile_db.task_ids import normalize_task_id


def analysis(conn, run_id, family):
    doc = read_snapshot(conn, run_id, family)
    fields = {
        k: doc[k]
        for k in (
            "family",
            "status",
            "analysis_version",
            "adapter_version",
            "backend_revision",
            "compatibility",
            "reason",
        )
        if k in doc
    }
    fields["run_id"] = run_id
    header = Fact("ANALYSIS", fields, Evidence.PROVEN if doc["status"] == "ok" else Evidence.UNAVAILABLE)
    return doc.get("payload") if doc["status"] == "ok" else None, [header]


def scheduler(conn, params):
    payload, facts = analysis(conn, params.run_id, "scheduler")
    if payload is None:
        return facts
    if params.task_id is not None or params.view == "rows":
        for row in payload["rows"]:
            if params.task_id is None or row["task_id"] == params.task_id:
                facts.append(Fact("SCHED_ROW", dict(run_id=params.run_id, **row), Evidence.MEASURED))
    elif params.view == "phases":
        if payload.get("producer") == "aicore":
            facts.append(
                Fact(
                    "SCHED_PHASES",
                    dict(producer="aicore", totals_us=payload["phase_totals_us"], capture=payload["capture"]),
                    Evidence.MEASURED,
                )
            )
        facts.extend(
            Fact("SCHED_PHASES", dict(lane=k, metrics=v), Evidence.MEASURED)
            for k, v in payload["threads"].items()
        )
    elif params.view == "cores":
        facts.append(
            Fact(
                "SCHED_CORES",
                dict(
                    per_core=payload["per_core"],
                    switch_split=payload["switch_split"],
                    domain="per-core; cannot be added to elapsed latency",
                ),
                Evidence.MEASURED,
            )
        )
    else:
        summary = {k: v for k, v in payload["overhead"].items() if k != "ready_steps"}
        facts.append(
            Fact(
                "SCHED_SUMMARY",
                dict(
                    run_id=params.run_id,
                    overhead=summary,
                    loops=payload["loops"],
                    producer=payload.get("producer"),
                    capture=payload.get("capture"),
                ),
                Evidence.MEASURED,
            )
        )
    return facts


def deps(conn, params):
    if params.view == "direct":
        if params.task_id is None:
            raise QueryError("direct deps requires task_id")
        return None
    payload, facts = analysis(conn, params.run_id, "deps")
    if payload is None:
        return facts
    key = "reduced" if params.view == "reduced" else "dataflow"
    facts.extend(
        Fact("DEP", dict(pred=str(p), succ=str(s), view=params.view), Evidence.PROVEN)
        for p, s in payload[key]
        if params.task_id is None
        or (
            (params.direction in ("out", "all") and str(p) == params.task_id)
            or (params.direction in ("in", "all") and str(s) == params.task_id)
        )
    )
    return facts


def counter_window(conn, run_id, t0, t1, engine=None):
    payload, facts = analysis(conn, run_id, "scheduler")
    if payload is None:
        return facts
    by_name = {}
    for event in payload["counters"]:
        if (
            engine is None
            or engine in event["name"]
            or event["name"] in ("oh_all_overhead", "oh_has_overhead")
        ):
            by_name.setdefault(event["name"], []).append(event)
    for name, series in sorted(by_name.items()):
        area, peak = 0.0, 0
        for i, event in enumerate(series):
            end = series[i + 1]["ts"] if i + 1 < len(series) else payload["window_us"][1]
            duration = max(0, min(t1, end) - max(t0, event["ts"]))
            value = event["args"][name]
            if duration:
                area += duration * value
                peak = max(peak, value)
        integral_field = (
            "integral_us"
            if "overhead" in name
            else ("integral_task_us" if name.endswith("_ready") else "integral_core_us")
        )
        facts.append(
            Fact(
                "OVERHEAD_WINDOW",
                dict(track=name, t0_us=t0, t1_us=t1, **{integral_field: area}, peak=peak),
                Evidence.MEASURED,
            )
        )
    return facts


def args_dump(conn, params):
    payload, facts = analysis(conn, params.run_id, "args")
    if payload is None:
        return facts
    if params.view == "values" and params.seq is None:
        raise QueryError("argument values require seq")
    for seq, entry in enumerate(payload["entries"]):
        task = normalize_task_id(entry["task_id"]).canonical if entry.get("task_id") is not None else None
        if params.seq is not None and seq != params.seq:
            continue
        if params.task_id is not None and task != params.task_id:
            continue
        if params.stage is not None and entry.get("stage") != params.stage:
            continue
        meta = {k: v for k, v in entry.items() if k not in ("values", "data", "value")}
        meta.update(seq=seq, task_id=task, run_id=params.run_id)
        if entry.get("kind") == "scalar":
            meta["value"] = entry.get("value")
        if params.view == "values" and entry.get("kind") != "scalar":
            try:
                path = resolve(conn, params.run_id, "args_payload", bounded=True)
                decoder = module("dump_viewer")
                dtype = str(entry.get("dtype", "")).lower()
                _, size = decoder.DTYPE_INFO.get(dtype, (None, 1))
                start = int(entry.get("bin_offset", 0))
                total = int(entry.get("bin_size", 0))
                if start < 0 or total < 0:
                    raise QueryError("invalid argument byte range")
                available = min(total, max(0, path.stat().st_size - start))
                count = min(params.limit, max(0, available // size - params.offset))
                calls = Calls()
                raw = calls.call(
                    "dump_viewer", "read_arg_data", path, start + params.offset * size, count * size
                )
                meta.update(
                    values=calls.call("dump_viewer", "decode_elements", raw, dtype, len(raw) // size),
                    functions=calls.functions,
                    offset=params.offset,
                    count=len(raw) // size,
                    layout="physical",
                    truncated=bool(entry.get("truncated")) or available < total,
                    next_offset=params.offset + count if (params.offset + count) * size < available else None,
                )
            except QueryError as exc:
                facts.append(Fact("ARGS", dict(meta, reason=str(exc)), Evidence.UNAVAILABLE))
                continue
        facts.append(Fact("ARGS", meta, Evidence.MEASURED))
    return facts


def scope_stats(conn, params):
    payload, facts = analysis(conn, params.run_id, "scope")
    if payload is None:
        return facts
    rows = payload["resources"] if params.view == "summary" else payload["records"]
    for row in rows:
        if params.site is not None and row.get("site") != params.site:
            continue
        if params.ring is not None and row.get("ring") != params.ring:
            continue
        facts.append(Fact("SCOPE", dict(run_id=params.run_id, **row), Evidence.MEASURED))
    if not rows:
        facts.append(
            Fact(
                "SCOPE", dict(run_id=params.run_id, reason="no paired resource records"), Evidence.UNAVAILABLE
            )
        )
    return facts


def incore_events(conn, params):
    if params.kernel is None:
        raise QueryError("in-core events require kernel")
    if params.t0_us is not None and params.t1_us is not None and params.t1_us <= params.t0_us:
        raise QueryError("event window must have t1_us > t0_us")
    try:
        path = resolve(conn, params.run_id, "incore_trace:" + params.kernel, bounded=True)
        calls = Calls()
        doc = events(path, calls)
    except (QueryError, ImportError, ValueError, KeyError) as exc:
        return [Fact("INCORE", dict(kernel=params.kernel, reason=str(exc)), Evidence.UNAVAILABLE)]
    lanes = {
        e.get("tid"): e.get("args", {}).get("name", "")
        for e in doc.get("traceEvents", [])
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    rows = [
        e
        for e in doc.get("traceEvents", [])
        if "ts" in e
        and (
            params.pipe is None
            or params.pipe in str(e.get("cat", ""))
            or params.pipe in str(e.get("name", ""))
            or params.pipe in str(lanes.get(e.get("tid"), e.get("tid", "")))
        )
        and (params.t0_us is None or e["ts"] + e.get("dur", 0) >= params.t0_us)
        and (params.t1_us is None or e["ts"] < params.t1_us)
    ]
    return [
        Fact(
            "INCORE_PAGE",
            dict(
                functions=calls.functions,
                total=len(rows),
                offset=params.offset,
                next_offset=params.offset + params.limit
                if len(rows) > params.offset + params.limit
                else None,
            ),
            Evidence.MEASURED,
        )
    ] + [
        Fact("INCORE_EVENT", dict(index=i + params.offset, event=e), Evidence.MEASURED)
        for i, e in enumerate(rows[params.offset : params.offset + params.limit])
    ]


def host(conn, params):
    payload, facts = analysis(conn, params.run_id, "host")
    if payload is None:
        return facts
    if params.view == "cpu" and (params.inv is not None or params.hid is not None):
        raise QueryError("host CPU aggregates accept pid; use tree/rounds for invocation or HID filters")
    key = {"summary": "rounds", "rounds": "rounds", "tree": "spans", "cpu": "cpu"}[params.view]
    rows = [
        r
        for r in payload[key]
        if all(
            value is None or r.get(field) == value
            for field, value in (("pid", params.pid), ("inv", params.inv), ("hid", params.hid))
        )
    ]
    if params.view == "summary":
        facts.append(
            Fact(
                "HOST",
                dict(
                    rounds=len(rows),
                    cold_keys=payload["cold_keys"],
                    warm_binds=len(payload["warm_binds"]),
                    warm_stats=payload.get("warm_stats", []),
                    drops=payload["drops"],
                    phase_orphaned_passes=payload.get("phase_orphaned_passes", 0),
                    phase_skipped_records=payload.get("phase_skipped_records", 0),
                    incomplete_records=payload.get("incomplete_records"),
                    cpu_incomplete=sum(
                        any(
                            r.get(k) is None
                            for k in ("cpu_ns", "rec_cpu_ns", "minflt", "tminflt", "nvcsw", "nivcsw")
                        )
                        for r in payload["cpu_rows"]
                    ),
                ),
                Evidence.MEASURED,
            )
        )
    else:
        if params.view == "cpu" and not rows:
            return facts + [Fact("HOST", dict(reason="no complete warm CPU counters"), Evidence.UNAVAILABLE)]
        facts.append(
            Fact(
                "HOST_PAGE",
                dict(
                    total=len(rows),
                    offset=params.offset,
                    next_offset=params.offset + params.limit
                    if len(rows) > params.offset + params.limit
                    else None,
                ),
                Evidence.MEASURED,
            )
        )
        facts.extend(
            Fact("HOST", r, Evidence.MEASURED) for r in rows[params.offset : params.offset + params.limit]
        )
    return facts
