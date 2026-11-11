# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Existing host logs, argument manifests and instruction traces."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from profile_db.runtime_adapter import plain
from profile_db.runtime_adapter.backend import require


def host(paths, calls):
    require("strace_timing", "parse_spans", "count_record_heads", "load_host_phase_records",
            "host_record_spans", "invocation_spans", "group_invocations", "_round_metrics",
            "parse_drop_summaries")
    require("hbg_bind_phases", "parse_binds", "cold_binds", "spread")
    require("phase_time_split", "parse", "cold_keys", "summarise")
    logs = [p for k, p in paths.items() if k.startswith("host_log:")]
    lines = [line for p in logs for line in Path(p).read_text(errors="replace").splitlines()]
    spans = list(calls.call("strace_timing", "parse_spans", lines))
    heads = calls.call("strace_timing", "count_record_heads", lines)
    incomplete_records = heads - len(spans)
    if not spans:
        raise ValueError("host artifacts contain no complete STRACE records")
    phase_files = [p for k, p in paths.items() if k.startswith("host_phase:")]
    orphaned = skipped = 0
    if phase_files:
        passes = calls.call("strace_timing", "load_host_phase_records", phase_files)
        extra, orphaned, skipped = calls.call("strace_timing", "host_record_spans", spans, passes)
        spans += extra
    invocation_spans = calls.call("strace_timing", "invocation_spans", spans)
    invocations = calls.call("strace_timing", "group_invocations", invocation_spans)
    rounds = []
    names = require("strace_timing")._ROUNDS_TABLE_NAMES
    for inv in invocations:
        values = calls.call("strace_timing", "_round_metrics", inv)
        present = inv.by_name()
        keys = ("host_us", "device_us", "effective_us", "orch_us", "sched_us")
        row = dict(pid=inv.pid, inv=inv.inv, hid=inv.hid)
        for key, value, name in zip(keys, values, ("run", "device", None, "orch", "sched")):
            row[key] = (
                value
                if (name is None and any(names[n] in present for n in ("orch", "sched")))
                or (name is not None and names[name] in present)
                else None
            )
        rounds.append(row)
    binds = calls.call("hbg_bind_phases", "parse_binds", logs)
    cold = calls.call("hbg_bind_phases", "cold_binds", binds)
    warm = [b for b in binds if (b["pid"], b["inv"]) not in cold]
    rows = calls.call("phase_time_split", "parse", logs)
    cpu_cold = calls.call("phase_time_split", "cold_keys", rows)
    cpu = []
    for pid, phase in sorted({(r["pid"], r["phase"]) for r in rows}):
        group = [
            r
            for r in rows
            if r["phase"] == phase
            and r["pid"] == pid
            and (r["pid"], r["inv"]) not in cpu_cold
            and all(
                r[f] is not None for f in ("cpu_ns", "rec_cpu_ns", "minflt", "tminflt", "nvcsw", "nivcsw")
            )
        ]
        if group:
            cpu.append(dict(pid=pid, phase=phase, **calls.call("phase_time_split", "summarise", group)))
    warm_stats = []
    for pid in sorted({b["pid"] for b in warm}):
        for phase in sorted({k for b in warm if b["pid"] == pid for k in b if k not in ("pid", "inv", "ts")}):
            values = [b[phase] for b in warm if b["pid"] == pid and phase in b]
            if values:
                low, median, high = calls.call("hbg_bind_phases", "spread", values)
                warm_stats.append(
                    dict(pid=pid, phase=phase, min_ms=low, median_ms=median, max_ms=high, n=len(values))
                )
    drops = calls.call("strace_timing", "parse_drop_summaries", lines)
    return dict(
        rounds=rounds,
        spans=[asdict(s) for s in spans],
        binds=binds,
        warm_binds=warm,
        cold_keys=sorted(cold),
        warm_stats=warm_stats,
        cpu=cpu,
        cpu_rows=rows,
        drops=plain(drops),
        record_heads=heads,
        parsed_records=heads - incomplete_records,
        incomplete_records=incomplete_records,
        phase_orphaned_passes=orphaned,
        phase_skipped_records=skipped,
    )


def args(path):
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError("argument manifest must be an object")
    entries = doc.get("args", doc.get("tensors", []))
    if not isinstance(entries, list):
        raise ValueError("argument manifest must contain an array")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("argument entries must be objects")
        for field in ("bin_offset", "bin_size"):
            if int(entry.get(field) or 0) < 0:
                raise ValueError(f"negative argument {field}")
    return dict(entries=entries, bin_file=doc.get("bin_file", "args.bin"))


def incore(manifest, calls):
    import csv

    rows = list(csv.DictReader(Path(manifest).open()))
    return dict(kernels=rows)


def events(path, calls):
    doc = json.loads(Path(path).read_text())
    calls.call("core_swimlane", "_to_perfetto", doc)
    return doc
